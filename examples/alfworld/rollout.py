"""Custom generate function for ALFWorld VLM agent training.

This module provides the ``generate()`` coroutine for ALFWorld environment
rollout via slime's ``--custom-generate-function-path`` mechanism.

Design follows examples/android_world/rollout.py with key differences:
- Uses AlfWorldEnvPool for Ray actor management
- Prompt includes action history and admissible actions
- Reward from environment evaluation (50*won + goal_success - illegal)
"""

from __future__ import annotations

import logging
from typing import Any

from examples.alfworld.env_alfworld import AlfWorldEnv
from examples.alfworld.env_pool import AlfWorldEnvPool
from examples.geo3k_vlm_multi_turn.rollout import (
    _append_to_sample,
    _encode_observation_for_generation,
    _finalize_sample,
    _run_inference_step,
    _should_stop_on_finish,
    _update_budget,
    _update_multimodal_state,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# System prompt for VLM chat template
ALFWORLD_SYSTEM_PROMPT = "You are a helpful assistant."


def _build_initial_prompt(
    tokenizer,
    processor,
    system_prompt: str,
    first_user_message: dict,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """Construct the tokenized initial prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        first_user_message,
    ]

    apply_kwargs = apply_chat_template_kwargs or {}
    if apply_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
    else:
        prompt_text = messages

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info

        images, _ = process_vision_info([first_user_message])
        multimodal_inputs = {"images": images}
        processor_output = processor(text=prompt_text, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
            and "video" not in k
        } or None
    else:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]

    return prompt_text, prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample:
    """Multi-turn rollout for an ALFWorld VLM agent."""
    assert not getattr(args, "partial_rollout", False), (
        "Partial rollout is not supported for ALFWorld interaction rollouts."
    )

    # Acquire worker from pool
    pool = await AlfWorldEnvPool.get_instance(vars(args))
    worker_ref, worker_id = await pool.acquire()

    max_turns = getattr(args, "max_turns", 50)
    image_size = getattr(args, "image_size", None)
    action_only = getattr(args, "action_only", False)

    env = AlfWorldEnv(
        worker_ref=worker_ref,
        worker_id=worker_id,
        pool=pool,
        max_turns=max_turns,
        image_size=image_size,
        action_only=action_only,
    )

    sampling_params = sampling_params.copy()

    try:
        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        sample.metadata = sample.metadata or {}

        # Reset env
        obs, info = await env.reset()

        # Build initial prompt
        first_user_message = env.format_observation(obs, is_initial=True)

        prompt_text, prompt_ids, current_image_data, multimodal_inputs, init_mm_train = _build_initial_prompt(
            tokenizer=state.tokenizer,
            processor=state.processor,
            system_prompt=ALFWORLD_SYSTEM_PROMPT,
            first_user_message=first_user_message,
            apply_chat_template=getattr(args, "apply_chat_template", True),
            apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
        )

        sample.prompt = prompt_text
        sample.multimodal_inputs = multimodal_inputs
        sample.tokens = list(prompt_ids)

        multimodal_train_inputs_buffer: list[dict | None] = []
        if init_mm_train:
            multimodal_train_inputs_buffer.append(init_mm_train)

        response_tokens: list[int] = []
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.response_length = 0

        budget = None
        max_ctx = getattr(args, "max_context_len", None) or getattr(args, "rollout_max_context_len", None)
        if max_ctx is not None:
            budget = max_ctx - len(sample.tokens)
        elif sampling_params.get("max_new_tokens") is not None:
            budget = sampling_params["max_new_tokens"]

        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            sample.reward = 0.0
            return sample

        # Multi-turn loop
        cur_sampling_params = sampling_params
        for turn_idx in range(max_turns):
            if budget is not None:
                cur_sampling_params = sampling_params.copy()
                cur_sampling_params["max_new_tokens"] = budget

            response_text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
                url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer
            )

            _append_to_sample(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1)
            budget = _update_budget(budget, len(new_tokens))

            if _should_stop_on_finish(sample, finish_type):
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            step_obs, done, step_info = await env.step(response_text)

            if done or step_obs is None:
                sample.status = Sample.Status.COMPLETED
                break

            next_user_message = env.format_observation(step_obs, is_initial=False)
            obs_prompt_ids, obs_image_data, obs_mm, obs_mm_train = _encode_observation_for_generation(
                state.tokenizer,
                state.processor,
                next_user_message,
                sample.metadata,
                getattr(args, "apply_chat_template", True),
                getattr(args, "apply_chat_template_kwargs", None),
            )

            bos_id = state.tokenizer.bos_token_id
            if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
                obs_prompt_ids = obs_prompt_ids[1:]

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            budget = _update_budget(budget, len(obs_prompt_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_image_data,
                obs_mm,
                obs_mm_train,
                multimodal_train_inputs_buffer,
            )

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            if turn_idx + 1 >= max_turns:
                sample.status = Sample.Status.COMPLETED
                break

        sample.reward = env.get_reward()

        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)

    finally:
        try:
            env.close()
        except Exception:
            pass
