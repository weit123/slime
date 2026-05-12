"""Custom generate function for Android World VLM agent training.

This module provides the ``generate()`` coroutine used via slime's
``--custom-generate-function-path examples.android_world.rollout.generate``.

The design closely follows ``examples/geo3k_vlm_multi_turn/rollout.py`` and
reuses its helper functions for token management, multimodal encoding, and
SGLang HTTP inference. The key differences are:

1. **Dynamic prompt construction** — the initial prompt (system + first
   screenshot) is built after ``env.reset()``, not from the JSONL dataset.
2. **Persistent env pool** — workers are acquired from / released to an
   ``AndroidWorldEnvPool`` singleton.
3. **In-loop reward** — ``sample.reward`` is set directly from the env
   evaluation, so no external RM is needed.
"""

from __future__ import annotations

import logging
from typing import Any

from examples.android_world.env_android_world import AndroidWorldEnv
from examples.android_world.env_pool import AndroidWorldEnvPool
from examples.android_world.prompts import ANDROID_WORLD_SYSTEM_PROMPT
from examples.geo3k_vlm_multi_turn.rollout import (
    _append_to_sample,
    _encode_observation_for_generation,
    _finalize_sample,
    _merge_multimodal_train_inputs,
    _run_inference_step,
    _should_stop_on_finish,
    _update_budget,
    _update_multimodal_state,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Dummy messages used for calculating trim length in chat template encoding.
# Must match what _encode_observation_for_generation uses internally.
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _build_initial_prompt(
    tokenizer,
    processor,
    system_prompt: str,
    first_user_message: dict,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """Construct the tokenized initial prompt from the system prompt and first observation.

    Returns:
        (prompt_text, prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs)
    """
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
        prompt_text = messages  # unlikely path for VLM

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
    """Multi-turn rollout for an Android World VLM agent.

    This is the entry point invoked by slime's ``generate_and_rm`` dispatch
    when ``--custom-generate-function-path`` points here.

    The flow:
    1. Acquire a worker from the env pool.
    2. Reset the env with the task from ``sample.metadata``.
    3. Build the initial prompt dynamically (system + screenshot + task text).
    4. Multi-turn loop: SGLang generate → env.step → encode obs → repeat.
    5. Set ``sample.reward`` from the env evaluation.
    6. Release the worker back to the pool.
    """
    assert not getattr(args, "partial_rollout", False), (
        "Partial rollout is not supported for Android World interaction rollouts."
    )

    # -- 1. Pool & worker --
    pool = await AndroidWorldEnvPool.get_instance(vars(args))
    worker_ref, worker_id = await pool.acquire()

    task_name = (sample.metadata or {}).get("task_name")
    params_idx = (sample.metadata or {}).get("params_idx", 0)
    max_turns = getattr(args, "max_turns", 15)

    env = AndroidWorldEnv(
        worker_ref=worker_ref,
        worker_id=worker_id,
        pool=pool,
        task_name=task_name,
        params_idx=params_idx,
        max_turns=max_turns,
        image_size=getattr(args, "image_size", None),
    )

    sampling_params = sampling_params.copy()

    try:
        # -- 2. SGLang state --
        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        sample.metadata = sample.metadata or {}

        # -- 3. Reset env --
        obs, info = await env.reset()

        # -- 4. Build initial prompt dynamically --
        first_user_message = env.format_observation(obs)

        prompt_text, prompt_ids, current_image_data, multimodal_inputs, init_mm_train = _build_initial_prompt(
            tokenizer=state.tokenizer,
            processor=state.processor,
            system_prompt=ANDROID_WORLD_SYSTEM_PROMPT,
            first_user_message=first_user_message,
            apply_chat_template=getattr(args, "apply_chat_template", True),
            apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
        )

        # Override the sample's prompt (JSONL placeholder) with the real one
        sample.prompt = prompt_text
        sample.multimodal_inputs = multimodal_inputs
        sample.tokens = list(prompt_ids)

        # -- 5. Initialize tracking --
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

        # -- 6. Multi-turn loop --
        cur_sampling_params = sampling_params
        for turn_idx in range(max_turns):
            if budget is not None:
                cur_sampling_params = sampling_params.copy()
                cur_sampling_params["max_new_tokens"] = budget

            # Generate via SGLang HTTP
            response_text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
                url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer
            )

            # Append model output (loss_mask=1)
            _append_to_sample(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1)
            budget = _update_budget(budget, len(new_tokens))

            # Check SGLang finish reason
            if _should_stop_on_finish(sample, finish_type):
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            # Environment step
            step_obs, done, step_info = await env.step(response_text)

            if done or step_obs is None:
                sample.status = Sample.Status.COMPLETED
                break

            # Encode observation (screenshot + step counter, loss_mask=0)
            next_user_message = env.format_observation(step_obs, is_initial=False)
            obs_prompt_ids, obs_image_data, obs_mm, obs_mm_train = _encode_observation_for_generation(
                state.tokenizer,
                state.processor,
                next_user_message,
                sample.metadata,
                getattr(args, "apply_chat_template", True),
                getattr(args, "apply_chat_template_kwargs", None),
            )

            # Strip leading BOS if present
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

        # -- 7. Set reward --
        sample.reward = env.get_reward()

        # -- 8. Finalize --
        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)

    finally:
        try:
            env.close()
        except Exception:
            pass
