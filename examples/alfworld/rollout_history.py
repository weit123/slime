"""History-based multi-turn rollout for ALFWorld VLM agent training.

Usage:
    --custom-generate-function-path examples.alfworld.rollout_history.generate
    --custom-reward-post-process-path examples.alfworld.rollout_history.post_process_rewards_history
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import torch
from PIL import Image as PILImage

import ray
from examples.alfworld.env_pool import AlfWorldEnvPool
from examples.alfworld.env_worker import AlfWorldWorker
from examples.alfworld.prompts import get_alfworld_prompt, ALFWORLD_SYSTEM_PROMPT
from examples.geo3k_vlm_multi_turn.rollout import (
    _merge_multimodal_train_inputs,
    _run_inference_step,
    _should_stop_on_finish,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _build_step_prompt(tokenizer, processor, user_message, apply_chat_template, apply_chat_template_kwargs):
    """Build standalone prompt for one step."""
    messages = [{"role": "system", "content": ALFWORLD_SYSTEM_PROMPT}, user_message]
    apply_kwargs = apply_chat_template_kwargs or {}

    if apply_chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **apply_kwargs)
    else:
        prompt_text = messages

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info
        images, _ = process_vision_info([user_message])
        multimodal_inputs = {"images": images}
        processor_output = processor(text=prompt_text, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"] and "video" not in k} or None
    else:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs.get("images", [])] if multimodal_inputs else []
    return prompt_text, prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _make_turn_sample(original_sample, prompt_text, prompt_ids, mm_inputs, mm_train, response_tokens, response_logprobs, tokenizer):
    """Create standalone Sample for one step."""
    full_tokens = list(prompt_ids) + response_tokens
    return Sample(
        group_index=original_sample.group_index,
        index=original_sample.index,
        label=original_sample.label,
        metadata=original_sample.metadata,
        generate_function_path=original_sample.generate_function_path,
        prompt=prompt_text,
        tokens=full_tokens,
        loss_mask=[1] * len(response_tokens),
        rollout_log_probs=response_logprobs,
        response=tokenizer.decode(response_tokens, skip_special_tokens=False),
        response_length=len(response_tokens),
        multimodal_inputs=mm_inputs,
        multimodal_train_inputs=_merge_multimodal_train_inputs([mm_train] if mm_train else []),
        status=Sample.Status.COMPLETED,
    )


def post_process_rewards_history(args, samples: List[Sample]) -> tuple[List[float], List[float]]:
    """GRPO reward normalization for variable-length trajectories."""
    raw_rewards = [s.get_reward_value(args) for s in samples]
    if not (args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization):
        return raw_rewards, raw_rewards

    normalized = [0.0] * len(samples)
    groups = {}
    for pos, s in enumerate(samples):
        groups.setdefault(s.group_index, []).append((pos, s))

    for _, members in groups.items():
        trajs = {}
        for pos, s in members:
            trajs.setdefault(s.index, []).append(pos)
        traj_rewards = [raw_rewards[positions[0]] for positions in trajs.values()]
        traj_positions = list(trajs.values())

        tr = torch.tensor(traj_rewards, dtype=torch.float)
        tr = tr - tr.mean()
        if len(tr) > 1:
            tr = tr / (tr.std() + 1e-6)

        for norm_reward, positions in zip(tr.tolist(), traj_positions):
            for pos in positions:
                normalized[pos] = norm_reward

    return raw_rewards, normalized


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> List[Sample] | Sample:
    """History-based multi-turn rollout for ALFWorld."""
    assert not getattr(args, "partial_rollout", False), "Partial rollout not supported."

    pool = await AlfWorldEnvPool.get_instance(vars(args))
    worker_ref, worker_id = await pool.acquire()

    max_turns = getattr(args, "max_turns", 50)
    image_size = getattr(args, "image_size", None)
    action_only = getattr(args, "action_only", False)

    try:
        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        # Reset
        import asyncio
        obs, info = await asyncio.to_thread(ray.get, worker_ref.reset.remote())
        action_history = []
        trajectory_samples = []

        for turn_idx in range(max_turns):
            # Build user message
            task = obs.get("task", "")
            admissible = obs.get("admissible_actions", [])
            user_message = {"role": "user", "content": []}

            image = obs.get("image")
            if image is not None:
                if isinstance(image, np.ndarray):
                    image = PILImage.fromarray(image)
                if image_size:
                    image = image.resize(tuple(image_size), PILImage.LANCZOS)
                user_message["content"].append({"type": "image", "image": image})

            text = get_alfworld_prompt(task, action_history, admissible, action_only)
            user_message["content"].append({"type": "text", "text": text})

            prompt_text, prompt_ids, image_data, mm_inputs, mm_train = _build_step_prompt(
                state.tokenizer, state.processor, user_message,
                getattr(args, "apply_chat_template", True),
                getattr(args, "apply_chat_template_kwargs", None),
            )

            cur_params = sampling_params.copy()
            max_ctx = getattr(args, "max_context_len", None)
            if max_ctx:
                budget = max_ctx - len(prompt_ids)
                if budget <= 0:
                    break
                cur_params["max_new_tokens"] = budget

            response_text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
                url, list(prompt_ids), cur_params, image_data, state.tokenizer
            )

            turn_sample = _make_turn_sample(sample, prompt_text, list(prompt_ids), mm_inputs, mm_train, new_tokens, new_logprobs, state.tokenizer)

            if _should_stop_on_finish(turn_sample, finish_type):
                trajectory_samples.append(turn_sample)
                break

            trajectory_samples.append(turn_sample)

            # Step
            step_obs, reward, done, step_info = await asyncio.to_thread(ray.get, worker_ref.step.remote(response_text))
            action_history = step_obs.get("action_history", action_history) if step_obs else action_history

            if done or step_obs is None:
                break
            obs = step_obs

        # Get final reward
        final_reward = await asyncio.to_thread(ray.get, worker_ref.get_reward.remote())

        if evaluation:
            final = trajectory_samples[-1] if trajectory_samples else sample
            final.reward = final_reward
            return final

        for s in trajectory_samples:
            s.reward = final_reward

        return trajectory_samples if trajectory_samples else [sample]

    finally:
        pool.release(worker_id)
