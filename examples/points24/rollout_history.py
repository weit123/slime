"""History-based multi-turn rollout for Points24 VLM agent training.

This module provides an alternative rollout strategy where each turn produces
an independent Sample with full history. A trajectory of length T returns T
Sample objects.

This is compatible with standard VLM RL methods (R1-V, InternVL-RL style)
where each turn is trained independently with full context.

Usage:
    --custom-generate-function-path examples.points24.rollout_history.generate
    --custom-reward-post-process-path examples.points24.rollout_history.post_process_rewards_history
    --n-samples-per-prompt N  # e.g., 8
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage

from examples.geo3k_vlm_multi_turn.rollout import (
    _merge_multimodal_train_inputs,
    _run_inference_step,
    _should_stop_on_finish,
)
from examples.points24.env_worker import Points24Worker
from examples.points24.prompts import get_points24_prompt, POINTS24_SYSTEM_PROMPT
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _build_step_prompt(
    tokenizer,
    processor,
    user_message: dict,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """Build a standalone tokenized prompt for one trajectory step.

    Returns:
        (prompt_text, prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs)
    """
    messages = [
        {"role": "system", "content": POINTS24_SYSTEM_PROMPT},
        user_message,
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

        images, _ = process_vision_info([user_message])
        multimodal_inputs = {"images": images}
        processor_output = processor(text=prompt_text, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v
            for k, v in processor_output.items()
            if k not in ["input_ids", "attention_mask"]
            and "video" not in k
        } or None
    else:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]

    return prompt_text, prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _make_turn_sample(
    original_sample: Sample,
    prompt_text: str | list,
    prompt_ids: list[int],
    multimodal_inputs: dict | None,
    multimodal_train_inputs: dict | None,
    response_tokens: list[int],
    response_logprobs: list[float],
    tokenizer,
) -> Sample:
    """Create a complete, standalone Sample for one trajectory step.

    The slime training pipeline expects:
    - tokens = prompt_ids + response_tokens  (full sequence)
    - loss_mask = [1]*response_len           (response-only, NOT full sequence)
    - rollout_log_probs = response_logprobs  (response-only)
    - response_length = len(response_tokens)
    """
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
        multimodal_inputs=multimodal_inputs,
        multimodal_train_inputs=_merge_multimodal_train_inputs(
            [multimodal_train_inputs] if multimodal_train_inputs else []
        ),
        status=Sample.Status.COMPLETED,
    )


def _extract_action_summary(response_text: str) -> str:
    """Extract the action from model response for history tracking."""
    # Try to extract from JSON "action" field
    match = re.search(r'"action"\s*:\s*"([^"]+)"', response_text)
    if match:
        return match.group(1)
    # Fallback to first 50 chars
    return response_text[:50].strip()


def check_reward_nonzero_std_history(args, group, **kwargs):
    """Dynamic filter for history-based rollout groups.

    Each element of ``group`` is ``list[Sample]`` (one trajectory's steps).
    All steps in a trajectory share the same reward, so we take the first.
    Drop the group if all N trajectory rewards are identical (std == 0).
    """
    traj_rewards = []
    for traj in group:
        if isinstance(traj, list):
            traj_rewards.append(traj[0].get_reward_value(args))
        else:
            traj_rewards.append(traj.get_reward_value(args))

    keep = bool(torch.tensor(traj_rewards, dtype=torch.float).std() > 0.0)
    return type('DynamicFilterOutput', (), {
        'keep': keep,
        'reason': None if keep else f"zero_std_{round(float(traj_rewards[0]), 1)}",
    })()


def post_process_rewards_history(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """GRPO reward normalization that correctly handles variable-length trajectories.

    The default _post_process_rewards in slime/ray/rollout.py reshapes rewards as
    (-1, n_samples_per_prompt) and falls back to (1, total) when total != N*B. With
    history-based rollout, each trajectory produces a variable number of step-samples
    (all sharing the same reward), so the total is always != N*B. The fallback
    normalizes all samples as one group, mixing prompts and biasing toward longer
    trajectories.

    This function fixes:
    1. Cross-prompt normalization: different prompts are never mixed.
    2. Length bias: each trajectory contributes equally regardless of step count.
    3. Cross-trajectory mixing: normalization is strictly per-prompt across N trajectories.
    """
    raw_rewards = [s.get_reward_value(args) for s in samples]

    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    normalized = [0.0] * len(samples)

    # group_index identifies the prompt, index identifies the trajectory copy
    groups: dict[int, list[tuple[int, Sample]]] = {}
    for pos, s in enumerate(samples):
        groups.setdefault(s.group_index, []).append((pos, s))

    for _group_index, members in groups.items():
        # Within a group, sub-group by trajectory (s.index)
        trajs: dict[int, list[int]] = {}
        for pos, s in members:
            trajs.setdefault(s.index, []).append(pos)

        # Each trajectory has a single reward (all steps share it)
        traj_rewards = []
        traj_positions = []
        for _idx, positions in trajs.items():
            reward = raw_rewards[positions[0]]
            traj_rewards.append(reward)
            traj_positions.append(positions)

        # Normalize across N trajectories for this prompt
        tr = torch.tensor(traj_rewards, dtype=torch.float)
        tr = tr - tr.mean()
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            if len(tr) > 1:
                std = tr.std()
                tr = tr / (std + 1e-6)

        # Broadcast normalized reward back to all step-samples in each trajectory
        for norm_reward, positions in zip(tr.tolist(), traj_positions):
            for pos in positions:
                normalized[pos] = norm_reward

    return raw_rewards, normalized


async def generate(
    args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False
) -> list[Sample] | Sample:
    """Non-incremental multi-turn rollout for Points24.

    Each turn is an independent training sample: (full_history_prompt + screenshot_t
    → response_t). A trajectory of length T returns T Sample objects.

    Invoked by slime when --custom-generate-function-path points here.

    Args:
        args: Parsed slime arguments.
        sample: Input sample (prompt is ignored; task generated dynamically).
        sampling_params: SGLang sampling parameters.
        evaluation: When True, return a single Sample instead of a list so that
            eval_rollout_single_dataset can log sample.prompt and compute_pass_rate
            gets one reward per prompt.

    Returns:
        list[Sample] during training (one per trajectory step), or a single Sample
        during evaluation.
    """
    assert not getattr(args, "partial_rollout", False), (
        "Partial rollout is not supported for Points24 history-based rollouts."
    )

    # Get config from args
    max_turns = getattr(args, "max_turns", 30)
    image_size = getattr(args, "image_size", None)
    action_only = getattr(args, "action_only", False)
    treat_face_cards_as_10 = getattr(args, "treat_face_cards_as_10", True)
    target_points = getattr(args, "target_points", 24)

    # Create worker directly (no pool needed)
    worker = Points24Worker(
        max_steps=max_turns,
        image_size=image_size,
        treat_face_cards_as_10=treat_face_cards_as_10,
        target_points=target_points,
    )

    sampling_params = sampling_params.copy()

    try:
        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        # Reset environment
        obs, info = worker.reset()

        trajectory_samples: list[Sample] = []

        for turn_idx in range(max_turns):
            # Build standalone prompt for this step
            user_message = {
                "role": "user",
                "content": [],
            }

            # Add image
            image = obs.get("image")
            if image is not None:
                if isinstance(image, np.ndarray):
                    image = PILImage.fromarray(image)
                if image_size is not None:
                    image = image.resize(tuple(image_size), PILImage.LANCZOS)
                user_message["content"].append({"type": "image", "image": image})

            # Add text prompt
            formula = obs.get("formula", [])
            text = get_points24_prompt(formula, action_only=action_only)
            user_message["content"].append({"type": "text", "text": text})

            prompt_text, prompt_ids, image_data, mm_inputs, mm_train = _build_step_prompt(
                tokenizer=state.tokenizer,
                processor=state.processor,
                user_message=user_message,
                apply_chat_template=getattr(args, "apply_chat_template", True),
                apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
            )

            # Compute per-step budget
            cur_params = sampling_params.copy()
            max_ctx = getattr(args, "max_context_len", None) or getattr(
                args, "rollout_max_context_len", None
            )
            if max_ctx is not None:
                step_budget = max_ctx - len(prompt_ids)
                if step_budget <= 0:
                    logger.warning(
                        "Step %d prompt length %d exceeds max_context_len %d; stopping",
                        turn_idx, len(prompt_ids), max_ctx,
                    )
                    break
                cur_params["max_new_tokens"] = step_budget

            # Single-turn SGLang inference
            response_text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
                url, list(prompt_ids), cur_params, image_data, state.tokenizer
            )

            # Build standalone Sample for this step
            turn_sample = _make_turn_sample(
                original_sample=sample,
                prompt_text=prompt_text,
                prompt_ids=list(prompt_ids),
                multimodal_inputs=mm_inputs,
                multimodal_train_inputs=mm_train,
                response_tokens=new_tokens,
                response_logprobs=new_logprobs,
                tokenizer=state.tokenizer,
            )

            # Check for early termination
            if _should_stop_on_finish(turn_sample, finish_type):
                trajectory_samples.append(turn_sample)
                break

            trajectory_samples.append(turn_sample)

            # Step the environment
            step_obs, reward, done, step_info = worker.step(response_text)

            if done or step_obs is None:
                break

            obs = step_obs

            if turn_idx + 1 >= max_turns:
                break

        # Get final reward
        reward = worker.get_reward()

        # For evaluation, return a single Sample with trajectory-level reward
        if evaluation:
            final = trajectory_samples[-1] if trajectory_samples else sample
            final.reward = reward
            final.status = Sample.Status.COMPLETED
            return final

        # Assign the trajectory-level reward to every step sample
        for s in trajectory_samples:
            s.reward = reward

        if not trajectory_samples:
            sample.reward = reward
            sample.status = Sample.Status.COMPLETED
            return [sample]

        return trajectory_samples

    finally:
        try:
            worker.close()
        except Exception:
            pass
