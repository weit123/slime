"""Non-incremental history-based rollout for Android World.

Each turn produces an independent (prompt_with_full_history + screenshot → response)
training sample, using ANDROID_WORLD_TEMPLATE. A trajectory of length T yields T
Sample objects returned as list[Sample].

Unlike rollout.py which uses the incremental token-append architecture (one sample per
trajectory, context grows each turn), this module produces one standalone sample per
turn. Each step's prompt includes the full task description and complete action history
built from scratch, compatible with standard VLM RL methods (R1-V, InternVL-RL style).

GRPO usage:
    --custom-generate-function-path examples.android_world.rollout_history.generate
    --custom-reward-post-process-path examples.android_world.rollout_history.post_process_rewards_history
    --n-samples-per-prompt N  # e.g. 8

The custom post_process_rewards_history function is required because variable-length
trajectories break the default _post_process_rewards reshape logic. Without it, the
default code path normalizes all samples as a single group (cross-prompt), biasing
gradients toward longer trajectories. The custom function groups samples by group_index
(prompt), computes per-trajectory rewards, and normalizes across N trajectories per
prompt — matching correct GRPO semantics.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from PIL import Image as PILImage

from examples.android_world.env_android_world import AndroidWorldEnv
from examples.android_world.env_pool import AndroidWorldEnvPool
from examples.android_world.prompts import (
    ANDROID_WORLD_SYSTEM_PROMPT,
    ANDROID_WORLD_TEMPLATE,
)
from examples.geo3k_vlm_multi_turn.rollout import (
    _merge_multimodal_train_inputs,
    _run_inference_step,
    _should_stop_on_finish,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def check_reward_nonzero_std_history(args, group, **kwargs):
    """Dynamic filter for history-based rollout groups.

    Each element of ``group`` is ``list[Sample]`` (one trajectory's steps).
    All steps in a trajectory share the same reward, so we take the first.
    Drop the group if all N trajectory rewards are identical (std == 0).
    """
    import torch

    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    traj_rewards = []
    for traj in group:
        if isinstance(traj, list):
            traj_rewards.append(traj[0].get_reward_value(args))
        else:
            traj_rewards.append(traj.get_reward_value(args))

    keep = bool(torch.tensor(traj_rewards, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(float(traj_rewards[0]), 1)}",
    )


def _format_observation_with_history(
    obs: dict[str, Any],
    action_history: list[str],
    max_steps: int,
    image_size: tuple[int, int] | None = None,
    history_window_size: int | None = None,
) -> dict:
    """Format observation as a standalone user message with full action history.

    Uses ANDROID_WORLD_TEMPLATE which includes task description and all prior
    actions, making each turn's prompt self-contained.

    Args:
        obs: Observation dict with 'image', 'task', etc.
        action_history: List of action summaries from prior steps.
        max_steps: Maximum allowed steps for the task.
        image_size: Optional (width, height) to resize the screenshot.
        history_window_size: If set, only include the last N actions in the
            history. None means include all actions.

    Returns:
        A chat message dict: {"role": "user", "content": [...]}
    """
    content: list[dict] = []

    # Screenshot first
    if obs and obs.get("image") is not None:
        image = obs["image"]
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        if image_size is not None:
            image = image.resize(tuple(image_size), PILImage.LANCZOS)
        content.append({"type": "image", "image": image})

    task = obs.get("task", "") if obs else ""
    step_count = len(action_history)
    windowed_history = (
        action_history[-history_window_size:] if history_window_size is not None else action_history
    )
    # Offset step numbers so they reflect the original step index in the trajectory
    history_offset = step_count - len(windowed_history)
    action_history_str = (
        "\n".join(
            f"Step {history_offset + i + 1}: {action}" for i, action in enumerate(windowed_history)
        )
        if windowed_history
        else "None"
    )
    text = ANDROID_WORLD_TEMPLATE.format(
        task_description=task,
        step_count=step_count,
        max_steps=max_steps,
        action_history=action_history_str,
        current_step=step_count + 1,
    )
    content.append({"type": "text", "text": text})

    return {"role": "user", "content": content}


def _build_step_prompt(
    tokenizer,
    processor,
    system_prompt: str,
    step_user_message: dict,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """Build a standalone tokenized prompt for one trajectory step.

    Wraps the system prompt and user message into a fresh conversation and
    applies the chat template, producing independent prompt IDs for each step.

    Returns:
        (prompt_text, prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        step_user_message,
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

        images, _ = process_vision_info([step_user_message])
        multimodal_inputs = {"images": images}
        processor_output = processor(text=prompt_text, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v
            for k, v in processor_output.items()
            if k not in ["input_ids", "attention_mask"] and "video" not in k
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
    - rollout_log_probs = response_logprobs  (response-only, NOT full sequence)
    - response_length = len(response_tokens)

    loss_mask and rollout_log_probs must have length == response_length (the
    assertion in _convert_samples_to_train_data checks this).
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
    """Extract the <conclusion> block from the model response as an action summary.

    Falls back to a truncated version of the raw response if no conclusion is
    found. Leading/trailing whitespace and special symbols are stripped.

    Args:
        response_text: Raw model response text.

    Returns:
        A cleaned string for inclusion in action_history.
    """
    match = re.search(r"<conclusion>(.*?)</conclusion>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text[:200].strip()


def post_process_rewards_history(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """GRPO reward normalization that correctly handles variable-length trajectories.

    The default _post_process_rewards in slime/ray/rollout.py reshapes rewards as
    (-1, n_samples_per_prompt) and falls back to (1, total) when total != N*B. With
    history-based rollout, each trajectory produces a variable number of step-samples
    (all sharing the same reward), so the total is always != N*B. The fallback
    normalizes all samples as one group, mixing prompts and biasing toward longer
    trajectories.

    This function fixes three problems:
    1. Cross-prompt normalization: different prompts are never mixed.
    2. Length bias: each trajectory contributes equally regardless of step count.
    3. Cross-trajectory mixing: normalization is strictly per-prompt across N trajectories.

    Groups samples by group_index (prompt), identifies trajectories by index
    (sample copy within group), computes per-trajectory reward, normalizes
    across N trajectories, and broadcasts back to all step-samples.
    """
    import torch

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
            reward = raw_rewards[positions[0]]  # all steps have the same reward
            traj_rewards.append(reward)
            traj_positions.append(positions)

        # Normalize across N trajectories for this prompt
        tr = torch.tensor(traj_rewards, dtype=torch.float)
        tr = tr - tr.mean()
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            # std() with Bessel correction on a single element returns nan;
            # guard against this (can happen when trimming splits a group)
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
    """Non-incremental multi-turn rollout for Android World.

    Each turn is an independent training sample: (full_history_prompt + screenshot_t
    → response_t). A trajectory of length T returns T Sample objects.

    Invoked by slime when --custom-generate-function-path points here.

    Args:
        args: Parsed slime arguments.
        sample: Input sample (carries task metadata; prompt is overridden per step).
        sampling_params: SGLang sampling parameters.
        evaluation: When True (set by slime during eval), return a single Sample
            instead of a list so that eval_rollout_single_dataset can log
            sample.prompt and compute_pass_rate gets one reward per prompt.

    Returns:
        list[Sample] during training (one per trajectory step), or a single Sample
        during evaluation.
    """
    assert not getattr(args, "partial_rollout", False), (
        "Partial rollout is not supported for Android World history-based rollouts."
    )

    pool = await AndroidWorldEnvPool.get_instance(vars(args))
    worker_ref, worker_id = await pool.acquire()

    task_name = (sample.metadata or {}).get("task_name")
    params_idx = (sample.metadata or {}).get("params_idx", 0)
    max_turns = getattr(args, "max_turns", 15)
    image_size = getattr(args, "image_size", None)
    history_window_size = getattr(args, "history_window_size", None)
    apply_chat_template = getattr(args, "apply_chat_template", True)
    apply_chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", None)

    env = AndroidWorldEnv(
        worker_ref=worker_ref,
        worker_id=worker_id,
        pool=pool,
        task_name=task_name,
        params_idx=params_idx,
        max_turns=max_turns,
        image_size=image_size,
    )

    sampling_params = sampling_params.copy()

    try:
        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        # Reset environment
        obs, info = await env.reset()
        max_steps = info.get("max_steps", max_turns)

        action_history: list[str] = []
        trajectory_samples: list[Sample] = []

        for turn_idx in range(max_turns):
            # Build a standalone prompt for this step with full history
            step_user_msg = _format_observation_with_history(
                obs, action_history, max_steps, image_size, history_window_size
            )
            prompt_text, prompt_ids, image_data, mm_inputs, mm_train = _build_step_prompt(
                tokenizer=state.tokenizer,
                processor=state.processor,
                system_prompt=ANDROID_WORLD_SYSTEM_PROMPT,
                step_user_message=step_user_msg,
                apply_chat_template=apply_chat_template,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
            )

            # Compute per-step budget (independent of other steps)
            cur_params = sampling_params.copy()
            max_ctx = getattr(args, "max_context_len", None) or getattr(
                args, "rollout_max_context_len", None
            )
            if max_ctx is not None:
                step_budget = max_ctx - len(prompt_ids)
                if step_budget <= 0:
                    # Prompt already exceeds context limit; skip this step
                    logger.warning(
                        "Step %d prompt length %d exceeds max_context_len %d; skipping",
                        turn_idx,
                        len(prompt_ids),
                        max_ctx,
                    )
                    break
                cur_params["max_new_tokens"] = step_budget

            # Single-turn SGLang inference with fresh prompt (no accumulated context)
            response_text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
                url, list(prompt_ids), cur_params, image_data, state.tokenizer
            )

            # Build a standalone Sample for this step (reward assigned later)
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

            # Propagate truncated/aborted status but still record the sample
            if _should_stop_on_finish(turn_sample, finish_type):
                trajectory_samples.append(turn_sample)
                break

            trajectory_samples.append(turn_sample)

            # Step the environment
            step_obs, done, _step_info = await env.step(response_text)

            # Record action summary for next step's history
            action_history.append(_extract_action_summary(response_text))

            if done or step_obs is None:
                break

            obs = step_obs

            if turn_idx + 1 >= max_turns:
                break

        reward = env.get_reward()

        # For evaluation, return a single Sample with trajectory-level reward.
        # eval_rollout_single_dataset expects one Sample per prompt (not a list)
        # so it can log sample.prompt and compute_pass_rate gets the right count.
        if evaluation:
            final = trajectory_samples[-1] if trajectory_samples else sample
            final.reward = reward
            final.status = Sample.Status.COMPLETED
            return final

        # Assign the trajectory-level reward to every step sample
        for s in trajectory_samples:
            s.reward = reward

        if not trajectory_samples:
            # Degenerate case: no steps produced (e.g., env.reset failed mid-way)
            sample.reward = reward
            sample.status = Sample.Status.COMPLETED
            return [sample]

        return trajectory_samples

    finally:
        try:
            env.close()
        except Exception:
            pass
