"""History-based rollout for ALFWorld.

Each turn produces an independent Sample with full conversation history.
"""

from __future__ import annotations

import copy
from typing import Any

from examples.gtr_turbo.alfworld.env_alfworld import build_env as async_build_env
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

SYSTEM_PROMPT = "You are a helpful assistant."


def _build_step_prompt(messages, tokenizer, processor, apply_chat_template, apply_chat_template_kwargs):
    apply_kwargs = apply_chat_template_kwargs or {}
    if apply_chat_template:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **apply_kwargs,
        )
    else:
        formatted = messages

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info
        images, videos = process_vision_info(messages)
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]

    return list(prompt_ids), image_data, multimodal_inputs, multimodal_train_inputs


def _make_turn_sample(base_sample, prompt_ids, response_ids, response_log_probs,
                      multimodal_inputs, multimodal_train_inputs, tokenizer,
                      reward, turn_idx, env_info=None):
    turn_sample = copy.deepcopy(base_sample)
    turn_sample.tokens = prompt_ids + response_ids
    turn_sample.loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    turn_sample.rollout_log_probs = [0.0] * len(prompt_ids) + list(response_log_probs)
    turn_sample.response_length = len(response_ids)
    turn_sample.response = tokenizer.decode(response_ids, skip_special_tokens=False)
    turn_sample.prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    turn_sample.multimodal_inputs = multimodal_inputs
    turn_sample.multimodal_train_inputs = multimodal_train_inputs
    turn_sample.reward = reward
    turn_sample.status = Sample.Status.COMPLETED
    turn_sample.metadata = turn_sample.metadata or {}
    turn_sample.metadata["turn_idx"] = turn_idx
    turn_sample.metadata["env_reward"] = float(reward)
    if env_info is not None:
        turn_sample.metadata["env_info"] = env_info
    return turn_sample


def post_process_rewards_history(args, samples, **kwargs):
    """Normalize rewards within trajectory groups."""
    all_rewards = []
    all_original = []
    for group in samples:
        group_rewards = [s.reward for s in group]
        mean_r = sum(group_rewards) / max(len(group_rewards), 1)
        std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / max(len(group_rewards), 1)) ** 0.5
        for s in group:
            normalized = (s.reward - mean_r) / std_r if std_r > 1e-8 else 0.0
            all_rewards.append(normalized)
            all_original.append(s.reward)
    return all_original, all_rewards


async def generate(args: Any, sample: Sample, sampling_params) -> list[Sample]:
    """History-based multi-turn rollout for ALFWorld.

    Wired via: --custom-generate-function-path examples.gtr_turbo.alfworld.rollout_history.generate
    """
    config = getattr(args, "custom_config", None) or {}
    if not config:
        for key in ["max_turns", "max_context_len", "repetition_penalty"]:
            if hasattr(args, key):
                config[key] = getattr(args, key)
    max_turns = config.get("max_turns", getattr(args, "max_turns", 40))

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    env = await async_build_env(sample, args)
    sample.metadata = sample.metadata or {}
    sampling_params = sampling_params.copy()

    # Apply repetition_penalty from custom config if set
    rep_penalty = config.get("repetition_penalty")
    if rep_penalty is not None:
        sampling_params["repetition_penalty"] = float(rep_penalty)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_samples = []

    try:
        task_file = sample.metadata.get("task_file") if sample.metadata else None
        obs, _ = env.reset(task_file=task_file)
        first_msg = env.format_observation(obs)
        messages.append(first_msg)

        for turn_idx in range(max_turns):
            prompt_ids, image_data, mm_inputs, mm_train = _build_step_prompt(
                messages, state.tokenizer, state.processor,
                args.apply_chat_template, getattr(args, "apply_chat_template_kwargs", None),
            )

            budget = None
            if args.rollout_max_context_len is not None:
                budget = args.rollout_max_context_len - len(prompt_ids)
            cur_params = sampling_params.copy()
            if budget is not None:
                cur_params["max_new_tokens"] = max(budget, 1)

            payload = {
                "input_ids": prompt_ids,
                "sampling_params": cur_params,
                "return_logprob": True,
            }
            if image_data:
                payload["image_data"] = image_data

            output = await post(url, payload)
            response_text = output["text"]

            if "output_token_logprobs" in output["meta_info"]:
                new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                new_tokens, new_log_probs = [], []

            obs, done, info = env.step(response_text)
            step_reward = info.get("reward", 0.0)

            turn_sample = _make_turn_sample(
                sample, prompt_ids, new_tokens, new_log_probs,
                mm_inputs, mm_train, state.tokenizer,
                reward=step_reward, turn_idx=turn_idx, env_info=info,
            )
            turn_samples.append(turn_sample)

            if done:
                break

            messages.append({"role": "assistant", "content": response_text})
            next_msg = env.format_observation(obs)
            messages.append(next_msg)

        # Sparse-reward fix: propagate episode outcome to all steps.
        # ALFWorld rewards are sparse (big reward only on task success),
        # so per-step rewards give no credit to preceding good actions.
        # Use the cumulative episode reward as outcome for every step.
        if turn_samples:
            episode_outcome = env.get_reward()
            for ts in turn_samples:
                ts.reward = episode_outcome
                ts.metadata["env_reward"] = float(episode_outcome)

        return turn_samples if turn_samples else [sample]
    finally:
        try:
            env.close()
        except Exception:
            pass
