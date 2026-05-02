"""Multi-turn incremental rollout for Points24.

One Sample per trajectory: context grows each turn with model responses
and environment observations. Follows the pattern in
examples/geo3k_vlm_multi_turn/rollout.py.
"""

from __future__ import annotations

from typing import Any

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from examples.gtr_turbo.points24.env_points24 import build_env
from examples.gtr_turbo.points24.prompts import SYSTEM_PROMPT
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

DUMMY_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "I am a user."},
]


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """Encode a single observation turn into tokens and multimodal data."""
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0
    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]

    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


import torch


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    if not chunks:
        return None
    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is not None:
                values_by_key.setdefault(key, []).append(val)
    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)
    return merged or None


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout for Points24 environment.

    Wired via: --custom-generate-function-path examples.gtr_turbo.points24.rollout.generate
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout not supported"

    config = getattr(args, "custom_config", {}) or {}
    max_turns = config.get("max_turns", args.max_turns or 20)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    env = build_env(sample, args)
    sample.metadata = sample.metadata or {}
    sampling_params = sampling_params.copy()

    if state.processor:
        processor_output = state.processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        init_mm_train = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
        init_mm_train = None

    current_image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        current_image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]

    mm_train_buffer: list[dict | None] = []
    if init_mm_train:
        mm_train_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids):] if len(sample.tokens) >= len(prompt_ids) else []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)

    try:
        env.reset()
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        for turn_idx in range(max_turns):
            cur_params = sampling_params.copy()
            if budget is not None:
                cur_params["max_new_tokens"] = budget

            payload = {
                "input_ids": sample.tokens,
                "sampling_params": cur_params,
                "return_logprob": True,
            }
            if current_image_data:
                payload["image_data"] = current_image_data

            output = await post(url, payload)
            response_text = output["text"]

            if "output_token_logprobs" in output["meta_info"]:
                new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                new_tokens, new_log_probs = [], []

            sample.tokens.extend(new_tokens)
            response_tokens.extend(new_tokens)
            sample.loss_mask.extend([1] * len(new_tokens))
            sample.rollout_log_probs.extend(new_log_probs)
            sample.response_length = len(response_tokens)

            if budget is not None:
                budget -= len(new_tokens)

            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                break
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            obs, done, info = env.step(response_text)
            if done:
                sample.status = Sample.Status.COMPLETED
                sample.metadata["env_reward"] = env.get_reward()
                sample.metadata["env_info"] = info
                break

            next_msg = env.format_observation(obs)
            obs_ids, obs_img, obs_mm, obs_mm_train, *_ = _encode_observation_for_generation(
                state.tokenizer,
                state.processor,
                next_msg,
                sample.metadata,
                args.apply_chat_template,
                getattr(args, "apply_chat_template_kwargs", None),
            )

            bos_id = state.tokenizer.bos_token_id
            if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
                obs_ids = obs_ids[1:]

            sample.tokens.extend(obs_ids)
            response_tokens.extend(obs_ids)
            sample.loss_mask.extend([0] * len(obs_ids))
            sample.rollout_log_probs.extend([0.0] * len(obs_ids))
            sample.response_length = len(response_tokens)

            if budget is not None:
                budget -= len(obs_ids)

            if obs_img:
                current_image_data = (current_image_data or []) + obs_img
            if obs_mm:
                if not sample.multimodal_inputs:
                    sample.multimodal_inputs = obs_mm
                elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_mm, dict):
                    for key, val in obs_mm.items():
                        if val and key in sample.multimodal_inputs and isinstance(sample.multimodal_inputs[key], list):
                            sample.multimodal_inputs[key].extend(val)
            if obs_mm_train:
                mm_train_buffer.append(obs_mm_train)

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= max_turns:
                sample.metadata["env_reward"] = env.get_reward()
                sample.status = Sample.Status.COMPLETED
                break

        if "env_reward" not in sample.metadata:
            sample.metadata["env_reward"] = env.get_reward()

        sample.multimodal_train_inputs = _merge_multimodal_train_inputs(mm_train_buffer)
        sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)
        if sample.status is None:
            sample.status = Sample.Status.COMPLETED
        return sample
    finally:
        try:
            env.close()
        except Exception:
            pass
