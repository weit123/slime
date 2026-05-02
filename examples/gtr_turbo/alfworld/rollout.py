"""Step-level ALFWorld rollout for VLM RL training.

The environment interaction is multi-turn, but returned training samples are
single-step samples: current observation/image prompt + current action only.
"""

from __future__ import annotations

from typing import Any

import torch

from examples.gtr_turbo.alfworld.env_alfworld import build_env as async_build_env
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

SYSTEM_PROMPT = "You are a helpful assistant."

DUMMY_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "I am a user."},
]


def _to_token_list(token_ids) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        return [int(x) for x in token_ids.detach().cpu().tolist()]
    return [int(x) for x in token_ids]


def _encode_observation_for_generation(
    tokenizer, processor, message, metadata,
    apply_chat_template, apply_chat_template_kwargs,
    qwen3_vl_eval_prompt_format: bool = False,
):
    """Encode an observation turn into tokens and multimodal data."""
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}
    trim_length = 0

    if qwen3_vl_eval_prompt_format:
        formatted_prompt = _format_qwen3_vl_observation(message)
    elif apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES, tools=tools, tokenize=False,
            add_generation_prompt=False, **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message], tools=tools, tokenize=False,
            add_generation_prompt=True, **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info
        images, videos = process_vision_info([message])
        multimodal_inputs = {}
        if images:
            multimodal_inputs["images"] = images
        if videos:
            multimodal_inputs["videos"] = videos
        processor_text = [formatted_prompt] if qwen3_vl_eval_prompt_format else formatted_prompt
        processor_output = processor(text=processor_text, **multimodal_inputs)
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

    return _to_token_list(prompt_ids), image_data, multimodal_inputs, multimodal_train_inputs


def _format_qwen3_vl_observation(message: dict[str, Any]) -> str:
    """Use the same single-image Qwen3-VL chat string as the successful eval."""
    text_chunks = []
    has_image = False
    for item in message.get("content", []):
        if isinstance(item, dict) and item.get("type") == "image":
            has_image = True
        elif isinstance(item, dict) and item.get("type") == "text":
            text_chunks.append(str(item.get("text", "")))

    vision_prefix = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
    prompt_text = "\n".join(text_chunks)
    return (
        "<|im_start|>user\n"
        f"{vision_prefix}{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _merge_multimodal_train_inputs(chunks):
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


def _append_context_tokens(
    sample: Sample,
    response_tokens: list[int],
    token_ids,
    log_prob_value: float = 0.0,
) -> None:
    """Append non-trainable environment/user context after the dataset prompt."""
    token_ids = _to_token_list(token_ids)
    if not token_ids:
        return
    sample.tokens.extend(token_ids)
    response_tokens.extend(token_ids)
    sample.loss_mask.extend([0] * len(token_ids))
    sample.rollout_log_probs.extend([log_prob_value] * len(token_ids))
    sample.response_length = len(response_tokens)


def _merge_multimodal_inputs(sample: Sample, obs_mm) -> None:
    if not obs_mm:
        return
    if not sample.multimodal_inputs:
        sample.multimodal_inputs = obs_mm
        return
    if isinstance(sample.multimodal_inputs, dict) and isinstance(obs_mm, dict):
        for key, val in obs_mm.items():
            if val and key in sample.multimodal_inputs and isinstance(sample.multimodal_inputs[key], list):
                sample.multimodal_inputs[key].extend(val)
            elif val and key not in sample.multimodal_inputs:
                sample.multimodal_inputs[key] = val


def _message_text(message: dict[str, Any]) -> str:
    text_chunks = []
    for item in message.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            text_chunks.append(str(item.get("text", "")))
    return "\n".join(text_chunks)


def _build_step_sample(
    *,
    base_sample: Sample,
    obs_ids: list[int],
    obs_mm_train,
    message: dict[str, Any],
    response_text: str,
    response_tokens: list[int],
    response_log_probs: list[float],
    step_idx: int,
    info: dict[str, Any],
) -> Sample | None:
    if not response_tokens:
        return None

    step_sample = Sample(
        group_index=base_sample.group_index,
        index=(base_sample.index or 0) * 1000 + step_idx,
        prompt=_message_text(message),
        tokens=list(obs_ids) + list(response_tokens),
        response=response_text,
        response_length=len(response_tokens),
        label=base_sample.label,
        loss_mask=[1] * len(response_tokens),
        rollout_log_probs=list(response_log_probs),
        metadata=dict(base_sample.metadata or {}),
        generate_function_path=base_sample.generate_function_path,
    )
    step_sample.multimodal_train_inputs = obs_mm_train
    step_sample.status = Sample.Status.COMPLETED
    step_sample.metadata.update({
        "alfworld_step_index": step_idx,
        "alfworld_group_index": base_sample.group_index,
        "alfworld_trajectory_index": base_sample.index,
        "alfworld_action_taken": info.get("action_taken"),
        "alfworld_action_legal": bool(info.get("legal", False)),
        "alfworld_step_reward": float(info.get("reward", 0.0)),
    })
    _assert_rollout_alignment(step_sample)
    return step_sample


def _finalize_step_samples(
    step_samples: list[Sample],
    *,
    env,
    final_status: Sample.Status,
    final_info: dict[str, Any] | None = None,
) -> list[Sample]:
    env_reward = env.get_reward()
    env_metrics = env.get_metrics()
    for step_sample in step_samples:
        step_sample.status = final_status
        step_sample.reward = env_reward
        step_sample.metadata["env_reward"] = env_reward
        step_sample.metadata["raw_reward"] = env_reward
        if final_info is not None:
            step_sample.metadata["env_info"] = final_info

    if step_samples:
        # Log episode-level ALFWorld metrics once per trajectory, not once per step.
        step_samples[-1].metadata["alfworld_metrics"] = env_metrics
        step_samples[-1].metadata["alfworld_episode_final"] = True
    return step_samples


def _assert_rollout_alignment(sample: Sample) -> None:
    assert sample.loss_mask is not None
    assert sample.rollout_log_probs is not None
    assert len(sample.loss_mask) == sample.response_length, (
        f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}"
    )
    assert len(sample.rollout_log_probs) == sample.response_length, (
        f"rollout_log_probs length {len(sample.rollout_log_probs)} != response_length {sample.response_length}"
    )
    assert len(sample.tokens) >= sample.response_length, (
        f"tokens length {len(sample.tokens)} < response_length {sample.response_length}"
    )


async def generate(args: Any, sample: Sample, sampling_params) -> list[Sample]:
    """Custom multi-turn rollout for ALFWorld environment.

    Wired via: --custom-generate-function-path examples.gtr_turbo.alfworld.rollout.generate
    """
    assert not getattr(args, "partial_rollout", False)

    config = getattr(args, "custom_config", None) or {}
    if not config:
        for key in ["max_turns", "max_context_len", "repetition_penalty"]:
            if hasattr(args, key):
                config[key] = getattr(args, key)
    max_turns = config.get("max_turns", getattr(args, "max_turns", 40))
    max_action_tokens = int(config.get("max_action_tokens", 512))
    qwen3_vl_eval_prompt_format = bool(config.get("qwen3_vl_eval_prompt_format", True))
    ignore_dataset_prompt = bool(config.get("ignore_dataset_prompt", qwen3_vl_eval_prompt_format))
    env_reset_retries = int(config.get("env_reset_retries", 1))

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    env = await async_build_env(sample, args)
    sample.metadata = sample.metadata or {}
    sampling_params = sampling_params.copy()
    rep_penalty = config.get("repetition_penalty")
    if rep_penalty is not None:
        sampling_params["repetition_penalty"] = float(rep_penalty)

    if ignore_dataset_prompt:
        prompt_ids = []
        prompt_mm_train = None
    elif state.processor:
        processor_output = state.processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        prompt_mm_train = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
        prompt_mm_train = None

    prompt_image_data = []
    if not ignore_dataset_prompt and sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        prompt_image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]

    prompt_ids = _to_token_list(prompt_ids)

    try:
        step_samples: list[Sample] = []
        task_file = sample.metadata.get("task_file") if sample.metadata else None
        for reset_attempt in range(env_reset_retries + 1):
            try:
                obs, _ = env.reset(task_file=task_file)
                break
            except Exception as exc:
                if reset_attempt >= env_reset_retries:
                    raise
                sample.metadata["alfworld_reset_error"] = f"{type(exc).__name__}: {exc}"
                await env.restart_worker()
        init_msg = env.format_observation(obs)
        init_ids, init_img, init_mm, init_obs_mm_train = _encode_observation_for_generation(
            state.tokenizer, state.processor, init_msg, sample.metadata,
            args.apply_chat_template, getattr(args, "apply_chat_template_kwargs", None),
            qwen3_vl_eval_prompt_format=qwen3_vl_eval_prompt_format,
        )
        bos_id = state.tokenizer.bos_token_id
        if bos_id is not None and init_ids and init_ids[0] == bos_id:
            init_ids = init_ids[1:]

        current_msg = init_msg
        current_obs_ids = init_ids
        current_obs_image_data = init_img
        current_obs_mm_train = init_obs_mm_train

        for turn_idx in range(max_turns):
            input_ids = prompt_ids + current_obs_ids
            context_budget = None
            if args.rollout_max_context_len is not None:
                context_budget = int(args.rollout_max_context_len) - len(input_ids)
            elif sampling_params.get("max_new_tokens") is not None:
                context_budget = int(sampling_params["max_new_tokens"])
            if context_budget is not None and context_budget <= 0:
                return _finalize_step_samples(step_samples, env=env, final_status=Sample.Status.TRUNCATED)

            cur_params = sampling_params.copy()
            if context_budget is not None:
                cur_params["max_new_tokens"] = min(context_budget, max_action_tokens)
            else:
                cur_params["max_new_tokens"] = min(
                    int(cur_params.get("max_new_tokens", max_action_tokens)),
                    max_action_tokens,
                )

            payload = {
                "input_ids": input_ids,
                "sampling_params": cur_params,
                "return_logprob": True,
            }
            image_data = prompt_image_data + current_obs_image_data
            if image_data:
                payload["image_data"] = image_data

            output = await post(url, payload)
            response_text = output["text"]

            if "output_token_logprobs" in output["meta_info"]:
                new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                new_tokens, new_log_probs = [], []

            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "abort":
                return _finalize_step_samples(step_samples, env=env, final_status=Sample.Status.ABORTED)

            obs, done, info = env.step(response_text)
            step_sample = _build_step_sample(
                base_sample=sample,
                obs_ids=input_ids,
                obs_mm_train=_merge_multimodal_train_inputs([prompt_mm_train, current_obs_mm_train]),
                message=current_msg,
                response_text=response_text,
                response_tokens=new_tokens,
                response_log_probs=new_log_probs,
                step_idx=turn_idx,
                info=info,
            )
            if step_sample is not None:
                step_samples.append(step_sample)

            if done:
                return _finalize_step_samples(
                    step_samples,
                    env=env,
                    final_status=Sample.Status.COMPLETED,
                    final_info=info,
                )
            if finish_type == "length":
                return _finalize_step_samples(step_samples, env=env, final_status=Sample.Status.TRUNCATED)

            next_msg = env.format_observation(obs)
            obs_ids, obs_img, obs_mm, obs_mm_train = _encode_observation_for_generation(
                state.tokenizer, state.processor, next_msg, sample.metadata,
                args.apply_chat_template, getattr(args, "apply_chat_template_kwargs", None),
                qwen3_vl_eval_prompt_format=qwen3_vl_eval_prompt_format,
            )

            bos_id = state.tokenizer.bos_token_id
            if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
                obs_ids = obs_ids[1:]

            current_msg = next_msg
            current_obs_ids = obs_ids
            current_obs_image_data = obs_img
            current_obs_mm_train = obs_mm_train

            if turn_idx + 1 >= max_turns:
                return _finalize_step_samples(step_samples, env=env, final_status=Sample.Status.COMPLETED)

        return _finalize_step_samples(step_samples, env=env, final_status=Sample.Status.COMPLETED)
    finally:
        try:
            env.close()
        except Exception:
            pass
