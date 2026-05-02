"""GTR-Turbo reward post-processor.

Combines OPD teacher log-probs extraction with environment task rewards.
Standard OPD returns 0.0 (pure distillation); GTR-Turbo needs task rewards
from the environment PLUS teacher log-probs for KL penalty.

Wired via:
  --custom-rm-path examples.gtr_turbo.gtr_turbo_train.gtr_turbo_reward.reward_func
  --custom-reward-post-process-path examples.gtr_turbo.gtr_turbo_train.gtr_turbo_reward.post_process_rewards
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os

import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_teacher_cycle = None
_teacher_lock = asyncio.Lock()
_teacher_semaphores: dict[str, asyncio.Semaphore] = {}
_teacher_session: aiohttp.ClientSession | None = None


def _teacher_urls(args) -> list[str]:
    raw = os.environ.get("GTR_TEACHER_URLS") or getattr(args, "rm_url", "")
    urls = [item.strip() for item in raw.split(",") if item.strip()]
    if not urls:
        raise ValueError("No teacher URLs configured. Set GTR_TEACHER_URLS or --rm-url.")
    return urls


async def _next_teacher_url(args) -> str:
    global _teacher_cycle
    urls = _teacher_urls(args)
    async with _teacher_lock:
        if _teacher_cycle is None:
            _teacher_cycle = itertools.cycle(urls)
        return next(_teacher_cycle)


def _teacher_limit(args) -> int:
    return int(os.environ.get("GTR_TEACHER_MAX_CONCURRENCY_PER_SERVER", "4"))


async def _get_teacher_session() -> aiohttp.ClientSession:
    global _teacher_session
    if _teacher_session is None or _teacher_session.closed:
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_connect=60)
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, enable_cleanup_closed=True)
        _teacher_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _teacher_session


def _build_teacher_payload(sample: Sample) -> dict:
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image)
            for image in sample.multimodal_inputs["images"]
        ]
    return payload


async def _teacher_reward(args, sample: Sample):
    payload = _build_teacher_payload(sample)
    last_error: Exception | None = None
    for attempt in range(3):
        url = await _next_teacher_url(args)
        semaphore = _teacher_semaphores.setdefault(url, asyncio.Semaphore(_teacher_limit(args)))
        try:
            async with semaphore:
                session = await _get_teacher_session()
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(1.0 + attempt)
    raise last_error

async def reward_func(args, sample, **kwargs):
    """Batch-aware OPD teacher reward for ALFWorld step-level rollouts."""
    if isinstance(sample, list):
        return [await reward_func(args, item, **kwargs) for item in sample]
    return await _teacher_reward(args, sample)


def _extract_teacher_log_probs(raw_reward, response_length: int) -> torch.Tensor:
    """Extract teacher log-probs from the OPD reward_func response.

    The raw_reward is the sglang server response containing
    meta_info.input_token_logprobs.
    """
    log_probs = torch.tensor(
        [item[0] for item in raw_reward["meta_info"]["input_token_logprobs"][1:]],
        dtype=torch.float32,
    )
    if log_probs.numel() < response_length:
        raise ValueError(
            f"Teacher returned {log_probs.numel()} token logprobs, "
            f"but response_length is {response_length}."
        )
    return log_probs[-response_length:]


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards combining OPD teacher log-probs with environment task rewards.

    This function:
    1. Extracts teacher log-probs from the OPD reward response (same as standard OPD)
    2. Stores them in sample.teacher_log_probs for KL penalty computation
    3. Returns the ENVIRONMENT task reward (not 0.0) as the scalar reward

    The KL penalty against the merged teacher is applied during
    compute_advantages_and_returns() via the standard OPD KL mechanism.
    """
    raw_teacher_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = [
        _extract_teacher_log_probs(raw_teacher_reward, resp_len)
        for raw_teacher_reward, resp_len in zip(raw_teacher_rewards, response_lengths)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs):
        if sample.loss_mask is None:
            raise ValueError("GTR-Turbo OPD requires sample.loss_mask for multi-turn alignment.")
        if len(sample.loss_mask) != sample.response_length:
            raise ValueError(
                f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}"
            )
        if len(t_log_probs) != sample.response_length:
            raise ValueError(
                f"teacher_log_probs length {len(t_log_probs)} != response_length {sample.response_length}"
            )
        if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) != sample.response_length:
            raise ValueError(
                f"rollout_log_probs length {len(sample.rollout_log_probs)} != "
                f"response_length {sample.response_length}"
            )
        sample.teacher_log_probs = t_log_probs

    raw_task_rewards = []
    for sample in samples:
        env_reward = 0.0
        if sample.metadata and "env_reward" in sample.metadata:
            env_reward = float(sample.metadata["env_reward"])
        raw_task_rewards.append(env_reward)

    task_rewards = _normalize_task_rewards_for_grpo(args, samples, raw_task_rewards)

    logger.info(
        "GTR-Turbo rewards: %d samples, mean_raw_task_reward=%.3f, mean_norm_task_reward=%.3f, "
        "mean_teacher_logprob_len=%.1f",
        len(samples),
        sum(raw_task_rewards) / max(len(raw_task_rewards), 1),
        sum(task_rewards) / max(len(task_rewards), 1),
        sum(len(t) for t in teacher_log_probs) / max(len(teacher_log_probs), 1),
    )

    return raw_task_rewards, task_rewards


def _normalize_task_rewards_for_grpo(args, samples: list[Sample], raw_rewards: list[float]) -> list[float]:
    """Normalize ALFWorld step rewards by original prompt and step index.

    GTR-Turbo uses OPD teacher log-probs as an auxiliary KL term, but the
    scalar reward should still follow the same step-level GRPO grouping as the
    ALFWorld baseline.
    """
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards

    grouped: dict[tuple[int | None, int | None], list[int]] = {}
    for idx, sample in enumerate(samples):
        metadata = sample.metadata or {}
        key = (
            metadata.get("alfworld_group_index", sample.group_index),
            metadata.get("alfworld_step_index"),
        )
        grouped.setdefault(key, []).append(idx)

    rewards = [0.0] * len(samples)
    for indices in grouped.values():
        values = torch.tensor([raw_rewards[i] for i in indices], dtype=torch.float)
        values = values - values.mean()
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization and values.numel() > 1:
            values = values / (values.std(unbiased=False) + 1e-6)
        for idx, value in zip(indices, values.tolist(), strict=False):
            rewards[idx] = value

    return rewards
