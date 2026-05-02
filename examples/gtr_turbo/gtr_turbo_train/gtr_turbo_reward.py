"""GTR-Turbo reward post-processor.

Combines OPD teacher log-probs extraction with environment task rewards.
Standard OPD returns 0.0 (pure distillation); GTR-Turbo needs task rewards
from the environment PLUS teacher log-probs for KL penalty.

Wired via:
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path examples.gtr_turbo.gtr_turbo_train.gtr_turbo_reward.post_process_rewards
"""

from __future__ import annotations

import logging

import torch

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


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
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = [
        _extract_teacher_log_probs(raw_reward, resp_len)
        for raw_reward, resp_len in zip(raw_rewards, response_lengths)
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

    task_rewards = []
    for sample in samples:
        env_reward = 0.0
        if sample.metadata and "env_reward" in sample.metadata:
            env_reward = float(sample.metadata["env_reward"])
        task_rewards.append(env_reward)

    logger.info(
        "GTR-Turbo rewards: %d samples, mean_task_reward=%.3f, mean_teacher_logprob_len=%.1f",
        len(samples),
        sum(task_rewards) / max(len(task_rewards), 1),
        sum(len(t) for t in teacher_log_probs) / max(len(teacher_log_probs), 1),
    )

    return task_rewards, task_rewards
