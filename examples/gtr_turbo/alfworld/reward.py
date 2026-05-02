"""Reward function for ALFWorld environment.

Wired via: --custom-rm-path examples.gtr_turbo.alfworld.reward.reward_func
"""

from __future__ import annotations

import torch


async def reward_func(args, sample, **kwargs):
    """Return the environment reward stored during rollout."""
    if isinstance(sample, list):
        return [await reward_func(args, item, **kwargs) for item in sample]
    if sample.metadata and "env_reward" in sample.metadata:
        return sample.metadata["env_reward"]
    return 0.0


def post_process_step_rewards(args, samples):
    """GRPO-compatible reward normalization for step-level ALFWorld samples.

    The generic slime postprocess assumes flattened samples are ordered in
    prompt groups. ALFWorld returns multiple step samples per trajectory, so we
    normalize across rollouts from the same original prompt and step index.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

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

    return raw_rewards, rewards
