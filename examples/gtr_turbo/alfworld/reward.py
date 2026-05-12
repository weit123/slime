"""Reward function for ALFWorld environment.

Wired via: --custom-rm-path examples.gtr_turbo.alfworld.reward.reward_func
"""

from __future__ import annotations

import torch


def _env_reward_value(args, sample) -> float:
    """Read the ALFWorld trajectory reward, independent of OPD teacher payloads."""
    if sample.metadata and "env_reward" in sample.metadata:
        return float(sample.metadata["env_reward"])
    return float(sample.get_reward_value(args))


def _trajectory_key(sample) -> tuple[int | None, int | None]:
    metadata = sample.metadata or {}
    return (
        metadata.get("alfworld_group_index", sample.group_index),
        metadata.get("alfworld_trajectory_index", sample.index),
    )


async def reward_func(args, sample, **kwargs):
    """Return the environment reward stored during rollout."""
    if isinstance(sample, list):
        return [await reward_func(args, item, **kwargs) for item in sample]
    if sample.metadata and "env_reward" in sample.metadata:
        return sample.metadata["env_reward"]
    return 0.0


def post_process_step_rewards(args, samples):
    """GRPO-compatible reward normalization for step-level ALFWorld samples.

    ALFWorld returns multiple step samples per trajectory. Match the Android
    World history rollout semantics: normalize one trajectory-level reward per
    prompt across the GRPO samples, then broadcast that normalized reward to
    every step sample from the same trajectory.
    """
    raw_rewards = [_env_reward_value(args, sample) for sample in samples]
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    prompt_groups: dict[int | None, dict[int | None, list[int]]] = {}
    for idx, sample in enumerate(samples):
        group_id, traj_id = _trajectory_key(sample)
        prompt_groups.setdefault(group_id, {}).setdefault(traj_id, []).append(idx)

    rewards = [0.0] * len(samples)
    for trajectories in prompt_groups.values():
        traj_positions = list(trajectories.values())
        traj_rewards = [raw_rewards[positions[0]] for positions in traj_positions]
        values = torch.tensor(traj_rewards, dtype=torch.float)
        values = values - values.mean()
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization and values.numel() > 1:
            values = values / (values.std() + 1e-6)
        for value, positions in zip(values.tolist(), traj_positions, strict=False):
            for idx in positions:
                rewards[idx] = value

    return raw_rewards, rewards


def check_reward_nonzero_std(args, group, **kwargs):
    """Drop GRPO groups whose trajectory rewards have no variance."""
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    traj_rewards = []
    for traj in group:
        if isinstance(traj, list):
            if not traj:
                return DynamicFilterOutput(keep=False, reason="empty_trajectory")
            traj_rewards.append(_env_reward_value(args, traj[0]))
        else:
            traj_rewards.append(_env_reward_value(args, traj))

    keep = bool(torch.tensor(traj_rewards, dtype=torch.float64).std() > 1e-6)
    reason = None if keep else f"zero_std_{round(float(traj_rewards[0]), 1)}"
    return DynamicFilterOutput(keep=keep, reason=reason)
