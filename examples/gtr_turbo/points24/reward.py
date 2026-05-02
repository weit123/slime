"""Reward function for Points24 environment.

Extracts the environment reward stored in sample.metadata by the rollout function.
Wired via: --custom-rm-path examples.gtr_turbo.points24.reward.reward_func
"""

from __future__ import annotations


async def reward_func(args, sample, **kwargs):
    """Return the environment reward stored during rollout."""
    if sample.metadata and "env_reward" in sample.metadata:
        return sample.metadata["env_reward"]
    return 0.0
