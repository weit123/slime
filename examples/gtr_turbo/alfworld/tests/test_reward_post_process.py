"""Unit tests for ALFWorld trajectory-level GRPO reward processing."""

from __future__ import annotations

from argparse import Namespace

from slime.utils.types import Sample


def _args(**overrides):
    defaults = dict(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        reward_key=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _sample(group: int, traj: int, step: int, reward: float) -> Sample:
    return Sample(
        group_index=group,
        index=traj * 1000 + step,
        reward=reward,
        tokens=[1, 2, 3],
        response_length=1,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        metadata={
            "alfworld_group_index": group,
            "alfworld_trajectory_index": traj,
            "alfworld_step_index": step,
            "env_reward": reward,
        },
    )


def test_post_process_step_rewards_normalizes_by_trajectory():
    from examples.gtr_turbo.alfworld.reward import post_process_step_rewards

    samples = [
        _sample(0, 0, 0, 1.0),
        _sample(0, 0, 1, 1.0),
        _sample(0, 1, 0, 0.0),
        _sample(0, 1, 1, 0.0),
    ]

    raw, normalized = post_process_step_rewards(_args(grpo_std_normalization=False), samples)

    assert raw == [1.0, 1.0, 0.0, 0.0]
    assert normalized == [0.5, 0.5, -0.5, -0.5]


def test_post_process_step_rewards_keeps_prompt_groups_separate():
    from examples.gtr_turbo.alfworld.reward import post_process_step_rewards

    samples = [
        _sample(0, 0, 0, 1.0),
        _sample(0, 1, 0, 0.0),
        _sample(1, 0, 0, 10.0),
        _sample(1, 1, 0, 8.0),
    ]

    _, normalized = post_process_step_rewards(_args(grpo_std_normalization=False), samples)

    assert normalized == [0.5, -0.5, 1.0, -1.0]


def test_check_reward_nonzero_std_uses_env_reward_metadata_for_opd_samples():
    from examples.gtr_turbo.alfworld.reward import check_reward_nonzero_std

    group = [
        [_sample(0, 0, 0, 1.0), _sample(0, 0, 1, 1.0)],
        [_sample(0, 1, 0, 0.0)],
    ]
    for traj in group:
        for sample in traj:
            sample.reward = {"teacher": "payload"}

    result = check_reward_nonzero_std(_args(), group)

    assert result.keep is True
    assert result.reason is None


def test_check_reward_nonzero_std_drops_uniform_rewards():
    from examples.gtr_turbo.alfworld.reward import check_reward_nonzero_std

    group = [
        [_sample(0, 0, 0, 0.0)],
        [_sample(0, 1, 0, 0.0), _sample(0, 1, 1, 0.0)],
    ]

    result = check_reward_nonzero_std(_args(), group)

    assert result.keep is False
    assert result.reason == "zero_std_0.0"


def test_gtr_turbo_reward_normalizes_by_trajectory():
    from examples.gtr_turbo.gtr_turbo_train.gtr_turbo_reward import _normalize_task_rewards_for_grpo

    samples = [
        _sample(0, 0, 0, 1.0),
        _sample(0, 0, 1, 1.0),
        _sample(0, 1, 0, 0.0),
    ]

    normalized = _normalize_task_rewards_for_grpo(
        _args(grpo_std_normalization=False),
        samples,
        [1.0, 1.0, 0.0],
    )

    assert normalized == [0.5, 0.5, -0.5]
