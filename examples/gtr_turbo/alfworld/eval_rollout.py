"""Evaluation rollout for ALFWorld.

Returns one Sample per task trajectory so eval metrics are task-level rather
than step-level.
"""

from __future__ import annotations

import copy
from typing import Any

from examples.gtr_turbo.alfworld import rollout as train_rollout
from examples.gtr_turbo.alfworld.metrics import task_type_from_path
from slime.utils.types import Sample


async def generate(args: Any, sample: Sample, sampling_params, evaluation: bool = True) -> Sample:
    step_samples = await train_rollout.generate(args, sample, sampling_params)
    if not step_samples:
        result = copy.deepcopy(sample)
        result.reward = 0.0
        result.status = Sample.Status.ABORTED
        result.metadata = dict(result.metadata or {})
        result.metadata.setdefault("eval_return", 0.0)
        result.metadata.setdefault("eval_success", False)
        return result

    final_step = step_samples[-1]
    result = copy.deepcopy(final_step)
    result.index = sample.index
    result.metadata = dict(result.metadata or {})

    metrics = result.metadata.get("alfworld_metrics", {}) or {}
    success = bool(metrics.get("success", False))
    eval_return = float(result.metadata.get("env_reward", 0.0))
    task_file = result.metadata.get("task_file")
    task_type = result.metadata.get("task_type") or task_type_from_path(task_file)
    category = result.metadata.get("category")

    result.reward = 1.0 if success else 0.0
    result.status = Sample.Status.COMPLETED
    result.metadata["eval_success"] = success
    result.metadata["eval_return"] = eval_return
    result.metadata["eval_goal_condition_success_rate"] = float(
        metrics.get("goal_condition_success_rate", 0.0)
    )
    result.metadata["eval_steps"] = int(metrics.get("steps", 0))
    result.metadata["eval_illegal_actions"] = int(metrics.get("illegal_actions", 0))
    result.metadata["eval_total_actions"] = int(metrics.get("total_actions", 0))
    if task_type is not None:
        result.metadata["task_type"] = task_type
    if category is not None:
        result.metadata["category"] = category
    return result
