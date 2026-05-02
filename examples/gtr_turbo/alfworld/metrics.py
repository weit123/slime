"""ALFWorld rollout metrics for slime logging."""

from __future__ import annotations

from typing import Any

import numpy as np


TASK_TYPE_TO_CATEGORY = {
    "pick_and_place_simple": "Pick & Place",
    "pick_two_obj_and_place": "Pick Two & Place",
    "pick_clean_then_place_in_recep": "Clean & Place",
    "pick_heat_then_place_in_recep": "Heat & Place",
    "pick_cool_then_place_in_recep": "Cool & Place",
    "look_at_obj_in_light": "Examine in Light",
}


def _mean(values: list[float]) -> float:
    return float(np.mean(values).item()) if values else 0.0


def _task_type_from_path(task_file: str | None) -> str | None:
    if not task_file:
        return None
    task_dir = task_file.split("/")[-3] if "/" in task_file else task_file
    for task_type in TASK_TYPE_TO_CATEGORY:
        if task_dir.startswith(task_type):
            return task_type
    return None


def collect_alfworld_metrics(samples) -> dict[str, Any]:
    sample_metric_pairs = [
        (sample, sample.metadata.get("alfworld_metrics", {}))
        for sample in samples
        if sample.metadata and sample.metadata.get("alfworld_metrics") is not None
    ]
    metrics_by_sample = [item for _, item in sample_metric_pairs]
    if not metrics_by_sample:
        return {}

    total_actions = sum(int(item.get("total_actions", 0)) for item in metrics_by_sample)
    illegal_actions = sum(int(item.get("illegal_actions", 0)) for item in metrics_by_sample)
    log_dict: dict[str, Any] = {
        "alfworld/success_rate": _mean([float(item.get("success", False)) for item in metrics_by_sample]),
        "alfworld/mean_goal_condition_success_rate": _mean(
            [float(item.get("goal_condition_success_rate", 0.0)) for item in metrics_by_sample]
        ),
        "alfworld/mean_steps": _mean([float(item.get("steps", 0)) for item in metrics_by_sample]),
        "alfworld/illegal_action_rate": illegal_actions / total_actions if total_actions else 0.0,
    }

    category_to_metrics: dict[str, list[dict[str, Any]]] = {}
    for sample, item in sample_metric_pairs:
        task_type = _task_type_from_path(sample.metadata.get("task_file") if sample.metadata else None)
        category = TASK_TYPE_TO_CATEGORY.get(task_type or "")
        if category:
            category_to_metrics.setdefault(category, []).append(item)

    for category, items in category_to_metrics.items():
        key = category.lower().replace(" & ", "_").replace(" ", "_")
        log_dict[f"alfworld/{key}/success_rate"] = _mean([float(item.get("success", False)) for item in items])
        log_dict[f"alfworld/{key}/mean_goal_condition_success_rate"] = _mean(
            [float(item.get("goal_condition_success_rate", 0.0)) for item in items]
        )

    return log_dict


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Inject ALFWorld metrics into slime's normal rollout logger."""
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(collect_alfworld_metrics(samples))
    return False
