#!/usr/bin/env python3
"""Build lightweight ALFWorld task JSONL datasets.

This module owns the shared, non-oracle parts of ALFWorld dataset generation:
task discovery, task category metadata, stable ordering, and prompt JSONL
writing.  The oracle scripts import these helpers and add environment rollout
validation on top.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

TASK_TYPE_BY_ID = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

TASK_TYPE_TO_CATEGORY = {
    "pick_and_place_simple": "Pick & Place",
    "pick_two_obj_and_place": "Pick Two & Place",
    "pick_clean_then_place_in_recep": "Clean & Place",
    "pick_heat_then_place_in_recep": "Heat & Place",
    "pick_cool_then_place_in_recep": "Cool & Place",
    "look_at_obj_in_light": "Examine in Light",
}

CATEGORY_ORDER = [
    "Pick & Place",
    "Pick Two & Place",
    "Clean & Place",
    "Heat & Place",
    "Cool & Place",
    "Examine in Light",
]

DEFAULT_PROMPT = "Begin ALFWorld task."


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def task_sort_key(task: dict[str, Any]) -> tuple[int, str]:
    category = task.get("category", "")
    try:
        category_index = CATEGORY_ORDER.index(category)
    except ValueError:
        category_index = len(CATEGORY_ORDER)
    return category_index, str(task.get("task_file", ""))


def flatten_tasks(tasks_by_category: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [task for category in CATEGORY_ORDER for task in tasks_by_category.get(category, [])]


def allowed_task_types(task_type_ids: Iterable[int] | None = None) -> set[str]:
    ids = list(task_type_ids or sorted(TASK_TYPE_BY_ID))
    unknown = [task_id for task_id in ids if task_id not in TASK_TYPE_BY_ID]
    if unknown:
        raise ValueError(f"Unknown ALFWorld task type ids: {unknown}")
    return {TASK_TYPE_BY_ID[task_id] for task_id in ids}


def is_supported_task_dir(path: Path, *, include_movable: bool = False, include_sliced: bool = False) -> bool:
    path_text = str(path)
    if not include_movable and "movable" in path_text:
        return False
    if not include_sliced and "Sliced" in path_text:
        return False
    return True


def load_task(task_file: str | Path) -> dict[str, str] | None:
    task_path = Path(task_file)
    with task_path.open(encoding="utf-8") as f:
        traj = json.load(f)

    task_type = traj.get("task_type", "")
    category = TASK_TYPE_TO_CATEGORY.get(task_type)
    if category is None:
        return None
    return {
        "task_file": str(task_path),
        "task_type": task_type,
        "category": category,
    }


def discover_tasks(
    data_path: str | Path,
    *,
    task_types: set[str] | None = None,
    include_movable: bool = False,
    include_sliced: bool = False,
) -> dict[str, list[dict[str, str]]]:
    tasks_by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for traj_path in sorted(Path(data_path).rglob("traj_data.json")):
        if not is_supported_task_dir(traj_path.parent, include_movable=include_movable, include_sliced=include_sliced):
            continue
        task = load_task(traj_path)
        if task is None:
            continue
        if task_types is not None and task["task_type"] not in task_types:
            continue
        tasks_by_category[task["category"]].append(task)
    return tasks_by_category


def write_prompt_jsonl(
    output: str | Path,
    tasks: list[dict[str, Any]],
    *,
    prompt_content: str = DEFAULT_PROMPT,
    id_prefix: str = "alfworld",
    metadata_keys: tuple[str, ...] = ("task_file",),
    metadata_extra: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
    include_label: bool = False,
    sort: bool = True,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_tasks = sorted(tasks, key=task_sort_key) if sort else list(tasks)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, task in enumerate(ordered_tasks):
            metadata = {key: task[key] for key in metadata_keys if key in task}
            if metadata_extra is not None:
                metadata.update(metadata_extra(task, idx))
            record: dict[str, Any] = {
                "prompt": [{"role": "user", "content": prompt_content}],
                "metadata": metadata,
                "id": f"{id_prefix}_{idx}",
            }
            if include_label:
                record["label"] = ""
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to an ALFWorld split directory, e.g. json_2.1.1/train")
    parser.add_argument("--output", default="examples/gtr_turbo/alfworld/data/alfworld_prompts.jsonl")
    parser.add_argument("--task-types", nargs="+", type=int, default=sorted(TASK_TYPE_BY_ID))
    parser.add_argument("--prompt-content", default=DEFAULT_PROMPT)
    parser.add_argument("--id-prefix", default="alfworld")
    parser.add_argument("--metadata-keys", nargs="+", default=["task_file"])
    parser.add_argument("--include-label", action="store_true")
    args = parser.parse_args()

    task_types = allowed_task_types(args.task_types)
    tasks = flatten_tasks(discover_tasks(args.data_path, task_types=task_types))
    write_prompt_jsonl(
        args.output,
        tasks,
        prompt_content=args.prompt_content,
        id_prefix=args.id_prefix,
        metadata_keys=tuple(args.metadata_keys),
        include_label=args.include_label,
    )
    print(f"Generated {len(tasks)} task entries -> {args.output}")


if __name__ == "__main__":
    main()
