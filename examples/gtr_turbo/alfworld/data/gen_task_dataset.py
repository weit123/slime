#!/usr/bin/env python3
"""Generate ALFWorld dataset with task_file paths for GRPO task-specific resets.

Usage:
    python examples/gtr_turbo/alfworld/data/gen_task_dataset.py \
        --data-path $ALFWORLD_DATA/json_2.1.1/train \
        --output examples/gtr_turbo/alfworld/data/alfworld_prompts.jsonl
"""

import argparse
import json
import os

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to ALFWorld task directory (e.g. json_2.1.1/train)")
    parser.add_argument("--output", default="examples/gtr_turbo/alfworld/data/alfworld_prompts.jsonl")
    parser.add_argument("--task-types", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    allowed_types = [TASK_TYPES[t] for t in args.task_types if t in TASK_TYPES]
    tasks = []

    for root, dirs, files in os.walk(args.data_path, topdown=False):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            continue

        json_path = os.path.join(root, "traj_data.json")
        with open(json_path) as f:
            traj_data = json.load(f)

        if traj_data["task_type"] not in allowed_types:
            continue

        tasks.append(json_path)

    tasks.sort()

    with open(args.output, "w") as f:
        for i, task_file in enumerate(tasks):
            record = {
                "prompt": [{"role": "user", "content": "Begin ALFWorld task."}],
                "metadata": {"task_file": task_file},
                "id": f"alfworld_{i}",
            }
            f.write(json.dumps(record) + "\n")

    print(f"Generated {len(tasks)} task entries -> {args.output}")


if __name__ == "__main__":
    main()
