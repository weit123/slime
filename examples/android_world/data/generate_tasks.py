"""Generate a diverse task JSONL dataset for Android World RL training.

Reads task names from task_list.txt and produces 116 tasks x 20 params = 2320 entries.
The entries are shuffled so that consecutive samples in training come from different
tasks, improving gradient diversity.

Usage:
    python examples/android_world/data/generate_tasks.py
"""

import json
import random

TASK_LIST_PATH = "examples/android_world/data/task_list.txt"
OUTPUT_PATH = "examples/android_world/data/tasks_eval.jsonl"
PARAMS_PER_TASK = 1
SEED = 42


def load_task_names(path: str) -> list[str]:
    with open(path) as f:
        raw = f.read().strip()
    # Strip surrounding brackets and split by comma
    raw = raw.strip("[]")
    return [name.strip() for name in raw.split(",") if name.strip()]


def main():
    task_names = load_task_names(TASK_LIST_PATH)
    print(f"Loaded {len(task_names)} tasks from {TASK_LIST_PATH}")

    entries = []
    for task_name in task_names:
        for params_idx in range(PARAMS_PER_TASK):
            entries.append({
                "prompt": "Complete a task on Android",
                "label": "",
                "metadata": {
                    "task_name": task_name,
                    "params_idx": params_idx,
                },
            })

    print(f"Generated {len(entries)} entries ({len(task_names)} tasks x {PARAMS_PER_TASK} params)")

    # Shuffle for training diversity
    random.seed(SEED)
    random.shuffle(entries)

    with open(OUTPUT_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
