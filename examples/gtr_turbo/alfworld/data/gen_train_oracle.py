#!/usr/bin/env python3
"""Generate an oracle-validated ALFWorld train dataset."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.gtr_turbo.alfworld.data.gen_task_dataset import (
    CATEGORY_ORDER,
    discover_tasks,
    flatten_tasks,
    repo_root,
    write_prompt_jsonl,
)
from examples.gtr_turbo.alfworld.data.gen_eval_valid_seen import (
    _build_raw_alf_env,
    _is_oracle_solvable,
)


def _worker_loop(
    worker_id: int,
    config_path: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    display: str | None,
    max_oracle_steps: int | None,
    reset_retries: int,
) -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    env = None
    try:
        env, max_steps = _build_raw_alf_env(config_path, display, worker_id)
        if max_oracle_steps is not None:
            max_steps = max_oracle_steps
        while True:
            task = task_queue.get()
            if task is None:
                break
            for attempt in range(reset_retries + 1):
                try:
                    solved, oracle_steps, oracle_gc_sr = _is_oracle_solvable(env, task["task_file"], max_steps)
                    result = {
                        "task": task,
                        "solved": solved,
                        "oracle_steps": oracle_steps,
                        "oracle_goal_condition_success_rate": oracle_gc_sr,
                    }
                    result_queue.put(result)
                    break
                except TimeoutError:
                    if attempt >= reset_retries:
                        result_queue.put(
                            {
                                "task": task,
                                "solved": False,
                                "error": f"TimeoutError: Timed out resetting task: {task['task_file']}",
                            }
                        )
                        break
                    try:
                        env.close()
                    except Exception:
                        pass
                    env, max_steps = _build_raw_alf_env(config_path, display, worker_id)
                    if max_oracle_steps is not None:
                        max_steps = max_oracle_steps
                except Exception as exc:
                    result_queue.put(
                        {
                            "task": task,
                            "solved": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    break
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


def _filter_parallel(
    *,
    tasks: list[dict[str, str]],
    config_path: str,
    num_workers: int,
    display: str | None,
    progress_interval: int,
    max_oracle_steps: int | None,
    reset_retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not tasks:
        return [], []

    task_queue: mp.Queue = mp.Queue(maxsize=max(num_workers * 4, 1))
    result_queue: mp.Queue = mp.Queue()
    workers = [
        mp.Process(
            target=_worker_loop,
            args=(worker_id, config_path, task_queue, result_queue, display, max_oracle_steps, reset_retries),
            daemon=True,
        )
        for worker_id in range(num_workers)
    ]
    for proc in workers:
        proc.start()

    solved_tasks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    submitted = 0
    completed = 0
    last_log = time.monotonic()

    try:
        task_iter = iter(tasks)

        def submit_more() -> None:
            nonlocal submitted
            while submitted < len(tasks) and submitted - completed < num_workers * 4:
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                task_queue.put(task)
                submitted += 1

        submit_more()
        while completed < len(tasks):
            try:
                result = result_queue.get(timeout=600)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for oracle filter workers") from exc

            completed += 1
            task = dict(result["task"])
            if result.get("solved"):
                task["oracle_steps"] = result["oracle_steps"]
                task["oracle_goal_condition_success_rate"] = result["oracle_goal_condition_success_rate"]
                solved_tasks.append(task)
            else:
                failure = {"task": task, "error": result.get("error")}
                failures.append(failure)

            now = time.monotonic()
            if completed % progress_interval == 0 or now - last_log > 60 or completed == len(tasks):
                kept_by_category = Counter(task["category"] for task in solved_tasks)
                print(
                    f"[PROGRESS] {completed}/{len(tasks)} checked, "
                    f"{len(solved_tasks)} kept; kept_by_category={dict(kept_by_category)}",
                    flush=True,
                )
                last_log = now

            submit_more()

        return solved_tasks, failures
    finally:
        for _ in workers:
            task_queue.put(None)
        for proc in workers:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)


def _write_prompts(output: Path, tasks: list[dict[str, Any]]) -> None:
    write_prompt_jsonl(
        output,
        tasks,
        id_prefix="alfworld_oracle_train",
        metadata_keys=(
            "task_file",
            "task_type",
            "category",
            "oracle_steps",
            "oracle_goal_condition_success_rate",
        ),
    )


def _write_stats(
    stats_output: Path,
    *,
    total_by_category: Counter,
    kept_by_category: Counter,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total": sum(total_by_category.values()),
        "kept": sum(kept_by_category.values()),
        "ratio": sum(kept_by_category.values()) / sum(total_by_category.values()) if total_by_category else 0.0,
        "by_category": {},
        "errors": Counter(failure.get("error") for failure in failures if failure.get("error")),
    }
    for category in CATEGORY_ORDER:
        total = total_by_category.get(category, 0)
        kept = kept_by_category.get(category, 0)
        stats["by_category"][category] = {
            "total": total,
            "kept": kept,
            "ratio": kept / total if total else 0.0,
        }

    stats["errors"] = dict(stats["errors"])
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    with stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/root/.cache/alfworld/json_2.1.1/train")
    parser.add_argument(
        "--config",
        default=str(repo_root() / "examples/gtr_turbo/alfworld/config.yaml"),
        help="ALFWorld slime config.yaml.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root() / "examples/gtr_turbo/alfworld/data/alfworld_oracle_train_prompts.jsonl"),
    )
    parser.add_argument(
        "--stats-output",
        default=str(repo_root() / "examples/gtr_turbo/alfworld/data/alfworld_oracle_train_stats.json"),
    )
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument(
        "--max-oracle-steps",
        type=int,
        default=40,
        help="Verifier step budget. Keep this aligned with training rollout max_turns.",
    )
    parser.add_argument("--display", default=None)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    stats_output = Path(args.stats_output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to regenerate.")

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    tasks_by_category = discover_tasks(args.data_path)
    tasks = flatten_tasks(tasks_by_category)
    total_by_category = Counter(task["category"] for task in tasks)
    print(f"Discovered {len(tasks)} train tasks: {dict(total_by_category)}", flush=True)
    print(f"Starting {args.num_workers} CPU/Xvfb oracle filter workers...", flush=True)

    solved_tasks, failures = _filter_parallel(
        tasks=tasks,
        config_path=args.config,
        num_workers=args.num_workers,
        display=args.display,
        progress_interval=args.progress_interval,
        max_oracle_steps=args.max_oracle_steps,
        reset_retries=args.reset_retries,
    )
    kept_by_category = Counter(task["category"] for task in solved_tasks)
    _write_prompts(output, solved_tasks)
    stats = _write_stats(
        stats_output,
        total_by_category=total_by_category,
        kept_by_category=kept_by_category,
        failures=failures,
    )

    print(f"Generated {len(solved_tasks)} oracle-solvable train tasks -> {output}", flush=True)
    print(f"Wrote stats -> {stats_output}", flush=True)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
