#!/usr/bin/env python3
"""Generate a fixed oracle-validated ALFWorld valid_seen eval set."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import re
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
    discover_tasks as _discover_tasks,
    repo_root as _repo_root,
    task_sort_key,
    write_prompt_jsonl,
)


def _proportional_quotas(tasks_by_category: dict[str, list[dict[str, str]]], total: int) -> dict[str, int]:
    counts = {category: len(tasks_by_category.get(category, [])) for category in CATEGORY_ORDER}
    total_count = sum(counts.values())
    if total_count <= 0:
        raise RuntimeError("No valid_seen tasks found after filtering.")

    raw = {category: total * counts[category] / total_count for category in CATEGORY_ORDER}
    quotas = {category: int(raw[category]) for category in CATEGORY_ORDER}
    remaining = total - sum(quotas.values())
    order = sorted(CATEGORY_ORDER, key=lambda category: raw[category] - quotas[category], reverse=True)
    for category in order[:remaining]:
        quotas[category] += 1
    return quotas


def _normalize_list(value):
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value or []


def _first_value(value, default=None):
    value = _normalize_list(value)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _strip_num_ids(text: str) -> str:
    tokens = "".join(ch for ch in text.lower() if not ch.isdigit()).split()
    # Some ALFWorld admissible commands render "sofa 1" as "sof1", and
    # occasionally drop the final vowel before the numeric id, e.g. "spatul1".
    aliases = {
        "sof": "sofa",
        "spatul": "spatula",
    }
    tokens = [aliases.get(token, token) for token in tokens]
    return " ".join(tokens)


def _high_action(entry: dict[str, Any]) -> tuple[str, list[str]]:
    discrete = entry.get("discrete_action", {})
    return discrete.get("action", ""), [str(arg).lower() for arg in discrete.get("args", [])]


def _planner_object_id(entry: dict[str, Any]) -> str | None:
    planner_action = entry.get("planner_action", {})
    object_id = planner_action.get("cleanObjectId") or planner_action.get("objectId")
    return str(object_id) if object_id else None


def _object_type_from_id(object_id: str | None) -> str | None:
    if not object_id:
        return None
    return str(object_id).split("|", 1)[0].lower()


def _receptacle_type_from_id(object_id: str | None) -> str | None:
    if not object_id:
        return None
    parts = str(object_id).split("|")
    if len(parts) >= 5 and parts[-1]:
        return parts[-1].lower()
    return parts[0].lower()


def _action_object_arg(entry: dict[str, Any], args: list[str]) -> str | None:
    if args and args[0]:
        return args[0]
    return _object_type_from_id(_planner_object_id(entry))


def _planner_receptacle_object_id(entry: dict[str, Any]) -> str | None:
    planner_action = entry.get("planner_action", {})
    receptacle_id = planner_action.get("receptacleObjectId") or planner_action.get("objectId")
    return str(receptacle_id) if receptacle_id else None


def _worker_from_env(env):
    try:
        return env.envs[0]
    except Exception:
        return None


def _num_id_for_receptacle(env, object_id: str | None) -> str | None:
    if not object_id:
        return None
    worker = _worker_from_env(env)
    if worker is None:
        return None
    receptacles = getattr(getattr(worker, "controller", None), "receptacles", {})
    recep = receptacles.get(object_id)
    if recep:
        return recep.get("num_id")
    for candidate in receptacles.values():
        if candidate.get("object_id") == object_id:
            return candidate.get("num_id")
    return None


def _num_id_for_visible_object(env, object_id: str | None) -> str | None:
    if not object_id:
        return None
    worker = _worker_from_env(env)
    if worker is None:
        return None
    objects = getattr(getattr(worker, "controller", None), "objects", {})
    obj = objects.get(object_id)
    if obj:
        return obj.get("num_id")
    for candidate in objects.values():
        if candidate.get("object_id") == object_id:
            return candidate.get("num_id")
    return None


def _held_object_id(env) -> str | None:
    worker = _worker_from_env(env)
    if worker is None:
        return None
    try:
        inventory = worker.env.last_event.metadata.get("inventoryObjects") or []
    except Exception:
        return None
    if not inventory:
        return None
    object_id = inventory[0].get("objectId")
    return str(object_id) if object_id else None


def _pickup_matches_planner_object(env, entry: dict[str, Any]) -> bool:
    target_object_id = _planner_object_id(entry)
    return not target_object_id or _held_object_id(env) == target_object_id


def _parent_receptacle_num_id(env, object_id: str | None) -> str | None:
    worker = _worker_from_env(env)
    if worker is None or not object_id:
        return None
    try:
        objects = worker.env.last_event.metadata.get("objects", [])
    except Exception:
        return None
    for obj in objects:
        if obj.get("objectId") != object_id:
            continue
        for parent_id in obj.get("parentReceptacles") or []:
            num_id = _num_id_for_receptacle(env, parent_id)
            if num_id:
                return num_id
    return None


def _object_location_goto_candidates(object_id: str | None, admissible: list[str], env) -> list[str]:
    worker = _worker_from_env(env)
    if worker is None or not object_id:
        return []
    try:
        objects = worker.env.last_event.metadata.get("objects", [])
    except Exception:
        return []
    position = None
    for obj in objects:
        if obj.get("objectId") == object_id:
            position = obj.get("position") or {}
            break
    if position is None or "x" not in position or "z" not in position:
        return []

    receptacles = getattr(getattr(worker, "controller", None), "receptacles", {})
    by_num_id = {recep.get("num_id"): recep for recep in receptacles.values()}
    scored = []
    for command in admissible:
        if not command.lower().startswith("go to "):
            continue
        num_id = command.lower().removeprefix("go to ")
        recep = by_num_id.get(num_id)
        locs = recep.get("locs", {}) if recep else {}
        if "x" not in locs or "z" not in locs:
            continue
        score = (float(locs["x"]) - float(position["x"])) ** 2 + (float(locs["z"]) - float(position["z"])) ** 2
        scored.append((score, command))
    return [command for _score, command in sorted(scored, key=lambda item: item[0])]


def _commands_for_num_id(admissible: list[str], prefix: str, num_id: str | None) -> list[str]:
    if not num_id:
        return []
    expected = f"{prefix}{num_id}".lower()
    return [cmd for cmd in admissible if cmd.lower() == expected]


def _ordered_unique(commands: list[str]) -> list[str]:
    seen = set()
    unique = []
    for command in commands:
        key = command.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(command)
    return unique


def _repair_command_aliases(command: str) -> str:
    # "take spatul1 ..." can appear in admissible_commands but is rejected by
    # the ALFWorld parser; the parser accepts the explicit "spatula 1" form.
    return re.sub(r"\bspatul(\d+)\b", r"spatula \1", command, flags=re.IGNORECASE)


def _candidate_result(commands: list[str]) -> list[str]:
    return _ordered_unique([_repair_command_aliases(command) for command in commands])


def _preferred_goto_candidates(next_entry: dict[str, Any] | None, admissible: list[str], env) -> list[str]:
    if next_entry is None:
        return []
    action, _args = _high_action(next_entry)
    target_num_id = None
    if action == "PickupObject":
        target_num_id = _parent_receptacle_num_id(env, _planner_object_id(next_entry))
    elif action in {"PutObject", "OpenObject", "CloseObject", "CleanObject", "HeatObject", "CoolObject"}:
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(next_entry))
    elif action == "ToggleObject":
        target_num_id = _num_id_for_visible_object(env, _planner_object_id(next_entry)) or _parent_receptacle_num_id(
            env, _planner_object_id(next_entry)
        )
    preferred = _commands_for_num_id(admissible, "go to ", target_num_id)
    if preferred:
        return preferred
    if action in {"PickupObject", "ToggleObject"}:
        return _object_location_goto_candidates(_planner_object_id(next_entry), admissible, env)
    return []


def _planner_location_goto_candidates(entry: dict[str, Any], admissible: list[str], env) -> list[str]:
    location = entry.get("planner_action", {}).get("location")
    if not location:
        return []
    parts = str(location).split("|")
    if len(parts) < 3 or parts[0] != "loc":
        return []
    try:
        target_x = float(parts[1]) / 4.0
        target_z = float(parts[2]) / 4.0
    except ValueError:
        return []

    worker = _worker_from_env(env)
    if worker is None:
        return []
    receptacles = getattr(getattr(worker, "controller", None), "receptacles", {})
    by_num_id = {recep.get("num_id"): recep for recep in receptacles.values()}
    scored = []
    for command in admissible:
        if not command.lower().startswith("go to "):
            continue
        num_id = command.lower().removeprefix("go to ")
        recep = by_num_id.get(num_id)
        locs = recep.get("locs", {}) if recep else {}
        if "x" not in locs or "z" not in locs:
            continue
        score = (float(locs["x"]) - target_x) ** 2 + (float(locs["z"]) - target_z) ** 2
        scored.append((score, command))
    return [command for _score, command in sorted(scored, key=lambda item: item[0])]


def _high_confidence_goto_candidates(
    entry: dict[str, Any],
    next_entry: dict[str, Any] | None,
    admissible: list[str],
    env,
) -> list[str]:
    _goto_action, goto_args = _high_action(entry)
    return _ordered_unique(
        _preferred_goto_candidates(next_entry, admissible, env)
        + _goto_target_type_candidates(goto_args[0] if goto_args else None, admissible)
        + _planner_location_goto_candidates(entry, admissible, env)[:8]
    )


def _candidate_actions(entry: dict[str, Any], admissible: list[str], env=None) -> list[str]:
    action, args = _high_action(entry)
    commands = list(admissible)

    def starts_with(prefix: str) -> list[str]:
        return [cmd for cmd in commands if cmd.lower().startswith(prefix)]

    if action in {"NoOp", "End"}:
        return []
    if action == "GotoLocation" and args:
        target = args[0]
        return _candidate_result(
            starts_with(f"go to {target} ")
            + [cmd for cmd in commands if _strip_num_ids(cmd) == f"go to {target}"]
            + [cmd for cmd in commands if cmd.lower().startswith("go to ")]
        )
    if action == "PickupObject" and args:
        obj = args[0]
        parent_num_id = _parent_receptacle_num_id(env, _planner_object_id(entry))
        object_num_id = _num_id_for_visible_object(env, _planner_object_id(entry))
        return _candidate_result(
            _metadata_pickup_candidates_for_object(entry, commands, env)
            + [
                cmd
                for cmd in commands
                if parent_num_id and cmd.lower().startswith(f"take {obj} ") and cmd.lower().endswith(f" from {parent_num_id}")
            ]
            + [
                cmd
                for cmd in commands
                if object_num_id and cmd.lower().startswith(f"take {object_num_id} ")
            ]
            + starts_with(f"take {obj} ")
            + [
                cmd for cmd in commands if _strip_num_ids(cmd).startswith(f"take {obj} ")
            ]
        )
    if action == "PutObject" and len(args) >= 2:
        obj, recep = args[0], args[1]
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
        return _candidate_result(
            [
                cmd
                for cmd in commands
                if target_num_id
                and cmd.lower().startswith(f"move {obj} ")
                and cmd.lower().endswith(f" to {target_num_id}")
            ]
            + [cmd for cmd in commands if cmd.lower().startswith(f"move {obj} ") and f" to {recep} " in f"{cmd.lower()} "]
            + [cmd for cmd in commands if cmd.lower().startswith(f"put {obj} ") and recep in cmd.lower()]
            + [
                cmd
                for cmd in commands
                if _strip_num_ids(cmd).startswith(f"move {obj} ") and _strip_num_ids(cmd).endswith(f" to {recep}")
            ]
            + [
                cmd
                for cmd in commands
                if _strip_num_ids(cmd).startswith(f"put {obj} ") and recep in _strip_num_ids(cmd)
            ]
        )
    if action == "CleanObject":
        obj = _action_object_arg(entry, args)
        if not obj:
            return []
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
        target_type = _receptacle_type_from_id(_planner_receptacle_object_id(entry))
        return _candidate_result(
            [
                cmd
                for cmd in commands
                if target_num_id and cmd.lower() == f"clean {obj} 1 with {target_num_id}"
            ]
            + [
                cmd
                for cmd in commands
                if target_num_id and cmd.lower().startswith(f"clean {obj} ") and cmd.lower().endswith(f" with {target_num_id}")
            ]
            + [
                cmd
                for cmd in commands
                if not target_num_id
                and target_type
                and cmd.lower().startswith(f"clean {obj} ")
                and f" with {target_type} " in f"{cmd.lower()} "
            ]
            + starts_with(f"clean {obj} ")
        )
    if action == "HeatObject":
        obj = _action_object_arg(entry, args)
        if not obj:
            return []
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
        target_type = _receptacle_type_from_id(_planner_receptacle_object_id(entry))
        return _candidate_result(
            [
                cmd
                for cmd in commands
                if target_num_id and cmd.lower().startswith(f"heat {obj} ") and cmd.lower().endswith(f" with {target_num_id}")
            ]
            + [
                cmd
                for cmd in commands
                if not target_num_id
                and target_type
                and cmd.lower().startswith(f"heat {obj} ")
                and f" with {target_type} " in f"{cmd.lower()} "
            ]
            + starts_with(f"heat {obj} ")
        )
    if action == "CoolObject":
        obj = _action_object_arg(entry, args)
        if not obj:
            return []
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
        target_type = _receptacle_type_from_id(_planner_receptacle_object_id(entry))
        return _candidate_result(
            [
                cmd
                for cmd in commands
                if target_num_id and cmd.lower().startswith(f"cool {obj} ") and cmd.lower().endswith(f" with {target_num_id}")
            ]
            + [
                cmd
                for cmd in commands
                if not target_num_id
                and target_type
                and cmd.lower().startswith(f"cool {obj} ")
                and f" with {target_type} " in f"{cmd.lower()} "
            ]
            + starts_with(f"cool {obj} ")
        )
    if action == "ToggleObject" and args:
        target_num_id = _num_id_for_visible_object(env, _planner_object_id(entry))
        return _commands_for_num_id(commands, "use ", target_num_id) or starts_with(f"use {args[0]} ") or starts_with(f"toggle {args[0]} ")
    if action == "OpenObject" and args:
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
        return _commands_for_num_id(commands, "open ", target_num_id) or starts_with(f"open {args[0]} ")
    if action == "CloseObject" and args:
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
        return _commands_for_num_id(commands, "close ", target_num_id) or starts_with(f"close {args[0]} ")
    if action == "SliceObject" and args:
        return starts_with(f"slice {args[0]} ")

    normalized = " ".join([action.lower(), *args]).strip()
    return _candidate_result([cmd for cmd in commands if _strip_num_ids(cmd) == _strip_num_ids(normalized)])


def _has_planner_process_candidate(entry: dict[str, Any], candidates: list[str], env=None) -> bool:
    action, args = _high_action(entry)
    if action == "PickupObject":
        target_object_id = _planner_object_id(entry)
        if not target_object_id:
            return bool(candidates)
        if _metadata_pickup_candidates_for_object(entry, candidates, env):
            return True
        object_num_id = _num_id_for_visible_object(env, target_object_id)
        return bool(object_num_id and any(cmd.lower().startswith(f"take {object_num_id} ") for cmd in candidates))
    if action not in {"CleanObject", "HeatObject", "CoolObject"}:
        return bool(candidates)
    obj = _action_object_arg(entry, args)
    target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
    target_type = _receptacle_type_from_id(_planner_receptacle_object_id(entry))
    if not obj or not target_num_id:
        if not obj or not target_type:
            return bool(candidates)
        prefix = {
            "CleanObject": "clean",
            "HeatObject": "heat",
            "CoolObject": "cool",
        }[action]
        return any(
            cmd.lower().startswith(f"{prefix} {obj} ") and f" with {target_type} " in f"{cmd.lower()} "
            for cmd in candidates
        )
    prefix = {
        "CleanObject": "clean",
        "HeatObject": "heat",
        "CoolObject": "cool",
    }[action]
    return any(
        cmd.lower().startswith(f"{prefix} {obj} ") and cmd.lower().endswith(f" with {target_num_id}")
        for cmd in candidates
    )


def _metadata_rescue_action_keys(env) -> set[str]:
    worker = _worker_from_env(env)
    if worker is None:
        return set()
    return set(getattr(getattr(worker, "controller", None), "_gtr_slime_metadata_action_map", {}).keys())


def _metadata_pickup_candidates_for_object(
    entry: dict[str, Any], commands: list[str], env=None
) -> list[str]:
    target_object_id = _planner_object_id(entry)
    if not target_object_id:
        return []
    worker = _worker_from_env(env)
    if worker is None:
        return []
    rescue_map = getattr(getattr(worker, "controller", None), "_gtr_slime_metadata_action_map", {})
    if not rescue_map:
        return []
    by_key = {command.lower(): command for command in commands}
    return [
        by_key[key]
        for key, rescue in rescue_map.items()
        if key in by_key and rescue.get("kind") == "pickup" and rescue.get("object_id") == target_object_id
    ]


def _non_metadata_rescue_candidates(candidates: list[str], env=None) -> list[str]:
    rescue_keys = _metadata_rescue_action_keys(env)
    if not rescue_keys:
        return candidates
    return [candidate for candidate in candidates if candidate.lower() not in rescue_keys]


def _open_candidates_for(entry: dict[str, Any], admissible: list[str], env=None) -> list[str]:
    action, args = _high_action(entry)
    if action == "PutObject" and len(args) >= 2:
        recep = args[1]
        target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
        return _commands_for_num_id(admissible, "open ", target_num_id) or [
            cmd
            for cmd in admissible
            if cmd.lower().startswith(f"open {recep} ") or _strip_num_ids(cmd) == f"open {recep}"
        ]
    if action == "PickupObject":
        parent_num_id = _parent_receptacle_num_id(env, _planner_object_id(entry))
        preferred = _commands_for_num_id(admissible, "open ", parent_num_id)
        if preferred:
            return preferred
        return [cmd for cmd in admissible if cmd.lower().startswith("open ")]
    return []


def _goto_receptacle_type_candidates(recep: str, admissible: list[str]) -> list[str]:
    return _ordered_unique(
        [
            cmd
            for cmd in admissible
            if cmd.lower().startswith(f"go to {recep} ") or _strip_num_ids(cmd) == f"go to {recep}"
        ]
    )


def _goto_target_type_candidates(target: str | None, admissible: list[str]) -> list[str]:
    if not target:
        return []
    return _ordered_unique(
        [
            cmd
            for cmd in admissible
            if cmd.lower().startswith(f"go to {target} ") or _strip_num_ids(cmd) == f"go to {target}"
        ]
    )


def _step_env(env, action: str):
    step_result = env.step([action])
    if len(step_result) == 4:
        obs, _scores, dones, infos = step_result
    else:
        obs, dones, infos = step_result
    feedback = obs[0] if isinstance(obs, list) else obs
    done = dones[0] if isinstance(dones, list) else dones
    won = _first_value(infos.get("won", [False]), False)
    gc_sr = _first_value(infos.get("goal_condition_success_rate", [0.0]), 0.0)
    admissible = _normalize_list(infos.get("admissible_commands", [[]]))
    return feedback, bool(done), admissible, bool(won), float(gc_sr)


def _next_executable_high_action(plan: list[dict[str, Any]], start_idx: int) -> dict[str, Any] | None:
    for entry in plan[start_idx + 1 :]:
        action, _args = _high_action(entry)
        if action not in {"NoOp", "End"}:
            return entry
    return None


def _build_raw_alf_env(config_path: str, display: str | None, worker_id: int = 0):
    from examples.gtr_turbo.alfworld.alf_utils import (
        force_legacy_thor_build,
        install_alfworld_compat_patches,
        load_config_file,
    )
    from examples.gtr_turbo.alfworld.env_worker import (
        _configure_render_flags,
        _ensure_xvfb,
        _patch_flask_jinja2_compat,
    )

    _patch_flask_jinja2_compat()
    config = load_config_file(config_path)
    legacy_build_path = config.get("legacy_build_path")
    force_legacy_thor_build(legacy_build_path)
    install_alfworld_compat_patches()
    _configure_render_flags(
        render_image=config.get("render_image"),
        render_depth_image=config.get("render_depth_image"),
        render_class_image=config.get("render_class_image"),
        render_object_image=config.get("render_object_image"),
    )
    os.environ.pop("DISPLAY", None)
    base_display = int(config.get("xvfb_display_base", 190))
    _ensure_xvfb(display or f":{base_display + worker_id}")

    from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv

    thor_config = load_config_file(config["alfworld_config_file"])
    env = AlfredThorEnv(thor_config, train_eval="eval_in_distribution")
    env.init_env(batch_size=1)
    return env, int(config.get("max_turns", 40))


def _reset_to_task(env, task_file: str, timeout_sec: float = 120.0):
    worker = env.envs[0]
    action_queue = env.action_queues[0]
    action_queue.put((None, True, task_file))
    deadline = time.monotonic() + timeout_sec
    while getattr(action_queue, "unfinished_tasks", 0):
        if not worker.is_alive():
            raise RuntimeError(f"ALFWorld worker died during reset: {task_file}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out resetting task: {task_file}")
        time.sleep(0.1)
    return worker.get_results()


def _is_oracle_solvable(env, task_file: str, max_steps: int) -> tuple[bool, int, float]:
    feedback, done, admissible, won, gc_sr, _expert = _reset_to_task(env, task_file)
    admissible = _normalize_list(admissible)
    steps = 0
    committed_actions: list[str] = []

    def replay_committed():
        replay_feedback, replay_done, replay_admissible, replay_won, replay_gc_sr, _ = _reset_to_task(env, task_file)
        replay_admissible = _normalize_list(replay_admissible)
        for replay_action in committed_actions:
            replay_feedback, replay_done, replay_admissible, replay_won, replay_gc_sr = _step_env(env, replay_action)
            if replay_done or replay_won:
                break
        return replay_feedback, replay_done, replay_admissible, replay_won, replay_gc_sr

    with open(task_file, encoding="utf-8") as f:
        traj = json.load(f)
    plan = traj.get("plan", {}).get("high_pddl", [])

    for idx, entry in enumerate(plan):
        action_name, _args = _high_action(entry)
        if done or won:
            break
        if action_name in {"NoOp", "End"}:
            continue

        next_entry = _next_executable_high_action(plan, idx)
        candidates = _candidate_actions(entry, admissible, env)
        high_confidence_goto_candidates: list[str] = []
        if action_name == "GotoLocation":
            _goto_action, goto_args = _high_action(entry)
            high_confidence_goto_candidates = _high_confidence_goto_candidates(entry, next_entry, admissible, env)
            candidates = _ordered_unique(
                _preferred_goto_candidates(next_entry, admissible, env)
                + _goto_target_type_candidates(goto_args[0] if goto_args else None, admissible)
                + candidates
                + _planner_location_goto_candidates(entry, admissible, env)
            )
        if action_name == "PutObject":
            open_candidates = _open_candidates_for(entry, admissible, env)
            if open_candidates:
                feedback, done, admissible, won, gc_sr = _step_env(env, open_candidates[0])
                committed_actions.append(open_candidates[0])
                steps += 1
                if done or won or steps >= max_steps:
                    break
                candidates = _candidate_actions(entry, admissible, env)
        if not candidates and action_name in {"CleanObject", "HeatObject", "CoolObject"} and steps < max_steps:
            target_num_id = _num_id_for_receptacle(env, _planner_receptacle_object_id(entry))
            for goto_candidate in _commands_for_num_id(admissible, "go to ", target_num_id):
                feedback, done, admissible, won, gc_sr = _step_env(env, goto_candidate)
                committed_actions.append(goto_candidate)
                steps += 1
                if done or won or steps >= max_steps:
                    break
                candidates = _candidate_actions(entry, admissible, env)
                if candidates:
                    break
        if not candidates:
            open_candidates = _open_candidates_for(entry, admissible, env)
            if open_candidates:
                feedback, done, admissible, won, gc_sr = _step_env(env, open_candidates[0])
                committed_actions.append(open_candidates[0])
                steps += 1
                if done or won or steps >= max_steps:
                    break
                candidates = _candidate_actions(entry, admissible, env)
        if not candidates:
            return False, steps, float(gc_sr)

        chosen = candidates[0]
        if action_name == "GotoLocation" and next_entry is not None and len(candidates) > 1:
            next_action_name, _next_args = _high_action(next_entry)
            next_candidates = _candidate_actions(next_entry, admissible, env)
            visible_next_candidates = _non_metadata_rescue_candidates(next_candidates, env)
            if visible_next_candidates and _has_planner_process_candidate(next_entry, visible_next_candidates, env):
                continue

            chosen = ""
            # Branching by reset/replay avoids poisoning the verifier state with
            # a bad navigation guess. Keep the search bounded because full train
            # filtering runs this over thousands of tasks.
            search_passes: list[tuple[bool, list[str]]] = []
            candidate_keys = {cmd.lower() for cmd in candidates}
            high_confidence_goto_candidates = [
                candidate for candidate in high_confidence_goto_candidates if candidate.lower() in candidate_keys
            ]
            if high_confidence_goto_candidates:
                search_passes.append((False, high_confidence_goto_candidates))
                search_passes.append((True, high_confidence_goto_candidates))
            search_passes.append((False, candidates[:24]))
            search_passes.append((True, candidates[:24]))

            tried_by_rescue_mode: dict[bool, set[str]] = {False: set(), True: set()}
            for allow_rescue, goto_candidates in search_passes:
                for candidate in goto_candidates:
                    candidate_key = candidate.lower()
                    if candidate_key in tried_by_rescue_mode[allow_rescue]:
                        continue
                    tried_by_rescue_mode[allow_rescue].add(candidate_key)
                    trial_actions: list[str] = []
                    trial_steps = 0
                    feedback, done, admissible, won, gc_sr = _step_env(env, candidate)
                    trial_actions.append(candidate)
                    trial_steps += 1
                    next_candidates = _candidate_actions(next_entry, admissible, env)
                    acceptable_next_candidates = (
                        next_candidates if allow_rescue else _non_metadata_rescue_candidates(next_candidates, env)
                    )
                    if not done and not won and not acceptable_next_candidates:
                        for open_candidate in _open_candidates_for(next_entry, admissible, env):
                            feedback, done, admissible, won, gc_sr = _step_env(env, open_candidate)
                            trial_actions.append(open_candidate)
                            trial_steps += 1
                            next_candidates = _candidate_actions(next_entry, admissible, env)
                            acceptable_next_candidates = (
                                next_candidates
                                if allow_rescue
                                else _non_metadata_rescue_candidates(next_candidates, env)
                            )
                            if done or won or acceptable_next_candidates or steps + trial_steps >= max_steps:
                                break
                    if (
                        not done
                        and not won
                        and next_action_name == "PutObject"
                        and acceptable_next_candidates
                    ):
                        for open_candidate in _open_candidates_for(next_entry, admissible, env):
                            feedback, done, admissible, won, gc_sr = _step_env(env, open_candidate)
                            trial_actions.append(open_candidate)
                            trial_steps += 1
                            next_candidates = _candidate_actions(next_entry, admissible, env)
                            acceptable_next_candidates = (
                                next_candidates
                                if allow_rescue
                                else _non_metadata_rescue_candidates(next_candidates, env)
                            )
                            if done or won or acceptable_next_candidates or steps + trial_steps >= max_steps:
                                break
                        if done or won:
                            chosen = candidate
                            committed_actions.extend(trial_actions)
                            steps += trial_steps
                            break
                        if not acceptable_next_candidates:
                            feedback, done, admissible, won, gc_sr = replay_committed()
                            if steps + trial_steps >= max_steps:
                                return bool(won), steps, float(gc_sr)
                            continue
                        put_feedback, put_done, _put_admissible, put_won, put_gc_sr = _step_env(
                            env, acceptable_next_candidates[0]
                        )
                        put_worked = (
                            put_done
                            or put_won
                            or (
                                not str(put_feedback).startswith("Nothing happens")
                                and put_gc_sr >= gc_sr
                            )
                        )
                        if put_worked:
                            chosen = candidate
                            committed_actions.extend(trial_actions)
                            steps += trial_steps
                            feedback, done, admissible, won, gc_sr = replay_committed()
                            break
                        feedback, done, admissible, won, gc_sr = replay_committed()
                        if steps + trial_steps >= max_steps:
                            return bool(won), steps, float(gc_sr)
                        continue
                    if done or won or (
                        acceptable_next_candidates
                        and _has_planner_process_candidate(next_entry, acceptable_next_candidates, env)
                    ):
                        chosen = candidate
                        committed_actions.extend(trial_actions)
                        steps += trial_steps
                        break
                    feedback, done, admissible, won, gc_sr = replay_committed()
                    if steps + trial_steps >= max_steps:
                        return bool(won), steps, float(gc_sr)
                if chosen:
                    break
            if not chosen:
                return False, steps, float(gc_sr)
            continue

        before_gc_sr = gc_sr
        for attempt_idx, candidate in enumerate(candidates):
            feedback, done, admissible, won, gc_sr = _step_env(env, candidate)
            steps += 1
            wrong_pickup_target = (
                action_name == "PickupObject"
                and not done
                and not won
                and not str(feedback).startswith("Nothing happens")
                and not _pickup_matches_planner_object(env, entry)
            )
            reject_candidate = (
                not done
                and not won
                and attempt_idx < len(candidates) - 1
                and (wrong_pickup_target or str(feedback).startswith("Nothing happens") or gc_sr < before_gc_sr)
            )
            if reject_candidate:
                feedback, done, admissible, won, gc_sr = replay_committed()
                if steps >= max_steps:
                    break
                continue
            committed_actions.append(candidate)
            if (
                done
                or won
                or not str(feedback).startswith("Nothing happens")
                or gc_sr > before_gc_sr
                or attempt_idx == len(candidates) - 1
            ):
                break
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break

    return bool(won), steps, float(gc_sr)


def _worker_loop(
    worker_id: int,
    config_path: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    display: str | None,
) -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    env = None
    try:
        env, max_steps = _build_raw_alf_env(config_path, display, worker_id)
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                solved, oracle_steps, oracle_gc_sr = _is_oracle_solvable(env, task["task_file"], max_steps)
                if solved:
                    task = dict(task)
                    task["oracle_steps"] = oracle_steps
                    task["oracle_goal_condition_success_rate"] = oracle_gc_sr
                    result_queue.put({"ok": True, "task": task})
                else:
                    result_queue.put({"ok": False, "category": task["category"], "task_file": task["task_file"]})
            except Exception as exc:
                result_queue.put(
                    {
                        "ok": False,
                        "category": task["category"],
                        "task_file": task["task_file"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


def _select_oracle_solvable_parallel(
    *,
    tasks_by_category: dict[str, list[dict[str, str]]],
    quotas: dict[str, int] | None,
    config_path: str,
    num_workers: int,
    display: str | None,
) -> list[dict[str, Any]]:
    selected_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORY_ORDER}
    checked_by_category: Counter[str] = Counter()

    task_queue: mp.Queue = mp.Queue(maxsize=max(num_workers * 2, 1))
    result_queue: mp.Queue = mp.Queue()
    workers = [
        mp.Process(
            target=_worker_loop,
            args=(worker_id, config_path, task_queue, result_queue, display),
            daemon=True,
        )
        for worker_id in range(num_workers)
    ]
    for proc in workers:
        proc.start()

    try:
        task_iters = {category: iter(tasks_by_category.get(category, [])) for category in CATEGORY_ORDER}
        exhausted_categories: set[str] = set()
        pending = 0
        category_cursor = 0

        def category_target_reached(category: str) -> bool:
            return quotas is not None and len(selected_by_category[category]) >= quotas[category]

        def all_targets_reached() -> bool:
            if quotas is None:
                return len(exhausted_categories) == len(CATEGORY_ORDER)
            return all(category_target_reached(category) for category in CATEGORY_ORDER)

        def next_task() -> dict[str, str] | None:
            nonlocal category_cursor
            for _ in range(len(CATEGORY_ORDER)):
                category = CATEGORY_ORDER[category_cursor % len(CATEGORY_ORDER)]
                category_cursor += 1
                if category in exhausted_categories or category_target_reached(category):
                    continue
                try:
                    return next(task_iters[category])
                except StopIteration:
                    exhausted_categories.add(category)
            return None

        def submit_more() -> None:
            nonlocal pending
            while pending < num_workers * 2 and not all_targets_reached():
                task = next_task()
                if task is None:
                    break
                task_queue.put(task)
                pending += 1

        submit_more()
        while pending > 0:
            try:
                result = result_queue.get(timeout=300)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for oracle workers") from exc

            result_category = result.get("task", {}).get("category") or result.get("category")
            if result_category not in selected_by_category:
                raise RuntimeError(f"Unexpected result category {result_category}")
            pending -= 1
            checked_by_category[result_category] += 1
            if result.get("ok") and not category_target_reached(result_category):
                selected_by_category[result_category].append(result["task"])
                target = quotas[result_category] if quotas is not None else "all"
                print(
                    f"[OK] {result_category}: {len(selected_by_category[result_category])}/{target} "
                    f"{result['task']['task_file']}",
                    flush=True,
                )
            elif result.get("error"):
                print(f"[SKIP] {result['task_file']}: {result['error']}", flush=True)

            submit_more()

        if quotas is not None:
            for category in CATEGORY_ORDER:
                if len(selected_by_category[category]) < quotas[category]:
                    raise RuntimeError(
                        f"Only found {len(selected_by_category[category])}/{quotas[category]} "
                        f"oracle-solvable tasks for {category}."
                    )
        else:
            for category in CATEGORY_ORDER:
                print(
                    f"[DONE] {category}: checked={checked_by_category[category]}, "
                    f"kept={len(selected_by_category[category])}",
                    flush=True,
                )

        return [task for category in CATEGORY_ORDER for task in selected_by_category[category]]
    finally:
        for _ in workers:
            task_queue.put(None)
        for proc in workers:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)


def _write_stats(
    stats_output: str | Path,
    *,
    tasks_by_category: dict[str, list[dict[str, str]]],
    selected: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    total_by_category = Counter({category: len(tasks_by_category.get(category, [])) for category in CATEGORY_ORDER})
    kept_by_category = Counter(task["category"] for task in selected)
    total = sum(total_by_category.values())
    kept = len(selected)
    stats: dict[str, Any] = {
        "mode": mode,
        "total": total,
        "kept": kept,
        "success_rate": kept / total if total else 0.0,
        "by_category": {},
    }
    for category in CATEGORY_ORDER:
        category_total = total_by_category.get(category, 0)
        category_kept = kept_by_category.get(category, 0)
        stats["by_category"][category] = {
            "total": category_total,
            "kept": category_kept,
            "success_rate": category_kept / category_total if category_total else 0.0,
        }

    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/root/.cache/alfworld/json_2.1.1/valid_seen")
    parser.add_argument(
        "--config",
        default=str(_repo_root() / "examples/gtr_turbo/alfworld/config.yaml"),
        help="ALFWorld slime config.yaml.",
    )
    parser.add_argument(
        "--output",
        default=str(_repo_root() / "examples/gtr_turbo/alfworld/data/valid_seen_oracle_full_prompts.jsonl"),
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only evaluate oracle solvability and write stats; do not save the selected eval dataset.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=64,
        help="Quota-sampled eval size. Use 0 or --all-solvable to keep every oracle-solvable valid_seen task.",
    )
    parser.add_argument(
        "--all-solvable",
        action="store_true",
        help="Disable quota sampling and keep every oracle-solvable task from the split.",
    )
    parser.add_argument("--stats-output", default=None, help="Optional JSON path for oracle success-rate stats.")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.stats_only and not args.stats_output:
        raise ValueError("--stats-only requires --stats-output")
    if not args.stats_only and output.exists() and not args.overwrite:
        print(f"Eval set already exists: {output}")
        return

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tasks_by_category = _discover_tasks(args.data_path)
    use_all_solvable = args.all_solvable or args.num_tasks <= 0
    quotas = None if use_all_solvable else _proportional_quotas(tasks_by_category, args.num_tasks)
    rng = random.Random(args.seed)
    for tasks in tasks_by_category.values():
        rng.shuffle(tasks)

    if quotas is None:
        print("Using all oracle-solvable valid_seen tasks; quota sampling disabled.", flush=True)
    else:
        print(f"Using quotas: {quotas}", flush=True)
    print(f"Starting {args.num_workers} CPU/Xvfb oracle workers...", flush=True)
    selected = _select_oracle_solvable_parallel(
        tasks_by_category=tasks_by_category,
        quotas=quotas,
        config_path=args.config,
        num_workers=args.num_workers,
        display=args.display,
    )

    selected.sort(key=task_sort_key)
    eval_id_prefix = "valid_seen_oracle_full" if quotas is None else f"valid_seen_oracle{args.num_tasks}"
    if not args.stats_only:
        write_prompt_jsonl(
            output,
            selected,
            prompt_content="Begin ALFWorld eval task.",
            id_prefix="alfworld_valid_seen",
            metadata_keys=(
                "task_file",
                "task_type",
                "category",
                "oracle_steps",
                "oracle_goal_condition_success_rate",
            ),
            metadata_extra=lambda _task, idx: {"eval_task_id": f"{eval_id_prefix}_{idx}"},
            include_label=True,
            sort=False,
        )
        print(f"Generated {len(selected)} oracle-solvable eval tasks -> {output}")
    else:
        print(f"Stats-only mode: evaluated {len(selected)} oracle-solvable tasks; no dataset written.", flush=True)
    if args.stats_output:
        mode = "all_solvable" if quotas is None else "quota"
        stats = _write_stats(args.stats_output, tasks_by_category=tasks_by_category, selected=selected, mode=mode)
        print(f"Wrote stats -> {args.stats_output}", flush=True)
        print(json.dumps(stats, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
