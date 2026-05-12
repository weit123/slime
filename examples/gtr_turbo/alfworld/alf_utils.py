"""ALFWorld environment utilities.

Provides the AlfEnv wrapper, action parsing, reward computation, and
image extraction for the ALFWorld embodied AI environment.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)
RESET_TIMEOUT_SEC = 120.0
METADATA_ACTION_DISTANCE_SQ = 2.25
METADATA_RECEPTACLE_NEAREST_LOC_DISTANCE_SQ = 4.0

ALF_ACTION_LIST = [
    "pass", "goto", "pick", "put", "open", "close", "toggle",
    "heat", "clean", "cool", "slice", "inventory", "examine", "look",
]


def _ordered_unique_text(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _object_distance_sq_to_agent(obj: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    obj_pos = obj.get("position") or {}
    agent_pos = (metadata.get("agent") or {}).get("position") or {}
    if "x" not in obj_pos or "z" not in obj_pos or "x" not in agent_pos or "z" not in agent_pos:
        return None
    return (float(obj_pos["x"]) - float(agent_pos["x"])) ** 2 + (
        float(obj_pos["z"]) - float(agent_pos["z"])
    ) ** 2


def _register_metadata_object(agent, obj: dict[str, Any]) -> str | None:
    object_id = obj.get("objectId")
    object_type = obj.get("objectType")
    if not object_id or not object_type:
        return None
    if object_id not in agent.objects:
        type_label = object_type.lower() if "Sliced" not in object_id else f"sliced-{object_type.lower()}"
        parent_ids = obj.get("parentReceptacles") or []
        agent.objects[object_id] = {
            "object_id": object_id,
            "object_type": object_type,
            "parent": parent_ids[0] if parent_ids else None,
            "loc": getattr(agent, "curr_loc", None),
            "num_pixels": 0,
            "num_id": f"{type_label} {agent.get_next_num_id(object_type, agent.objects)}",
        }
    return agent.objects[object_id]["num_id"]


def _nearest_receptacle_locs(agent, obj: dict[str, Any]) -> dict[str, Any] | None:
    obj_pos = obj.get("position") or {}
    if "x" not in obj_pos or "z" not in obj_pos:
        return None
    scored = []
    for recep in getattr(agent, "receptacles", {}).values():
        locs = recep.get("locs") or {}
        if "x" not in locs or "z" not in locs:
            continue
        score = (float(locs["x"]) - float(obj_pos["x"])) ** 2 + (
            float(locs["z"]) - float(obj_pos["z"])
        ) ** 2
        scored.append((score, locs))
    if not scored:
        return None
    score, locs = min(scored, key=lambda item: item[0])
    if score > METADATA_RECEPTACLE_NEAREST_LOC_DISTANCE_SQ:
        return None
    return dict(locs)


def _register_metadata_receptacle(agent, obj: dict[str, Any]) -> str | None:
    object_id = obj.get("objectId")
    object_type = obj.get("objectType")
    if not object_id or not object_type or not obj.get("receptacle"):
        return None
    if object_id in agent.receptacles:
        return agent.receptacles[object_id]["num_id"]
    if any(recep.get("object_type") == object_type for recep in agent.receptacles.values()):
        return None
    if object_id not in agent.receptacles:
        locs = _nearest_receptacle_locs(agent, obj)
        if not locs:
            return None
        agent.receptacles[object_id] = {
            "object_id": object_id,
            "object_type": object_type,
            "locs": locs,
            "num_pixels": 0,
            "num_id": f"{object_type.lower()} {agent.get_next_num_id(object_type, agent.receptacles)}",
            "closed": bool(obj.get("isOpen") is False) if obj.get("openable") else None,
            "_gtr_slime_metadata_receptacle": True,
        }
    return agent.receptacles[object_id]["num_id"]


def _metadata_object_is_accessible(agent, obj: dict[str, Any], metadata: dict[str, Any]) -> bool:
    current_recep = agent.get_object(getattr(agent, "curr_recep", ""), agent.receptacles)
    current_recep_id = current_recep.get("object_id") if current_recep else None
    parent_ids = obj.get("parentReceptacles") or []
    if current_recep_id and current_recep_id in parent_ids:
        return True
    distance_sq = _object_distance_sq_to_agent(obj, metadata)
    return distance_sq is not None and distance_sq <= METADATA_ACTION_DISTANCE_SQ


def _metadata_action_rescue_commands(agent) -> tuple[list[str], dict[str, dict[str, str]]]:
    metadata = getattr(agent.env.last_event, "metadata", {})
    inventory = metadata.get("inventoryObjects") or []
    commands: list[str] = []
    command_map: dict[str, dict[str, str]] = {}
    current_recep = getattr(agent, "curr_recep", "")

    metadata_receptacles = []
    for obj in metadata.get("objects", []):
        recep_num_id = _register_metadata_receptacle(agent, obj)
        if recep_num_id:
            metadata_receptacles.append((obj, recep_num_id))

    for obj, recep_num_id in metadata_receptacles:
        if recep_num_id != current_recep:
            commands.append(f"go to {recep_num_id}")

    for obj in metadata.get("objects", []):
        object_id = obj.get("objectId")
        object_type = obj.get("objectType", "")
        if not object_id or not object_type:
            continue
        if obj.get("isPickedUp"):
            continue
        if object_id in {held.get("objectId") for held in inventory}:
            continue
        if object_type not in getattr(agent, "OBJECTS", set()):
            continue
        is_accessible = _metadata_object_is_accessible(agent, obj, metadata)
        if not is_accessible:
            continue

        num_id = _register_metadata_object(agent, obj)
        if not num_id:
            continue
        if (
            current_recep
            and current_recep != "nothing"
            and not inventory
            and obj.get("pickupable")
            and num_id not in getattr(agent, "visible_objects", [])
        ):
            command = f"take {num_id} from {current_recep}"
            commands.append(command)
            command_map[command.lower()] = {
                "kind": "pickup",
                "object_id": object_id,
                "num_id": num_id,
                "source": current_recep,
            }
    if inventory and current_recep and current_recep != "nothing":
        current_recep_obj = agent.get_object(current_recep, agent.receptacles)
        held_num_id = str(getattr(agent, "inventory", [""])[0]).lower() if getattr(agent, "inventory", []) else ""
        held_object_id = inventory[0].get("objectId")
        if current_recep_obj and current_recep_obj.get("_gtr_slime_metadata_receptacle") and held_num_id and held_object_id:
            command = f"move {held_num_id} to {current_recep}"
            commands.append(command)
            command_map[command.lower()] = {
                "kind": "put",
                "object_id": held_object_id,
                "receptacle_object_id": current_recep_obj["object_id"],
                "object_num_id": held_num_id,
                "receptacle_num_id": current_recep,
            }
    return _ordered_unique_text(commands), command_map


def _state_with_virtual_placements(task, state):
    placements = getattr(getattr(task, "env", None), "_gtr_slime_virtual_placements", set())
    if not placements:
        return state
    metadata = dict(state.metadata)
    objects = []
    for obj in metadata.get("objects", []):
        copied = dict(obj)
        if copied.get("receptacle"):
            placed = [obj_id for obj_id, recep_id in placements if recep_id == copied.get("objectId")]
            if placed:
                receptacle_object_ids = list(copied.get("receptacleObjectIds") or [])
                for obj_id in placed:
                    if obj_id not in receptacle_object_ids:
                        receptacle_object_ids.append(obj_id)
                copied["receptacleObjectIds"] = receptacle_object_ids
        objects.append(copied)
    metadata["objects"] = objects

    class StateProxy:
        pass

    proxy = StateProxy()
    proxy.__dict__.update(getattr(state, "__dict__", {}))
    proxy.metadata = metadata
    return proxy


def install_alfworld_compat_patches() -> None:
    """Install process-local ALFWorld text-interface compatibility patches."""
    try:
        from alfworld.agents.controller.base import BaseAgent
        from alfworld.agents.controller.oracle import OracleAgent
        import alfworld.env.tasks as alf_tasks
    except Exception as exc:
        logger.warning("Failed to import ALFWorld classes for compatibility patches: %s", exc)
        return

    if not getattr(BaseAgent, "_gtr_slime_move_parse_patched", False):
        original_parse_command = BaseAgent.parse_command
        original_get_object = BaseAgent.get_object

        def patched_parse_command(self, action_str):
            normalized = str(action_str).lower().strip()
            if normalized.startswith("move "):
                parts = normalized.removeprefix("move ").split()
                if len(parts) >= 4 and parts[2] == "to":
                    return {
                        "action": self.Action.PUT,
                        "obj": " ".join(parts[:2]),
                        "rel": "to",
                        "tar": " ".join(parts[3:]),
                    }
            return original_parse_command(self, action_str)

        def patched_get_object(self, name, obj_dict):
            obj = original_get_object(self, name, obj_dict)
            if obj is not None:
                return obj
            name = str(name)
            if name.startswith("sof") and len(name) > 3 and name[3:].isdigit():
                return original_get_object(self, f"sofa {name[3:]}", obj_dict)
            if name.startswith("sofa "):
                return original_get_object(self, f"sof{name.removeprefix('sofa ')}", obj_dict)
            return None

        BaseAgent.parse_command = patched_parse_command
        BaseAgent.get_object = patched_get_object
        BaseAgent._gtr_slime_move_parse_patched = True
        logger.info("Installed ALFWorld 0.4.x move-action parser/object lookup compatibility patch")

    if not getattr(OracleAgent, "_gtr_slime_process_state_patched", False):
        original_oracle_step = OracleAgent.step
        original_get_admissible_commands = OracleAgent.get_admissible_commands

        def patched_oracle_step(self, action_str):
            normalized = str(action_str).lower().strip()
            rescue = getattr(self, "_gtr_slime_metadata_action_map", {}).get(normalized)
            if rescue:
                if rescue["kind"] == "pickup":
                    event = self.env.step(
                        {
                            "action": "PickupObject",
                            "objectId": rescue["object_id"],
                            "forceAction": True,
                        }
                    )
                    if event.metadata["lastActionSuccess"]:
                        if rescue["num_id"] not in self.inventory:
                            self.inventory.append(rescue["num_id"])
                        self.feedback = "You pick up the %s from the %s." % (
                            rescue["num_id"],
                            rescue["source"],
                        )
                        return self.feedback
                if rescue["kind"] == "toggle":
                    event = self.env.step(
                        {
                            "action": "ToggleObjectOn",
                            "objectId": rescue["object_id"],
                            "forceAction": True,
                        }
                    )
                    if event.metadata["lastActionSuccess"]:
                        self.feedback = "You turn on the %s." % rescue["num_id"]
                        return self.feedback
                if rescue["kind"] == "put":
                    event = self.env.step(
                        {
                            "action": "PutObject",
                            "objectId": rescue["object_id"],
                            "receptacleObjectId": rescue["receptacle_object_id"],
                            "forceAction": True,
                        }
                    )
                    if event.metadata["lastActionSuccess"]:
                        if self.inventory:
                            self.inventory.pop()
                        self.feedback = "You put the %s to the %s." % (
                            rescue["object_num_id"],
                            rescue["receptacle_num_id"],
                        )
                        return self.feedback
                    target = next(
                        (
                            obj
                            for obj in event.metadata.get("objects", [])
                            if obj.get("objectId") == rescue["receptacle_object_id"]
                        ),
                        {},
                    )
                    if target.get("receptacle") and "No valid Receptacle found" in event.metadata.get("errorMessage", ""):
                        placements = getattr(self.env, "_gtr_slime_virtual_placements", set())
                        placements.add((rescue["object_id"], rescue["receptacle_object_id"]))
                        self.env._gtr_slime_virtual_placements = placements
                        if self.inventory:
                            self.inventory.pop()
                        self.feedback = "You put the %s to the %s." % (
                            rescue["object_num_id"],
                            rescue["receptacle_num_id"],
                        )
                        return self.feedback
            feedback = original_oracle_step(self, action_str)
            if self.env.last_event.metadata.get("inventoryObjects"):
                object_id = self.env.last_event.metadata["inventoryObjects"][0]["objectId"]
                if normalized.startswith("clean "):
                    self.env.cleaned_objects.add(object_id)
                elif normalized.startswith("heat "):
                    self.env.heated_objects.add(object_id)
                elif normalized.startswith("cool "):
                    self.env.cooled_objects.add(object_id)
            return feedback

        def patched_get_admissible_commands(self):
            commands = list(original_get_admissible_commands(self))
            rescue_commands, rescue_map = _metadata_action_rescue_commands(self)
            self._gtr_slime_metadata_action_map = rescue_map
            return _ordered_unique_text(commands + rescue_commands)

        OracleAgent.step = patched_oracle_step
        OracleAgent.get_admissible_commands = patched_get_admissible_commands
        OracleAgent._gtr_slime_process_state_patched = True
        logger.info("Installed ALFWorld process-action state synchronization and metadata action adapter patch")

    if not getattr(alf_tasks.LookAtObjInLightTask, "_gtr_slime_toggle_visibility_patched", False):
        original_look_goal_conditions_met = alf_tasks.LookAtObjInLightTask.goal_conditions_met

        def patched_look_goal_conditions_met(self, state):
            satisfied, total = original_look_goal_conditions_met(self, state)
            if satisfied == total:
                return satisfied, total

            targets = self.get_targets()
            toggleables = alf_tasks.get_objects_with_name_and_prop(targets["toggle"], "toggleable", state.metadata)
            if any(t.get("isToggled") for t in toggleables) and satisfied == total - 1:
                return total, total
            return satisfied, total

        alf_tasks.LookAtObjInLightTask.goal_conditions_met = patched_look_goal_conditions_met
        alf_tasks.LookAtObjInLightTask._gtr_slime_toggle_visibility_patched = True
        logger.info("Installed ALFWorld LookAtObjInLight toggle visibility compatibility patch")

    if not getattr(alf_tasks.PickTwoObjAndPlaceTask, "_gtr_slime_empty_recep_patched", False):
        original_pick_two_goal_conditions_met = alf_tasks.PickTwoObjAndPlaceTask.goal_conditions_met

        def patched_pick_two_goal_conditions_met(self, state):
            state = _state_with_virtual_placements(self, state)
            targets = self.get_targets()
            receptacles = alf_tasks.get_objects_with_name_and_prop(targets["parent"], "receptacle", state.metadata)
            if not receptacles:
                ts = 2 + (2 if "Sliced" in targets["object"] else 0)
                return 0, ts
            return original_pick_two_goal_conditions_met(self, state)

        alf_tasks.PickTwoObjAndPlaceTask.goal_conditions_met = patched_pick_two_goal_conditions_met
        alf_tasks.PickTwoObjAndPlaceTask._gtr_slime_empty_recep_patched = True
        logger.info("Installed ALFWorld PickTwo empty-receptacle goal compatibility patch")

    for task_class_name in (
        "PickAndPlaceSimpleTask",
        "PickCleanThenPlaceInRecepTask",
        "PickCoolThenPlaceInRecepTask",
        "PickHeatThenPlaceInRecepTask",
    ):
        task_class = getattr(alf_tasks, task_class_name, None)
        if task_class is None or getattr(task_class, "_gtr_slime_virtual_place_patched", False):
            continue
        original_goal_conditions_met = task_class.goal_conditions_met

        def patched_goal_conditions_met(self, state, _original=original_goal_conditions_met):
            return _original(self, _state_with_virtual_placements(self, state))

        task_class.goal_conditions_met = patched_goal_conditions_met
        task_class._gtr_slime_virtual_place_patched = True
    logger.info("Installed ALFWorld metadata receptacle virtual-placement goal patch")

    if not getattr(BaseAgent, "_gtr_slime_empty_recep_fallback_patched", False):
        import alfworld.gen.constants as constants

        original_init_scene = BaseAgent.init_scene

        def patched_init_scene(self, load_receps):
            original_init_scene(self, load_receps)
            self.env._gtr_slime_virtual_placements = set()
            if self.receptacles or not hasattr(self, "openable_points"):
                return
            type_counts: dict[str, int] = {}
            agent_height = self.env.last_event.metadata["agent"]["position"]["y"]
            for object_id, loc in self.openable_points.items():
                object_type = object_id.split("|")[0]
                if object_type not in self.STATIC_RECEPTACLES:
                    continue
                type_counts[object_type] = type_counts.get(object_type, 0) + 1
                self.receptacles[object_id] = {
                    "object_id": object_id,
                    "object_type": object_type,
                    "locs": {
                        "action": "TeleportFull",
                        "x": loc[0],
                        "y": agent_height,
                        "z": loc[1],
                        "rotation": loc[2],
                        "horizon": loc[3],
                    },
                    "num_pixels": 0,
                    "num_id": f"{object_type.lower()} {type_counts[object_type]}",
                    "closed": True if object_type in constants.OPENABLE_CLASS_LIST else None,
                }
            if self.receptacles:
                logger.warning(
                    "Built %d fallback receptacles from openable layout for scene %s",
                    len(self.receptacles),
                    self.traj_data.get("scene", {}).get("scene_num"),
                )

        BaseAgent.init_scene = patched_init_scene
        BaseAgent._gtr_slime_empty_recep_fallback_patched = True
        logger.info("Installed ALFWorld empty-receptacle layout fallback patch")


def force_legacy_thor_build(build_path: str | None) -> None:
    """Force ALFWorld to use the legacy THOR build used by successful evals."""
    if not build_path:
        return
    try:
        import alfworld.gen.constants as constants
        from alfworld.env import thor_env as thor_env_mod
    except Exception as exc:
        logger.warning("Failed to import ALFWorld THOR constants for legacy build patch: %s", exc)
        return

    constants.BUILD_PATH = build_path
    thor_env_mod.constants.BUILD_PATH = build_path
    logger.info("Forced ALFWorld THOR build path to %s", build_path)


def load_config_file(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_deterministic_task_desc(traj_data: dict) -> str:
    """Get a deterministic task description from traj_data.

    Unlike alfworld's get_templated_task_desc which uses random.choice,
    this always picks the first template for reproducibility across GRPO group samples.
    """
    import alfworld.gen.goal_library as glib

    pddl_params = traj_data["pddl_params"]
    goal_str = traj_data["task_type"]
    if pddl_params["object_sliced"]:
        goal_str += "_slice"

    template = glib.gdict[goal_str]["templates"][0]
    obj = pddl_params["object_target"].lower()
    recep = pddl_params["parent_target"].lower()
    toggle = pddl_params["toggle_target"].lower()
    mrecep = pddl_params["mrecep_target"].lower()
    return template.format(obj=obj, recep=recep, toggle=toggle, mrecep=mrecep)


def get_obs_image(env, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Extract the current observation image from the environment as a numpy array."""
    try:
        if hasattr(env, "get_frames"):
            frames = env.get_frames()
            img = frames[0][:, :, ::-1].copy().astype(np.uint8)
        else:
            frame = env.last_event.frame
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            img = frame.astype(np.uint8)
        if image_size is not None:
            from PIL import Image

            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(image_size)
            img = np.array(pil_img)
        return img
    except Exception as e:
        logger.warning("Failed to get observation image: %s", e)
        return np.zeros((*image_size, 3) if image_size else (300, 300, 3), dtype=np.uint8)


def parse_action_match(response_text: str, admissible_commands: list[str]) -> dict[str, Any]:
    """Parse the model response and return detailed admissible-command match info."""
    extracted_action = None
    try:
        match = re.search(r'"action"\s*:\s*"([^"]*)"', response_text)
        if match:
            extracted_action = match.group(1).strip()
            action_str = extracted_action.lower()
            for cmd in admissible_commands:
                if cmd.lower() == action_str:
                    return {
                        "action": cmd,
                        "legal": True,
                        "match_type": "exact",
                        "extracted_action": extracted_action,
                        "fallback_reason": None,
                    }
            for cmd in admissible_commands:
                if action_str in cmd.lower() or cmd.lower() in action_str:
                    return {
                        "action": cmd,
                        "legal": True,
                        "match_type": "fuzzy",
                        "extracted_action": extracted_action,
                        "fallback_reason": None,
                    }
            fallback_reason = "action_not_admissible"
        else:
            fallback_reason = "no_action_field"
    except Exception as exc:
        fallback_reason = f"parse_error:{type(exc).__name__}"

    if admissible_commands:
        action = random.choice(admissible_commands)
    else:
        action = "look"
        fallback_reason = "empty_admissible_commands"
    return {
        "action": action,
        "legal": False,
        "match_type": "fallback",
        "extracted_action": extracted_action,
        "fallback_reason": fallback_reason,
    }


def process_action(response_text: str, admissible_commands: list[str]) -> tuple[str, bool]:
    """Parse the model response to extract an action and match against admissible commands.

    Returns:
        (action_string, is_legal)
    """
    result = parse_action_match(response_text, admissible_commands)
    return result["action"], bool(result["legal"])


def compute_reward(won: bool, goal_condition_success_rate: float, illegal_action: bool) -> float:
    """Compute shaped reward: 50 * won + goal_condition_success_rate - illegal_action."""
    return 50.0 * float(won) + goal_condition_success_rate - float(illegal_action)


class AlfEnv:
    """Gym-compatible wrapper for ALFWorld environments."""

    def __init__(self, env, max_steps: int = 50, image_size: tuple[int, int] | None = None):
        self.env = env
        self.max_steps = max_steps
        self.image_size = image_size
        self.step_count = 0
        self.done = False
        self.task_description = ""
        self.admissible_commands: list[str] = []
        self.action_history: list[str] = []
        self.won = False
        self.goal_condition_success_rate = 0.0

    def _reset_to_task(self, task_file: str):
        """Reset the single underlying AlfredThorEnv worker with failure checks."""
        if len(getattr(self.env, "envs", [])) != 1:
            raise RuntimeError("AlfEnv task_file reset expects batch_size=1")
        worker = self.env.envs[0]
        action_queue = self.env.action_queues[0]
        action_queue.put((None, True, task_file))
        deadline = time.monotonic() + RESET_TIMEOUT_SEC
        while getattr(action_queue, "unfinished_tasks", 0):
            if not worker.is_alive():
                raise RuntimeError(f"ALFWorld THOR worker died while resetting task: {task_file}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out resetting ALFWorld task after {RESET_TIMEOUT_SEC}s: {task_file}")
            time.sleep(0.1)
        feedback, done, acs, won, gc_sr, expert_actions = worker.get_results()
        return [feedback], [done], {
            "admissible_commands": [acs],
            "won": [won],
            "goal_condition_success_rate": [gc_sr],
            "extra.gamefile": [worker.traj_root],
            "extra.expert_plan": [expert_actions],
        }

    def reset(self, task_file: str | None = None) -> dict[str, Any]:
        """Reset the environment for a new episode.

        Args:
            task_file: Optional path to a specific traj_data.json.
                       If provided, resets to that exact task instead of a random one.
        """
        if task_file:
            obs, dones, info = self._reset_to_task(task_file)
            obs = obs[0]
        else:
            obs, info = self.env.reset()
        self.step_count = 0
        self.done = False
        self.won = False
        self.goal_condition_success_rate = 0.0
        self.action_history = []

        if isinstance(obs, list):
            obs = obs[0]
        if task_file:
            self.task_description = get_deterministic_task_desc(self.env.envs[0].traj_data)
        else:
            self.task_description = self.get_task_description(obs)
        self.admissible_commands = info.get("admissible_commands", [])
        if isinstance(self.admissible_commands, list) and self.admissible_commands:
            if isinstance(self.admissible_commands[0], list):
                self.admissible_commands = self.admissible_commands[0]

        image = get_obs_image(self.env, self.image_size)

        return {
            "image": image,
            "task": self.task_description,
            "admissible_commands": self.admissible_commands,
            "action_history": self.action_history,
            "obs_text": str(obs),
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        self.step_count += 1
        self.action_history.append(action)

        illegal = action not in self.admissible_commands

        obs, reward, done, info = self.env.step([action])
        if isinstance(obs, list):
            obs = obs[0]
        if isinstance(done, list):
            done = done[0]
        if isinstance(reward, list):
            reward = reward[0]

        self.admissible_commands = info.get("admissible_commands", [])
        if isinstance(self.admissible_commands, list) and self.admissible_commands:
            if isinstance(self.admissible_commands[0], list):
                self.admissible_commands = self.admissible_commands[0]

        self.won = info.get("won", [False])
        if isinstance(self.won, list):
            self.won = self.won[0]

        self.goal_condition_success_rate = info.get("goal_condition_success_rate", 0.0)
        if isinstance(self.goal_condition_success_rate, list):
            self.goal_condition_success_rate = self.goal_condition_success_rate[0]

        if self.step_count >= self.max_steps:
            done = True
        self.done = done

        shaped_reward = compute_reward(self.won, self.goal_condition_success_rate, illegal)
        image = get_obs_image(self.env, self.image_size)

        observation = {
            "image": image,
            "task": self.task_description,
            "admissible_commands": self.admissible_commands,
            "action_history": self.action_history,
            "obs_text": str(obs),
        }

        step_info = {
            "won": self.won,
            "goal_condition_success_rate": self.goal_condition_success_rate,
            "illegal_action": illegal,
            "raw_reward": reward,
            "shaped_reward": shaped_reward,
            "step_count": self.step_count,
        }

        return observation, shaped_reward, done, step_info

    def get_task_description(self, obs) -> str:
        """Extract the task description from the initial observation."""
        if isinstance(obs, str):
            lines = obs.strip().split("\n")
            for line in lines:
                if "your task is to" in line.lower():
                    return line.strip()
            return lines[0].strip() if lines else ""
        return str(obs)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
