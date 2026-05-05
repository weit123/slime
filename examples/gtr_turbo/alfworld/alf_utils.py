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

ALF_ACTION_LIST = [
    "pass", "goto", "pick", "put", "open", "close", "toggle",
    "heat", "clean", "cool", "slice", "inventory", "examine", "look",
]


def install_thor5_compat_patches(*, patch_put_object: bool = True) -> None:
    """Install process-local ALFWorld compatibility patches for ai2thor 5.x.

    ALFWorld 0.4.x still emits the pre-THOR5 PutObject call shape:
        PutObject(objectId=<held object>, receptacleObjectId=<target receptacle>)

    ai2thor 5.x removed receptacleObjectId from PutObject. When the agent is
    holding an object, the single objectId argument is the receptacle target.
    Without this remap, legal ALFWorld commands like "move apple 1 to table 1"
    become no-ops and every placement task gets zero success.
    """
    try:
        from alfworld.agents.controller.base import BaseAgent
        import alfworld.env.tasks as alf_tasks
        from alfworld.env.thor_env import ThorEnv
    except Exception as exc:
        logger.warning("Failed to import ALFWorld classes for THOR5 compatibility patch: %s", exc)
        return

    if not getattr(BaseAgent, "_gtr_slime_move_parse_patched", False):
        original_parse_command = BaseAgent.parse_command

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

        BaseAgent.parse_command = patched_parse_command
        BaseAgent._gtr_slime_move_parse_patched = True
        logger.info("Installed ALFWorld 0.4.x move-action parser compatibility patch")

    if patch_put_object and not getattr(ThorEnv, "_gtr_slime_thor5_compat_patched", False):
        original_step = ThorEnv.step

        def patched_step(self, action, smooth_nav=False, **kwargs):
            if isinstance(action, dict) and action.get("action") == "PutObject":
                action = dict(action)
                receptacle_object_id = action.pop("receptacleObjectId", None)
                if receptacle_object_id:
                    action["objectId"] = receptacle_object_id
                    action.setdefault("forceAction", True)
                    action.setdefault("placeStationary", True)
            return original_step(self, action, smooth_nav=smooth_nav, **kwargs)

        ThorEnv.step = patched_step
        ThorEnv._gtr_slime_thor5_compat_patched = True
        logger.info("Installed ALFWorld ai2thor 5.x compatibility patch for PutObject")

    if not getattr(alf_tasks.PickTwoObjAndPlaceTask, "_gtr_slime_empty_recep_patched", False):
        original_pick_two_goal_conditions_met = alf_tasks.PickTwoObjAndPlaceTask.goal_conditions_met

        def patched_pick_two_goal_conditions_met(self, state):
            targets = self.get_targets()
            receptacles = alf_tasks.get_objects_with_name_and_prop(targets["parent"], "receptacle", state.metadata)
            if not receptacles:
                ts = 2 + (2 if "Sliced" in targets["object"] else 0)
                return 0, ts
            return original_pick_two_goal_conditions_met(self, state)

        alf_tasks.PickTwoObjAndPlaceTask.goal_conditions_met = patched_pick_two_goal_conditions_met
        alf_tasks.PickTwoObjAndPlaceTask._gtr_slime_empty_recep_patched = True
        logger.info("Installed ALFWorld PickTwo empty-receptacle goal compatibility patch")

    if not getattr(BaseAgent, "_gtr_slime_empty_recep_fallback_patched", False):
        import alfworld.gen.constants as constants

        original_init_scene = BaseAgent.init_scene

        def patched_init_scene(self, load_receps):
            original_init_scene(self, load_receps)
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


def process_action(response_text: str, admissible_commands: list[str]) -> tuple[str, bool]:
    """Parse the model response to extract an action and match against admissible commands.

    Returns:
        (action_string, is_legal)
    """
    try:
        match = re.search(r'"action"\s*:\s*"([^"]*)"', response_text)
        if match:
            action_str = match.group(1).strip().lower()
            for cmd in admissible_commands:
                if cmd.lower() == action_str:
                    return cmd, True
            for cmd in admissible_commands:
                if action_str in cmd.lower() or cmd.lower() in action_str:
                    return cmd, True
    except Exception:
        pass

    if admissible_commands:
        return random.choice(admissible_commands), False
    return "look", False


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
