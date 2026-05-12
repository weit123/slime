"""Android World environment worker for slime.

Contains the Ray remote AndroidWorldWorker that wraps a single Android emulator instance,
plus helper functions for AVD management, action parsing, and port allocation.

Ported from verl-agent (Apache-2.0 licensed) with adaptations for slime's architecture:
- Removed AndroidWorldEnvs batched wrapper (replaced by env_pool.py)
- Removed AndroidWorldEnvironmentManager (replaced by env_android_world.py)

"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from time import sleep
from typing import Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Android World SDK imports (expected to be on sys.path)
# ---------------------------------------------------------------------------
from android_world.env import env_launcher, json_action
from android_world import registry, suite_utils


# ---------------------------------------------------------------------------
# AVD management helpers
# ---------------------------------------------------------------------------


def clone_avd(src_avd_name: str, tar_avd_name: str, android_avd_home: str):
    """Clone the source AVD to the target AVD.

    Copies the AVD folder and .ini file and updates internal paths.
    """
    src_avd_dir = os.path.join(android_avd_home, src_avd_name + ".avd")
    tar_avd_dir = os.path.join(android_avd_home, tar_avd_name + ".avd")
    src_ini_file = os.path.join(android_avd_home, src_avd_name + ".ini")
    tar_ini_file = os.path.join(android_avd_home, tar_avd_name + ".ini")

    if not os.path.exists(tar_avd_dir):
        shutil.copytree(src_avd_dir, tar_avd_dir)

    with open(src_ini_file, "r") as src_ini, open(tar_ini_file, "w") as tar_ini:
        for line in src_ini:
            tar_ini.write(line.replace(src_avd_name, tar_avd_name))

    for ini_name in ["config.ini", "hardware-qemu.ini"]:
        ini_path = os.path.join(tar_avd_dir, ini_name)
        if os.path.exists(ini_path):
            with open(ini_path, "r") as file:
                lines = file.readlines()
            with open(ini_path, "w") as file:
                for line in lines:
                    file.write(line.replace(src_avd_name, tar_avd_name))

    snapshots_hw_ini = os.path.join(tar_avd_dir, "snapshots", "default_boot", "hardware.ini")
    if os.path.exists(snapshots_hw_ini):
        with open(snapshots_hw_ini, "r") as file:
            lines = file.readlines()
        with open(snapshots_hw_ini, "w") as file:
            for line in lines:
                file.write(line.replace(src_avd_name, tar_avd_name))


def remove_cache_avd(cache_avd_path: str, cache_avd_ini_path: str, interval: int = 1):
    """Remove cache AVD files with retry."""
    for _ in range(3):
        try:
            if os.path.exists(cache_avd_path):
                shutil.rmtree(cache_avd_path, ignore_errors=False)
            if os.path.exists(cache_avd_ini_path):
                try:
                    os.remove(cache_avd_ini_path)
                except Exception:
                    continue
            if not os.path.exists(cache_avd_path) and not os.path.exists(cache_avd_ini_path):
                break
        except Exception as e:
            logger.warning("remove_cache_avd error: %s", e)
            sleep(interval)


def kill_single_emulator(
    adb_path: str,
    emulator_name: str,
    adb_server_port: int = 5037,
    max_retries: int = 10,
    retry_interval: float = 0.5,
) -> bool:
    """Kill a single emulator instance via ADB."""
    logger.info("Attempting to kill %s via ADB server on port %d", emulator_name, adb_server_port)

    env = os.environ.copy()
    env["ANDROID_ADB_SERVER_PORT"] = str(adb_server_port)

    result = subprocess.run([adb_path, "devices"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        subprocess.run([adb_path, "kill-server"], env=env)
        sleep(1)
        subprocess.run([adb_path, "start-server"], env=env)
        sleep(2)
        result = subprocess.run([adb_path, "devices"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    devices_output = result.stdout.decode("utf-8")
    if emulator_name not in devices_output:
        return True

    subprocess.run(
        [adb_path, "-s", emulator_name, "emu", "kill"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for attempt in range(max_retries):
        sleep(retry_interval)
        result = subprocess.run([adb_path, "devices"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        devices_output = result.stdout.decode("utf-8")
        if emulator_name not in devices_output:
            logger.info("%s has been shut down successfully", emulator_name)
            return True

    # Force kill
    port_match = re.search(r"emulator-(\d+)", emulator_name)
    if port_match:
        port = port_match.group(1)
        subprocess.run(["pkill", "-9", "-f", f"emulator.*-port {port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sleep(1)

    result = subprocess.run([adb_path, "devices"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    devices_output = result.stdout.decode("utf-8")
    if emulator_name not in devices_output:
        return True
    logger.error("Failed to kill %s", emulator_name)
    return False


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def rescale_fn(
    x: int, y: int, width: int, height: int, ori_width: int = 1080, ori_height: int = 2400
) -> tuple[int, int]:
    """Rescale coordinates from [width x height] to actual screen size."""
    return round(x / width * ori_width), round(y / height * ori_height)


def clean_action_text(text: str) -> str:
    """Clean special characters from action text."""
    if not text:
        return text
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = text.replace("\\", "")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def is_terminal_action(action: Optional[json_action.JSONAction]) -> bool:
    """Check if an action is terminal (ends the episode)."""
    if action is None:
        return False
    return action.action_type == "status"


def parse_ui_action_from_response(
    response: str,
    norm_width: int = 999,
    norm_height: int = 999,
    screen_width: int = 1080,
    screen_height: int = 2400,
    whether_direction: bool = False,
    rescale_coords: bool = True,
) -> Optional[json_action.JSONAction]:
    """Parse action from model response (tool_call XML format)."""
    try:
        action_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if not action_match:
            logger.debug("No tool_call found in response")
            return json_action.JSONAction(action_type="wait")

        action_str = action_match.group(1).strip()
        try:
            action_dict = json.loads(action_str)
        except json.JSONDecodeError:
            try:
                fixed_str = re.sub(r"(\w+)=", r'"\1":', action_str)
                action_dict = json.loads(fixed_str)
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON from tool_call")
                return json_action.JSONAction(action_type="wait")

        action_type = action_dict.get("arguments", {}).get("action", "")
        if not action_type:
            return json_action.JSONAction(action_type="wait")

        x, y, x_, y_, text, direction, goal_status, app_name = (
            None, None, None, None, None, None, None, None,
        )

        if action_type in ["click", "long_press"]:
            coord = action_dict["arguments"].get("coordinate")
            if not coord or len(coord) != 2:
                return json_action.JSONAction(action_type="wait")
            x, y = coord
            if rescale_coords:
                x, y = rescale_fn(x, y, norm_width, norm_height, screen_width, screen_height)

        elif action_type == "swipe":
            coord1 = action_dict["arguments"].get("coordinate")
            coord2 = action_dict["arguments"].get("coordinate2")
            if not coord1 or len(coord1) != 2 or not coord2 or len(coord2) != 2:
                return json_action.JSONAction(action_type="wait")
            x1, y1 = coord1
            x2, y2 = coord2
            if whether_direction:
                x_, y_ = -1, -1
                dx, dy = x2 - x1, y2 - y1
                direction = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
            else:
                if rescale_coords:
                    x, y = rescale_fn(x1, y1, norm_width, norm_height, screen_width, screen_height)
                    x_, y_ = rescale_fn(x2, y2, norm_width, norm_height, screen_width, screen_height)
                else:
                    x, y, x_, y_ = x1, y1, x2, y2

        elif action_type in ["open", "open_app"]:
            action_type = "open_app"
            app_name = action_dict["arguments"].get("text", "")
            if not app_name:
                return json_action.JSONAction(action_type="wait")

        elif action_type in ["type", "input_text"]:
            action_type = "input_text"
            text = action_dict["arguments"].get("text", "")
            if not text:
                return json_action.JSONAction(action_type="wait")

        elif action_type == "wait":
            pass

        elif action_type == "system_button":
            button = action_dict["arguments"].get("button", "")
            button_map = {"Back": "navigate_back", "Home": "navigate_home", "Enter": "keyboard_enter"}
            action_type = button_map.get(button, "wait")

        elif action_type == "terminate":
            action_type = "status"
            goal_status = action_dict["arguments"].get("status", "success")

        elif action_type == "answer":
            text = action_dict["arguments"].get("text", "")

        else:
            logger.debug("Unknown action type: %s", action_type)
            return json_action.JSONAction(action_type="wait")

        return json_action.JSONAction(
            action_type=action_type,
            direction=direction,
            x=x,
            y=y,
            x_=x_,
            y_=y_,
            text=text,
            goal_status=goal_status,
            app_name=app_name,
        )
    except Exception as e:
        logger.debug("Error parsing action: %s", e)
        return json_action.JSONAction(action_type="wait")


# ---------------------------------------------------------------------------
# Port allocation helpers
# ---------------------------------------------------------------------------


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False
        except OSError:
            return True


def find_available_ports(base_port: int, count: int, port_pairs: bool = True) -> list[int]:
    """Find available ports starting from *base_port* (must be even for emulator console)."""
    available: list[int] = []
    current = base_port if base_port % 2 == 0 else base_port + 1
    attempts = 0
    while len(available) < count and attempts < 1000:
        if port_pairs:
            grpc = current + 8554 - 5556
            if not is_port_in_use(current) and not is_port_in_use(grpc):
                available.append(current)
        else:
            if not is_port_in_use(current):
                available.append(current)
        current += 2
        attempts += 1
    if len(available) < count:
        raise RuntimeError(f"Could not find {count} available ports")
    return available


def find_available_adb_ports(base_port: int, count: int) -> list[int]:
    available: list[int] = []
    current = base_port
    attempts = 0
    while len(available) < count and attempts < 1000:
        if not is_port_in_use(current):
            available.append(current)
        current += 1
        attempts += 1
    if len(available) < count:
        raise RuntimeError(f"Could not find {count} available ADB server ports")
    return available


def generate_env_configs(
    base_avd_name_pattern: str,
    base_console_port: int,
    base_grpc_port: int,
    num_envs: int,
    base_adb_server_port: int = 5037,
) -> tuple[list[str], list[int], list[int], list[int]]:
    """Generate AVD names, console ports, gRPC ports, and ADB server ports."""
    cache_avd_names = [base_avd_name_pattern.format(i) for i in range(1, num_envs + 1)]
    console_ports = find_available_ports(base_console_port, num_envs, port_pairs=True)
    grpc_ports = [p + (base_grpc_port - base_console_port) for p in console_ports]
    adb_server_ports = find_available_adb_ports(base_adb_server_port, num_envs)
    return cache_avd_names, console_ports, grpc_ports, adb_server_ports


# ---------------------------------------------------------------------------
# AndroidWorldWorker
# ---------------------------------------------------------------------------


class AndroidWorldWorker:
    """Wraps a single AndroidWorld emulator instance.

    This class is intended to be used as a Ray remote actor (decorated in env_pool.py).
    Each worker hosts one emulator and operates it via method calls.
    """

    def __init__(
        self,
        worker_id: int,
        console_port: int,
        grpc_port: int,
        adb_path: str,
        adb_server_port: int,
        emulator_path: str,
        android_sdk_root: str,
        android_avd_home: str,
        avd_name: str,
        max_steps: int,
        temp_path: str,
        task_family: str,
        save_images: bool = False,
        image_size: Optional[tuple[int, int]] = None,
        base_avd_name: Optional[str] = None,
    ):
        # Suppress noisy logs
        import warnings

        logging.getLogger("absl").setLevel(logging.CRITICAL)
        logging.getLogger("grpc").setLevel(logging.CRITICAL)
        logging.getLogger("android_env").setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        self.worker_id = worker_id
        self.console_port = console_port
        self.grpc_port = grpc_port
        self.adb_path = os.path.expanduser(adb_path)
        self.adb_server_port = adb_server_port
        self.emulator_path = os.path.expanduser(emulator_path)
        self.android_sdk_root = android_sdk_root
        self.android_avd_home = android_avd_home
        self.avd_name = avd_name
        self.max_steps = max_steps
        self.temp_path = temp_path
        self.task_family = task_family
        self.save_images = save_images
        self.image_size = image_size
        self.base_avd_name = base_avd_name

        os.makedirs(temp_path, exist_ok=True)

        self.emulator_id = f"worker_{worker_id}_{time.time()}"

        # State
        self.env = None
        self.current_task = None
        self.task_instance = None
        self.history: list[str] = []
        self.steps = 0
        self.thinking_tokens = 0
        self.terminated = False

        # Reward shaping
        self.MAX_THINKING_TOKENS = 100
        self.THINKING_REWARD_SCALE = 0.1
        self.STEP_DECAY_FACTOR = 0.1
        self.MAX_PREMATURE_PENALTY = 0.5
        self.thinking_reward = True
        self.positive_step_reward = True
        self.negative_step_reward = True

        if base_avd_name:
            self._clone_avd_locally()

        self._initialize_env()

        # Pre-generate task parameter suite
        task_registry = registry.TaskRegistry()
        self.suite = suite_utils.create_suite(
            task_registry.get_registry(family="android_world"),
            n_task_combinations=20,
            seed=30,
            tasks=None,
            use_identical_params=False,
        )
        self.suite.suite_family = "android_world"

    # ---- AVD management ----

    def _clone_avd_locally(self):
        logger.info("[Worker %d] Cloning AVD: %s -> %s", self.worker_id, self.base_avd_name, self.avd_name)
        cache_avd_path = os.path.join(self.android_avd_home, self.avd_name + ".avd")
        cache_avd_ini_path = os.path.join(self.android_avd_home, self.avd_name + ".ini")
        for _ in range(3):
            try:
                remove_cache_avd(cache_avd_path, cache_avd_ini_path)
                break
            except Exception as e:
                logger.warning("[Worker %d] Failed to remove existing AVD: %s", self.worker_id, e)
                sleep(1)
        for attempt in range(3):
            try:
                clone_avd(self.base_avd_name, self.avd_name, self.android_avd_home)
                logger.info("[Worker %d] AVD cloned successfully", self.worker_id)
                return
            except Exception as e:
                logger.warning("[Worker %d] Clone attempt %d failed: %s", self.worker_id, attempt + 1, e)
                sleep(2)
        raise RuntimeError(f"Failed to clone AVD {self.base_avd_name} -> {self.avd_name} after 3 attempts")

    def _initialize_env(self):
        logger.info("[Worker %d] Initializing AndroidWorld environment...", self.worker_id)
        avd_dir = os.path.join(self.android_avd_home, f"{self.avd_name}.avd")
        avd_ini = os.path.join(self.android_avd_home, f"{self.avd_name}.ini")

        max_wait_time = 360
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            if os.path.exists(avd_dir) and os.path.exists(avd_ini):
                break
            sleep(5)
        else:
            raise RuntimeError(f"AVD files for {self.avd_name} not ready after {max_wait_time}s")

        os.environ["ANDROID_ADB_SERVER_PORT"] = str(self.adb_server_port)

        max_retries = 10
        for attempt in range(max_retries):
            try:
                self.env = env_launcher.load_and_setup_env(
                    console_port=self.console_port,
                    grpc_port=self.grpc_port,
                    adb_path=self.adb_path,
                    adb_server_port=self.adb_server_port,
                    emulator_path=self.emulator_path,
                    avd_name=self.avd_name,
                    android_sdk_root=self.android_sdk_root,
                    android_avd_home=self.android_avd_home,
                    emulator_setup=False,
                    freeze_datetime=True,
                )
                sleep(10)
                self.env.reset(go_home=True)
                logger.info("[Worker %d] Environment initialized successfully", self.worker_id)
                break
            except Exception as e:
                logger.warning("[Worker %d] Init attempt %d/%d failed: %s", self.worker_id, attempt + 1, max_retries, e)
                sleep(10)
                if attempt < max_retries - 1:
                    self.close()
                else:
                    raise

    # ---- Episode management ----

    def reset(self, task_name: str, params_idx: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment with a new task."""
        self.history = []
        self.steps = 0
        self.thinking_tokens = 0
        self.terminated = False

        self.task_instance = self.suite[task_name][params_idx]
        self.current_task = self.task_instance.goal

        max_init_retries = 10
        for init_attempt in range(max_init_retries):
            try:
                self.env.reset(go_home=True)
                if self.task_instance.initialized:
                    self.task_instance.tear_down(self.env)
                self.task_instance.initialize_task(self.env)
                sleep(5)

                obs = self._get_obs()
                self.max_steps = int(self.task_instance.complexity * 10)
                logger.info(
                    "[Worker %d] Task initialized: [%d]-[%s]-[%s]",
                    self.worker_id, self.max_steps, task_name, self.current_task,
                )
                if obs is None:
                    raise RuntimeError("Failed to get initial observation")

                info = {
                    "task": self.current_task,
                    "max_steps": self.max_steps,
                    "task_name": task_name,
                }
                return obs, info

            except Exception as e:
                logger.warning(
                    "[Worker %d] Task init failed (attempt %d/%d): %s",
                    self.worker_id, init_attempt + 1, max_init_retries, e,
                )
                self.close()
                sleep(5)
                self._initialize_env()
                if init_attempt >= max_init_retries - 1:
                    raise RuntimeError(
                        f"Task init failed for {task_name} after {max_init_retries} attempts: {e}"
                    )
                sleep(10)

    def _get_obs(self) -> Optional[dict[str, Any]]:
        """Get current observation from the environment."""
        try:
            state = self.env.get_state(wait_to_stabilize=True)
            image_path = None
            if self.save_images:
                image = Image.fromarray(state.pixels)
                image_path = os.path.join(self.temp_path, f"{self.emulator_id}_{self.steps}.png")
                image.save(image_path)

            return {
                "text": self.current_task,
                "image": state.pixels,
                "history": self.history.copy(),
                "image_path": image_path,
                "task": self.current_task,
                "max_steps": self.max_steps,
            }
        except Exception as e:
            logger.error("[Worker %d] Error getting observation at step %d: %s", self.worker_id, self.steps, e)
            return None

    def step(self, raw_action: str) -> tuple[Optional[dict], float, bool, dict]:
        """Execute an action and return (observation, reward, done, info)."""
        if self.terminated:
            return None, 0.0, True, {"won": False}

        before_action_obs = self._get_obs()
        if before_action_obs is None:
            self.terminated = True
            return None, 0.0, True, {"won": False}

        try:
            # Extract thinking tokens
            thought_match = re.search(r"<think>(.*?)</think>", raw_action, re.DOTALL)
            if thought_match:
                self.thinking_tokens += len(thought_match.group(1).split())

            action = parse_ui_action_from_response(
                response=raw_action,
                norm_width=999,
                norm_height=999,
                rescale_coords=True,
            )
            if action is None:
                sleep(1.0)
                return before_action_obs, 0.0, False, {"won": False}

        except Exception as e:
            logger.warning("[Worker %d] Failed to parse action: %s", self.worker_id, e)
            sleep(1.0)
            return before_action_obs, 0.0, False, {"won": False}

        # Record history
        action_description = self._get_action_description(raw_action)
        self.history.append(action_description)
        self.steps += 1

        try:
            if is_terminal_action(action):
                try:
                    base_reward = self.task_instance.is_successful(self.env)
                    logger.info("[Worker %d] Task evaluation: %s", self.worker_id, base_reward)
                except Exception as e:
                    logger.error("[Worker %d] Task evaluation failed: %s", self.worker_id, e)
                    base_reward = 0.0

                # step_reward = self._compute_step_reward(action, base_reward)
                step_reward = base_reward  # For now, use base reward directly without shaping
                self.terminated = True
                info = {"won": base_reward >= 1.0, "base_reward": base_reward, "step_count": self.steps}
                return before_action_obs, step_reward, True, info

            # Non-terminal action
            if action.action_type == "wait":
                try:
                    tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", raw_action, re.DOTALL)
                    if tool_call_match:
                        action_data = json.loads(tool_call_match.group(1).strip())
                        wait_time = action_data.get("arguments", {}).get("time", 2.0)
                    else:
                        wait_time = 2.0
                except Exception:
                    wait_time = 2.0
                sleep(wait_time)
            else:
                self.env.execute_action(action)
                sleep(2)

            after_action_obs = self._get_obs()
            if after_action_obs is None:
                self.terminated = True
                return None, 0.0, True, {"won": False}

            if self.steps >= self.max_steps:
                self.terminated = True
                return after_action_obs, 0.0, True, {"won": False, "step_count": self.steps}

            return after_action_obs, 0.0, False, {"won": False, "step_count": self.steps}

        except Exception as e:
            logger.error("[Worker %d] Error during step %d: %s", self.worker_id, self.steps, e)
            self.terminated = True
            return None, 0.0, True, {"won": False}

    def _get_action_description(self, response: str) -> str:
        """Extract action description from response for history tracking."""
        if "<conclusion>" in response:
            later_half = response.split("<conclusion>")[1].strip("\n")
            if "</conclusion>" in later_half:
                action = later_half.split("</conclusion>")[0].strip("\n")
                return clean_action_text(action)
            else:
                return clean_action_text(later_half.split("\n")[0])

        if "Action:" in response:
            action_start = response.find("Action:") + len("Action:")
            action_end = response.find("\n", action_start)
            if action_end == -1:
                action_end = len(response)
            return clean_action_text(response[action_start:action_end].strip())

        return clean_action_text(response[:200])

    def _compute_step_reward(self, action: json_action.JSONAction, base_reward: float) -> float:
        """Compute shaped reward with thinking and step penalties/bonuses."""
        is_complete = action.action_type == "status" and getattr(action, "goal_status", "") == "complete"

        if base_reward >= 1:
            avg_thinking = self.thinking_tokens / max(1, self.steps)
            normalized_thinking = 1 / (1 + np.exp(-(avg_thinking - self.MAX_THINKING_TOKENS / 2) / (self.MAX_THINKING_TOKENS / 4)))
            thinking_bonus = normalized_thinking * self.THINKING_REWARD_SCALE

            step_scale = np.exp(-self.STEP_DECAY_FACTOR * self.steps)
            step_scale = max(0.1, min(1.5, step_scale))

            if self.thinking_reward and self.positive_step_reward:
                return base_reward * step_scale + thinking_bonus
            elif self.thinking_reward:
                return base_reward + thinking_bonus
            elif self.positive_step_reward:
                return base_reward * step_scale
            else:
                return base_reward

        elif base_reward == 0 and is_complete:
            if self.negative_step_reward:
                premature_penalty = self.MAX_PREMATURE_PENALTY * (1 - self.steps / self.max_steps)
                return base_reward - premature_penalty
            return base_reward

        return base_reward

    def close(self):
        """Clean up the environment (kill emulator)."""
        logger.info("[Worker %d] Closing environment...", self.worker_id)
        try:
            if self.env is not None:
                self.env.close()
                self.env = None
        except Exception as e:
            logger.warning("[Worker %d] Error closing env: %s", self.worker_id, e)

        if self.save_images:
            try:
                if os.path.exists(self.temp_path):
                    for f in os.listdir(self.temp_path):
                        if f.startswith(self.emulator_id):
                            os.remove(os.path.join(self.temp_path, f))
            except Exception:
                pass

        kill_single_emulator(self.adb_path, f"emulator-{self.console_port}", self.adb_server_port)
