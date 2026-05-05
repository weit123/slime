"""ALFWorld environment wrapper for slime's BaseInteractionEnv interface.

Wraps a Ray-based AlfWorldWorker and adapts to the 3-tuple step() interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from examples.gtr_turbo.alfworld.alf_utils import process_action
from examples.gtr_turbo.alfworld.prompts import get_alfworld_prompt

logger = logging.getLogger(__name__)

OBS_TEXT_FEEDBACK_PREFIXES = (
    "move ",
    "put ",
    "take ",
    "heat ",
    "clean ",
    "cool ",
    "slice ",
    "use ",
    "close ",
)


class ALFWorldEnv(BaseInteractionEnv):
    """Multi-turn interaction environment for ALFWorld tasks.

    Wraps a remote Ray worker and conforms to BaseInteractionEnv's 3-tuple step() interface.
    """

    def __init__(
        self,
        *,
        worker,
        worker_id: int,
        pool,
        max_turns: int = 40,
        image_size: tuple[int, int] | None = (300, 300),
        action_only: bool = False,
        remote_reset_timeout_sec: float = 180.0,
        remote_step_timeout_sec: float = 90.0,
    ):
        self.worker = worker
        self.worker_id = worker_id
        self.pool = pool
        self.max_turns = max_turns
        self.image_size = image_size
        self.action_only = action_only
        self.remote_reset_timeout_sec = remote_reset_timeout_sec
        self.remote_step_timeout_sec = remote_step_timeout_sec

        self.task_description = ""
        self.admissible_commands: list[str] = []
        self.action_history: list[str] = []
        self.response_history: list[str] = []
        self.legal_history: list[bool] = []
        self._cumulative_reward = 0.0
        self._obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._released = False
        self._worker_failed = False

    def reset(self, task_file=None) -> tuple[dict, dict]:
        import ray

        start = time.monotonic()
        try:
            obs = ray.get(self.worker.reset.remote(task_file=task_file), timeout=self.remote_reset_timeout_sec)
        except Exception:
            self._worker_failed = True
            raise
        if self.pool is not None:
            self.pool.record_timing(self.worker_id, reset_sec=time.monotonic() - start)
        self._apply_reset_obs(obs)
        return obs, {}

    async def async_reset(self, task_file=None) -> tuple[dict, dict]:
        import ray

        start = time.monotonic()
        try:
            obs = await asyncio.to_thread(
                ray.get,
                self.worker.reset.remote(task_file=task_file),
                timeout=self.remote_reset_timeout_sec,
            )
        except Exception:
            self._worker_failed = True
            raise
        if self.pool is not None:
            self.pool.record_timing(self.worker_id, reset_sec=time.monotonic() - start)
        self._apply_reset_obs(obs)
        return obs, {}

    def _apply_reset_obs(self, obs: dict[str, Any]) -> None:
        self.task_description = obs.get("task", "")
        self.admissible_commands = obs.get("admissible_commands", [])
        self.action_history = []
        self.response_history = []
        self.legal_history = []
        self._cumulative_reward = 0.0
        self._obs = obs
        self._last_info = {}

    def step(self, response_text: str) -> tuple[dict, bool, dict]:
        """Execute model's response as an ALFWorld action.

        Returns:
            (observation, done, info) — 3-tuple per BaseInteractionEnv contract.
        """
        import ray

        action, legal = process_action(response_text, self.admissible_commands)
        self.action_history.append(action)
        self.response_history.append(response_text)
        self.legal_history.append(legal)

        start = time.monotonic()
        try:
            obs, reward, done, info = ray.get(
                self.worker.step.remote(action),
                timeout=self.remote_step_timeout_sec,
            )
        except Exception:
            self._worker_failed = True
            raise
        if self.pool is not None:
            self.pool.record_timing(self.worker_id, step_sec=time.monotonic() - start)
        return self._apply_step_result(action, legal, obs, reward, done, info)

    async def async_step(self, response_text: str) -> tuple[dict, bool, dict]:
        """Async version of step() that does not block the rollout event loop."""
        import ray

        action, legal = process_action(response_text, self.admissible_commands)
        self.action_history.append(action)
        self.response_history.append(response_text)
        self.legal_history.append(legal)

        start = time.monotonic()
        try:
            obs, reward, done, info = await asyncio.to_thread(
                ray.get,
                self.worker.step.remote(action),
                timeout=self.remote_step_timeout_sec,
            )
        except Exception:
            self._worker_failed = True
            raise
        if self.pool is not None:
            self.pool.record_timing(self.worker_id, step_sec=time.monotonic() - start)
        return self._apply_step_result(action, legal, obs, reward, done, info)

    def _apply_step_result(
        self,
        action: str,
        legal: bool,
        obs: dict[str, Any],
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> tuple[dict, bool, dict]:
        self._obs = obs
        self._last_info = info
        self._cumulative_reward += reward
        self.admissible_commands = obs.get("admissible_commands", [])

        info["reward"] = reward
        info["cumulative_reward"] = self._cumulative_reward
        info["legal"] = legal
        info["action_taken"] = action
        info["total_actions"] = len(self.action_history)
        info["illegal_actions"] = sum(1 for item in self.legal_history if not item)
        return obs, done, info

    def _should_include_obs_text(self) -> bool:
        if not self.action_history:
            return False
        last_action = self.action_history[-1].lower().strip()
        return last_action == "inventory" or any(
            last_action.startswith(prefix) for prefix in OBS_TEXT_FEEDBACK_PREFIXES
        )

    def format_observation(self, observation: dict) -> dict:
        """Convert observation to VLM-compatible chat message."""
        content = []

        raw_img = observation.get("image")
        if raw_img is not None:
            if isinstance(raw_img, np.ndarray):
                pil_img = Image.fromarray(raw_img)
            else:
                pil_img = raw_img
            content.append({"type": "image", "image": pil_img})

        prompt_text = get_alfworld_prompt(
            task_description=observation.get("task", self.task_description),
            action_history=self.action_history,
            admissible_actions=observation.get("admissible_commands", self.admissible_commands),
            action_only=self.action_only,
        )
        if self._should_include_obs_text():
            prompt_text += f"Previous action feedback: {observation.get('obs_text', '')}. "
        content.append({"type": "text", "text": prompt_text})

        return {"role": "user", "content": content}

    def get_reward(self) -> float:
        return self._cumulative_reward

    def get_metrics(self) -> dict[str, Any]:
        info = self._last_info
        total_actions = len(self.action_history)
        illegal_actions = sum(1 for item in self.legal_history if not item)
        return {
            "success": bool(info.get("won", False)),
            "goal_condition_success_rate": float(info.get("goal_condition_success_rate", 0.0)),
            "steps": total_actions,
            "illegal_actions": illegal_actions,
            "total_actions": total_actions,
            "illegal_action_rate": illegal_actions / total_actions if total_actions else 0.0,
            "actions": list(self.action_history),
            "legal": list(self.legal_history),
            "responses": list(self.response_history),
        }

    async def restart_worker(self) -> None:
        self.worker = await self.pool.restart(self.worker_id)
        self._worker_failed = False

    def close(self):
        if not self._released and self.pool is not None and self.worker_id is not None:
            if self._worker_failed:
                self.pool.discard(self.worker_id)
            else:
                self.pool.release(self.worker_id)
            self._released = True


async def build_env(sample, args) -> ALFWorldEnv:
    """Async factory function for ALFWorldEnv.

    Acquires a worker from the singleton pool. Called by the rollout function.
    """
    from examples.gtr_turbo.alfworld.env_pool import AlfWorldEnvPool

    config = {}
    if hasattr(args, "custom_config") and args.custom_config:
        config = args.custom_config
    elif hasattr(args, "alfworld_config_file"):
        # custom_config_path sets individual args attributes, not a dict
        for key in [
            "alfworld_config_file", "max_turns", "max_context_len", "num_workers",
            "resources_per_worker", "action_only", "image_size", "repetition_penalty",
            "legacy_build_path", "xvfb_display_base", "qwen3_vl_eval_prompt_format",
            "use_gpu_xorg", "xorg_display_base", "xorg_num_displays", "prewarm_workers",
            "render_image", "render_depth_image", "render_class_image", "render_object_image",
            "remote_reset_timeout_sec", "remote_step_timeout_sec",
        ]:
            if hasattr(args, key):
                config[key] = getattr(args, key)

    pool = await AlfWorldEnvPool.get_instance(config)
    worker, worker_id = await pool.acquire()

    image_size = config.get("image_size", [300, 300])
    if isinstance(image_size, list):
        image_size = tuple(image_size)

    return ALFWorldEnv(
        worker=worker,
        worker_id=worker_id,
        pool=pool,
        max_turns=config.get("max_turns", 40),
        image_size=image_size,
        action_only=config.get("action_only", False),
        remote_reset_timeout_sec=float(config.get("remote_reset_timeout_sec", 180.0)),
        remote_step_timeout_sec=float(config.get("remote_step_timeout_sec", 90.0)),
    )
