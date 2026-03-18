"""Per-sample ALFWorld environment wrapper for slime rollout."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from PIL import Image as PILImage

import ray
from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from examples.alfworld.env_pool import AlfWorldEnvPool
from examples.alfworld.prompts import get_alfworld_prompt

logger = logging.getLogger(__name__)


class AlfWorldEnv(BaseInteractionEnv):
    """Per-sample ALFWorld environment for slime rollout.

    This class wraps a remote AlfWorldWorker and adapts it to the
    BaseInteractionEnv interface. Key differences from Points24:
    - Uses Ray actor pool instead of direct worker
    - Async remote calls via asyncio.to_thread + ray.get
    - Tracks action history for prompt generation
    """

    def __init__(
        self,
        worker_ref: ray.ObjectRef,
        worker_id: int,
        pool: AlfWorldEnvPool,
        max_turns: int = 50,
        image_size: tuple[int, int] | None = None,
        action_only: bool = False,
        **kwargs,
    ):
        self.worker_ref = worker_ref
        self.worker_id = worker_id
        self.pool = pool
        self.max_turns = max_turns
        self._image_size = tuple(image_size) if image_size else None
        self._action_only = action_only

        self._final_reward = 0.0
        self._action_history: list[str] = []
        self._current_task = ""
        self._admissible_actions: list[str] = []

    async def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment."""
        self._final_reward = 0.0
        self._action_history = []

        obs, info = await asyncio.to_thread(ray.get, self.worker_ref.reset.remote())

        self._current_task = obs.get("task", "")
        self._admissible_actions = obs.get("admissible_actions", [])

        return obs, info

    async def step(self, response_text: str) -> tuple[dict | None, bool, dict]:
        """Execute one step."""
        obs, reward, done, info = await asyncio.to_thread(
            ray.get, self.worker_ref.step.remote(response_text)
        )

        if done:
            self._final_reward = reward

        if obs:
            self._admissible_actions = obs.get("admissible_actions", [])
            self._action_history = obs.get("action_history", [])

        return obs, done, info

    def get_reward(self) -> float:
        """Get the final reward."""
        return self._final_reward

    def format_observation(self, observation: dict, is_initial: bool = True) -> dict:
        """Format observation into VLM-compatible message.

        Uses GTR-Turbo's prompt format with task, history, and admissible actions.
        """
        content: list[dict] = []

        # Add image
        image = observation.get("image")
        if image is not None:
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image)
            if self._image_size is not None:
                image = image.resize(self._image_size, PILImage.LANCZOS)
            content.append({"type": "image", "image": image})

        # Generate text using GTR-Turbo's prompt format
        task = observation.get("task", self._current_task)
        action_history = observation.get("action_history", self._action_history)
        admissible_actions = observation.get("admissible_actions", self._admissible_actions)

        text = get_alfworld_prompt(
            task_description=task,
            action_history=action_history,
            admissible_actions=admissible_actions,
            action_only=self._action_only,
        )
        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    def close(self) -> None:
        """Release worker back to pool."""
        if self.pool is not None and self.worker_id is not None:
            self.pool.release(self.worker_id)
