"""Per-sample Android World environment wrapper for slime.

Implements the BaseInteractionEnv interface used by the multi-turn rollout pattern.
Each AndroidWorldEnv wraps a single Ray actor worker reference acquired from the pool.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import numpy as np
import ray
from PIL import Image as PILImage

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from examples.android_world.prompts import (
    ANDROID_WORLD_TEMPLATE_NO_HIS,
    ANDROID_WORLD_TEMPLATE_STEP,
)

logger = logging.getLogger(__name__)


class AndroidWorldEnv(BaseInteractionEnv):
    """Per-sample environment that delegates to a remote AndroidWorldWorker."""

    def __init__(
        self,
        worker_ref: Any,
        worker_id: int,
        pool: Any,
        task_name: str,
        params_idx: int,
        max_turns: int,
        image_size: tuple[int, int] | None = None,
    ):
        self.worker_ref = worker_ref
        self.worker_id = worker_id
        self.pool = pool
        self.task_name = task_name
        self.params_idx = params_idx
        self.max_turns = max_turns
        self.max_steps = max_turns  # updated after reset
        self.image_size = tuple(image_size) if image_size else None
        self.turn = 0
        self.final_reward = 0.0
        self.task_won = False
        self._closed = False

    async def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the remote worker with the assigned task."""
        obs, info = await asyncio.to_thread(
            ray.get, self.worker_ref.reset.remote(self.task_name, self.params_idx)
        )
        self.max_steps = info.get("max_steps", self.max_turns)
        return obs, info

    async def step(self, response_text: str) -> tuple[Optional[dict], bool, dict]:
        """Execute agent action on the remote worker.

        Returns:
            (observation, done, info)
        """
        self.turn += 1
        obs, step_reward, done, info = await asyncio.to_thread(
            ray.get, self.worker_ref.step.remote(response_text)
        )
        if done:
            self.final_reward = step_reward
            self.task_won = info.get("won", False)
        if obs is None:
            done = True
        return obs, done, info

    def get_reward(self) -> float:
        """Return the reward from the final step of the episode."""
        return self.final_reward

    def format_observation(self, observation: dict[str, Any], is_initial: bool = True) -> dict:
        """Format observation into a VLM-compatible chat message dict.

        Args:
            observation: Dict with 'image', 'task', 'history', 'max_steps' keys.
            is_initial: If True, include task description and instructions (first turn).
                If False, use a minimal step template since the task and prior actions
                are already in the token sequence from earlier turns.

        Returns a dict like::

            {"role": "user", "content": [
                {"type": "image", "image": <PIL.Image>},
                {"type": "text", "text": "..."}
            ]}
        """
        content: list[dict] = []

        # Screenshot
        if observation and observation.get("image") is not None:
            image = observation["image"]
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image)
            if self.image_size is not None:
                image = image.resize(self.image_size, PILImage.LANCZOS)
            content.append({"type": "image", "image": image})

        # Text
        task = observation.get("task", "") if observation else ""
        history = observation.get("history", []) if observation else []
        max_steps = observation.get("max_steps", self.max_steps) if observation else self.max_steps

        if is_initial:
            text = ANDROID_WORLD_TEMPLATE_NO_HIS.format(task_description=task)
        else:
            # Subsequent turns: only step counter + screenshot.
            # Task description and prior actions are already in the token sequence.
            text = ANDROID_WORLD_TEMPLATE_STEP.format(
                current_step=len(history) + 1,
                max_steps=max_steps,
            )

        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def close(self) -> None:
        """Release the worker back to the pool (does NOT destroy the emulator)."""
        if not self._closed and self.pool is not None and self.worker_id is not None:
            self.pool.release(self.worker_id)
            self._closed = True
