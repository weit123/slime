"""Per-sample Points24 environment wrapper for slime rollout.

This module provides the Points24Env class that adapts the lightweight
Points24Worker to slime's BaseInteractionEnv interface for multi-turn
rollout generation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image as PILImage

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from examples.points24.env_worker import Points24Worker
from examples.points24.prompts import get_points24_prompt

logger = logging.getLogger(__name__)


class Points24Env(BaseInteractionEnv):
    """Per-sample Points24 environment for slime rollout.

    This class wraps a Points24Worker and adapts it to the BaseInteractionEnv
    interface used by slime's multi-turn rollout pipeline. Key responsibilities:

    1. Managing the environment lifecycle (reset, step, close)
    2. Formatting observations into VLM-compatible message format
    3. Tracking episode state and computing rewards

    Note: Unlike android_world, Points24 does NOT use Ray actors or a pool.
    The Points24Worker is created on-demand per sample since the underlying
    Point24Env is a lightweight pure-Python environment.
    """

    def __init__(
        self,
        worker: Points24Worker | None = None,
        max_turns: int = 30,
        image_size: tuple[int, int] | None = None,
        action_only: bool = False,
        **kwargs,
    ):
        """Initialize the Points24 environment wrapper.

        Args:
            worker: Optional pre-created Points24Worker. If None, one will be
                    created on the first reset.
            max_turns: Maximum number of turns per episode
            image_size: Optional (width, height) to resize observations
            action_only: If True, omit thoughts field in prompts
            **kwargs: Additional arguments passed to Points24Worker
        """
        self._worker = worker
        self._max_turns = max_turns
        self._image_size = tuple(image_size) if image_size else None
        self._action_only = action_only
        self._worker_kwargs = kwargs

        # Episode state
        self._final_reward = 0.0
        self._steps = 0

    @property
    def worker(self) -> Points24Worker:
        """Get or create the underlying Points24Worker."""
        if self._worker is None:
            self._worker = Points24Worker(
                max_steps=self._max_turns,
                image_size=self._image_size,
                **self._worker_kwargs,
            )
        return self._worker

    async def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Returns:
            Tuple of (observation_dict, info_dict)
        """
        self._final_reward = 0.0
        self._steps = 0
        return self.worker.reset()

    async def step(self, response_text: str) -> tuple[dict | None, bool, dict]:
        """Execute one step in the environment.

        Args:
            response_text: Model's text response (JSON format)

        Returns:
            Tuple of (observation, done, info)
        """
        self._steps += 1
        obs, reward, done, info = self.worker.step(response_text)

        if done:
            self._final_reward = reward

        return obs, done, info

    def get_reward(self) -> float:
        """Get the final reward from the episode.

        Returns:
            Final reward (10 for success, -1 for failure)
        """
        return self._final_reward

    def format_observation(
        self,
        observation: dict[str, Any],
        is_initial: bool = True,
    ) -> dict:
        """Format observation into a VLM-compatible chat message dict.

        This method adapts GTR-Turbo's prompt format to slime's interface.
        The text content is generated using get_points24_prompt() which
        matches GTR-Turbo exactly.

        Args:
            observation: Dict with 'image', 'cards', 'formula', 'numbers' keys
            is_initial: If True, this is the first turn (unused for Points24
                        since prompt format is consistent across turns)

        Returns:
            Dict like:
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": "..."}
                ]
            }
        """
        content: list[dict] = []

        # 1. Add image (from observation)
        image = observation.get("image")
        if image is not None:
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image)
            if self._image_size is not None and image is not None:
                image = image.resize(self._image_size, PILImage.LANCZOS)
            content.append({"type": "image", "image": image})

        # 2. Generate text using GTR-Turbo's prompt format (keep text consistent!)
        formula = observation.get("formula", [])
        text = get_points24_prompt(formula, action_only=self._action_only)
        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    def close(self) -> None:
        """Close the environment."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None
