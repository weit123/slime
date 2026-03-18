"""ALFWorld environment worker as Ray actor.

This module provides AlfWorldWorker, a Ray remote actor that wraps AlfEnv
for distributed environment execution. Each worker manages a single THOR
environment instance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image as PILImage

from examples.alfworld.alf_utils import AlfEnv, process_action, compute_reward

logger = logging.getLogger(__name__)


class AlfWorldWorker:
    """Worker wrapping a single AlfEnv instance.

    This class is designed to be used as a Ray remote actor. It provides
    synchronous methods that can be called remotely via ray.get().

    Note: The @ray.remote decorator is applied in env_pool.py to allow
    dynamic resource configuration.
    """

    def __init__(
        self,
        worker_id: int,
        config_file: str,
        max_steps: int = 50,
        image_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        """Initialize the ALFWorld worker.

        Args:
            worker_id: Unique identifier for this worker
            config_file: Path to ALFWorld config YAML
            max_steps: Maximum steps per episode
            image_size: Optional (width, height) to resize observations
            **kwargs: Additional arguments (ignored)
        """
        self.worker_id = worker_id
        self.config_file = config_file
        self.max_steps = max_steps
        self.image_size = tuple(image_size) if image_size else None

        # Initialize environment
        self.env = AlfEnv(config_file)

        # Episode state
        self.steps = 0
        self.final_reward = 0.0
        self.terminated = False
        self.action_history: list[str] = []
        self.current_task = ""
        self.prev_admissible_commands: list[str] = []

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional random seed

        Returns:
            Tuple of (observation_dict, info_dict)
        """
        self.steps = 0
        self.final_reward = 0.0
        self.terminated = False
        self.action_history = []

        obs_image, info = self.env.reset(seed=seed or 42)

        # Extract state
        self.prev_admissible_commands = info.get('admissible_commands', [[]])
        if isinstance(self.prev_admissible_commands, list) and len(self.prev_admissible_commands) > 0:
            if isinstance(self.prev_admissible_commands[0], list):
                self.prev_admissible_commands = self.prev_admissible_commands[0]

        # Get task description
        try:
            self.current_task = self.env.get_task_description()
        except Exception:
            self.current_task = info.get('observation_text', '')

        # Process image
        image = self._process_image(obs_image)

        observation = {
            "image": image,
            "task": self.current_task,
            "action_history": [],
            "admissible_actions": list(self.prev_admissible_commands),
        }

        info_dict = {
            "task": self.current_task,
            "admissible_actions": list(self.prev_admissible_commands),
            "max_steps": self.max_steps,
        }

        return observation, info_dict

    def step(self, response_text: str) -> tuple[dict | None, float, bool, dict]:
        """Execute one step in the environment.

        Args:
            response_text: Model's text response (JSON format)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.terminated:
            return None, 0.0, True, {"terminated": True}

        # Step the environment
        obs_image, reward, done, info = self.env.step(response_text)

        self.steps += 1

        # Track action history
        parsed_action = info.get('parsed_action', '')
        if parsed_action:
            self.action_history.append(parsed_action)

        # Update state
        admissible = info.get('admissible_commands', [[]])
        if isinstance(admissible, list) and len(admissible) > 0:
            if isinstance(admissible[0], list):
                admissible = admissible[0]
        self.prev_admissible_commands = list(admissible)

        # Check termination
        done = done or self.steps >= self.max_steps
        self.terminated = done

        if done:
            self.final_reward = reward

        # Process image
        image = self._process_image(obs_image) if not done else None

        observation = {
            "image": image,
            "task": self.current_task,
            "action_history": list(self.action_history),
            "admissible_actions": list(self.prev_admissible_commands),
        } if not done else None

        info_dict = {
            "task": self.current_task,
            "admissible_actions": list(self.prev_admissible_commands),
            "legal_action": info.get('legal_action', False),
            "last_action": parsed_action,
            "step": self.steps,
            "won": info.get('won', [False])[0] if isinstance(info.get('won'), list) else info.get('won', False),
        }

        return observation, reward, done, info_dict

    def get_reward(self) -> float:
        """Get the final reward from the episode."""
        return self.final_reward

    def _process_image(self, image: np.ndarray) -> PILImage.Image:
        """Process observation image.

        Args:
            image: numpy array (H, W, C)

        Returns:
            PIL Image (optionally resized)
        """
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        if self.image_size is not None:
            pil_image = pil_image.resize(self.image_size, PILImage.LANCZOS)

        return pil_image

    def close(self) -> None:
        """Close the environment."""
        self.terminated = True
        try:
            self.env.close()
        except Exception as e:
            logger.debug(f"Error closing worker {self.worker_id}: {e}")
