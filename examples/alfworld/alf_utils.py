"""ALFWorld environment utilities adapted from GTR-Turbo.

This module provides the AlfEnv wrapper class and helper functions for
interacting with ALFWorld environments. Adapted from GTR-Turbo/Turbo_ALF/alf_utils.py
with the following modifications:
- Removed CUDA-specific code (slime handles device management)
- Simplified get_obs_image() to return numpy array
- Adapted process_action() for single-action use case

Original source: GTR-Turbo/Turbo_ALF/alf_utils.py (Apache-2.0 licensed)
"""

from __future__ import annotations

import logging
import os
import random
import re
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

logger = logging.getLogger(__name__)


# Action list for ALFWorld (must match GTR-Turbo)
ALF_ACTION_LIST = [
    "pass", "goto", "pick", "put", "open", "close",
    "toggle", "heat", "clean", "cool", "slice",
    "inventory", "examine", "look",
]


def load_config_file(path: str) -> dict:
    """Load ALFWorld configuration from YAML file."""
    assert os.path.exists(path), f"Config file not found: {path}"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def get_obs_image(env) -> np.ndarray:
    """Get observation image as numpy array (H, W, C).

    Adapted from GTR-Turbo to return numpy array instead of CUDA tensor.

    Args:
        env: ALFWorld environment instance

    Returns:
        numpy array of shape (H, W, C) with dtype uint8
    """
    current_frames = env.get_frames()
    if not current_frames:
        return np.zeros((300, 300, 3), dtype=np.uint8)

    # Get first frame (batch_size=1)
    image = np.array(current_frames[0])

    # Convert RGB to BGR if needed (THOR outputs RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Already RGB, keep as is for PIL compatibility
        pass

    return image.astype(np.uint8)


def process_action(
    action_text: str,
    admissible_commands: list[str],
) -> tuple[str, bool]:
    """Parse action from model response and match against admissible commands.

    This function extracts the action from the model's JSON response and
    finds a matching admissible command. If no match is found, a random
    admissible command is selected.

    Args:
        action_text: Model's text response (JSON format with "action" field)
        admissible_commands: List of valid commands for current state

    Returns:
        Tuple of (command_string, is_legal)
    """
    if not admissible_commands:
        return "look", False

    legal_action = False
    action_text = action_text.lower() if action_text else ""

    if not action_text:
        # Empty action, choose random
        return admissible_commands[random.randint(0, len(admissible_commands) - 1)], False

    try:
        # Find "action": field in the response
        action_index = action_text.find('"action":')
        if action_index == -1:
            # If no "action": field, use last 50 characters
            string = action_text[-50:]
        else:
            string = action_text[action_index:]

        # Try to match against admissible commands
        for cmd in admissible_commands:
            cmd_lower = cmd.lower()
            if cmd_lower in string:
                return cmd, True

        # No match found
        legal_action = False

    except Exception as e:
        logger.debug(f"Error parsing action: {e}")
        legal_action = False

    # Fallback: random admissible command
    if not legal_action:
        random_cmd = admissible_commands[random.randint(0, len(admissible_commands) - 1)]
        return random_cmd, False

    return admissible_commands[0], False


def compute_reward(infos: dict, legal_action: bool) -> float:
    """Compute shaped reward for ALFWorld environment.

    Reward formula: r = 50 * won + goal_condition_success_rate - 1 * illegal_action

    Args:
        infos: Info dict from environment step
        legal_action: Whether the action was legal

    Returns:
        Scalar reward value
    """
    try:
        won = float(infos.get('won', [0])[0])
        goal_success = float(infos.get('goal_condition_success_rate', [0])[0])
    except (IndexError, TypeError):
        won = 0.0
        goal_success = 0.0

    reward = 50 * won + goal_success
    if not legal_action:
        reward -= 1

    return reward


class AlfEnv(gym.Env):
    """ALFWorld environment wrapper compatible with slime.

    This class wraps the ALFWorld environment (AlfredThorEnv) and provides
    a standard Gym interface. It handles:
    - Environment initialization from config file
    - Action parsing and execution
    - Reward computation
    - Observation extraction (image + text)
    """

    def __init__(self, config_file: str):
        """Initialize the ALFWorld environment.

        Args:
            config_file: Path to ALFWorld config YAML file
        """
        import alfworld.agents.environment as environment

        config = load_config_file(config_file)
        env_type = config['env']['type']

        env = getattr(environment, env_type)(config, train_eval='train')
        self.env = env.init_env(batch_size=1)

        self.action_space = spaces.Discrete(len(ALF_ACTION_LIST))
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)

        self.prev_admissible_commands: list[str] = []
        self.num_envs = 1
        self._last_observation_text = ""

    def reset(self, seed: int = 42) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: Random seed

        Returns:
            Tuple of (observation_image, info_dict)
        """
        self.env.seed(seed)
        obs, infos = self.env.reset()

        # Extract observation text
        self._last_observation_text = obs[0] if isinstance(obs, list) else str(obs)
        infos['observation_text'] = self._last_observation_text

        # Store admissible commands
        admissible = infos.get('admissible_commands', [[]])
        self.prev_admissible_commands = list(admissible[0]) if admissible else []

        return self._get_obs(), infos

    def step(self, action_text: str) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment.

        Args:
            action_text: Model's text response (JSON format)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Parse action from text
        parsed_action, legal_action = process_action(action_text, self.prev_admissible_commands)

        # Step the environment
        obs, scores, dones, infos = self.env.step([parsed_action])

        # Extract observation text
        self._last_observation_text = obs[0] if isinstance(obs, list) else str(obs)
        infos['observation_text'] = self._last_observation_text

        # Compute reward
        reward = compute_reward(infos, legal_action)

        # Update admissible commands
        admissible = infos.get('admissible_commands', [[]])
        self.prev_admissible_commands = list(admissible[0]) if admissible else []

        # Extract done flag
        done = bool(dones[0]) if isinstance(dones, list) else bool(dones)

        # Add legal action info
        infos['legal_action'] = legal_action
        infos['parsed_action'] = parsed_action

        return self._get_obs(), reward, done, infos

    def _get_obs(self) -> np.ndarray:
        """Get current observation image."""
        return get_obs_image(self.env)

    def get_task_description(self) -> str:
        """Get the current task description.

        Requires alfworld.agents.utils.misc.get_templated_task_desc
        """
        try:
            from alfworld.agents.utils.misc import get_templated_task_desc
            task_desc = get_templated_task_desc(self.env.envs[0].traj_data)
            return task_desc
        except Exception as e:
            logger.debug(f"Could not get task description: {e}")
            return self._last_observation_text

    def close(self):
        """Close the environment."""
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
        except Exception as e:
            logger.debug(f"Error closing environment: {e}")
