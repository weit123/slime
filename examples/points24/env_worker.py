"""Lightweight Points24 environment wrapper for slime.

No Ray actor needed - the Point24Env is a pure Python environment that can be
created on-demand per sample with minimal overhead.
"""

from __future__ import annotations

import logging
import random
from itertools import permutations, product, chain, zip_longest
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage

from examples.points24.prompts import parse_single_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 24 Game Solvability Checker
# Reference: https://github.com/LeslieTrue/SFTvsRL/blob/master/gym/gym_cards/envs/general_points_oneline.py
# Original source: https://rosettacode.org/wiki/24_game/Solve#Python
# ---------------------------------------------------------------------------

def is_solvable_24(digits: list[int], target: float = 24.0) -> bool:
    """Check if a set of 4 digits can form an expression equal to the target.

    Uses exhaustive search over all permutations of:
    - Digit orderings
    - Operator combinations (+, -, *, /)
    - Bracket insertion points

    Args:
        digits: List of 4 integers (1-10)
        target: Target value (default 24)

    Returns:
        True if solvable, False otherwise

    Reference:
        - https://github.com/LeslieTrue/SFTvsRL/blob/master/gym/gym_cards/envs/general_points_oneline.py
        - https://rosettacode.org/wiki/24_game/Solve#Python
    """
    from fractions import Fraction as F

    digilen = len(digits)
    # length of an exp without brackets
    exprlen = 2 * digilen - 1
    # permute all the digits
    # added shuffle to avoid always the same solution
    digiperm = sorted(set(permutations(digits)))
    random.shuffle(digiperm)
    # All the possible operator combinations
    opcomb = list(product('+-*/', repeat=digilen-1))
    # All the bracket insertion points:
    brackets = ([()] + [(x, y)
                for x in range(0, exprlen, 2)
                for y in range(x+4, exprlen+2, 2)
                if (x, y) != (0, exprlen+1)]
                + [(0, 3+1, 4+2, 7+3)])  # double brackets case

    for d in digiperm:
        for ops in opcomb:
            if '/' in ops:
                d2 = [('F(%s)' % i) for i in d]  # Use Fractions for accuracy
            else:
                d2 = d
            ex = list(chain.from_iterable(zip_longest(d2, ops, fillvalue='')))
            for b in brackets:
                exp = ex[::]
                for insertpoint, bracket in zip(b, '()'*(len(b)//2)):
                    exp.insert(insertpoint, bracket)
                txt = ''.join(str(i) for i in exp)
                try:
                    num = eval(txt)
                except ZeroDivisionError:
                    continue
                if num == target:
                    return True
    return False


class Points24Worker:
    """Lightweight wrapper around Point24Env for slime compatibility.

    This class wraps the gym_cards Point24Env and provides an interface
    compatible with slime's rollout pipeline. It handles:
    - Resetting the environment with random cards
    - Verifying task solvability before each episode
    - Parsing model JSON responses into discrete actions
    - Tracking episode state and computing rewards

    Note: This is NOT a Ray actor. Point24Env is lightweight enough to be
    created on-demand during each rollout sample.
    """

    def __init__(
        self,
        worker_id: int = 0,
        max_steps: int = 20,  # Changed from 30 to 20
        treat_face_cards_as_10: bool = True,
        target_points: int = 24,
        image_size: tuple[int, int] | None = None,
        max_resets: int = 100,  # Max attempts to find solvable task
        **kwargs,
    ):
        """Initialize the Points24 worker.

        Args:
            worker_id: Identifier for this worker (for logging)
            max_steps: Maximum number of steps per episode (default: 20)
            treat_face_cards_as_10: If True, J/Q/K count as 10
            target_points: Target value for the formula (default: 24)
            image_size: Optional (width, height) to resize observations
            max_resets: Maximum attempts to find a solvable task
            **kwargs: Additional arguments (ignored)
        """
        from gym_cards.envs.points import Point24Env

        self.env = Point24Env(
            treat_face_cards_as_10=treat_face_cards_as_10,
            target_points=target_points,
        )
        self.worker_id = worker_id
        self.max_steps = max_steps
        self.image_size = tuple(image_size) if image_size else None
        self.max_resets = max_resets
        self.target_points = target_points

        # Episode state
        self.steps = 0
        self.final_reward = 0.0
        self.terminated = False
        self.current_cards = []
        self.current_formula = []
        self.current_numbers = []

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and return initial observation.

        Verifies that the generated task is solvable before returning.
        If the task is unsolvable, regenerates until a solvable one is found.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Tuple of (observation_dict, info_dict) where:
            - observation_dict contains 'image', 'cards', 'formula', 'numbers'
            - info_dict contains metadata about the task
        """
        self.steps = 0
        self.final_reward = 0.0
        self.terminated = False

        # Try to find a solvable task
        for attempt in range(self.max_resets):
            # Reset the underlying environment
            # Use seed + attempt to ensure different cards each try
            current_seed = None if seed is None else seed + attempt
            obs_image, info = self.env.reset(seed=current_seed)

            # Extract state from info
            self.current_cards = info.get("Cards", [])
            self.current_formula = info.get("Formula", [])
            self.current_numbers = info.get("Numbers", [])

            # Check solvability
            if is_solvable_24(self.current_numbers, target=float(self.target_points)):
                break

            logger.debug(
                "[Worker %d] Task unsolvable (attempt %d): cards=%s, numbers=%s",
                self.worker_id, attempt, self.current_cards, self.current_numbers
            )
        else:
            logger.warning(
                "[Worker %d] Failed to find solvable task after %d attempts, using last generated task",
                self.worker_id, self.max_resets
            )

        # Process image
        if isinstance(obs_image, np.ndarray):
            image = PILImage.fromarray(obs_image)
        else:
            image = obs_image

        if self.image_size is not None:
            image = image.resize(self.image_size, PILImage.LANCZOS)

        # Build observation dict
        observation = {
            "image": image,
            "cards": self.current_cards,
            "formula": self.current_formula.copy(),
            "numbers": self.current_numbers,
        }

        # Build info dict
        info_dict = {
            "cards": self.current_cards,
            "formula": self.current_formula.copy(),
            "numbers": self.current_numbers,
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

        # Parse action from model response
        action_idx, is_legal = parse_single_action(response_text)

        # Step the environment
        obs_image, reward, terminated, truncated, info = self.env.step(action_idx)

        self.steps += 1
        self.current_formula = info.get("Formula", [])

        # Update termination state
        done = terminated or truncated or self.steps >= self.max_steps
        self.terminated = done

        # Track final reward
        if done:
            self.final_reward = reward

        # Process image
        if obs_image is not None:
            if isinstance(obs_image, np.ndarray):
                image = PILImage.fromarray(obs_image)
            else:
                image = obs_image

            if self.image_size is not None:
                image = image.resize(self.image_size, PILImage.LANCZOS)
        else:
            image = None

        # Build observation dict
        observation = {
            "image": image,
            "cards": self.current_cards,
            "formula": self.current_formula.copy(),
            "numbers": info.get("Numbers", []),
        } if not done else None

        # Build info dict
        info_dict = {
            "cards": self.current_cards,
            "formula": self.current_formula.copy(),
            "numbers": info.get("Numbers", []),
            "is_legal": is_legal,
            "step": self.steps,
            "last_action_idx": action_idx,
        }

        return observation, reward, done, info_dict

    def get_reward(self) -> float:
        """Get the final reward from the episode.

        Returns:
            Final reward (10 for success, -1 for failure, 0 otherwise)
        """
        return self.final_reward

    def close(self) -> None:
        """Close the environment (no-op for lightweight env)."""
        self.terminated = True
