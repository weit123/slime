"""Points24 environment wrapper for slime's BaseInteractionEnv interface.

Adapts Points24Worker to the 3-tuple step() interface expected by the
current slime multi-turn rollout framework.
"""

from __future__ import annotations

from typing import Any

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from examples.gtr_turbo.points24.env_worker import Points24Worker
from examples.gtr_turbo.points24.prompts import (
    SYSTEM_PROMPT,
    get_points24_prompt,
    parse_single_action,
)


class Points24Env(BaseInteractionEnv):
    """Multi-turn interaction environment for the 24-point card game.

    Wraps Points24Worker and conforms to BaseInteractionEnv's 3-tuple step() interface.
    """

    def __init__(self, *, worker: Points24Worker, max_turns: int = 20, action_only: bool = False):
        self.worker = worker
        self.max_turns = max_turns
        self.action_only = action_only
        self._obs: dict[str, Any] | None = None
        self._cumulative_reward = 0.0

    def reset(self) -> tuple[dict, dict]:
        obs = self.worker.reset()
        self._obs = obs
        self._cumulative_reward = 0.0
        return obs, {}

    def step(self, response_text: str) -> tuple[dict, bool, dict]:
        """Execute model's response as an action.

        Returns:
            (observation, done, info) — 3-tuple per BaseInteractionEnv contract.
            Reward is stored in info["reward"] for extraction by the rollout function.
        """
        action, legal = parse_single_action(response_text)
        obs, reward, done, info = self.worker.step(action)
        self._obs = obs
        self._cumulative_reward += reward
        info["reward"] = reward
        info["cumulative_reward"] = self._cumulative_reward
        info["legal"] = legal
        return obs, done, info

    def format_observation(self, observation: dict) -> dict:
        """Convert observation to VLM-compatible chat message."""
        content = []

        img = observation.get("image")
        if img is not None:
            content.append({"type": "image", "image": img})

        prompt_text = get_points24_prompt(
            cards=observation.get("cards", []),
            formula=observation.get("formula", "(empty)"),
            action_only=self.action_only,
        )
        content.append({"type": "text", "text": prompt_text})

        return {"role": "user", "content": content}

    def close(self):
        self.worker.close()

    def get_reward(self) -> float:
        return self.worker.get_reward()


def build_env(sample, args) -> Points24Env:
    """Factory function required by slime's rollout framework.

    Called by geo3k_vlm_multi_turn/rollout.py via _build_env().
    """
    config = {}
    if hasattr(args, "custom_config") and args.custom_config:
        config = args.custom_config

    cards = None
    if hasattr(sample, "metadata") and sample.metadata:
        cards = sample.metadata.get("cards")

    worker = Points24Worker(
        cards=cards,
        target=config.get("target_value", 24),
        max_steps=config.get("max_steps", 20),
        image_size=config.get("image_size", 300),
        treat_face_cards_as_10=config.get("treat_face_cards_as_10", True),
        solvability_check=config.get("solvability_check", True),
        max_resets=config.get("max_resets", 100),
    )

    return Points24Env(
        worker=worker,
        max_turns=config.get("max_turns", 20),
        action_only=config.get("action_only", False),
    )
