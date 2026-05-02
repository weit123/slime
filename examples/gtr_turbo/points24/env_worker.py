"""Points24 game engine and worker.

Pure-Python environment for the 24-point card game. No Ray dependency needed
since the environment is lightweight (image rendering + expression validation).
"""

from __future__ import annotations

import itertools
import json
import logging
import operator
import random
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
ACTION_LIST = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "+", "-", "*", "/", "(", ")", "="]


def is_solvable_24(digits: list[int], target: int = 24) -> bool:
    """Check if four digits can be combined to reach the target value via exhaustive search."""
    if len(digits) != 4:
        return False
    for perm in itertools.permutations(digits):
        for ops in itertools.product(OPS.keys(), repeat=3):
            expressions = [
                f"(({perm[0]}{ops[0]}{perm[1]}){ops[1]}{perm[2]}){ops[2]}{perm[3]}",
                f"({perm[0]}{ops[0]}({perm[1]}{ops[1]}{perm[2]})){ops[2]}{perm[3]}",
                f"({perm[0]}{ops[0]}{perm[1]}){ops[1]}({perm[2]}{ops[2]}{perm[3]})",
                f"{perm[0]}{ops[0]}(({perm[1]}{ops[1]}{perm[2]}){ops[2]}{perm[3]})",
                f"{perm[0]}{ops[0]}({perm[1]}{ops[1]}({perm[2]}{ops[2]}{perm[3]}))",
            ]
            for expr in expressions:
                try:
                    if abs(eval(expr) - target) < 1e-9:
                        return True
                except (ZeroDivisionError, OverflowError):
                    continue
    return False


def _render_cards_image(cards: list[int], size: tuple[int, int] = (300, 300)) -> Image.Image:
    """Render a simple image showing the current cards."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    n = len(cards)
    if n == 0:
        draw.text((size[0] // 2 - 40, size[1] // 2), "No cards", fill=(0, 0, 0))
        return img

    card_w = min(60, (size[0] - 20) // max(n, 1))
    card_h = 80
    start_x = (size[0] - n * card_w - (n - 1) * 10) // 2
    y = (size[1] - card_h) // 2

    for i, card in enumerate(cards):
        x = start_x + i * (card_w + 10)
        draw.rectangle([x, y, x + card_w, y + card_h], outline=(0, 0, 0), width=2)
        text = str(card)
        try:
            bbox = draw.textbbox((0, 0), text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(text)
        draw.text((x + (card_w - tw) // 2, y + (card_h - th) // 2), text, fill=(0, 0, 0))

    return img


class Points24Worker:
    """Manages a single episode of the 24-point card game.

    Args:
        cards: Initial set of 4 cards (integers 1-13).
        target: Target value (default 24).
        max_steps: Maximum number of actions before timeout.
        image_size: Size of rendered card images.
        treat_face_cards_as_10: If True, J/Q/K are treated as 10.
    """

    def __init__(
        self,
        cards: list[int] | None = None,
        target: int = 24,
        max_steps: int = 20,
        image_size: int = 300,
        treat_face_cards_as_10: bool = True,
        solvability_check: bool = True,
        max_resets: int = 100,
    ):
        self.target = target
        self.max_steps = max_steps
        self.image_size = image_size
        self.treat_face_cards_as_10 = treat_face_cards_as_10
        self.solvability_check = solvability_check
        self.max_resets = max_resets

        self.cards: list[int] = []
        self.formula: list[str] = []
        self.step_count = 0
        self.done = False
        self.won = False
        self.reward = 0.0

        if cards is not None:
            self.cards = list(cards)
        else:
            self._generate_solvable_cards()

    def _generate_solvable_cards(self) -> None:
        """Generate a random set of 4 solvable cards."""
        for _ in range(self.max_resets):
            cards = [random.randint(1, 13) for _ in range(4)]
            if self.treat_face_cards_as_10:
                cards = [min(c, 10) for c in cards]
            if not self.solvability_check or is_solvable_24(cards, self.target):
                self.cards = cards
                return
        self.cards = [1, 2, 3, 4]

    def reset(self) -> dict[str, Any]:
        """Reset for a new episode. Returns initial observation."""
        self.formula = []
        self.step_count = 0
        self.done = False
        self.won = False
        self.reward = 0.0
        if not self.cards:
            self._generate_solvable_cards()
        return self._get_observation()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        self.step_count += 1
        info: dict[str, Any] = {"legal": True, "won": False}

        if action not in ACTION_LIST:
            info["legal"] = False
            self.reward = -1.0
            self.done = True
            return self._get_observation(), self.reward, self.done, info

        if action == "=":
            success = self._evaluate_formula()
            self.won = success
            self.done = True
            self.reward = 10.0 if success else -1.0
            info["won"] = success
            return self._get_observation(), self.reward, self.done, info

        self.formula.append(action)

        if self.step_count >= self.max_steps:
            self.done = True
            self.reward = -1.0

        return self._get_observation(), self.reward, self.done, info

    def _evaluate_formula(self) -> bool:
        """Evaluate the accumulated formula and check if it equals the target."""
        expr = "".join(self.formula)
        try:
            result = eval(expr)
            return abs(result - self.target) < 1e-9
        except Exception:
            return False

    def _get_observation(self) -> dict[str, Any]:
        """Build the observation dict."""
        img = _render_cards_image(self.cards, (self.image_size, self.image_size))
        formula_str = "".join(self.formula) if self.formula else "(empty)"
        remaining = [str(c) for c in self.cards]

        return {
            "image": img,
            "cards": self.cards,
            "formula": formula_str,
            "remaining": remaining,
            "step": self.step_count,
            "done": self.done,
        }

    def get_reward(self) -> float:
        return self.reward

    def close(self) -> None:
        pass
