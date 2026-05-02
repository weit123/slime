"""Points24 prompt templates and action parsing.

Follows GTR-Turbo's structured JSON prompt format for the 24-point card game.
"""

from __future__ import annotations

import json
import random
import re

ACTION_LIST = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "+", "-", "*", "/", "(", ")", "="]

SYSTEM_PROMPT = "You are an expert 24 points card game player."


def get_points24_prompt(
    cards: list[int],
    formula: str = "(empty)",
    action_only: bool = False,
) -> str:
    """Generate a structured prompt for the Points24 task.

    Args:
        cards: Current available cards.
        formula: Current accumulated formula string.
        action_only: If True, omit the "thoughts" field from the expected response.
    """
    cards_str = ", ".join(str(c) for c in cards)
    actions_str = ", ".join(f'"{a}"' for a in ACTION_LIST)

    if action_only:
        response_format = json.dumps({"action": "<your chosen action>"}, indent=2)
    else:
        response_format = json.dumps(
            {
                "cards": "<list of available cards>",
                "formula": "<current formula>",
                "thoughts": "<your reasoning>",
                "action": "<your chosen action>",
            },
            indent=2,
        )

    prompt = (
        f"You have the following cards: [{cards_str}].\n"
        f"Current formula: {formula}\n"
        f"Your goal is to construct an arithmetic expression using these cards that equals 24.\n\n"
        f"Available actions: [{actions_str}]\n\n"
        f"Select the next action to build towards reaching 24.\n"
        f"Respond in valid JSON format:\n{response_format}"
    )
    return prompt


def text_projection_points24(response_text: str) -> tuple[int, bool]:
    """Extract the action index from a model response.

    Returns:
        (action_index, is_legal): Index into ACTION_LIST and whether the action was valid.
    """
    try:
        match = re.search(r'"action"\s*:\s*"([^"]*)"', response_text)
        if match:
            action_str = match.group(1).strip()
            if action_str == "10" and "10" in ACTION_LIST:
                return ACTION_LIST.index("10"), True
            for i, a in enumerate(ACTION_LIST):
                if a == action_str:
                    return i, True
    except Exception:
        pass

    return random.randint(0, len(ACTION_LIST) - 1), False


def parse_single_action(response_text: str) -> tuple[str, bool]:
    """Parse a single action string from model response.

    Returns:
        (action_string, is_legal)
    """
    idx, legal = text_projection_points24(response_text)
    return ACTION_LIST[idx], legal
