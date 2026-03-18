"""Points24 prompt templates - MUST match GTR-Turbo exactly.

Reference: GTR-Turbo/Turbo_P24/a2c_ppo_acktr/rl_utils.py

This module provides prompt generation and action parsing functions for the
24 Game environment. The text format is kept identical to GTR-Turbo to ensure
compatibility with trained models.
"""

from __future__ import annotations

import random
from typing import List

import torch


# Action list for Points24 environment (must match GTR-Turbo)
POINTS24_ACTION_LIST = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "+", "-", "*", "/", "(", ")", "="
]


def get_points24_prompt(formula: list, action_only: bool = False) -> str:
    """Generate prompt for Points24 environment.

    This function MUST match GTR-Turbo/Turbo_P24/a2c_ppo_acktr/rl_utils.py exactly.

    Args:
        formula: List of formula elements (e.g., ['8', '*', '3'])
        action_only: If True, omit the "thoughts" field in the prompt

    Returns:
        Prompt string in the exact format used by GTR-Turbo
    """
    # Convert formula list to string
    text_formula = ''.join(str(element) for element in formula)

    # Build prompt - MUST match GTR-Turbo format exactly
    qs = "You are an expert 24 points card game player. You are observing four cards in the image. "
    qs = qs + f"You are observing the current formula: {text_formula}. "
    qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. "
    qs = qs + "The number or operator you choose will be appended to the current formula. "
    qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
    qs = qs + "Your goal is to output a formula that evaluates to 24, and each number can only be used once. "
    qs = qs + "Return exactly one JSON object and nothing else. Do not use markdown fences, lists, or extra explanation. "
    qs = qs + 'Your response should **only contain a valid json file** in the following format:\n{\n'
    if not action_only:
        qs = qs + '  "cards": ["x", "y", "z", "w"],\n'
        qs = qs + f'  "formula": "{text_formula}",\n'
        qs = qs + '  "thoughts": "First check whether the current formula equals 24. If so, output \'=\'. Otherwise consider which number or operator should be appended to make it equal 24.",\n'
    qs = qs + '  "action": "number or operator"\n}'

    return qs


def text_projection_points24(text_actions: List[str]) -> tuple[torch.Tensor, int]:
    """Project text actions to discrete action indices.

    This function MUST match GTR-Turbo implementation exactly.
    It parses the model's text output and extracts the action.

    Args:
        text_actions: List of text responses from the model

    Returns:
        Tuple of (action_indices_tensor, legal_action_flag)
        - action_indices_tensor: Tensor of shape (batch_size, 1) with action indices
        - legal_action_flag: 1 if action was legal, 0 otherwise
    """
    output_indices = []

    for string in text_actions:
        if not isinstance(string, str):
            # Directly output a random action if the string is not a string
            output_indices.append(random.randint(0, len(POINTS24_ACTION_LIST) - 1))
            legal_action = 0
            continue

        string = string.lower()
        action_index = string.find('"action":')

        if action_index == -1:
            output_indices.append(random.randint(0, len(POINTS24_ACTION_LIST) - 1))
            legal_action = 0
            continue

        string = string[action_index:]
        contained_actions = []

        # Handle '10' separately to prevent it from being counted as '1'
        if '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')  # Remove '10' to prevent counting as '1'

        # Find all actions that are contained in the string
        for action in POINTS24_ACTION_LIST:
            if action in string:
                contained_actions.append(action)

        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))

        if len(contained_actions) == 1 and contained_actions[0] in POINTS24_ACTION_LIST:
            # Only one keyword from action_list is in the string
            output_indices.append(POINTS24_ACTION_LIST.index(contained_actions[0]))
            legal_action = 1
        else:
            # The string contains none or multiple keywords, randomly select
            output_indices.append(random.randint(0, len(POINTS24_ACTION_LIST) - 1))
            legal_action = 0

    return torch.Tensor([output_indices]).long().reshape(-1, 1), legal_action


def parse_single_action(response_text: str) -> tuple[int, bool]:
    """Parse a single model response to get the action index.

    Simplified wrapper for single-sample inference.

    Args:
        response_text: Model's text response

    Returns:
        Tuple of (action_index, is_legal)
    """
    indices, legal = text_projection_points24([response_text])
    return int(indices[0].item()), bool(legal)
