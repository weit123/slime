"""Verification tests for ALFWorld environment integration.

Prerequisites:
    - ALFWorld installed (pip install alfworld)
    - AI2-THOR installed
    - VLM model downloaded
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


def test_imports():
    """Test that all modules can be imported."""
    from examples.alfworld.prompts import get_alfworld_prompt, ALF_ACTION_LIST
    from examples.alfworld.alf_utils import AlfEnv, process_action, compute_reward

    assert callable(get_alfworld_prompt)
    assert len(ALF_ACTION_LIST) == 14
    assert callable(process_action)
    assert callable(compute_reward)


def test_prompt_generation():
    """Test ALFWorld prompt generation matches GTR-Turbo format."""
    from examples.alfworld.prompts import get_alfworld_prompt

    prompt = get_alfworld_prompt(
        task_description="pick up the apple",
        action_history=["go to kitchen", "open fridge"],
        admissible_actions=["pick apple", "close fridge", "look"],
        action_only=False,
    )

    assert "ALFRED Embodied Environment" in prompt
    assert "pick up the apple" in prompt
    assert "pick apple" in prompt
    assert '"thoughts"' in prompt
    assert '"action"' in prompt

    # Action-only mode
    prompt_action_only = get_alfworld_prompt(
        task_description="pick up the apple",
        action_history=["go to kitchen"],
        admissible_actions=["pick apple"],
        action_only=True,
    )
    assert '"thoughts"' not in prompt_action_only
    assert '"action"' in prompt_action_only


def test_action_processing():
    """Test action parsing from model output."""
    from examples.alfworld.alf_utils import process_action

    admissible = ["pick apple 1", "go to countertop", "look", "open fridge"]

    # Valid action
    action, legal = process_action('{"action": "pick apple 1"}', admissible)
    assert legal == True
    assert action == "pick apple 1"

    # Action with extra text
    action, legal = process_action('Some text {"action": "look"} more text', admissible)
    assert legal == True
    assert action == "look"

    # Invalid action - should return random from admissible
    action, legal = process_action('{"action": "invalid"}', admissible)
    assert legal == False
    assert action in admissible


def test_reward_computation():
    """Test reward calculation."""
    from examples.alfworld.alf_utils import compute_reward

    # Success
    infos = {'won': [1.0], 'goal_condition_success_rate': [1.0]}
    reward = compute_reward(infos, legal_action=True)
    assert reward == 51.0  # 50*1 + 1

    # Partial success
    infos = {'won': [0.0], 'goal_condition_success_rate': [0.5]}
    reward = compute_reward(infos, legal_action=True)
    assert reward == 0.5

    # Illegal action penalty
    infos = {'won': [0.0], 'goal_condition_success_rate': [0.0]}
    reward = compute_reward(infos, legal_action=False)
    assert reward == -1.0


if __name__ == "__main__":
    test_imports()
    test_prompt_generation()
    test_action_processing()
    test_reward_computation()
    print("All tests passed!")
