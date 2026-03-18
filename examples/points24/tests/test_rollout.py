"""Verification tests for Points24 environment integration.

Run from the slime repo root:
    pytest examples/points24/tests/test_rollout.py -v

Prerequisites:
    - VLM model downloaded (e.g. Qwen3-VL-2B-Instruct)
    - GPU available for SGLang
    - SGLang server running (optional for unit tests)
"""

from __future__ import annotations

import os
import sys

# Ensure slime is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


def test_imports():
    """Test that all modules can be imported."""
    from examples.points24.prompts import get_points24_prompt, text_projection_points24
    from examples.points24.env_worker import Points24Worker
    from examples.points24.env_points24 import Points24Env

    assert callable(get_points24_prompt)
    assert callable(text_projection_points24)
    assert callable(Points24Worker)
    assert callable(Points24Env)


def test_prompt_generation():
    """Test prompt generation matches GTR-Turbo format."""
    from examples.points24.prompts import get_points24_prompt

    # Empty formula
    prompt = get_points24_prompt([], action_only=False)
    assert "24 points card game" in prompt
    assert "formula: " in prompt
    assert '"cards":' in prompt
    assert '"thoughts":' in prompt
    assert '"action":' in prompt

    # With formula
    prompt = get_points24_prompt(["8", "*", "3"], action_only=False)
    assert "8*3" in prompt

    # Action-only mode
    prompt = get_points24_prompt([], action_only=True)
    assert '"thoughts":' not in prompt
    assert '"action":' in prompt


def test_text_projection():
    """Test action parsing from model output."""
    from examples.points24.prompts import text_projection_points24
    import torch

    # Valid action
    indices, legal = text_projection_points24(['{"action": "8"}'])
    assert legal == 1
    assert indices[0].item() == 7  # "8" is index 7

    # Operator
    indices, legal = text_projection_points24(['{"action": "+"}'])
    assert legal == 1
    assert indices[0].item() == 10  # "+" is index 10

    # Invalid action
    indices, legal = text_projection_points24(['invalid'])
    assert legal == 0

    # Handle '10' correctly
    indices, legal = text_projection_points24(['{"action": "10"}'])
    assert legal == 1
    assert indices[0].item() == 9  # "10" is index 9


def test_worker_reset():
    """Test Points24Worker reset functionality."""
    from examples.points24.env_worker import Points24Worker

    worker = Points24Worker(max_steps=30)
    obs, info = worker.reset(seed=42)

    assert "image" in obs
    assert "cards" in obs
    assert "formula" in obs
    assert "numbers" in obs
    assert len(obs["cards"]) == 4
    assert obs["formula"] == []
    assert worker.steps == 0


def test_worker_step():
    """Test Points24Worker step functionality."""
    from examples.points24.env_worker import Points24Worker

    worker = Points24Worker(max_steps=30)
    obs, info = worker.reset(seed=42)

    # Step with a number action
    obs, reward, done, info = worker.step('{"action": "8"}')
    assert obs is not None
    assert reward == 0  # No reward until done
    assert not done
    assert "8" in obs["formula"]

    worker.close()


def test_worker_full_episode():
    """Test a complete episode."""
    from examples.points24.env_worker import Points24Worker

    worker = Points24Worker(max_steps=30)
    obs, info = worker.reset(seed=42)

    # Simulate a game (exact solution depends on cards)
    actions = ['{"action": "8"}', '{"action": "*"}', '{"action": "3"}', '{"action": "="}']

    for action in actions:
        obs, reward, done, info = worker.step(action)
        if done:
            break

    worker.close()


def test_env_format_observation():
    """Test Points24Env format_observation method."""
    from examples.points24.env_points24 import Points24Env
    from examples.points24.env_worker import Points24Worker

    worker = Points24Worker(max_steps=30)
    env = Points24Env(worker=worker, max_turns=30)

    obs, info = worker.reset(seed=42)
    message = env.format_observation(obs, is_initial=True)

    assert message["role"] == "user"
    assert isinstance(message["content"], list)
    assert len(message["content"]) == 2  # image + text
    assert message["content"][0]["type"] == "image"
    assert message["content"][1]["type"] == "text"

    env.close()


if __name__ == "__main__":
    test_imports()
    test_prompt_generation()
    test_text_projection()
    test_worker_reset()
    test_worker_step()
    test_worker_full_episode()
    test_env_format_observation()
    print("All tests passed!")
