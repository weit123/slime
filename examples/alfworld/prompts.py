"""ALFWorld prompt templates - MUST match GTR-Turbo exactly.

Reference: GTR-Turbo/Turbo_ALF/a2c_ppo_acktr/rl_utils.py

This module provides prompt generation for the ALFWorld environment.
The text format is kept identical to GTR-Turbo to ensure compatibility
with trained models.

Key differences from Points24:
- Prompts include task description from environment
- Prompts include action history
- Prompts include admissible actions list
- No image embedding in prompt text (image provided separately)
"""

from __future__ import annotations

from typing import List


def get_alfworld_prompt(
    task_description: str,
    action_history: List[str],
    admissible_actions: List[str],
    action_only: bool = False,
) -> str:
    """Generate prompt for ALFWorld environment.

    This function MUST match GTR-Turbo/Turbo_ALF/a2c_ppo_acktr/rl_utils.py exactly.

    Args:
        task_description: Task description from get_templated_task_desc()
        action_history: List of previous actions taken
        admissible_actions: List of valid actions for current state
        action_only: If True, omit the "thoughts" field in the prompt

    Returns:
        Prompt string in the exact format used by GTR-Turbo
    """
    # Format admissible actions as newline-separated quoted strings
    reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions)

    # Build prompt - MUST match GTR-Turbo format exactly
    qs = "Your are an expert in the ALFRED Embodied Environment."
    qs = qs + f"Your task is to " + task_description + ". "
    qs = qs + f"You are also given the previous actions you have taken: {action_history}. "
    qs = qs + f"Your admissible actions of the current situation are: [{reformatted_admissible_actions}]. "

    if not action_only:
        qs = qs + "Your response should be a valid json file in the following format: \n{\n"
        qs = qs + '"thoughts": "{first describe what do you see in the image using the text description, then carefully think about which action to complete the task based on your observation, action history and admissible actions. }", \n'
        qs = qs + '"action": "{an admissible action}"\n}'
    else:
        qs = qs + "Your response should be a valid json file in the following format: \n{\n"
        qs = qs + '"action": "{an admissible action}"\n}'

    return qs


def get_alfworld_prompt_from_env(
    env,
    action_history: List[str],
    admissible_actions: List[str],
    action_only: bool = False,
) -> tuple[str, str]:
    """Generate prompt using environment's task description.

    Convenience wrapper that extracts task description from the environment.

    Args:
        env: ALFWorld environment instance (AlfredThorEnv or wrapper)
        action_history: List of previous actions taken
        admissible_actions: List of valid actions for current state
        action_only: If True, omit the "thoughts" field

    Returns:
        Tuple of (prompt_string, task_description)
    """
    try:
        from alfworld.agents.utils.misc import get_templated_task_desc

        # Extract task description from environment
        task = get_templated_task_desc(env.env.envs[0].traj_data)
    except Exception:
        # Fallback if extraction fails
        task = "complete the current task"

    prompt = get_alfworld_prompt(task, action_history, admissible_actions, action_only)
    return prompt, task


# Action list for ALFWorld (must match GTR-Turbo)
ALF_ACTION_LIST = [
    "pass", "goto", "pick", "put", "open", "close",
    "toggle", "heat", "clean", "cool", "slice",
    "inventory", "examine", "look",
]
