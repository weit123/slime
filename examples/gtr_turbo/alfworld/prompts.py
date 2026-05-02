"""ALFWorld prompt templates.

Generates prompts for the ALFWorld embodied AI environment following
the GTR-Turbo inline prompt format (Turbo_ALF/a2c_ppo_acktr/rl_utils.py).
"""

from __future__ import annotations

ALF_ACTION_LIST = [
    "pass", "goto", "pick", "put", "open", "close", "toggle",
    "heat", "clean", "cool", "slice", "inventory", "examine", "look",
]


def get_alfworld_prompt(
    task_description: str,
    action_history: list[str],
    admissible_actions: list[str],
    action_only: bool = False,
) -> str:
    """Generate a prompt for ALFWorld tasks matching the GTR-Turbo reference format."""
    reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions)

    qs = "Your are an expert in the ALFRED Embodied Environment."
    qs += f"Your task is to {task_description}. "
    qs += f"You are also given the previous actions you have taken: {action_history}. "
    qs += f"Your admissible actions of the current situation are: [{reformatted_admissible_actions}]. "

    if not action_only:
        qs += "Your response should be a valid json file in the following format: \n{\n"
        qs += '"thoughts": "{first describe what do you see in the image using the text description, '
        qs += "then carefully think about which action to complete the task based on your observation, "
        qs += 'action history and admissible actions. }", \n'
        qs += '"action": "{an admissible action}"\n}'
    else:
        qs += "Your response should be a valid json file in the following format: \n{\n"
        qs += '"action": "{an admissible action}"\n}'

    return qs


def get_alfworld_prompt_from_env(env, action_only: bool = False) -> tuple[str, str]:
    """Convenience wrapper that extracts task info from an AlfEnv instance.

    Returns:
        (prompt, task_description)
    """
    prompt = get_alfworld_prompt(
        task_description=env.task_description,
        action_history=env.action_history,
        admissible_actions=env.admissible_commands,
        action_only=action_only,
    )
    return prompt, env.task_description


# Keep backward compat alias
get_prompt = get_alfworld_prompt
