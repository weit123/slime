# Prompt templates for Android World environment interaction.
# Ported from verl-agent (Apache-2.0 licensed).

# System prompt with comprehensive tool definitions (adapted from Qwen agent)
ANDROID_WORLD_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call one function per step to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `open`: Open an app on the device.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate", "answer"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type`, `action=answer`, and `action=open`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

ANDROID_WORLD_TEMPLATE_NO_HIS = """
The user query: {task_description}

Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.

After answering, summarize your observation and action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.

<image>
"""

ANDROID_WORLD_TEMPLATE = """
The user query: {task_description}

Task progress (You have done the following {step_count} operations out of {max_steps} on the current device):
{action_history}

Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.

After answering, summarize your observation and action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.

Step {current_step}: <image>
"""

# Lightweight per-step observation for slime's incremental token-append architecture.
# The task description and previous actions are already in the token sequence from
# the initial prompt and prior model outputs, so we only need the step counter and
# the new screenshot.
ANDROID_WORLD_TEMPLATE_STEP = """
Step {current_step} of {max_steps}: <image>
"""
