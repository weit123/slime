# Android World Rollout — Prompt Format and Context Evolution Analysis

This document provides a detailed analysis of how the model context is constructed in `examples/android_world/rollout.py`, and how the context grows incrementally across multi-turn interactions.

---

## 1. Prompt Template Overview

All templates are defined in `examples/android_world/prompts.py`.

### 1.1 System Prompt (`ANDROID_WORLD_SYSTEM_PROMPT`)

Defines the `mobile_use` tool in Qwen agent tool-call style:

```
You are a helpful assistant.

# Tools

You may call one function per step to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

Supported action types: `click`, `long_press`, `swipe`, `type`, `answer`, `system_button`, `open`, `wait`, `terminate`.

The screen coordinate space is **999×999** (coordinates output by the model). At inference time, `rescale_fn` in `env_worker.py` maps these to the actual screen size (default 1080×2400).

### 1.2 First-Turn User Message (`ANDROID_WORLD_TEMPLATE_NO_HIS`)

Used only for the first turn (`is_initial=True`):

```
The user query: {task_description}

Before answering, explain your reasoning step-by-step in <thinking></thinking> tags,
and insert them before the <tool_call></tool_call> XML tags.

After answering, summarize your observation and action in <conclusion></conclusion> tags,
and insert them after the <tool_call></tool_call> XML tags.

<image>
```

- `{task_description}` is retrieved from `observation["task"]` (the task description returned by the environment reset)
- `<image>` is the Qwen VL placeholder corresponding to the current screenshot
- The first turn does **not** include any action history

### 1.3 Subsequent-Turn User Message (`ANDROID_WORLD_TEMPLATE_STEP`)

Used from the second turn onward (`is_initial=False`), extremely lightweight:

```
Step {current_step} of {max_steps}: <image>
```

- Contains only a step counter and the new screenshot
- Does **not repeat** the task description or action history — this information already exists in the preceding token sequence (leveraging KV cache)

### 1.4 Unused Full Template (`ANDROID_WORLD_TEMPLATE`)

```
The user query: {task_description}

Task progress (You have done the following {step_count} operations ...):
{action_history}

Before answering, ...

Step {current_step}: <image>
```

This template includes the full task description and action history, but is **not used** in slime's incremental token-appending architecture. It may be intended for non-incremental approaches or debugging.

---

## 2. Expected Model Output Format

The model should output the following structure each turn:

```xml
<thinking>
I see a Contacts app icon on the screen. I need to click it to open the contacts list...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}
</tool_call>
<conclusion>
Clicked the Contacts app icon to open the contacts application.
</conclusion>
```

The parsing logic is in `parse_ui_action_from_response` in `env_worker.py`, which uses the regex `<tool_call>(.*?)</tool_call>` to extract the JSON.

### Action Parameter Examples

| action | Required parameters | Example |
|--------|---------------------|---------|
| `click` | `coordinate: [x, y]` | `{"action": "click", "coordinate": [500, 300]}` |
| `long_press` | `coordinate: [x, y]`, `time` | `{"action": "long_press", "coordinate": [500, 300], "time": 2}` |
| `swipe` | `coordinate`, `coordinate2` | `{"action": "swipe", "coordinate": [500, 800], "coordinate2": [500, 200]}` |
| `type` | `text` | `{"action": "type", "text": "Hello"}` |
| `open` | `text` | `{"action": "open", "text": "Chrome"}` |
| `system_button` | `button` | `{"action": "system_button", "button": "Back"}` |
| `wait` | `time` (optional) | `{"action": "wait", "time": 2}` |
| `terminate` | `status` | `{"action": "terminate", "status": "success"}` |
| `answer` | `text` | `{"action": "answer", "text": "42"}` |

---

## 3. Initial Prompt Construction

Handled by `_build_initial_prompt()` in `rollout.py`.

### 3.1 Message Assembly

```python
messages = [
    {"role": "system", "content": ANDROID_WORLD_SYSTEM_PROMPT},
    first_user_message,  # return value of format_observation(obs, is_initial=True)
]
```

The structure of `first_user_message` (returned by `format_observation`):

```python
{
    "role": "user",
    "content": [
        {"type": "image", "image": <PIL.Image>},   # screenshot
        {"type": "text",  "text": "The user query: ...\\n\\n...<image>"}
    ]
}
```

### 3.2 Chat Template Application

```python
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, **apply_kwargs
)
```

For Qwen VL models, this produces text roughly like the following (simplified):

```
<|im_start|>system
You are a helpful assistant.
# Tools
...
<|im_end|>
<|im_start|>user
The user query: Add a new contact named "Alice" with phone 123456.
...
<|vision_start|><|image_pad|>...<|vision_end|>
<|im_end|>
<|im_start|>assistant
```

### 3.3 Processor Encoding

For VLMs, `qwen_vl_utils.process_vision_info` extracts images, then the processor encodes them to produce:
- `prompt_ids`: token ID sequence
- `multimodal_train_inputs`: tensors such as `pixel_values` (used for image feature computation during training)
- `image_data`: base64-encoded images (sent to the SGLang inference engine)

---

## 4. Context Evolution Across Turns

Below is the complete token sequence evolution for a 3-turn interaction.

### Turn 0: Initial Prompt (environment input only, loss_mask=0)

```
┌─────────────────────────────────────────────────────────────┐
│  sample.tokens:                                             │
│  [ system_prompt_tokens | user_msg_tokens(task+screenshot) ]│
│                                                             │
│  sample.loss_mask:                                          │
│  (not in response_tokens, not included in loss_mask)        │
│                                                             │
│  image_data: [screenshot_0_base64]                          │
└─────────────────────────────────────────────────────────────┘
```

Payload sent to SGLang:
```json
{
  "input_ids": [<prompt_tokens>],
  "image_data": ["<screenshot_0_base64>"],
  "sampling_params": {"max_new_tokens": <budget>, ...},
  "return_logprob": true
}
```

### Turn 1: Model Generation → Environment Execution

**SGLang returns** (model's first response):

```
<thinking>I see the home screen. I need to open the Contacts app.</thinking>
<tool_call>{"name": "mobile_use", "arguments": {"action": "open", "text": "Contacts"}}</tool_call>
<conclusion>Opening the Contacts app.</conclusion>
```

**Append model tokens** (loss_mask=**1**):

```
sample.tokens:  [...prompt, response_1_tokens]
sample.loss_mask:       [..., 1, 1, 1, ..., 1]
                              ^^^^^^^^^^^^^^^^
                              model-generated tokens, included in training loss
```

**Environment executes** `env.step(response_text)` → returns new observation (new screenshot)

**Encode new observation** (loss_mask=**0**):

`_encode_observation_for_generation` applies the chat template to the new user message, then uses the DUMMY_MESSAGES trim trick to remove the prefix:

```
formatted = chat_template([dummy_sys, dummy_user, new_obs_message])
trim_len  = len(tokenize(chat_template([dummy_sys, dummy_user])))
obs_tokens = tokenize(formatted)[trim_len:]   # keep only the new_obs_message portion
```

If the first token of `obs_tokens` is a BOS token, it is also stripped.

The resulting `obs_tokens` roughly correspond to:

```
<|im_start|>user
Step 2 of 10: <|vision_start|><|image_pad|>...<|vision_end|>
<|im_end|>
<|im_start|>assistant
```

After appending:

```
sample.tokens:  [...prompt, response_1, obs_2_tokens]
sample.loss_mask:       [..., 1...1,     0...0      ]
                                         ^^^^^^^^^^^^
                                         environment observation tokens, excluded from loss

image_data:     [screenshot_0, screenshot_1]  ← all images accumulated
```

### Turn 2: Model Generates Again

**SGLang payload**:
```json
{
  "input_ids": [<prompt + response_1 + obs_2>],
  "image_data": ["<screenshot_0>", "<screenshot_1>"],
  "sampling_params": {"max_new_tokens": <remaining_budget>},
  "return_logprob": true
}
```

**SGLang returns** → append response_2 (loss_mask=1) → environment executes → append obs_3 (loss_mask=0) → ...

### Final Token Sequence Structure

```
sample.tokens:
┌──────────┬────────────┬────────┬────────────┬────────┬────────────┐
│  prompt  │ response_1 │ obs_2  │ response_2 │ obs_3  │ response_3 │
│(sys+user)│ (model)    │ (env)  │ (model)    │ (env)  │ (model)    │
└──────────┴────────────┴────────┴────────────┴────────┴────────────┘

sample.loss_mask:
┌──────────┬────────────┬────────┬────────────┬────────┬────────────┐
│ (no mask)│  1 1 1 ... │ 0 0 0 │  1 1 1 ... │ 0 0 0 │  1 1 1 ... │
└──────────┴────────────┴────────┴────────────┴────────┴────────────┘
                ↑ train      ↑ ignore    ↑ train     ↑ ignore  ↑ train

image_data (full history sent on every inference request):
[screenshot_0, screenshot_1, screenshot_2, ...]
```

---

## 5. Token Budget Management

Context length is controlled by two parameters:

```python
max_ctx = args.max_context_len or args.rollout_max_context_len
budget = max_ctx - len(initial_prompt_tokens)
# or
budget = sampling_params["max_new_tokens"]
```

After each token append (whether model-generated or environment observation):

```python
budget = budget - len(new_tokens)
```

When `budget <= 0`, `sample.status = TRUNCATED` and the loop terminates.

Other termination conditions:
- SGLang returns `finish_reason.type == "length"` → `TRUNCATED`
- SGLang returns `finish_reason.type == "abort"` → `ABORTED`
- Environment returns `done=True` → `COMPLETED`
- `max_turns` reached → `COMPLETED`
- Model outputs `terminate` action → environment returns `done=True`

---

## 6. Multimodal Data Management

### 6.1 Inference Side (SGLang)

The `current_image_data` list appends the base64-encoded screenshot on each turn. **Every inference request sends the full image sequence**:

```python
# Turn 0: [img_0]
# Turn 1: [img_0, img_1]
# Turn 2: [img_0, img_1, img_2]
# ...
```

The SGLang engine aligns images with tokens using the positions of `<|image_pad|>` in the token sequence.

### 6.2 Training Side

`multimodal_train_inputs_buffer` is a `list[dict | None]`. Each turn (if images are present) appends a dict containing processor output tensors such as `pixel_values` and `image_grid_thw`.

At the end, `_finalize_sample` calls `_merge_multimodal_train_inputs` to concatenate all turns' tensors along dim=0:

```python
# pixel_values: [turn_0_pixels; turn_1_pixels; turn_2_pixels; ...]  (torch.cat dim=0)
```

### 6.3 `sample.multimodal_inputs`

Accumulates raw PIL Image objects:

```python
sample.multimodal_inputs = {"images": [img_0, img_1, img_2, ...]}
```

---

## 7. Complete Interaction Example

Assume the task is "Add a new contact named Alice with phone 123456", with `max_turns=10`.

### First Prompt After Reset

```
<|im_start|>system
You are a helpful assistant.

# Tools
...(mobile_use tool definition)...
<|im_end|>
<|im_start|>user

The user query: Add a new contact named "Alice" with phone number 123456.

Before answering, explain your reasoning step-by-step in <thinking></thinking> tags,
and insert them before the <tool_call></tool_call> XML tags.

After answering, summarize your observation and action in <conclusion></conclusion> tags,
and insert them after the <tool_call></tool_call> XML tags.

<|vision_start|><|image_pad|>...<|vision_end|>
<|im_end|>
<|im_start|>assistant
```

### Model Turn 1 Output

```
<thinking>
I see the Android home screen with several app icons. I need to open the Contacts app
to add a new contact. I can see the Contacts icon on the screen.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "open", "text": "Contacts"}}
</tool_call>
<conclusion>
Opening the Contacts app to begin adding a new contact.
</conclusion>
```

### Environment Returns → Turn 2 Observation (appended to token sequence)

```
<|im_start|>user

Step 2 of 10: <|vision_start|><|image_pad|>...<|vision_end|>
<|im_end|>
<|im_start|>assistant
```

### Model Turn 2 Output

```
<thinking>
The Contacts app is now open showing the contact list. I need to click the "+"
button to add a new contact. I can see it at the bottom right corner.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [900, 900]}}
</tool_call>
<conclusion>
Clicked the "+" (add contact) button in the Contacts app.
</conclusion>
```

### ...(similar turns continue)...

### Model Final Turn Output (termination)

```
<thinking>
I have successfully filled in the contact name "Alice" and phone number "123456",
and saved the contact. The task should be complete.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>
<conclusion>
Terminated the task with success status after adding the contact.
</conclusion>
```

---

## 8. Key Design Decision Summary

| Design Point | Choice | Rationale |
|--------------|--------|-----------|
| Subsequent-turn template | Only "Step N of M: \<image\>" | Leverages KV cache, avoids repeating verbose task description and history |
| Image sending strategy | Send full image history on every inference request | SGLang needs the complete image sequence to align with image placeholders in the token stream |
| loss_mask | Model output=1, environment obs=0 | Only train on model decisions (actions), not on environment feedback |
| Initial prompt | Dynamically constructed (not from JSONL) | Screenshot and task description are only known after environment reset |
| DUMMY_MESSAGES trim | Subtract dummy prefix length | Prevents chat template from re-adding system/user prefixes |
| Coordinate space | 999×999 normalized | Unifies across different device resolutions; env_worker maps to actual screen |
