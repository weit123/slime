# Android World VLM Agent RL Training

Train vision-language model (VLM) agents to complete tasks on Android devices using reinforcement learning. The agent interacts with Android emulators through screenshots and tool-call actions, and is trained with GRPO on task success rewards.

Two rollout modes are provided:

| Mode | Module | `--custom-generate-function-path` | `--custom-reward-post-process-path` | `--dynamic-sampling-filter-path` | Launch script |
|------|--------|-----------------------------------|-------------------------------------|----------------------------------|---------------|
| **Incremental** (default) | `rollout.py` | `examples.android_world.rollout.generate` | _(none, default works)_ | _(none)_ | `run_android_world_opt.sh` |
| **History-based** | `rollout_history.py` | `examples.android_world.rollout_history.generate` | `examples.android_world.rollout_history.post_process_rewards_history` | `examples.android_world.rollout_history.check_reward_nonzero_std_history` | `run_android_world_history.sh` |

See [Rollout Modes](#rollout-modes) for a detailed comparison.

## Architecture

```
                      slime Training Loop
                ┌─────────────────────────────┐
                │   Data Buffer (prompts)      │
                │        ↓                     │
                │   Rollout (SGLang + Pool)     │
                │     ┌────────────────────┐   │
                │     │  generate() loop   │   │
                │     │  per sample:       │   │
                │     │    SGLang ──► VLM   │   │
                │     │      ↕             │   │
                │     │    EnvPool ──► AVD  │   │
                │     └────────────────────┘   │
                │        ↓                     │
                │   Training (Megatron/FSDP)    │
                │        ↓                     │
                │   Weight Sync → SGLang       │
                └─────────────────────────────┘
```

Each rollout sample runs an independent async multi-turn loop:

1. Acquire an Android emulator worker from the pool
2. Reset the environment with a task (e.g., "Add a contact named John")
3. Build the initial VLM prompt (system prompt + screenshot + task text)
4. Loop: SGLang generates an action → environment executes it → screenshot returned → repeat
5. Episode ends when the agent calls `terminate` or hits `max_turns`
6. Reward = task success (0 or 1) + step shaping, only evaluated on explicit termination; timeout = 0.0
7. Release the worker back to the pool

## Prerequisites

- **Android emulators**: A base AVD (e.g., `AndroidWorldAvd`) must be configured. Workers clone from it automatically.
- **Android World SDK**: The `android_world` package must be on `PYTHONPATH`.
- **VLM model**: A Qwen3-VL model (2B/4B/8B, Instruct or Thinking variant).
- **slime**: Installed via `pip install -e . --no-deps` from the repo root.

## Quick Start

### Shell script (recommended)

The shell script handles process cleanup, Ray startup, model download, and job submission automatically:

```bash
# FSDP backend (default), 8 GPUs
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct \
SLIME_SCRIPT_NUM_GPUS=8 \
  bash examples/android_world/run_android_world.sh
```

```bash
# Megatron backend
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct \
SLIME_SCRIPT_NUM_GPUS=8 \
SLIME_SCRIPT_TRAIN_BACKEND=megatron \
  bash examples/android_world/run_android_world.sh
```

### Multi-node training

For multi-node setups, start the Ray cluster first with `multi_node_ray_start.sh`, then launch training with external Ray:

```bash
# 1. Start Ray across all nodes (head + workers)
bash examples/android_world/multi_node_ray_start.sh

# 2. Launch training with external Ray (e.g., 2 nodes x 8 GPUs)
SLIME_SCRIPT_EXTERNAL_RAY=1 \
SLIME_SCRIPT_NUM_NODES=2 \
SLIME_SCRIPT_NUM_GPUS=8 \
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct \
SLIME_SCRIPT_TRAIN_BACKEND=megatron \
  bash examples/android_world/run_android_world.sh
```

### Python entry point

If you prefer to manage Ray and model downloads yourself:

```bash
# 1. Download the model
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /root/models/Qwen3-VL-8B-Instruct

# 2. Start Ray cluster
ray start --head --num-gpus 8

# 3. Run training
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct \
SLIME_SCRIPT_NUM_GPUS=8 \
  python examples/android_world/run_android_world.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLIME_SCRIPT_MODEL_NAME` | `Qwen3-VL-2B-Instruct` | VLM model name (Qwen3-VL-{2B,4B,8B}-{Instruct,Thinking}) |
| `SLIME_SCRIPT_NUM_NODES` | `1` | Number of nodes for training |
| `SLIME_SCRIPT_NUM_GPUS` | `8` | Number of GPUs per node |
| `SLIME_SCRIPT_TRAIN_BACKEND` | `fsdp` | Training backend: `fsdp` or `megatron` |
| `SLIME_SCRIPT_TASK_DATA` | `examples/android_world/data/tasks.jsonl` | Path to task dataset |
| `SLIME_SCRIPT_EXTERNAL_RAY` | `0` | Set to `1` if Ray is already running |
| `WANDB_API_KEY` | _(none)_ | Enables W&B logging if set |

## Configuration

### `config.yaml` — Environment Settings

Edit `examples/android_world/config.yaml` to adjust:

```yaml
# Multi-turn interaction
max_turns: 5               # Max agent-environment interaction steps per episode
max_context_len: 16384     # Total token budget (prompt + all turns)

# Environment pool
num_workers: 64            # Number of persistent Android emulator workers
resources_per_worker:
  num_cpus: 4
  memory: 8589934592       # 8 GB per worker

# Android emulator paths
avd_name: "AndroidWorldAvd"               # Base AVD to clone from
base_avd_name_pattern: "slime_aw_{}"      # Worker AVD naming pattern
base_console_port: 5556
base_grpc_port: 8554
base_adb_server_port: 5037
android_avd_home: "/root/android/avd/"
android_sdk_root: "/root/android/"
emulator_path: "/root/android/emulator/emulator"
adb_path: "/root/android/platform-tools/adb"
```

**Sizing `num_workers`**: Should be >= the number of concurrent in-flight samples. This is bounded by the SGLang concurrency semaphore (`sglang_server_concurrency * rollout_num_gpus / rollout_num_gpus_per_engine`). If the pool has fewer workers than concurrent samples, excess `generate()` calls will await until a worker is released.

### `data/tasks.jsonl` — Task Dataset

Each line defines one task for training:

```json
{"prompt": "Complete a task on Android", "label": "", "metadata": {"task_name": "ContactsAddContact", "params_idx": 0}}
```

- `prompt`: Placeholder text (overridden dynamically by `generate()` after `env.reset()`)
- `metadata.task_name`: Android World task identifier (must match `registry.TaskRegistry`)
- `metadata.params_idx`: Parameter variation index (0-19, each task has 20 pre-generated variations)

To regenerate the dataset (e.g., after editing `task_list.txt`):

```bash
python examples/android_world/data/generate_tasks.py
```

This reads the 116 task names from `data/task_list.txt`, generates all `task x params_idx` combinations (116 x 20 = 2320), shuffles them with a fixed seed for gradient diversity, and writes to `data/tasks.jsonl`.

With GRPO (`--n-samples-per-prompt 8`), slime creates 8 copies of each entry sharing the same `task_name`/`params_idx`. Each copy runs on its own emulator with different model sampling, producing diverse trajectories for variance reduction.

### Training Hyperparameters

Key args in `run_android_world.sh` / `run_android_world.py` (edit directly or override via slime CLI):

| Argument | Value | Description |
|----------|-------|-------------|
| `--num-rollout` | 16 | Number of rollout iterations |
| `--rollout-batch-size` | 8 | Unique prompts (groups) that survive filtering per rollout step |
| `--n-samples-per-prompt` | 8 | GRPO copies per prompt |
| `--over-sampling-batch-size` | 32 | Prompts requested per data source call (history-based mode; 2x rollout-batch-size for filter headroom) |
| `--dynamic-sampling-filter-path` | _(see history mode)_ | Drop groups with zero reward variance (history-based mode only) |
| `--rollout-max-response-len` | 4096 | Max new tokens per SGLang generation call |
| `--rollout-temperature` | 0.7 | Sampling temperature |
| `--global-batch-size` | 64 | Samples per optimizer step |
| `--lr` | 1e-6 | Learning rate |
| `--advantage-estimator` | grpo | RL algorithm |

**How these relate**: Training runs for `num_rollout` (16) iterations. In each iteration, `rollout_batch_size` (8) unique prompts are sampled and each is expanded into `n_samples_per_prompt` (8) independent copies, launching `8 × 8 = 64` concurrent `generate()` calls (each on its own emulator). The 64 samples produced equal `global_batch_size` (64), so each iteration performs exactly 1 optimizer step.

**History-based mode differences**: With `--dynamic-sampling-filter-path`, more than `rollout_batch_size` groups are attempted — `--over-sampling-batch-size 32` pre-submits 32 prompt groups per data source call, but only 16 surviving groups (those passing the filter) are used. Additionally, each trajectory produces T step-samples (not 1), so `--global-batch-size` is omitted from the history script (uses `--num-steps-per-rollout 1` with `--balance-data` instead). The total flattened sample count = `rollout_batch_size × n_samples_per_prompt × avg_trajectory_length`, trimmed to a multiple of `global_batch_size` if set.

`max_context_len` (config.yaml, 16384) sets a **total token budget** for each episode. At the start of a `generate()` call the remaining budget is computed as `max_context_len - len(prompt_tokens)`. Every token produced during the episode — both model output tokens and environment observation tokens — is deducted from this budget. Before each SGLang inference call, `max_new_tokens` is clamped to the remaining budget so the model cannot overshoot. If the budget reaches zero mid-episode, the sample is marked `TRUNCATED` and the loop exits early. This is distinct from `--rollout-max-response-len`, which caps a single SGLang generation call; `max_context_len` caps the cumulative length across all turns.

## File Structure

```
examples/android_world/
  __init__.py
  README.md                  # This file
  config.yaml                # Environment and emulator settings
  run_android_world.sh       # Shell launch script (cleanup + Ray + job submit)
  run_android_world_opt.sh   # Optimized launch script for H20 multi-node
  run_android_world_history.sh # Launch script for history-based rollout
  run_android_world.py       # Python entry script (uses execute_train)
  cleanup.sh                 # Kill emulators, ADB servers, remove cached AVDs
  prompts.py                 # System prompt + turn templates
  env_worker.py              # AndroidWorldWorker (single emulator wrapper)
  env_pool.py                # AndroidWorldEnvPool (async worker pool singleton)
  env_android_world.py       # AndroidWorldEnv (per-sample BaseInteractionEnv)
  rollout.py                 # generate() — incremental multi-turn rollout
  rollout_history.py         # generate() — non-incremental history-based rollout
  data/
    task_list.txt            # All 116 Android World task names
    generate_tasks.py        # Script to generate tasks.jsonl from task_list.txt
    tasks.jsonl              # Task dataset (116 tasks x 20 params = 2320 entries)
  tests/
    test_imports.py          # Import path verification
    test_env_pool.py         # Pool acquire/release + worker ops
    test_rollout.py          # End-to-end rollout with SGLang
    test_reward_post_process.py # Unit tests for GRPO reward normalization
```

### Module Roles

- **`rollout.py`** is the default entry point (incremental mode), specified via `--custom-generate-function-path examples.android_world.rollout.generate`. It orchestrates the multi-turn loop: pool acquisition, prompt construction, SGLang inference, env stepping, token/loss_mask management, and reward computation. Produces 1 training sample per trajectory.

- **`rollout_history.py`** is the non-incremental entry point (history-based mode), specified via `--custom-generate-function-path examples.android_world.rollout_history.generate`. Each step builds a standalone prompt with full action history (using `ANDROID_WORLD_TEMPLATE`), producing T independent training samples per trajectory. Also provides `post_process_rewards_history` (wired via `--custom-reward-post-process-path`) for correct GRPO normalization with variable-length trajectories, and `check_reward_nonzero_std_history` (wired via `--dynamic-sampling-filter-path`) to drop zero-variance groups. See [Rollout Modes](#rollout-modes) for details.

- **`env_pool.py`** manages a singleton pool of persistent `AndroidWorldWorker` Ray actors. Workers are created once (emulator boot is expensive) and reused across all training iterations. The `asyncio.Queue`-based acquire/release is safe for concurrent async generate calls.

- **`env_android_world.py`** implements slime's `BaseInteractionEnv` interface, wrapping a single remote worker with async `reset()`, `step()`, plus sync `format_observation()`, `get_reward()`, and `close()`. The async methods use `asyncio.to_thread(ray.get, ...)` so concurrent samples can overlap on the event loop. On close, the worker is released back to the pool (not destroyed). See [Async Environment Calls](#async-environment-calls-rayget-fix) for background.

- **`env_worker.py`** contains the actual Android emulator interaction: AVD cloning, emulator lifecycle, action parsing (`<tool_call>` XML to `JSONAction`), coordinate rescaling, and reward shaping.

## Reward Design

Rewards are computed inside `env_worker.py`'s `step()` when the agent explicitly terminates (no external `--rm-type` needed). If the agent hits `max_turns` without terminating, the reward is `0.0` — no task evaluation is performed.

| Condition | Reward |
|-----------|--------|
| Task success (`is_successful = 1`) + agent terminated | `1.0 * step_scale + thinking_bonus` |
| Task failure + agent terminated | `0.0 - premature_penalty` |
| Max turns reached (no explicit termination) | `0.0` |

- **Step scale**: `exp(-0.1 * steps)`, clamped to [0.1, 1.5] — rewards fewer steps
- **Thinking bonus**: Sigmoid-scaled bonus based on average thinking tokens per step
- **Premature penalty**: Up to 0.5, proportional to how early the agent terminated

### Dynamic Sampling

When `--dynamic-sampling-filter-path` is set (as in `run_android_world_history.sh`), the rollout loop filters out prompt groups before they enter the training batch. For Android World, the filter `check_reward_nonzero_std_history` drops groups where all N trajectory rewards are identical (std == 0), because those groups produce zero advantage and zero gradients.

#### How the rollout loop works with filtering

The rollout loop in `sglang_rollout.py` collects groups until it has `rollout_batch_size` (16) **surviving** groups:

```
target_data_size = args.rollout_batch_size  # 16 groups

while len(data) < target_data_size:
    # Request over_sampling_batch_size prompts from data source
    samples = data_source(args.over_sampling_batch_size)  # 32 prompts
    state.submit_generate_tasks(samples)

    # Wait for completions
    for each completed group:
        all_data.append(group)                    # always kept (for optional post-processing)
        if dynamic_filter(group).keep == False:
            remaining_batch_size -= 1             # triggers more submissions
            continue
        data.append(group)                        # only surviving groups
```

The loop tracks `remaining_batch_size` — when a group is dropped, it decrements, which triggers more prompt submissions on the next iteration. Excess pending tasks are aborted once 16 groups survive.

#### `--over-sampling-batch-size` (32)

This controls how many **prompts** are requested from the data source per call (not samples). Each prompt is expanded into `n_samples_per_prompt` (8) copies. With `rollout_batch_size=16`, setting `over_sampling_batch_size=32` means each `data_source()` call pre-submits 32 prompt groups (256 sample copies). The loop only keeps 16 surviving groups and aborts the rest.

The 2x ratio provides headroom for the expected filter drop rate. If ~50% of groups have uniform rewards (common in early Android World training), 32 is reasonable — submit 32, expect ~16 to survive. If the drop rate is lower, excess groups are simply aborted. The only downside is wasted rollout compute on aborted groups, which is minor compared to the latency of iterating the while loop to request more prompts one-by-one.

#### Filter function signature

```python
def check_reward_nonzero_std_history(args, group, **kwargs):
    """Each element of group is list[Sample] (one trajectory's steps).
    Drop the group if all N trajectory rewards are identical (std == 0)."""
    traj_rewards = [traj[0].get_reward_value(args) for traj in group]
    keep = bool(torch.tensor(traj_rewards).std() > 0.0)
    return DynamicFilterOutput(keep=keep, reason=f"zero_std_{round(traj_rewards[0], 1)}" if not keep else None)
```

The `reason` string (e.g., `"zero_std_0.0"`, `"zero_std_1.0"`) is passed to `MetricGatherer` and becomes part of the logged metric name — see below. Note: `float()` is applied to the reward before `round()` to ensure consistent metric names (without this, `int` rewards like `0` would produce `drop_zero_std_0` instead of `drop_zero_std_0.0`).

### TensorBoard Metrics

#### Reward metrics (post-filter)

Three reward-related metrics appear under `rollout/` in TensorBoard. **All three are computed on post-filter samples only** — groups dropped by the dynamic sampling filter are excluded. This is correct for training (you only train on surviving groups) but means these metrics do not reflect the raw policy output distribution.

| Metric | Source | Computation | Meaning |
|--------|--------|-------------|---------|
| `rollout/raw_reward` | `rollout_data["raw_reward"]` | Simple average across samples | Original shaped reward from `env.get_reward()`, before any normalization. **Post-filter**: groups with uniform rewards (e.g., all-zero) are excluded, so this value is biased upward. |
| `rollout/rewards` | `rollout_data["rewards"]` | Simple average across samples | Reward after GRPO group normalization (mean-subtracted, optionally std-divided within each group of `n_samples_per_prompt`). Post-filter. |
| `rollout/returns` | `rollout_data["returns"]` | Loss-mask-weighted average, gathered across DP ranks | GRPO-normalized reward broadcast to every response token, then weighted by `loss_mask`. Post-filter. |

For GRPO, `returns = advantages = normalized_reward` repeated for each response token (`slime/backends/megatron_utils/loss.py`). The `rollout/returns` logging weights each sample by its loss-mask sum, while `rollout/rewards` weights all samples equally — this is why the two can differ even though they derive from the same normalized values. In practice, they differ only when `loss_mask` contains zeros (e.g., incremental mode where observation tokens have `mask=0`).

**History-based mode note**: In history-based rollout, `rollout/rewards` and `rollout/returns` will be **identical**. This is expected: each step-sample has `loss_mask = [1] * response_length` (all ones, no observation tokens), so the loss-mask-weighted average in `rollout/returns` degenerates to a simple average — the same computation as `rollout/rewards`. The custom `post_process_rewards_history` function controls how both are computed: it groups samples by `group_index` (prompt), normalizes across N trajectories per prompt with equal weight per trajectory regardless of step count, and broadcasts the result to all step-samples. `rollout/raw_reward` is unaffected (always the original `env.get_reward()` value, averaged across all step-samples — longer trajectories contribute more step-samples to this average, but it is logging-only and does not affect training).

#### Dynamic filter metrics

When `--dynamic-sampling-filter-path` is set, the `MetricGatherer` tracks drop counts and logs them via `rollout_extra_metrics` at the rollout actor level (`ray/rollout.py:1161`). These appear in TensorBoard as:

| Metric | Meaning |
|--------|---------|
| `rollout/dynamic_filter/drop_zero_std_0.0` | Number of groups dropped because all N trajectories scored 0.0 (all-fail) |
| `rollout/dynamic_filter/drop_zero_std_1.0` | Number of groups dropped because all N trajectories scored ~1.0 (all-succeed) |

These counts are per-rollout-iteration. A high `drop_zero_std_0.0` early in training is expected (many tasks are too hard initially). As training progresses, expect `drop_zero_std_0.0` to decrease and `drop_zero_std_1.0` to increase (tasks become "solved" and all N trajectories succeed).

#### Interpreting metrics with filtering enabled

Because `rollout/raw_reward` is post-filter, it does **not** represent the true policy success rate. To estimate the actual success rate, combine it with the filter drop counts:

```
total_groups_attempted = rollout_batch_size + drop_zero_std_0.0 + drop_zero_std_1.0
pre_filter_success_rate ≈ (rollout/raw_reward * rollout_batch_size + 1.0 * drop_zero_std_1.0) / total_groups_attempted
```

This is approximate because `raw_reward` includes step-shaping (not exactly 0 or 1), but it gives a rough picture. The `rollout/zero_std/*` metrics (logged separately by `compute_metrics_from_samples` on the rollout side) count zero-std groups **within the surviving batch** — these should be zero when the dynamic filter is active, since the filter removes them before they reach the batch.

## Token and Loss Mask Layout

For a 3-turn episode, the token sequence looks like:

```
[prompt tokens] [model output 1] [env obs 1] [model output 2] [env obs 2] [model output 3]
                 loss_mask=1       mask=0       mask=1           mask=0       mask=1
```

- **Prompt tokens**: System prompt + task description + initial screenshot (first turn only)
- **Model output tokens** (`loss_mask=1`): Trained with RL loss, have real log-probs from SGLang
- **Environment observation tokens** (`loss_mask=0`): Minimal step counter + new screenshot, tokenized and appended with `logprob=0.0`, excluded from loss

Observation tokens are kept lightweight: each subsequent turn only includes `"Step N of M: <image>"` (~15 tokens of text). The task description and all prior actions are already in the token sequence from the initial prompt and earlier model outputs, so repeating them would waste context budget. SGLang's KV cache retains the full history.

## Rollout Modes

### Incremental (`rollout.py`) — default

The entire trajectory is a single training sample. Context grows turn-by-turn via token appending, and SGLang's KV cache avoids recomputation. Efficient for long episodes but produces only 1 training sample per trajectory.

```
--custom-generate-function-path examples.android_world.rollout.generate
```

### Non-Incremental History-Based (`rollout_history.py`)

Each turn is an **independent** training sample: the prompt is rebuilt from scratch with the full task description and complete action history (using `ANDROID_WORLD_TEMPLATE`), plus the current screenshot. A trajectory of T steps yields T standalone samples, all sharing the same trajectory-level reward. This matches standard VLM RL methods (R1-V, InternVL-RL).

```
--custom-generate-function-path examples.android_world.rollout_history.generate
--custom-reward-post-process-path examples.android_world.rollout_history.post_process_rewards_history
--dynamic-sampling-filter-path examples.android_world.rollout_history.check_reward_nonzero_std_history
--over-sampling-batch-size 32
```

**Launch script:**

```bash
# Multi-node with external Ray
SLIME_SCRIPT_EXTERNAL_RAY=1 bash examples/android_world/run_android_world_history.sh

# Single-node
SLIME_SCRIPT_NUM_NODES=1 SLIME_SCRIPT_NUM_GPUS=8 bash examples/android_world/run_android_world_history.sh
```

#### How it works

For a 3-turn episode, history-based mode produces 3 independent samples:

```
Sample 1:  [sys + task + screenshot_0]     → [response_0]
Sample 2:  [sys + task + history(0) + screenshot_1]  → [response_1]
Sample 3:  [sys + task + history(0,1) + screenshot_2]  → [response_2]
```

Each sample has:
- `tokens` = full sequence (prompt + response)
- `loss_mask` = `[1] * response_length` (response-only)
- `rollout_log_probs` = log-probs from SGLang (response-only)
- `reward` = trajectory-level reward (same across all T samples)

The action history text is built by `_extract_action_summary()`, which extracts the `<conclusion>` block from each prior response and formats it as:

```
Step 1: I clicked on the Contacts app icon to open it.
Step 2: I typed "John" into the name field.
```

#### Token budget

Each turn's prompt is built independently, so the budget is per-step:

```
step_budget = max_context_len - len(prompt_ids_for_this_step)
```

There is no cross-turn accumulated budget. If a step's prompt exceeds `max_context_len` (e.g., very long action history), that step is skipped and the trajectory ends.

#### Comparison

| | Incremental (`rollout.py`) | History-based (`rollout_history.py`) |
|---|---|---|
| Training samples per trajectory | 1 | T (one per step) |
| Prompt template | `ANDROID_WORLD_TEMPLATE_NO_HIS` (step 1) + `ANDROID_WORLD_TEMPLATE_STEP` (subsequent) | `ANDROID_WORLD_TEMPLATE` (every step) |
| Context growth | Accumulated (KV cache reuse) | Rebuilt from scratch each step |
| SGLang calls per T-step trajectory | T (incremental, cached) | T (independent, no cache reuse) |
| Token budget | Global across all turns | Independent per step |
| GRPO reward normalization | Default `_post_process_rewards` (reshape works because 1 sample per trajectory) | Custom `post_process_rewards_history` (groups by prompt, normalizes per-trajectory) |
| Dynamic sampling filter | Not needed (1 sample/trajectory, default reshape handles it) | `check_reward_nonzero_std_history` drops zero-variance groups; `--over-sampling-batch-size 32` provides filter headroom |
| Matching RL method | Custom multi-turn RL | R1-V / InternVL-RL style VLM RL |

#### GRPO with history-based rollout

With `--n-samples-per-prompt N`, each task prompt runs N independent trajectories. Each trajectory produces T step-samples (T varies per trajectory). After flattening:

```
[traj1_step1, ..., traj1_stepT1, traj2_step1, ..., traj2_stepT2, ..., trajN_stepTN]
```

Because trajectory lengths vary, the total sample count != `N * rollout_batch_size`. The default `_post_process_rewards` reshape falls through to a `(1, total)` fallback, which normalizes all rewards across all prompts as a single group — breaking GRPO's per-prompt normalization semantics and biasing gradients toward longer trajectories.

The `--custom-reward-post-process-path examples.android_world.rollout_history.post_process_rewards_history` flag fixes this by:
1. Grouping samples by `group_index` (one group per prompt)
2. Within each group, identifying trajectories by `index` (the sample copy ID)
3. Computing one reward per trajectory (all steps share the same reward)
4. Normalizing across N trajectory rewards per prompt (mean-subtraction, optional std normalization)
5. Broadcasting the normalized reward back to every step-sample in each trajectory

This ensures each trajectory contributes equally to the gradient regardless of step count, and different prompts are never mixed during normalization.

**Trimming edge case**: After flattening, the total sample count is trimmed to a multiple of `global_batch_size` by chopping trailing samples (`ray/rollout.py:528`). This can split the last group, leaving only a subset of its trajectories' step-samples. If only 1 trajectory survives trimming for a group, `torch.std()` with Bessel correction (default `ddof=1`) returns `nan` (division by `N-1 = 0`). `post_process_rewards_history` guards against this by skipping std normalization when `len(traj_rewards) <= 1`. In that case, mean-subtraction of a single value yields `0.0`, which is correct (a single trajectory provides no contrastive signal).

#### Dynamic sampling filter

With `--dynamic-sampling-filter-path examples.android_world.rollout_history.check_reward_nonzero_std_history`, prompt groups where all N trajectories have the same reward are dropped before entering the training batch. This is important because:

- **Zero-variance groups produce zero advantages** (GRPO mean-subtraction yields 0 for all), which means zero gradients for those samples
- **Zero gradients can cause NaN** in some optimizer configurations, particularly with gradient clipping or mixed precision
- **Wasted compute**: training on zero-gradient samples provides no learning signal but consumes GPU time

The filter operates at the group level (one group = N trajectory copies of a single prompt). In early training, many groups will be all-fail (`reward ≈ 0` for all N), and late in training, some groups will be all-succeed (`reward ≈ 1` for all N). Both are dropped.

See [Dynamic Sampling](#dynamic-sampling) for the full mechanism and [TensorBoard Metrics](#tensorboard-metrics) for how to monitor filter activity.

## Verification Tests

Run from the slime repo root:

```bash
# Step 0: Reward normalization unit tests (no emulators/GPU needed)
pytest examples/android_world/tests/test_reward_post_process.py -v

# Step 1: Import verification (no emulators/GPU needed)
python examples/android_world/tests/test_imports.py

# Step 2: Env pool + worker ops (needs emulators + Ray)
python examples/android_world/tests/test_env_pool.py

# Step 3: End-to-end rollout (needs emulators + GPU + SGLang)
# First start SGLang:
python -m sglang.launch_server --model-path /root/models/Qwen3-VL-8B-Instruct \
    --port 30000 --mem-fraction-static 0.5

# Then run:
SLIME_TEST_MODEL=/root/models/Qwen3-VL-8B-Instruct \
SLIME_TEST_SGLANG_PORT=30000 \
  python examples/android_world/tests/test_rollout.py
```

## Cleanup

Kill emulators and ADB servers after training or when recovering from a crash:

```bash
# Kill emulator processes and ADB servers only (safe default)
bash examples/android_world/cleanup.sh

# Also remove cloned AVD files (slime_aw_1, slime_aw_2, ...)
bash examples/android_world/cleanup.sh --avd

# Full cleanup: AVDs + temp images + ADB lock files
bash examples/android_world/cleanup.sh --all
```

The script reads `config.yaml` for paths and the AVD naming pattern, so it only touches resources created by slime. It will not affect emulators from other projects.

## Async Environment Calls (`ray.get` Fix)

### Problem

Running with `num_workers=64` vs `num_workers=128` in `config.yaml` produced nearly identical rollout times (~7000 seconds per step). Doubling the emulator workers had no measurable effect on throughput.

### Root Cause: `ray.get()` Blocking the asyncio Event Loop

The `generate()` function in `rollout.py` is an `async` coroutine. slime launches many `generate()` calls concurrently (one per sample) on a shared asyncio event loop. The expectation is that while one sample waits on I/O (SGLang inference, emulator interaction), other samples can make progress.

However, `AndroidWorldEnv.reset()` and `step()` were using synchronous `ray.get()` to wait for remote worker results:

```python
# OLD — blocks the entire event loop thread
def step(self, response_text):
    obs, step_reward, done, info = ray.get(self.worker_ref.step.remote(response_text))
```

`ray.get()` is a **blocking call** — it suspends the calling thread until the Ray task completes. Since all `generate()` coroutines share the same event loop thread, a single `ray.get()` call blocks **every** concurrent coroutine from making progress. This serializes what should be parallel work:

```
Timeline with blocking ray.get() (only 1 sample progresses at a time):

Event loop thread: [sample_1.step() BLOCKED 5s] [sample_2.step() BLOCKED 3s] [sample_3.step() BLOCKED 7s] ...
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    Other coroutines cannot run while ray.get() holds the thread
```

With 128 concurrent samples each doing 5-30 turns of `env.step()` (3-7 seconds per turn), the serialization means only one emulator interaction runs at any time. Adding more workers cannot help — the bottleneck is the single event loop thread, not worker availability.

### Fix: `asyncio.to_thread(ray.get, ...)`

The two blocking methods in `env_android_world.py` were converted to async using `asyncio.to_thread()`:

```python
# NEW — offloads blocking ray.get() to a thread pool, freeing the event loop
async def step(self, response_text):
    obs, step_reward, done, info = await asyncio.to_thread(
        ray.get, self.worker_ref.step.remote(response_text)
    )
```

`asyncio.to_thread()` runs the blocking function in a separate thread from Python's default `ThreadPoolExecutor`. The `await` yields control back to the event loop, allowing other coroutines to run while the Ray call completes in the background.

```
Timeline with async ray.get() (all samples progress concurrently):

Event loop thread: [sample_1 await] [sample_2 await] [sample_3 await] [sample_1 resumes] ...
Thread pool:       [ray.get(s1)     ─────────────────────────────────── done]
                   [ray.get(s2)     ───────────────── done]
                   [ray.get(s3)     ──────────────────────────── done]
```

**Files changed:**

| File | Change |
|------|--------|
| `env_android_world.py` | `reset()`, `step()` changed from `def` to `async def`, `ray.get(...)` wrapped with `await asyncio.to_thread(ray.get, ...)` |
| `rollout.py` | All call sites updated: `env.reset()` / `env.step()` now use `await` |

### Why This Matters

Each emulator turn takes 3-7 seconds (`execute_action` + `sleep(2)` + `get_state(wait_to_stabilize)`). A full episode runs 5-30 turns, so a single sample takes 40-200 seconds. With 128 concurrent samples serialized by blocking `ray.get()`, the event loop can only complete ~1 emulator interaction at a time, turning what should be a parallel workload into a sequential one.

After this fix, all 128 samples can have their `ray.get()` calls in-flight simultaneously across the thread pool, and the event loop remains free to dispatch SGLang inference calls, process completions, and manage other async I/O. The rollout time for a batch should now scale with the **slowest single sample** rather than the **sum of all emulator wait times**.

### Notes

- `asyncio.to_thread()` uses Python's default `ThreadPoolExecutor`. The default pool size is `min(32, os.cpu_count() + 4)`. If you run >32 concurrent samples, the thread pool itself becomes a bottleneck. In that case, increase the pool size at startup: `import asyncio; loop = asyncio.get_event_loop(); loop.set_default_executor(ThreadPoolExecutor(max_workers=256))`.
- The `BaseInteractionEnv` base class in `geo3k_vlm_multi_turn/base_env.py` still defines sync `reset()`/`step()`. This is fine — `Geo3kEnv` performs only local CPU work (no blocking I/O), so sync calls do not block the event loop. The async override is specific to `AndroidWorldEnv` where remote Ray calls introduce multi-second blocking waits.
- `format_observation()`, `get_reward()`, and `close()` remain synchronous. `format_observation()` does local image resizing and string formatting (microseconds, no I/O). `get_reward()` returns a stored value. `close()` calls `pool.release()` which is a simple queue put (also non-blocking).

## Troubleshooting

**Emulator boot failures**: Check that the base AVD exists at `android_avd_home/avd_name.avd`. Workers retry up to 10 times with 10s intervals.

**OOM on emulator nodes**: Reduce `num_workers` or `resources_per_worker.memory` in `config.yaml`.

**Long token sequences**: With many turns, episodes can reach 20k+ tokens. Reduce `max_turns` or `--rollout-max-response-len` for initial experiments.

**Pool exhaustion (generate calls hanging)**: If `num_workers < rollout_batch_size`, some generate calls will block waiting for a worker. Increase `num_workers` or decrease `--rollout-batch-size`.

**Port conflicts**: Workers auto-detect available ports starting from `base_console_port`. If other emulators are running, increase the base port values in `config.yaml`.

**Stale emulators after crash**: If a previous run was interrupted, orphaned emulators may hold ports and memory. Run `bash examples/android_world/cleanup.sh --avd` to kill them and remove cloned AVDs before restarting.

**`assert video_embeds is None, "not support video now"` with Megatron backend**: This occurs when `--context-parallel-size` is set to a value greater than 1. Context parallelism (CP) splits token sequences across CP ranks via `slice_with_cp`, so each rank only sees a portion of the `input_ids`. The Megatron bridge Qwen3-VL model computes `video_start_index = image_mask.sum()` using the local CP-sliced tokens, but compares it against `vision_embeds.shape[0]` from the full (unsliced) `pixel_values` tensor. When CP > 1, the local image token count is smaller than the total vision embeddings, causing the model to misinterpret the remainder as video embeddings and trigger the assertion. The fix is to set `--context-parallel-size 1`. For multi-node setups needing long-sequence support, increase `--tensor-model-parallel-size` instead (up to 8, constrained to a single node due to inter-node bandwidth). TP=8 with CP=1 on 2 × 8 GPUs gives DP=2.
