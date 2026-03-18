# Android World 环境集成分析报告

本文档详细分析了 `slime_android` 目录下的代码如何将 Android World 环境集成到 slime 框架中，包括整体工作流、函数层面和代码层面的深入分析。

---

## 目录

1. [概述](#概述)
2. [架构对比](#架构对比)
3. [核心模块详解](#核心模块详解)
4. [两种 Rollout 模式](#两种-rollout-模式)
5. [数据流与训练流程](#数据流与训练流程)
6. [配置与使用指南](#配置与使用指南)
7. [代码层面的关键设计](#代码层面的关键设计)
8. [与原始 slime 框架的差异](#与原始-slime-框架的差异)

---

## 概述

### 什么是 Android World

Android World 是一个 Android 操作系统环境基准测试，用于评估 AI 智能体在真实 Android 设备上执行复杂任务的能力。它包含 116 个不同的任务类型（如日历操作、联系人管理、短信发送、相机操作等），每个任务有 20 个不同的参数配置，共 2320 个任务实例。

### 集成目标

将 Android World 集成到 slime 框架的目标是：
- **训练 VLM 智能体**：使用强化学习（GRPO/GSPO）训练视觉语言模型来操作 Android 设备
- **多轮交互**：支持截图-动作-反馈的多轮交互模式
- **高效环境池**：通过 Ray actor 池管理多个 Android 模拟器实例

### 代码结构

```
slime_android/examples/android_world/
├── __init__.py                    # 模块初始化
├── config.yaml                    # 配置文件（环境池、模拟器参数等）
├── run_android_world.py           # 训练入口脚本
├── prompts.py                     # 系统提示词和模板
├── env_android_world.py           # 单样本环境封装
├── env_pool.py                    # 异步环境池（单例）
├── env_worker.py                  # Ray actor 工作器（封装模拟器）
├── rollout.py                     # 增量式多轮 rollout
├── rollout_history.py             # 历史式多轮 rollout
├── data/
│   ├── generate_tasks.py          # 任务 JSONL 生成脚本
│   ├── task_list.txt              # 任务名称列表
│   ├── tasks.jsonl                # 训练任务数据
│   └── tasks_eval.jsonl           # 评估任务数据
└── tests/
    ├── test_env_pool.py           # 环境池测试
    ├── test_imports.py            # 导入测试
    ├── test_reward_post_process.py # 奖励后处理测试
    └── test_rollout.py            # Rollout 集成测试
```

---

## 架构对比

### slime 原框架的多轮交互模式

slime 原框架提供了 `examples/geo3k_vlm_multi_turn/` 作为多轮 VLM 交互的参考实现：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Rollout       │     │   SGLang        │     │   Environment   │
│   Actor         │────▶│   Engine        │────▶│   (geo3k)       │
│   (Ray Actor)   │     │   (Inference)   │     │   (sync)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

关键特点：
- 环境通过 `--rollout-interaction-env-path` 指定
- 环境模块必须实现 `build_env(sample, args)` 函数
- 单个环境实例，同步执行

### Android World 的改进架构

Android World 集成对架构进行了重要改进：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────────────┐
│   Rollout       │     │   SGLang        │     │   AndroidWorldEnvPool (单例)    │
│   Actor         │────▶│   Router        │────▶│   ┌─────────┐  ┌─────────┐     │
│   (Ray Actor)   │     │   (HTTP)        │     │   │ Worker0 │  │ Worker1 │ ... │
└─────────────────┘     └─────────────────┘     │   │(Ray)    │  │(Ray)    │     │
                                                │   │Emulator │  │Emulator │     │
                                                │   └─────────┘  └─────────┘     │
                                                └─────────────────────────────────┘
```

关键改进：
1. **环境池化**：预创建多个模拟器实例，避免每次重置的高开销
2. **异步获取/释放**：通过 `asyncio.Queue` 实现非阻塞的工作器调度
3. **Ray 远程执行**：每个工作器是独立的 Ray actor，支持分布式部署
4. **持久化环境**：模拟器在整个训练过程中保持运行，只是重置任务状态

---

## 核心模块详解

### 1. `env_worker.py` - Android World 工作器

这是整个集成的核心，封装了与 Android World SDK 的所有交互。

#### 类：`AndroidWorldWorker`

```python
class AndroidWorldWorker:
    """封装单个 Android 模拟器实例。

    作为 Ray remote actor 使用，每个工作器管理一个模拟器。
    """
```

**初始化流程**：

```python
def __init__(self, worker_id, console_port, grpc_port, adb_path, ...):
    # 1. 克隆 AVD（Android Virtual Device）
    if base_avd_name:
        self._clone_avd_locally()

    # 2. 初始化 AndroidWorld 环境
    self._initialize_env()

    # 3. 预生成任务参数套件
    task_registry = registry.TaskRegistry()
    self.suite = suite_utils.create_suite(
        task_registry.get_registry(family="android_world"),
        n_task_combinations=20,
        seed=30,
    )
```

**关键方法**：

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `reset(task_name, params_idx)` | 重置环境并初始化指定任务 | `(observation, info)` |
| `step(raw_action)` | 执行动作并返回新观察 | `(observation, reward, done, info)` |
| `close()` | 关闭模拟器 | - |
| `_get_obs()` | 获取当前截图和状态 | `dict` |
| `_compute_step_reward()` | 计算奖励塑形 | `float` |

**动作解析**：

`parse_ui_action_from_response()` 函数将模型输出解析为 `json_action.JSONAction`：

```python
# 输入格式（模型输出）：
# {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}

# 输出：json_action.JSONAction 对象
JSONAction(
    action_type="click",
    x=540,  # 重缩放后的坐标
    y=720,
)
```

支持的动作类型：
- `click`: 点击坐标
- `long_press`: 长按
- `swipe`: 滑动
- `type`/`input_text`: 输入文本
- `open_app`: 打开应用
- `system_button`: 系统按钮（返回、主页等）
- `wait`: 等待
- `terminate`: 终止任务
- `answer`: 回答问题

### 2. `env_pool.py` - 异步环境池

#### 类：`AndroidWorldEnvPool`

```python
class AndroidWorldEnvPool:
    """单例异步池，管理持久的 AndroidWorldWorker Ray actors。"""

    _instance: AndroidWorldEnvPool | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, config: dict) -> AndroidWorldEnvPool:
        """返回单例池，首次调用时创建。"""
```

**设计模式**：
- **单例模式**：所有并发 `generate()` 调用共享同一池
- **双重检查锁定**：确保线程安全的延迟初始化
- **异步队列**：`asyncio.Queue` 实现工作器的 acquire/release

**工作流程**：

```python
# 初始化（首次调用）
pool = await AndroidWorldEnvPool.get_instance(vars(args))

# 获取工作器（阻塞直到可用）
worker_ref, worker_id = await pool.acquire()

# 使用工作器
obs, info = await asyncio.to_thread(ray.get, worker_ref.reset.remote(...))
obs, reward, done, info = await asyncio.to_thread(ray.get, worker_ref.step.remote(...))

# 释放工作器
pool.release(worker_id)
```

**资源配置**（来自 config.yaml）：

```yaml
num_workers: 128
resources_per_worker:
  num_cpus: 4
  memory: 8589934592  # 8 GB
```

### 3. `env_android_world.py` - 单样本环境封装

#### 类：`AndroidWorldEnv`

```python
class AndroidWorldEnv(BaseInteractionEnv):
    """单个样本的环境封装，委托给远程 AndroidWorldWorker。"""
```

**继承关系**：

```
BaseInteractionEnv (geo3k_vlm_multi_turn/base_env.py)
    │
    └── AndroidWorldEnv
            │
            ├── worker_ref: Ray actor 引用
            ├── worker_id: 工作器 ID
            └── pool: 环境池引用
```

**关键方法**：

```python
async def reset(self) -> tuple[dict, dict]:
    """重置远程工作器。"""
    obs, info = await asyncio.to_thread(
        ray.get, self.worker_ref.reset.remote(self.task_name, self.params_idx)
    )
    return obs, info

async def step(self, response_text: str) -> tuple[dict, bool, dict]:
    """在远程工作器上执行动作。"""
    obs, step_reward, done, info = await asyncio.to_thread(
        ray.get, self.worker_ref.step.remote(response_text)
    )
    if done:
        self.final_reward = step_reward
    return obs, done, info
```

**观察格式化**：

`format_observation()` 方法将环境观察转换为 VLM 兼容的聊天消息：

```python
def format_observation(self, observation, is_initial=True) -> dict:
    """返回格式：
    {
        "role": "user",
        "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "..."}
        ]
    }
    """
```

- **初始轮**：使用 `ANDROID_WORLD_TEMPLATE_NO_HIS`（包含任务描述）
- **后续轮**：使用 `ANDROID_WORLD_TEMPLATE_STEP`（仅步骤计数和截图）

### 4. `prompts.py` - 提示词模板

#### 系统提示词

`ANDROID_WORLD_SYSTEM_PROMPT` 定义了工具调用格式：

```python
ANDROID_WORLD_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call one function per step to assist with the user query.

<tools>
{"type": "function", "function": {"name": "mobile_use", ...}}
</tools>

For each function call, return a json object within <tool_call XML tags:
<tool_call:
{"name": <function-name>, "arguments": <args-json-object>}
"""
```

#### 观察模板

```python
# 初始轮（无历史）
ANDROID_WORLD_TEMPLATE_NO_HIS = """
The user query: {task_description}

Before answering, explain your reasoning step-by-step in <thinking></thinking> tags.
After answering, summarize your observation and action in <conclusion></conclusion> tags.

<image>
"""

# 后续轮（轻量级）
ANDROID_WORLD_TEMPLATE_STEP = """
Step {current_step} of {max_steps}: <image>
"""

# 完整历史模板（用于 rollout_history.py）
ANDROID_WORLD_TEMPLATE = """
The user query: {task_description}

Task progress (You have done the following {step_count} operations):
{action_history}

Step {current_step}: <image>
"""
```

---

## 两种 Rollout 模式

Android World 集成提供了两种不同的 rollout 策略：

### 模式 1：增量式 Rollout（`rollout.py`）

**特点**：一个轨迹生成一个 Sample，上下文逐轮增长

```
Turn 0: [System][User: Task + Screenshot0] → [Assistant: Action0]
Turn 1: [System][User: Task + Screenshot0][Assistant: Action0][User: Screenshot1] → [Assistant: Action1]
Turn 2: [System][User: Task + Screenshot0][Assistant: Action0][User: Screenshot1][Assistant: Action1][User: Screenshot2] → [Assistant: Action2]
...
```

**关键代码**：

```python
async def generate(args, sample, sampling_params) -> Sample:
    # 初始化
    state = GenerateState(args)
    obs, info = await env.reset()

    # 构建初始提示
    first_user_message = env.format_observation(obs)
    prompt_text, prompt_ids, ... = _build_initial_prompt(...)

    sample.tokens = list(prompt_ids)
    response_tokens = []
    multimodal_train_inputs_buffer = []

    # 多轮循环
    for turn_idx in range(max_turns):
        # 生成
        response_text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
            url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer
        )

        # 追加模型输出（loss_mask=1）
        _append_to_sample(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1)

        # 环境步进
        step_obs, done, step_info = await env.step(response_text)
        if done:
            break

        # 编码观察（loss_mask=0）
        obs_prompt_ids, obs_image_data, obs_mm, obs_mm_train = _encode_observation_for_generation(...)
        _append_to_sample(sample, response_tokens, obs_prompt_ids, [0.0]*len(obs_prompt_ids), loss_mask_val=0)

        # 更新多模态状态
        current_image_data = _update_multimodal_state(...)

    # 设置奖励
    sample.reward = env.get_reward()

    return _finalize_sample(sample, ...)
```

**loss_mask 语义**：
- `loss_mask=1`：模型生成的 token（计算 loss）
- `loss_mask=0`：环境观察的 token（不计算 loss）

**优点**：
- 内存效率高（一个 Sample 存储整个轨迹）
- 上下文连续，模型可以利用完整历史

**缺点**：
- 训练信号稀疏（只有一个奖励）
- 上下文长度限制更严格

### 模式 2：历史式 Rollout（`rollout_history.py`）

**特点**：每轮生成独立的 Sample，每轮包含完整历史

```
Turn 0: [System][User: Task + Screenshot0] → [Assistant: Action0] → Sample0
Turn 1: [System][User: Task + Screenshot0 + History0 + Screenshot1] → [Assistant: Action1] → Sample1
Turn 2: [System][User: Task + Screenshot0 + History0-1 + Screenshot2] → [Assistant: Action2] → Sample2
...
```

**关键代码**：

```python
async def generate(args, sample, sampling_params, evaluation=False) -> list[Sample] | Sample:
    trajectory_samples: list[Sample] = []
    action_history: list[str] = []

    for turn_idx in range(max_turns):
        # 每轮构建独立提示（包含完整历史）
        step_user_msg = _format_observation_with_history(
            obs, action_history, max_steps, image_size, history_window_size
        )
        prompt_text, prompt_ids, ... = _build_step_prompt(...)

        # 独立推理
        response_text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
            url, list(prompt_ids), cur_params, image_data, state.tokenizer
        )

        # 创建独立 Sample
        turn_sample = _make_turn_sample(
            original_sample=sample,
            prompt_text=prompt_text,
            prompt_ids=list(prompt_ids),
            response_tokens=new_tokens,
            response_logprobs=new_logprobs,
            tokenizer=state.tokenizer,
        )

        trajectory_samples.append(turn_sample)

        # 记录动作历史
        action_history.append(_extract_action_summary(response_text))

        # 环境步进
        step_obs, done, _ = await env.step(response_text)
        if done:
            break

    # 分配轨迹级奖励到所有步骤
    reward = env.get_reward()
    for s in trajectory_samples:
        s.reward = reward

    return trajectory_samples
```

**Sample 结构**：

```python
def _make_turn_sample(...) -> Sample:
    return Sample(
        group_index=original_sample.group_index,
        index=original_sample.index,
        prompt=prompt_text,              # 完整提示
        tokens=full_tokens,              # prompt_ids + response_tokens
        loss_mask=[1] * len(response_tokens),  # 只对响应计算 loss
        rollout_log_probs=response_logprobs,
        response=tokenizer.decode(response_tokens),
        response_length=len(response_tokens),
        multimodal_inputs=multimodal_inputs,
        multimodal_train_inputs=...,
        status=Sample.Status.COMPLETED,
    )
```

**优点**：
- 每轮独立训练信号
- 兼容标准 VLM RL 方法（R1-V、InternVL-RL 风格）
- 可以限制历史窗口（`history_window_size`）

**缺点**：
- 内存开销更大
- 每轮需要重新编码完整提示

### 自定义奖励后处理

历史式 rollout 需要特殊的奖励归一化逻辑：

```python
def post_process_rewards_history(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """GRPO 奖励归一化，正确处理变长轨迹。

    问题：默认的 _post_process_rewards 将奖励 reshape 为 (-1, n_samples_per_prompt)，
    但变长轨迹导致总数 != N*B，回退到全局归一化，混合了不同 prompt。

    解决：
    1. 按 group_index（prompt）分组
    2. 按 index（轨迹）子分组
    3. 每个轨迹取一个奖励（所有步骤共享）
    4. 在 N 个轨迹间归一化
    5. 广播回所有步骤样本
    """
    groups: dict[int, list[tuple[int, Sample]]] = {}
    for pos, s in enumerate(samples):
        groups.setdefault(s.group_index, []).append((pos, s))

    for _group_index, members in groups.items():
        # 按轨迹子分组
        trajs: dict[int, list[int]] = {}
        for pos, s in members:
            trajs.setdefault(s.index, []).append(pos)

        # 提取轨迹奖励
        traj_rewards = [raw_rewards[positions[0]] for positions in trajs.values()]

        # 归一化
        tr = torch.tensor(traj_rewards, dtype=torch.float)
        tr = tr - tr.mean()
        if len(tr) > 1:
            tr = tr / (tr.std() + 1e-6)

        # 广播
        for norm_reward, positions in zip(tr.tolist(), traj_positions):
            for pos in positions:
                normalized[pos] = norm_reward

    return raw_rewards, normalized
```

---

## 数据流与训练流程

### 完整训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           slime 训练流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 数据加载                                                                │
│     tasks.jsonl ──▶ RolloutDataSource ──▶ Sample 对象列表                  │
│                                                                             │
│  2. Rollout 生成                                                            │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ for each sample:                                                │    │
│     │   ┌─────────────────────────────────────────────────────────┐  │    │
│     │   │ AndroidWorldEnvPool.acquire()                            │  │    │
│     │   │ env.reset(task_name, params_idx)                         │  │    │
│     │   │                                                           │  │    │
│     │   │ for turn in range(max_turns):                            │  │    │
│     │   │   screenshot = env.get_obs()                             │  │    │
│     │   │   action = SGLang.generate(screenshot + prompt)          │  │    │
│     │   │   obs, done = env.step(action)                           │  │    │
│     │   │   if done: break                                         │  │    │
│     │   │                                                           │  │    │
│     │   │ reward = env.get_reward()                                │  │    │
│     │   │ AndroidWorldEnvPool.release(worker_id)                   │  │    │
│     │   └─────────────────────────────────────────────────────────┘  │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  3. 奖励计算                                                                │
│     AndroidWorldEnv.get_reward() ──▶ task_instance.is_successful(env)      │
│                                                                             │
│  4. 数据处理                                                                │
│     Sample + reward ──▶ 数据缓冲区 ──▶ 训练批次                            │
│                                                                             │
│  5. 训练更新                                                                │
│     Megatron/FSDP 训练器 ──▶ GRPO loss ──▶ 梯度更新                        │
│                                                                             │
│  6. 权重同步                                                                │
│     训练器 ──▶ SGLang 引擎                                                 │
│                                                                             │
│  7. 重复 2-6                                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 任务数据格式

**tasks.jsonl 格式**：

```json
{"prompt": "Complete a task on Android", "label": "", "metadata": {"task_name": "ContactsAddContact", "params_idx": 0}}
{"prompt": "Complete a task on Android", "label": "", "metadata": {"task_name": "CameraTakePhoto", "params_idx": 4}}
...
```

**字段说明**：
- `prompt`：占位符（实际提示在 `generate()` 中动态构建）
- `label`：空（奖励来自环境）
- `metadata.task_name`：任务类型名称
- `metadata.params_idx`：任务参数索引（0-19）

### 配置文件（config.yaml）

```yaml
# 多轮交互
max_turns: 30
max_context_len: 16384
history_window_size: 10  # 限制动作历史

# 环境池
num_workers: 128
resources_per_worker:
  num_cpus: 4
  memory: 8589934592  # 8 GB

# Android 模拟器设置
avd_name: "AndroidWorldAvd"
base_avd_name_pattern: "slime_aw_{}"
base_console_port: 5556
base_grpc_port: 8554
base_adb_server_port: 5037
android_avd_home: "/root/android/avd/"
android_sdk_root: "/root/android/"
emulator_path: "/root/android/emulator/emulator"
adb_path: "/root/android/platform-tools/adb"
temp_path: "/tmp/android_world_images"
save_images: false
task_family: "android_world"
image_size:
  - 540
  - 1200
```

---

## 配置与使用指南

### 环境准备

#### 1. 安装 Android SDK

```bash
# 下载 Android SDK
mkdir -p /root/android
cd /root/android

# 下载命令行工具
wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip
unzip commandlinetools-linux-9477386_latest.zip

# 安装必要组件
yes | ./cmdline-tools/bin/sdkmanager --sdk_root=/root/android "platform-tools" "emulator" "platforms;android-34"
```

#### 2. 创建 AVD

```bash
# 创建 AVD（首次）
echo "no" | avdmanager create avd -n AndroidWorldAvd -k "system-images;android-34;google_apis;x86_64"

# 或从现有 AVD 复制
cp -r ~/.android/avd/AndroidWorldAvd.avd /root/android/avd/
cp ~/.android/avd/AndroidWorldAvd.ini /root/android/avd/
```

#### 3. 安装 Android World SDK

```bash
pip install android-world
```

#### 4. 下载模型

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /root/models/Qwen3-VL-2B-Instruct
```

### 训练命令

#### 使用增量式 Rollout

```bash
# 启动 Ray
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
    -- python3 examples/android_world/run_android_world.py
```

#### 使用历史式 Rollout

修改 `run_android_world.py` 中的参数：

```python
rollout_args = (
    f"--prompt-data {TASK_DATA} "
    "--custom-generate-function-path examples.android_world.rollout_history.generate "
    "--custom-reward-post-process-path examples.android_world.rollout_history.post_process_rewards_history "
    "--custom-config-path examples/android_world/config.yaml "
    # ... 其他参数
)
```

### 评估模式

```bash
# 评估脚本
python -m slime.eval \
    --hf-checkpoint /root/models/Qwen3-VL-2B-Instruct \
    --prompt-data examples/android_world/data/tasks_eval.jsonl \
    --custom-generate-function-path examples.android_world.rollout_history.generate \
    --custom-config-path examples/android_world/config.yaml \
    --eval-on-the-fly
```

### 测试

```bash
# 启动 SGLang 服务器
python -m sglang.launch_server \
    --model-path /root/models/Qwen3-VL-2B-Instruct \
    --port 30000 \
    --mem-fraction-static 0.5

# 运行测试
SLIME_TEST_MODEL=/root/models/Qwen3-VL-2B-Instruct \
SLIME_TEST_SGLANG_IP=127.0.0.1 \
SLIME_TEST_SGLANG_PORT=30000 \
python examples/android_world/tests/test_rollout.py
```

---

## 代码层面的关键设计

### 1. 异步与同步的桥接

Android World SDK 是同步的，但 slime 的 rollout 是异步的。使用 `asyncio.to_thread` 桥接：

```python
# 同步 Ray 调用包装为异步
obs, info = await asyncio.to_thread(
    ray.get, self.worker_ref.reset.remote(self.task_name, self.params_idx)
)
```

### 2. 端口分配策略

每个模拟器需要三个独立端口：

```python
def generate_env_configs(base_avd_name_pattern, base_console_port, base_grpc_port, num_envs, base_adb_server_port):
    # 控制台端口（必须是偶数）
    console_ports = find_available_ports(base_console_port, num_envs, port_pairs=True)

    # gRPC 端口（与控制台端口有固定偏移）
    grpc_ports = [p + (base_grpc_port - base_console_port) for p in console_ports]

    # ADB 服务器端口（独立分配）
    adb_server_ports = find_available_adb_ports(base_adb_server_port, num_envs)

    return cache_avd_names, console_ports, grpc_ports, adb_server_ports
```

### 3. AVD 克隆

每个工作器需要独立的 AVD 实例：

```python
def clone_avd(src_avd_name, tar_avd_name, android_avd_home):
    """克隆 AVD，复制文件夹和 .ini 文件，更新内部路径。"""
    src_avd_dir = os.path.join(android_avd_home, src_avd_name + ".avd")
    tar_avd_dir = os.path.join(android_avd_home, tar_avd_name + ".avd")

    shutil.copytree(src_avd_dir, tar_avd_dir)

    # 更新配置文件中的 AVD 名称
    for ini_name in ["config.ini", "hardware-qemu.ini"]:
        ini_path = os.path.join(tar_avd_dir, ini_name)
        # 替换 src_avd_name 为 tar_avd_name
```

### 4. 多模态编码

使用 Qwen VL 的工具处理图像：

```python
from qwen_vl_utils import process_vision_info

# 从消息中提取图像
images, videos = process_vision_info([message])

# 处理器编码
processor_output = processor(text=prompt_text, images=images)

# 提取训练用的多模态输入
multimodal_train_inputs = {
    k: v for k, v in processor_output.items()
    if k not in ["input_ids", "attention_mask"] and "video" not in k
}
```

### 5. 动态提示构建

提示不是从 JSONL 加载，而是在运行时构建：

```python
async def generate(args, sample, sampling_params):
    # 1. 从 sample.metadata 获取任务信息
    task_name = sample.metadata.get("task_name")
    params_idx = sample.metadata.get("params_idx", 0)

    # 2. 重置环境
    obs, info = await env.reset()

    # 3. 动态构建提示
    first_user_message = env.format_observation(obs)
    prompt_text, prompt_ids, ... = _build_initial_prompt(
        system_prompt=ANDROID_WORLD_SYSTEM_PROMPT,
        first_user_message=first_user_message,
        ...
    )

    # 4. 覆盖 sample 的占位符提示
    sample.prompt = prompt_text
    sample.tokens = list(prompt_ids)
```

### 6. 奖励计算

奖励来自 Android World 的任务评估：

```python
def step(self, raw_action: str):
    if is_terminal_action(action):
        # 评估任务成功与否
        base_reward = self.task_instance.is_successful(self.env)
        # base_reward 是 float：
        #   1.0 = 完全成功
        #   0.5 = 部分成功
        #   0.0 = 失败

        self.terminated = True
        return before_action_obs, base_reward, True, {"won": base_reward >= 1.0}
```

---

## 与原始 slime 框架的差异

### 核心模块

**完全一致**：`slime_android/slime/` 目录与 `slime/slime/` 目录内容完全相同，没有修改核心框架代码。

### 新增示例

**slime_android 新增**：
- `examples/android_world/` - 完整的 Android World 集成

**slime 原有**（slime_android 保留）：
- `examples/geo3k_vlm/` - 单轮 VLM
- `examples/geo3k_vlm_multi_turn/` - 多轮 VLM（Android World 的参考实现）
- `examples/multi_agent/` - 多智能体
- `examples/tau-bench/` - Tau Bench 集成
- 等等...

### 关键设计差异

| 方面 | geo3k_vlm_multi_turn | android_world |
|------|---------------------|---------------|
| 环境管理 | 单环境，同步 | 环境池，异步 |
| 环境生命周期 | 每次创建/销毁 | 持久化，重用 |
| 提示来源 | JSONL 文件 | 动态构建 |
| 奖励来源 | 外部 RM | 环境评估 |
| Rollout 模式 | 增量式 | 增量式 + 历史式 |
| 资源需求 | 低（无模拟器） | 高（多个模拟器） |

### 复用的代码

Android World 集成大量复用了 geo3k_vlm_multi_turn 的辅助函数：

```python
from examples.geo3k_vlm_multi_turn.rollout import (
    _append_to_sample,
    _encode_observation_for_generation,
    _finalize_sample,
    _merge_multimodal_train_inputs,
    _run_inference_step,
    _should_stop_on_finish,
    _update_budget,
    _update_multimodal_state,
)
```

这确保了与 slime 框架的完全兼容性。

---

## 总结

Android World 集成是一个精心设计的 slime 框架扩展，它：

1. **保持框架兼容**：没有修改任何核心 slime 模块
2. **遵循最佳实践**：复用 geo3k_vlm_multi_turn 的模式
3. **解决实际挑战**：环境池化、异步调度、端口管理
4. **提供灵活性**：两种 rollout 模式适应不同训练需求
5. **完整的测试覆盖**：单元测试和集成测试

这种集成方式可以作为将其他交互式环境（如 Web 浏览、游戏、机器人模拟器）接入 slime 框架的参考模板。
