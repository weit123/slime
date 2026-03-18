# ALFWorld 环境集成文档

本文档详细介绍 slime_android 框架中 ALFWorld 环境的集成实现，包括整体工作流、函数接口和代码层面的说明。

---

## 目录

1. [环境概述](#环境概述)
2. [文件结构](#文件结构)
3. [整体工作流](#整体工作流)
4. [函数接口说明](#函数接口说明)
5. [代码层面说明](#代码层面说明)
6. [配置与使用](#配置与使用)

---

## 环境概述

### 什么是 ALFWorld

ALFWorld 是一个基于 AI2-THOR 的具身智能基准测试环境：
- 模拟家庭环境中的日常任务（如取物、清洁、烹饪等）
- 提供 134 个不同的任务类型
- 结合文本观察和视觉观察
- 支持多种任务复杂度

### 环境特点

| 特性 | 说明 |
|------|------|
| 观察空间 | 300x300 RGB 图像 + 文本描述 |
| 动作空间 | 14 种基础动作（goto, pick, put, open, close, toggle, heat, clean, cool, slice, inventory, examine, look, pass） |
| 奖励 | 50 * won + goal_condition_success_rate - 1 * illegal_action |
| 最大步数 | 40 步（可配置） |
| 环境类型 | 重量级（需要 AI2-THOR + Ray actor 池） |

---

## 文件结构

```
examples/alfworld/
├── __init__.py              # 模块初始化
├── alf_utils.py             # AlfEnv 环境封装（从 GTR-Turbo 适配）
├── env_worker.py            # Ray actor 工作器
├── env_pool.py              # 单例异步环境池
├── env_alfworld.py          # Per-sample 环境封装
├── rollout.py               # 增量式多轮 rollout
├── rollout_history.py       # 历史式多轮 rollout
├── prompts.py               # 提示词模板（与 GTR-Turbo 一致）
├── config.yaml              # 配置文件
├── run_alfworld.py          # 训练入口脚本
├── tests/
│   ├── __init__.py
│   └── test_rollout.py
└── README.md                # 本文档
```

---

## 整体工作流

### 1. 数据流

```
┌─────────────────┐
│ tasks.jsonl     │  占位符（任务从 ALFWorld 环境获取）
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ generate() 函数                                                  │
│                                                                  │
│  1. 从 AlfWorldEnvPool 获取 worker                               │
│  2. reset() → 获取任务描述、admissible_actions                   │
│  3. format_observation() → 构建包含图像和提示词的消息              │
│  4. 循环:                                                        │
│     ├── SGLang HTTP 生成 → response_text                        │
│     ├── process_action() → 解析动作                              │
│     ├── env.step() → 执行动作，获取奖励                           │
│     └── 如果未完成，编码新观察并追加到 token 序列                   │
│  5. 设置 sample.reward = 50*won + goal_success - illegal         │
│  6. 释放 worker 回池                                             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ GRPO/GSPO 训练   │
└─────────────────┘
```

### 2. 环境池架构

```
┌─────────────────────────────────────────────────────────────────┐
│ AlfWorldEnvPool (单例)                                          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      ┌──────────┐   │
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │ ...  │ Worker N │   │
│  │ (Ray)    │  │ (Ray)    │  │ (Ray)    │      │ (Ray)    │   │
│  │ THOR     │  │ THOR     │  │ THOR     │      │ THOR     │   │
│  └──────────┘  └──────────┘  └──────────┘      └──────────┘   │
│                                                                  │
│  acquire() ← asyncio.Queue → release()                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 函数接口说明

### 1. alf_utils.py

#### `class AlfEnv(gym.Env)`

ALFWorld 环境的 Gym 封装。

**主要方法**：
- `reset(seed)`: 重置环境，返回 (image, info)
- `step(action_text)`: 执行动作，返回 (image, reward, done, info)
- `get_task_description()`: 获取当前任务描述

#### `process_action(action_text, admissible_commands) -> (str, bool)`

解析模型输出中的动作。

**参数**：
- `action_text`: 模型的 JSON 响应
- `admissible_commands`: 当前可用的动作列表

**返回**：
- 匹配的动作字符串
- 是否合法标志

#### `compute_reward(infos, legal_action) -> float`

计算奖励。

**公式**：`reward = 50 * won + goal_condition_success_rate - 1 * illegal_action`

### 2. prompts.py

#### `get_alfworld_prompt(task, action_history, admissible_actions, action_only) -> str`

生成 ALFWorld 提示词，**必须与 GTR-Turbo 完全一致**。

**参数**：
- `task_description`: 任务描述
- `action_history`: 之前的动作列表
- `admissible_actions`: 当前可用的动作列表
- `action_only`: 是否省略 thoughts 字段

**返回格式**：
```
Your are an expert in the ALFRED Embodied Environment.
Your task is to {task}.
You are also given the previous actions you have taken: {history}.
Your admissible actions of the current situation are: [{actions}].
Your response should be a valid json file in the following format:
{
  "thoughts": "...",
  "action": "{an admissible action}"
}
```

### 3. env_worker.py

#### `class AlfWorldWorker`

Ray actor 封装单个 AlfEnv 实例。

**主要方法**：
- `reset(seed)`: 重置环境
- `step(response_text)`: 执行动作
- `get_reward()`: 获取最终奖励
- `close()`: 关闭环境

### 4. env_pool.py

#### `class AlfWorldEnvPool`

单例异步环境池。

**主要方法**：
- `get_instance(config)`: 获取单例池
- `acquire()`: 获取可用 worker
- `release(worker_id)`: 释放 worker
- `close()`: 关闭所有 workers

---

## 代码层面说明

### 1. 动作解析

```python
# ALFWorld 动作是自然语言命令
# 例如: "pick apple 1", "go to countertop", "open fridge"

# 解析流程：
# 1. 在响应中查找 "action": 字段
# 2. 提取动作文本
# 3. 与 admissible_commands 匹配
# 4. 如果匹配成功，返回该命令
# 5. 否则随机选择一个 admissible 命令
```

### 2. 奖励塑形

```python
# 奖励公式：
reward = 50 * won + goal_condition_success_rate
if not legal_action:
    reward -= 1

# won: 0 或 1（任务是否完全成功）
# goal_condition_success_rate: 0.0 到 1.0（子目标完成率）
# legal_action: 动作是否在 admissible_commands 中
```

### 3. Ray Actor 配置

```yaml
# config.yaml
num_workers: 16
resources_per_worker:
  num_cpus: 2
  num_gpus: 0.25  # THOR 需要部分 GPU 资源
  memory: 4294967296  # 4 GB
```

---

## 配置与使用

### config.yaml

```yaml
max_turns: 40
max_context_len: 16384

num_workers: 16
resources_per_worker:
  num_cpus: 2
  num_gpus: 0.25
  memory: 4294967296

alfworld_config_file: "/root/alfworld/configs/base_config.yaml"
image_size: [300, 300]
```

### 训练命令

```bash
# 基本训练
SLIME_SCRIPT_NUM_GPUS=8 python examples/alfworld/run_alfworld.py

# 使用 WandB
WANDB_API_KEY=your_key python examples/alfworld/run_alfworld.py

# 历史式 rollout
python train.py \
    --hf-checkpoint /root/models/Qwen3-VL-2B-Instruct \
    --custom-generate-function-path examples.alfworld.rollout_history.generate \
    --custom-reward-post-process-path examples.alfworld.rollout_history.post_process_rewards_history \
    --custom-config-path examples/alfworld/config.yaml \
    ...
```

### 环境要求

1. 安装 ALFWorld:
   ```bash
   pip install alfworld
   ```

2. 安装 AI2-THOR:
   ```bash
   pip install ai2thor
   ```

3. 下载 THOR 资源（首次运行时自动下载）

---

## 与 Points24 的主要区别

| 方面 | ALFWorld | Points24 |
|------|----------|----------|
| 环境池 | 需要（16 workers） | 不需要 |
| Ray actor | 使用 | 不使用 |
| 任务来源 | 从 ALFWorld 套件获取 | reset 时随机生成 |
| 提示词 | 包含 admissible actions | 不包含 |
| 动作类型 | 自然语言命令 | 单个 token |
| 奖励范围 | 0-51 | -1 或 10 |
| 资源需求 | 高（THOR + GPU） | 低（纯 Python） |

---

## 参考文件

- GTR-Turbo 源代码：`GTR-Turbo/Turbo_ALF/alf_utils.py`
- GTR-Turbo 提示词：`GTR-Turbo/Turbo_ALF/a2c_ppo_acktr/rl_utils.py`
- ALFWorld 官方文档：https://github.com/alfworld/alfworld
