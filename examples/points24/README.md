# Points24 环境集成文档

本文档详细介绍 slime_android 框架中 Points24（24点游戏）环境的集成实现，包括整体工作流、函数接口和代码层面的说明。

---

## 目录

1. [环境概述](#环境概述)
2. [文件结构](#文件结构)
3. [整体工作流](#整体工作流)
4. [函数接口说明](#函数接口说明)
5. [代码层面说明](#代码层面说明)
6. [配置与使用](#配置与使用)
7. [训练示例](#训练示例)

---

## 环境概述

### 什么是 Points24

Points24（24点游戏）是一个经典的数学益智游戏：
- 给定 4 张扑克牌（1-13，其中 J/Q/K 算作 10）
- 使用运算符 `+`、`-`、`*`、`/` 和括号 `(`、`)`
- 构建一个等于 24 的表达式
- 每个数字只能使用一次

### 环境特点

| 特性 | 说明 |
|------|------|
| 观察空间 | 300x300 RGB 图像 + 文本公式 |
| 动作空间 | 17 个离散动作（1-10, +, -, *, /, (, ), =） |
| 奖励 | +10 成功，-1 失败/非法动作 |
| 最大步数 | 20 步（可配置） |
| 环境类型 | 轻量级纯 Python，无需 Ray 池 |
| 可解性验证 | 每次 reset 验证任务有解 |

---

## 文件结构

```
examples/points24/
├── __init__.py              # 模块初始化
├── env_worker.py            # 轻量级环境包装器
├── env_points24.py          # Per-sample 环境封装（继承 BaseInteractionEnv）
├── rollout.py               # 增量式多轮 rollout
├── rollout_history.py       # 历史式多轮 rollout
├── prompts.py               # 提示词模板（与 GTR-Turbo 一致）
├── config.yaml              # 配置文件
├── run_points24.py          # 训练入口脚本
├── gym_cards/               # 从 GTR-Turbo 复制的环境代码
│   ├── __init__.py
│   └── envs/
│       ├── __init__.py
│       ├── points.py        # Point24Env 核心环境
│       ├── img/             # 扑克牌图片
│       └── font/            # 字体文件
├── data/
│   └── tasks.jsonl          # 占位符任务文件
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
│ tasks.jsonl     │  占位符（实际任务在 reset 时生成）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RolloutDataSource │  加载 Sample 对象
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ generate() 函数                                                  │
│                                                                  │
│  1. 创建 Points24Worker（轻量级，无需池）                         │
│  2. reset() → 生成随机 4 张牌                                    │
│  3. format_observation() → 构建包含图像和提示词的消息              │
│  4. 循环:                                                        │
│     ├── SGLang HTTP 生成 → response_text                        │
│     ├── worker.step(response_text) → 解析动作，执行，获取奖励      │
│     └── 如果未完成，编码新观察并追加到 token 序列                   │
│  5. 设置 sample.reward = 最终奖励                                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Sample 对象      │  包含 tokens, loss_mask, reward 等
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GRPO/GSPO 训练   │  使用 rollout 数据更新模型
└─────────────────┘
```

### 2. 两种 Rollout 模式

#### 增量式 Rollout（rollout.py）

- 一个轨迹生成一个 Sample
- Token 序列逐轮增长
- `loss_mask=1` 用于模型输出，`loss_mask=0` 用于环境观察

```
Turn 0: [Prompt0] → [Response0]
Turn 1: [Prompt0][Response0][Obs1] → [Response1]
Turn 2: [Prompt0][Response0][Obs1][Response1][Obs2] → [Response2]
...
```

#### 历史式 Rollout（rollout_history.py）

- 每轮生成独立的 Sample
- 每轮包含完整历史（通过 prompt 重建）
- 兼容标准 VLM RL 方法

```
Turn 0: [Prompt_with_history_0] → [Response0] → Sample0
Turn 1: [Prompt_with_history_1] → [Response1] → Sample1
Turn 2: [Prompt_with_history_2] → [Response2] → Sample2
...
```

---

## 函数接口说明

### 1. prompts.py

#### `get_points24_prompt(formula: list, action_only: bool = False) -> str`

生成 Points24 环境的提示词，**必须与 GTR-Turbo 完全一致**。

**参数**：
- `formula`: 当前公式列表，如 `['8', '*', '3']`
- `action_only`: 如果为 True，省略 "thoughts" 字段

**返回**：
- 提示词字符串，格式如下：

```
You are an expert 24 points card game player. You are observing four cards in the image.
You are observing the current formula: 8*3.
You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '='].
...
Your response should **only contain a valid json file** in the following format:
{
  "cards": ["x", "y", "z", "w"],
  "formula": "8*3",
  "thoughts": "...",
  "action": "number or operator"
}
```

#### `text_projection_points24(text_actions: List[str]) -> Tuple[Tensor, int]`

将模型文本输出解析为离散动作索引。

**参数**：
- `text_actions`: 模型响应文本列表

**返回**：
- 动作索引张量
- 合法动作标志（0 或 1）

#### `parse_single_action(response_text: str) -> Tuple[int, bool]`

单样本动作解析的简化包装器。

### 2. env_worker.py

#### `class Points24Worker`

轻量级环境包装器，不使用 Ray actor。

**主要方法**：

| 方法 | 说明 |
|------|------|
| `reset(seed=None)` | 重置环境，返回 (observation_dict, info_dict) |
| `step(response_text)` | 执行动作，返回 (obs, reward, done, info) |
| `get_reward()` | 获取最终奖励 |
| `close()` | 关闭环境 |

**observation_dict 结构**：
```python
{
    "image": PIL.Image,  # 300x300 游戏截图
    "cards": ["H8", "S3", "D3", "C1"],  # 卡牌名称
    "formula": ["8", "*", "3"],  # 当前公式
    "numbers": [8, 3, 3, 1],  # 卡牌数值
}
```

### 3. env_points24.py

#### `class Points24Env(BaseInteractionEnv)`

Per-sample 环境封装，实现 slime 的 BaseInteractionEnv 接口。

**主要方法**：

| 方法 | 说明 |
|------|------|
| `async reset()` | 异步重置环境 |
| `async step(response_text)` | 异步执行一步 |
| `format_observation(obs, is_initial)` | 格式化观察为 VLM 消息格式 |
| `get_reward()` | 获取最终奖励 |
| `close()` | 关闭环境 |

**format_observation 返回格式**：
```python
{
    "role": "user",
    "content": [
        {"type": "image", "image": <PIL.Image>},
        {"type": "text", "text": "<GTR-Turbo prompt text>"},
    ]
}
```

### 4. rollout.py

#### `async generate(args, sample, sampling_params) -> Sample`

增量式多轮 rollout 的入口函数。

**参数**：
- `args`: 解析后的 slime 参数
- `sample`: 输入 Sample 对象
- `sampling_params`: SGLang 采样参数

**返回**：
- 完成的 Sample 对象，包含 tokens, loss_mask, reward 等

### 5. rollout_history.py

#### `async generate(args, sample, sampling_params, evaluation=False) -> List[Sample] | Sample`

历史式多轮 rollout 的入口函数。

**参数**：
- 同上，额外有 `evaluation` 标志

**返回**：
- 训练时：Sample 列表（每轮一个）
- 评估时：单个 Sample

#### `post_process_rewards_history(args, samples) -> Tuple[List[float], List[float]]`

GRPO 奖励归一化，正确处理变长轨迹。

---

## 代码层面说明

### 1. 动作解析逻辑

```python
# 动作列表（索引 0-16）
ACTION_LIST = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
               "+", "-", "*", "/", "(", ")", "="]

# 解析流程：
# 1. 查找 "action": 字段
# 2. 提取包含的动作
# 3. 如果只有一个合法动作，返回其索引
# 4. 否则随机选择一个动作
```

**特殊处理**：
- `'10'` 需要特殊处理，避免被误识别为 `'1'`
- 不区分大小写
- 支持部分 JSON 格式容错

### 2. 奖励计算

```python
# 在 Point24Env.step() 中：
if action == '=':
    try:
        result = eval(formula_str)
        if result == 24 and all_cards_used:
            reward = 10  # 成功
        else:
            reward = -1  # 失败
    except:
        reward = -1  # 公式无效
```

### 3. 任务可解性验证

每次 reset 时，系统会验证当前生成的 4 张牌是否可以通过运算得到 24。

使用穷举搜索算法，尝试所有可能的：
- 数字排列组合
- 运算符组合（+, -, *, /）
- 括号插入位置（包括双层括号）

```python
def is_solvable_24(digits: list[int], target: float = 24.0) -> bool:
    """检查 4 个数字是否可以组成等于 target 的表达式。

    参考：
    - https://github.com/LeslieTrue/SFTvsRL/blob/master/gym/gym_cards/envs/general_points_oneline.py
    - https://rosettacode.org/wiki/24_game/Solve#Python
    """
    from fractions import Fraction as F

    # 所有数字排列
    digiperm = sorted(set(permutations(digits)))
    random.shuffle(digiperm)

    # 所有运算符组合
    opcomb = list(product('+-*/', repeat=len(digits)-1))

    # 所有括号插入点（包括双层括号）
    brackets = ([()] + [(x, y)
                for x in range(0, exprlen, 2)
                for y in range(x+4, exprlen+2, 2)
                if (x, y) != (0, exprlen+1)]
                + [(0, 3+1, 4+2, 7+3)])  # 双层括号

    for d in digiperm:
        for ops in opcomb:
            # 使用 Fraction 处理除法精度问题
            # 构建表达式并求值...
            if num == target:
                return True
    return False
```

如果生成的任务无解，系统会自动重新生成，最多尝试 100 次。

### 4. Token 管理

增量式 rollout 的 token 管理：

```python
# 初始 prompt
sample.tokens = list(prompt_ids)
response_tokens = []

# 多轮循环中：
# 1. 模型生成 → loss_mask=1
_append_to_sample(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1)

# 2. 环境观察 → loss_mask=0
_append_to_sample(sample, response_tokens, obs_prompt_ids, [0.0]*len(obs_prompt_ids), loss_mask_val=0)
```

---

## 配置与使用

### config.yaml

```yaml
# 多轮交互设置
max_turns: 30
max_context_len: 8192
history_window_size: 10

# Points24 环境设置
target_points: 24
treat_face_cards_as_10: true
action_only: false

# 图像设置
image_size:
  - 300
  - 300
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SLIME_SCRIPT_MODEL_NAME` | VLM 模型名称 | `Qwen3-VL-2B-Instruct` |
| `SLIME_SCRIPT_NUM_GPUS` | GPU 数量 | `8` |
| `SLIME_SCRIPT_TRAIN_BACKEND` | 训练后端 | `fsdp` |
| `WANDB_API_KEY` | WandB API 密钥 | - |

---

## 训练示例

### 1. 使用增量式 Rollout

```bash
# 基本训练
SLIME_SCRIPT_NUM_GPUS=8 python examples/points24/run_points24.py

# 使用 WandB 日志
WANDB_API_KEY=your_key \
SLIME_SCRIPT_NUM_GPUS=8 \
python examples/points24/run_points24.py

# 使用 Megatron 后端
SLIME_SCRIPT_TRAIN_BACKEND=megatron \
SLIME_SCRIPT_NUM_GPUS=8 \
python examples/points24/run_points24.py
```

### 2. 使用历史式 Rollout

```bash
python train.py \
    --hf-checkpoint /root/models/Qwen3-VL-2B-Instruct \
    --prompt-data examples/points24/data/tasks.jsonl \
    --input-key prompt \
    --apply-chat-template \
    --custom-generate-function-path examples.points24.rollout_history.generate \
    --custom-reward-post-process-path examples.points24.rollout_history.post_process_rewards_history \
    --custom-config-path examples/points24/config.yaml \
    --rollout-shuffle \
    --num-rollout 256 \
    --n-samples-per-prompt 8 \
    --rollout-max-response-len 2048 \
    --rollout-temperature 1.0 \
    --advantage-estimator grpo \
    --train-backend fsdp \
    --colocate \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8
```

### 3. 评估模式

```bash
python -m slime.eval \
    --hf-checkpoint /root/models/Qwen3-VL-2B-Instruct \
    --custom-generate-function-path examples.points24.rollout_history.generate \
    --custom-config-path examples/points24/config.yaml \
    --eval-on-the-fly
```

### 4. 单元测试

```bash
# 运行所有测试
pytest examples/points24/tests/ -v

# 运行特定测试
pytest examples/points24/tests/test_rollout.py::test_prompt_generation -v
```

---

## 与 android_world 的主要区别

| 方面 | Points24 | android_world |
|------|----------|---------------|
| 环境池 | 不需要（轻量级） | 需要（128 workers） |
| Ray actor | 不使用 | 使用 |
| 任务来源 | reset 时随机生成 | 从 JSONL 加载 |
| 提示词 | 包含在 user message 中 | 单独的 system prompt |
| 观察格式 | 图像 + 文本公式 | 图像 + 任务描述 |

---

## 参考文件

- GTR-Turbo 源代码：`GTR-Turbo/gym-cards/gym_cards/envs/points.py`
- GTR-Turbo 提示词：`GTR-Turbo/Turbo_P24/a2c_ppo_acktr/rl_utils.py`
- slime_android 参考实现：`examples/android_world/`
