# GTR-Turbo: Merged Checkpoint as Free Teacher for Agentic VLM Training

This example implements [GTR-Turbo](https://arxiv.org/abs/2512.13043) on the slime framework, featuring:

1. **ALFWorld Environment** — Embodied AI household tasks (AI2-THOR)
2. **TIES Checkpoint Merging** — Merge RL checkpoints into a teacher model
3. **GTR-Turbo KL Training** — On-policy distillation with merged teacher

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ GTR-Turbo Training Loop (train_gtr_turbo.py)            │
│                                                         │
│  Epoch 0: GRPO only → save HF checkpoint               │
│  Epoch 1: GRPO only → save → TIES merge → teacher_1    │
│  Epoch 2: GRPO + KL(student, teacher_1) → merge → ...  │
│  ...                                                    │
└──────────┬──────────────────────────────────────────────┘
           │
              ▼
        ┌──────────────┐
        │  ALFWorld    │    Environment
        │  (Ray+THOR)  │
        └──────────────┘
```

## Quick Start

### 1. ALFWorld GRPO Baseline

```bash
python examples/gtr_turbo/alfworld/run_alfworld.py
```

### 2. GTR-Turbo on ALFWorld

```bash
bash examples/gtr_turbo/gtr_turbo_train/run_gtr_turbo_alfworld.sh
```

### 3. Standalone TIES Merge

```bash
python -m examples.gtr_turbo.ties_merge.merge_teacher \
    --base-model /path/to/base \
    --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
    --output /path/to/merged \
    --weighting ema --ema-alpha 0.5 --density 0.8
```

## How It Works

### TIES Merging

TIES (Trim, Elect, Select) merges multiple RL checkpoints:

1. **Trim**: Compute task vectors (checkpoint - base), keep top 80% by magnitude
2. **Elect Sign**: Weighted majority vote for each parameter's sign
3. **Selective Average**: Average only values matching the elected sign

Supports SMA (equal weights) and EMA (recent-biased, α=0.5) weighting.

### GTR-Turbo KL Variant

The merged checkpoint serves as a "free" teacher for on-policy distillation:

- Standard OPD reward function fetches teacher log-probs via SGLang
- Custom `post_process_rewards` returns **environment task rewards** (not 0.0)
- KL penalty: `advantage' = advantage - β * (student_logp - teacher_logp)`
- Teacher is periodically updated with newly merged checkpoints

### Key CLI Arguments

| Argument | Purpose |
|---|---|
| `--custom-generate-function-path` | Multi-turn rollout function |
| `--rollout-interaction-env-path` | Environment module |
| `--custom-config-path` | Environment YAML config |
| `--custom-rm-path` | Reward function |
| `--custom-reward-post-process-path` | GTR-Turbo reward post-processor |
| `--use-opd --opd-type sglang` | Enable OPD with teacher server |
| `--opd-kl-coef` | KL penalty weight (paper: β=1.0) |
| `--save-hf` | Save HF checkpoints for TIES merging |

## Configuration

### GTR-Turbo hyperparameters (`gtr_turbo_train/config.yaml`)

```yaml
ties_density: 0.8     # TIES trimming density
weighting: sma        # "sma" or "ema"
ema_alpha: 0.5        # EMA decay factor
opd_kl_coef: 1.0      # KL penalty weight (β)
num_epochs: 30        # Training epochs
merge_interval: 1     # Merge every N epochs
n_samples_per_prompt: 32  # GRPO group size
lr: 1.0e-5
lr_decay_style: cosine
min_lr: 1.0e-7
```

### ALFWorld (`alfworld/config.yaml`)

- `max_turns: 40`, `max_context_len: 16384`
- 14 actions, reward: `50*won + goal_rate - illegal`
- Requires AI2-THOR and `alfworld` package

## Directory Structure

```
examples/gtr_turbo/
├── ties_merge/           # TIES checkpoint merging
│   ├── ties_merging.py   # Core algorithm
│   ├── checkpoint_buffer.py  # SMA/EMA buffer
│   └── merge_teacher.py  # CLI merge tool
├── alfworld/             # Embodied AI tasks
│   ├── env_alfworld.py   # BaseInteractionEnv wrapper
│   ├── env_worker.py     # Ray remote actor
│   ├── env_pool.py       # Worker pool
│   ├── alf_utils.py      # Utilities
│   ├── prompts.py        # Prompt templates
│   ├── rollout.py        # Incremental rollout
│   ├── rollout_history.py  # History-based rollout
│   ├── reward.py         # Reward extraction
│   ├── config.yaml       # Environment config
│   └── run_alfworld.py   # Training script
└── gtr_turbo_train/      # Training orchestration
    ├── gtr_turbo_reward.py  # OPD + env reward combiner
    ├── train_gtr_turbo.py   # Merge-deploy-train loop
    ├── config.yaml       # GTR-Turbo hyperparams
    └── run_gtr_turbo_alfworld.sh
```

## Reference

```bibtex
@article{wei2024gtr,
  title={GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training},
  author={Wei, et al.},
  journal={arXiv preprint arXiv:2512.13043},
  year={2024}
}
```
