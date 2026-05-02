"""Points24 VLM Agent RL Training Script.

Configures and launches distributed RL training for Points24 using slime.
Supports FSDP and Megatron backends with Qwen3-VL models.

Usage:
    python examples/gtr_turbo/points24/run_points24.py

Environment variables:
    SLIME_SCRIPT_MODEL_NAME: VLM model name (default: Qwen/Qwen3-VL-2B-Instruct)
    SLIME_SCRIPT_NUM_GPUS: Number of GPUs (default: 8)
    SLIME_SCRIPT_TRAIN_BACKEND: Training backend, fsdp or megatron (default: fsdp)
    WANDB_API_KEY: Optional W&B API key for experiment tracking
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SUPPORTED_MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "fsdp")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")


def prepare():
    """Download model and create directories."""
    ckpt_dir = Path(f"/root/models/{MODEL_NAME.split('/')[-1]}")
    if not ckpt_dir.exists():
        print(f"Downloading {MODEL_NAME}...")
        subprocess.run(
            ["huggingface-cli", "download", MODEL_NAME, "--local-dir", str(ckpt_dir)],
            check=True,
        )
    return ckpt_dir


def execute():
    ckpt_dir = prepare()
    save_dir = Path(f"/root/outputs/points24_{ckpt_dir.name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        # Checkpoint
        f"--hf-checkpoint={ckpt_dir}",
        f"--save={save_dir / 'checkpoints'}",
        f"--save-hf={save_dir / 'hf_ckpt_{{rollout_id}}'}",
        "--save-interval=50",

        # Rollout
        "--rollout-function-path=slime.rollout.sglang_rollout.generate_rollout",
        "--custom-generate-function-path=examples.gtr_turbo.points24.rollout.generate",
        "--rollout-interaction-env-path=examples.gtr_turbo.points24.env_points24",
        "--custom-config-path=examples/gtr_turbo/points24/config.yaml",
        "--num-rollout=256",
        "--rollout-batch-size=4",
        "--n-samples-per-prompt=4",
        "--rollout-temperature=1.0",
        "--rollout-top-p=1.0",
        "--rollout-max-response-len=2048",
        "--rollout-max-context-len=8192",
        "--prompt-data=/root/data/points24_prompts.jsonl",

        # Reward
        "--custom-rm-path=examples.gtr_turbo.points24.reward.reward_func",

        # GRPO advantage
        "--advantage-estimator=grpo",
        "--ppo-clip=0.2",
        "--kl-loss-coef=0.00",
        "--entropy-coef=0.00",

        # Optimizer
        "--lr=1e-6",
        "--lr-warmup-init=1e-7",
        "--lr-warmup-iters=10",
        "--lr-decay-style=cosine",
        "--min-lr=1e-7",
        "--adam-beta1=0.9",
        "--adam-beta2=0.99",
        "--adam-eps=1e-8",
        "--clip-grad=1.0",
        "--weight-decay=0.01",
        "--bf16",
        "--global-batch-size=16",

        # SGLang
        f"--rollout-num-gpus={NUM_GPUS}",
        "--sglang-mem-fraction-static=0.75",

        # Multi-turn
        f"--max-turns=20",
    ]

    if TRAIN_BACKEND == "fsdp":
        args.extend([
            "--train-backend=fsdp",
            "--gradient-checkpointing",
            "--use-flash-attn-3",
            f"--actor-num-nodes=1",
            f"--actor-num-gpus-per-node={NUM_GPUS}",
            f"--colocate",
        ])
    else:
        args.extend([
            "--train-backend=megatron",
            f"--actor-num-nodes=1",
            f"--actor-num-gpus-per-node={NUM_GPUS}",
            "--tensor-model-parallel-size=2",
            "--sequence-parallel",
            f"--colocate",
        ])

    if WANDB_API_KEY:
        args.extend([
            "--wandb-project=points24-rl",
            f"--wandb-name=points24-{ckpt_dir.name}",
        ])

    cmd = [sys.executable, "-m", "slime.launch", "train"] + args
    print(f"Launching training: {' '.join(cmd[:10])}...")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    execute()
