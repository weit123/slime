"""Training entry script for ALFWorld VLM agent RL training.

Usage:
    # FSDP backend (default)
    python examples/alfworld/run_alfworld.py

    # With WandB
    WANDB_API_KEY=your_key SLIME_SCRIPT_NUM_GPUS=8 python examples/alfworld/run_alfworld.py

Environment variables:
    SLIME_SCRIPT_MODEL_NAME     VLM model name (default: Qwen3-VL-2B-Instruct)
    SLIME_SCRIPT_NUM_GPUS       Number of GPUs (default: 8)
    SLIME_SCRIPT_TRAIN_BACKEND  Training backend: fsdp or megatron (default: fsdp)
    WANDB_API_KEY               Weights & Biases API key (optional)
"""

from __future__ import annotations

import os

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "fsdp").lower()
ALFWORLD_CONFIG = os.environ.get("SLIME_SCRIPT_ALFWORLD_CONFIG", "base_config.yaml")


def prepare():
    """Prepare model and ALFWorld dependencies."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")


def execute():
    """Execute the training run."""
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    wandb_args = (
        f"--use-wandb --wandb-project slime-dev --wandb-group alfworld --wandb-key '{os.environ.get('WANDB_API_KEY')}' "
        if os.environ.get("WANDB_API_KEY") else ""
    )

    rollout_args = (
        "--prompt-data examples/alfworld/data/tasks.jsonl "
        "--input-key prompt "
        "--apply-chat-template "
        "--custom-generate-function-path examples.alfworld.rollout.generate "
        "--custom-config-path examples/alfworld/config.yaml "
        "--rollout-shuffle "
        "--num-rollout 128 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    grpo_args = "--advantage-estimator grpo --kl-loss-coef 0.00 --eps-clip 0.2 "

    optimizer_args = "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 "

    fsdp_args = "--train-backend fsdp --gradient-checkpointing --sglang-attention-backend fa3 "

    misc_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --rollout-num-gpus {NUM_GPUS} --colocate "

    train_args = f"{ckpt_args}{rollout_args}{optimizer_args}{grpo_args}{sglang_args}{fsdp_args}{misc_args}{wandb_args}"

    execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS)


if __name__ == "__main__":
    prepare()
    execute()
