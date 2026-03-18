"""Training entry script for Points24 VLM agent RL training.

This script configures and launches slime's distributed RL training pipeline
for the Points24 (24 Game) environment.

Usage:
    # FSDP backend (default)
    python examples/points24/run_points24.py

    # Megatron backend
    SLIME_SCRIPT_TRAIN_BACKEND=megatron python examples/points24/run_points24.py

    # Custom model / GPU count
    SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Instruct SLIME_SCRIPT_NUM_GPUS=8 \
        python examples/points24/run_points24.py

    # History-based rollout (alternative mode)
    python train.py \
        --hf-checkpoint /root/models/Qwen3-VL-2B-Instruct \
        --custom-generate-function-path examples.points24.rollout_history.generate \
        --custom-reward-post-process-path examples.points24.rollout_history.post_process_rewards_history \
        --custom-config-path examples/points24/config.yaml \
        ...

Environment variables:
    SLIME_SCRIPT_MODEL_NAME     VLM model name (default: Qwen3-VL-2B-Instruct)
    SLIME_SCRIPT_NUM_GPUS       Number of GPUs (default: 8)
    SLIME_SCRIPT_TRAIN_BACKEND  Training backend: fsdp or megatron (default: fsdp)
    SLIME_SCRIPT_EXTERNAL_RAY   Whether Ray is already running (default: 0)
    WANDB_API_KEY               Weights & Biases API key (optional)
"""

from __future__ import annotations

import os

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train

# Supported models
SUPPORTED_MODELS = {
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Thinking",
}

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
assert MODEL_NAME in SUPPORTED_MODELS, f"Unsupported model: {MODEL_NAME}"

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
EXTERNAL_RAY = int(os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY", "0"))
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "fsdp").lower()
assert TRAIN_BACKEND in {"fsdp", "megatron"}


def get_megatron_model_type(model_name: str) -> str:
    """Convert model name to Megatron model type."""
    model_type = model_name.replace("-Instruct", "").replace("-Thinking", "")
    model_type = model_type.replace("Qwen3-VL-", "qwen3-")
    return model_type.replace("-2B", "-1.7B")


def prepare():
    """Prepare model and datasets."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")


def execute():
    """Execute the training run."""
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project slime-dev "
            "--wandb-group points24 "
            f"--wandb-key '{wandb_api_key}' "
        )
        if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
        else ""
    )

    rollout_args = (
        "--prompt-data examples/points24/data/tasks.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--custom-generate-function-path examples.points24.rollout.generate "
        "--custom-config-path examples/points24/config.yaml "
        "--rollout-shuffle "
        "--num-rollout 256 "
        "--rollout-batch-size 32 "  # Higher batch for lightweight env
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 64 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.35 "
        f"--sglang-cuda-graph-bs {' '.join(map(str, [1, 2, 4, 8] + list(range(16, 129, 8))))} "
    )

    fsdp_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
        "--update-weight-buffer-size 536870912 "
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load /root/models/{MODEL_NAME} "
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} "
        "--colocate "
    )

    if TRAIN_BACKEND == "megatron":
        backend_args = megatron_args
        megatron_model_type = get_megatron_model_type(MODEL_NAME)
        os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    else:
        backend_args = fsdp_args
        megatron_model_type = None

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{backend_args} "
        f"{misc_args} "
        f"{wandb_args} "
    )

    execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=megatron_model_type,
        extra_env_vars=({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
    )


if __name__ == "__main__":
    prepare()
    execute()
