"""Training entry script for Android World VLM agent RL training.

Usage:
    # FSDP backend (default)
    python examples/android_world/run_android_world.py

    # Megatron backend
    SLIME_SCRIPT_TRAIN_BACKEND=megatron python examples/android_world/run_android_world.py

    # Custom model / GPU count
    SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct SLIME_SCRIPT_NUM_GPUS=8 \
        python examples/android_world/run_android_world.py

Environment variables:
    SLIME_SCRIPT_MODEL_NAME     VLM model name (default: Qwen3-VL-2B-Instruct)
    SLIME_SCRIPT_NUM_GPUS       Number of GPUs (default: 8)
    SLIME_SCRIPT_TRAIN_BACKEND  Training backend: fsdp or megatron (default: fsdp)
    SLIME_SCRIPT_EXTERNAL_RAY   Whether Ray is already running (default: 0)
    SLIME_SCRIPT_TASK_DATA      Path to task JSONL (default: examples/android_world/data/tasks.jsonl)
    WANDB_API_KEY               Weights & Biases API key (optional)
"""

from __future__ import annotations

import os

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
assert MODEL_NAME in {
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Thinking",
}, f"Unsupported model: {MODEL_NAME}"

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
EXTERNAL_RAY = int(os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY", "0"))
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "fsdp").lower()
assert TRAIN_BACKEND in {"fsdp", "megatron"}

TASK_DATA = os.environ.get(
    "SLIME_SCRIPT_TASK_DATA",
    "examples/android_world/data/tasks.jsonl",
)


def get_megatron_model_type(model_name: str) -> str:
    model_type = model_name.replace("-Instruct", "").replace("-Thinking", "")
    model_type = model_type.replace("Qwen3-VL-", "qwen3-")
    return model_type.replace("-2B", "-1.7B")


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project slime-dev "
            "--wandb-group android_world "
            f"--wandb-key '{wandb_api_key}' "
        )
        if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
        else ""
    )

    rollout_args = (
        f"--prompt-data {TASK_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--custom-generate-function-path examples.android_world.rollout.generate "
        "--custom-config-path examples/android_world/config.yaml "
        "--rollout-shuffle "
        "--num-rollout 256 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 "
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
