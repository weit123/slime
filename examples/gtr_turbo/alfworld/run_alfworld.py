"""ALFWorld VLM Agent RL Training Script.

Usage:
    python examples/gtr_turbo/alfworld/run_alfworld.py

Environment variables:
    SLIME_SCRIPT_MODEL_NAME: VLM model name or local path (default: /workspace/wt/Qwen3-VL-8B-Instruct if present)
    SLIME_SCRIPT_NUM_GPUS: Number of GPUs (default: 8)
    SLIME_SCRIPT_TRAIN_BACKEND: megatron (default: megatron)
    WANDB_API_KEY: Optional W&B API key
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "megatron")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WORKSPACE = os.environ.get("WORKSPACE", "/workspace/wt")
DEFAULT_LOCAL_MODEL = f"{WORKSPACE}/Qwen3-VL-8B-Instruct"
MODEL_NAME = os.environ.get(
    "SLIME_SCRIPT_MODEL_NAME",
    DEFAULT_LOCAL_MODEL if Path(DEFAULT_LOCAL_MODEL).exists() else "Qwen/Qwen3-VL-8B-Instruct",
)
NUM_ROLLOUT = int(os.environ.get("SLIME_SCRIPT_NUM_ROLLOUT", "128"))
ROLLOUT_BATCH_SIZE = int(os.environ.get("SLIME_SCRIPT_ROLLOUT_BATCH_SIZE", "4"))
N_SAMPLES_PER_PROMPT = int(os.environ.get("SLIME_SCRIPT_N_SAMPLES_PER_PROMPT", "4"))
GLOBAL_BATCH_SIZE = int(os.environ.get("SLIME_SCRIPT_GLOBAL_BATCH_SIZE", "16"))
SAVE_INTERVAL = int(os.environ.get("SLIME_SCRIPT_SAVE_INTERVAL", "50"))
LR_WARMUP_ITERS = int(os.environ.get("SLIME_SCRIPT_LR_WARMUP_ITERS", "0"))
LR_DECAY_STYLE = os.environ.get("SLIME_SCRIPT_LR_DECAY_STYLE", "cosine")
SG_LANG_MEM_FRACTION_STATIC = os.environ.get("SLIME_SCRIPT_SGLANG_MEM_FRACTION_STATIC", "0.70")
MAX_TURNS = int(os.environ.get("SLIME_SCRIPT_MAX_TURNS", "40"))
ROLLOUT_MAX_RESPONSE_LEN = int(os.environ.get("SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN", "4096"))
ROLLOUT_MAX_CONTEXT_LEN = int(os.environ.get("SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN", "16384"))


def prepare():
    model_path = Path(MODEL_NAME).expanduser()
    if model_path.exists():
        return model_path.resolve()

    ckpt_dir = Path(f"{WORKSPACE}/models/{MODEL_NAME.split('/')[-1]}")
    if not ckpt_dir.exists():
        print(f"Downloading {MODEL_NAME}...")
        subprocess.run(
            ["huggingface-cli", "download", MODEL_NAME, "--local-dir", str(ckpt_dir)],
            check=True,
        )
    return ckpt_dir


def execute():
    if TRAIN_BACKEND != "megatron":
        raise ValueError("This gtr_slime checkout only supports SLIME_SCRIPT_TRAIN_BACKEND=megatron.")

    megatron_path = "/root/Megatron-LM"
    os.environ["PYTHONPATH"] = (
        f"{megatron_path}:{os.environ['PYTHONPATH']}"
        if os.environ.get("PYTHONPATH")
        else megatron_path
    )
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    ckpt_dir = prepare()
    repo_root = Path(__file__).resolve().parents[3]
    alfworld_dir = repo_root / "examples/gtr_turbo/alfworld"
    save_dir = Path(f"{WORKSPACE}/outputs/alfworld_{ckpt_dir.name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        f"--hf-checkpoint={ckpt_dir}",
        f"--load={ckpt_dir}",

        "--rollout-function-path=slime.rollout.sglang_rollout.generate_rollout",
        "--custom-generate-function-path=examples.gtr_turbo.alfworld.rollout.generate",
        "--rollout-interaction-env-path=examples.gtr_turbo.alfworld.env_alfworld",
        f"--custom-config-path={alfworld_dir / 'config.yaml'}",
        "--custom-rollout-log-function-path=examples.gtr_turbo.alfworld.metrics.log_rollout",
        "--multimodal-keys={\"image\": \"images\"}",
        f"--num-rollout={NUM_ROLLOUT}",
        f"--rollout-batch-size={ROLLOUT_BATCH_SIZE}",
        f"--n-samples-per-prompt={N_SAMPLES_PER_PROMPT}",
        "--rollout-temperature=1.0",
        "--rollout-top-p=1.0",
        f"--rollout-max-response-len={ROLLOUT_MAX_RESPONSE_LEN}",
        f"--rollout-max-context-len={ROLLOUT_MAX_CONTEXT_LEN}",
        f"--prompt-data={alfworld_dir / 'data/alfworld_prompts.jsonl'}",
        "--input-key=prompt",

        "--custom-rm-path=examples.gtr_turbo.alfworld.reward.reward_func",
        "--custom-reward-post-process-path=examples.gtr_turbo.alfworld.reward.post_process_step_rewards",

        "--advantage-estimator=grpo",
        "--ppo-clip=0.2",
        "--kl-loss-coef=0.00",
        "--entropy-coef=0.00",
        "--use-rollout-logprobs",

        "--lr=1e-6",
        "--lr-warmup-init=1e-7",
        f"--lr-warmup-iters={LR_WARMUP_ITERS}",
        f"--lr-decay-style={LR_DECAY_STYLE}",
        "--min-lr=1e-7",
        "--adam-beta1=0.9",
        "--adam-beta2=0.99",
        "--clip-grad=1.0",
        "--weight-decay=0.01",
        "--bf16",
        f"--global-batch-size={GLOBAL_BATCH_SIZE}",

        f"--rollout-num-gpus={NUM_GPUS}",
        f"--sglang-mem-fraction-static={SG_LANG_MEM_FRACTION_STATIC}",

        f"--max-turns={MAX_TURNS}",

        # Megatron Qwen3-8B architecture args; Qwen3-VL-8B shares the text backbone shape.
        "--swiglu",
        "--num-layers=36",
        "--hidden-size=4096",
        "--ffn-hidden-size=12288",
        "--num-attention-heads=32",
        "--group-query-attention",
        "--num-query-groups=8",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--normalization=RMSNorm",
        "--norm-epsilon=1e-6",
        "--rotary-base=5000000",
        "--vocab-size=151936",
        "--kv-channels=128",
        "--qk-layernorm",
        "--untie-embeddings-and-output-weights",
    ]

    args.extend([
        "--train-backend=megatron",
        "--actor-num-nodes=1",
        f"--actor-num-gpus-per-node={NUM_GPUS}",
        "--tensor-model-parallel-size=2",
        "--sequence-parallel",
        "--megatron-to-hf-mode=bridge",
        "--colocate",
    ])

    if SAVE_INTERVAL > 0:
        args.extend([
            f"--save={save_dir / 'checkpoints'}",
            f"--save-hf={save_dir / 'hf_ckpt_{{rollout_id}}'}",
            f"--save-interval={SAVE_INTERVAL}",
        ])

    if WANDB_API_KEY:
        args.extend([
            "--use-wandb",
            "--wandb-project=alfworld-rl",
            f"--wandb-name=alfworld-{ckpt_dir.name}",
            f"--wandb-key={WANDB_API_KEY}",
        ])

    cmd = [sys.executable, str(Path(__file__).resolve().parents[3] / "train.py")] + args
    print(f"Launching training: {' '.join(cmd[:10])}...")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    execute()
