"""ALFWorld VLM Agent GRPO baseline/smoke training script.

For the full GTR-Turbo algorithm on ALFWorld, use:
    bash examples/gtr_turbo/gtr_turbo_train/run_gtr_turbo_alfworld.sh

Usage:
    python examples/gtr_turbo/alfworld/run_alfworld.py

Environment variables:
    SLIME_SCRIPT_MODEL_NAME: VLM model name or local path (default: /workspace/wt/Qwen3-VL-8B-Instruct if present)
    SLIME_SCRIPT_VISIBLE_DEVICES: CUDA devices for baseline training (default: 0,1,2,3,4,5,6,7)
    SLIME_SCRIPT_NUM_GPUS: Number of visible GPUs (default: len(SLIME_SCRIPT_VISIBLE_DEVICES))
    SLIME_SCRIPT_TRAIN_BACKEND: megatron (default: megatron)
    SLIME_SCRIPT_USE_WANDB: 1/0 to enable W&B logging (default: 1)
    SLIME_SCRIPT_BACKGROUND: 1/0 to detach training and write logs under run_notes (default: 1)
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "megatron")
WORKSPACE = os.environ.get("WORKSPACE", "/workspace/wt")
DEFAULT_LOCAL_MODEL = f"{WORKSPACE}/Qwen3-VL-8B-Instruct"
MODEL_NAME = os.environ.get(
    "SLIME_SCRIPT_MODEL_NAME",
    DEFAULT_LOCAL_MODEL if Path(DEFAULT_LOCAL_MODEL).exists() else "Qwen/Qwen3-VL-8B-Instruct",
)
VISIBLE_DEVICES = os.environ.get("SLIME_SCRIPT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", str(len([d for d in VISIBLE_DEVICES.split(",") if d]))))
TENSOR_MODEL_PARALLEL_SIZE = int(os.environ.get("SLIME_SCRIPT_TENSOR_MODEL_PARALLEL_SIZE", "2"))
NUM_EPOCHS = int(os.environ.get("SLIME_SCRIPT_NUM_EPOCHS", "18"))
ROLLOUTS_PER_EPOCH = int(os.environ.get("SLIME_SCRIPT_ROLLOUTS_PER_EPOCH", "1"))
NUM_ROLLOUT = int(os.environ.get("SLIME_SCRIPT_NUM_ROLLOUT", str(NUM_EPOCHS * ROLLOUTS_PER_EPOCH)))
ROLLOUT_BATCH_SIZE = int(os.environ.get("SLIME_SCRIPT_ROLLOUT_BATCH_SIZE", "16"))
N_SAMPLES_PER_PROMPT = int(os.environ.get("SLIME_SCRIPT_N_SAMPLES_PER_PROMPT", "8"))
OVER_SAMPLING_BATCH_SIZE = int(
    os.environ.get("SLIME_SCRIPT_OVER_SAMPLING_BATCH_SIZE", str(ROLLOUT_BATCH_SIZE * 2))
)
GLOBAL_BATCH_SIZE = int(os.environ.get("SLIME_SCRIPT_GLOBAL_BATCH_SIZE", "128"))
USE_DYNAMIC_GLOBAL_BATCH_SIZE = os.environ.get(
    "SLIME_SCRIPT_USE_DYNAMIC_GLOBAL_BATCH_SIZE", "1"
).lower() in {"1", "true", "yes"}
SAVE_INTERVAL = int(os.environ.get("SLIME_SCRIPT_SAVE_INTERVAL", "3"))
SAVE_HF = os.environ.get("SLIME_SCRIPT_SAVE_HF", "1").lower() in {"1", "true", "yes"}
LR = os.environ.get("SLIME_SCRIPT_LR", "1.0e-6")
MIN_LR = os.environ.get("SLIME_SCRIPT_MIN_LR", "1.0e-7")
LR_WARMUP_ITERS = int(os.environ.get("SLIME_SCRIPT_LR_WARMUP_ITERS", "3"))
LR_DECAY_STYLE = os.environ.get("SLIME_SCRIPT_LR_DECAY_STYLE", "constant")
SG_LANG_MEM_FRACTION_STATIC = os.environ.get("SLIME_SCRIPT_SGLANG_MEM_FRACTION_STATIC", "0.82")
SGLANG_SERVER_CONCURRENCY = int(os.environ.get("SLIME_SCRIPT_SGLANG_SERVER_CONCURRENCY", "256"))
SGLANG_CUDA_GRAPH_BS = os.environ.get(
    "SLIME_SCRIPT_SGLANG_CUDA_GRAPH_BS",
    "1 2 4 8 16 24 32",
)
MAX_TURNS = int(os.environ.get("SLIME_SCRIPT_MAX_TURNS", "40"))
ROLLOUT_TEMPERATURE = os.environ.get("SLIME_SCRIPT_ROLLOUT_TEMPERATURE", "0.7")
ROLLOUT_MAX_RESPONSE_LEN = int(os.environ.get("SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN", "1024"))
ROLLOUT_MAX_CONTEXT_LEN = int(os.environ.get("SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN", "16384"))
SEED = int(os.environ.get("SLIME_SCRIPT_SEED", "42"))
RUN_TAG = os.environ.get("SLIME_SCRIPT_RUN_TAG", "grpo")
SAVE_DIR_BASE = os.environ.get("SLIME_SCRIPT_SAVE_DIR", f"{WORKSPACE}/gtr_runs")
EVAL_DATA_PATH = os.environ.get("SLIME_SCRIPT_EVAL_DATA_PATH", "/root/.cache/alfworld/json_2.1.1/valid_seen")
EVAL_NUM_TASKS = int(os.environ.get("SLIME_SCRIPT_EVAL_NUM_TASKS", "0"))
EVAL_SEED = int(os.environ.get("SLIME_SCRIPT_EVAL_SEED", "42"))
EVAL_GEN_WORKERS = int(os.environ.get("SLIME_SCRIPT_EVAL_GEN_WORKERS", "32"))
EVAL_PROMPT_DATA = os.environ.get(
    "SLIME_SCRIPT_EVAL_PROMPT_DATA",
    "examples/gtr_turbo/alfworld/data/valid_seen_all_prompts.jsonl",
)
PROMPT_DATA = os.environ.get(
    "SLIME_SCRIPT_PROMPT_DATA",
    "examples/gtr_turbo/alfworld/data/alfworld_oracle_train_prompts.jsonl",
)
USE_WANDB = os.environ.get("SLIME_SCRIPT_USE_WANDB", "1").lower() not in {"0", "false", "no"}
WANDB_KEY = os.environ.get(
    "SLIME_SCRIPT_WANDB_KEY",
    os.environ.get("WANDB_API_KEY", "local-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"),
)
WANDB_HOST = os.environ.get("SLIME_SCRIPT_WANDB_HOST", "https://wandb.glm.ai")
WANDB_PROJECT = os.environ.get("SLIME_SCRIPT_WANDB_PROJECT", "gtr")
BACKGROUND = os.environ.get("SLIME_SCRIPT_BACKGROUND", "1").lower() not in {"0", "false", "no"}


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


def ensure_eval_set(alfworld_dir: Path) -> Path:
    repo_root = alfworld_dir.parents[2]
    eval_path = Path(EVAL_PROMPT_DATA)
    if not eval_path.is_absolute():
        eval_path = repo_root / eval_path
    if eval_path.exists():
        return eval_path
    if EVAL_NUM_TASKS <= 0:
        subprocess.run(
            [
                sys.executable,
                str(alfworld_dir / "data/gen_task_dataset.py"),
                "--data-path",
                EVAL_DATA_PATH,
                "--output",
                str(eval_path),
                "--prompt-content",
                "Begin ALFWorld eval task.",
                "--id-prefix",
                "alfworld_valid_seen",
                "--metadata-keys",
                "task_file",
                "task_type",
                "category",
                "--include-label",
            ],
            check=True,
        )
        return eval_path
    subprocess.run(
        [
            sys.executable,
            str(alfworld_dir / "data/gen_eval_valid_seen.py"),
            "--data-path",
            EVAL_DATA_PATH,
            "--config",
            str(alfworld_dir / "config.yaml"),
            "--output",
            str(eval_path),
            "--seed",
            str(EVAL_SEED),
            "--num-workers",
            str(EVAL_GEN_WORKERS),
            "--num-tasks",
            str(EVAL_NUM_TASKS),
        ],
        check=True,
    )
    return eval_path


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
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", VISIBLE_DEVICES)
    os.environ.setdefault("SLIME_SCRIPT_NUM_GPUS", str(NUM_GPUS))

    ckpt_dir = prepare()
    repo_root = Path(__file__).resolve().parents[3]
    alfworld_dir = repo_root / "examples/gtr_turbo/alfworld"
    eval_path = ensure_eval_set(alfworld_dir)
    model_short = ckpt_dir.name.lower().replace("-", "").replace("_", "").replace("instruct", "")
    env_short = "alf"
    run_name = f"{RUN_TAG}_{env_short}_{model_short}_seed{SEED}"
    wandb_group = f"{RUN_TAG}_{env_short}_{model_short}"
    save_dir = Path(SAVE_DIR_BASE) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_output_dir = save_dir / "rollout_debug"
    base_config_path = alfworld_dir / "config.yaml"
    runtime_config_path = save_dir / "alfworld_runtime_config.yaml"
    runtime_config_text = base_config_path.read_text(encoding="utf-8").rstrip()
    runtime_config_path.write_text(
        runtime_config_text
        + "\n\n"
        + "# Per-run debug output settings.\n"
        + f"debug_output_log_dir: \"{debug_output_dir}\"\n"
        + "debug_output_sample_rate: 0.02\n",
        encoding="utf-8",
    )

    args = [
        f"--hf-checkpoint={ckpt_dir}",
        f"--load={ckpt_dir}",
        "--megatron-to-hf-mode=bridge",

        f"--prompt-data={Path(PROMPT_DATA) if Path(PROMPT_DATA).is_absolute() else repo_root / PROMPT_DATA}",
        "--input-key=prompt",
        "--rollout-function-path=slime.rollout.sglang_rollout.generate_rollout",
        "--custom-generate-function-path=examples.gtr_turbo.alfworld.rollout.generate",
        "--rollout-interaction-env-path=examples.gtr_turbo.alfworld.env_alfworld",
        f"--custom-config-path={runtime_config_path}",
        "--custom-rollout-log-function-path=examples.gtr_turbo.alfworld.metrics.log_rollout",
        "--multimodal-keys={\"image\": \"images\"}",
        f"--num-rollout={NUM_ROLLOUT}",
        f"--rollout-batch-size={ROLLOUT_BATCH_SIZE}",
        f"--over-sampling-batch-size={OVER_SAMPLING_BATCH_SIZE}",
        f"--n-samples-per-prompt={N_SAMPLES_PER_PROMPT}",
        f"--rollout-temperature={ROLLOUT_TEMPERATURE}",
        f"--rollout-max-response-len={ROLLOUT_MAX_RESPONSE_LEN}",
        f"--rollout-max-context-len={ROLLOUT_MAX_CONTEXT_LEN}",
        "--balance-data",

        "--custom-rm-path=examples.gtr_turbo.alfworld.reward.reward_func",
        "--custom-reward-post-process-path=examples.gtr_turbo.alfworld.reward.post_process_step_rewards",
        "--dynamic-sampling-filter-path=examples.gtr_turbo.alfworld.reward.check_reward_nonzero_std",

        "--advantage-estimator=grpo",
        "--eps-clip=0.2",
        "--eps-clip-high=0.28",
        "--kl-loss-coef=0.00",
        "--entropy-coef=0.00",
        "--use-tis",

        f"--lr={LR}",
        "--lr-warmup-init=1e-7",
        f"--lr-warmup-iters={LR_WARMUP_ITERS}",
        f"--lr-decay-style={LR_DECAY_STYLE}",
        f"--min-lr={MIN_LR}",
        "--adam-beta1=0.9",
        "--adam-beta2=0.99",
        "--clip-grad=1.0",
        "--weight-decay=0.01",
        "--bf16",
        f"--global-batch-size={GLOBAL_BATCH_SIZE}",

        f"--rollout-num-gpus={NUM_GPUS}",
        "--rollout-num-gpus-per-engine=1",
        f"--sglang-mem-fraction-static={SG_LANG_MEM_FRACTION_STATIC}",
        f"--sglang-server-concurrency={SGLANG_SERVER_CONCURRENCY}",
        "--sglang-cuda-graph-bs",
        *SGLANG_CUDA_GRAPH_BS.split(),

        f"--max-turns={MAX_TURNS}",

        "--eval-interval=1",
        "--skip-eval-before-train",
        f"--eval-config={save_dir / 'eval_valid_seen.yaml'}",
        "--n-samples-per-eval-prompt=1",
        "--eval-temperature=0.0",
        "--custom-eval-rollout-log-function-path=examples.gtr_turbo.alfworld.metrics.log_eval_rollout",

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

    eval_config_path = save_dir / "eval_valid_seen.yaml"
    eval_config_path.write_text(
        "eval:\n"
        "  defaults:\n"
        "    input_key: prompt\n"
        "    label_key: null\n"
        "    n_samples_per_eval_prompt: 1\n"
        "    temperature: 0.0\n"
        "    top_p: 1.0\n"
        f"    max_response_len: {ROLLOUT_MAX_RESPONSE_LEN}\n"
        "  datasets:\n"
        "    - name: alfworld_valid_seen\n"
        f"      path: {eval_path}\n"
        "      custom_generate_function_path: examples.gtr_turbo.alfworld.eval_rollout.generate\n",
        encoding="utf-8",
    )

    args.extend([
        "--train-backend=megatron",
        "--gradient-checkpointing",
        "--actor-num-nodes=1",
        f"--actor-num-gpus-per-node={NUM_GPUS}",
        f"--num-gpus-per-node={NUM_GPUS}",
        f"--tensor-model-parallel-size={TENSOR_MODEL_PARALLEL_SIZE}",
        "--rollout-shuffle",
        "--colocate",

        f"--seed={SEED}",
        f"--rollout-seed={SEED}",
    ])

    if USE_DYNAMIC_GLOBAL_BATCH_SIZE:
        args.append("--use-dynamic-global-batch-size")

    if SAVE_HF and SAVE_INTERVAL > 0:
        args.extend([
            f"--save-interval={SAVE_INTERVAL}",
            "--save-final-only",
            f"--save-hf={save_dir / 'hf_ckpt_{rollout_id}'}",
        ])

    if USE_WANDB:
        os.environ["WANDB_NAME"] = run_name
        os.environ["WANDB_TAGS"] = f"{RUN_TAG},{env_short},{model_short},seed{SEED}"
        args.extend([
            "--use-wandb",
            "--disable-wandb-random-suffix",
            f"--wandb-key={WANDB_KEY}",
            f"--wandb-host={WANDB_HOST}",
            f"--wandb-project={WANDB_PROJECT}",
            f"--wandb-group={wandb_group}",
        ])

    cmd = [sys.executable, str(Path(__file__).resolve().parents[3] / "train.py")] + args
    print(f"Launching training: {' '.join(cmd[:10])}...")
    if not BACKGROUND:
        os.execvp(cmd[0], cmd)

    run_notes_dir = alfworld_dir / "run_notes"
    run_notes_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = run_notes_dir / f"{run_name}_bg_{timestamp}.log"
    cmd_path = run_notes_dir / f"{run_name}_bg_{timestamp}.cmd"
    pid_path = run_notes_dir / f"{run_name}_bg_{timestamp}.pid"

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("ALFWORLD_ROLLOUT_DEBUG_DIR", str(debug_output_dir))
    env.setdefault("ALFWORLD_ROLLOUT_DEBUG_SAMPLE_RATE", "0.02")
    cmd_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    log_file = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"Started background training PID {proc.pid}")
    print(f"Log: {log_path}")
    print(f"Command: {cmd_path}")
    print(f"PID file: {pid_path}")


if __name__ == "__main__":
    execute()
