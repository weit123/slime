"""GTR-Turbo training orchestrator.

Manages the outer training loop:
  1. Run slime RL training for one epoch (saving HF checkpoints)
  2. Collect saved checkpoints into the TIES buffer
  3. Merge checkpoints via TIES to create a teacher
  4. Restart the SGLang teacher server with merged weights
  5. Repeat

Usage:
    python -m examples.gtr_turbo.gtr_turbo_train.train_gtr_turbo \
        --config examples/gtr_turbo/gtr_turbo_train/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def start_teacher_server(model_path: str, port: int, tp_size: int = 1, gpu_id: int = 0) -> subprocess.Popen:
    """Start an SGLang server as the OPD teacher."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["no_proxy"] = "localhost,127.0.0.1"
    env["NO_PROXY"] = "localhost,127.0.0.1"

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp-size", str(tp_size),
        "--trust-remote-code",
    ]
    logger.info("Starting teacher server: %s", " ".join(cmd))
    teacher_log = open(f"{os.environ.get('SAVE_DIR', '/tmp')}/teacher_server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=teacher_log, stderr=teacher_log)

    _wait_for_server(port, timeout=300)
    logger.info("Teacher server ready on port %d (PID %d)", port, proc.pid)
    return proc


def _wait_for_server(port: int, timeout: int = 300) -> None:
    """Wait until the SGLang server is responsive."""
    import urllib.request

    # Bypass proxy for localhost
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/health",
                method="GET",
            )
            opener.open(req, timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"Teacher server on port {port} did not start within {timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop a server process."""
    if proc and proc.poll() is None:
        logger.info("Stopping server PID %d", proc.pid)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def build_slime_train_args(
    config: dict,
    env_config: dict,
    epoch: int,
    base_model: str,
    save_dir: str,
    env_name: str,
    num_gpus: int,
    seed: int = 42,
    teacher_port: int | None = None,
    load_path: str | None = None,
) -> list[str]:
    """Build slime training CLI arguments for one epoch."""
    rollouts_per_epoch = config.get("rollouts_per_epoch", 50)
    start_rollout = epoch * rollouts_per_epoch
    end_rollout = (epoch + 1) * rollouts_per_epoch

    ckpt_dir = Path(base_model)
    hf_ckpt_path = f"{save_dir}/hf_ckpt_{{rollout_id}}"

    env_module_map = {
        "points24": "examples.gtr_turbo.points24",
        "alfworld": "examples.gtr_turbo.alfworld",
    }
    env_base = env_module_map.get(env_name, f"examples.gtr_turbo.{env_name}")

    rollout_module = f"{env_base}.rollout"

    _env_short = {"alfworld": "alf", "points24": "p24"}.get(env_name, env_name)
    _model_short = Path(base_model).name.lower().replace("-", "").replace("_", "").replace("instruct", "")
    _run_tag = config.get("run_tag", "gtrturbo")
    wandb_group = f"{_run_tag}_{_env_short}_{_model_short}"

    # Separate GPUs: actor on first 4, rollout on next 3, teacher on last 1
    train_gpus = 4
    rollout_gpus = num_gpus - 1 - train_gpus  # 3

    repo_root = Path(__file__).resolve().parents[3]

    args = [
        f"--hf-checkpoint={ckpt_dir}",
        "--megatron-to-hf-mode=bridge",
        f"--prompt-data={repo_root}/examples/gtr_turbo/{env_name}/data/{env_name}_prompts.jsonl",
        "--input-key=prompt",
        "--apply-chat-template",
        f"--save={save_dir}/checkpoints",
        f"--save-hf={hf_ckpt_path}",
        f"--save-interval={rollouts_per_epoch}",
        f"--start-rollout-id={start_rollout}",
        f"--num-rollout={end_rollout}",

        f"--rollout-function-path=slime.rollout.sglang_rollout.generate_rollout",
        f"--custom-generate-function-path={rollout_module}.generate",
        f"--rollout-interaction-env-path={env_base}.env_{env_name}",
        f"--custom-config-path={repo_root}/examples/gtr_turbo/{env_name}/config.yaml",

        "--rollout-batch-size=4",
        "--n-samples-per-prompt=4",
        "--rollout-temperature=0.6",
        "--rollout-top-p=1.0",
        f"--rollout-max-response-len={env_config.get('max_context_len', 8192)}",
        f"--rollout-max-context-len={env_config.get('max_context_len', 8192)}",
        f"--max-turns={env_config.get('max_turns', 20)}",

        "--advantage-estimator=grpo",
        "--ppo-clip=0.2",
        "--kl-loss-coef=0.00",
        "--entropy-coef=0.00",

        "--lr=1e-5",
        "--lr-warmup-init=1e-7",
        "--lr-warmup-iters=10",
        "--lr-decay-style=cosine",
        "--min-lr=1e-9",
        "--adam-beta1=0.9",
        "--adam-beta2=0.99",
        "--clip-grad=1.0",
        "--weight-decay=0.01",
        "--bf16",
        "--global-batch-size=16",

        f"--rollout-num-gpus={rollout_gpus}",
        "--sglang-mem-fraction-static=0.75",

        "--train-backend=megatron",
        "--gradient-checkpointing",
        "--use-flash-attn-3",
        "--actor-num-nodes=1",
        f"--actor-num-gpus-per-node={train_gpus}",
        "--tensor-model-parallel-size=1",
        "--rollout-shuffle",

        # Qwen3-VL-8B model architecture (from text_config)
        "--hidden-size=4096",
        "--num-attention-heads=32",
        "--num-layers=36",
        "--ffn-hidden-size=12288",
        "--group-query-attention",
        "--num-query-groups=8",
        "--vocab-size=151936",
        "--untie-embeddings-and-output-weights",
        "--norm-epsilon=1e-6",
        "--rotary-base=5000000",
        "--max-position-embeddings=262144",

        # Seeds
        f"--seed={seed}",
        f"--rollout-seed={seed}",

        # Wandb logging — disabled during initial debug
        # "--use-wandb",
        # "--disable-wandb-random-suffix",
        # f"--wandb-key={config.get('wandb_key', '')}",
        # f"--wandb-host={config.get('wandb_host', 'https://wandb.glm.ai')}",
        # f"--wandb-project={config.get('wandb_project', 'gtr')}",
        # f"--wandb-group={wandb_group}",
    ]

    if load_path is not None:
        args.append(f"--load={load_path}")

    model_args = config.get("model_args")
    if model_args:
        args = [arg for arg in args if not arg.startswith((
            "--hidden-size=",
            "--num-attention-heads=",
            "--num-layers=",
            "--ffn-hidden-size=",
            "--num-query-groups=",
            "--vocab-size=",
            "--norm-epsilon=",
            "--rotary-base=",
            "--max-position-embeddings=",
        )) and arg not in [
            "--group-query-attention",
            "--untie-embeddings-and-output-weights",
        ]]
        for key, value in model_args.items():
            cli_key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{cli_key}")
            else:
                args.append(f"--{cli_key}={value}")

    if teacher_port is not None:
        args.extend([
            "--use-opd",
            "--opd-type=sglang",
            f"--opd-kl-coef={config.get('opd_kl_coef', 1.0)}",
            f"--rm-url=http://localhost:{teacher_port}/generate",
            f"--custom-rm-path=slime.rollout.on_policy_distillation.reward_func",
            f"--custom-reward-post-process-path=examples.gtr_turbo.gtr_turbo_train.gtr_turbo_reward.post_process_rewards",
        ])
    else:
        args.append(f"--custom-rm-path={env_base}.reward.reward_func")

    return args


def _rollout_id_from_hf_ckpt(path: Path) -> int:
    match = re.fullmatch(r"hf_ckpt_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_latest_hf_checkpoint(save_dir: str, min_rollout_id: int | None = None) -> str | None:
    """Return latest HF checkpoint path, optionally requiring a minimum rollout id."""
    candidates = [
        p for p in Path(save_dir).glob("hf_ckpt_*")
        if p.is_dir() and _rollout_id_from_hf_ckpt(p) >= 0
    ]
    if min_rollout_id is not None:
        candidates = [p for p in candidates if _rollout_id_from_hf_ckpt(p) >= min_rollout_id]
    if not candidates:
        return None
    return str(max(candidates, key=_rollout_id_from_hf_ckpt))


def run_slime_training(args_list: list[str], extra_env: dict[str, str] | None = None) -> None:
    """Run one epoch of slime training as a subprocess."""
    train_script = Path(__file__).resolve().parents[3] / "train.py"
    cmd = [sys.executable, str(train_script)] + args_list
    logger.info("Running slime training: %s ...", " ".join(cmd[:15]))
    env = os.environ.copy()
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["WANDB_INIT_TIMEOUT"] = "30"
    env["SLIME_SKIP_WANDB_REINIT"] = "1"
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, check=True, env=env)
    logger.info("Training epoch completed (return code %d)", result.returncode)


def gtr_turbo_train_loop(config_path: str):
    """Main GTR-Turbo training loop with periodic checkpoint merging."""
    config = load_config(config_path)

    env_name = config.get("env", "points24")
    env_config_path = config.get("env_config", f"examples/gtr_turbo/{env_name}/config.yaml")
    env_config = load_config(env_config_path)

    base_model = config["base_model"]
    save_dir_base = config.get("save_dir", "/workspace/wt/gtr_runs")
    num_epochs = config.get("num_epochs", 10)
    num_gpus = config.get("num_gpus", 8)
    seed = config.get("seed", 42)
    run_tag = config.get("run_tag", "gtrturbo")
    teacher_port = config.get("teacher_port", 13141)
    teacher_gpu = config.get("teacher_gpu", num_gpus - 1)
    merge_interval = config.get("merge_interval", 1)
    ties_density = config.get("ties_density", 0.8)
    weighting = config.get("weighting", "ema")
    ema_alpha = config.get("ema_alpha", 0.5)

    # Build run name: {tag}_alf_qwen3vl8b_seed42
    env_short = {"alfworld": "alf", "points24": "p24"}.get(env_name, env_name)
    model_short = Path(base_model).name.lower().replace("-", "").replace("_", "").replace("instruct", "")
    run_name = f"{run_tag}_{env_short}_{model_short}_seed{seed}"
    save_dir = f"{save_dir_base}/{run_name}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    os.environ["SAVE_DIR"] = save_dir
    logger.info("Run name: %s", run_name)
    logger.info("Save directory: %s", save_dir)

    # Wandb: generate a fixed run ID so all epochs log to the same run.
    # group = tag_env (shared across seeds for mean±std aggregation)
    # name  = run_name (unique per seed for identification)
    import uuid
    wandb_run_id = uuid.uuid4().hex[:8]
    wandb_env = {
        "WANDB_RUN_ID": wandb_run_id,
        "WANDB_RESUME": "allow",
        "WANDB_NAME": f"seed{seed}",
    }
    logger.info("Wandb run ID: %s, name: %s", wandb_run_id, run_name)

    from examples.gtr_turbo.ties_merge.checkpoint_buffer import CheckpointBuffer

    buffer = CheckpointBuffer(base_model, weighting=weighting, ema_alpha=ema_alpha)
    teacher_proc = None
    current_teacher_path = None
    megatron_load_path = f"{save_dir}/checkpoints"

    try:
        # Start teacher as base model so epoch 1 already has OPD
        logger.info("Starting initial teacher server with base model: %s", base_model)
        teacher_proc = start_teacher_server(
            model_path=base_model,
            port=teacher_port,
            gpu_id=teacher_gpu,
        )
        current_teacher_path = base_model

        for epoch in range(num_epochs):
            logger.info("=" * 60)
            logger.info("GTR-Turbo Epoch %d/%d", epoch + 1, num_epochs)
            logger.info("=" * 60)

            train_args = build_slime_train_args(
                config=config,
                env_config=env_config,
                epoch=epoch,
                base_model=base_model,
                save_dir=save_dir,
                env_name=env_name,
                num_gpus=num_gpus,
                seed=seed,
                teacher_port=teacher_port,
                load_path=megatron_load_path if Path(megatron_load_path, "latest_checkpointed_iteration.txt").exists() else None,
            )

            run_slime_training(train_args, extra_env=wandb_env)

            rollouts_per_epoch = config.get("rollouts_per_epoch", 50)
            expected_rollout_id = (epoch + 1) * rollouts_per_epoch - 1
            hf_ckpt_path = f"{save_dir}/hf_ckpt_{expected_rollout_id}"
            if Path(hf_ckpt_path).exists():
                buffer.add_checkpoint(hf_ckpt_path)
            else:
                logger.warning("Expected checkpoint not found: %s", hf_ckpt_path)
                latest = find_latest_hf_checkpoint(save_dir, min_rollout_id=expected_rollout_id)
                if latest and latest not in buffer.checkpoint_paths:
                    buffer.add_checkpoint(latest)

            # Update teacher: always merge(base_model, ckpts) after each epoch
            if len(buffer) >= 1 and (epoch + 1) % merge_interval == 0:
                # Reuse a single temp path to avoid accumulating merged checkpoints
                merged_path = f"{save_dir}/_tmp_merged_teacher"
                logger.info("Merging %d checkpoints -> %s", len(buffer), merged_path)

                stop_server(teacher_proc)

                # Clean previous merge before writing new one
                import shutil
                if Path(merged_path).exists():
                    shutil.rmtree(merged_path)

                buffer.save_merged(merged_path, density=ties_density)

                teacher_proc = start_teacher_server(
                    model_path=merged_path,
                    port=teacher_port,
                    gpu_id=teacher_gpu,
                )
                current_teacher_path = merged_path
                logger.info("Teacher updated (epoch %d, %d ckpts merged)", epoch, len(buffer))
            else:
                logger.info(
                    "Skipping merge (buffer=%d, interval=%d, epoch=%d)",
                    len(buffer), merge_interval, epoch + 1,
                )

    finally:
        stop_server(teacher_proc)
        logger.info("GTR-Turbo training completed")


def parse_args():
    parser = argparse.ArgumentParser(description="GTR-Turbo training orchestrator")
    parser.add_argument("--config", required=True, help="Path to GTR-Turbo config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    gtr_turbo_train_loop(args.config)


if __name__ == "__main__":
    main()
