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
import shutil
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


def start_teacher_server(
    model_path: str,
    port: int,
    tp_size: int = 1,
    gpu_id: int = 0,
    mem_fraction_static: float = 0.70,
) -> subprocess.Popen:
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
        "--mem-fraction-static", str(mem_fraction_static),
        "--trust-remote-code",
    ]
    logger.info("Starting teacher server on GPU %d port %d: %s", gpu_id, port, " ".join(cmd))
    teacher_log = open(f"{os.environ.get('SAVE_DIR', '/tmp')}/teacher_server_{port}.log", "w")
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


def stop_servers(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        stop_server(proc)


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
    teacher_urls: list[str] | None = None,
    load_path: str | None = None,
) -> list[str]:
    """Build slime training CLI arguments for one epoch."""
    rollouts_per_epoch = config.get("rollouts_per_epoch", 50)
    start_rollout = epoch * rollouts_per_epoch
    end_rollout = (epoch + 1) * rollouts_per_epoch

    ckpt_dir = Path(base_model)
    hf_ckpt_path = f"{save_dir}/hf_ckpt_{{rollout_id}}"

    env_module_map = {
        "alfworld": "examples.gtr_turbo.alfworld",
    }
    env_base = env_module_map.get(env_name, f"examples.gtr_turbo.{env_name}")

    rollout_module = f"{env_base}.rollout"

    _env_short = {"alfworld": "alf"}.get(env_name, env_name)
    _model_short = Path(base_model).name.lower().replace("-", "").replace("_", "").replace("instruct", "")
    _run_tag = config.get("run_tag", "gtrturbo")
    wandb_group = f"{_run_tag}_{_env_short}_{_model_short}"

    train_gpus = int(config.get("actor_num_gpus_per_node", max(num_gpus - 2, 1)))
    rollout_gpus = train_gpus if config.get("colocate", True) else int(config.get("rollout_num_gpus", max(num_gpus - 1 - train_gpus, 1)))
    tensor_model_parallel_size = int(config.get("tensor_model_parallel_size", 2))
    global_batch_size = int(config.get("global_batch_size", 192))
    if train_gpus % tensor_model_parallel_size != 0:
        raise ValueError(
            f"actor_num_gpus_per_node={train_gpus} must be divisible by "
            f"tensor_model_parallel_size={tensor_model_parallel_size}."
        )
    if global_batch_size % max(train_gpus // tensor_model_parallel_size, 1) != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by data parallel size "
            f"{max(train_gpus // tensor_model_parallel_size, 1)}."
        )

    repo_root = Path(__file__).resolve().parents[3]

    args = [
        f"--hf-checkpoint={ckpt_dir}",
        "--megatron-to-hf-mode=bridge",
        f"--prompt-data={repo_root}/examples/gtr_turbo/{env_name}/data/{env_name}_prompts.jsonl",
        "--input-key=prompt",
        f"--save={save_dir}/checkpoints",
        f"--save-hf={hf_ckpt_path}",
        f"--save-interval={rollouts_per_epoch}",
        f"--start-rollout-id={start_rollout}",
        f"--num-rollout={end_rollout}",

        f"--rollout-function-path=slime.rollout.sglang_rollout.generate_rollout",
        f"--custom-generate-function-path={rollout_module}.generate",
        f"--rollout-interaction-env-path={env_base}.env_{env_name}",
        f"--custom-config-path={repo_root}/examples/gtr_turbo/{env_name}/config.yaml",
        "--multimodal-keys={\"image\": \"images\"}",

        f"--rollout-batch-size={config.get('rollout_batch_size', 2)}",
        f"--n-samples-per-prompt={config.get('n_samples_per_prompt', 32)}",
        f"--rollout-temperature={config.get('rollout_temperature', 1.0)}",
        f"--rollout-top-p={config.get('rollout_top_p', 1.0)}",
        f"--rollout-max-response-len={config.get('rollout_max_response_len', env_config.get('max_action_tokens', 512))}",
        f"--rollout-max-context-len={config.get('rollout_max_context_len', env_config.get('max_context_len', 4096))}",
        f"--max-turns={env_config.get('max_turns', 20)}",

        "--advantage-estimator=grpo",
        "--eps-clip=0.2",
        "--kl-loss-coef=0.00",
        "--entropy-coef=0.00",
        "--use-rollout-logprobs",

        f"--lr={config.get('lr', 1e-6)}",
        "--lr-warmup-init=1e-7",
        f"--lr-warmup-iters={config.get('lr_warmup_iters', 10)}",
        f"--lr-decay-style={config.get('lr_decay_style', 'cosine')}",
        f"--min-lr={config.get('min_lr', 1e-7)}",
        "--adam-beta1=0.9",
        "--adam-beta2=0.99",
        "--clip-grad=1.0",
        "--weight-decay=0.01",
        "--bf16",
        f"--global-batch-size={global_batch_size}",

        f"--rollout-num-gpus={rollout_gpus}",
        f"--sglang-mem-fraction-static={config.get('sglang_mem_fraction_static', 0.70)}",

        "--train-backend=megatron",
        "--gradient-checkpointing",
        "--actor-num-nodes=1",
        f"--actor-num-gpus-per-node={train_gpus}",
        f"--num-gpus-per-node={train_gpus}",
        f"--tensor-model-parallel-size={tensor_model_parallel_size}",
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

    ]

    if config.get("use_wandb", True):
        args.extend([
            "--use-wandb",
            "--disable-wandb-random-suffix",
            f"--wandb-key={config.get('wandb_key', '')}",
            f"--wandb-host={config.get('wandb_host', 'https://wandb.glm.ai')}",
            f"--wandb-project={config.get('wandb_project', 'gtr')}",
            f"--wandb-group={wandb_group}",
        ])

    if config.get("colocate", True):
        args.append("--colocate")
    if env_name == "alfworld":
        args.append(f"--custom-rollout-log-function-path={env_base}.metrics.log_rollout")

    if load_path is not None:
        args.append(f"--load={load_path}")
    else:
        args.append(f"--load={ckpt_dir}")

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

    if teacher_urls:
        # Ray rollout actors do not inherit arbitrary parent-process env vars.
        # Encode all teacher replicas in args.rm_url so the custom reward
        # function can load-balance without touching slime core runtime_env.
        rm_url = ",".join(teacher_urls)
    elif teacher_port is not None:
        rm_url = f"http://localhost:{teacher_port}/generate"
    else:
        rm_url = None

    if rm_url is not None:
        args.extend([
            "--use-opd",
            "--opd-type=sglang",
            f"--opd-kl-coef={config.get('opd_kl_coef', 1.0)}",
            f"--rm-url={rm_url}",
            f"--custom-rm-path=examples.gtr_turbo.gtr_turbo_train.gtr_turbo_reward.reward_func",
            f"--custom-reward-post-process-path=examples.gtr_turbo.gtr_turbo_train.gtr_turbo_reward.post_process_rewards",
        ])
    else:
        args.append(f"--custom-rm-path={env_base}.reward.reward_func")
        if env_name == "alfworld":
            args.append("--custom-reward-post-process-path=examples.gtr_turbo.alfworld.reward.post_process_step_rewards")

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


def _megatron_iter_id(path: Path) -> int:
    match = re.fullmatch(r"iter_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def cleanup_megatron_checkpoints(checkpoint_dir: str, keep_last: int = 1) -> None:
    """Keep only the newest Megatron iter_* checkpoints.

    HF checkpoints are intentionally not touched. The latest Megatron state is
    needed to continue the next outer-loop epoch without resetting optimizer
    and scheduler state.
    """
    if keep_last < 0:
        return

    root = Path(checkpoint_dir)
    if not root.exists():
        return

    iter_dirs = sorted(
        [path for path in root.glob("iter_*") if path.is_dir() and _megatron_iter_id(path) >= 0],
        key=_megatron_iter_id,
    )
    stale_dirs = iter_dirs[:-keep_last] if keep_last else iter_dirs
    for path in stale_dirs:
        logger.info("Removing stale Megatron checkpoint: %s", path)
        shutil.rmtree(path)


def merge_teacher_incremental(
    *,
    base_model: str,
    current_teacher_path: str,
    teacher_checkpoint_count: int,
    new_checkpoint_path: str,
    output_path: str,
    density: float,
    weighting: str,
    ema_alpha: float,
) -> tuple[str, int]:
    """Merge current teacher with the newest checkpoint.

    Online weights:
      - SMA after n checkpoints: old_teacher * n/(n+1) + new_ckpt * 1/(n+1)
      - EMA: old_teacher * (1-alpha) + new_ckpt * alpha

    This keeps TIES memory bounded with respect to training history: each merge
    loads only base, current teacher, and the newest checkpoint instead of all
    historical checkpoints. Task vectors are still computed relative to the
    original base model, as required by TIES.
    """
    import shutil
    from pathlib import Path

    from examples.gtr_turbo.ties_merge.ties_merging import (
        copy_hf_assets,
        save_state_dict,
        ties_merge_from_paths,
    )

    if Path(output_path).exists():
        shutil.rmtree(output_path)

    if teacher_checkpoint_count <= 0 or Path(current_teacher_path).resolve() == Path(base_model).resolve():
        checkpoint_paths = [new_checkpoint_path]
        weights = [1.0]
        next_count = 1
    else:
        checkpoint_paths = [current_teacher_path, new_checkpoint_path]
        if weighting == "sma":
            weights = [
                teacher_checkpoint_count / (teacher_checkpoint_count + 1),
                1.0 / (teacher_checkpoint_count + 1),
            ]
        elif weighting == "ema":
            weights = [1.0 - ema_alpha, ema_alpha]
        else:
            raise ValueError(f"Unknown weighting: {weighting}")
        next_count = teacher_checkpoint_count + 1

    logger.info(
        "Online teacher merge: weighting=%s, teacher_checkpoint_count=%d -> %d, weights=%s, paths=%s",
        weighting,
        teacher_checkpoint_count,
        next_count,
        [round(w, 6) for w in weights],
        checkpoint_paths,
    )
    merged_sd = ties_merge_from_paths(base_model, checkpoint_paths, weights, density=density)
    save_state_dict(merged_sd, output_path)
    copy_hf_assets(base_model, output_path)
    return output_path, next_count


def run_slime_training(args_list: list[str], extra_env: dict[str, str] | None = None) -> None:
    """Run one epoch of slime training as a subprocess."""
    train_script = Path(__file__).resolve().parents[3] / "train.py"
    cmd = [sys.executable, str(train_script)] + args_list
    logger.info("Running slime training: %s ...", " ".join(cmd[:15]))
    env = os.environ.copy()
    megatron_path = "/root/Megatron-LM"
    env["PYTHONPATH"] = (
        f"{megatron_path}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else megatron_path
    )
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["WANDB_INIT_TIMEOUT"] = "30"
    env["SLIME_SKIP_WANDB_REINIT"] = "1"
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, check=True, env=env)
    logger.info("Training epoch completed (return code %d)", result.returncode)


def _parse_int_list(value, default: list[int] | None = None) -> list[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _parse_visible_devices(devices: str | list[int] | None, num_gpus: int, teacher_gpus: list[int]) -> list[int]:
    if devices is None:
        teacher_set = set(teacher_gpus)
        return [idx for idx in range(num_gpus) if idx not in teacher_set]
    if isinstance(devices, str):
        return [int(item.strip()) for item in devices.split(",") if item.strip()]
    return [int(item) for item in devices]


def gtr_turbo_train_loop(config_path: str):
    """Main GTR-Turbo training loop with periodic checkpoint merging."""
    config = load_config(config_path)

    env_name = config.get("env", "alfworld")
    env_config_path = config.get("env_config", f"examples/gtr_turbo/{env_name}/config.yaml")
    env_config = load_config(env_config_path)

    base_model = config["base_model"]
    save_dir_base = config.get("save_dir", "/workspace/wt/gtr_runs")
    num_epochs = config.get("num_epochs", 10)
    num_gpus = config.get("num_gpus", 8)
    seed = config.get("seed", 42)
    run_tag = config.get("run_tag", "gtrturbo")
    teacher_gpus = _parse_int_list(config.get("teacher_gpus"), default=[config.get("teacher_gpu", num_gpus - 1)])
    teacher_ports = _parse_int_list(
        config.get("teacher_ports"),
        default=[int(config.get("teacher_port", 13141)) + idx for idx in range(len(teacher_gpus))],
    )
    if len(teacher_ports) != len(teacher_gpus):
        raise ValueError(f"teacher_ports={teacher_ports} must have the same length as teacher_gpus={teacher_gpus}.")
    teacher_urls = [f"http://localhost:{port}/generate" for port in teacher_ports]
    teacher_mem_fraction_static = float(config.get("teacher_mem_fraction_static", 0.70))
    slime_visible_devices = _parse_visible_devices(config.get("slime_visible_devices"), num_gpus, teacher_gpus)
    overlap = sorted(set(teacher_gpus).intersection(slime_visible_devices))
    if overlap and not config.get("allow_shared_teacher_gpu", False):
        raise ValueError(
            f"teacher_gpus={teacher_gpus} overlap slime_visible_devices={slime_visible_devices}: {overlap}. "
            "Keep SGLang teachers on dedicated GPUs by default, or set allow_shared_teacher_gpu=true "
            "after validating memory headroom."
        )
    expected_actor_gpus = int(config.get("actor_num_gpus_per_node", len(slime_visible_devices)))
    if expected_actor_gpus != len(slime_visible_devices):
        raise ValueError(
            f"actor_num_gpus_per_node={expected_actor_gpus} must match the number of "
            f"slime_visible_devices={slime_visible_devices}."
        )
    merge_interval = config.get("merge_interval", 1)
    keep_megatron_checkpoints = int(config.get("keep_megatron_checkpoints", 1))
    ties_density = config.get("ties_density", 0.8)
    weighting = config.get("weighting", "ema")
    ema_alpha = config.get("ema_alpha", 0.5)
    incremental_teacher_merge = bool(config.get("incremental_teacher_merge", True))

    # Build run name: {tag}_alf_qwen3vl8b_seed42
    env_short = {"alfworld": "alf"}.get(env_name, env_name)
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

    teacher_procs: list[subprocess.Popen] = []
    current_teacher_path = None
    teacher_checkpoint_count = 0
    megatron_load_path = f"{save_dir}/checkpoints"
    checkpoint_paths: list[str] = []

    try:
        # Start teacher as base model so epoch 1 already has OPD
        logger.info(
            "Starting initial teacher replicas with base model: %s, gpus=%s, ports=%s",
            base_model,
            teacher_gpus,
            teacher_ports,
        )
        teacher_procs = [
            start_teacher_server(
                model_path=base_model,
                port=port,
                gpu_id=gpu,
                mem_fraction_static=teacher_mem_fraction_static,
            )
            for gpu, port in zip(teacher_gpus, teacher_ports, strict=True)
        ]
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
                teacher_urls=teacher_urls,
                load_path=megatron_load_path if Path(megatron_load_path, "latest_checkpointed_iteration.txt").exists() else None,
            )

            train_env = dict(wandb_env)
            train_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in slime_visible_devices)
            train_env["SLIME_SCRIPT_NUM_GPUS"] = str(len(slime_visible_devices))
            train_env["GTR_TEACHER_URLS"] = ",".join(teacher_urls)
            train_env["GTR_TEACHER_MAX_CONCURRENCY_PER_SERVER"] = str(
                config.get("teacher_max_concurrency_per_server", 4)
            )
            run_slime_training(train_args, extra_env=train_env)

            rollouts_per_epoch = config.get("rollouts_per_epoch", 50)
            expected_rollout_id = (epoch + 1) * rollouts_per_epoch - 1
            hf_ckpt_path = f"{save_dir}/hf_ckpt_{expected_rollout_id}"
            if Path(hf_ckpt_path).exists():
                checkpoint_paths.append(hf_ckpt_path)
                latest_checkpoint = hf_ckpt_path
            else:
                logger.warning("Expected checkpoint not found: %s", hf_ckpt_path)
                latest = find_latest_hf_checkpoint(save_dir, min_rollout_id=expected_rollout_id)
                latest_checkpoint = latest
                if latest and latest not in checkpoint_paths:
                    checkpoint_paths.append(latest)

            # Update teacher after each merge interval. The default path uses
            # online merge(current_teacher, latest_checkpoint), not full-history
            # merge(base_model, all_checkpoints).
            if latest_checkpoint and (epoch + 1) % merge_interval == 0:
                # Alternate two temp paths so the current teacher can be used
                # as merge input while the next teacher is being written.
                merged_path = f"{save_dir}/_tmp_merged_teacher_{(epoch + 1) % 2}"
                logger.info("Merging teacher incrementally -> %s", merged_path)

                stop_servers(teacher_procs)
                teacher_procs = []

                if incremental_teacher_merge:
                    _, teacher_checkpoint_count = merge_teacher_incremental(
                        base_model=base_model,
                        current_teacher_path=current_teacher_path or base_model,
                        teacher_checkpoint_count=teacher_checkpoint_count,
                        new_checkpoint_path=latest_checkpoint,
                        output_path=merged_path,
                        density=ties_density,
                        weighting=weighting,
                        ema_alpha=ema_alpha,
                    )
                else:
                    import shutil

                    from examples.gtr_turbo.ties_merge.checkpoint_buffer import CheckpointBuffer

                    if Path(merged_path).exists():
                        shutil.rmtree(merged_path)
                    buffer = CheckpointBuffer(base_model, weighting=weighting, ema_alpha=ema_alpha)
                    for checkpoint_path in checkpoint_paths:
                        buffer.add_checkpoint(checkpoint_path)
                    buffer.save_merged(merged_path, density=ties_density)
                    teacher_checkpoint_count = len(checkpoint_paths)

                teacher_procs = [
                    start_teacher_server(
                        model_path=merged_path,
                        port=port,
                        gpu_id=gpu,
                        mem_fraction_static=teacher_mem_fraction_static,
                    )
                    for gpu, port in zip(teacher_gpus, teacher_ports, strict=True)
                ]
                current_teacher_path = merged_path
                logger.info(
                    "Teacher replicas updated (epoch %d, incremental=%s, historical_ckpts_seen=%d, urls=%s)",
                    epoch + 1,
                    incremental_teacher_merge,
                    len(checkpoint_paths),
                    teacher_urls,
                )
            else:
                logger.info(
                    "Skipping merge (latest_checkpoint=%s, interval=%d, epoch=%d)",
                    latest_checkpoint, merge_interval, epoch + 1,
                )

            cleanup_megatron_checkpoints(megatron_load_path, keep_last=keep_megatron_checkpoints)

    finally:
        stop_servers(teacher_procs)
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
