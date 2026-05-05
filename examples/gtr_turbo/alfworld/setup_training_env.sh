#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/gtr_turbo/alfworld/setup_training_env.sh [options]

Options:
  --block             Keep Ray head in the foreground. Useful in managed shells
                      where background Ray daemons may be reaped.
  --no-clean          Do not stop existing Ray/SGLang/Xvfb/THOR processes.
  --num-cpus N        Logical CPUs to register with Ray. Default: 64.
  --num-gpus N        GPUs to register with Ray. Default: detected by nvidia-smi, fallback 8.
  --master-addr IP    Ray head node IP. Default: first hostname -I entry, fallback 127.0.0.1.
  -h, --help          Show this help.

After setup, run:
  export RAY_ADDRESS=<master-addr>:6379
  export PYTHONPATH=/root/Megatron-LM:/workspace/wt/gtr_slime:${PYTHONPATH:-}
  python examples/gtr_turbo/alfworld/run_alfworld.py

The script also writes these exports to:
  /workspace/wt/gtr_slime/examples/gtr_turbo/alfworld/.training_env
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
WORKSPACE="${WORKSPACE:-/workspace/wt}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

BLOCK=0
CLEAN=1
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-}"
MASTER_ADDR="${MASTER_ADDR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --block)
      BLOCK=1
      shift
      ;;
    --no-clean)
      CLEAN=0
      shift
      ;;
    --num-cpus)
      RAY_NUM_CPUS="$2"
      shift 2
      ;;
    --num-gpus)
      RAY_NUM_GPUS="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

if [[ -z "${RAY_NUM_GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    RAY_NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
fi

export WORKSPACE
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SLIME_SCRIPT_NUM_GPUS="${SLIME_SCRIPT_NUM_GPUS:-${RAY_NUM_GPUS}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/.cache/alfworld}"
export PYTHONPATH="${MEGATRON_PATH}:${REPO_ROOT}:${PYTHONPATH:-}"
export RAY_ADDRESS="${MASTER_ADDR}:6379"

# Avoid routing local Ray/SGLang health checks through the machine HTTP proxy.
LOCAL_NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR}"
if [[ -n "${no_proxy:-}" ]]; then
  export no_proxy="${LOCAL_NO_PROXY},${no_proxy}"
else
  export no_proxy="${LOCAL_NO_PROXY}"
fi
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${LOCAL_NO_PROXY},${NO_PROXY}"
else
  export NO_PROXY="${LOCAL_NO_PROXY}"
fi

ENV_FILE="${SCRIPT_DIR}/.training_env"
cat > "${ENV_FILE}" <<EOF
export WORKSPACE="${WORKSPACE}"
export MEGATRON_PATH="${MEGATRON_PATH}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export SLIME_SCRIPT_NUM_GPUS="${SLIME_SCRIPT_NUM_GPUS}"
export ALFWORLD_DATA="${ALFWORLD_DATA}"
export PYTHONPATH="${MEGATRON_PATH}:${REPO_ROOT}:\${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS}"
export no_proxy="${no_proxy}"
export NO_PROXY="${NO_PROXY}"
EOF

echo "Repository: ${REPO_ROOT}"
echo "Ray head:   ${RAY_ADDRESS}"
echo "Ray CPUs:   ${RAY_NUM_CPUS}"
echo "Ray GPUs:   ${RAY_NUM_GPUS}"
echo "Env file:   ${ENV_FILE}"

if [[ "${CLEAN}" == "1" ]]; then
  echo "Cleaning old training-side processes..."
  pkill -9 sglang >/dev/null 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 ray >/dev/null 2>&1 || true
  pkill -f '^Xvfb :' >/dev/null 2>&1 || true
  pkill -f 'thor-201909061227-Linux64' >/dev/null 2>&1 || true
  pkill -f 'AI2-Thor' >/dev/null 2>&1 || true
  pkill -9 redis >/dev/null 2>&1 || true
  sleep 2
fi

echo "Starting Ray..."
RAY_CMD=(
  ray start
  --head
  --node-ip-address "${MASTER_ADDR}"
  --num-gpus "${RAY_NUM_GPUS}"
  --num-cpus "${RAY_NUM_CPUS}"
  --disable-usage-stats
)

if [[ "${BLOCK}" == "1" ]]; then
  echo "Ray will stay in the foreground. Use another shell to run training."
  exec "${RAY_CMD[@]}" --block
fi

"${RAY_CMD[@]}"

echo "Ray status:"
ray status --address "${RAY_ADDRESS}" || true

cat <<EOF

Training environment is ready.

For this shell, run:
  source "${ENV_FILE}"
  python examples/gtr_turbo/alfworld/run_alfworld.py

If Ray disappears after this script exits in your execution environment, rerun:
  bash examples/gtr_turbo/alfworld/setup_training_env.sh --block
and launch training from another shell.
EOF
