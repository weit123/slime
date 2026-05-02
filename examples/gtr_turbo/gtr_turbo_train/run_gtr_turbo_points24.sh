#!/bin/bash
# GTR-Turbo training on Points24 environment.
#
# This script launches the GTR-Turbo training loop which orchestrates:
#   1. RL training with GRPO on Points24
#   2. Periodic TIES merging of RL checkpoints
#   3. Merged model deployed as OPD teacher for KL distillation
#
# Usage:
#   bash examples/gtr_turbo/gtr_turbo_train/run_gtr_turbo_points24.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override with environment variables if needed
export SLIME_SCRIPT_MODEL_NAME="${SLIME_SCRIPT_MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
export SLIME_SCRIPT_NUM_GPUS="${SLIME_SCRIPT_NUM_GPUS:-8}"

python -m examples.gtr_turbo.gtr_turbo_train.train_gtr_turbo \
    --config "${SCRIPT_DIR}/config.yaml"
