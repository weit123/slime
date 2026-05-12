#!/bin/bash

# Android World VLM agent RL training with slime
# Optimized for: 2 nodes x 8 H20 GPUs (96GB HBM3, PCIe Gen5, no NVLink, InfiniBand)
# Model: Qwen3-VL-4B-Instruct, Backend: Megatron
#
# Key optimizations vs run_android_world.sh:
#   - TP=4 → TP=2: halves PCIe all-reduce overhead (H20 has no NVLink)
#   - 8 training + 8 rollout GPUs (dedicated nodes, non-colocate)
#   - max-tokens-per-gpu 2048 → 8192: better GPU utilization on 96GB HBM3
#   - sglang-mem-fraction-static 0.7 → 0.82: more KV cache on dedicated rollout GPUs
#   - CUDA graph BS trimmed to 1..32: saves GPU memory (max ~16 concurrent reqs/engine)
#   - NCCL: enable InfiniBand, disable NVLS (no NVLink), enable GPU Direct RDMA
#   - Weight sync buffer 512MB → 1GB: fewer broadcast rounds
#
# Usage:
#   # With external Ray cluster (recommended for multi-node):
#   bash examples/android_world/multi_node_ray_start.sh
#   SLIME_SCRIPT_EXTERNAL_RAY=1 bash examples/android_world/run_android_world_opt.sh
#
#   # Single-command (starts Ray internally):
#   SLIME_SCRIPT_NUM_NODES=2 SLIME_SCRIPT_NUM_GPUS=8 bash examples/android_world/run_android_world_opt.sh

# Configuration — defaults tuned for 2-node H20 setup
TRAIN_BACKEND=${SLIME_SCRIPT_TRAIN_BACKEND:-"megatron"}
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-4B-Instruct"}
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-2}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
TASK_DATA=${SLIME_SCRIPT_TASK_DATA:-"examples/android_world/data/tasks.jsonl"}

# Validate MODEL_NAME
VALID_MODELS="
  Qwen2.5-VL-3B-Instruct
  Qwen2.5-VL-7B-Instruct
  Qwen2.5-VL-32B-Instruct
  Qwen2.5-VL-72B-Instruct
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-30B-A3B-Instruct
  Qwen3-VL-235B-A22B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
  Qwen3-VL-30B-A3B-Thinking
  Qwen3-VL-235B-A22B-Thinking
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
   exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# Cleanup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "${SCRIPT_DIR}/cleanup.sh" ]; then
   bash "${SCRIPT_DIR}/cleanup.sh" 2>/dev/null || true
fi
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Detect NVLink (H20 should have none)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download model
mkdir -p /root/models
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
   hf download Qwen/${MODEL_NAME} --local-dir /root/models/${MODEL_NAME}
fi

# Common args
CKPT_ARGS=(
   --hf-checkpoint /root/models/${MODEL_NAME}
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data ${TASK_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --custom-generate-function-path examples.android_world.rollout.generate
   --custom-config-path examples/android_world/config.yaml
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.7
   --num-steps-per-rollout 1
   --balance-data
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# [OPT] Trimmed CUDA graph BS to 1..32. With 128 total samples across 8 engines,
# each engine handles ~16 concurrent requests. BS > 32 wastes GPU memory on
# CUDA graph captures that are never used.
# [OPT] mem-fraction-static 0.7 → 0.82: dedicated rollout GPUs (non-colocate)
# have no training weights competing for memory. More KV cache supports more
# concurrent long-context sequences (max_context_len=16384).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.82
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32
)

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   LOGGER_ARGS=(
      --use-wandb
      --wandb-project slime-android-world
      --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   LOGGER_ARGS=(
      --use-tensorboard
      --tb-project-name slime-android-world
      --tb-experiment-name ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
   )
fi

# [OPT] sglang-server-concurrency 160 → 256: scale with 8 engines (256/8=32 per engine)
MISC_ARGS=(
   --sglang-server-concurrency 256
)

# Backend-specific args
if [ "$TRAIN_BACKEND" = "fsdp" ]; then
   BACKEND_ARGS=(
      --train-backend fsdp
      --gradient-checkpointing
      --sglang-attention-backend fa3
      --attn-implementation flash_attention_3
      --update-weight-buffer-size 536870912
   )
   MODEL_ARGS=()
else
   # megatron backend — optimized for H20 (no NVLink, 96GB HBM3)
   BACKEND_ARGS=(
      --train-backend megatron
      --load /root/models/${MODEL_NAME}
      # [OPT] TP=4 → TP=2: H20 has no NVLink, all TP communication goes over PCIe.
      # Halving TP reduces all-reduce volume by ~50%. With 8 GPUs and TP=2, we get
      # DP=4 (4 data-parallel ranks), processing 128/4=32 samples per DP rank.
      --tensor-model-parallel-size 4
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1  # CP>1 incompatible with VLM in Megatron bridge
      --expert-model-parallel-size 1
      --expert-tensor-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --use-dynamic-batch-size
      # [OPT] max-tokens-per-gpu 2048 → 8192: 4B model on 96GB H20 has massive
      # memory headroom. Larger value = fewer micro-batches = better GPU utilization.
      --max-tokens-per-gpu 8192
      # [OPT] NEW: should be >= max-tokens-per-gpu for efficient log-prob computation
      --log-probs-max-tokens-per-gpu 16384
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash
      --megatron-to-hf-mode bridge
      # [OPT] NEW: 512MB → 1GB buffer. 4B model ≈ 8GB params, 1GB buffer = ~8
      # NCCL broadcast rounds instead of ~16. Faster weight sync to rollout engines.
      --update-weight-buffer-size 1073741824
   )

   # get MODEL_ARGS from scripts/models for megatron backend
   SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
   MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
   MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"
fi

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ulimit -n 1048576 || ulimit -n 65536 || true
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8080
   sleep 5
fi

# [OPT] NCCL env vars optimized for H20:
#   NCCL_IB_DISABLE: 1 → 0  — enable InfiniBand (~10x faster cross-node communication)
#   NCCL_NVLS_ENABLE: 1 → 0 — NVLink SHARP requires NVLink; H20 has none
#   NCCL_NET_GDR_LEVEL: 2 → 5 — enable GPU Direct RDMA on all links
ray job submit --address="http://127.0.0.1:8080" \
   --runtime-env-json='{
      "env_vars": {
         "PYTHONPATH": "/root/Megatron-LM/",
         "CUDA_DEVICE_MAX_CONNECTIONS": "1",
         "NCCL_IP_LOCAL_INTERFACE_TYPE": "IPv4",
         "NCCL_IB_GID_INDEX": "3",
         "NCCL_CHECK_DISABLE": "1",
         "NCCL_IB_SL": "3",
         "NCCL_P2P_DISABLE": "0",
         "NCCL_IB_DISABLE": "0",
         "NCCL_SOCKET_IFNAME": "bond1",
         "NCCL_LL_THRESHOLD": "16384",
         "NCCL_IB_CUDA_SUPPORT": "1",
         "UCX_NET_DEVICES": "bond1",
         "NCCL_IB_HCA": "mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6",
         "NCCL_COLLNET_ENABLE": "0",
         "SHARP_COLL_ENABLE_SAT": "0",
         "NCCL_NET_GDR_LEVEL": "5",
         "NCCL_IB_QPS_PER_CONNECTION": "4",
         "NCCL_IB_TC": "160",
         "NCCL_PXN_DISABLE": "0",
         "NCCL_NVLS_ENABLE": "0",
         "NCCL_PROFILE_PRIMS_ENABLE": "1"
      }
   }' \
   --no-wait -- python train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${LOGGER_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}
