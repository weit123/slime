#!/bin/bash

# Qwen3-VL-8B small-scale smoke test on geo3k dataset.
# Reuses /workspace/wt/Qwen3-VL-8B-Instruct via /root/models symlink.
# Reduced rollout/batch/response-len; eval disabled.

TRAIN_BACKEND="megatron"
MODEL_NAME="Qwen3-VL-8B-Instruct"
DATASET_NAME="chenhegu/geo3k_imgurl"
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# Cleanup any leftover processes from prior runs.
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 slime
sleep 3
pkill -9 ray
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Model is already on disk at /workspace/wt/Qwen3-VL-8B-Instruct, symlinked into /root/models.
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
   echo "Expected /root/models/${MODEL_NAME} to exist (symlink or real dir)." >&2
   exit 1
fi
if [ ! -d "/root/datasets/${DATASET_LOCAL_NAME}" ]; then
   echo "Expected /root/datasets/${DATASET_LOCAL_NAME}; run hf download first." >&2
   exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint /root/models/${MODEL_NAME}
   --rotary-base 5000000
)

# Small-scale rollout settings: just prove the pipeline runs end-to-end.
ROLLOUT_ARGS=(
   --prompt-data /root/datasets/${DATASET_LOCAL_NAME}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 3
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 0.8
   --global-batch-size 32
)

MULTIMODAL_KEYS='{"image": "images"}'

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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32
)

MISC_ARGS=(
   --colocate
)

BACKEND_ARGS=(
   --train-backend megatron
   --load /root/models/${MODEL_NAME}
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="*"
export NO_PROXY="*"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"*\",
    \"NO_PROXY\": \"*\",
    \"http_proxy\": \"\",
    \"https_proxy\": \"\",
    \"HTTP_PROXY\": \"\",
    \"HTTPS_PROXY\": \"\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}
