#!/bin/bash

# Direct runner: bypass `ray job submit` so train.py's stderr is not swallowed.
# Starts ray head locally, then runs python3 train.py in the same shell with
# RAY_ADDRESS=auto so it attaches as a client driver.

TRAIN_BACKEND="megatron"
MODEL_NAME="Qwen3-VL-8B-Instruct"
DATASET_NAME="chenhegu/geo3k_imgurl"
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

pkill -9 sglang 2>/dev/null
sleep 2
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 slime 2>/dev/null
pkill -9 redis 2>/dev/null
sleep 2

set -ex

export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/Megatron-LM/
export CUDA_DEVICE_MAX_CONNECTIONS=1
export no_proxy="*"
export NO_PROXY="*"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   export NCCL_NVLS_ENABLE=1
else
   export NCCL_NVLS_ENABLE=0
fi
echo "NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"

if [ ! -d "/root/models/${MODEL_NAME}" ]; then
   echo "Expected /root/models/${MODEL_NAME}" >&2; exit 1
fi
if [ ! -d "/root/datasets/${DATASET_LOCAL_NAME}" ]; then
   echo "Expected /root/datasets/${DATASET_LOCAL_NAME}" >&2; exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint /root/models/${MODEL_NAME}
   --rotary-base 5000000
)

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
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

export RAY_ADDRESS="${MASTER_ADDR}:6379"

cd "${SLIME_DIR}"

# Unbuffered python, stderr merged into stdout, in the same foreground shell.
exec python3 -u train.py \
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
