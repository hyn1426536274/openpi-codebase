#!/usr/bin/env bash
# OpenPI 本地 4090 调试训练启动脚本

# wandb 配置 (entity: huangyinuo321-uestc, project: pi05_research)
WANDB_API_KEY="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"
export WANDB_API_KEY

# 1. 环境激活与路径跳转
cd /home/hyn/code/pi05r/openpi-codebase
source .venv/bin/activate 2>/dev/null || true

# 防止 HuggingFace datasets 首次加载时尝试网络请求（本地数据集不需要 Hub）
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# 使用本地磁盘作为 HF datasets 缓存
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"

# 2. 参数处理与 GPU 自动检测
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    AVAILABLE_GPUS=1
fi

# 参数 1: 显卡数量
NUM_GPUS="${1:-1}"
# 参数 2: 配置名称（默认本地 debug 配置）
CONFIG_NAME="${2:-libero10_pi05ki_alltasks_local_debug}"
# 参数 3: 实验名称
EXP_NAME="${3:-pi05_libero_local_$(date +%Y%m%d_%H%M%S)}"
# 参数 4: 是否 resume（默认 false）
RESUME="${4:-false}"

LOG_DIR="/home/hyn/code/pi05r/tmp"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_libero_local_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

# 3. 动态生成 CUDA_VISIBLE_DEVICES
G_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${G_IDS}"

echo "---------------------------------------"
echo "Detected Total GPUs: ${AVAILABLE_GPUS}"
echo "Using GPUs:          ${CUDA_VISIBLE_DEVICES} (Count: ${NUM_GPUS})"
echo "Config:              ${CONFIG_NAME}"
echo "Exp:                 ${EXP_NAME}"
echo "Resume:              ${RESUME}"
echo "HF cache:            ${HF_DATASETS_CACHE}"
echo "Log file:            ${LOG_FILE}"
echo "---------------------------------------"

# 4. 根据是否 resume 选择标志
if [ "${RESUME}" = "true" ]; then
    TRAIN_MODE_FLAG="--resume"
else
    TRAIN_MODE_FLAG="--overwrite"
fi

# 5. 启动训练
# 用法示例：
#   本地 debug:      bash scripts/runshell/train_libero_local.sh
#   指定配置:        bash scripts/runshell/train_libero_local.sh 1 libero10_pi05ki_alltasks_local_debug my_local_exp
#   本地 resume:     bash scripts/runshell/train_libero_local.sh 1 libero10_pi05ki_alltasks_local_debug my_local_exp true
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG_NAME}" \
    --exp_name="${EXP_NAME}" \
    ${TRAIN_MODE_FLAG} 2>&1 | tee "${LOG_FILE}"
