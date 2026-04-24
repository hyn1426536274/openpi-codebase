#!/usr/bin/env bash
# OpenPI 动态多卡分布式训练启动脚本

# wandb 配置 (entity: huangyinuo321-uestc, project: pi05_research)
WANDB_API_KEY="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"
export WANDB_API_KEY

# 1. 环境激活与路径跳转
cd /workspace/code/openpi-codebase
source .venv/bin/activate 2>/dev/null || true

# 防止 HuggingFace datasets 首次加载时尝试网络请求（本地数据集不需要 Hub）
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# 使用本地磁盘作为 HF datasets 缓存（避免 NFS 上的缓存 I/O 慢）(持久化？)
export HF_DATASETS_CACHE="/workspace/tmp/hf_datasets_cache"

# 2. 参数处理与 GPU 自动检测
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    AVAILABLE_GPUS=1
fi

# 参数 1: 显卡数量
NUM_GPUS="${1:-$AVAILABLE_GPUS}"
# 参数 2: 配置名称（默认 val_test 验证配置）
# CONFIG_NAME="${2:-pi05_libero_val_test}" # 25
# CONFIG_NAME="${2:-pi05_libero_torch_debug}" # libero24
# CONFIG_NAME="${2:-pi05_ki_libero_torch_debug}" # 23
# CONFIG_NAME="${2:-pi05_ki_libero_task_filter_test}" # 26

# CONFIG_NAME="${2:-libero10_pi05ki_alltasks}" #
# CONFIG_NAME="${2:-libero10_pi05_alltasks}" #
# CONFIG_NAME="${2:-libero10_pi05ki_onetask}" #
# CONFIG_NAME="${2:-libero10_pi05_onetask}" #
# CONFIG_NAME="${2:-libero10_pi05_alltasks_f32}" #
# CONFIG_NAME="${2:-libero10_pi05_alltasks_f32_v2}" #

## ki debug
# CONFIG_NAME="${2:-libero10_pi05ki_ablate_subtask_only}" # ablate_subtask_only
# CONFIG_NAME="${2:-libero10_pi05ki_ablate_fast_only}" # ablate_fast_only
# CONFIG_NAME="${2:-libero10_pi05ki_ablate_ar_only_ki_off}" # ablate_ar_only_ki_off
# CONFIG_NAME="${2:-libero10_pi05ki_ablate_flow_only}" # ablate_flow_only
# CONFIG_NAME="${2:-libero10_pi05ki_ablate_flow_only_ki_off}" # ablate_flow_only_ki_off


## train form pi05 official (pi05_base jax to torch)
# CONFIG_NAME="${2:-libero10_pi05_alltasks_official}" 
CONFIG_NAME="${2:-libero10_pi05ki_alltasks_official}" 

## ki component ablation with pi05_base official model
# CONFIG_NAME="${2:-libero10_pi05ki_alltasks_official_no_fast}" # ki_official_no_fast
# resume : no_fast_continue
# CONFIG_NAME="${2:-libero10_pi05ki_alltasks_official_no_fast}"
# EXP_NAME="${3:-pi05_libero_20260421_214722}"
# RESUME="${4:-true}"
# EXTRA_TRAIN_ARGS=()
# CONFIG_NAME="${2:-libero10_pi05ki_alltasks_official_no_subtask}" # ki_official_no_subtask
# resume : no_subtask_continue
# CONFIG_NAME="${2:-libero10_pi05ki_alltasks_official_no_subtask}"
# EXP_NAME="${3:-pi05_libero_20260421_215506}"
# RESUME="${4:-true}"
# EXTRA_TRAIN_ARGS=()
# resume : no_ki_continue
# CONFIG_NAME="${2:-libero10_pi05ki_alltasks_official_no_ki}" # ki_official_no_ki
# CONFIG_NAME="${2:-libero10_pi05ki_alltasks_official_no_ki}"
# EXP_NAME="${3:-pi05_libero_20260421_215609}"
# RESUME="${4:-true}"
# EXTRA_TRAIN_ARGS=()

# 参数 3: 实验名称
EXP_NAME="${3:-pi05_libero_$(date +%Y%m%d_%H%M%S)}"
# 参数 4: 是否 resume（默认 false）
RESUME="${4:-false}"
# 额外透传给 train_pytorch.py 的参数。常用于 resume 时提高总步数。
# 示例:
# EXTRA_TRAIN_ARGS=(
#   --num_train_steps=40000
# )
EXTRA_TRAIN_ARGS=()

# continue training exp
# CONFIG_NAME="${2:-libero10_pi05_alltasks}"
# EXP_NAME="${3:-pi05_libero_20260421_214722}"
# RESUME="${4:-true}"
# EXTRA_TRAIN_ARGS=(
#   --num_train_steps=100000
# )
# CONFIG_NAME="${2:-libero10_pi05ki_alltasks}"
# EXP_NAME="${3:-pi05_libero_20260415_145622}"
# RESUME="${4:-true}"
# EXTRA_TRAIN_ARGS=(
#   --num_train_steps=100000
# )



# 3. 动态生成 CUDA_VISIBLE_DEVICES
G_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${G_IDS}"

echo "---------------------------------------"
echo "Detected Total GPUs: ${AVAILABLE_GPUS}"
echo "Using GPUs:          ${CUDA_VISIBLE_DEVICES} (Count: ${NUM_GPUS})"
echo "Config:              ${CONFIG_NAME}"
echo "Exp:                 ${EXP_NAME}"
echo "Resume:              ${RESUME}"
if [ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]; then
    echo "Extra Train Args:    ${EXTRA_TRAIN_ARGS[*]}"
fi
echo "---------------------------------------"

# 4. 根据是否 resume 选择标志
if [ "${RESUME}" = "true" ]; then
    TRAIN_MODE_FLAG="--resume"
else
    TRAIN_MODE_FLAG="--overwrite"
fi

# 5. 启动训练
# 用法示例：
#   验证 train/val:  bash scripts/runshell/train_libero.sh 1 pi05_libero_val_test
#   正式 pi05 训练:  bash scripts/runshell/train_libero.sh 8 pi05_libero_torch_debug my_exp
#   KI 训练:         bash scripts/runshell/train_libero.sh 1 pi05_ki_libero_torch_debug ki_exp
#   Resume 并增加总步数: 先在脚本里设置
#       RESUME="true"
#       EXTRA_TRAIN_ARGS=(--num_train_steps=40000)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG_NAME}" \
    --exp_name="${EXP_NAME}" \
    ${TRAIN_MODE_FLAG} \
    "${EXTRA_TRAIN_ARGS[@]}"
