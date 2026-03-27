#!/usr/bin/env bash
# OpenPI 动态多卡分布式训练启动脚本

# wandb 配置 (entity: huangyinuo321-uestc, project: pi05_research)
WANDB_API_KEY="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"
export WANDB_API_KEY

# 1. 环境激活与路径跳转
cd /root/Training/ki/openpi
source .venv/bin/activate 2>/dev/null || true

# 2. 参数处理与 GPU 自动检测
# 获取系统中可用的 GPU 总数
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

# 如果检测失败或为 0，默认设为 1（作为保底）
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    AVAILABLE_GPUS=1
fi

# 参数 1: 显卡数量 (如果未输入，则使用检测到的全部显卡)
NUM_GPUS="${1:-$AVAILABLE_GPUS}"
# 参数 2: 配置名称
CONFIG_NAME="${2:-pi05_libero_torch_debug}"
# 参数 3: 实验名称
EXP_NAME="${3:-pi05_libero_$(date +%Y%m%d_%H%M%S)}"

# 3. 动态生成 CUDA_VISIBLE_DEVICES
# 注意：在某些集群环境下（如使用了 SLURM 或 Docker 限制），
# 物理显卡索引不一定是从 0 到 N-1，但对于大多数九章/H800 机器，
# 默认按顺序排列是安全的。
G_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${G_IDS}"

echo "---------------------------------------"
echo "Detected Total GPUs: ${AVAILABLE_GPUS}"
echo "Using GPUs:          ${CUDA_VISIBLE_DEVICES} (Count: ${NUM_GPUS})"
echo "Config:              ${CONFIG_NAME}"
echo "Exp:                 ${EXP_NAME}"
echo "---------------------------------------"

# 4. 自动应用必要的补丁 (按需取消注释)
# cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ 2>/dev/null || true

# 5. 启动训练
# 使用 torchrun 进行分布式训练
# 注意：如果只使用 1 张卡，torchrun 依然有效，但在多卡（如 H800）下效率更高

# datasets==0.36.0 huggingface-hub=0.32.3 | you may add patch on your code
# see https://github.com/Physical-Intelligence/openpi/issues/561
# # Monkey-patch to fix 'List' feature type error in old datasets
# try:
#     import datasets.features.features as features

#     _OLD_GENERATE_FROM_DICT = features.generate_from_dict

#     def _new_generate_from_dict(obj):
#         if isinstance(obj, dict) and obj.get("_type") == "List":
#             obj["_type"] = "Sequence"
#         return _OLD_GENERATE_FROM_DICT(obj)

#     features.generate_from_dict = _new_generate_from_dict
# except (ImportError, AttributeError):
#     # If datasets or the function doesn't exist, do nothing.
#     pass
# # End of monkey-patch
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --overwrite