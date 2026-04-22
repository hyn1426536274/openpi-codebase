# openpi 迁移修改计划

> 目标：在 `/root/Training/ki/openpi` 上实现 **PI05 Subtask**、**FAST**、**Knowledge Isolation (KI)** 训练框架，暂时使用 LIBERO 数据集。
>
> 参考实现：`/root/Training/ki/openpi-comet-test`（改动详见 `framework_changes.md`）

---

## ⭐ 最新进展（2026-04-22）

**PI05_KI Official 权重消融实验、Loss 加权与梯度分组调试**

核心改进：
- ✅ `Pi0Config` 新增 `action_loss_alpha` / `ar_loss_alpha`，用于显式控制 PI05_KI 多 loss 训练中的 flow/action loss 与 AR loss 权重
- ✅ PyTorch 训练与验证路径统一使用加权总 loss：`action_loss_alpha * action + ar_loss_alpha * subtask + ar_loss_alpha * fast`
- ✅ 新增 grouped grad norm 调试能力，按 `vlm_backbone` / `lm_head` / `action_expert` / `other` 分组统计梯度范数和有梯度参数数量
- ✅ 修正 PaliGemma tied LM head 的统计口径：`paligemma.model.language_model.embed_tokens.weight` 归入 `lm_head`，避免 `grad_params/lm_head=0` 的误判
- ✅ 新增 `/workspace/gradcheck.txt` 调试 dump：每次 backward 后记录参数名、是否有梯度、shape 与单参数 grad norm，方便定位梯度流向
- ✅ WandB resume 初始化改为使用 `WANDB_ENTITY` 环境变量，默认回退到当前项目 entity，避免恢复历史 run 时 entity 不一致导致失败
- ✅ 新增 official 权重训练与消融配置：基于 `/workspace/data/pi_official_models/torch/pi05_base` 的 PI05 / PI05_KI official 训练，以及 no-fast、no-subtask、no-KI 三个 component ablation
- ✅ official component ablation 默认开启 grouped grad logging，`grouped_grad_log_interval=100`，用于低开销监控 KI 梯度隔离效果
- ✅ `train_libero.sh` 增加 official / ablation / resume 模板与 `EXTRA_TRAIN_ARGS` 透传，便于中断后按原 `config + exp_name + --resume` 继续训练

**实验配置说明**：
- `libero10_pi05ki_alltasks_official`：PI05_KI，使用从 OpenPI official/JAX 权重转换来的 torch `pi05_base`
- `libero10_pi05_alltasks_official`：非 KI PI05 official baseline
- `libero10_pi05ki_alltasks_official_no_fast`：关闭 FAST AR loss，仅保留 subtask AR loss 与 flow/action loss
- `libero10_pi05ki_alltasks_official_no_subtask`：关闭 subtask AR loss，仅保留 FAST AR loss 与 flow/action loss
- `libero10_pi05ki_alltasks_official_no_ki`：关闭 KI attention detach，用于对比无知识隔离时的梯度耦合

**文件变更清单**：

| 文件 | 改动 |
|------|------|
| `src/openpi/models/pi0_config.py` | 新增 `action_loss_alpha` / `ar_loss_alpha`，支持 PI05_KI loss 权重消融 |
| `scripts/train_pytorch.py` | 多 loss 加权训练/验证；WandB resume entity 修复；新增 gradcheck dump 与 grouped grad norm 统计；修正 tied LM head 分组 |
| `src/openpi/training/config.py` | `TrainConfig` 新增 grouped grad logging 参数；新增 official PI05/PI05_KI 训练配置与 no-fast/no-subtask/no-KI 消融配置 |
| `scripts/runshell/train_libero.sh` | 增加 official 消融与 resume 模板；支持 `EXTRA_TRAIN_ARGS` 透传 |

---

## ⭐ 最新进展（2026-04-17）

**PI05_KI Subtask 推理前缀与训练对齐**

核心改进：
- ✅ `pi0_pytorch.py` 的 `_build_subtask_prefix_tokens()` 不再优先从 `observation.tokenized_prompt + observation.state` 手工重建 `Task: ... State: ...\nSubtask: ` 前缀
- ✅ 推理时优先直接复用 `TokenizeKIInputs` 生成的 `subtask_tokenized_prompt` / `subtask_tokenized_prompt_mask`
- ✅ 利用 `subtask_token_loss_mask` 自动定位 prefix/postfix 分界，只保留训练时真正的 prefix（`Task + State + "\n"`），让模型自己生成完整的 `Subtask: xxx.` postfix
- ✅ 保留旧的手工重建逻辑作为 fallback，兼容不带 KI subtask token 流的旧 observation

**解决的问题**：
- 旧实现是在完整 transform 之后，从 `observation.state` 重新离散化构造 subtask prefix；而 `PadStatesAndActions` 在 `TokenizeKIInputs` 之后执行，推理阶段这里的 `state` 可能已经 pad，和训练时 `SubtaskTokenizer` 看到的 state 不一致
- 旧实现的 prefix 直接包含 `Subtask: `，而训练时 subtask 分支的 target postfix 本身就是从 `Subtask: {subtask}.` 开始，存在 train/infer mismatch
- 新实现直接复用训练同源的 token 前缀，避免了 padded state 和 prefix 起始位置两类不对齐

**行为变化**：
- 推理时 `generate_subtask()` 现在会优先使用 `TokenizeKIInputs` 产出的 subtask token 流
- 如果当前 observation 中存在 `subtask_tokenized_prompt`，则 subtask 生成的 prefix 与训练时 subtask loss 使用的 prefix 保持一致
- 对外接口无变化；`Policy.generate_subtask()`、`auto_subtask` 和服务端 CLI 用法不需要修改

**文件变更清单**：

| 文件 | 改动 |
|------|------|
| `src/openpi/models_pytorch/pi0_pytorch.py` | `_build_subtask_prefix_tokens()` 改为优先复用 `subtask_tokenized_prompt`，并用 `subtask_token_loss_mask` 截取训练同源 prefix；旧的 state 重建逻辑保留为 fallback |

---

## ⭐ 最新进展（2026-04-17）

**推理闭环：PI05_KI Auto-Subtask → Action 自动联动**

核心改进：
- ✅ `Policy.infer()` 新增 `auto_subtask` 支持：在 transform 之前自动调用模型自身的 `generate_subtask()` 生成 subtask，注入 obs 后经 `TokenizeKIInputs` 正确条件化 action expert（与训练一致）
- ✅ Subtask 缓存机制：每 `subtask_refresh_interval` 步重新生成 subtask，避免每步都做 AR 解码
- ✅ 控制完全在服务端：通过 `serve_policy.py` 的 `--auto_subtask` / `--subtask_refresh_interval` CLI 参数控制，评估脚本（client 端）无需任何改动
- ✅ 向后兼容：不启用 `--auto_subtask` 时行为完全不变

**解决的问题**：
- 训练时 action expert 以 subtask 为条件（通过 `TokenizeKIInputs` 的 `FASTTokenizer` 和 `PaligemmaTokenizer`），但推理时直接用 full task prompt → 训练/推理不对称
- 模型自带的 `generate_subtask()` 方法（`pi0_pytorch.py:664-781`）已实现但从未被推理流程调用

**架构设计**：
```
serve_policy.py (--auto_subtask --subtask_refresh_interval 10)
  ↓
policy_config.py (create_trained_policy 透传参数)
  ↓
Policy.__init__(auto_subtask=True, subtask_refresh_interval=10)
  ↓
Policy.infer():
  1. 检查 auto_subtask 标志 + 模型是否支持 generate_subtask
  2. 缓存过期时调用 self.generate_subtask(obs) → AR 解码 subtask 文本
  3. 注入 obs["subtask"] = cached_subtask
  4. 后续 transform pipeline（TokenizeKIInputs）正确使用 subtask 条件化
  ↓
WebSocket 服务器 / 评估脚本（无需改动）
```

**使用方式**：
```bash
# 启动带 auto_subtask 的推理服务
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=libero10_pi05ki_alltasks \
  --policy.dir=/path/to/checkpoint \
  --auto_subtask --subtask_refresh_interval 10

# 评估脚本不需要任何改动
python examples/libero/main.py --task_suite_name libero_10
```

**已知限制**：
- `reset_subtask_cache()` 不会在 episode 边界自动调用（WebSocket 协议无 episode 边界信号），缓存通过 `subtask_refresh_interval` 自然刷新
- 仅支持 PyTorch PI05_KI 模型（JAX 路径不支持 `generate_subtask`）

**文件变更清单**：

| 文件 | 改动 |
|------|------|
| `src/openpi/policies/policy.py` | `__init__` 新增 `auto_subtask`/`subtask_refresh_interval` 参数 + 缓存状态；`infer()` 开头添加 auto_subtask 逻辑；新增 `reset_subtask_cache()` 方法 |
| `src/openpi/policies/policy_config.py` | `create_trained_policy()` 签名新增 `auto_subtask`/`subtask_refresh_interval`，透传到 `Policy()` 构造函数 |
| `scripts/serve_policy.py` | `Args` 新增 `auto_subtask: bool = False` 和 `subtask_refresh_interval: int = 10`；`create_policy()` 透传到 `create_trained_policy()` |

详见 [9.1 Policy 推理层](#91-新增-pi05_ki-推理支持)

---

## ⭐ 进展（2026-04-11）

**按 Task 加载数据：支持 task 级别的数据过滤**

核心改进：
- ✅ `DataConfig` 新增 `tasks: Sequence[str] | None` 和 `episodes_index: Sequence[int] | None` 两个字段（对齐 openpi-comet-test B1K 的设计，数据筛选放在数据配置层）
- ✅ `data_loader.py` 新增 `_resolve_task_episodes()` 函数：从 `meta.tasks` 获取 task_name → task_index 映射，从 `meta.episodes` 按 task 分组 episodes，在每个 task 内按 `episodes_index` 选取
- ✅ `create_torch_dataset()` 自动消费 `DataConfig` 中的 task 过滤参数，与现有 `episodes` 参数取交集
- ✅ `build_datasets()` 无需修改——task 参数通过 `data_config` 透传

**`episodes_index` vs `debug_episodes` 注意事项**：
- `episodes_index`（DataConfig）是 **per-task** 的位置索引——每个 task 内取第 0/1/2… 个 episode
- `debug_episodes`（TrainConfig）是 **全局** 前 N 个 episode 索引（`list(range(N))`）
- 两者同时使用时取交集，但语义不同可能导致空集（如 `debug_episodes=5` 只取 `[0-4]`，而 task_b 的 episodes 是 `[50-99]`）。建议使用 `episodes_index` 替代 `debug_episodes` 做数据量控制

**使用示例**：
```python
data=LeRobotLiberoSubtaskDataConfig(
    repo_id="/workspace/data/libero/libero_10_subtasks_fixed",
    base_config=DataConfig(
        prompt_from_task=True,
        tasks=["turn on the stove and put the moka pot on it"],  # 只训练这个 task
        episodes_index=list(range(20)),  # 每个 task 取前 20 个 episode
    ),
),
```

**文件变更清单**：

| 文件 | 改动 |
|------|------|
| `src/openpi/training/config.py` | `DataConfig` 新增 `tasks` 和 `episodes_index` 字段 |
| `src/openpi/training/data_loader.py` | 新增 `_resolve_task_episodes()`；`create_torch_dataset()` 集成 task 过滤逻辑；新增 `from collections import defaultdict` |

---

## ⭐ 进展（2026-04-10）

**代码质量改进 + Device mismatch 修复 + 命名规范化**

核心改进：
- ✅ **类名重命名**：`LeRobotLiberoV3DataConfig` → `LeRobotLiberoSubtaskDataConfig`（`config.py`），名称更准确地反映其用途（subtask 数据集配置），而非仅与 v3 格式绑定
- ✅ **Device mismatch 修复**（`pi0_pytorch.py`）：`_prepare_attention_masks_4d()` 中 `torch.zeros` / `torch.full` 默认创建 CPU tensor，但输入 `att_2d_masks_4d` 在 CUDA 上，导致 `RuntimeError: Expected all tensors on same device`。修复：提取 `device = att_2d_masks_4d.device` 并传入 tensor 创建函数
- ✅ **Train/Val split 完整性验证**（`data_loader.py`）：在 `create_torch_data_loader_with_val()` 中新增 4 条 assert 检查——episode 不重叠、episode 不遗漏、frame 不重叠、frame 不遗漏
- ✅ **参数命名统一**：`debug_episodes` → `episodes`，涉及 `data_loader.py` 的 `create_data_loader()` 和 `create_torch_data_loader_with_val()`，以及 `train_pytorch.py` 的调用点。新名称更通用，不暗示仅用于调试

**文件变更清单**：

| 文件 | 改动 |
|------|------|
| `src/openpi/training/config.py` | 类名 `LeRobotLiberoV3DataConfig` → `LeRobotLiberoSubtaskDataConfig`，含引用更新 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `_prepare_attention_masks_4d()` 添加 `device=` 参数，修复 CUDA/CPU 混合报错 |
| `src/openpi/training/data_loader.py` | `create_data_loader()` 和 `create_torch_data_loader_with_val()` 参数 `debug_episodes` → `episodes`；新增 train/val split 的 4 条 assert |
| `scripts/train_pytorch.py` | 调用 `episodes=debug_eps`（对齐参数重命名） |

---

## ⭐ 进展（2026-04-09 v2）

**正式迁移到 lerobot 0.4.4 + LeRobot v3.0 数据集支持**

核心改进：
- ✅ `pyproject.toml` 锁定 `lerobot==0.4.4`（从 PyPI 安装，不再用 git rev），解决 `uv run` 反复降级问题
- ✅ `data_loader.py` 导入路径：`lerobot.datasets.lerobot_dataset`（0.4.4 移除了 `lerobot.common`）
- ✅ `data_loader.py` 的 `PromptFromLeRobotTask` 兼容：0.4.4 的 `meta.tasks` 是 DataFrame，自动转换为 `dict[int, str]`
- ✅ `data_loader.py` 的 `episode_data_index` 重建：使用 `meta.episodes` 的 `dataset_from_index/dataset_to_index` 列（0.4.4 移除了 `episode_data_index` 属性）
- ✅ `config.py` 新增 `LeRobotLiberoSubtaskDataConfig`（原名 `LeRobotLiberoV3DataConfig`）：subtask 数据集 key 映射（`images.agentview_rgb` → `image`，`images.wrist_rgb` → `wrist_image`）
- ✅ `config.py` 的 `pi05_ki_libero_torch_debug` 配置改用 `LeRobotLiberoSubtaskDataConfig`
- ✅ `config.py` 的 `_load_norm_stats()` fallback：自动从 `meta/stats.json` 加载归一化统计
- ✅ 原有 `LeRobotLiberoDataConfig`（v2 key 映射）保持不变，兼容 `/workspace/data/libero/lerobot` 等 v2 数据集

**lerobot 0.4.4 vs 0.1.0 关键 API 差异**：

| 项目 | lerobot 0.1.0 | lerobot 0.4.4 |
|------|--------------|---------------|
| 导入路径 | `lerobot.common.datasets.lerobot_dataset` | `lerobot.datasets.lerobot_dataset` |
| `meta.tasks` 类型 | `dict[int, str]` | `pandas.DataFrame` |
| tasks 文件 | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| episode 边界 | `dataset.episode_data_index` | `meta.episodes[dataset_from_index/dataset_to_index]` |
| `__getitem__` | 不注入 task/subtask | 自动注入 `task` 和 `subtask` 字段 |

**快速开始**：
```bash
uv run scripts/train_pytorch.py pi05_ki_libero_torch_debug --exp_name my_run
```

详见 [8.4 实现更新](#84--2026-04-09-lerobot-v30-兼容性修复)

---

## ⭐ 进展（2026-04-08）

**PI05_KI 真实 Subtask 数据管道已实装**

核心改进：
- ✅ 实现 `_EnsureSubtask` 类，支持**三层优先级**处理 subtask 来源
- ✅ 新增 `pi05_ki_libero_torch_debug` 训练配置，指向 `/workspace/data/libero/libero_10_subtasks_fixed`
- ✅ 支持多种数据集格式：直接字段 → 索引映射 → 伪标注回退

详见 [8.3 实现更新](#83--2026-04-08-实现更新真实-subtask-数据支持)

---

## 目录

1. [全局概览](#1-全局概览)
2. [模型定义层](#2-模型定义层)
3. [Tokenizer 层](#3-tokenizer-层)
4. [Transforms 层](#4-transforms-层)
5. [模型前向传播层（PyTorch）](#5-模型前向传播层pytorch)
6. [Attention 机制层（Knowledge Isolation）](#6-attention-机制层knowledge-isolation)
7. [训练配置层](#7-训练配置层)
8. [数据加载层（LIBERO）](#8-数据加载层libero)
9. [Policy 推理层](#9-policy-推理层)
10. [训练脚本层](#10-训练脚本层)
11. [LoRA 与 merge_lora](#11-lora-与-merge_lora)
12. [修改文件汇总表](#12-修改文件汇总表)
13. [建议实施顺序](#13-建议实施顺序)

---

## 1. 全局概览

### 三个框架的关系

```
┌─────────────────────────────────────────────────────────┐
│                    PI05 KI 训练框架                       │
│                                                         │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  PI05 Subtask │  │    FAST AR   │  │  KI Attention│ │
│  │ （子任务生成头）│  │ （动作离散化）│  │（梯度隔离机制）│ │
│  └──────┬────────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┴──────────────────┘         │
│                     联合训练损失                           │
│   loss = flow_matching + λ_fast * fast_ar + λ_sub * sub │
└─────────────────────────────────────────────────────────┘
```

- **PI05 Subtask**：在现有 PI05 基础上，增加一个子任务文本生成头（autoregressive），模型能从当前观测推断出下一步子任务描述
- **FAST**：将动作序列编码为离散 token，模型同时做 flow-matching 和 AR 预测
- **Knowledge Isolation (KI)**：在 attention 计算中，将 prefix（视觉+语言）的 K/V detach，防止 action expert 的梯度污染 VLM backbone

### openpi 与 openpi-comet 的主要差异（影响迁移的部分）

| 方面 | openpi | openpi-comet |
|------|--------|--------------|
| 已有 ModelType | PI0, PI0_FAST, PI05 | PI0, PI05 |
| FAST 实现 | 已有 FASTTokenizer, TokenizeFASTInputs | 有 |
| LIBERO 支持 | 已有 libero_policy.py + 配置 | 无 |
| 训练脚本 | train_pytorch.py | train_pytorch.py + train_pytorch_test.py |
| KI 机制 | **无** | 有 eager_attention_forward_ki() |
| SubtaskTokenizer | **无** | 有 |
| PI05_KI 模型类型 | **无** | 有 |

---

## 2. 模型定义层

### 文件：`src/openpi/models/model.py`

#### 2.1 扩展 ModelType 枚举

```python
# 现有（openpi baseline）
class ModelType(enum.Enum):
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"

# 修改后：新增 PI05_KI
class ModelType(enum.Enum):
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
    PI05_KI = "pi05_ki"    # ← 新增：PI05 + 子任务生成 + KI 梯度隔离
```

#### 2.2 扩展 Observation dataclass

在现有字段基础上，新增 8 个可选字段，支持 subtask 和 FAST 的并行 token 流：

```python
@dataclasses.dataclass
class Observation:
    # 现有字段（不变）
    images: dict[str, Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, Bool[ArrayT, "*b"]]
    state: Float[ArrayT, "*b s"]
    tokenized_prompt: Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None = None
    token_ar_mask: Int[ArrayT, "*b l"] | None = None       # PI0_FAST 用
    token_loss_mask: Bool[ArrayT, "*b l"] | None = None    # PI0_FAST 用

    # 新增：subtask 生成头的 token 流
    subtask_tokenized_prompt: Int[ArrayT, "*b l"] | None = None
    subtask_tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None = None
    subtask_token_ar_mask: Int[ArrayT, "*b l"] | None = None
    subtask_token_loss_mask: Bool[ArrayT, "*b l"] | None = None

    # 新增：KI 训练时的 FAST token 流
    fast_tokenized_prompt: Int[ArrayT, "*b l"] | None = None
    fast_tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None = None
    fast_token_ar_mask: Int[ArrayT, "*b l"] | None = None
    fast_token_loss_mask: Bool[ArrayT, "*b l"] | None = None
```

**注意**：`from_dict()` 方法需同步更新，识别这 8 个新字段。

---

### 文件：`src/openpi/models/pi0_config.py`

#### 2.3 新增 PI05_KI 相关配置字段

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    # 现有字段（不变）...
    pi05: bool = False
    discrete_state_input: bool = None

    # 新增字段
    pi05_ki: bool = False                               # ← 启用 PI05+KI 模式
    fast_model_tokenizer: str | None = None             # ← FAST tokenizer HF repo ID
    fast_model_tokenizer_kwargs: dict | None = None     # ← 额外初始化参数

    @property
    def model_type(self):
        if self.pi05_ki:
            return ModelType.PI05_KI
        if self.pi05:
            return ModelType.PI05
        # ... 现有逻辑不变

    def __post_init__(self):
        # 现有逻辑...
        if self.pi05_ki:
            # pi05_ki 强制使用 PI05 的 tokenizer 设置
            object.__setattr__(self, "max_token_len", 200)
            object.__setattr__(self, "discrete_state_input", True)
```

---

## 3. Tokenizer 层

### 文件：`src/openpi/models/tokenizer.py`

#### 3.1 新增 SubtaskTokenizer 类

该类用于将"当前任务描述 + 机器人状态 → 子任务文本"进行 tokenize，并生成训练所需的 loss mask（只在 subtask 文本部分计算损失）。

```python
class SubtaskTokenizer:
    """
    为 PI05_KI 的子任务生成头生成 token 序列。

    输入：
      - prompt (str): 高层任务描述，如 "Turn on the radio."
      - state (np.ndarray): 离散化后的机器人状态 token
      - subtask (str): 当前子任务标注，如 "pick up radio from table"

    输出：四元组 (tokens, mask, ar_mask, loss_mask)，每个 shape=[max_len]

    序列格式：
      [BOS] "Task: {prompt}, State: {state_str};\n"  ← prefix（不计算损失）
      "Subtask: {subtask}." [EOS]                     ← postfix（计算损失）

    Mask 说明：
      - ar_mask:   prefix=0（双向注意力），postfix=1（causal 自回归）
      - loss_mask: prefix=False，postfix=True
    """
    def __init__(self, max_len: int = 200):
        self._tokenizer = _load_sentencepiece_tokenizer()  # 复用现有 paligemma tokenizer
        self.max_len = max_len

    def tokenize(
        self,
        prompt: str | np.ndarray,
        state: np.ndarray | None = None,
        subtask: str | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1. 构建 prefix: "Task: {text}, State: {state_str};\n"
        # 2. 构建 postfix: "Subtask: {subtask}." + EOS
        # 3. 拼接、截断/填充到 max_len
        # 4. 生成 ar_mask 和 loss_mask
        ...
```

#### 3.2 修改现有 PaligemmaTokenizer

在 `tokenize()` 中对输入文本添加 `.capitalize()` 规范化（与 openpi-comet-test 对齐）：

```python
def tokenize(self, prompt, state=None):
    if isinstance(prompt, np.ndarray):
        prompt = str(bytes(prompt[prompt != 0]).decode("utf-8"))
    prompt = prompt.replace("_", " ").replace("\n", " ").strip().capitalize()  # ← 加 .capitalize()
    ...
```

---

## 4. Transforms 层

### 文件：`src/openpi/transforms.py`

#### 4.1 新增 TokenizeKIInputs transform

这是 PI05_KI 训练模式的核心 transform，在一次调用中同时生成三路 token 流：

```python
@dataclasses.dataclass(frozen=True)
class TokenizeKIInputs(DataTransformFn):
    """
    PI05_KI 模式下的联合 tokenization。

    同时处理：
      1. 主任务 prompt (PaligemmaTokenizer)  → tokenized_prompt + mask
      2. 子任务生成头 (SubtaskTokenizer)     → subtask_tokenized_prompt + ar/loss masks
      3. FAST 动作 token (FASTTokenizer)    → fast_tokenized_prompt + ar/loss masks

    输入 data 中需要有：
      - "prompt": 高层任务描述文本
      - "subtask": 子任务文本（来自 annotation 中的 skill_prompt）
      - "observation/state" 或 "state": 机器人状态
      - "actions": 动作序列（FASTTokenizer 需要）
    """
    tokenizer: PaligemmaTokenizer
    subtask_tokenizer: SubtaskTokenizer
    fast_tokenizer: FASTTokenizer
    discrete_state_input: bool = True

    def __call__(self, data: dict) -> dict:
        prompt = data.pop("prompt", None)
        subtask = data.pop("subtask", None)
        state = data.get("state") if self.discrete_state_input else None
        actions = data.get("actions")

        # 1. 主任务 tokenize
        tokens, mask = self.tokenizer.tokenize(prompt, state)
        data["tokenized_prompt"] = tokens
        data["tokenized_prompt_mask"] = mask

        # 2. subtask tokenize（若 subtask 可用）
        if subtask is not None:
            sub_tokens, sub_mask, sub_ar, sub_loss = self.subtask_tokenizer.tokenize(
                prompt, state, subtask
            )
            data["subtask_tokenized_prompt"] = sub_tokens
            data["subtask_tokenized_prompt_mask"] = sub_mask
            data["subtask_token_ar_mask"] = sub_ar
            data["subtask_token_loss_mask"] = sub_loss

        # 3. FAST tokenize
        if actions is not None:
            f_tokens, f_mask, f_ar, f_loss = self.fast_tokenizer.tokenize(
                prompt, state, actions
            )
            data["fast_tokenized_prompt"] = f_tokens
            data["fast_tokenized_prompt_mask"] = f_mask
            data["fast_token_ar_mask"] = f_ar
            data["fast_token_loss_mask"] = f_loss

        return data
```

#### 4.2 新增 ExtractSubtaskOutput transform（推理端）

```python
@dataclasses.dataclass(frozen=True)
class ExtractSubtaskOutput(DataTransformFn):
    """
    推理时从模型输出中解码子任务文本。
    将模型生成的 token IDs 解码为字符串。
    """
    tokenizer: SubtaskTokenizer

    def __call__(self, data: dict) -> dict:
        if "subtask_tokens" in data:
            data["generated_subtask"] = self.tokenizer.decode(data.pop("subtask_tokens"))
        return data
```

---

## 5. 模型前向传播层（PyTorch）

### 文件：`src/openpi/models_pytorch/pi0_pytorch.py`

这是改动最大的文件，需要支持三路损失的联合训练和 KI 推理。

#### 5.1 添加 KI 模式标记和子任务头

在 `PI0Pytorch.__init__()` 中新增：

```python
def __init__(self, config: Pi0Config):
    super().__init__()
    self.pi05 = config.pi05 or config.pi05_ki
    self.pi05_ki = config.pi05_ki     # ← 新增

    # 现有层（不变）...

    # 新增：子任务生成头（与主任务 prompt 共享 embedding）
    # 注意：embedding 层来自 paligemma_with_expert，不需要新建，只需要在 forward 中传入
    # SubtaskHead 实际上复用 Gemma LM head（语言模型头），不需要单独定义
```

#### 5.2 修改 forward() 支持三路联合损失

```python
def forward(self, observation: Observation, actions: torch.Tensor,
            noise=None, time=None) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Returns:
      - PI0/PI05 模式（不变）：返回 flow_matching loss tensor
      - PI05_KI 模式：返回 dict {
            "flow_loss": ...,
            "subtask_loss": ...,   # CE loss on subtask tokens
            "fast_loss": ...,      # CE loss on FAST tokens
        }
    """
    # -------- 现有 flow-matching 部分（基本不变）--------
    # 1. 预处理 observation
    # 2. 采样噪声和时间
    # 3. 计算 x_t, u_t
    # 4. embed prefix（语言 + 图像）
    # 5. embed suffix（state + action + timestep）

    if not self.pi05_ki:
        # 现有逻辑，直接返回 flow_loss（不变）
        ...
        return flow_loss

    # -------- PI05_KI 额外逻辑 --------
    # 6. 联合前向：prefix + suffix 共同经过 KI attention
    #    （在 paligemma_with_expert.forward 中传入 use_ki=True）
    prefix_out, suffix_out, lm_logits = self.paligemma_with_expert.forward(
        ...,
        use_ki=True,                               # ← 启用 KI attention
        subtask_tokens=observation.subtask_tokenized_prompt,
        fast_tokens=observation.fast_tokenized_prompt,
    )

    # 7. Flow-matching loss（与现有完全一致）
    v_t = self.action_out_proj(suffix_out)
    flow_loss = F.mse_loss(u_t, v_t, reduction="none").mean()

    # 8. Subtask AR loss（使用 Gemma LM head 计算 CE）
    subtask_loss = self._compute_lm_loss(
        lm_logits,                                 # 语言模型 logits
        observation.subtask_tokenized_prompt,      # target tokens
        observation.subtask_token_loss_mask,       # 只在 postfix 计算
    )

    # 9. FAST AR loss
    fast_loss = self._compute_lm_loss(
        lm_logits,
        observation.fast_tokenized_prompt,
        observation.fast_token_loss_mask,
    )

    return {
        "flow_loss": flow_loss,
        "subtask_loss": subtask_loss,
        "fast_loss": fast_loss,
    }

def _compute_lm_loss(self, logits, target_tokens, loss_mask):
    """计算语言模型 cross-entropy loss，只在 loss_mask=True 的位置计算。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_tokens[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    loss = (loss * shift_mask.view(-1).float()).sum() / (shift_mask.float().sum() + 1e-8)
    return loss
```

#### 5.3 新增 forward_language_model() 方法

用于纯 LM 推理（生成 subtask）：

```python
def forward_language_model(
    self,
    prefix_embeds,      # 已经计算好的 prefix KV cache
    input_tokens,       # 新输入 token
    past_key_values=None,
) -> tuple[torch.Tensor, any]:
    """
    自回归生成一步。用于推理时生成子任务文本。
    复用 paligemma_with_expert 的 prefix KV cache。
    """
    ...

def generate_subtask(
    self,
    observation: Observation,
    max_new_tokens: int = 30,
) -> list[str]:
    """
    推理时：从当前观测自回归生成子任务描述。
    1. 计算 prefix KV cache（一次）
    2. 用 SubtaskTokenizer 构建生成起始序列 "Task: ..., State: ...;\nSubtask: "
    3. 自回归 decode 到 EOS 或 max_new_tokens
    4. 返回解码后的文本列表
    """
    ...
```

#### 5.4 添加 LoRA 支持（复用 openpi-comet-test 实现）

```python
def apply_lora(self, rank: int = 16, alpha: float = 16.0, target_modules=None):
    """
    对 paligemma_with_expert 中的 attention Q/K/V/O 投影层应用 LoRA。
    冻结除 LoRA 参数外的所有权重。
    """
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    self.paligemma_with_expert = get_peft_model(
        self.paligemma_with_expert, lora_config
    )
```

---

## 6. Attention 机制层（Knowledge Isolation）

### 文件：`src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`

这是 KI 框架的核心，需要新增一个特殊的 attention 函数。

#### 6.1 新增 eager_attention_forward_ki() 函数

```python
def eager_attention_forward_ki(
    module,
    query,         # shape: [B, n_heads, seq_len, head_dim]
    key,           # shape: [B, n_kv_heads, seq_len, head_dim]
    value,         # shape: [B, n_kv_heads, seq_len, head_dim]
    attention_mask,
    prefix_len: int,        # prefix 部分的长度（VLM backbone token 数）
    scaling=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Knowledge Isolation Attention：

    将序列分为 prefix（视觉+语言 token）和 suffix（action token）两段。

    对于 suffix 的 query，在 attend to prefix K/V 时使用 detach()：
      - suffix_q attend to prefix_k.detach() 和 prefix_v.detach()
      - 这样 action expert 的梯度无法通过 prefix K/V 反传到 VLM backbone

    对于 prefix 的 query，正常 attend to 所有 K/V（无 detach）。

    图示：
      prefix_q → attend → prefix_k (梯度正常流动)
      suffix_q → attend → prefix_k.detach() + suffix_k (仅 suffix 有梯度)
    """
    scaling = scaling or (module.head_dim ** -0.5)

    # 分离 prefix 和 suffix 的 query
    prefix_q = query[:, :, :prefix_len, :]
    suffix_q = query[:, :, prefix_len:, :]

    # 对于 prefix query：正常 attention（双向，含完整 K/V）
    # 对于 suffix query：K/V 中的 prefix 部分 detach
    key_for_suffix = torch.cat([key[:, :, :prefix_len, :].detach(),
                                 key[:, :, prefix_len:, :]], dim=2)
    val_for_suffix = torch.cat([value[:, :, :prefix_len, :].detach(),
                                 value[:, :, prefix_len:, :]], dim=2)

    # 分别计算 attention
    # prefix attention
    prefix_attn_out = _sdpa(prefix_q, key, value, attention_mask[..., :prefix_len, :], scaling)
    # suffix attention
    suffix_attn_out = _sdpa(suffix_q, key_for_suffix, val_for_suffix,
                             attention_mask[..., prefix_len:, :], scaling)

    attn_out = torch.cat([prefix_attn_out, suffix_attn_out], dim=2)
    return attn_out, None
```

#### 6.2 修改 eager_attention_forward() 的调用路径

在 `GemmaAttention.forward()` 中，根据 `use_ki` 标志选择不同的 attention 函数：

```python
def forward(self, hidden_states, attention_mask, position_ids, past_key_values,
            cache_position, use_ki=False, prefix_len=0, **kwargs):
    ...
    if use_ki and prefix_len > 0:
        attn_output, _ = eager_attention_forward_ki(
            self, query_states, key_states, value_states,
            attention_mask, prefix_len=prefix_len
        )
    else:
        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states,
            attention_mask, **kwargs
        )
    ...
```

---

### 文件：`src/openpi/models_pytorch/gemma_pytorch.py`

#### 6.3 修改 PaliGemmaWithExpertModel.forward() 支持 KI 模式

在 `compute_layer_complete()` 中传递 `use_ki` 和 `prefix_len` 参数：

```python
def compute_layer_complete(
    self,
    layer_idx,
    prefix_embeds, prefix_mask, prefix_pos_ids,
    suffix_embeds, suffix_mask, suffix_pos_ids,
    adarms_cond=None,
    use_ki=False,        # ← 新增
):
    """
    拼接 prefix 和 suffix，通过 VLM 和 expert 层进行联合 attention。
    use_ki=True 时：suffix query 在 attend prefix K/V 时使用 detach。
    """
    prefix_len = prefix_embeds.shape[1]

    # 拼接
    combined_embeds = torch.cat([prefix_embeds, suffix_embeds], dim=1)
    combined_mask = ...  # 构建联合 attention mask

    # VLM 层前向（prefix 部分）
    vlm_layer = self.paligemma.language_model.model.layers[layer_idx]
    # Expert 层前向（suffix 部分）
    expert_layer = self.gemma_expert.model.layers[layer_idx]

    # 在 attention 计算时传入 use_ki 和 prefix_len
    # （需要修改 GemmaAttention.forward 的调用接口）
    ...
```

#### 6.4 修复 embed_tokens 的 LoRA 兼容问题

```python
# 原始：直接设为 None 可能与 LoRA 不兼容
# self.gemma_expert.model.embed_tokens = None

# 修改：保留 embed_tokens 引用，但在 forward 中显式不使用
# 或者：将 expert 的 embed_tokens 指向 paligemma 的 embed_tokens（共享）
self.gemma_expert.model.embed_tokens = self.paligemma.language_model.model.embed_tokens
```

---

## 7. 训练配置层

### 文件：`src/openpi/training/config.py`

#### 7.1 扩展 ModelTransformFactory

在 `model_transforms()` 中新增 `PI05_KI` 分支：

```python
class ModelTransformFactory:
    def model_transforms(self, model_config: BaseModelConfig) -> Group:
        match model_config.model_type:
            case ModelType.PI0:
                return Group(inputs=[
                    _transforms.TokenizePrompt(
                        PaligemmaTokenizer(model_config.max_token_len)
                    ),
                    _transforms.PadStatesAndActions(model_config),
                ])
            case ModelType.PI05:
                return Group(inputs=[
                    _transforms.TokenizePrompt(
                        PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=True,
                    ),
                    _transforms.PadStatesAndActions(model_config),
                ])
            case ModelType.PI0_FAST:
                # 现有逻辑不变
                ...

            case ModelType.PI05_KI:           # ← 新增
                fast_tokenizer = FASTTokenizer(
                    max_len=model_config.max_token_len,
                    fast_tokenizer_path=model_config.fast_model_tokenizer or "physical-intelligence/fast",
                )
                return Group(
                    inputs=[
                        _transforms.TokenizeKIInputs(
                            tokenizer=PaligemmaTokenizer(model_config.max_token_len),
                            subtask_tokenizer=SubtaskTokenizer(model_config.max_token_len),
                            fast_tokenizer=fast_tokenizer,
                            discrete_state_input=True,
                        ),
                        _transforms.PadStatesAndActions(model_config),
                    ],
                    outputs=[
                        _transforms.ExtractSubtaskOutput(...),  # 推理时解码子任务
                    ],
                )
```

#### 7.2 新增 LIBERO KI 训练配置

```python
# 在 _CONFIGS 列表中新增以下配置：

TrainConfig(
    name="pi05_ki_libero",
    model=Pi0Config(
        pi05_ki=True,
        action_dim=7,
        action_horizon=50,
        max_token_len=200,
        # 可选：加 LoRA
        # paligemma_variant="gemma_2b_lora",
        # action_expert_variant="gemma_300m_lora",
    ),
    weight_loader=CheckpointWeightLoader("s3://..."),  # 从 pi05_libero 预训练权重初始化
    pytorch_weight_path=None,
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        fine_grained_level=1,   # ← 使用 skill 级别的子任务 prompt
    ),
    batch_size=16,
    num_train_steps=50000,
    ...
),

TrainConfig(
    name="pi05_ki_libero_lora",
    # 同上，但使用 LoRA 微调
    model=Pi0Config(
        pi05_ki=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        ...
    ),
    ...
),
```

#### 7.3 扩展 LeRobotLiberoDataConfig 支持 fine_grained_level

```python
@dataclasses.dataclass
class LeRobotLiberoDataConfig(DataConfigFactory):
    repo_id: str = "physical-intelligence/libero"
    fine_grained_level: int = 0    # ← 新增：0=整任务，1=skill级别子任务

    def create(self, assets_dirs, model_config) -> DataConfig:
        # 在 data_transforms 中加入子任务相关的 transform
        data_transforms = Group(inputs=[
            _policies.LiberoInputs(model_config.model_type),
            # 若是 PI05_KI 且 fine_grained_level > 0，data 中会包含 "subtask" 字段
        ])
        ...
```

---

## 8. 数据加载层（LIBERO）

### 文件：`src/openpi/training/data_loader.py`

#### 8.1 修改 create_torch_dataset() 支持 subtask 字段

LIBERO 使用 LeRobot 格式，每个 item 有 `"task"` 字段（整任务描述）。PI05_KI 模式需要额外提供 `"subtask"` 字段（当前帧所在的 skill 描述）。

```python
def create_torch_dataset(config: DataConfig, model_config: BaseModelConfig) -> Dataset:
    """
    在现有逻辑基础上，若 model_config 是 PI05_KI 模式：
    - 加载 annotations 文件（含 skill_annotation + frame_duration）
    - 为每帧 lookup 当前 skill 的 prompt 文本
    - 在 item 中注入 "subtask" 字段
    """
    dataset = LeRobotDataset(config.repo_id, ...)

    # 现有：extract "task" -> "prompt"
    dataset = TransformedDataset(dataset, [PromptFromLeRobotTask()])

    # 新增：PI05_KI 模式下 extract "subtask"
    if getattr(model_config, "pi05_ki", False):
        dataset = TransformedDataset(dataset, [SubtaskFromAnnotation(
            annotation_path=config.asset_id,   # 或从 dataset 本身读取
            fine_grained_level=config.fine_grained_level,
        )])

    return dataset
```

#### 8.2 新增 SubtaskFromAnnotation transform

```python
@dataclasses.dataclass(frozen=True)
class SubtaskFromAnnotation(DataTransformFn):
    """
    根据当前帧的 frame_index，从 annotation 文件中查找对应的子任务文本。

    annotation 格式（与 openpi-comet-test 中的 build_orchestrator_levels_from_annotations 一致）：
      level 1: skill 级别，每个 skill 有 start_frame/end_frame/task 字段

    输出：在 data 中注入 "subtask" 字段（字符串）。
    若当前帧不在任何 skill 范围内，subtask = "" (空字符串，loss_mask 全 False)。
    """
    annotation_dir: str
    fine_grained_level: int = 1

    # 内部：按 episode_index 缓存 annotation
    _cache: dict = dataclasses.field(default_factory=dict, init=False, compare=False)

    def __call__(self, data: dict) -> dict:
        ep_idx = data.get("episode_index")
        frame_idx = data.get("frame_index")
        annotation = self._load_annotation(ep_idx)

        subtask = ""
        for seg in annotation.get(self.fine_grained_level, []):
            if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
                subtask = seg["task"]
                break

        data["subtask"] = subtask
        return data
```

**注意**：LIBERO 数据集是否已有 annotation 文件需要确认。如果没有，有两个方案：
- **方案 A**：用 `format_skill_prompt()` 从 LIBERO 任务描述自动生成 pseudo-subtask（简单）
- **方案 B**：手动标注或用 openpi-comet-test 中的 annotation pipeline 生成（完整）

暂时建议先用**方案 A**，后续升级到方案 B。

#### 8.3 ⭐ 2026-04-08 实现更新：真实 Subtask 数据支持

实现已升级为支持**真实 subtask 标注**的数据集（如 LeRobot v3 格式的 `libero_10_subtasks_fixed`）。

**变更**：
- 移除 `_SubtaskFromPrompt`，替换为更鲁棒的 `_EnsureSubtask` 类
- `_EnsureSubtask` 支持三层优先级：
  1. **直接字段**：优先使用数据中已存在的 `subtask` 字符串字段（真实标注）
  2. **索引映射**：若无直接字段，尝试通过 `subtask_index` 查询 `meta/subtasks.json` 映射
  3. **伪标注回退**：若上述都无，使用 `prompt` 作为伪 subtask（向后兼容）

```python
class _EnsureSubtask:
    """Ensures 'subtask' field exists, preserving real annotations when present."""
    
    def __init__(self, subtasks_mapping: dict[int, str] | None = None):
        self._subtasks_mapping = subtasks_mapping
    
    def __call__(self, data: dict) -> dict:
        # 优先级 1：已有 subtask 字段 → 保留
        if data.get("subtask") is not None:
            return data
        
        # 优先级 2：索引映射（若有）
        if self._subtasks_mapping is not None and "subtask_index" in data:
            idx = int(data["subtask_index"])
            if idx in self._subtasks_mapping:
                data["subtask"] = self._subtasks_mapping[idx]
                return data
        
        # 优先级 3：回退到 prompt
        data["subtask"] = data.get("prompt", "")
        return data
```

**新增配置**：`LeRobotLiberoSubtaskDataConfig`
- 处理 LeRobot v3 key 命名（`observation.images.image` → `image`）
- 支持 `action_sequence_keys=("action",)`（v3 使用 `action` 而非 `actions`）
- 自动检测数据中的 subtask 字段（无需外部 annotation 文件）

**新增训练配置**：`pi05_ki_libero_torch_debug`
```python
TrainConfig(
    name="pi05_ki_libero_torch_debug",
    model=Pi0Config(pi05_ki=True, action_horizon=10),
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="/workspace/data/libero/libero_10_subtasks_fixed",
        base_config=DataConfig(prompt_from_task=True),
    ),
    # ... 其他参数同 pi05_libero_torch_debug
)
```

**数据流**：
```
libero_10_subtasks_fixed (LeRobot v3, 含 subtask 字符串字段)
  ↓
LeRobotDataset (自动加载 subtask 字段)
  ↓
PromptFromLeRobotTask (添加 prompt)
  ↓
_EnsureSubtask (检查 → 保留 → 不做任何改动)
  ↓
RepackTransform (v3 key 映射)
  ↓
LiberoInputs / TokenizeKIInputs / forward() → 三路 loss
```

**迁移指南**：
- 已有真实 subtask 标注的数据集：直接使用 `_EnsureSubtask`，无需额外配置
- 仅有 `subtask_index` 的数据集：提供 `meta/subtasks.json` 映射文件，`_load_subtasks_mapping()` 自动加载
- 无 subtask 标注的数据集：自动回退到 prompt-based pseudo-subtask，行为同原 `_SubtaskFromPrompt`

#### 8.4 ⭐ 2026-04-09 LeRobot v3.0 兼容性修复（正式迁移到 lerobot 0.4.4）

**问题背景**：原始 openpi 使用 `lerobot==0.1.0`（通过 git rev 锁定），只支持 v2.1 格式数据集。现在需要支持 v3.0 格式的 `libero_10_subtasks_fixed` 数据集。

**变更 1：`pyproject.toml` — 锁定 lerobot 0.4.4**

```toml
# 旧
dependencies = ["lerobot", ...]
[tool.uv.sources]
lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864..." }

# 新
dependencies = ["lerobot==0.4.4", ...]
# [tool.uv.sources] 中移除 lerobot 的 git 源
```

效果：`uv run` 不再降级 lerobot，从 PyPI 安装稳定版本。

**变更 2：`data_loader.py` — lerobot 导入路径更新**

```python
# 旧（lerobot 0.1.0）
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# 新（lerobot 0.4.4，lerobot.common 已移除）
import lerobot.datasets.lerobot_dataset as lerobot_dataset
```

**变更 3：`data_loader.py` — `PromptFromLeRobotTask` 兼容 DataFrame**

lerobot 0.4.4 的 `meta.tasks` 返回 `pandas.DataFrame`（index=task text, column=task_index），不再是 `dict[int, str]`。

```python
# 自动转换 DataFrame → dict[int, str]
tasks = dataset_meta.tasks
if hasattr(tasks, "iterrows"):
    tasks_dict = {int(row["task_index"]): str(idx) for idx, row in tasks.iterrows()}
else:
    tasks_dict = tasks  # lerobot <= 0.1.0 兼容
```

**变更 4：`data_loader.py` — episode_data_index 重建**

lerobot 0.4.4 移除了 `LeRobotDataset.episode_data_index` 属性。改用 `meta.episodes` 的 `dataset_from_index` / `dataset_to_index` 列：

```python
if hasattr(raw_ds, "episode_data_index"):
    # lerobot <= 0.1.0
    episode_data_index = raw_ds.episode_data_index
elif hasattr(raw_ds, "meta") and raw_ds.meta.episodes is not None:
    # lerobot >= 0.4.4
    episodes_table = raw_ds.meta.episodes
    episode_data_index = {
        "from": torch.tensor(episodes_table["dataset_from_index"]),
        "to": torch.tensor(episodes_table["dataset_to_index"]),
    }
```

**变更 5：`config.py` — 新增 `LeRobotLiberoSubtaskDataConfig`（原名 `LeRobotLiberoV3DataConfig`）**

v3 数据集 `libero_10_subtasks_fixed` 的 feature key 与 `lerobot` 数据集不同，需要独立的 RepackTransform。

注意：`RepackTransform` 的 dict 中，**key = 目标 key（LiberoInputs 期望的）**，**value = 源 key（数据集中的）**。

| 配置 | 数据集源 key | RepackTransform 目标 key |
|---|---|---|
| `LeRobotLiberoDataConfig` (lerobot) | `image`, `wrist_image`, `state` | `observation/image`, `observation/wrist_image`, `observation/state` |
| `LeRobotLiberoSubtaskDataConfig` (subtask) | `images.agentview_rgb`, `images.wrist_rgb`, `state` | `observation/image`, `observation/wrist_image`, `observation/state` |

`pi05_ki_libero_torch_debug` 配置改用 `LeRobotLiberoSubtaskDataConfig`。
原有 `LeRobotLiberoDataConfig` 保持不变，兼容 `/workspace/data/libero/lerobot` 数据集。

**变更 6：`config.py` — norm stats fallback**

`_load_norm_stats()` 在找不到 openpi 标准格式时，自动从 `{repo_id}/meta/stats.json` 加载。
注意：openpi 只归一化 `state` 和 `actions`，图像不参与归一化，所以 v3 stats.json 中的图像 key 不匹配不影响训练。

---

## 9. Policy 推理层

### 文件：`src/openpi/policies/policy.py`

#### 9.1 新增 PI05_KI 推理支持

**已实现方案（2026-04-17）**：Auto-subtask 在 `Policy.infer()` 中完成，控制通过构造函数参数传入（由 `serve_policy.py` CLI 参数驱动），评估脚本无需改动。

```python
class Policy(BasePolicy):
    def __init__(self, model, *, ..., auto_subtask=False, subtask_refresh_interval=10):
        # ... 现有初始化 ...
        self._auto_subtask = auto_subtask
        self._subtask_refresh_interval = subtask_refresh_interval
        self._cached_subtask: str | None = None
        self._subtask_step_counter: int = 0
        # PyTorch 模型：绑定 generate_subtask 方法
        if self._is_pytorch_model:
            self._generate_subtask = getattr(model, "generate_subtask", None)

    def infer(self, obs: dict, *, noise=None) -> dict:
        # --- Auto-subtask（transform 之前执行）---
        if self._auto_subtask and self._generate_subtask is not None:
            if self._cached_subtask is None or self._subtask_step_counter >= self._subtask_refresh_interval:
                subtask_result = self.generate_subtask(obs)
                self._cached_subtask = subtask_result.get("subtask")
                self._subtask_step_counter = 0
                logging.info(f"[auto_subtask] Generated subtask: {self._cached_subtask}")
            if self._cached_subtask:
                obs = {**obs, "subtask": self._cached_subtask}
            self._subtask_step_counter += 1
        # ... 后续 transform + 推理逻辑不变 ...

    def reset_subtask_cache(self):
        """Episode 边界调用，重置缓存。"""
        self._cached_subtask = None
        self._subtask_step_counter = 0
```

**关键设计**：subtask 注入发生在 transform 之前，这样 `TokenizeKIInputs` 中的 `FASTTokenizer` 和 `PaligemmaTokenizer` 都能用 subtask 条件化（与训练一致）。

### 文件：`src/openpi/policies/policy_config.py`

#### 9.1.1 透传 auto_subtask 参数

`create_trained_policy()` 新增 `auto_subtask` 和 `subtask_refresh_interval` 参数，直接传给 `Policy()` 构造函数。

### 文件：`scripts/serve_policy.py`

#### 9.1.2 CLI 参数控制

`Args` 新增：
```python
auto_subtask: bool = False          # 启用自动 subtask 生成
subtask_refresh_interval: int = 10  # 每 N 步刷新 subtask
```

`create_policy()` 将这两个参数透传到 `create_trained_policy()`。

### 文件：`src/openpi/policies/libero_policy.py`

#### 9.2 扩展 LiberoInputs 支持 PI05_KI 模式

```python
@dataclasses.dataclass(frozen=True)
class LiberoInputs(DataTransformFn):
    model_type: ModelType = ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # 现有图像映射逻辑（不变）...

        # PI05_KI：right_wrist mask 逻辑与 PI0_FAST 相同
        right_mask = self.model_type in (ModelType.PI0_FAST, ModelType.PI05_KI)

        # 传递 subtask 字段（若存在）
        if "subtask" in data:
            result["subtask"] = data["subtask"]

        return result
```

---

## 10. 训练脚本层

### 文件：`scripts/train_pytorch.py`

#### 10.1 修改 loss 计算，支持多路损失

```python
# 现有（单一 loss）
loss = model(observation, actions)
loss.mean().backward()

# 修改后（PI05_KI 多路 loss）
loss_output = model(observation, actions)

if isinstance(loss_output, dict):
    # PI05_KI 模式
    lambda_fast = config.model.get("fast_loss_weight", 0.1)
    lambda_sub  = config.model.get("subtask_loss_weight", 0.1)

    total_loss = (
        loss_output["flow_loss"]
        + lambda_fast * loss_output["fast_loss"]
        + lambda_sub  * loss_output["subtask_loss"]
    )

    # 分别记录到 wandb
    log_dict = {
        "loss": total_loss.item(),
        "loss/flow": loss_output["flow_loss"].item(),
        "loss/fast": loss_output["fast_loss"].item(),
        "loss/subtask": loss_output["subtask_loss"].item(),
    }
else:
    total_loss = loss_output.mean()
    log_dict = {"loss": total_loss.item()}

total_loss.backward()
```

#### 10.2 新增 loss 权重配置项

在 `Pi0Config` 或 `TrainConfig` 中加入：

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    ...
    fast_loss_weight: float = 0.1      # ← FAST AR loss 权重
    subtask_loss_weight: float = 0.1   # ← Subtask CE loss 权重
```

#### 10.3 修改 prompt 日志解码

```python
# 现有：解码 tokenized_prompt
# 修改：PI05_KI 模式额外解码 subtask_tokenized_prompt
if config.model.pi05_ki and "subtask_tokenized_prompt" in observation:
    subtask_text = decode_tokens(observation.subtask_tokenized_prompt[0])
    pbar.write(f"[subtask step={global_step}] {subtask_text}")
```

---

## 11. LoRA 与 merge_lora

### 新建：`test-scripts/merge_lora.py`

（从 openpi-comet-test 直接移植，做路径适配）

主要功能：
1. 加载含 LoRA 权重的 checkpoint
2. 调用 `peft.merge_and_unload()` 合并 LoRA 到基础权重
3. 保存为标准 `model.safetensors` 格式，供后续推理使用

```python
# 用法示例
python test-scripts/merge_lora.py \
    --checkpoint_dir checkpoints/pi05_ki_libero_lora/step_50000 \
    --output_dir checkpoints/pi05_ki_libero_merged
```

---

## 12. 修改文件汇总表

| 文件 | 改动类型 | 涉及框架 | 优先级 | 状态 |
|------|---------|---------|--------|------|
| `src/openpi/models/model.py` | 扩展枚举 + Observation 字段 | PI05_KI, FAST, Subtask | P0 | ✅ 已完成 |
| `src/openpi/models/pi0_config.py` | 新增 pi05_ki、fast_model_tokenizer 等配置字段；更新 __post_init__ 和 model_type | PI05_KI | P0 | ✅ 已完成 |
| `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` | 新增 `eager_attention_forward_ki()` | KI | P0 | ✅ 已完成 |
| `src/openpi/models_pytorch/gemma_pytorch.py` | `forward()` 和 `compute_layer_complete()` 新增 `knowledge_isolation` 参数 | KI | P0 | ✅ 已完成 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `__init__` 支持 pi05_ki；`forward()` 返回多路 loss dict；新增 `forward_language_model()` | 全部 | P0 | ✅ 已完成（LoRA 留 P2） |
| `src/openpi/models/tokenizer.py` | 新增 SubtaskTokenizer | Subtask | P0 | ✅ 已完成 |
| `src/openpi/transforms.py` | 新增 TokenizeKIInputs, ExtractSubtaskOutput | PI05_KI, FAST, Subtask | P0 | ✅ 已完成 |
| `src/openpi/training/config.py` | ModelTransformFactory 新增 PI05_KI 分支；新增 LeRobotLiberoSubtaskDataConfig (subtask 数据集 key 映射)；pi05_ki_libero_torch_debug 改用 Subtask config；_load_norm_stats fallback | 全部 | P0-P1 | ✅ 已完成 |
| `src/openpi/training/data_loader.py` | 导入路径迁移；PromptFromLeRobotTask 兼容 DataFrame；episode_data_index 从 meta.episodes 重建；_EnsureSubtask | Subtask, 兼容性 | P1 | ✅ 已完成 |
| `pyproject.toml` | lerobot 从 git rev 改为 `==0.4.4`（PyPI） | 兼容性 | P0 | ✅ 已完成 |
| `src/openpi/policies/policy.py` | PI05_KI 推理, generate_subtask, auto_subtask 缓存机制, reset_subtask_cache | Subtask, 推理闭环 | P1 | ✅ 已完成 |
| `src/openpi/policies/policy_config.py` | `create_trained_policy()` 透传 auto_subtask/subtask_refresh_interval | 推理闭环 | P1 | ✅ 已完成 |
| `scripts/serve_policy.py` | Args 新增 auto_subtask/subtask_refresh_interval CLI 参数 | 推理闭环 | P1 | ✅ 已完成 |
| `src/openpi/policies/libero_policy.py` | LiberoInputs 支持 PI05_KI | LIBERO | P1 | ✅ 已完成 |
| `scripts/train_pytorch.py` | 多路损失合并, 分别记录 wandb；新增验证支持 | 全部 | P1 | ✅ 已完成 |
| `test-scripts/merge_lora.py` | 新建（从 comet-test 移植） | LoRA | P2 | ⬜ 未完成 |
| `src/openpi/models_pytorch/preprocessing_pytorch.py` | 轻微调整（对齐 comet-test） | 通用 | P2 | ⬜ 未完成 |

---

## 13. 建议实施顺序

### Phase 1：核心架构（P0，约 2-3 天）

1. ✅ `model.py`：扩展 ModelType + Observation 字段
2. ✅ `pi0_config.py`：新增 PI05_KI 配置
3. ✅ `modeling_gemma.py`：实现 `eager_attention_forward_ki()`
4. ✅ `gemma_pytorch.py`：修改 `compute_layer_complete()` 支持 `use_ki`
5. ✅ `pi0_pytorch.py`：修改 `forward()` 返回多路 loss，添加 `forward_language_model()`

**验证**：写单元测试，确认 KI attention 下 prefix K/V 的梯度为零，suffix 梯度正常。

### Phase 2：Token 化与 Transforms（P0，约 1-2 天）

6. ✅ `tokenizer.py`：实现 `SubtaskTokenizer`
7. ✅ `transforms.py`：实现 `TokenizeKIInputs` 和 `ExtractSubtaskOutput`

**验证**：检查 tokenize 后 loss_mask 正确（只有 subtask postfix 部分为 True）。

### Phase 3：配置与数据（P0-P1，约 1 天）

8. ✅ `training/config.py`：新增 PI05_KI 模型 transform 分支 + LIBERO KI 训练配置
9. ✅ `training/data_loader.py`：新增 `_SubtaskFromPrompt`，PI05_KI 模式下注入 subtask 字段（方案 A pseudo-subtask）
10. ✅ `libero_policy.py`：`LiberoInputs` 支持 PI05_KI（image_mask + subtask 透传）

**验证**：跑 `FakeDataConfig` 的 debug 训练，确认数据 pipeline 通路。

### Phase 4：训练脚本与推理（P1，约 1 天）

11. ✅ `train_pytorch.py`：多路 loss 合并 + wandb 分别记录 `loss/action`、`loss/subtask`、`loss/fast`
12. ✅ `policy.py`：PI05_KI 推理 + subtask 生成（`generate_subtask()`）
13. ✅ `pi0_pytorch.py`：模型侧 `generate_subtask()` 自回归解码实现

**验证**：运行 `pi05_ki_libero` 配置 100 步，确认三路 loss 均在下降。

### Phase 4.5：Train/Val Split（P1）

14. ✅ `training/config.py`：TrainConfig 新增 `val_ratio` / `val_interval` / `val_batches`
15. ✅ `training/data_loader.py`：`_split_episodes()` + `create_torch_data_loader_with_val()` + `create_torch_dataset(episodes=)`
16. ✅ `train_pytorch.py`：`validate()` 函数 + 训练循环中定期验证 + wandb 记录 `val_loss/*`

**验证**：设置 `val_ratio=0.1`，确认 wandb 出现 `val_loss/action` 等指标；设置 `val_ratio=0` 确认行为与原来一致。

### Phase 4.6：推理闭环 — Auto-Subtask（P1）

17. ✅ `policy.py`：`infer()` 添加 auto_subtask 逻辑 + subtask 缓存 + `reset_subtask_cache()`
18. ✅ `policy_config.py`：`create_trained_policy()` 透传 `auto_subtask` / `subtask_refresh_interval`
19. ✅ `serve_policy.py`：CLI 参数 `--auto_subtask` / `--subtask_refresh_interval`

**验证**：启动 `--auto_subtask` 服务，观察日志中 `[auto_subtask] Generated subtask: ...` 输出，确认 subtask 被正确注入 transform pipeline。

### Phase 5：LoRA 与工具（P2，可后续进行）

13. `pi0_pytorch.py`：`apply_lora()` 方法
14. `test-scripts/merge_lora.py`：移植合并工具
15. 新增 `pi05_ki_libero_lora` 训练配置

---

## 附录：关键代码对照

### KI Attention 梯度流动验证

```python
# 验证 KI 正确实现的单元测试伪代码
prefix_k.requires_grad_(True)
suffix_q.requires_grad_(True)

output = eager_attention_forward_ki(q=suffix_q, k=prefix_k, ...)
loss = output.sum()
loss.backward()

assert prefix_k.grad is None, "KI: prefix K 不应收到来自 suffix 的梯度"
assert suffix_q.grad is not None, "suffix Q 应有梯度"
```

### SubtaskTokenizer 输出格式验证

```python
tok = SubtaskTokenizer(max_len=50)
tokens, mask, ar_mask, loss_mask = tok.tokenize(
    prompt="Turn on the radio.",
    state=np.zeros(256),
    subtask="pick up radio from table",
)
# loss_mask 应在 "pick up radio from table." 对应的 token 上为 True
# ar_mask 应在 subtask postfix 上为 1（causal）
assert loss_mask[:prefix_len].sum() == 0   # prefix 不计算 loss
assert loss_mask[prefix_len:].sum() > 0    # postfix 计算 loss
```
