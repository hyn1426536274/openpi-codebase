# Beyond Migration: 后续开发计划

> 本文档记录**超出 framework_changes.md 框架迁移范畴**的改动计划。
> 框架迁移关注的是"把 openpi-comet-test 的功能搬到 openpi"，而本文档关注的是"在 openpi 上做的新设计、改进、以及服务于消融实验的工程工作"。

---

## 目录

1. [消融实验基础设施](#1-消融实验基础设施)
2. [Loss 权重可配置化](#2-loss-权重可配置化)
3. [Subtask 数据：从 Pseudo 到真实标注](#3-subtask-数据从-pseudo-到真实标注)
4. [推理闭环：Subtask → Action 联动](#4-推理闭环subtask--action-联动)
5. [KI 开关与消融变体](#5-ki-开关与消融变体)
6. [LoRA 支持与 Merge 工具](#6-lora-支持与-merge-工具)
7. [评测与指标体系](#7-评测与指标体系)
8. [训练过程验证：Train/Val Split](#8-训练过程验证trainval-split)
9. [已知技术债务](#9-已知技术债务)
10. [实施优先级总览](#10-实施优先级总览)

---

## 1. 消融实验基础设施

### 动机

研究核心是消融三板斧（FAST / Subtask / KI）在 post-training 阶段对 π0.5 的必要性。需要能灵活开关每个组件，并组合出所有实验变体。

### 计划

#### 1.1 训练配置模板化

在 `training/config.py` 中定义一组消融配置，遵循命名规范：

```
pi05_ki_libero_{variant}

variant 命名：
  full          = subtask + fast + ki  （三板斧全开）
  no_ki         = subtask + fast       （无梯度隔离）
  no_fast       = subtask + ki         （无 FAST AR loss）
  no_subtask    = fast + ki            （无 subtask AR loss）
  flow_only     = 仅 flow matching     （社区常见做法，baseline）
  subtask_only  = 仅 subtask + flow
  fast_only     = 仅 fast + flow
  ki_only       = 仅 ki + flow
```

#### 1.2 组件开关机制

在 `Pi0Config` 中新增细粒度开关：

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    # 现有
    pi05_ki: bool = False

    # 新增：消融开关（仅 pi05_ki=True 时生效）
    enable_fast_loss: bool = True      # 是否计算 FAST AR loss
    enable_subtask_loss: bool = True   # 是否计算 subtask AR loss
    enable_ki_attention: bool = True   # 是否在 attention 中启用 KI detach
```

**涉及文件：**
- `src/openpi/models/pi0_config.py` — 新增字段
- `src/openpi/models_pytorch/pi0_pytorch.py` — `forward()` 根据开关决定是否调用 `forward_language_model()`、是否传 `knowledge_isolation=True`
- `src/openpi/training/config.py` — 生成消融配置组合

**验证：** 配置 `flow_only` 变体，确认 forward 只返回 action loss，不调用 `forward_language_model()`。

---

## 2. Loss 权重可配置化

### 现状

当前 `train_pytorch.py` 中三路 loss 直接求和：`loss = sum(losses.values())`，无权重控制。

### 计划

#### 2.1 在 TrainConfig 中新增 loss 权重

```python
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 现有字段...

    # 新增
    fast_loss_weight: float = 1.0
    subtask_loss_weight: float = 1.0
    # action (flow matching) loss 权重隐式为 1.0
```

#### 2.2 修改 train_pytorch.py 的 loss 合并逻辑

```python
if isinstance(losses, dict):
    loss = losses.get("action", 0.0)
    if "fast" in losses:
        loss = loss + config.fast_loss_weight * losses["fast"]
    if "subtask" in losses:
        loss = loss + config.subtask_loss_weight * losses["subtask"]
    per_loss_dict = {f"loss/{k}": v.item() for k, v in losses.items()}
    per_loss_dict["loss/total"] = loss.item()
```

**涉及文件：**
- `src/openpi/training/config.py` — `TrainConfig` 新增字段
- `scripts/train_pytorch.py` — 修改 loss 合并逻辑

**验证：** 设置 `fast_loss_weight=0.0`，确认 wandb 中 `loss/fast` 仍有记录但不影响梯度。

---

## 3. Subtask 数据：从 Pseudo 到真实标注

### 现状

`data_loader.py` 中 `_SubtaskFromPrompt` 将全局 task prompt 复制为 subtask（方案 A），这是权宜之计。

### 计划

#### 3.1 方案 B：基于 LLM 的自动标注 Pipeline

为 LIBERO 数据集生成真实 subtask 标注：

```
输入：LIBERO episode 视频帧序列 + 全局 task 描述
处理：
  1. 将 episode 按关键帧分段（基于 gripper action 变化 / 接触事件）
  2. 对每个分段，用 LLM（GPT-4o / Claude）+ 关键帧图片生成 subtask 描述
  3. 输出 JSON annotation 文件：
     [{
       "episode_id": 0,
       "segments": [
         {"start_frame": 0, "end_frame": 30, "subtask": "reach for the plate on the left"},
         {"start_frame": 31, "end_frame": 60, "subtask": "pick up the plate"},
         ...
       ]
     }]
```

#### 3.2 替换 `_SubtaskFromPrompt` 为 `_SubtaskFromAnnotation`

```python
class _SubtaskFromAnnotation:
    def __init__(self, annotation_path: str):
        self._annotations = self._load(annotation_path)

    def __call__(self, data: dict) -> dict:
        ep_id = data["episode_index"]
        frame_idx = data["frame_index"]
        subtask = self._lookup(ep_id, frame_idx)
        data["subtask"] = subtask
        return data
```

**涉及文件：**
- 新建 `scripts/generate_subtask_annotations.py` — LLM 标注脚本
- `src/openpi/training/data_loader.py` — 新增 `_SubtaskFromAnnotation`
- `src/openpi/training/config.py` — 在 data config 中添加 annotation_path 字段

**验证：** 对 LIBERO-LONG 的一个 episode 手动检查标注质量。

#### 3.3 对比实验

方案 A（pseudo-subtask）vs 方案 B（LLM 标注）本身就是一个有价值的消融变量——LLM 生成 subtask 标注是否比简单复制 task prompt 更好？

---

## 4. 推理闭环：Subtask → Action 联动

### 现状

`policy.py` 中 `generate_subtask()` 已实现，但与 `infer()` 是独立的两个方法。推理时需要外部调用者（如 `eval_wrapper`）手动协调：先调 `generate_subtask()`，再把 subtask 塞回 obs，再调 `infer()`。

### 计划

#### 4.1 在 Policy.infer() 中自动注入 subtask（可选）

```python
def infer(self, obs: dict, *, noise=None, auto_subtask: bool = False) -> dict:
    if auto_subtask and self._generate_subtask is not None:
        subtask_result = self.generate_subtask(obs)
        obs["subtask"] = subtask_result["subtask"]

    # ... 现有推理逻辑 ...
```

这样可以通过 `auto_subtask=True` 一步完成 subtask 生成 + action 推理。

#### 4.2 Subtask 缓存与更新频率

论文中 subtask 不是每步都重新生成的。可以添加缓存机制：

```python
class Policy:
    def __init__(self, ...):
        self._cached_subtask = None
        self._subtask_refresh_interval = 10  # 每 N 步重新生成

    def infer(self, obs, *, noise=None, auto_subtask=False):
        if auto_subtask and self._should_refresh_subtask():
            self._cached_subtask = self.generate_subtask(obs)["subtask"]
        if self._cached_subtask:
            obs["subtask"] = self._cached_subtask
        # ...
```

**涉及文件：**
- `src/openpi/policies/policy.py` — 修改 `infer()` + 添加缓存逻辑

**验证：** 在 LIBERO eval 中对比每步生成 subtask vs 每 10 步生成 subtask 的性能差异。

---

## 5. KI 开关与消融变体

### 现状

KI (Knowledge Isolation) 当前是 `pi05_ki=True` 时硬编码启用的。消融需要能在保持其他框架不变的情况下单独关闭 KI。

### 计划

#### 5.1 独立 KI 开关

在 `pi0_pytorch.py` `forward()` 中：

```python
# 现有逻辑
knowledge_isolation = False
if self.pi05_ki and (fast_prompt is not None and subtask_prompt is not None):
    knowledge_isolation = True

# 修改为
knowledge_isolation = False
if self.pi05_ki and self.config.enable_ki_attention and (...):
    knowledge_isolation = True
```

#### 5.2 KI vs Joint Training 对比

这是一个关键消融维度——原论文中提到的"attention mask isolation"与我们实现的"gradient detach isolation"是否等价？可以实现第三种变体做对比：

| 变体 | 实现方式 | 描述 |
|------|---------|------|
| `ki_detach` | prefix K/V detach | 当前实现，梯度层面隔离 |
| `ki_mask` | attention mask 隔离 | FAST 和 flow matching token 流互不 attend（更接近原论文） |
| `joint` | 无隔离 | 所有 loss 的梯度自由流动 |

`ki_mask` 变体需要修改 attention mask 构建逻辑（`make_att_2d_masks`），而非 attention 计算本身。

**涉及文件：**
- `src/openpi/models/pi0_config.py` — `enable_ki_attention` 开关
- `src/openpi/models_pytorch/pi0_pytorch.py` — 条件分支
- 可选：`pi0_pytorch.py` 新增 attention mask 隔离模式

**验证：** KI 关闭后，确认 prefix K/V 可以收到来自 action loss 的梯度（用简单的梯度检查脚本）。

---

## 6. LoRA 支持与 Merge 工具

### 现状

`modify_history.md` 标记 LoRA 为 P2 未完成。训练配置中已引用 `gemma_2b_lora` 和 `gemma_300m_lora` variant。

### 计划

#### 6.1 `apply_lora()` 方法

在 `PI0Pytorch` 中添加：

```python
def apply_lora(self, rank=16, alpha=16.0):
    from peft import get_peft_model, LoraConfig
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules)
    self.paligemma_with_expert.paligemma = get_peft_model(
        self.paligemma_with_expert.paligemma, lora_config
    )
    self.paligemma_with_expert.gemma_expert = get_peft_model(
        self.paligemma_with_expert.gemma_expert, lora_config
    )
```

#### 6.2 LoRA-aware Checkpoint

修改 `train_pytorch.py` 的 checkpoint 保存逻辑，当检测到 PeftModel 时分别保存 adapter：

```
checkpoint_dir/
  paligemma_lora/     # adapter_model.safetensors + adapter_config.json
  expert_lora/        # adapter_model.safetensors + adapter_config.json
  lora_config.json    # base model path + adapter dir names
  metadata.pt         # Pi0Config
```

#### 6.3 `merge_lora.py` 工具

从 openpi-comet-test 移植，适配 openpi 的权重加载路径：

```bash
python scripts/merge_lora.py \
  --checkpoint_dir checkpoints/pi05_ki_libero_lora/step_50000 \
  --output_dir checkpoints/pi05_ki_libero_merged
```

#### 6.4 LoRA 与 KI 的兼容性

当前 `gemma_pytorch.py` 中有 LoRA unwrapping 逻辑（`hasattr(gemma_model, "model")`）。需要确保：
- KI attention 的 `compute_layer_complete()` 能正确获取被 LoRA 包裹的底层 layer
- `generate_subtask()` 在 LoRA 模式下仍能正确调用 `lm_head`

**涉及文件：**
- `src/openpi/models_pytorch/pi0_pytorch.py` — `apply_lora()`
- `scripts/train_pytorch.py` — LoRA checkpoint 保存
- 新建 `scripts/merge_lora.py`
- `src/openpi/models_pytorch/gemma_pytorch.py` — LoRA unwrapping 验证

**验证：** 用 LoRA 训练 100 步 → 保存 → merge → 加载 merged 模型 → 推理，确认结果一致。

---

## 7. 评测与指标体系

### 现状

训练时记录三路 loss 到 wandb，但缺少系统化的评测。

### 计划

#### 7.1 训练指标

除现有的 `loss/action`、`loss/subtask`、`loss/fast` 外，新增：

| 指标 | 说明 |
|------|------|
| `subtask/accuracy` | subtask AR 预测的 token-level accuracy（在 loss_mask 区域内） |
| `fast/accuracy` | FAST AR 预测的 token-level accuracy |
| `gradient/prefix_norm` | prefix 参数的梯度范数（监控 KI 是否在工作） |
| `gradient/suffix_norm` | suffix 参数的梯度范数 |

#### 7.2 评测脚本

需要为 LIBERO benchmark 搭建评测管线：

```bash
python scripts/eval_libero.py \
  --config pi05_ki_libero_full \
  --checkpoint_dir checkpoints/step_50000 \
  --suite libero_90 \
  --num_episodes 50
```

输出指标：
- 成功率 (success rate)
- 平均完成步数 (average steps to completion)
- 生成的 subtask 质量评估（可选：与 ground-truth subtask 的语义相似度）

#### 7.3 消融结果可视化

创建标准化的结果对比表格模板，方便汇总所有消融变体的结果。

**涉及文件：**
- `scripts/train_pytorch.py` — 新增 accuracy 指标计算
- 新建 `scripts/eval_libero.py` — LIBERO 评测脚本
- 新建 `docs/ablation_results.md` — 结果汇总模板

---

## 8. 训练过程验证：Train/Val Split

### 动机

每次消融实验都跑 LIBERO 仿真来验证效果，成本太高且反馈周期长。需要一种**训练过程中就能初步判断效果**的方法：将 LIBERO 数据按 episode 划分为训练集和验证集，在训练过程中定期计算 val loss，通过 train/val loss 曲线的走势来快速比较不同消融变体的效果。

### 核心设计

#### 8.1 按 Episode 划分 Train/Val

LIBERO 的数据是按 episode 组织的。划分应按 episode 而非按 frame 切分，避免同一条轨迹的帧同时出现在训练集和验证集中（信息泄露）。

`LeRobotDataset` 原生支持 `episodes` 参数来过滤 episode：

```python
# LeRobotDataset 构造函数签名
LeRobotDataset(repo_id, episodes: list[int] | None = None, ...)
```

因此只需在 `create_torch_dataset()` 层面传入不同的 episode 列表即可。

#### 8.2 数据层改动

在 `data_loader.py` 中新增 `create_torch_dataset_with_split()`：

```python
def create_torch_dataset_with_split(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """创建 train/val 分割的数据集。

    按 episode 级别划分，保证同一条轨迹不会同时出现在 train 和 val 中。

    Args:
        val_ratio: 验证集占比（按 episode 数量）。
        seed: 随机种子，确保划分可复现。

    Returns:
        (train_dataset, val_dataset) 元组。
    """
    repo_id = data_config.repo_id
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    total_episodes = dataset_meta.total_episodes
    all_episode_ids = list(range(total_episodes))

    # 固定种子划分
    rng = np.random.RandomState(seed)
    rng.shuffle(all_episode_ids)
    val_count = max(1, int(total_episodes * val_ratio))
    val_episodes = sorted(all_episode_ids[:val_count])
    train_episodes = sorted(all_episode_ids[val_count:])

    logging.info(
        f"Train/Val split: {len(train_episodes)} train episodes, "
        f"{len(val_episodes)} val episodes (ratio={val_ratio})"
    )

    # 创建两个独立的 LeRobotDataset，各自只加载对应 episode
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }
    train_ds = lerobot_dataset.LeRobotDataset(
        repo_id, episodes=train_episodes, delta_timestamps=delta_timestamps
    )
    val_ds = lerobot_dataset.LeRobotDataset(
        repo_id, episodes=val_episodes, delta_timestamps=delta_timestamps
    )

    # 应用相同的 transforms（prompt, subtask 等）
    if data_config.prompt_from_task:
        tasks = dataset_meta.tasks
        train_ds = TransformedDataset(train_ds, [_transforms.PromptFromLeRobotTask(tasks)])
        val_ds = TransformedDataset(val_ds, [_transforms.PromptFromLeRobotTask(tasks)])

    if getattr(model_config, "pi05_ki", False):
        train_ds = TransformedDataset(train_ds, [_SubtaskFromPrompt()])
        val_ds = TransformedDataset(val_ds, [_SubtaskFromPrompt()])

    return train_ds, val_ds
```

#### 8.3 训练配置新增字段

在 `TrainConfig` 中新增验证相关配置：

```python
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 现有字段...

    # 新增：验证集
    val_ratio: float = 0.1               # 验证集比例（按 episode 划分）
    val_interval: int = 500              # 每 N 步计算一次 val loss
    val_batches: int = 10                # 每次验证跑多少个 batch
```

#### 8.4 训练脚本改动

在 `train_pytorch.py` 中加入验证循环：

```python
# ---- 构建数据集 ----
if config.val_ratio > 0:
    train_ds, val_ds = create_torch_dataset_with_split(
        data_config, config.model.action_horizon, config.model,
        val_ratio=config.val_ratio
    )
    # 对 train_ds 和 val_ds 分别应用 transform_dataset
    train_ds = transform_dataset(train_ds, data_config)
    val_ds = transform_dataset(val_ds, data_config)

    # 构建 val DataLoader（不 shuffle，不 DDP 分片）
    val_loader = TorchDataLoader(val_ds, local_batch_size=config.batch_size, shuffle=False)
else:
    # 不划分，使用全量数据训练（现有行为）
    train_ds = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    train_ds = transform_dataset(train_ds, data_config)

# ---- 训练循环中 ----
if (
    config.val_ratio > 0
    and global_step % config.val_interval == 0
    and global_step > 0
    and is_main
):
    val_metrics = validate(model, val_loader, device, config.val_batches)
    wandb.log(val_metrics, step=global_step)
    model.train()  # 恢复训练模式


def validate(model, val_loader, device, num_batches):
    """在验证集上计算 loss，不更新梯度。"""
    model.eval()
    val_losses = defaultdict(list)

    with torch.no_grad():
        for i, (observation, actions) in enumerate(val_loader):
            if i >= num_batches:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(device).float()

            losses = model(observation, actions)

            if isinstance(losses, dict):
                for k, v in losses.items():
                    val_losses[f"val_loss/{k}"].append(v.item())
                val_losses["val_loss/total"].append(sum(v.item() for v in losses.values()))
            else:
                val_losses["val_loss/action"].append(losses.mean().item())

    return {k: sum(v) / len(v) for k, v in val_losses.items()}
```

#### 8.5 wandb 可视化效果

训练过程中 wandb 将同时显示：

| 指标 | 含义 |
|------|------|
| `loss/action` | 训练集 flow-matching loss |
| `loss/subtask` | 训练集 subtask AR loss |
| `loss/fast` | 训练集 FAST AR loss |
| `val_loss/action` | 验证集 flow-matching loss |
| `val_loss/subtask` | 验证集 subtask AR loss |
| `val_loss/fast` | 验证集 FAST AR loss |
| `val_loss/total` | 验证集三路 loss 之和 |

通过对比 `loss/action` vs `val_loss/action` 的走势，可以判断：
- **两者同步下降** → 模型在学到有效表征，没有过拟合
- **train loss 下降但 val loss 上升** → 过拟合，需要调整（early stopping / 增大数据 / 减小模型）
- **不同消融变体的 val loss 对比** → 快速判断哪种组合更有效，无需跑仿真

#### 8.6 与消融实验的配合

消融实验时只需对比不同变体在**相同验证集**上的 `val_loss/action`：

```
pi05_ki_libero_full       val_loss/action = 0.023  ← 三板斧全开
pi05_ki_libero_no_ki      val_loss/action = 0.025  ← 无 KI
pi05_ki_libero_no_fast    val_loss/action = 0.028  ← 无 FAST
pi05_ki_libero_flow_only  val_loss/action = 0.031  ← 仅 flow matching
```

这样在正式跑仿真前就能快速筛选出有潜力的变体。

**涉及文件：**
- `src/openpi/training/data_loader.py` — 新增 `create_torch_dataset_with_split()`
- `src/openpi/training/config.py` — `TrainConfig` 新增 `val_ratio` / `val_interval` / `val_batches`
- `scripts/train_pytorch.py` — 新增 `validate()` 函数 + 训练循环中集成验证

**验证：**
1. 确认 train/val episode 不重叠（打印 episode 列表检查）
2. 跑 100 步，确认 wandb 中出现 `val_loss/*` 指标
3. 确认 val_ratio=0 时行为与原来完全一致（不 break 现有流程）

---

## 9. 已知技术债务

以下问题不阻塞消融实验，但应在后续解决：

| 问题 | 位置 | 描述 | 优先级 |
|------|------|------|--------|
| Image format hack | `preprocessing_pytorch.py` | 存在 TODO：处理 [B,C,H,W] vs [B,H,W,C] 格式不一致的 hack | 低 |
| Gradient checkpointing 强制启用 | `gemma_pytorch.py:139-143` | `compute_layer_complete` 中强制为 expert 启用 gradient checkpointing，应改为从配置控制 | 中 |
| `embed_tokens = None` | `gemma_pytorch.py:59` | Expert 的 embed_tokens 设为 None，与 LoRA/PEFT 不兼容（PEFT 需要完整 module tree） | 高（阻塞 LoRA） |
| Subtask 解析脆弱 | `policy.py:generate_subtask()` | 依赖字符串分割 `"Subtask:"` 和 `"."`，模型生成格式稍有偏差就会失败 | 中 |
| Debug print 残留 | `gemma_pytorch.py:141,146-156` | `_debug_gc_printed` 和 print 语句，应改为 logging 或移除 | 低 |
| `discrete_state_input` 硬约束 | `transforms.py:TokenizeKIInputs` | PI05_KI 路径强制要求 `discrete_state_input=True`，限制了实验灵活性 | 低 |

---

## 10. 实施优先级总览

### Phase A：消融实验基础（下一步优先）

| 任务 | 涉及 | 状态 |
|------|------|------|
| **Train/Val Split + 训练中验证** | §8 | ⬜ |
| Pi0Config 消融开关 (`enable_fast_loss` / `enable_subtask_loss` / `enable_ki_attention`) | §1, §5 | ⬜ |
| Loss 权重可配置化 | §2 | ⬜ |
| 消融配置组合生成 | §1.1 | ⬜ |
| 训练指标：token accuracy + gradient norm | §7.1 | ⬜ |

### Phase B：数据与评测

| 任务 | 涉及 | 状态 |
|------|------|------|
| LLM subtask 自动标注 pipeline | §3.1 | ⬜ |
| `_SubtaskFromAnnotation` 替换 pseudo-subtask | §3.2 | ⬜ |
| LIBERO 评测脚本 | §7.2 | ⬜ |

### Phase C：推理与工具

| 任务 | 涉及 | 状态 |
|------|------|------|
| `infer()` 中 auto_subtask 集成 | §4.1 | ⬜ |
| Subtask 缓存与更新频率控制 | §4.2 | ⬜ |
| LoRA apply / checkpoint / merge | §6 | ⬜ |

### Phase D：技术债务清理

| 任务 | 涉及 | 状态 |
|------|------|------|
| `embed_tokens` LoRA 兼容修复 | §9 | ⬜ |
| Gradient checkpointing 配置化 | §9 | ⬜ |
| Debug print 清理 | §9 | ⬜ |

---

*最后更新：2025-07-06*
