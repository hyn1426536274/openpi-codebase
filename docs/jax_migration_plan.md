# JAX 迁移计划：将 PI05_KI / FAST / Subtask / LIBERO 改动迁移到原版 JAX 训练链路

> 目标：把当前已经在共享层和 PyTorch 侧落地的 `PI05_KI` / `FAST` / `Subtask` / `LIBERO v3` 相关改动，系统迁移到原版 JAX 训练入口 [`scripts/train.py`](/workspace/code/openpi-codebase/scripts/train.py)，并以 JAX 模型实现 [`src/openpi/models/pi0.py`](/workspace/code/openpi-codebase/src/openpi/models/pi0.py) 为主线完成训练支持。
>
> 工作方式约束：
> - 先写计划，再逐步实施
> - 每一步只改一小部分，优先单文件修改
> - 每一步修改后必须做本步可验证检查
> - 每一步完成后停下来，等确认再继续

---

## 1. 当前状态梳理

### 1.1 已经在共享层存在的改动

这些改动已经不再是 PyTorch 专属，JAX 迁移可以直接复用：

- `ModelType.PI05_KI` 已加入共享枚举，见 [`src/openpi/models/model.py`](/workspace/code/openpi-codebase/src/openpi/models/model.py)
- `Observation` 已扩展 `subtask_*` 与 `fast_*` 两组 token 流字段，见 [`src/openpi/models/model.py`](/workspace/code/openpi-codebase/src/openpi/models/model.py)
- `Pi0Config` 已有 `pi05_ki`、`enable_fast_loss`、`enable_subtask_loss`、`enable_ki_attention`，见 [`src/openpi/models/pi0_config.py`](/workspace/code/openpi-codebase/src/openpi/models/pi0_config.py)
- `SubtaskTokenizer` 与 `TokenizeKIInputs` 已在共享 tokenizer / transforms 层存在，见 [`src/openpi/models/tokenizer.py`](/workspace/code/openpi-codebase/src/openpi/models/tokenizer.py) 和 [`src/openpi/transforms.py`](/workspace/code/openpi-codebase/src/openpi/transforms.py)
- `ModelTransformFactory` 已经能为 `PI05_KI` 构造 `TokenizeKIInputs`，见 [`src/openpi/training/config.py`](/workspace/code/openpi-codebase/src/openpi/training/config.py)
- `LeRobotLiberoSubtaskDataConfig`、task 过滤、`episodes_index`、真实 subtask 注入逻辑都已经在共享数据层，见 [`src/openpi/training/config.py`](/workspace/code/openpi-codebase/src/openpi/training/config.py) 和 [`src/openpi/training/data_loader.py`](/workspace/code/openpi-codebase/src/openpi/training/data_loader.py)
- `libero_policy.py` 已认识 `PI05_KI` 并支持传入 `subtask`

### 1.2 仍然是 PyTorch-only 的能力

这些能力目前只在 PyTorch 模型或训练脚本中存在，是 JAX 迁移的主要缺口：

- `PI05_KI` 多 loss 前向：`action + subtask + fast`
- `forward_language_model()` 的 subtask / fast AR loss 计算
- `Knowledge Isolation` attention 路径
- `generate_subtask()` 推理生成
- `train_pytorch.py` 对多 loss 的汇总与日志
- PyTorch 特有的 validation 实现

### 1.3 当前 JAX 侧的真实状态

- JAX 训练入口 [`scripts/train.py`](/workspace/code/openpi-codebase/scripts/train.py) 仍假设 `model.compute_loss(...)` 返回单个可直接 `mean` 的 loss 张量
- JAX 模型 [`src/openpi/models/pi0.py`](/workspace/code/openpi-codebase/src/openpi/models/pi0.py) 仍只实现标准 `PI0/PI05` flow matching loss
- JAX `compute_loss()` 当前没有 subtask/fast 语言建模支路，也没有 KI attention 控制

这意味着：

- 配置层和数据层已经具备大部分入口
- 训练核心仍缺 JAX 模型实现与 JAX train loop 对多 loss 的适配

---

## 2. 迁移原则

为了满足“每次只改一小部分、且每步可验证”，这次迁移遵循下面原则：

- 每一步只改一个文件
- 每一步先做“让代码结构支持下一步”的最小改动，不追求一次完成完整功能
- 先打通配置与接口，再做模型计算，再做训练日志，再做验证与推理
- 每一步都必须有一个本地可执行或至少可静态检查的验证动作

---

## 3. 拆分后的实施顺序

下面顺序刻意偏保守，方便逐步验证。

### Step 0. 写计划文档

目标：
- 在 `docs/` 中落地本计划

修改文件：
- `docs/jax_migration_plan.md`

验证：
- 回读文档，确认范围、顺序、单文件约束、验证方式都明确

状态：
- 当前步骤

### Step 1. 让 JAX 模型显式识别 `pi05_ki`

目标：
- 在 JAX 模型 [`src/openpi/models/pi0.py`](/workspace/code/openpi-codebase/src/openpi/models/pi0.py) 中先把 `config.pi05_ki` 纳入实例状态
- 但这一阶段先不引入新 loss，只做状态对齐，为后续分支留入口

修改文件：
- `src/openpi/models/pi0.py`

预期改动：
- `self.pi05 = config.pi05 or config.pi05_ki`
- 保存 `self.pi05_ki`
- 保存 `enable_fast_loss` / `enable_subtask_loss` / `enable_ki_attention`

验证：
- 运行一次静态搜索，确认字段已被 JAX 模型使用
- 运行最小导入检查，确认 `Pi0Config(pi05_ki=True).create(...)` 不报错

### Step 2. 为 JAX 模型补一个“多 loss 返回接口”，先不改 train loop

目标：
- 先在 JAX 模型中新增一个内部辅助接口，例如 `_compute_pi05_ki_losses(...)`
- 暂时允许它只返回 `{"action": flow_loss}`，先建立结构

修改文件：
- `src/openpi/models/pi0.py`

预期改动：
- 增加内部 helper
- `compute_loss()` 暂时保持兼容，仍返回单 loss，避免一次改太大

验证：
- 最小 shape 检查：fake obs / fake act 跑 `compute_loss()` 仍返回原 shape
- 新 helper 可被调用并返回 dict

### Step 3. 让 JAX train loop 能接收 dict loss

目标：
- 修改 [`scripts/train.py`](/workspace/code/openpi-codebase/scripts/train.py)，使其兼容：
  - 标量/张量 loss
  - dict 多 loss

修改文件：
- `scripts/train.py`

预期改动：
- `loss_fn` 不再强依赖 `jnp.mean(chunked_loss)` 的单一路径
- 当模型返回 dict 时：
  - 聚合出总 loss
  - 单独记录 `loss/action`、`loss/subtask`、`loss/fast`
- 训练日志先和 PyTorch 保持命名对齐

验证：
- 静态检查 `train_step()` 的返回 `info` 中含多 loss 字段
- 用 fake batch 跑一次最小 JIT 编译，确认不会因为返回 dict 结构报错

### Step 4. 在 JAX 模型中正式引入 `PI05_KI` 的 action loss 分支

目标：
- 让 `compute_loss()` 在 `pi05_ki=True` 时先走 dict 路径，但初期仍只包含 `action`
- 这一步的目的是让 train.py + pi0.py 的 dict 接口真正打通

修改文件：
- `src/openpi/models/pi0.py`

预期改动：
- `compute_loss()` 在 `pi05_ki=True` 时返回 `{"action": ...}` 风格的可聚合结果

验证：
- 最小训练步 smoke test
- 观察 `train.py` 中 `loss/action` 是否能正常产生

### Step 5. 把 subtask AR loss 迁移到 JAX

目标：
- 在 JAX 模型里实现 subtask token 流的 prefix-LM loss

修改文件：
- `src/openpi/models/pi0.py`

预期改动：
- 读取 `observation.subtask_*`
- 复用已有 tokenizer 产物
- 构造 masked CE loss
- 按 `enable_subtask_loss` 开关控制

验证：
- fake batch 下 subtask loss 分支 shape 正确
- 当关闭 `enable_subtask_loss=False` 时，该项不存在或为零

### Step 6. 把 FAST AR loss 迁移到 JAX

目标：
- 在 JAX 模型里实现 fast token 流的 AR loss

修改文件：
- `src/openpi/models/pi0.py`

预期改动：
- 读取 `observation.fast_*`
- 构造 masked CE loss
- 按 `enable_fast_loss` 开关控制

验证：
- fake batch 下 fast loss 可计算
- 关闭 `enable_fast_loss=False` 时，该项不存在或为零

### Step 7. 在 JAX train loop 中支持总 loss 权重和完整日志

目标：
- 若需要，补齐 `loss/total`
- 与 PyTorch 命名对齐，便于横向比较

修改文件：
- `scripts/train.py`

预期改动：
- `wandb.log` 增加 `loss/total`、`loss/action`、`loss/subtask`、`loss/fast`
- 保持现有 `grad_norm` / `param_norm`

验证：
- 检查 `reduced_info` 字段是否齐全
- 最小训练步下日志字典可序列化

### Step 8. 迁移 JAX 侧 Knowledge Isolation

目标：
- 在 JAX attention 实现中加入 KI 机制

修改文件：
- 优先候选：`src/openpi/models/gemma.py`

说明：
- 这是高风险步骤，也是整次迁移最难的一步
- 必须先读清 JAX Gemma 的 attention 结构后再动手
- 很可能需要拆成多个更小步骤；如果发现必须涉及第二个文件，需要先停下来确认

验证：
- 编译通过
- 在 `enable_ki_attention=False/True` 时前向都可运行
- 至少做一个梯度流向的 sanity check

### Step 9. 为 JAX 训练补 validation

目标：
- 把当前 PyTorch 训练里有而 JAX 训练里没有的 validation 逻辑迁回 JAX 入口

修改文件：
- `scripts/train.py`

说明：
- 这一步不一定要立即做，取决于你是否把 JAX train/val 作为第一阶段刚需
- 数据层已有 `create_torch_data_loader_with_val()`，但 JAX 目前训练链路不直接消费它，需要重新设计

验证：
- 可以周期性产出 `val_loss/*`
- 明确 train/val 口径一致

### Step 10. JAX 推理侧是否支持 `generate_subtask()`

目标：
- 评估是否要把 PyTorch-only 的 `generate_subtask()` 推到 JAX

说明：
- 这一步不是训练阻塞项
- 目前 auto-subtask 明确是 PyTorch-only，JAX 可以先不做

验证：
- 若做，则需最小推理 smoke test
- 若不做，则在文档中明确“JAX 训练支持，JAX 推理暂不支持 auto_subtask”

---

## 4. 建议的每步验证模板

为了保证每一步都能停下来检查，建议固定采用下面模板：

1. 静态检查
- `rg` 确认新字段/新分支已接入目标路径

2. 最小导入检查
- 能 import 对应模块
- dataclass/config 构造不报错

3. 最小 shape/smoke test
- 用 fake obs / fake act 跑一小步前向
- 若涉及 train loop，则跑单 step 或最小 JIT

4. 结果回报
- 说明本步改了什么
- 说明验证通过了什么
- 说明下一步准备做什么
- 然后停下来等待确认

---

## 5. 风险评估

### 低风险

- 配置和共享数据层
- train.py 的多 loss 日志适配
- JAX 模型保存 `pi05_ki` / loss 开关状态

### 中风险

- JAX subtask / FAST token loss 接入
- JAX train loop 的多 loss JIT 返回结构

### 高风险

- JAX Knowledge Isolation attention
- JAX `generate_subtask()` 推理能力

---

## 6. 第一阶段完成标准

如果先做一个“可训练版本”的 JAX 迁移，建议第一阶段完成标准是：

- `scripts/train.py` 可以跑 `pi05_ki` 配置
- JAX 模型能计算 `loss/action`、`loss/subtask`、`loss/fast`
- `wandb` 中能看到这些 loss
- 数据层正确读到真实 subtask 数据
- 暂不要求 JAX 推理支持 `generate_subtask()`
- 暂不要求第一阶段就完成 KI attention，前提是文档中明确这一点

这会得到一个“JAX 版联合训练框架”，然后第二阶段再补 KI attention 与推理闭环。

---

## 7. 当前建议的下一步

按你的要求，下一步只改一个文件，建议从这里开始：

- **下一步建议：Step 1，只修改 [`src/openpi/models/pi0.py`](/workspace/code/openpi-codebase/src/openpi/models/pi0.py)**

理由：

- 风险最低
- 不会立刻影响训练脚本
- 能先把 JAX 模型状态与现有共享配置对齐
- 改完后可以立刻做最小导入验证

