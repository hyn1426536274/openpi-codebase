# JAX 修改记录（2026-04-24 重置版）

> 这份记录从今天开始只服务新的 JAX 迁移方向。
>
> 新方向一句话概括：
> - Torch 不动
> - JAX 核心参考 `openpi-comet-clean`
> - JAX 外围行为尽量向 Torch 对齐
> - 当前目标数据集仍然是 LIBERO

---

## 1. 状态重置

### 2026-04-24

今天确认旧的 JAX 迁移方向已经过时。

旧问题在于：

- 它默认继续沿着 Torch 的 KI attention 设计补 JAX
- 但实际对照 `openpi-comet-clean` 后，JAX 更合适的主干是 cache-based KI
- 当前 `openpi-codebase` 的 JAX 真缺的也不只是 attention，而是整套：
  - prompt 合约
  - `compute_loss_and_metrics`
  - 训练日志
  - validation

因此，从今天开始：

- 旧计划不再作为后续改代码的依据
- 以 `docs/jax_migration_plan.md` 为新的唯一计划文档

---

## 2. 今日确认的代码基线

下面这些是已经确认过的当前事实。

### 2.1 JAX 模型当前状态

文件：
- `src/openpi/models/pi0.py`

确认结果：

- 已经识别 `pi05_ki`
- 已经保存：
  - `enable_fast_loss`
  - `enable_subtask_loss`
  - `enable_ki_attention`
- 但当前 `_compute_pi05_ki_losses(...)` 只返回 `action`
- 当前 `compute_loss(...)` 本质上仍然是单分支 flow loss

结论：

- 当前 JAX 代码只具备“接入 KI 配置的壳”
- 还不具备目标中的 LIBERO 多分支训练能力

### 2.2 JAX 训练入口当前状态

文件：
- `scripts/train.py`

确认结果：

- 仍然按单个 loss tensor 写的
- 当前只稳定记录：
  - `loss`
  - `grad_norm`
  - `param_norm`
- 没有对齐 Torch 的 validation 行为

### 2.3 共享 prompt / Observation 当前状态

文件：
- `src/openpi/models/model.py`
- `src/openpi/transforms.py`

确认结果：

- 已有：
  - `tokenized_prompt`
  - `fast_*`
  - `subtask_*`
- 尚无：
  - `flow_tokenized_prompt`
  - `flow_tokenized_prompt_mask`
  - `flow_loss_mask`
  - `is_vqa`

结论：

- 当前共享层更偏 Torch KI 组织
- 如果 JAX 要参考 `openpi-comet-clean`，需要补一层向后兼容的字段扩展

---

## 3. 今日文档修改

### 2026-04-24

修改文件：

- `docs/jax_migration_plan.md`
- `docs/jax_modify_history.md`

改了什么：

- 重写 JAX 迁移计划
- 明确 JAX 的核心方向切换为：
  - `openpi-comet-clean` 风格的 cache-based KI
  - 而不是继续沿 Torch 的 KI attention 方案补丁式迁移
- 明确 Torch 暂时不动
- 明确当前目标任务仍是 LIBERO
- 明确新的阶段顺序：
  1. 先固定共享字段和 prompt 合约
  2. 再迁移 JAX 核心 `pi0.py`
  3. 再补 fast / subtask
  4. 再整理 grad / loss / wandb
  5. 再接 validation
  6. 最后补 JAX LIBERO configs

验证通过：

- 文档已按新方向重写
- 新计划与当前代码基线一致
- 新计划和 `openpi-comet-clean` 的参考方向一致

---

## 4. 当前有效的执行顺序

后续如果开始真正改代码，当前有效顺序如下：

### Phase 0

状态：
- 已完成

内容：
- 重写文档

### Phase 1

状态：
- 待开始

内容：
- 固定共享字段和 LIBERO prompt 合约

### Phase 2

状态：
- 待开始

内容：
- 把 JAX `src/openpi/models/pi0.py` 的 flow 主干迁到 `openpi-comet-clean` 风格

### Phase 3

状态：
- 待开始

内容：
- 在新的 JAX 主干上补回 `loss/fast` 和 `loss/subtask`

### Phase 4

状态：
- 待开始

内容：
- 整理 JAX grad / loss / wandb 日志

### Phase 5

状态：
- 待开始

内容：
- 接回 validation

### Phase 6

状态：
- 待开始

内容：
- 补 JAX 的 LIBERO config

---

## 5. 历史说明

### 关于 2026-04-17 的旧记录

旧记录里提到的这件事仍然成立：

- `src/openpi/models/pi0.py` 已经接入 `pi05_ki` 和几个消融开关

但从今天开始，这件事只作为“当前代码基线”的说明存在，不再代表后续迁移的主路线。

换句话说：

- 那一步代码没有白做
- 但后续路线要按新的计划继续，而不是沿旧计划继续展开

---

## 6. 下一步建议

下一步如果要开始改代码，建议先做：

1. 明确共享 `Observation` 还要不要加 `flow_*` 字段
2. 明确 LIBERO 下：
   - `tokenized_prompt`
   - `flow_tokenized_prompt`
   - `fast_tokenized_prompt`
   - `subtask_tokenized_prompt`
   的最终语义
3. 只在这个接口固定之后，再开始改 `src/openpi/models/pi0.py`

当前不建议的起手式：

- 直接去补 JAX attention 细节
- 直接去补 grouped grad
- 直接去补一大串 config，而模型主干还没定
