# JAX 迁移计划（2026-04-24 重置版）

> 这份文档从今天开始作为 JAX 迁移的唯一计划文档。
>
> 结论先写在最前面：
> - Torch 侧设计暂时不动
> - JAX 模型核心实现不再沿着 Torch 的 KI attention 方案继续堆补丁
> - JAX 的 `compute_loss` / `forward` / KI 主干，改为参考 `openpi-comet-clean`
> - JAX 的配置、日志、validation、prompt 组织方式，尽量和当前 Torch 训练链路保持一致
> - 当前目标任务仍然是 LIBERO

---

## 1. 迁移目标

这次迁移的目标不是“把 Torch 版 KI 原样翻译成 JAX”，而是分成两层来做：

### 1.1 核心层

JAX 模型核心参考 `openpi-comet-clean`：

- 用 cache-based KI，而不是继续走 Torch 当前那套 attention 级别的改写
- 参考 `compute_loss_and_metrics(...)`
- 参考 `embed_prefix_for_flow(...)`
- 参考 `embed_prefix_for_fast(...)`
- 参考 full-prefix forward 后 slice KV cache，再对 flow 分支做 suffix-only forward 的做法

### 1.2 外围行为层

JAX 对外行为尽量向 Torch 靠齐：

- config 命名和开关
- wandb metric 命名
- grad / loss 日志习惯
- validation 触发方式
- LIBERO 的 prompt / subtask 组织语义

### 1.3 明确非目标

下面这些不是本轮第一优先级：

- 不改 Torch 训练代码
- 不把 Torch 的 KI attention 逻辑继续硬搬到 JAX
- 不要求 JAX 先支持所有非 LIBERO 数据集
- 不要求第一步就把 JAX 推理侧全部补齐

---

## 2. 当前状态

### 2.1 `openpi-codebase` 当前 JAX 状态

当前仓库里的 JAX 路径已经有一部分 KI 痕迹，但整体还是半成品。

#### 模型

文件：
- `src/openpi/models/pi0.py`

当前事实：

- 已经读取了 `pi05_ki`
- 已经保存了 `enable_fast_loss`
- 已经保存了 `enable_subtask_loss`
- 已经保存了 `enable_ki_attention`
- `_compute_pi05_ki_losses(...)` 目前只返回 `{"action": ...}`
- `compute_loss(...)` 在 `pi05_ki=True` 时，本质上还是只返回 action / flow loss

也就是说：

- JAX 现在还没有真正落地 LIBERO 所需的 `flow + fast + subtask`
- 更没有落地 `openpi-comet-clean` 那种 cache-based KI 主干

#### 训练入口

文件：
- `scripts/train.py`

当前事实：

- 仍然假设 `model.compute_loss(...)` 返回的是单个 loss tensor
- 当前默认只记录：
  - `loss`
  - `grad_norm`
  - `param_norm`
- 没有和 Torch 对齐的 validation 逻辑
- 也没有 `compute_loss_and_metrics(...)` 这条更适合 JAX 的多指标接口

#### 共享 Observation / prompt 字段

文件：
- `src/openpi/models/model.py`
- `src/openpi/transforms.py`

当前事实：

- 共享 `Observation` 里已经有：
  - `tokenized_prompt`
  - `tokenized_prompt_mask`
  - `fast_*`
  - `subtask_*`
- 但还没有：
  - `flow_tokenized_prompt`
  - `flow_tokenized_prompt_mask`
  - `flow_loss_mask`
  - `is_vqa`

这说明当前共享层更偏 Torch 的 KI 组织方式，不是 `openpi-comet-clean` 的 JAX 组织方式。

#### validation 配置

文件：
- `src/openpi/training/config.py`

当前事实：

- 共享 `TrainConfig` 已经有：
  - `log_interval`
  - `val_ratio`
  - `val_interval`
  - `val_batches`
- 这些字段已经足够表达 Torch 风格的 train/val 行为
- 但当前 JAX `scripts/train.py` 还没有真正消费这套 validation 逻辑

### 2.2 `openpi-comet-clean` 参考实现给出的方向

参考文件：

- `openpi-comet-clean/src/openpi/models/pi0.py`
- `openpi-comet-clean/src/openpi/models/model.py`
- `openpi-comet-clean/src/openpi/training/jax_train_step.py`

关键点有四个：

#### 1. KI 是 cache-level 的

不是额外魔改 attention kernel，而是：

- 先做一次 full-prefix forward
- 拿到 full KV cache
- slice 出 flow prefix 对应那一段
- 对这段 cache `stop_gradient`
- flow 分支只跑 suffix

这是当前 JAX 迁移应该优先学习的核心设计。

#### 2. `compute_loss_and_metrics(...)` 更适合 JAX

`openpi-comet-clean` 不是只返回一个 loss，而是：

- 主 loss
- 外加一组 aux metrics

这比当前 `openpi-codebase/scripts/train.py` 的“只会吃一个 loss tensor”更适合后续做：

- 多分支 loss
- wandb 细粒度日志
- validation 对齐

#### 3. flow prompt 和 language prompt 是显式分开的

`openpi-comet-clean` 里把 flow prompt 单独做成：

- `flow_tokenized_prompt`
- `flow_tokenized_prompt_mask`

这比在一个字段里混着塞不同语义更清楚，也更适合 KI。

#### 4. 训练指标比当前 `openpi-codebase` 的 JAX 版本更完整

参考 `jax_train_step.py` 可以看到它除了基础 loss 外，还会记录：

- `update_norm`
- `grad_param_ratio`
- `update_param_ratio`
- prompt token 长度统计
- flow prompt 长度统计
- flow 生效比例等

这些很适合作为 JAX 迁移后的基础日志框架。

---

## 3. 新的总体设计

### 3.1 设计原则

这次 JAX 迁移采用下面这个原则：

- JAX 核心参考 `openpi-comet-clean`
- Torch 保持不动
- 共享层只做“向后兼容”的扩展
- LIBERO 当前需要的行为优先
- 不为了“和 Torch 实现细节完全一致”而牺牲 JAX 主干设计

### 3.2 JAX 与 Torch 的职责边界

#### Torch

Torch 继续保持现在的实现和训练逻辑：

- 现有 config
- 现有 grouped grad
- 现有 validation
- 现有 prompt 组织
- 现有推理路径

#### JAX

JAX 单独演进：

- 主干 forward / compute_loss 用 `openpi-comet-clean` 风格
- 训练入口改成能接 `compute_loss_and_metrics(...)`
- 共享层增加 JAX 需要的可选字段，但不破坏 Torch

### 3.3 Prompt 组织的目标口径

当前最重要的是把 prompt 语义先固定下来，避免后面一边改模型一边改数据字段。

对 JAX-LIBERO，建议采用下面的语义：

- `tokenized_prompt`
  - 保留为通用/兼容字段
  - 继续服务非 KI 路径或旧逻辑
- `flow_tokenized_prompt`
  - JAX flow 分支专用
  - 语义必须是 leakage-free prompt
- `fast_tokenized_prompt`
  - JAX / Torch 都可共用
  - 表示 FAST 自回归监督序列
- `subtask_tokenized_prompt`
  - JAX / Torch 都可共用
  - 表示 subtask 自回归监督序列

这个设计的关键点是：

- 不去重定义 Torch 已经在用的 `fast_*` / `subtask_*`
- 只新增 JAX 真正缺的 `flow_*`
- 让 flow prompt 不再偷用别的字段

### 3.4 JAX 损失组织的目标口径

对 LIBERO，JAX 迁移后的目标不是照抄 `openpi-comet-clean` 的 VQA 场景，而是借它的主干设计，输出更接近 Torch 的 loss 视图。

目标日志口径建议如下：

- `loss`
  - 总 loss
- `loss/action`
  - flow / action 分支
- `loss/fast`
  - FAST 自回归分支
- `loss/subtask`
  - subtask 自回归分支

如果后续调试确实需要，也可以额外保留 JAX 风格别名：

- `loss_flow`
- `loss_fast_raw`
- `loss_subtask_raw`

但主展示口径优先靠近 Torch。

### 3.5 JAX validation 的目标口径

validation 目标和当前 Torch 版本尽量一致：

- 只有 `val_ratio > 0` 时启用
- 使用 `val_interval`
- 每次跑 `val_batches`
- 每次 validation 单独平均，不和 train log interval 再做混合平均
- wandb 中明确记录 `val/*` 或 `val_loss/*`

---

## 4. 分阶段迁移方案

下面是新的执行顺序。后续真正改代码时，仍然遵循“每次只动一小块，改前确认”的原则。

### Phase 0. 文档重置

目标：

- 重写当前迁移计划
- 明确旧计划过时
- 把 JAX 目标从“跟着 Torch KI attention 走”切换到“核心参考 `openpi-comet-clean`”

涉及文件：

- `docs/jax_migration_plan.md`
- `docs/jax_modify_history.md`

状态：

- 当前阶段

### Phase 1. 固定共享字段和 LIBERO prompt 合约

目标：

- 在不影响 Torch 的前提下，把 JAX 需要的字段补齐
- 把 LIBERO 下各 prompt 字段的语义完全写死

建议动作：

- 在共享 `Observation` 中补可选字段：
  - `flow_tokenized_prompt`
  - `flow_tokenized_prompt_mask`
  - `flow_loss_mask`
  - `is_vqa`
- 对 LIBERO 先约定默认行为：
  - `flow_loss_mask=True`
  - `is_vqa=False`
- 重新整理 JAX 要用的 transforms / tokenizer 输出口径

涉及文件候选：

- `src/openpi/models/model.py`
- `src/openpi/transforms.py`
- `src/openpi/models/tokenizer.py`

完成标准：

- Torch 侧现有数据流不被破坏
- JAX 所需字段齐备
- 文档中对每个 prompt 字段的语义没有歧义

### Phase 2. 先迁移 JAX 模型核心，再谈优化

目标：

- 把 `src/openpi/models/pi0.py` 的 action / flow 主干改成 `openpi-comet-clean` 风格

优先迁移的内容：

- `_embed_visual_prefix(...)`
- `embed_prefix_for_flow(...)`
- full-prefix LM forward
- KV cache slicing
- `stop_gradient`
- suffix-only flow forward
- `compute_loss_and_metrics(...)`

注意：

- 这一阶段先把 JAX 核心主干改对
- 不急着一口气把所有 LIBERO 分支都塞进去
- 先保证 `loss/action` 路径是干净、稳定、可解释的

涉及文件：

- `src/openpi/models/pi0.py`

完成标准：

- JAX `compute_loss` 可以作为 `compute_loss_and_metrics` 的包装
- action / flow 路径确实是 cache-based KI
- Torch 不受影响

### Phase 3. 在新的 JAX 主干上接回 LIBERO 的 fast / subtask 分支

目标：

- 以新的 JAX 核心为底座，恢复 LIBERO 所需的多分支训练

建议顺序：

1. 先接 `loss/action`
2. 再接 `loss/fast`
3. 最后接 `loss/subtask`

原因：

- action / flow 是主干
- fast 更接近 `openpi-comet-clean` 已有的 language loss 结构
- subtask 是 LIBERO 特有附加分支，最后接最稳

重要约束：

- 尽量不要把三条分支写成互相缠绕的复杂逻辑
- 先保证每条分支单独可读，再做共享前缀优化

涉及文件：

- `src/openpi/models/pi0.py`

完成标准：

- `compute_loss_and_metrics(...)` 能稳定返回总 loss 和分支指标
- 输出指标至少包括：
  - `loss/action`
  - `loss/fast`
  - `loss/subtask`

### Phase 4. 对齐 JAX 的 grad / loss / wandb 日志

目标：

- 把 JAX 训练日志从“只有 `loss`/`grad_norm`/`param_norm`”提升到可用状态
- 同时尽量靠近 Torch 的可读性

建议动作：

- 让 JAX train loop 接 `compute_loss_and_metrics(...)`
- 保留基础字段：
  - `loss`
  - `grad_norm`
  - `param_norm`
- 增加分支 loss：
  - `loss/action`
  - `loss/fast`
  - `loss/subtask`
- 参考 `openpi-comet-clean` 增加辅助统计：
  - `update_norm`
  - `grad_param_ratio`
  - `update_param_ratio`
  - prompt 长度统计

暂不承诺的项：

- Torch 那套 grouped grad norm 不作为第一阶段阻塞项
- 如果要做，也放在主干稳定之后，再单独设计 JAX 参数分组口径

涉及文件：

- `scripts/train.py`
- 视情况可抽公共 helper，但第一步不强求拆文件

完成标准：

- wandb 能看到总 loss 和分支 loss
- 日志平均方式清晰，不混淆 train / val

### Phase 5. 把 JAX validation 接回当前共享 config 口径

目标：

- 让 JAX 真正使用当前共享 `TrainConfig` 里已有的：
  - `val_ratio`
  - `val_interval`
  - `val_batches`

要求：

- validation 与 train 的调用链分清楚
- val 指标只在 validation run 内平均
- 不把单次 val 结果错误地和 train log interval 再做二次平均

涉及文件候选：

- `scripts/train.py`
- `src/openpi/training/data_loader.py`
- 若需要，补 JAX 侧可复用的 val loader 创建逻辑

完成标准：

- 能周期性产出稳定的 `val` 指标
- 口径与 Torch 当前配置语义一致

### Phase 6. 补 JAX 的 LIBERO 专用 config，并与 Torch 命名尽量对齐

目标：

- 给 JAX 增加当前 LIBERO 需要的 config 入口
- 和 Torch 版本的实验命名、开关含义尽量一致

原则：

- 开关名尽量复用已有字段
- 不为了“完全一样”去扭曲 JAX 主干
- config 负责表达实验，不负责承载模型 hack

涉及文件：

- `src/openpi/training/config.py`

完成标准：

- JAX 有独立的 LIBERO config
- 语义和 Torch 当前实验保持可对照

### Phase 7. 最后再决定是否补 JAX 推理闭环

这一步不是当前训练迁移的阻塞项。

可选内容：

- JAX `sample_actions(...)` 路径继续对齐 `openpi-comet-clean`
- 是否需要 JAX 侧 `generate_subtask()`

当前建议：

- 先把训练链路做好
- 推理闭环单独开下一轮

---

## 5. 当前推荐的实际执行顺序

如果下一步要开始动代码，建议按这个顺序做：

1. 先做 Phase 1：补共享字段，固定 prompt 合约
2. 再做 Phase 2：把 JAX `pi0.py` 的 flow 主干切到 `openpi-comet-clean` 风格
3. 再做 Phase 3：补回 fast / subtask
4. 然后做 Phase 4：整理 JAX train 日志
5. 然后做 Phase 5：接 validation
6. 最后做 Phase 6：补 JAX LIBERO configs

这个顺序的原因很简单：

- 先固定输入和字段语义
- 再改最核心的模型计算
- 最后再处理日志和训练外围

否则很容易出现：

- prompt 语义还没定
- 模型已经改了一半
- 日志字段又开始漂移

---

## 6. 风险与注意事项

### 6.1 最大风险不是“少一个 if”，而是接口语义混乱

真正高风险的地方不是某个 loss 少算了，而是下面这些：

- 同一个字段在 Torch 和 JAX 里表示不同语义
- flow prompt 和 language prompt 混用
- train / val 的平均口径不一致
- 总 loss 和分支 loss 的记录口径漂移

所以这次迁移第一步必须先把接口定义好。

### 6.2 subtask 分支不要一开始就过度优化

`openpi-comet-clean` 给的是 flow + language 主干参考，但 LIBERO 还有 subtask 分支。

这里更稳妥的做法是：

- 先把 subtask 当成独立、清晰的一条 JAX LM 分支接回来
- 确认行为正确后，再讨论是否和 fast 分支进一步共享更多计算

### 6.3 grouped grad 不要抢在主干之前做

Torch 当前 grouped grad 是有价值的，但它不是 JAX 主干迁移的第一阻塞项。

优先级应该是：

1. 核心 forward / compute_loss 正确
2. train / val 口径清楚
3. 分支日志可读
4. 再决定是否做 JAX grouped grad

---

## 7. 当前结论

从今天开始，JAX 迁移按下面这句话执行：

> JAX 核心跟 `openpi-comet-clean`，JAX 外围行为尽量对齐 Torch，Torch 本身不动，当前只服务 LIBERO。

下一步如果开始改代码，第一步不应该再去碰 JAX KI attention 细节，而应该先做：

- 共享字段补齐
- prompt 合约固定
- 然后再改 JAX `pi0.py`
