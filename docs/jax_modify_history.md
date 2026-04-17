# JAX 迁移修改记录

> 目标：将当前已在共享层与 PyTorch 训练链路中实现的 `PI05_KI` / `FAST` / `Subtask` / `LIBERO` 相关改动，逐步迁移到原版 JAX 训练链路。
>
> 主训练入口：
> - JAX: `scripts/train.py`
> - JAX 模型主实现：`src/openpi/models/pi0.py`
>
> 约束：
> - 每次只做一个小步骤
> - 每次只改最少量代码，优先单文件修改
> - 每一步改完后必须做当前步骤可验证检查
> - 每一步完成后暂停，等待确认再继续

---

## 0. 工作方法

### 0.1 记录模板

按下面模板记录：

1. 改了什么
2. 验证通过

验证说明：

- 只做当前步骤必需的检查
- 语法、基础可运行性这类默认检查直接包含在“验证通过”里，不单独展开记录
- 如果某一步需要额外验证，只记录最关键的结果

### 0.2 当前阶段目标

第一阶段先达成：

- JAX 训练入口可以承接 `pi05_ki` 配置
- JAX 模型能够逐步接入多 loss 训练结构
- 训练日志能逐步与 PyTorch 版本对齐

第一阶段暂不强求：

- JAX `generate_subtask()` 推理
- JAX 版完整 KI attention

这两项属于高风险项，放在后续阶段。

---

## 1. 已建立的规划文档

- 总计划文档：`docs/jax_migration_plan.md`

当前总计划已经明确：

- 哪些改动已经在共享层存在
- 哪些能力仍是 PyTorch-only
- 推荐的逐步迁移顺序
- 每一步的验证模板

---

## 2. 修改记录

### Step 0. 新增 JAX 修改记录文档

时间：
- 2026-04-17

修改文件：
- `docs/jax_modify_history.md`

改了什么：
- 新建本文件
- 写入 JAX 迁移范围、工作约束、记录模板、阶段目标
- 将约束更新为：
  - 每次只做一个小步骤
  - 每次只改最少量代码，优先单文件修改
  - 每一步改完后必须做当前步骤可验证检查
  - 完成后暂停，等待确认再继续
- 简化验证记录方式：
  - 只记录当前步骤必需的检查结果
  - 不再详细展开语法等默认基础检查

验证通过：
- 文件创建成功
- 内容已回读，包含：
  - 迁移目标
  - 记录模板
  - 当前阶段目标
  - 修改记录区

### Step 1. 子计划：让 JAX `Pi0` 显式识别 `pi05_ki`

时间：
- 2026-04-17

实际修改文件：
- `src/openpi/models/pi0.py`

改了什么：
- 新增 `self.pi05_ki = config.pi05_ki`
- 将 `self.pi05 = config.pi05` 改为 `self.pi05 = config.pi05 or config.pi05_ki`
- 新增保存消融开关：
  - `self.enable_fast_loss = config.enable_fast_loss`
  - `self.enable_subtask_loss = config.enable_subtask_loss`
  - `self.enable_ki_attention = config.enable_ki_attention`
- 将 JAX PaliGemma/Gemma 初始化中的 PI05 判断改为使用 `self.pi05`
  - `adarms=self.pi05`
  - `use_adarms=[False, True] if self.pi05 else [False, False]`
  - `if self.pi05:` 初始化 time MLP

验证通过：
- 静态检查通过：
  - `rg -n "pi05_ki|enable_fast_loss|enable_subtask_loss|enable_ki_attention|self\\.pi05|adarms=self\\.pi05|use_adarms=\\[False, True\\] if self\\.pi05|if self\\.pi05" src/openpi/models/pi0.py`
  - 确认 `pi05_ki`、三个开关、`self.pi05`、`adarms`、`use_adarms`、time MLP 分支均已接入
- 语法检查通过：
  - `python -m py_compile /workspace/code/openpi-codebase/src/openpi/models/pi0.py`
- 最小配置检查通过：
  - `uv run python -c "from openpi.models import pi0_config, model; cfg = pi0_config.Pi0Config(pi05_ki=True, action_horizon=10); print(cfg.model_type == model.ModelType.PI05_KI, cfg.discrete_state_input, cfg.max_token_len)"`
  - 输出：`True True 200`
- diff 检查通过：
  - `git diff -- src/openpi/models/pi0.py`
  - 确认本步代码修改只涉及 `src/openpi/models/pi0.py`
