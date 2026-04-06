# Framework Changes: openpi-comet vs openpi-comet-test

This document compares the **reference codebase** (`/root/Training/ki/openpi-comet/`) against the **modified implementation** (`/root/Training/ki/openpi-comet-test/`) across 16 files. Changes are grouped by framework or feature area.

---

## Table of Contents

1. [Overview](#overview)
2. [PI05 Subtask-Conditioned Policy](#1-pi05-subtask-conditioned-policy)
3. [FAST Action Tokenization](#2-fast-action-tokenization)
4. [Knowledge Isolation (KI)](#3-knowledge-isolation-ki)
5. [LoRA / PEFT Fine-Tuning](#4-lora--peft-fine-tuning)
6. [Dataset and Data Loading](#5-dataset-and-data-loading)
7. [Inference and Serving](#6-inference-and-serving)
8. [Other / Minor Changes](#7-other--minor-changes)
9. [File Change Summary Table](#8-file-change-summary-table)

---

## Overview

The modified codebase introduces three interlocking frameworks on top of the baseline PI0 policy:

**PI05 Subtask-Conditioned Policy** — PI05 differs from PI0 in two ways: (1) the robot proprioceptive state is embedded as discrete language tokens (prefix) rather than as a continuous suffix; (2) the action expert uses adaRMSNorm to inject the flow-matching timestep. The new implementation extends PI05 with a learned subtask generation head — the model can autoregressively predict a short natural-language subtask description from the current observation, enabling hierarchical planning without an external reasoner.

**FAST Action Tokenization** — FAST encodes action trajectories as sequences of discrete vocabulary tokens using a learned tokenizer (`FASTTokenizer`), then trains the model to predict those tokens autoregressively (a language-model-style auxiliary loss). This runs in parallel with the flow-matching action loss during KI training.

**Knowledge Isolation (KI)** — A training strategy that prevents gradients produced by the action expert (suffix tokens) from flowing back through the backbone (prefix tokens) during the attention computation. Implemented by detaching backbone key/value tensors before they are used by suffix queries. This allows multi-task training (flow matching + FAST AR + subtask AR) without polluting the visual-language representations.

---

## 1. PI05 Subtask-Conditioned Policy

### Concept

Standard PI05 feeds the robot state as discrete tokens and uses adaRMSNorm for timestep conditioning. The new implementation adds a second output head: during both training and inference the model can generate a natural-language subtask description. At training time this creates a supervised sequence prediction loss; at inference time the generated subtask is fed back into the prompt for the next query.

### Changed Files

#### `src/openpi/models/model.py`

Added `PI05_KI = "pi05_ki"` to the `ModelType` enum. Extended the `Observation` dataclass with eight new optional fields supporting parallel token streams:

```python
# subtask prediction stream
subtask_tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
subtask_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
subtask_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
subtask_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

# FAST action token stream
fast_tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
fast_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
fast_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
fast_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
```

#### `src/openpi/models/pi0_config.py`

Added three new config fields:

```python
pi05_ki: bool = False
fast_model_tokenizer: str | None = None
fast_model_tokenizer_kwargs: dict | None = None
```

`model_type` property returns `ModelType.PI05_KI` when `pi05_ki=True`. `__post_init__` enforces `max_token_len=200` and `discrete_state_input=True` automatically when `pi05_ki=True`.

#### `src/openpi/models/tokenizer.py`

Added new `SubtaskTokenizer` class. Its `tokenize()` method constructs a prefix of the form `"Task: {prompt}, State: {state_str};\n"` and a postfix of the form `"Subtask: {subtask}."` terminated by an EOS token. It returns four arrays: token IDs, attention mask, autoregressive mask, and per-token loss mask. The loss mask is set to `True` only over the postfix (the model is supervised only on the subtask text, not the context).

Also added `.capitalize()` calls to `PaligemmaTokenizer.tokenize` for consistent text normalization.

#### `src/openpi/transforms.py`

Added `TokenizeKIInputs` transform class:

```python
@dataclasses.dataclass(frozen=True)
class TokenizeKIInputs(DataTransformFn):
    fast_tokenizer: _tokenizer.FASTTokenizer
    subtask_tokenizer: _tokenizer.SubtaskTokenizer
    paligemma_tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
```

Its `__call__` produces three parallel token sequences from a single observation:
- `tokenized_prompt / tokenized_prompt_mask` — for the action expert's prefix
- `subtask_tokenized_prompt / mask / ar_mask / loss_mask` — for subtask prediction
- `fast_tokenized_prompt / mask / ar_mask / loss_mask` — for FAST action token prediction

Removed the debug `_log_prompt` infrastructure that was present in the baseline.

#### `src/openpi/training/config.py`

Added `PI05_KI` to the `ModelTransformFactory.create_transforms()` dispatch using `TokenizeKIInputs`. Added `"subtask": "subtask"` to the repack transforms in `LeRobotB1KDataConfig` so the subtask label propagates from dataset to model input. Removed `LeRobotB1KSkillDataConfig`. Added `b1k_assets_dir` config field and reduced the default `save_interval` from 5000 to 1000.

#### `src/openpi/models_pytorch/pi0_pytorch.py`

`PI0Pytorch.__init__` now sets `self.pi05 = config.pi05 or config.pi05_ki`, stores a reference to the PaliGemma tokenizer, and disables `torch.compile` (compatibility requirement for LoRA/KI).

New `generate_subtask()` method (~130 lines): performs autoregressive generation using greedy decoding. Constructs the subtask prefix prompt, runs forward passes token-by-token through the PaliGemma backbone, and stops at the EOS token. Returns a list of decoded strings.

`forward()` now returns a `dict` instead of a scalar, with keys `action`, `subtask`, and `fast`:

```python
loss = defaultdict(float)
# ...
if observation.fast_tokenized_prompt is not None and \
   observation.subtask_tokenized_prompt is not None:
    knowledge_isolation = True
    loss_language = self.forward_language_model(observation)
    loss.update(loss_language)
# ... flow matching ...
loss['action'] = F.mse_loss(u_t, v_t, reduction="none").mean()
return loss
```

New `forward_language_model()` computes the FAST and subtask AR losses by running the backbone with the respective token streams and comparing predicted logits against ground-truth next tokens.

#### `src/openpi/policies/b1k_policy.py`

`B1kInputs.__call__` extended to handle `PI05_KI` model type alongside `PI0` and `PI05`. When `subtask` is present in the data dict, it is forwarded to the model input:

```python
case _model.ModelType.PI0 | _model.ModelType.PI05 | _model.ModelType.PI05_KI:
    ...
    if "subtask" in data:
        inputs["subtask"] = data["subtask"]
```

#### `src/openpi/policies/policy.py`

Added `generate_subtask()` method that calls the model's subtask generation, parses the output format `"Subtask: {text}."`, and returns the extracted text. Also wires up `self._generate_subtask = model.generate_subtask` when the model has that attribute.

---

## 2. FAST Action Tokenization

### Concept

FAST (Frequency-space Action Sequence Tokenization) converts a continuous action trajectory into a sequence of discrete vocabulary tokens. During KI training the model is trained to predict these tokens autoregressively — a standard language modeling loss over the action tokens. This auxiliary task encourages the backbone to learn a compact, language-compatible representation of motion.

### Changed Files

#### `src/openpi/models/tokenizer.py`

`FASTTokenizer.tokenize` now accepts an optional `state` argument (was required). Added `.capitalize()` to the prompt string for normalization consistency.

#### `src/openpi/transforms.py`

`TokenizeKIInputs.__call__` calls `fast_tokenizer.tokenize(prompt, state, actions)` to produce the `fast_tokenized_prompt` stream used in training.

#### `src/openpi/models_pytorch/pi0_pytorch.py`

`forward_language_model()` computes the FAST AR loss:

```python
# Run backbone with FAST token stream
fast_logits = self.backbone(fast_tokenized_prompt, fast_mask, fast_ar_mask)
fast_loss = F.cross_entropy(
    fast_logits[fast_loss_mask],
    fast_targets[fast_loss_mask]
)
loss['fast'] = fast_loss
```

#### `src/openpi/training/config.py`

`Pi0Config` instances for KI training set `fast_model_tokenizer` to point to a pretrained FAST tokenizer checkpoint. The `TokenizeKIInputs` transform receives the instantiated `FASTTokenizer`.

#### `scripts/train_pytorch_test.py`

Training loop handles the multi-loss dict returned by the model:

```python
elif isinstance(losses, dict):
    action_loss = losses.get("action", None)
    fast_loss   = losses.get("fast", None)
    subtask_loss = losses.get("subtask", None)
    loss_list = [v for v in losses.values() if v is not None]
    losses = torch.stack(loss_list)
```

WandB logging records `action_loss`, `fast_loss`, and `subtask_loss` as separate metrics.

---

## 3. Knowledge Isolation (KI)

### Concept

During multi-task training the action expert (suffix tokens) attends to both its own tokens and the backbone's visual-language tokens (prefix). Without KI, gradients from the action expert loss flow back through the cross-attention into the backbone weights, potentially degrading the backbone's visual-language representations. KI addresses this by detaching the backbone's key and value tensors before they are used in suffix queries, breaking the gradient path from suffix loss to prefix parameters.

Prefix tokens (images, language) continue to attend to each other with full gradients (bidirectional within-prefix attention is unchanged). Only the information flow from prefix to suffix is made gradient-free.

### Changed Files

#### `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`

New function `eager_attention_forward_ki()` (~60 lines). Splits the query, key, and value tensors at `num_prefix_tokens`:

```python
prefix_query = query_states[:, :, :num_prefix_tokens, :]
suffix_query = query_states[:, :, num_prefix_tokens:, :]
prefix_key   = key_states[:, :, :num_prefix_tokens, :]
suffix_key   = key_states[:, :, num_prefix_tokens:, :]
prefix_value = value_states[:, :, :num_prefix_tokens, :]
suffix_value = value_states[:, :, num_prefix_tokens:, :]

# Prefix queries: full attention, full gradient
weights_prefix = torch.matmul(prefix_query, key_states.transpose(2, 3)) * scaling

# Suffix queries: attend to DETACHED prefix keys (no grad to backbone)
ki_key   = torch.cat([prefix_key.detach(),   suffix_key],   dim=2)
ki_value = torch.cat([prefix_value.detach(), suffix_value], dim=2)
weights_suffix = torch.matmul(suffix_query, ki_key.transpose(2, 3)) * scaling
output_suffix  = torch.matmul(weights_for_suffix, ki_value)
```

The final attention output concatenates the prefix and suffix results along the sequence dimension, maintaining the correct tensor shape for the rest of the model.

#### `src/openpi/models_pytorch/gemma_pytorch.py`

`GemmaExpert.forward()` accepts a new `knowledge_isolation: bool = False` parameter. When enabled it resolves the LoRA-wrapped model to the underlying `gemma_model` and passes `knowledge_isolation=True` down through `compute_layer_complete()`. The layer function dispatches to `eager_attention_forward_ki()` instead of the standard `eager_attention_forward()`.

Added LoRA-unwrapping logic to obtain the raw model reference:

```python
gemma_model = self.gemma_expert.model
if hasattr(gemma_model, "model"):
    gemma_model = gemma_model.model
```

#### `src/openpi/models_pytorch/pi0_pytorch.py`

`forward()` sets `knowledge_isolation = True` when both `fast_tokenized_prompt` and `subtask_tokenized_prompt` are present in the observation. This flag is passed to the expert's forward call, activating the KI attention path for that forward pass.

#### `src/openpi/models/pi0_config.py`

`pi05_ki: bool = False` field controls whether the KI training path is activated. When `True`, the `model_type` property returns `ModelType.PI05_KI` and the config enforces `discrete_state_input=True`.

---

## 4. LoRA / PEFT Fine-Tuning

### Concept

Low-Rank Adaptation (LoRA) is applied to both PaliGemma (the vision-language backbone) and the Gemma action expert. LoRA adapters are injected after the full base model weights are loaded, keeping the pre-trained weights frozen and only training the low-rank matrices. At the end of training, LoRA weights are saved separately and can be merged back into the base model.

### Changed Files

#### `src/openpi/models_pytorch/pi0_pytorch.py`

New `apply_lora()` method injects LoRA adapters into both sub-models after base weight loading:

```python
def apply_lora(self):
    self.paligemma_with_expert.paligemma = self.lora_wrapper(
        self.paligemma_with_expert.paligemma
    )
    self.paligemma_with_expert.gemma_expert = self.lora_wrapper(
        self.paligemma_with_expert.gemma_expert
    )
```

New `lora_wrapper()` method creates a `LoraConfig` targeting the attention and feed-forward projection layers:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

Returns a `get_peft_model(model, lora_config)` PEFT model.

#### `src/openpi/models_pytorch/gemma_pytorch.py`

`embed_tokens` changed from `None` to `nn.Identity()`. This is required for LoRA compatibility — PEFT needs all expected attributes to exist on the module tree.

Also added the LoRA-unwrapping pattern in `forward()` to obtain the raw Gemma model object regardless of whether LoRA is currently applied:

```python
gemma_model = self.gemma_expert.model
if hasattr(gemma_model, "model"):
    gemma_model = gemma_model.model
```

#### `scripts/train_pytorch_test.py`

After loading base weights, the training loop calls `model.apply_lora()` to inject adapters:

```python
model.apply_lora()
```

Checkpoint saving is LoRA-aware: when LoRA is detected it saves adapters separately using `save_pretrained()` and writes a `lora_config.json` with the base model path and adapter directory names. When LoRA is not present it falls back to `safetensors.torch.save_model()`:

```python
from peft import PeftModel
has_paligemma_lora = isinstance(model_to_save.paligemma_with_expert.paligemma, PeftModel)
has_expert_lora    = isinstance(model_to_save.paligemma_with_expert.gemma_expert, PeftModel)

if has_paligemma_lora and has_expert_lora:
    lora_metadata = {
        "base_model_path": str(config.pytorch_weight_path),
        "paligemma_lora_dir": "paligemma_lora",
        "expert_lora_dir":    "expert_lora",
    }
    model_to_save.paligemma_with_expert.paligemma.save_pretrained(
        tmp_ckpt_dir / "paligemma_lora"
    )
    model_to_save.paligemma_with_expert.gemma_expert.save_pretrained(
        tmp_ckpt_dir / "expert_lora"
    )
else:
    safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
```

#### `test-scripts/merge_lora.py` (new file, only in test codebase)

Utility script for post-training LoRA merging. Workflow:

1. Reads `lora_config.json` from the checkpoint directory to obtain the base model path.
2. Reads `metadata.pt` to reconstruct the `Pi0Config`.
3. Instantiates `PI0Pytorch` and loads base weights via `safetensors.torch.load_model()`.
4. Wraps PaliGemma and the expert with `PeftModel.from_pretrained()`.
5. Calls `.merge_and_unload()` on each to fold the LoRA deltas into the base weights.
6. Saves the merged model as `model.safetensors`, and copies `assets/`, `metadata.pt`, and `optimizer.pt` alongside it.

---

## 5. Dataset and Data Loading

### Concept

The KI training regime requires multi-level task annotations: each timestep needs both a global task label and a finer-grained subtask label. The dataset was refactored to load all annotation levels simultaneously and expose them as `task` and `subtask` keys.

### Changed Files

#### `src/behavior/learning/datas/dataset.py`

`prepare_task()` now loads all annotation levels for every episode simultaneously:

```python
def prepare_task(self):
    self.task_sizes = {}
    for ep_id, ep_orch in self.meta.orchestrators.items():
        self.task_sizes[ep_id] = {}
        for level, tasks in ep_orch.items():
            self.task_sizes[ep_id][level] = [
                task_info["end_frame"] for task_info in tasks
            ]
```

`__getitem__` now always sets `item["task"]` (global task, level 0) and also sets `item["subtask"]` when `fine_grained_level > 0`, otherwise sets it to an empty string:

```python
item["task"] = self._get_fine_grained_task(item, 0)
if self.fine_grained_level > 0:
    item["subtask"] = self._get_fine_grained_task(item, self.fine_grained_level)
else:
    item["subtask"] = ""
```

`_get_fine_grained_task()` now accepts an explicit `fine_grained_level` parameter instead of using `self.fine_grained_level`.

`BehaviorLerobotDatasetMetadata.load_orchestrators_data()` refactored as an instance method (was a class method / standalone function). Now reads directly from `skill_annotation` and `primitive_annotation` data, building three annotation levels:

- Level 0: global task name
- Level 1: skill type (navigation, uncoordinated manipulation, coordinated manipulation, etc.)
- Level 2: skill description with object names cleaned by stripping trailing `_N` instance suffixes
- Level 3: primitive description with objects

Removed: `build_orchestrator_levels_from_annotations()` standalone function, skill stream infrastructure, and action chunk masking logic.

---

## 6. Inference and Serving

### Concept

At inference time the model's subtask generation capability replaces the external LLM reasoner. Instead of calling a separate model to produce a subtask description, the policy itself generates the subtask autoregressively given the current observation, then injects that subtask back into the action-generation prompt.

### Changed Files

#### `src/openpi/shared/eval_b1k_wrapper.py`

JSON path for `task_mapping.json` changed from a relative to an absolute path (minor environment fix).

External `reasoner` Client initialization disabled — the `self.reasoner = None` assignment is now unconditional and the `Client` import block is commented out.

In `act()`, when `fine_grained_level > 0`, the wrapper now calls the policy's built-in subtask generator instead of the external reasoner:

```python
subtask = self.policy.generate_subtask(batch)["subtask"]
self.last_subtask = subtask
batch["subtask"] = subtask
```

The generated subtask is stored as `self.last_subtask` for fallback on exception.

#### `src/openpi/policies/policy.py`

Added `generate_subtask()` method:

```python
def generate_subtask(self, obs: dict) -> dict:
    inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
    # convert to tensors ...
    observation = _model.Observation.from_dict(inputs)
    subtasks = self._generate_subtask(device, observation)
    subtask = subtasks[0]  # take first batch element

    def extract_subtask_content(raw_text):
        if "Subtask:" in raw_text and "." in raw_text.split("Subtask:")[-1]:
            return raw_text.split("Subtask:")[1].split(".")[0].strip()
        return None

    return {"subtask": extract_subtask_content(subtask), ...}
```

`__init__` wires this up when the model exposes the `generate_subtask` attribute:

```python
if hasattr(model, "generate_subtask"):
    self._generate_subtask = model.generate_subtask
```

#### `scripts/serve_b1k.py`

Default `port` argument changed from `8000` to `8002` to avoid conflict with other running services.

---

## 7. Other / Minor Changes

#### `src/openpi/models_pytorch/preprocessing_pytorch.py`

No functional changes. Minor whitespace and formatting differences only.

#### `scripts/train_pytorch_test.py` — administrative changes

- `build_datasets`: `shuffle=False` (was `True`). Likely for reproducibility during debugging.
- `init_wandb`: added `entity="huangyinuo321-uestc"` for team-level WandB logging.
- Removed validation infrastructure: `_validation_is_enabled()`, `validate()`, `build_val_loader()`, `_decode_prompt_for_logging()`, `_log_trainable_params_summary_pytorch()`.
- Added `import openpi.training.sharding as sharding`.

---

## 8. File Change Summary Table

| File | PI05 Subtask | FAST | KI | LoRA | Dataset | Inference | Other |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `scripts/serve_b1k.py` | | | | | | x | x |
| `scripts/train_pytorch_test.py` | | x | | x | | | x |
| `src/behavior/learning/datas/dataset.py` | x | | | | x | | |
| `src/openpi/models/model.py` | x | x | x | | | | |
| `src/openpi/models/pi0_config.py` | x | x | x | | | | |
| `src/openpi/models/tokenizer.py` | x | x | | | | | |
| `src/openpi/models_pytorch/gemma_pytorch.py` | | | x | x | | | |
| `src/openpi/models_pytorch/pi0_pytorch.py` | x | x | x | x | | | |
| `src/openpi/models_pytorch/preprocessing_pytorch.py` | | | | | | | x |
| `src/openpi/models_pytorch/transformers_replace/.../modeling_gemma.py` | | | x | | | | |
| `src/openpi/policies/b1k_policy.py` | x | | | | | | |
| `src/openpi/policies/policy.py` | x | | | | | x | |
| `src/openpi/shared/eval_b1k_wrapper.py` | x | | | | | x | |
| `src/openpi/training/config.py` | x | x | | | x | | x |
| `src/openpi/transforms.py` | x | x | | | | | x |
| `test-scripts/merge_lora.py` | | | | x | | | |

**Legend:** x = file has changes belonging to that framework/category.

---

## Appendix: Key File Paths

| Component | Path |
|-----------|------|
| KI attention kernel | `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` |
| Multi-task forward pass | `src/openpi/models_pytorch/pi0_pytorch.py` |
| Parallel tokenization transform | `src/openpi/transforms.py` (`TokenizeKIInputs`) |
| Subtask / FAST tokenizers | `src/openpi/models/tokenizer.py` |
| Dataset multi-level labels | `src/behavior/learning/datas/dataset.py` |
| LoRA injection | `src/openpi/models_pytorch/pi0_pytorch.py` (`apply_lora`, `lora_wrapper`) |
| LoRA checkpoint merge | `test-scripts/merge_lora.py` |
| Training loop (KI) | `scripts/train_pytorch_test.py` |
| Inference subtask generation | `src/openpi/policies/policy.py`, `src/openpi/shared/eval_b1k_wrapper.py` |
