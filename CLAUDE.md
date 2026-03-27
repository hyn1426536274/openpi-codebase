# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenPI is an open-source package from Physical Intelligence for training and running Vision-Language-Action (VLA) models for robotics. It contains three model architectures:
- **π₀** (`pi0`): Flow-based VLA model
- **π₀-FAST** (`pi0_fast`): Autoregressive VLA using FAST action tokenizer
- **π₀.₅** (`pi05`): Upgraded π₀ with improved open-world generalization

The codebase has two parallel implementations: **JAX** (primary, in `src/openpi/models/`) and **PyTorch** (newer, in `src/openpi/models_pytorch/`).

## Setup

```bash
# Install with uv (required package manager)
uv sync

# Initialize git submodules
git submodule update --init --recursive
```

Python 3.11 is required (see `.python-version`). Ubuntu 22.04 only.

## Common Commands

### Linting & Formatting
```bash
# Lint
uv run ruff check <path>
uv run ruff check --fix <path>

# Format
uv run ruff format <path>
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest src/openpi/shared/normalize_test.py

# Run a specific test
uv run pytest src/openpi/shared/normalize_test.py::test_name

# Skip tests marked as manual
uv run pytest -m "not manual"
```

Tests are co-located with source files as `*_test.py`. `conftest.py` automatically switches JAX to CPU backend when no GPU is detected.

### Training

**JAX training:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name>
```

**PyTorch training (single GPU):**
```bash
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

**PyTorch training (multi-GPU):**
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name> [--resume]
```

Config names are defined in `src/openpi/training/config.py` in the `_CONFIGS` list (e.g., `pi0_libero`, `pi0_fast_libero`, `pi05_libero_torch_debug`).

### Data Preprocessing
```bash
# Compute normalization stats (required before training on a new dataset)
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

### Inference / Policy Server
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_path>
```

## Architecture

### Package Structure

```
src/openpi/
├── models/           # JAX model implementations (Pi0, Pi0-FAST, Gemma, SigLIP, ViT)
├── models_pytorch/   # PyTorch equivalents (newer, parallel implementation)
├── policies/         # Robot-specific adapters (ALOHA, DROID, LIBERO)
├── training/         # Training pipeline (config, data loading, checkpoints, optimizer)
├── shared/           # Utilities (download, normalize, image tools, array typing)
├── serving/          # WebSocket policy server for remote inference
└── transforms.py     # Core data transformation pipeline (~15k lines)

packages/openpi-client/  # Standalone client package for connecting to policy server
scripts/                 # Entry-point scripts (train.py, train_pytorch.py, serve_policy.py)
examples/                # Per-robot examples (aloha_real, aloha_sim, droid, libero, ur5)
```

### Configuration System (`src/openpi/training/config.py`)

All training experiments are defined as `TrainConfig` instances in the `_CONFIGS` list. Each config specifies:
- `model`: A model config object (e.g., `Pi0Config`, `Pi0FASTConfig`)
- `data`: A `DataConfigFactory` subclass that builds the data pipeline
- `weight_loader`: Where to load pretrained weights from (GCS or local path)
- Optimizer, LR schedule, batch size, checkpoint directory, etc.

**Data pipeline factory pattern:** `DataConfigFactory.create()` returns a `DataConfig` with three transform stages:
1. `repack_transforms` — remaps dataset-specific key names to canonical names
2. `data_transforms` — robot-specific transformations (applied before normalization), also used at inference
3. `model_transforms` — model-specific transforms like prompt tokenization (applied after normalization)

Concrete factories: `LeRobotLiberoDataConfig`, `LeRobotAlohaDataConfig`, `RLDSDroidDataConfig`, `LeRobotDROIDDataConfig`, `FakeDataConfig`.

### Transform System (`src/openpi/transforms.py`)

The transform system uses `DataTransformFn` protocol objects. A `Group` holds separate `inputs` and `outputs` transform lists. Use `Group.push()` to append transforms. All transforms receive/return a `DataDict` (nested dict of numpy arrays, unbatched).

### Policy Adapters (`src/openpi/policies/`)

Each robot platform has an `*Inputs` and `*Outputs` transform class that handles platform-specific preprocessing (image resizing, state remapping, action space normalization). These are reused at both training and inference time.

### Model Architectures

- **π₀**: SigLIP vision encoder + PaliGemma language model + flow matching action decoder
- **π₀-FAST**: Same backbone but uses FAST tokenizer for autoregressive action generation
- **π₀.₅**: Extends π₀ with knowledge insulation for better generalization

LoRA fine-tuning is supported via `paligemma_variant="gemma_2b_lora"` and `freeze_filter` in `TrainConfig`.

### Checkpoints

- JAX checkpoints use Orbax (`src/openpi/training/checkpoints.py`)
- PyTorch weights can be converted from JAX via `examples/convert_jax_model_to_pytorch.py`
- Base model checkpoints are hosted on GCS at `gs://openpi-assets/checkpoints/`

### Assets & Normalization Stats

Normalization stats are computed per-dataset and stored as assets in the checkpoint directory. The `AssetsConfig` in a `TrainConfig` controls where to load them from (useful for loading base model's stats during fine-tuning).

## Key Conventions

- CLI argument parsing uses `tyro` throughout (not argparse)
- Array shapes are annotated with `openpi.shared.array_typing` (`at` module)
- JAX code uses `flax.nnx` (not the older `flax.linen`)
- Pre-commit hooks run `ruff` for linting/formatting; `third_party/` is excluded
- Test files are `*_test.py`, co-located with source; markers: `manual` for slow/integration tests
- The `openpi-client` package is a workspace member and can be installed independently for robot runtime use
