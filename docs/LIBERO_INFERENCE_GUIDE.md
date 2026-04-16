# Complete Story: Running LIBERO Inference with PI05_KI

## Overview
This document traces the complete flow of how someone would run inference on LIBERO with a PI0.5-KI model in OpenPI.

## Key Components

### 1. Policy Server (Backend)
**File**: `scripts/serve_policy.py`

The policy server is the main entry point for serving a model. Usage:

```bash
# Default LIBERO policy
uv run scripts/serve_policy.py --env LIBERO

# Custom PI05_KI checkpoint
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_ki_libero_torch_debug \
  --policy.dir ./my_checkpoint
```

**Flow**:
1. `serve_policy.py` parses arguments
2. Calls `create_policy()` which uses `policy_config.create_trained_policy()`
3. Sets up WebSocket server on port 8000 (default)
4. Server waits for client connections

### 2. Policy Creation: `policy_config.create_trained_policy()`
**File**: `src/openpi/policies/policy_config.py`

This function does the heavy lifting:

```python
def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    pytorch_device: str | None = None,
) -> _policy.Policy:
```

**Key Steps**:

1. **Detect Model Type**
   - Checks if `model.safetensors` exists in checkpoint
   - If yes → PyTorch model (newer, PI0.5/PI05_KI models)
   - If no → JAX model (legacy)

2. **Load Model**
   ```python
   if is_pytorch:
       model = train_config.model.load_pytorch(train_config, weight_path)
       model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
   ```
   - For PI05_KI: model has `generate_subtask()` method

3. **Create Data Config**
   ```python
   data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
   ```
   - For LIBERO with PI05_KI: uses `LeRobotLiberoSubtaskDataConfig`
   - This defines transform pipelines

4. **Load Normalization Stats**
   - Loads from `checkpoint_dir/assets/{asset_id}/`
   - Critical for denormalizing model outputs

5. **Build Transform Pipeline**
   ```python
   Policy(
       model,
       transforms=[
           *repack_transforms.inputs,
           transforms.InjectDefaultPrompt(default_prompt),
           *data_config.data_transforms.inputs,
           transforms.Normalize(norm_stats),
           *data_config.model_transforms.inputs,
       ],
       output_transforms=[
           *data_config.model_transforms.outputs,
           transforms.Unnormalize(norm_stats),
           *data_config.data_transforms.outputs,
           *repack_transforms.outputs,
       ],
       is_pytorch=is_pytorch,
       pytorch_device=pytorch_device,
   )
   ```

### 3. Data Configuration for LIBERO with PI05_KI
**File**: `src/openpi/training/config.py` → `LeRobotLiberoSubtaskDataConfig`

This factory creates the data config with appropriate transforms:

```python
class LeRobotLiberoSubtaskDataConfig(DataConfigFactory):
    """Config for LIBERO subtask datasets in LeRobot v3.0 format."""
    
    def create(self, assets_dirs, model_config):
        # 1. REPACK TRANSFORMS: Map dataset keys to canonical keys
        repack_mapping = {
            "observation/image": "images.agentview_rgb",
            "observation/wrist_image": "images.wrist_rgb",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        if getattr(model_config, "pi05_ki", False):
            repack_mapping["subtask"] = "subtask"  # ← PI05_KI SUBTASK SUPPORT
        
        # 2. DATA TRANSFORMS: Robot-specific preprocessing
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        
        # 3. MODEL TRANSFORMS: Tokenization and padding
        model_transforms = ModelTransformFactory()(model_config)
        # For PI05_KI, this creates TokenizeKIInputs
```

### 4. LIBERO Policy Transforms
**File**: `src/openpi/policies/libero_policy.py`

#### `LiberoInputs` (Data Transform Input)
Converts raw observations to model input format:

```python
@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images: handle both uint8 and float32, both CHW and HWC
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        
        # Build model inputs
        inputs = {
            "state": data["observation/state"],  # [8]: 3 pos + 3 axis-angle + 2 gripper
            "image": {
                "base_0_rgb": base_image,        # [224, 224, 3]
                "left_wrist_0_rgb": wrist_image, # [224, 224, 3]
                "right_wrist_0_rgb": np.zeros_like(base_image),  # Not used in LIBERO
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if model_type in (PI0_FAST, PI05_KI) else np.False_,
            },
        }
        
        # Pass prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        # PI05_KI: Pass subtask if available
        if self.model_type == _model.ModelType.PI05_KI and "subtask" in data:
            inputs["subtask"] = data["subtask"]
        
        return inputs
```

**Key Detail**: PI05_KI models mask the right_wrist image (set to np.True_), unlike PI0.

#### `LiberoOutputs` (Data Transform Output)
Extracts action predictions from model:

```python
def __call__(self, data: dict) -> dict:
    # Model outputs 10 actions (padded), LIBERO only uses first 7
    return {"actions": np.asarray(data["actions"][:, :7])}
```

### 5. Model Transforms for PI05_KI
**File**: `src/openpi/training/config.py` → `ModelTransformFactory`

For PI05_KI models, creates:

```python
_transforms.Group(
    inputs=[
        _transforms.InjectDefaultPrompt(self.default_prompt),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizeKIInputs(
            fast_tokenizer=FASTTokenizer(max_token_len),
            subtask_tokenizer=SubtaskTokenizer(max_token_len),
            paligemma_tokenizer=PaligemmaTokenizer(max_token_len),
            discrete_state_input=model_config.discrete_state_input,
        ),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ],
    outputs=[
        _transforms.ExtractFASTActions(
            FASTTokenizer(...),
            action_horizon=model_config.action_horizon,
            action_dim=model_config.action_dim,
        )
    ],
)
```

**TokenizeKIInputs** is PI05_KI specific:
- Takes both `prompt` and `subtask` fields
- Uses subtask for action prediction (expert conditioning)
- Produces multiple tokenized sequences for multi-loss training

### 6. Policy Inference: `policy.infer()`
**File**: `src/openpi/policies/policy.py` → `Policy.infer()`

```python
def infer(self, obs: dict, noise: np.ndarray | None = None) -> dict:
    # 1. Copy observation
    inputs = jax.tree.map(lambda x: x, obs)
    
    # 2. Apply input transforms
    inputs = self._input_transform(inputs)
    # Sequence: RepackTransform → InjectDefaultPrompt → LiberoInputs 
    #         → Normalize → ResizeImages → TokenizeKIInputs → PadStatesAndActions
    
    # 3. Convert to PyTorch tensors and batch
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
        inputs
    )
    
    # 4. Create Observation object
    observation = _model.Observation.from_dict(inputs)
    
    # 5. Sample actions
    outputs = {
        "state": inputs["state"],
        "actions": self._sample_actions(sample_rng_or_pytorch_device, observation),
    }
    
    # 6. Convert back to numpy
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
    
    # 7. Apply output transforms
    outputs = self._output_transform(outputs)
    # Sequence: ExtractFASTActions → Unnormalize → LiberoOutputs
    
    return outputs
```

### 7. Subtask Generation (PI05_KI Only)
**File**: `src/openpi/policies/policy.py` → `Policy.generate_subtask()`

PI05_KI models can generate intermediate subtask descriptions:

```python
def generate_subtask(self, obs: dict) -> dict:
    """Generate subtask from current observation.
    
    Only works for PI05_KI PyTorch models.
    Returns: {"subtask": "subtask text or None"}
    """
    if not self._is_pytorch_model or self._generate_subtask is None:
        raise RuntimeError("Only for PI05_KI PyTorch models")
    
    # Same input transform pipeline
    inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
        inputs
    )
    
    observation = _model.Observation.from_dict(inputs)
    
    # Call model's generate_subtask method
    raw_subtasks = self._generate_subtask(self._pytorch_device, observation)
    subtask_text = raw_subtasks[0] if raw_subtasks else None
    
    # Parse format "Subtask: {text}."
    if subtask_text and "Subtask:" in subtask_text:
        after = subtask_text.split("Subtask:")[-1]
        if "." in after:
            subtask_text = after.split(".")[0].strip()
        else:
            subtask_text = after.strip()
    
    return {"subtask": subtask_text}
```

**Wiring**: Policy automatically detects `generate_subtask` method on PyTorch models:
```python
self._generate_subtask = getattr(model, "generate_subtask", None)
```

### 8. WebSocket Communication
**File**: `src/openpi/serving/websocket_policy_server.py`

Server receives observations and returns actions:

```python
async def _handler(self, websocket):
    # Send metadata
    await websocket.send(packer.pack(self._metadata))
    
    while True:
        # Receive observation
        obs = msgpack_numpy.unpackb(await websocket.recv())
        
        # Run inference
        action = self._policy.infer(obs)
        
        # Add timing info
        action["server_timing"] = {
            "infer_ms": infer_time * 1000,
            "prev_total_ms": prev_total_time * 1000,
        }
        
        # Send action
        await websocket.send(packer.pack(action))
```

### 9. Example Client: LIBERO Evaluation
**File**: `examples/libero/main.py`

Shows how to use the policy:

```python
# Connect to server
client = WebsocketClientPolicy("0.0.0.0", 8000)

# Per timestep:
element = {
    "observation/image": img,                    # [224, 224, 3] uint8
    "observation/wrist_image": wrist_img,        # [224, 224, 3] uint8
    "observation/state": np.concatenate([
        obs["robot0_eef_pos"],                    # [3]
        _quat2axisangle(obs["robot0_eef_quat"]), # [3]
        obs["robot0_gripper_qpos"],               # [2]
    ]),                                          # Total [8]
    "prompt": str(task_description),             # e.g., "pick up cube"
}

# Call infer
action_chunk = client.infer(element)["actions"]  # Shape: [10, 7]

# Optionally (PI05_KI only):
# subtask = client.generate_subtask(element)["subtask"]
```

## PI05_KI Specific Flow

### Training Configuration
```python
TrainConfig(
    name="pi05_ki_libero_torch_debug",
    model=pi0_config.Pi0Config(pi05_ki=True),  # Enable PI05_KI
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="/workspace/data/libero/libero_10_subtasks_fixed",
    ),
)
```

### Data Flow During Training
1. `LeRobotLiberoSubtaskDataConfig` repack mapping includes "subtask"
2. `_EnsureSubtask` class ensures "subtask" field always exists
3. `TokenizeKIInputs` handles both prompt and subtask tokenization
4. Model loss includes:
   - Action prediction loss (conditioned on subtask)
   - Subtask generation loss (predict subtask from task+state)

### Data Flow During Inference
1. Observation arrives without subtask
2. **Option A**: Use task prompt directly (subtask ≈ prompt)
   - Policy infer() uses prompt as fallback
3. **Option B**: Generate subtask first
   - Call `policy.generate_subtask(obs)` 
   - Get intermediate subtask description
   - Use in next infer() call for better action prediction

### Subtask Preprocessing in Data Loading
**File**: `src/openpi/training/data_loader.py` → `_EnsureSubtask`

During training, subtasks are handled:

```python
class _EnsureSubtask:
    """Ensures 'subtask' field exists, preserving real annotations."""
    
    def __call__(self, data):
        # Priority 1: Real annotation
        if "subtask" in data and data["subtask"] is not None:
            return data
        
        # Priority 2: Index-based lookup (if mapping available)
        if self._subtasks_mapping and "subtask_index" in data:
            data["subtask"] = self._subtasks_mapping[int(data["subtask_index"])]
            return data
        
        # Priority 3: Fallback to prompt
        if "prompt" in data:
            data["subtask"] = data["prompt"]
        
        return data
```

## Complete Inference Journey

### Step 1: Start Server
```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_ki_libero_torch_debug \
  --policy.dir /workspace/data/pi_models/pi05_base
```

**What happens**:
- Loads `pi05_ki_libero_torch_debug` config
- Detects PyTorch model (model.safetensors exists)
- Creates PI05_KI model with `generate_subtask` support
- Loads normalization stats
- Sets up transform pipelines
- Creates WebSocket server on :8000

### Step 2: Client Connects
```python
from openpi_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy("localhost", 8000)
```

### Step 3: For Each Timestep

#### Minimal Usage:
```python
obs = {
    "observation/image": image_224x224_uint8,
    "observation/wrist_image": wrist_224x224_uint8,
    "observation/state": state_8d,
    "prompt": "pick up the mug",
}

action = client.infer(obs)["actions"]  # [10, 7]
```

#### With Subtask (PI05_KI Feature):
```python
# Generate subtask
subtask_result = client.generate_subtask(obs)
print(f"Subtask: {subtask_result['subtask']}")

# Use subtask for better action prediction
obs_with_subtask = obs.copy()
obs_with_subtask["subtask"] = subtask_result["subtask"]

action = client.infer(obs_with_subtask)["actions"]
```

### Step 4: Transform Pipeline (What Happens Inside)

1. **RepackTransform**: Map dataset keys → canonical keys
   - "observation/image" → "image", etc.
   - "subtask" → "subtask" (if PI05_KI)

2. **InjectDefaultPrompt**: Add default prompt if missing

3. **LiberoInputs**: 
   - Parse images (handle CHW/HWC and float/uint8)
   - Build image dict with 3 views
   - Set right_wrist mask to True (PI05_KI specific)
   - Pass prompt and optional subtask

4. **Normalize**: Apply z-score normalization with loaded stats

5. **ResizeImages**: Ensure 224x224

6. **TokenizeKIInputs**:
   - Tokenize main prompt (action expert conditioning)
   - Tokenize subtask sequence (for subtask generation)
   - Produce tokenized_prompt, subtask_tokenized_prompt, etc.

7. **PadStatesAndActions**: Pad to model action dimension

### Step 5: Model Forward Pass

PI05_KI model (PyTorch):
- Takes observation with tokenized sequences
- Runs PaliGemma vision encoder
- Processes images through vision tower
- Action Expert: predicts actions conditioned on subtask
- Optionally: Subtask head predicts subtask from task+state

### Step 6: Reverse Transforms

1. **ExtractFASTActions**: Decode tokenized action tokens back to continuous
2. **Unnormalize**: Denormalize using loaded stats
3. **LiberoOutputs**: Extract first 7 actions (skip padding)

### Step 7: Return to Client

```python
{
    "state": state_8d,
    "actions": action_chunk_7x7,  # 7 timesteps × 7 dims
    "server_timing": {
        "infer_ms": 45.2,
        "prev_total_ms": 52.1,
    },
}
```

## Key Files Summary

| File | Purpose |
|------|---------|
| `scripts/serve_policy.py` | Entry point, creates policy and WebSocket server |
| `src/openpi/policies/policy_config.py` | `create_trained_policy()` - loads model + transforms |
| `src/openpi/policies/policy.py` | `Policy` class with `infer()` and `generate_subtask()` |
| `src/openpi/policies/libero_policy.py` | `LiberoInputs`/`LiberoOutputs` - LIBERO-specific preprocessing |
| `src/openpi/training/config.py` | Config definitions including `LeRobotLiberoSubtaskDataConfig` |
| `src/openpi/serving/websocket_policy_server.py` | WebSocket server implementation |
| `src/openpi/models_pytorch/pi0_pytorch.py` | PI05_KI model with `generate_subtask()` method |
| `examples/libero/main.py` | LIBERO evaluation example using client |

## Notable Observations

1. **Subtask Handling in PI05_KI**:
   - Enabled via `pi05_ki=True` in model config
   - Automatically adds "subtask" → "subtask" to repack mapping
   - `LiberoInputs` conditionally passes subtask if model_type == PI05_KI
   - Right wrist image masking differs for PI05_KI

2. **Image Masking Logic**:
   ```python
   "right_wrist_0_rgb": np.True_ if model_type in (PI0_FAST, PI05_KI) else np.False_
   ```
   - PI0.5 and PI0.5-KI mask padding images
   - PI0 (original) doesn't mask

3. **Subtask Generation**:
   - Only works on PyTorch PI05_KI models
   - Returns None for JAX models
   - Uses autoregressive decoding through PaliGemma
   - Parsed to extract clean text from "Subtask: {text}." format

4. **Transform Composition**:
   - Input transforms: 7 stages (repack → normalize → tokenize)
   - Output transforms: 3 stages (extract → denormalize → output)
   - Applied symmetrically during inference
