from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
# import lerobot.common.datasets.lerobot_dataset as lerobot_dataset # lerobot<=0.4.0
import lerobot.datasets.lerobot_dataset as lerobot_dataset # lerobot>=0.4.4
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class _EnsureSubtask:
    """Ensures 'subtask' field exists, preserving real annotations when present.

    Handles three scenarios:
    1. subtask field already exists in data → preserve it (real annotation)
    2. subtask_index exists + mapping available → lookup via index
    3. Neither → fall back to prompt as pseudo-subtask

    This gracefully handles both annotated datasets (like libero_10_subtasks_fixed
    where subtask is a direct field) and index-based datasets.
    """

    def __init__(self, subtasks_mapping: dict[int, str] | None = None):
        """Initialize with optional subtask index→text mapping.

        Args:
            subtasks_mapping: Dict mapping subtask_index (int) to subtask text (str).
                             If None, will skip index-based lookup.
        """
        self._subtasks_mapping = subtasks_mapping

    def __call__(self, data: dict) -> dict:
        # First priority: check if subtask already exists (real annotation)
        subtask = data.get("subtask")
        if subtask is not None:
            # Convert from various formats to string
            if not isinstance(subtask, str):
                try:
                    data["subtask"] = str(subtask) if not hasattr(subtask, "item") else subtask.item()
                except Exception:
                    pass  # Keep as is if conversion fails
            return data

        # Second priority: try index-based lookup (if mapping available)
        if self._subtasks_mapping is not None:
            subtask_index = data.get("subtask_index")
            if subtask_index is not None:
                try:
                    idx = int(subtask_index)
                    if idx in self._subtasks_mapping:
                        data["subtask"] = self._subtasks_mapping[idx]
                        return data
                except (ValueError, TypeError):
                    pass  # Fall through to prompt-based fallback

        # Final fallback: use prompt as pseudo-subtask
        prompt = data.get("prompt")
        if prompt is not None:
            data["subtask"] = prompt if isinstance(prompt, str) else prompt.item()

        return data


def _load_subtasks_mapping(repo_id: str) -> dict[int, str] | None:
    """Load subtask index→text mapping from meta/subtasks.json or subtasks.parquet.

    Args:
        repo_id: Dataset repo ID (local path or HuggingFace repo ID).

    Returns:
        Dict mapping subtask_index to subtask text, or None if mapping file
        doesn't exist or cannot be loaded.
    """
    root = pathlib.Path(repo_id)
    subtasks_json = root / "meta" / "subtasks.json"
    subtasks_parquet = root / "meta" / "subtasks.parquet"

    if subtasks_json.exists():
        try:
            with open(subtasks_json) as f:
                entries = json.load(f)
            # Each entry should have {"subtask_index": int, "subtask": str, ...}
            return {entry["subtask_index"]: entry["subtask"] for entry in entries}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.warning(f"Failed to load subtasks from {subtasks_json}: {e}")

    if subtasks_parquet.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(subtasks_parquet)
            # Standard LeRobot subtask columns: subtask_index, subtask
            return dict(zip(df["subtask_index"], df["subtask"]))
        except Exception as e:
            logging.warning(f"Failed to load subtask mapping from {subtasks_parquet}: {e}")

    return None


def _split_episodes(total_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Split episode indices into train/val sets.

    Deterministic given the same seed, so all ablation variants share
    the same val episodes for fair comparison.

    Returns:
        (train_episode_ids, val_episode_ids)
    """
    all_ids = list(range(total_episodes))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_ids)
    val_count = max(1, int(total_episodes * val_ratio))
    return sorted(all_ids[val_count:]), sorted(all_ids[:val_count])


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig,
    episodes: list[int] | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    import time as _time
    _t0 = _time.time()
    print(f"[DATASET-DEBUG] Creating LeRobotDatasetMetadata(repo_id={repo_id})...", flush=True)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    print(f"[DATASET-DEBUG] Metadata done in {_time.time()-_t0:.1f}s. fps={dataset_meta.fps}, total_episodes={dataset_meta.total_episodes}", flush=True)

    _t1 = _time.time()
    print(f"[DATASET-DEBUG] Creating LeRobotDataset(episodes={episodes})...", flush=True)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        episodes=episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    print(f"[DATASET-DEBUG] LeRobotDataset done in {_time.time()-_t1:.1f}s. len={len(dataset)}", flush=True)

    if data_config.prompt_from_task:
        # lerobot 0.4.4: meta.tasks is a DataFrame (index=task_text, column=task_index)
        # Convert to dict[int, str] for PromptFromLeRobotTask compatibility.
        tasks = dataset_meta.tasks
        if hasattr(tasks, "iterrows"):
            # DataFrame: index is task text, "task_index" is the column
            tasks_dict = {int(row["task_index"]): str(idx) for idx, row in tasks.iterrows()}
        else:
            # Already a dict (lerobot <= 0.1.0)
            tasks_dict = tasks
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks_dict)])

    # PI05_KI mode: inject "subtask" field, preserving real annotations when available.
    # Try to load subtask mapping from meta/subtasks.json; fall back to prompt-based pseudo-subtask.
    if getattr(model_config, "pi05_ki", False):
        subtasks_mapping = _load_subtasks_mapping(repo_id)
        dataset = TransformedDataset(dataset, [_EnsureSubtask(subtasks_mapping)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    episodes: list[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        episodes: If provided, only load these episode indices. (new param, not support all dataset types now)
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        episodes=episodes,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    episodes: list[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        episodes: If provided, only load these episode indices.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, episodes=episodes)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_torch_data_loader_with_val(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    *,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    framework: str = "pytorch",
    episodes: list[int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders with episode-level split.

    Loads the full dataset once, then uses torch.utils.data.Subset to split
    by frame indices derived from episode boundaries. This avoids constructing
    two LeRobotDataset instances (which doubles the parquet loading time).

    Args:
        episodes: If provided, only load these episode indices (applied before train/val split).
    """
    # 1. Load and transform full dataset (single LeRobotDataset construction)
    print(f"[VAL-DEBUG] Step 1: Creating full dataset (episodes={episodes})...", flush=True)
    full_dataset = create_torch_dataset(data_config, action_horizon, model_config, episodes=episodes)
    print(f"[VAL-DEBUG] Step 1a: Dataset created, len={len(full_dataset)}", flush=True)
    full_dataset = transform_dataset(full_dataset, data_config, skip_norm_stats=skip_norm_stats)
    print(f"[VAL-DEBUG] Step 1b: Transforms applied, len={len(full_dataset)}", flush=True)

    # 2. Unwrap TransformedDataset layers to reach the underlying LeRobotDataset
    raw_ds = full_dataset
    while isinstance(raw_ds, TransformedDataset):
        raw_ds = raw_ds._dataset

    # Build episode_data_index compatible with both lerobot versions. （now use lerobot==0.4.4 for v3.0 dataset version）
    if hasattr(raw_ds, "episode_data_index"):
        # lerobot <= 0.1.0: has episode_data_index directly
        episode_data_index = raw_ds.episode_data_index
        total_episodes = len(episode_data_index["from"])
    elif hasattr(raw_ds, "meta") and hasattr(raw_ds.meta, "episodes") and raw_ds.meta.episodes is not None:
        # lerobot >= 0.4.4: use meta.episodes which has dataset_from_index / dataset_to_index
        episodes_table = raw_ds.meta.episodes
        from_col = episodes_table["dataset_from_index"]
        to_col = episodes_table["dataset_to_index"]
        episode_data_index = {
            "from": torch.tensor(from_col, dtype=torch.int64),
            "to": torch.tensor(to_col, dtype=torch.int64),
        }
        total_episodes = len(from_col)
    else:
        raise RuntimeError(
            "Cannot determine episode boundaries: LeRobotDataset has neither "
            "'episode_data_index' nor 'meta.episodes'. Check your lerobot version."
        )
    print(f"[VAL-DEBUG] Step 2: total_episodes={total_episodes}", flush=True)

    # 3. Split episodes
    train_eps, val_eps = _split_episodes(total_episodes, val_ratio, seed)
    logging.info(
        f"Train/Val split: {len(train_eps)} train episodes, {len(val_eps)} val episodes "
        f"(ratio={val_ratio}, seed={seed})"
    )
    print(f"[VAL-DEBUG] Step 3: train_eps={len(train_eps)}, val_eps={len(val_eps)}", flush=True)

    # 4. Convert episode ids to frame indices
    def _eps_to_indices(ep_ids):
        indices = []
        for ep in ep_ids:
            start = episode_data_index["from"][ep].item()
            end = episode_data_index["to"][ep].item()
            indices.extend(range(start, end))
        return indices

    train_indices = _eps_to_indices(train_eps)
    val_indices = _eps_to_indices(val_eps)

    # Sanity checks: no overlap, no leakage
    assert len(set(train_eps) & set(val_eps)) == 0, "Episode overlap between train and val!"
    assert len(train_eps) + len(val_eps) == total_episodes, "Episode leakage: some episodes are missing!"
    assert len(set(train_indices) & set(val_indices)) == 0, "Frame overlap between train and val!"
    assert len(train_indices) + len(val_indices) == len(full_dataset), (
        f"Frame leakage: train({len(train_indices)}) + val({len(val_indices)}) != total({len(full_dataset)})"
    )

    print(f"[VAL-DEBUG] Step 4: train_frames={len(train_indices)}, val_frames={len(val_indices)}", flush=True)

    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)

    # 5. Train loader
    sampler = None
    if framework == "pytorch" and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = batch_size
    # Safety: clamp batch size to dataset size (episodes filter may produce tiny datasets)
    local_batch_size = min(local_batch_size, len(train_ds))
    val_batch_size = min(batch_size, len(val_ds))
    print(f"[VAL-DEBUG] Step 5: local_batch_size={local_batch_size}, val_batch_size={val_batch_size}, sampler={'DDP' if sampler else 'None'}", flush=True)

    train_loader = DataLoaderImpl(
        data_config,
        TorchDataLoader(
            train_ds,
            local_batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            seed=seed,
            framework=framework,
        ),
    )
    print(f"[VAL-DEBUG] Step 5b: Train loader created", flush=True)

    # 6. Val loader (no shuffle, no DDP)
    val_loader = DataLoaderImpl(
        data_config,
        TorchDataLoader(
            val_ds,
            local_batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            seed=seed,
            framework=framework,
        ),
    )
    print(f"[VAL-DEBUG] Step 6: Val loader created. Done!", flush=True)

    return train_loader, val_loader


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            print(f"[ITER-DEBUG] Creating new data_iter (num_items={num_items}, num_batches={self._num_batches}, dataset_len={len(self._data_loader.dataset)})", flush=True)
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
