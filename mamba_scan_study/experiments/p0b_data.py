"""Frozen CIFAR-10 train/validation data access for P0-B only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mamba_scan_study.experiments.p0b_path_bank import (
    P0BSourcePaths,
    default_source_paths,
    verify_source_hashes,
)


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
DATASET_LENGTH = 50_000
TRAIN_LENGTH = 45_000
VALIDATION_LENGTH = 5_000


@dataclass(frozen=True)
class FrozenSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    images_sha256: str
    targets_sha256: str
    train_indices_sha256: str
    validation_indices_sha256: str
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class P0BLoaders:
    train: DataLoader
    validation: DataLoader
    frozen_split: FrozenSplit


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def int64_c_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return _sha256_bytes(array.tobytes(order="C"))


def uint8_c_sha256(values: object) -> str:
    array = np.asarray(values)
    if array.dtype != np.uint8:
        raise ValueError(f"CIFAR images must be uint8, got {array.dtype}")
    return _sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_indices(payload: dict, name: str, expected_length: int) -> np.ndarray:
    values = payload.get(name)
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    _require(array.ndim == 1, f"frozen split {name} must be one-dimensional")
    _require(len(array) == expected_length, f"frozen split {name} length is invalid")
    _require(np.all((array >= 0) & (array < DATASET_LENGTH)), f"frozen split {name} is out of range")
    _require(np.array_equal(array, np.sort(array)), f"frozen split {name} must be sorted")
    _require(np.unique(array).size == expected_length, f"frozen split {name} contains duplicates")
    declared = payload.get(f"{name}_sha256")
    observed = int64_c_sha256(array)
    _require(declared == observed, f"frozen split {name} SHA-256 mismatch")
    return array


def load_frozen_split(source_paths: P0BSourcePaths | None = None) -> FrozenSplit:
    """Verify all P0-B source gates before parsing the frozen split arrays."""
    paths = default_source_paths() if source_paths is None else source_paths
    source_sha256 = verify_source_hashes(paths)
    payload = json.loads(Path(paths.validation_split).read_text(encoding="utf-8"))

    _require(payload.get("dataset") == "CIFAR10", "P0-B requires frozen CIFAR10 split")
    _require(payload.get("dataset_train_flag") is True, "P0-B split must target train=True")
    _require(payload.get("dataset_length") == DATASET_LENGTH, "frozen split population is invalid")
    _require(payload.get("split_seed") == 20260720, "frozen split seed is invalid")
    _require(payload.get("train_per_class") == 4500, "frozen train-per-class is invalid")
    _require(payload.get("validation_per_class") == 500, "frozen validation-per-class is invalid")
    _require(payload.get("class_iteration_order") == list(range(10)), "frozen class order is invalid")

    train_indices = _validated_indices(payload, "train_indices", TRAIN_LENGTH)
    validation_indices = _validated_indices(payload, "validation_indices", VALIDATION_LENGTH)
    combined = np.sort(np.concatenate((train_indices, validation_indices)))
    _require(
        np.array_equal(combined, np.arange(DATASET_LENGTH, dtype=np.int64)),
        "frozen split must partition all 50,000 official training examples",
    )

    images_sha256 = payload.get("images_uint8_c_sha256")
    targets_sha256 = payload.get("targets_int64_c_sha256")
    _require(isinstance(images_sha256, str) and len(images_sha256) == 64, "missing image SHA-256")
    _require(isinstance(targets_sha256, str) and len(targets_sha256) == 64, "missing target SHA-256")
    return FrozenSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        images_sha256=images_sha256,
        targets_sha256=targets_sha256,
        train_indices_sha256=payload["train_indices_sha256"],
        validation_indices_sha256=payload["validation_indices_sha256"],
        source_sha256=source_sha256,
    )


def seed_worker(worker_id: int) -> None:
    """Synchronize Python, NumPy, and PyTorch worker RNGs from DataLoader state."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _validate_dataset_arrays(dataset: object, frozen_split: FrozenSplit, label: str) -> np.ndarray:
    images = np.asarray(getattr(dataset, "data"))
    _require(images.shape == (DATASET_LENGTH, 32, 32, 3), f"{label} CIFAR image shape is invalid")
    _require(uint8_c_sha256(images) == frozen_split.images_sha256, f"{label} CIFAR image SHA-256 mismatch")
    targets = np.ascontiguousarray(np.asarray(getattr(dataset, "targets"), dtype=np.int64))
    _require(targets.shape == (DATASET_LENGTH,), f"{label} CIFAR target shape is invalid")
    _require(int64_c_sha256(targets) == frozen_split.targets_sha256, f"{label} CIFAR target SHA-256 mismatch")
    _require(np.array_equal(np.bincount(targets, minlength=10), np.full(10, 5000)), f"{label} class counts are invalid")
    _require(
        np.array_equal(np.bincount(targets[frozen_split.train_indices], minlength=10), np.full(10, 4500)),
        f"{label} train class counts are invalid",
    )
    _require(
        np.array_equal(np.bincount(targets[frozen_split.validation_indices], minlength=10), np.full(10, 500)),
        f"{label} validation class counts are invalid",
    )
    return targets


def _transforms():
    import torchvision.transforms as transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    return train_transform, validation_transform


def build_p0b_loaders(
    data_root: str | Path,
    batch_size: int,
    training_seed: int,
    *,
    num_workers: int = 4,
    download: bool = True,
    pin_memory: bool = True,
    source_paths: P0BSourcePaths | None = None,
    cifar10_factory: Callable[..., object] | None = None,
    frozen_split: FrozenSplit | None = None,
) -> P0BLoaders:
    """Build P0-B's frozen 45k/5k train/validation loaders without official test."""
    _require(batch_size > 0, "batch_size must be positive")
    _require(training_seed in (0, 1, 2, 3), "P0-B training seed must be 0, 1, 2, or 3")
    _require(num_workers >= 0, "num_workers must be non-negative")
    frozen = load_frozen_split(source_paths) if frozen_split is None else frozen_split
    train_transform, validation_transform = _transforms()
    if cifar10_factory is None:
        from torchvision.datasets import CIFAR10

        cifar10_factory = CIFAR10

    train_dataset = cifar10_factory(
        root=str(data_root), train=True, transform=train_transform, download=download
    )
    validation_dataset = cifar10_factory(
        root=str(data_root), train=True, transform=validation_transform, download=download
    )
    train_targets = _validate_dataset_arrays(train_dataset, frozen, "train")
    validation_targets = _validate_dataset_arrays(validation_dataset, frozen, "validation")
    _require(np.array_equal(train_targets, validation_targets), "train and validation datasets differ")

    train_subset = Subset(train_dataset, frozen.train_indices.tolist())
    validation_subset = Subset(validation_dataset, frozen.validation_indices.tolist())
    _require(len(train_subset) == TRAIN_LENGTH, "P0-B train subset must contain 45,000 examples")
    _require(len(validation_subset) == VALIDATION_LENGTH, "P0-B validation subset must contain 5,000 examples")
    generator = torch.Generator().manual_seed(int(training_seed))
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    return P0BLoaders(train=train_loader, validation=validation_loader, frozen_split=frozen)
