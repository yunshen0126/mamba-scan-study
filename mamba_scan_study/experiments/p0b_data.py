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
# Computed from P0B_EUROSAT_SPLIT_FROZEN.json SHA-256
# f5ddb2db3f8ffc74efb77295e0fac17d34df85179bcd78de3f4e638b685c4117:
# only its 22,000-sample train split, after PIL bilinear resize to 32x32.
EUROSAT_MEAN = (0.3447923299, 0.3808058131, 0.4081652860)
EUROSAT_STD = (0.1978067505, 0.1315186118, 0.1098431517)
ORGAN_NORMALIZATION = {
    "organamnist": ((0.4681101025,) * 3, (0.2801411101,) * 3),
    "organcmnist": ((0.4942488707,) * 3, (0.2674806004,) * 3),
    "organsmnist": ((0.4954148361,) * 3, (0.2679301867,) * 3),
}
DATASET_CLASS_COUNTS = {
    "cifar10": 10,
    "organamnist": 11,
    "organcmnist": 11,
    "organsmnist": 11,
    "eurosat": 10,
}
ORGAN_DATASETS = {
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
}
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
    frozen_split: FrozenSplit | None


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


class RepeatGrayscaleChannels:
    """Repeat a single grayscale tensor into the three shared input channels."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"expected a one-channel image tensor, got shape {tuple(tensor.shape)}")
        return tensor.repeat(3, 1, 1)


def _scalar_label(label: object) -> int:
    values = np.asarray(label)
    _require(values.size == 1, f"expected a scalar label, got shape {values.shape}")
    return int(values.item())


def _legacy_cifar10_transforms():
    """Return the original P0-B transforms without changing their construction."""
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


def _main_uniform_transforms(dataset: str):
    """Build PIL-first main-experiment transforms without directional reflection."""
    import torchvision.transforms as transforms
    from torchvision.transforms import InterpolationMode

    _require(dataset in DATASET_CLASS_COUNTS, f"unsupported P0-B dataset: {dataset}")
    if dataset == "cifar10":
        mean, std = CIFAR_MEAN, CIFAR_STD
        resize = []
        repeat_channels = []
    elif dataset in ORGAN_NORMALIZATION:
        mean, std = ORGAN_NORMALIZATION[dataset]
        resize = [transforms.Resize((32, 32), interpolation=InterpolationMode.BILINEAR)]
        repeat_channels = [RepeatGrayscaleChannels()]
    else:
        mean, std = EUROSAT_MEAN, EUROSAT_STD
        resize = [transforms.Resize((32, 32), interpolation=InterpolationMode.BILINEAR)]
        repeat_channels = []

    train_transform = transforms.Compose(
        resize
        + [transforms.RandomCrop(32, padding=4), transforms.ToTensor()]
        + repeat_channels
        + [transforms.Normalize(mean, std)]
    )
    validation_transform = transforms.Compose(
        resize + [transforms.ToTensor()] + repeat_channels + [transforms.Normalize(mean, std)]
    )
    return train_transform, validation_transform


def _transforms(dataset: str, augmentation: str):
    if augmentation == "p0b_legacy":
        _require(dataset == "cifar10", "p0b_legacy augmentation is only valid for cifar10")
        return _legacy_cifar10_transforms()
    _require(augmentation == "main_uniform", f"unsupported augmentation mode: {augmentation}")
    return _main_uniform_transforms(dataset)


def _build_generic_loaders(
    train_dataset: object,
    validation_dataset: object,
    *,
    batch_size: int,
    training_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> P0BLoaders:
    generator = torch.Generator().manual_seed(int(training_seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    return P0BLoaders(train=train_loader, validation=validation_loader, frozen_split=None)


def _validate_labels(dataset: object, dataset_name: str, expected_length: int) -> None:
    labels = np.asarray(getattr(dataset, "labels")).reshape(-1)
    class_count = DATASET_CLASS_COUNTS[dataset_name]
    _require(len(labels) == expected_length, f"{dataset_name} split length is invalid")
    _require(np.all((labels >= 0) & (labels < class_count)), f"{dataset_name} labels are out of range")
    _require(
        np.all(np.bincount(labels, minlength=class_count) > 0),
        f"{dataset_name} train/validation split is missing a class",
    )


def _build_organ_loaders(
    dataset: str,
    data_root: str | Path,
    *,
    train_transform: object,
    validation_transform: object,
    batch_size: int,
    training_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> P0BLoaders:
    import medmnist

    dataset_class = getattr(medmnist, ORGAN_DATASETS[dataset])
    train_dataset = dataset_class(
        split="train",
        root=str(data_root),
        transform=train_transform,
        target_transform=_scalar_label,
        download=False,
    )
    validation_dataset = dataset_class(
        split="val",
        root=str(data_root),
        transform=validation_transform,
        target_transform=_scalar_label,
        download=False,
    )
    _validate_labels(train_dataset, dataset, {"organamnist": 34561, "organcmnist": 12975, "organsmnist": 13932}[dataset])
    _validate_labels(validation_dataset, dataset, {"organamnist": 6491, "organcmnist": 2392, "organsmnist": 2452}[dataset])
    return _build_generic_loaders(
        train_dataset,
        validation_dataset,
        batch_size=batch_size,
        training_seed=training_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _resolve_eurosat_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    for candidate in (root / "eurosat", root / "EuroSAT", root):
        nested = candidate / "2750"
        if nested.is_dir():
            return nested
        if candidate.is_dir() and any(path.is_dir() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"EuroSAT ImageFolder data is unavailable below {root}")


def _sorted_imagefolder(root: Path, transform: object):
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(root), transform=transform)
    root_path = Path(dataset.root)
    samples = sorted(
        ((path, int(label)) for path, label in dataset.samples),
        key=lambda entry: Path(entry[0]).relative_to(root_path).as_posix(),
    )
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [label for _, label in samples]
    return dataset


def _load_eurosat_indices() -> tuple[np.ndarray, np.ndarray]:
    split_path = Path(__file__).resolve().parents[2] / "P0B_EUROSAT_SPLIT_FROZEN.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"missing frozen EuroSAT split: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    train_indices = np.asarray(payload.get("train_indices"), dtype=np.int64)
    validation_indices = np.asarray(payload.get("validation_indices"), dtype=np.int64)
    test_indices = np.asarray(payload.get("test_indices"), dtype=np.int64)
    _require(len(train_indices) == 22_000, "EuroSAT train split length is invalid")
    _require(len(validation_indices) == 2_500, "EuroSAT validation split length is invalid")
    _require(len(test_indices) == 2_500, "EuroSAT test split length is invalid")
    all_indices = np.concatenate((train_indices, validation_indices, test_indices))
    _require(np.array_equal(np.sort(all_indices), np.arange(27_000, dtype=np.int64)), "EuroSAT split is not a partition")
    return train_indices, validation_indices


def _build_eurosat_loaders(
    data_root: str | Path,
    *,
    train_transform: object,
    validation_transform: object,
    batch_size: int,
    training_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> P0BLoaders:
    image_root = _resolve_eurosat_root(data_root)
    train_indices, validation_indices = _load_eurosat_indices()
    train_dataset = _sorted_imagefolder(image_root, train_transform)
    validation_dataset = _sorted_imagefolder(image_root, validation_transform)
    _require(len(train_dataset) == 27_000 and len(validation_dataset) == 27_000, "EuroSAT sample count is invalid")
    _require(train_dataset.targets == validation_dataset.targets, "EuroSAT image ordering differs by transform")
    _require(len(set(train_dataset.targets)) == DATASET_CLASS_COUNTS["eurosat"], "EuroSAT class count is invalid")
    return _build_generic_loaders(
        Subset(train_dataset, train_indices.tolist()),
        Subset(validation_dataset, validation_indices.tolist()),
        batch_size=batch_size,
        training_seed=training_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


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
    augmentation: str = "p0b_legacy",
    dataset: str = "cifar10",
) -> P0BLoaders:
    """Build P0-B train/validation loaders without constructing official test data."""
    _require(batch_size > 0, "batch_size must be positive")
    _require(training_seed in (0, 1, 2, 3), "P0-B training seed must be 0, 1, 2, or 3")
    _require(num_workers >= 0, "num_workers must be non-negative")
    dataset = dataset.lower()
    _require(dataset in DATASET_CLASS_COUNTS, f"unsupported P0-B dataset: {dataset}")
    train_transform, validation_transform = _transforms(dataset, augmentation)

    if dataset in ORGAN_DATASETS:
        return _build_organ_loaders(
            dataset,
            data_root,
            train_transform=train_transform,
            validation_transform=validation_transform,
            batch_size=batch_size,
            training_seed=training_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    if dataset == "eurosat":
        return _build_eurosat_loaders(
            data_root,
            train_transform=train_transform,
            validation_transform=validation_transform,
            batch_size=batch_size,
            training_seed=training_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    frozen = load_frozen_split(source_paths) if frozen_split is None else frozen_split
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
