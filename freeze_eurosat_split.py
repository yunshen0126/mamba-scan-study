"""Freeze a stable, stratified EuroSAT train/validation/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torchvision
from torchvision.datasets import ImageFolder

from mamba_scan_study.experiments.p0b_data import _resolve_eurosat_root


SEED = 0
CLASS_COUNT = 10
TOTAL_SAMPLES = 27_000
TRAIN_SAMPLES = 22_000
VALIDATION_SAMPLES = 2_500
TEST_SAMPLES = 2_500


def _sorted_imagefolder(root: Path) -> ImageFolder:
    dataset = ImageFolder(str(root))
    samples = sorted(
        ((path, int(label)) for path, label in dataset.samples),
        key=lambda entry: Path(entry[0]).relative_to(root).as_posix(),
    )
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [label for _, label in samples]
    return dataset


def _distribution(indices: list[int], targets: list[int], classes: list[str]) -> dict[str, int]:
    selected = np.asarray(targets, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(selected, minlength=CLASS_COUNT)
    return {classes[class_id]: int(count) for class_id, count in enumerate(counts)}


def _proportional_holdout_counts(class_counts: np.ndarray, total: int) -> np.ndarray:
    quotas = class_counts.astype(np.float64) * total / int(class_counts.sum())
    allocated = np.floor(quotas).astype(np.int64)
    remainder = total - int(allocated.sum())
    order = sorted(
        range(CLASS_COUNT),
        key=lambda class_id: (-(quotas[class_id] - allocated[class_id]), class_id),
    )
    for class_id in order[:remainder]:
        allocated[class_id] += 1
    if int(allocated.sum()) != total:
        raise AssertionError("proportional holdout allocation did not reach its requested total")
    return allocated


def freeze(data_root: Path, output: Path) -> str:
    image_root = _resolve_eurosat_root(data_root)
    dataset = _sorted_imagefolder(image_root)
    if len(dataset) != TOTAL_SAMPLES:
        raise ValueError(f"expected 27000 EuroSAT samples, found {len(dataset)}")
    if len(dataset.classes) != CLASS_COUNT:
        raise ValueError(f"expected 10 EuroSAT classes, found {len(dataset.classes)}")

    targets = np.asarray(dataset.targets, dtype=np.int64)
    class_counts = np.bincount(targets, minlength=CLASS_COUNT)
    validation_counts = _proportional_holdout_counts(class_counts, VALIDATION_SAMPLES)
    test_counts = _proportional_holdout_counts(class_counts, TEST_SAMPLES)
    train_counts = class_counts - validation_counts - test_counts
    if np.any(train_counts < 0) or int(train_counts.sum()) != TRAIN_SAMPLES:
        raise ValueError("EuroSAT proportional allocation cannot satisfy the requested split sizes")
    generator = np.random.default_rng(SEED)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for class_id in range(CLASS_COUNT):
        class_indices = np.flatnonzero(targets == class_id)
        shuffled = generator.permutation(class_indices)
        train_count = int(train_counts[class_id])
        validation_count = int(validation_counts[class_id])
        train_indices.extend(int(index) for index in shuffled[:train_count])
        validation_indices.extend(
            int(index) for index in shuffled[train_count : train_count + validation_count]
        )
        test_indices.extend(int(index) for index in shuffled[train_count + validation_count :])

    train_indices.sort()
    validation_indices.sort()
    test_indices.sort()
    payload = {
        "dataset": "EuroSAT",
        "sample_count": len(dataset),
        "class_count": CLASS_COUNT,
        "classes": dataset.classes,
        "seed": SEED,
        "index_definition": (
            "Index i is the zero-based position in ImageFolder.samples after sorting every sample "
            "by its POSIX relative path from the resolved ImageFolder root."
        ),
        "sorting_rule": "relative_path.as_posix() ascending",
        "resolved_imagefolder_root": str(image_root),
        "stratification": {
            "algorithm": "numpy.random.default_rng(0).permutation within each class",
            "rounding_rule": (
                "For validation and test independently, floor(n_c * 2500 / 27000), then assign "
                "the remaining samples by descending fractional remainder; ties use ascending class index. "
                "Validation allocation is completed before the independent test allocation."
            ),
        },
        "torchvision_version": torchvision.__version__,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "class_distributions": {
            "overall": {dataset.classes[class_id]: int(class_counts[class_id]) for class_id in range(CLASS_COUNT)},
            "train": _distribution(train_indices, dataset.targets, dataset.classes),
            "validation": _distribution(validation_indices, dataset.targets, dataset.classes),
            "test": _distribution(test_indices, dataset.targets, dataset.classes),
        },
    }
    encoded = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root / "datasets" / "eurosat")
    parser.add_argument("--output", type=Path, default=repo_root / "P0B_EUROSAT_SPLIT_FROZEN.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        digest = freeze(args.data_root, args.output)
    except FileNotFoundError as error:
        print(f"SKIP: {error}")
        return 0
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
