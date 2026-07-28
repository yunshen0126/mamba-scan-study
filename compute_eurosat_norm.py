"""Compute EuroSAT normalization from the frozen train split only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from freeze_eurosat_split import _sorted_imagefolder
from mamba_scan_study.experiments.p0b_data import _resolve_eurosat_root


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root / "datasets" / "eurosat")
    parser.add_argument("--split", type=Path, default=repo_root / "P0B_EUROSAT_SPLIT_FROZEN.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _moments_from_images(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    images = images.to(torch.float64)
    return images.sum(dim=(0, 2, 3)), images.square().sum(dim=(0, 2, 3)), images.shape[0] * images.shape[2] * images.shape[3]


def _finalize(channel_sum: torch.Tensor, channel_sum_sq: torch.Tensor, pixel_count: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mean = channel_sum / pixel_count
    std = (channel_sum_sq / pixel_count - mean.square()).sqrt()
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def _compute_one_by_one(dataset: object, indices: list[int]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for index in indices:
        image, _ = dataset[index]
        image_sum, image_sum_sq, image_pixels = _moments_from_images(image.unsqueeze(0))
        channel_sum += image_sum
        channel_sum_sq += image_sum_sq
        pixel_count += image_pixels
    return _finalize(channel_sum, channel_sum_sq, pixel_count)


def _compute_batched(dataset: object, indices: list[int], batch_size: int, num_workers: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in loader:
        batch_sum, batch_sum_sq, batch_pixels = _moments_from_images(images)
        channel_sum += batch_sum
        channel_sum_sq += batch_sum_sq
        pixel_count += batch_pixels
    return _finalize(channel_sum, channel_sum_sq, pixel_count)


def _dataset_and_indices(data_root: Path, split_path: Path) -> tuple[object, list[int]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"frozen EuroSAT split does not exist: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    train_indices = [int(index) for index in payload.get("train_indices", [])]
    if len(train_indices) != 22_000 or len(set(train_indices)) != 22_000:
        raise ValueError("frozen EuroSAT split must contain 22,000 unique train indices")

    dataset = _sorted_imagefolder(_resolve_eurosat_root(data_root))
    if len(dataset) != 27_000 or any(index < 0 or index >= len(dataset) for index in train_indices):
        raise ValueError("frozen EuroSAT train indices do not match the sorted ImageFolder population")
    dataset.transform = transforms.Compose(
        [
            transforms.Resize((32, 32), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    return dataset, train_indices


def compute(data_root: Path, split_path: Path, batch_size: int = 128, num_workers: int = 0) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    dataset, train_indices = _dataset_and_indices(data_root, split_path)
    return _compute_batched(dataset, train_indices, batch_size, num_workers)


def main() -> int:
    args = parse_args()
    try:
        dataset, train_indices = _dataset_and_indices(args.data_root, args.split)
        mean, std = _compute_batched(dataset, train_indices, args.batch_size, args.num_workers)
        reference_mean, reference_std = _compute_one_by_one(dataset, train_indices[:200])
        batched_mean, batched_std = _compute_batched(dataset, train_indices[:200], args.batch_size, args.num_workers)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    if max(abs(left - right) for left, right in zip(reference_mean + reference_std, batched_mean + batched_std)) > 1e-6:
        print("ERROR: batched and one-by-one normalization disagree on the first 200 train samples")
        return 1
    print("first_200_batched_vs_one_by_one_max_abs_diff<=1e-6")
    print("mean=" + ", ".join(f"{value:.10f}" for value in mean))
    print("std=" + ", ".join(f"{value:.10f}" for value in std))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
