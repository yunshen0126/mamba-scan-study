"""CPU-only contract tests for P0-B's multi-dataset data layer."""

from __future__ import annotations

import itertools
from pathlib import Path
import random
import subprocess
import sys
import types
import unittest

import numpy as np
import torch

from mamba_scan_study.experiments import p0b_data
from mamba_scan_study.experiments.p0b_data import build_p0b_loaders


DATA_ROOT = Path("datasets")
ORGAN_LENGTHS = {
    "organamnist": (34_561, 6_491),
    "organcmnist": (12_975, 2_392),
    "organsmnist": (13_932, 2_452),
}


def _reset_rng() -> None:
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)


def _sample_identities(dataset: object) -> set[tuple[str, int]]:
    if isinstance(dataset, torch.utils.data.Subset):
        return {("imagefolder", int(index)) for index in dataset.indices}
    split = str(getattr(dataset, "split"))
    return {(split, index) for index in range(len(dataset))}


class P0BMultiDatasetTests(unittest.TestCase):
    def _assert_batch_contract(self, loaders, class_count: int) -> None:
        images, labels = next(iter(loaders.train))
        self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))
        self.assertEqual(images.dtype, torch.float32)
        self.assertTrue(torch.all((labels >= 0) & (labels < class_count)).item())

    def test_main_uniform_transform_structure(self) -> None:
        for dataset in p0b_data.DATASET_CLASS_COUNTS:
            train_transform, validation_transform = p0b_data._transforms(dataset, "main_uniform")
            train_names = [type(transform).__name__ for transform in train_transform.transforms]
            validation_names = [type(transform).__name__ for transform in validation_transform.transforms]
            self.assertIn("RandomCrop", train_names)
            self.assertNotIn("RandomHorizontalFlip", train_names)
            self.assertNotIn("RandomHorizontalFlip", validation_names)
            self.assertNotIn("RandomCrop", validation_names)

    def test_organ_datasets(self) -> None:
        for dataset, (expected_train, expected_validation) in ORGAN_LENGTHS.items():
            with self.subTest(dataset=dataset):
                loaders = build_p0b_loaders(
                    DATA_ROOT,
                    batch_size=32,
                    training_seed=0,
                    num_workers=0,
                    download=False,
                    pin_memory=False,
                    augmentation="main_uniform",
                    dataset=dataset,
                )
                self.assertEqual(len(loaders.train.dataset), expected_train)
                self.assertEqual(len(loaders.validation.dataset), expected_validation)
                self._assert_batch_contract(loaders, 11)
                train_labels = np.asarray(loaders.train.dataset.labels).reshape(-1)
                self.assertTrue(np.all(np.bincount(train_labels, minlength=11) > 0))
                self.assertFalse(
                    _sample_identities(loaders.train.dataset)
                    & _sample_identities(loaders.validation.dataset)
                )

    def test_eurosat_when_available(self) -> None:
        if not (DATA_ROOT / "eurosat").is_dir() or not Path("P0B_EUROSAT_SPLIT_FROZEN.json").is_file():
            self.skipTest("EuroSAT ImageFolder or frozen split is unavailable")
        loaders = build_p0b_loaders(
            DATA_ROOT,
            batch_size=32,
            training_seed=0,
            num_workers=0,
            download=False,
            pin_memory=False,
            augmentation="main_uniform",
            dataset="eurosat",
        )
        self.assertEqual(len(loaders.train.dataset), 22_000)
        self.assertEqual(len(loaders.validation.dataset), 2_500)
        self._assert_batch_contract(loaders, 10)
        train_indices = set(loaders.train.dataset.indices)
        validation_indices = set(loaders.validation.dataset.indices)
        self.assertFalse(train_indices & validation_indices)
        train_labels = np.asarray(loaders.train.dataset.dataset.targets)[list(train_indices)]
        self.assertTrue(np.all(np.bincount(train_labels, minlength=10) > 0))

    def test_cifar10_legacy_regression_when_reference_is_available(self) -> None:
        cifar_root = DATA_ROOT / "cifar-10-batches-py"
        self.assertTrue(cifar_root.is_dir(), "CIFAR-10 training data is required for the legacy regression")
        source = subprocess.check_output(
            ["git", "show", "b39ba1a:mamba_scan_study/experiments/p0b_data.py"], text=True
        )
        module_name = "mamba_scan_study.experiments._p0b_data_b39ba1a"
        legacy = types.ModuleType(module_name)
        legacy.__package__ = "mamba_scan_study.experiments"
        sys.modules[module_name] = legacy
        try:
            exec(compile(source, "b39ba1a:p0b_data.py", "exec"), legacy.__dict__)
            def collect(module):
                _reset_rng()
                loaders = module.build_p0b_loaders(
                    DATA_ROOT, 32, 2, num_workers=0, download=False, pin_memory=False
                )
                return list(itertools.islice(loaders.train, 3)), list(itertools.islice(loaders.validation, 3))
            expected_train, expected_validation = collect(legacy)
            observed_train, observed_validation = collect(p0b_data)
        finally:
            sys.modules.pop(module_name, None)
        for expected, observed in zip(expected_train + expected_validation, observed_train + observed_validation):
            self.assertTrue(torch.equal(expected[0], observed[0]))
            self.assertTrue(torch.equal(expected[1], observed[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
