"""CPU-only wiring tests for P0-B dataset and augmentation dimensions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import torch

from mamba_scan_study.experiments import run_p0b_feasibility as runner
from mamba_scan_study.experiments.p0b_path_bank import resolve_p0b_paths


class _CompletedModelDouble:
    def __init__(self, checkpoint: dict) -> None:
        state = checkpoint["model_state"]
        self.channel_permutations = state["channel_permutations"].detach().clone()
        self.channel_inverse_permutations = state["channel_inverse_permutations"].detach().clone()
        self._parameter = torch.nn.Parameter(torch.empty(282122))
        self.loaded = False

    def to(self, device: torch.device):
        del device
        return self

    def parameters(self):
        return iter((self._parameter,))

    def load_state_dict(self, state: dict, strict: bool = True):
        assert strict is True
        assert torch.equal(state["channel_permutations"], self.channel_permutations)
        assert torch.equal(state["channel_inverse_permutations"], self.channel_inverse_permutations)
        self.loaded = True
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


class _CapturingBackbone:
    def __init__(self, **kwargs) -> None:
        self.head = SimpleNamespace(out_features=kwargs["n_classes"])

    def to(self, device: torch.device):
        del device
        return self


class P0BRunnerWiringTests(unittest.TestCase):
    def test_default_path_is_byte_for_byte_legacy_format(self) -> None:
        observed = runner._run_directory("GEO_SG1", 8, 0, "formal", None)
        expected = runner.FORMAL_RUN_ROOT / "p0b_GEO_SG1_R_low_seed0"
        self.assertEqual(observed, expected)

    def test_dataset_augmentation_paths_are_unique(self) -> None:
        paths = {
            runner._run_directory("GEO_SG1", 8, 0, "formal", None, dataset, augmentation)
            for dataset in runner.DATASET_CLASS_COUNTS
            for augmentation in ("p0b_legacy", "main_uniform")
        }
        self.assertEqual(len(paths), len(runner.DATASET_CLASS_COUNTS) * 2)

    def test_existing_completed_run_skips_with_legacy_metadata(self) -> None:
        run_directory = runner.FORMAL_RUN_ROOT / "p0b_GEO_SG1_R_low_seed0"
        checkpoint = torch.load(run_directory / "final_checkpoint.pt", map_location="cpu")
        with mock.patch.object(runner, "construct_requested_model", side_effect=lambda *args: _CompletedModelDouble(checkpoint)):
            result = runner.run_one_cell(
                runner.parse_args(
                    [
                        "--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0",
                        "--data-root", "datasets", "--execute",
                    ]
                )
            )
        self.assertEqual(result, "COMPLETED_SKIP GEO_SG1 grid=8 seed=0")

    def test_dry_run_keeps_four_source_sha_gates_active(self) -> None:
        result = runner.run_one_cell(
            runner.parse_args(
                ["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run"]
            )
        )
        self.assertTrue(result.startswith("DRY_RUN_OK GEO_SG1 grid=8 seed=0"))

    def test_organ_model_uses_eleven_class_head(self) -> None:
        resolution = resolve_p0b_paths("GEO_SG1", 8, 0)
        with mock.patch.object(runner, "ChannelSplitBackbone", _CapturingBackbone):
            model = runner.construct_requested_model(resolution, torch.device("cpu"), "organamnist")
        self.assertEqual(model.head.out_features, 11)

    def test_metadata_and_c6_signatures_are_dataset_scoped(self) -> None:
        resolution = resolve_p0b_paths("GEO_SG1", 8, 0)
        checkpoint = torch.load(runner.FORMAL_RUN_ROOT / "p0b_GEO_SG1_R_low_seed0" / "final_checkpoint.pt", map_location="cpu")
        metadata = runner.build_metadata(
            resolution,
            _CompletedModelDouble(checkpoint),
            "ledger-sha",
            runtime_config=runner.replace(runner.FORMAL_CONFIG, dataset="organamnist"),
            dataset="organamnist",
            augmentation="main_uniform",
        )
        self.assertEqual(metadata["dataset"], "organamnist")
        self.assertEqual(metadata["augmentation"], "main_uniform")
        self.assertEqual(metadata["training_config"]["dataset"], "organamnist")
        self.assertEqual(
            runner.nominal_flops_equality_signature(8, "organamnist"),
            runner.nominal_flops_equality_signature(8, "organamnist"),
        )
        self.assertNotEqual(
            runner.nominal_flops_equality_signature(8, "organamnist"),
            runner.nominal_flops_equality_signature(8, "cifar10"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
