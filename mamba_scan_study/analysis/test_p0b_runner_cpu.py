"""CPU-only tests for P0-B data, ledger, checkpoint, and structural preflight."""

from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest import mock

import numpy as np
import torch

from mamba_scan_study.experiments import p0b_data
from mamba_scan_study.experiments.p0b_data import (
    build_p0b_loaders,
    int64_c_sha256,
    load_frozen_split,
    uint8_c_sha256,
)
from mamba_scan_study.experiments.p0b_path_bank import P0BSourcePaths, default_source_paths, resolve_p0b_paths
from mamba_scan_study.experiments import run_p0b_preflight
from mamba_scan_study.experiments.run_p0b_feasibility import (
    FORMAL_ACCUM_STEPS,
    FORMAL_CONFIG,
    FORMAL_MICRO_BATCH,
    _atomic_write_bytes,
    _validate_cli,
    build_metadata,
    construct_requested_model,
    generate_ledger_rows,
    parse_args,
    resolved_micro_batch,
    run_one_cell,
    validate_completed_run,
    verify_ledger,
    write_completed_run,
    write_ledger,
)
from mamba_scan_study.models.backbone import ChannelSplitBackbone


class FakeCIFAR10:
    calls: list[bool] = []
    data = np.zeros((50_000, 32, 32, 3), dtype=np.uint8)
    targets = np.zeros(50_000, dtype=np.int64).tolist()

    def __init__(self, *, root, train, transform, download):
        del root, transform, download
        if train is not True:
            raise AssertionError("P0-B must never construct CIFAR10(train=False)")
        self.__class__.calls.append(train)
        self.data = self.__class__.data
        self.targets = self.__class__.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        raise AssertionError("CPU data contract tests must not iterate a loader")


def _expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _fake_frozen_split():
    frozen = load_frozen_split()
    targets = np.empty(50_000, dtype=np.int64)
    targets[frozen.train_indices] = np.repeat(np.arange(10, dtype=np.int64), 4500)
    targets[frozen.validation_indices] = np.repeat(np.arange(10, dtype=np.int64), 500)
    FakeCIFAR10.targets = targets.tolist()
    return replace(
        frozen,
        images_sha256=uint8_c_sha256(FakeCIFAR10.data),
        targets_sha256=int64_c_sha256(targets),
    )


def _cpu_model(resolution):
    return ChannelSplitBackbone(
        img_size=32,
        patch_size=32 // resolution.grid,
        in_chans=3,
        d_model=256,
        n_layers=0,
        block_type="gru",
        n_classes=10,
        variant="channel_same_row_4",
        pos_mode="xy_learned",
        channel_orders=resolution.channel_orders,
    )


def _complete_validation_history(epochs=FORMAL_CONFIG.epochs):
    return [
        {
            "epoch": epoch,
            "learning_rate": 0.001,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "validation_loss": 1.0,
            "validation_accuracy": 0.5,
        }
        for epoch in range(1, epochs + 1)
    ]


def _cpu_metadata(resolution, model, ledger_sha256, *, validation_history=None):
    metadata = build_metadata(
        resolution,
        model,
        ledger_sha256,
        validation_history=_complete_validation_history() if validation_history is None else validation_history,
        micro_batch=FORMAL_MICRO_BATCH,
        accum_steps=FORMAL_ACCUM_STEPS,
    )
    assert metadata["micro_batch"] == 128
    assert metadata["accum_steps"] == 1
    assert metadata["training_config"]["micro_batch"] == 128
    assert metadata["training_config"]["accum_steps"] == 1
    return metadata


def _replace_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".replacement")
    with open(temporary, "wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def test_split_loader_contract_and_hash_failures():
    frozen = _fake_frozen_split()
    FakeCIFAR10.calls = []
    loaders = build_p0b_loaders(
        "temporary-data-root",
        128,
        2,
        num_workers=0,
        download=False,
        pin_memory=False,
        cifar10_factory=FakeCIFAR10,
        frozen_split=frozen,
    )
    assert len(loaders.train.dataset) == 45_000
    assert len(loaders.validation.dataset) == 5_000
    assert loaders.train.sampler.generator.initial_seed() == 2
    assert loaders.validation.sampler.__class__.__name__ == "SequentialSampler"
    assert FakeCIFAR10.calls == [True, True]
    assert np.array_equal(np.bincount(np.asarray(FakeCIFAR10.targets)[frozen.train_indices], minlength=10), np.full(10, 4500))
    assert np.array_equal(np.bincount(np.asarray(FakeCIFAR10.targets)[frozen.validation_indices], minlength=10), np.full(10, 500))
    _expect_value_error(
        lambda: build_p0b_loaders(
            "temporary-data-root", 128, 0, num_workers=0, download=False, pin_memory=False,
            cifar10_factory=FakeCIFAR10, frozen_split=replace(frozen, images_sha256="0" * 64),
        )
    )
    _expect_value_error(
        lambda: build_p0b_loaders(
            "temporary-data-root", 128, 0, num_workers=0, download=False, pin_memory=False,
            cifar10_factory=FakeCIFAR10, frozen_split=replace(frozen, targets_sha256="0" * 64),
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        paths = default_source_paths()
        tampered_split = Path(directory) / paths.validation_split.name
        tampered_split.write_bytes(paths.validation_split.read_bytes() + b"\n")
        _expect_value_error(lambda: load_frozen_split(replace(paths, validation_split=tampered_split)))
    source = Path(p0b_data.__file__).read_text(encoding="utf-8")
    assert "build_real_loaders" not in source
    assert "PCG64" not in source


def test_ledger_generation_validation_and_sha():
    rows = generate_ledger_rows()
    assert len(rows) == 104
    assert len({(row["exp_id"], row["grid"], row["training_seed"]) for row in rows}) == 104
    with tempfile.TemporaryDirectory() as directory:
        ledger = Path(directory) / "P0B_RUN_LEDGER_104.csv"
        written_sha = write_ledger(ledger)
        assert verify_ledger(ledger) == written_sha
        ledger.write_bytes(ledger.read_bytes() + b"\n")
        _expect_value_error(lambda: verify_ledger(ledger))


def test_cli_rejects_unfrozen_overrides_and_debug_leakage():
    with redirect_stderr(io.StringIO()):
        try:
            parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--epochs", "2"])
        except SystemExit:
            pass
        else:
            raise AssertionError("unfrozen --epochs CLI override was accepted")
    formal = parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run", "--micro-batch", "128"])
    _validate_cli(formal)
    assert resolved_micro_batch(formal) == (128, 1)
    _expect_value_error(lambda: _validate_cli(parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run", "--micro-batch", "64"])))
    _expect_value_error(lambda: _validate_cli(parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run", "--micro-batch", "1"])))
    debug = parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run", "--mode", "debug", "--debug-root", "temporary-debug", "--micro-batch", "64"])
    _validate_cli(debug)
    assert resolved_micro_batch(debug) == (64, 2)
    _expect_value_error(lambda: _validate_cli(parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run", "--micro-batch", "7"])))
    _expect_value_error(lambda: _validate_cli(parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run", "--mode", "debug"])))
    _expect_value_error(lambda: _validate_cli(parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run", "--debug-root", "temporary-debug"])))
    dry_run = run_one_cell(parse_args(["--exp-id", "GEO_SG1", "--grid", "8", "--training-seed", "0", "--dry-run"]))
    assert "micro_batch=128" in dry_run
    assert "accum_steps=1" in dry_run


def test_checkpoint_validation_skip_and_fail_closed_order():
    resolution = resolve_p0b_paths("GEO_DIV", 8, 1)
    with tempfile.TemporaryDirectory() as directory:
        ledger = Path(directory) / "P0B_RUN_LEDGER_104.csv"
        ledger_sha = write_ledger(ledger)
        model = _cpu_model(resolution)
        metadata = _cpu_metadata(resolution, model, ledger_sha)
        final_checkpoint, metadata_path, completed_marker = write_completed_run(directory, model, metadata)
        fresh_model = _cpu_model(resolution)
        expected = _cpu_metadata(resolution, fresh_model, ledger_sha)
        assert validate_completed_run(final_checkpoint, metadata_path, completed_marker, fresh_model, expected)

        def reject_history(label, history):
            broken_metadata = _cpu_metadata(resolution, model, ledger_sha, validation_history=history)
            broken_directory = Path(directory) / label
            paths = write_completed_run(broken_directory, model, broken_metadata)
            rejected_model = _cpu_model(resolution)
            with mock.patch.object(rejected_model, "load_state_dict", wraps=rejected_model.load_state_dict) as strict_load:
                _expect_value_error(lambda: validate_completed_run(*paths, rejected_model, expected))
                strict_load.assert_not_called()

        reject_history("empty-history", [])
        reject_history("ninety-nine-history", _complete_validation_history(99))
        duplicate_history = _complete_validation_history()
        duplicate_history[1]["epoch"] = 1
        reject_history("duplicate-epoch", duplicate_history)
        nan_history = _complete_validation_history()
        nan_history[0]["validation_loss"] = float("nan")
        reject_history("nan-history", nan_history)
        inf_history = _complete_validation_history()
        inf_history[0]["learning_rate"] = float("inf")
        reject_history("inf-history", inf_history)
        test_field_history = _complete_validation_history()
        test_field_history[0]["test_acc"] = 0.5
        reject_history("forbidden-history-field", test_field_history)

        bad_metadata = dict(metadata)
        bad_metadata["channel_order_sha256"] = list(metadata["channel_order_sha256"])
        bad_metadata["channel_order_sha256"][0] = "0" * 64
        mismatch_directory = Path(directory) / "path-mismatch"
        path_checkpoint, path_metadata, path_marker = write_completed_run(mismatch_directory, model, bad_metadata)
        _expect_value_error(lambda: validate_completed_run(path_checkpoint, path_metadata, path_marker, _cpu_model(resolution), expected))

        source_metadata = dict(metadata)
        source_metadata["split_source_sha256"] = "0" * 64
        source_directory = Path(directory) / "source-mismatch"
        source_checkpoint, source_metadata_path, source_marker = write_completed_run(source_directory, model, source_metadata)
        _expect_value_error(lambda: validate_completed_run(source_checkpoint, source_metadata_path, source_marker, _cpu_model(resolution), expected))

        checkpoint = torch.load(final_checkpoint, map_location="cpu")
        checkpoint["model_state"]["channel_permutations"] = checkpoint["model_state"]["channel_permutations"].clone()
        checkpoint["model_state"]["channel_permutations"][0, 0] = 1
        _replace_checkpoint(final_checkpoint, checkpoint)
        buffer_model = _cpu_model(resolution)
        with mock.patch.object(buffer_model, "load_state_dict", wraps=buffer_model.load_state_dict) as strict_load:
            _expect_value_error(lambda: validate_completed_run(final_checkpoint, metadata_path, completed_marker, buffer_model, expected))
            strict_load.assert_not_called()

        partial_directory = Path(directory) / "partial"
        partial_directory.mkdir()
        partial_checkpoint = partial_directory / "final_checkpoint.pt"
        torch.save({"status": "running"}, partial_checkpoint)
        assert not validate_completed_run(
            partial_checkpoint,
            partial_directory / "metadata.json",
            partial_directory / "completed.json",
            _cpu_model(resolution),
            expected,
        )


def test_atomic_helper_validation_schema_and_c6():
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "atomic.bin"
        _atomic_write_bytes(target, b"frozen")
        assert target.read_bytes() == b"frozen"
        assert not list(Path(directory).glob("*.tmp"))

        ledger = Path(directory) / "P0B_RUN_LEDGER_104.csv"
        write_ledger(ledger)
        preflight = run_p0b_preflight.run_preflight(ledger)
        assert set(preflight["c6_by_grid"]) == {"grid8", "grid32"}
        assert preflight["c6_by_grid"]["grid8"]["nominal_flops_equality_signature"] != preflight["c6_by_grid"]["grid32"]["nominal_flops_equality_signature"]
        for grid_key in ("grid8", "grid32"):
            c6 = preflight["c6_by_grid"][grid_key]
            assert c6["group_count"] == 4
            assert c6["blocks_per_group"] == (2, 2, 2, 2)
            assert all(len(group) == 2 for group in c6["group_block_schema"])
            assert all(block[0] == "GRUBlock" for group in c6["group_block_schema"] for block in group)
            assert ("micro_batch", 128) in c6["formal_training_plan_signature"]
            assert ("accum_steps", 1) in c6["formal_training_plan_signature"]
        assert preflight["c1_to_c5"]["grid8"]["four_g_macro"] > preflight["c1_to_c5"]["grid8"]["two_g_macro"]

        for grid in (8, 32):
            g_orders = [resolve_p0b_paths(f"GEO_SG{index}", grid, 0).channel_orders[0] for index in range(1, 5)]
            corrupted_g4 = torch.arange(grid * grid, dtype=torch.long)
            try:
                run_p0b_preflight.check_c2_edgewise(grid, *g_orders[:3], corrupted_g4)
            except AssertionError:
                pass
            else:
                raise AssertionError("C2 accepted a corrupted G4 order")

        resolution = resolve_p0b_paths("LOC_D", 32, 2)
        metadata = _cpu_metadata(resolution, _cpu_model(resolution), verify_ledger(ledger))
        encoded = json.dumps(metadata, sort_keys=True)
        assert "test_loss" not in encoded
        assert "test_acc" not in encoded
        assert "test_metrics" not in encoded


def test_stage1_files_remain_unmodified_and_runtime_has_no_randperm():
    root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "mamba_scan_study/experiments/run_stage1_seed0.py",
        "mamba_scan_study/data/real_datasets.py",
    ):
        head_bytes = subprocess.check_output(["git", "show", f"HEAD:{relative_path}"], cwd=root)
        worktree_bytes = (root / relative_path).read_bytes()
        assert head_bytes == worktree_bytes.replace(b"\r\n", b"\n")
    runner_source = Path(__file__).resolve().parents[1] / "experiments" / "run_p0b_feasibility.py"
    runner_text = runner_source.read_text(encoding="utf-8")
    assert "torch.randperm(" not in runner_text
    assert runner_text.index("set_seed(args.training_seed)") < runner_text.index("model = construct_requested_model")
    assert FORMAL_CONFIG.amp is True


def main():
    torch.set_num_threads(1)
    test_split_loader_contract_and_hash_failures()
    test_ledger_generation_validation_and_sha()
    test_cli_rejects_unfrozen_overrides_and_debug_leakage()
    test_checkpoint_validation_skip_and_fail_closed_order()
    test_atomic_helper_validation_schema_and_c6()
    test_stage1_files_remain_unmodified_and_runtime_has_no_randperm()
    print("PASS: P0-B data split, ledger, CLI, checkpoint, validation schema, C1-C6, and Stage-1 guards")


if __name__ == "__main__":
    main()
