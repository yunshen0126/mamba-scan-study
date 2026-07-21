"""Dedicated P0-B feasibility runner with frozen protocol guards."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Iterable, Mapping
import uuid

import torch

from mamba_scan_study.experiments.p0b_data import build_p0b_loaders
from mamba_scan_study.experiments.p0b_path_bank import (
    EXPECTED_SOURCE_SHA256,
    P0B_EXP_IDS,
    P0B_GRIDS,
    P0B_TRAINING_SEEDS,
    P0BPathResolution,
    P0BSourcePaths,
    default_source_paths,
    iter_p0b_design_cells,
    resolve_p0b_paths,
    verify_source_hashes,
)
from mamba_scan_study.models.backbone import ChannelSplitBackbone


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_FILENAME = "P0B_RUN_LEDGER_104.csv"
FORMAL_RUN_ROOT = REPO_ROOT / "outputs"
FORMAL_CONFIG_REQUIRED_TEXT = (
    "--arch channel_split",
    "--dataset cifar10",
    "--d-model 256",
    "--n-layers 2",
    "--block mamba",
    "--effective-batch 128",
    "--epochs 100",
    "--warmup-epochs 5",
    "--base-lr 0.001",
    "--weight-decay 0.05",
    "--grad-clip 1.0",
    "--pos-mode xy_learned",
    "--num-workers 4",
    "R_low = grid8 (patch4, L=64)",
    "R_high = grid32 (patch1, L=1024)",
    "**seed:** 0,1,2,3",
)


@dataclass(frozen=True)
class FormalP0BConfig:
    protocol: str = "P0B"
    dataset: str = "cifar10"
    arch: str = "channel_split"
    block_type: str = "mamba"
    d_model: int = 256
    n_layers: int = 2
    pos_mode: str = "xy_learned"
    epochs: int = 100
    warmup_epochs: int = 5
    optimizer: str = "AdamW"
    base_lr: float = 0.001
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    effective_batch: int = 128
    amp: bool = True
    num_workers: int = 4
    img_size: int = 32


FORMAL_CONFIG = FormalP0BConfig()
FORMAL_MICRO_BATCH = 128
FORMAL_ACCUM_STEPS = 1
LEDGER_FIELDS = (
    "exp_id",
    "grid",
    "patch_size",
    "training_seed",
    "reliance",
    "path_family",
    "single_or_diverse",
    "channel_path_id_0",
    "channel_path_id_1",
    "channel_path_id_2",
    "channel_path_id_3",
    "channel_order_sha256_0",
    "channel_order_sha256_1",
    "channel_order_sha256_2",
    "channel_order_sha256_3",
    "channel_inverse_order_sha256_0",
    "channel_inverse_order_sha256_1",
    "channel_inverse_order_sha256_2",
    "channel_inverse_order_sha256_3",
    "latin_square_rotation",
    "lmto_source_sha256",
    "random_source_sha256",
    "split_source_sha256",
    "config_source_sha256",
)
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "protocol",
        "exp_id",
        "reliance",
        "grid",
        "patch_size",
        "training_seed",
        "channel_path_ids",
        "channel_order_sha256",
        "channel_inverse_order_sha256",
        "lmto_source_sha256",
        "random_source_sha256",
        "split_source_sha256",
        "config_source_sha256",
        "ledger_sha256",
        "latin_square_rotation",
        "path_family",
        "single_or_diverse",
        "base_variant",
        "shuffle_seed",
        "shuffle_seed_note",
        "parameter_count",
        "architecture_signature",
        "operator_signature",
        "nominal_flops_method",
        "nominal_flops_equality_signature",
        "git_commit",
        "git_dirty",
        "micro_batch",
        "accum_steps",
        "training_config",
        "validation_history",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def patch_size_for_grid(grid: int) -> int:
    if grid not in P0B_GRIDS:
        raise ValueError(f"P0-B grid must be one of {P0B_GRIDS}")
    return FORMAL_CONFIG.img_size // grid


def reliance_for_grid(grid: int) -> str:
    return "R_low" if grid == 8 else "R_high"


def verify_formal_config(source_paths: P0BSourcePaths | None = None) -> dict[str, str]:
    """Verify source bytes, then assert every documented formal field."""
    paths = default_source_paths() if source_paths is None else source_paths
    source_sha256 = verify_source_hashes(paths)
    text = Path(paths.config).read_text(encoding="utf-8")
    for token in FORMAL_CONFIG_REQUIRED_TEXT:
        if token not in text:
            raise ValueError(f"frozen P0-B configuration is missing {token!r}")
    if (
        FORMAL_CONFIG.optimizer != "AdamW"
        or FORMAL_CONFIG.amp is not True
        or FORMAL_MICRO_BATCH != FORMAL_CONFIG.effective_batch
        or FORMAL_ACCUM_STEPS != 1
    ):
        raise AssertionError("P0-B optimizer and AMP are fixed protocol fields")
    return source_sha256


def architecture_operator_signature(grid: int) -> dict[str, object]:
    """Path-independent formal architecture signature; no numeric Mamba FLOPs claim."""
    return {
        "model": "ChannelSplitBackbone",
        "dataset": FORMAL_CONFIG.dataset,
        "block_type": FORMAL_CONFIG.block_type,
        "d_model": FORMAL_CONFIG.d_model,
        "n_layers": FORMAL_CONFIG.n_layers,
        "pos_mode": FORMAL_CONFIG.pos_mode,
        "img_size": FORMAL_CONFIG.img_size,
        "grid": grid,
        "patch_size": patch_size_for_grid(grid),
        "branch_dirs": ["row", "row", "row", "row"],
        "base_variant": "channel_same_row_4",
        "explicit_path_control_flow": "row_flatten+order_index_select+inverse_index_select",
    }


def nominal_flops_equality_signature(grid: int) -> str:
    return _sha256_bytes(_canonical_json_bytes(architecture_operator_signature(grid)))


def _ledger_row(resolution: P0BPathResolution) -> dict[str, str]:
    row = {
        "exp_id": resolution.exp_id,
        "grid": str(resolution.grid),
        "patch_size": str(patch_size_for_grid(resolution.grid)),
        "training_seed": str(resolution.training_seed),
        "reliance": reliance_for_grid(resolution.grid),
        "path_family": resolution.path_family,
        "single_or_diverse": resolution.single_or_diverse,
        "latin_square_rotation": "" if resolution.latin_square_rotation is None else str(resolution.latin_square_rotation),
        "lmto_source_sha256": resolution.source_sha256["lmto"],
        "random_source_sha256": resolution.source_sha256["random"],
        "split_source_sha256": resolution.source_sha256["validation_split"],
        "config_source_sha256": resolution.source_sha256["config"],
    }
    for index in range(4):
        row[f"channel_path_id_{index}"] = resolution.channel_path_ids[index]
        row[f"channel_order_sha256_{index}"] = resolution.channel_order_sha256[index]
        row[f"channel_inverse_order_sha256_{index}"] = resolution.channel_inverse_order_sha256[index]
    return row


def generate_ledger_rows(source_paths: P0BSourcePaths | None = None) -> list[dict[str, str]]:
    paths = default_source_paths() if source_paths is None else source_paths
    return [_ledger_row(resolve_p0b_paths(*cell, paths)) for cell in iter_p0b_design_cells()]


def _ledger_bytes(rows: Iterable[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_ledger(path: str | Path, source_paths: P0BSourcePaths | None = None) -> str:
    rows = generate_ledger_rows(source_paths)
    validate_ledger_rows(rows, source_paths)
    payload = _ledger_bytes(rows)
    _atomic_write_bytes(Path(path), payload)
    return _sha256_bytes(payload)


def validate_ledger_rows(
    rows: list[Mapping[str, str]], source_paths: P0BSourcePaths | None = None
) -> None:
    expected = generate_ledger_rows(source_paths)
    if len(rows) != 104 or len(expected) != 104:
        raise ValueError("P0-B ledger must contain exactly 104 rows")
    keys = {(row["exp_id"], row["grid"], row["training_seed"]) for row in rows}
    if len(keys) != 104:
        raise ValueError("P0-B ledger contains duplicate design keys")
    if list(rows) != expected:
        raise ValueError("P0-B ledger does not exactly match the frozen design")


def verify_ledger(path: str | Path, source_paths: P0BSourcePaths | None = None) -> str:
    ledger_path = Path(path)
    payload = ledger_path.read_bytes()
    expected_rows = generate_ledger_rows(source_paths)
    validate_ledger_rows(expected_rows, source_paths)
    expected_payload = _ledger_bytes(expected_rows)
    if payload != expected_payload:
        raise ValueError("P0-B ledger bytes or SHA-256 do not match deterministic frozen content")
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != LEDGER_FIELDS:
        raise ValueError("P0-B ledger header is invalid")
    observed_rows = list(reader)
    validate_ledger_rows(observed_rows, source_paths)
    return _sha256_bytes(payload)


def construct_requested_model(resolution: P0BPathResolution, device: torch.device) -> ChannelSplitBackbone:
    return ChannelSplitBackbone(
        img_size=FORMAL_CONFIG.img_size,
        patch_size=patch_size_for_grid(resolution.grid),
        in_chans=3,
        d_model=FORMAL_CONFIG.d_model,
        n_layers=FORMAL_CONFIG.n_layers,
        block_type=FORMAL_CONFIG.block_type,
        n_classes=10,
        variant="channel_same_row_4",
        pos_mode=FORMAL_CONFIG.pos_mode,
        channel_orders=resolution.channel_orders,
    ).to(device)


def _git_value(arguments: list[str], fallback: str) -> str:
    try:
        return subprocess.check_output(arguments, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def build_metadata(
    resolution: P0BPathResolution,
    model: ChannelSplitBackbone,
    ledger_sha256: str,
    *,
    validation_history: list[dict[str, float]] | None = None,
    runtime_config: FormalP0BConfig = FORMAL_CONFIG,
    micro_batch: int = FORMAL_MICRO_BATCH,
    accum_steps: int = FORMAL_ACCUM_STEPS,
) -> dict[str, object]:
    if micro_batch <= 0 or runtime_config.effective_batch % micro_batch:
        raise ValueError("metadata micro-batch must divide effective batch")
    if accum_steps != runtime_config.effective_batch // micro_batch:
        raise ValueError("metadata accum_steps is inconsistent with micro-batch")
    if runtime_config == FORMAL_CONFIG and (
        micro_batch != FORMAL_MICRO_BATCH or accum_steps != FORMAL_ACCUM_STEPS
    ):
        raise ValueError("formal P0-B metadata must use frozen micro-batch 128 and accum_steps 1")
    signature = architecture_operator_signature(resolution.grid)
    return {
        "protocol": "P0B",
        "exp_id": resolution.exp_id,
        "reliance": reliance_for_grid(resolution.grid),
        "grid": resolution.grid,
        "patch_size": patch_size_for_grid(resolution.grid),
        "training_seed": resolution.training_seed,
        "channel_path_ids": list(resolution.channel_path_ids),
        "channel_order_sha256": list(resolution.channel_order_sha256),
        "channel_inverse_order_sha256": list(resolution.channel_inverse_order_sha256),
        "lmto_source_sha256": resolution.source_sha256["lmto"],
        "random_source_sha256": resolution.source_sha256["random"],
        "split_source_sha256": resolution.source_sha256["validation_split"],
        "config_source_sha256": resolution.source_sha256["config"],
        "ledger_sha256": ledger_sha256,
        "latin_square_rotation": resolution.latin_square_rotation,
        "path_family": resolution.path_family,
        "single_or_diverse": resolution.single_or_diverse,
        "base_variant": "channel_same_row_4",
        "shuffle_seed": None,
        "shuffle_seed_note": "explicit P0-B paths do not use shuffle_seed to control path content",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "architecture_signature": _sha256_bytes(_canonical_json_bytes(signature)),
        "operator_signature": signature["explicit_path_control_flow"],
        "nominal_flops_method": "path-independent architecture equality signature; no absolute Mamba FLOPs estimator",
        "nominal_flops_equality_signature": nominal_flops_equality_signature(resolution.grid),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"], "UNKNOWN"),
        "git_dirty": bool(_git_value(["git", "status", "--porcelain"], "UNKNOWN")),
        "micro_batch": micro_batch,
        "accum_steps": accum_steps,
        "training_config": {
            **asdict(runtime_config),
            "micro_batch": micro_batch,
            "accum_steps": accum_steps,
        },
        "validation_history": [] if validation_history is None else validation_history,
    }


def _require_metadata_complete(metadata: Mapping[str, object]) -> None:
    missing = REQUIRED_METADATA_FIELDS - set(metadata)
    if missing:
        raise ValueError(f"P0-B metadata is missing required fields: {sorted(missing)}")
    if metadata.get("protocol") != "P0B":
        raise ValueError("P0-B metadata protocol is invalid")
    if not isinstance(metadata.get("validation_history"), list):
        raise ValueError("P0-B validation_history must be a list")


def _expected_metadata_values(expected: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(key for key in REQUIRED_METADATA_FIELDS if key != "validation_history")


def _compare_metadata(observed: Mapping[str, object], expected: Mapping[str, object]) -> None:
    _require_metadata_complete(observed)
    for key in _expected_metadata_values(expected):
        if observed.get(key) != expected.get(key):
            raise ValueError(f"P0-B completed checkpoint metadata mismatch for {key}")


_VALIDATION_HISTORY_FIELDS = frozenset(
    {
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    }
)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_completed_validation_history(
    metadata: Mapping[str, object], runtime_config: FormalP0BConfig
) -> None:
    """Reject incomplete or malformed completed validation history before state loading."""
    history = metadata.get("validation_history")
    if not isinstance(history, list) or len(history) != runtime_config.epochs:
        raise ValueError("completed P0-B validation history length is invalid")
    training_config = metadata.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError("completed P0-B training_config is invalid")
    expected_training_config = {
        **asdict(runtime_config),
        "micro_batch": metadata.get("micro_batch"),
        "accum_steps": metadata.get("accum_steps"),
    }
    if dict(training_config) != expected_training_config:
        raise ValueError("completed P0-B training_config does not match runtime configuration")
    micro_batch = metadata.get("micro_batch")
    accum_steps = metadata.get("accum_steps")
    if not isinstance(micro_batch, int) or not isinstance(accum_steps, int):
        raise ValueError("completed P0-B micro-batch metadata is invalid")
    if micro_batch <= 0 or runtime_config.effective_batch % micro_batch:
        raise ValueError("completed P0-B micro-batch does not divide effective batch")
    if accum_steps != runtime_config.effective_batch // micro_batch:
        raise ValueError("completed P0-B accum_steps is inconsistent with micro-batch")
    if runtime_config == FORMAL_CONFIG and (
        micro_batch != FORMAL_MICRO_BATCH or accum_steps != FORMAL_ACCUM_STEPS
    ):
        raise ValueError("completed formal P0-B metadata violates frozen micro-batch")
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or set(row) != _VALIDATION_HISTORY_FIELDS:
            raise ValueError("completed P0-B validation history schema is invalid")
        if {"test_loss", "test_acc", "test_metrics"} & set(row):
            raise ValueError("completed P0-B history contains forbidden test metric fields")
        if type(row["epoch"]) is not int or row["epoch"] != expected_epoch:
            raise ValueError("completed P0-B validation history epochs are invalid")
        for field in _VALIDATION_HISTORY_FIELDS - {"epoch"}:
            if not _finite_number(row[field]):
                raise ValueError(f"completed P0-B validation history {field} is non-finite")
        for field in ("train_accuracy", "validation_accuracy"):
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ValueError(f"completed P0-B validation history {field} is outside [0, 1]")


def _tensor_equal(left: object, right: torch.Tensor) -> bool:
    return isinstance(left, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())


def validate_completed_run(
    final_checkpoint: str | Path,
    metadata_path: str | Path,
    completed_marker: str | Path,
    model: ChannelSplitBackbone,
    expected_metadata: Mapping[str, object],
    runtime_config: FormalP0BConfig = FORMAL_CONFIG,
) -> bool:
    """Return True only after metadata/buffer checks and a strict state load succeed."""
    checkpoint_path = Path(final_checkpoint)
    metadata_file = Path(metadata_path)
    marker_file = Path(completed_marker)
    if not checkpoint_path.is_file() or not metadata_file.is_file() or not marker_file.is_file():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("status") != "completed":
        return False
    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("P0-B completed checkpoint lacks metadata")
    external_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    if marker.get("status") != "completed":
        return False
    if external_metadata != checkpoint_metadata:
        raise ValueError("P0-B final checkpoint and metadata file disagree")
    if marker.get("metadata_sha256") != _sha256_bytes(_canonical_json_bytes(external_metadata)):
        raise ValueError("P0-B completed marker metadata SHA-256 mismatch")
    _compare_metadata(checkpoint_metadata, expected_metadata)
    validate_completed_validation_history(checkpoint_metadata, runtime_config)
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("P0-B completed checkpoint lacks model_state")
    if not _tensor_equal(state.get("channel_permutations"), model.channel_permutations):
        raise ValueError("P0-B checkpoint permutation buffers do not match requested model")
    if not _tensor_equal(state.get("channel_inverse_permutations"), model.channel_inverse_permutations):
        raise ValueError("P0-B checkpoint inverse buffers do not match requested model")
    model.load_state_dict(state, strict=True)
    return True


def write_completed_run(
    run_directory: str | Path,
    model: ChannelSplitBackbone,
    metadata: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    """Atomically write final checkpoint, metadata, then completed marker in that order."""
    _require_metadata_complete(metadata)
    directory = Path(run_directory)
    final_checkpoint = directory / "final_checkpoint.pt"
    metadata_path = directory / "metadata.json"
    completed_marker = directory / "completed.json"
    checkpoint = {
        "status": "completed",
        "protocol": "P0B",
        "metadata": dict(metadata),
        "model_state": model.state_dict(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary = final_checkpoint.with_name(f".{final_checkpoint.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_write_bytes(metadata_path, _canonical_json_bytes(metadata))
    marker = {"status": "completed", "metadata_sha256": _sha256_bytes(_canonical_json_bytes(metadata))}
    _atomic_write_bytes(completed_marker, _canonical_json_bytes(marker))
    return final_checkpoint, metadata_path, completed_marker


def _run_directory(exp_id: str, grid: int, training_seed: int, mode: str, debug_root: str | None) -> Path:
    name = f"p0b_{exp_id}_{reliance_for_grid(grid)}_seed{training_seed}"
    if mode == "formal":
        return FORMAL_RUN_ROOT / name
    if not debug_root:
        raise ValueError("debug mode requires --debug-root and never writes formal results")
    return Path(debug_root) / name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One frozen P0-B feasibility design cell.")
    parser.add_argument("--exp-id", choices=P0B_EXP_IDS, required=True)
    parser.add_argument("--grid", type=int, choices=P0B_GRIDS, required=True)
    parser.add_argument("--training-seed", type=int, choices=P0B_TRAINING_SEEDS, required=True)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ledger", default=str(REPO_ROOT / LEDGER_FILENAME))
    parser.add_argument("--mode", choices=("formal", "debug"), default="formal")
    parser.add_argument("--debug-root")
    parser.add_argument("--debug-epochs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def resolved_micro_batch(args: argparse.Namespace) -> tuple[int, int]:
    if args.mode == "formal":
        if args.micro_batch not in (None, FORMAL_MICRO_BATCH):
            raise ValueError("formal P0-B micro-batch is frozen at 128")
        return FORMAL_MICRO_BATCH, FORMAL_ACCUM_STEPS
    micro_batch = FORMAL_MICRO_BATCH if args.micro_batch is None else args.micro_batch
    if micro_batch <= 0 or FORMAL_CONFIG.effective_batch % micro_batch:
        raise ValueError("micro-batch must be a positive divisor of frozen effective batch 128")
    return micro_batch, FORMAL_CONFIG.effective_batch // micro_batch


def _validate_cli(args: argparse.Namespace) -> None:
    resolved_micro_batch(args)
    if args.dry_run and args.execute:
        raise ValueError("--dry-run and --execute are mutually exclusive")
    if not args.dry_run and not args.execute:
        raise ValueError("refusing to run without explicit --dry-run or --execute")
    if args.mode == "formal" and args.debug_root is not None:
        raise ValueError("formal mode does not accept debug output paths")
    if args.mode == "debug" and not args.debug_root:
        raise ValueError("debug mode requires a separate --debug-root")
    if args.mode == "formal" and args.debug_epochs != 2:
        raise ValueError("formal P0-B configuration has no CLI epoch override")
    if args.mode == "debug" and args.debug_epochs <= 0:
        raise ValueError("debug epochs must be positive")


def run_one_cell(args: argparse.Namespace) -> str:
    _validate_cli(args)
    micro_batch, accum_steps = resolved_micro_batch(args)
    source_sha256 = verify_formal_config()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise AssertionError("P0-B source SHA-256 mapping changed unexpectedly")
    ledger_sha256 = verify_ledger(args.ledger)
    resolution = resolve_p0b_paths(args.exp_id, args.grid, args.training_seed)
    if args.dry_run:
        return (
            f"DRY_RUN_OK {args.exp_id} grid={args.grid} seed={args.training_seed} "
            f"micro_batch={micro_batch} accum_steps={accum_steps} ledger={ledger_sha256}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("formal P0-B execution requires CUDA; no CPU fallback is permitted")
    from mamba_scan_study.experiments.run_stage1_seed0 import set_seed

    device = torch.device("cuda")
    set_seed(args.training_seed)
    model = construct_requested_model(resolution, device)
    runtime_config = FORMAL_CONFIG if args.mode == "formal" else replace(FORMAL_CONFIG, epochs=args.debug_epochs, num_workers=0)
    metadata = build_metadata(
        resolution,
        model,
        ledger_sha256,
        runtime_config=runtime_config,
        micro_batch=micro_batch,
        accum_steps=accum_steps,
    )
    run_directory = _run_directory(args.exp_id, args.grid, args.training_seed, args.mode, args.debug_root)
    completed = validate_completed_run(
        run_directory / "final_checkpoint.pt",
        run_directory / "metadata.json",
        run_directory / "completed.json",
        model,
        metadata,
        runtime_config,
    )
    if completed:
        return f"COMPLETED_SKIP {args.exp_id} grid={args.grid} seed={args.training_seed}"

    from mamba_scan_study.experiments.run_stage1_seed0 import (
        evaluate,
        lr_scale,
        set_epoch_lr,
        train_epoch,
    )

    set_seed(args.training_seed)
    loaders = build_p0b_loaders(
        args.data_root,
        micro_batch,
        args.training_seed,
        num_workers=runtime_config.num_workers,
    )
    set_seed(args.training_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime_config.base_lr, weight_decay=runtime_config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=runtime_config.amp)
    validation_history: list[dict[str, float]] = []
    for epoch in range(1, runtime_config.epochs + 1):
        learning_rate = set_epoch_lr(optimizer, runtime_config.base_lr, lr_scale(epoch, runtime_config.epochs, runtime_config.warmup_epochs))
        train_metrics = train_epoch(model, loaders.train, optimizer, scaler, runtime_config, accum_steps, device)
        validation_metrics, _ = evaluate(model, loaders.validation, runtime_config, device, collect=False)
        validation_history.append(
            {
                "epoch": epoch,
                "learning_rate": float(learning_rate),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["acc"]),
                "validation_loss": float(validation_metrics["loss"]),
                "validation_accuracy": float(validation_metrics["acc"]),
            }
        )
    metadata = build_metadata(
        resolution,
        model,
        ledger_sha256,
        validation_history=validation_history,
        runtime_config=runtime_config,
        micro_batch=micro_batch,
        accum_steps=accum_steps,
    )
    write_completed_run(run_directory, model, metadata)
    return f"COMPLETED {args.exp_id} grid={args.grid} seed={args.training_seed}"


def main() -> None:
    print(run_one_cell(parse_args()), flush=True)


if __name__ == "__main__":
    main()
