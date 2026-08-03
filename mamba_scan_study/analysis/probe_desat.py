"""Run the P0-B desaturation probe through the frozen runner.

The d_model=256 cells are read from ``outputs_main``.  The d_model=128 and
d_model=64 cells are delegated to ``run_p0b_feasibility.py`` with a separate
run root.  Only epoch 80--100 ``train_accuracy`` is read and reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
import subprocess
import sys
from typing import Any, Mapping, Sequence


_SCRIPT_PATH = Path(__file__).resolve()
_PACKAGE_PARENT = _SCRIPT_PATH.parents[1]
# The cloud deployment lives under mamba_scan_study/analysis; retain local
# analysis/ compatibility while resolving the cloud layout to parents[2].
REPO_ROOT = (
    _SCRIPT_PATH.parents[2]
    if _PACKAGE_PARENT.name == "mamba_scan_study"
    else _PACKAGE_PARENT
)
DEFAULT_MAIN_ROOT = Path("/root/autodl-tmp/outputs_main")
DEFAULT_PROBE_ROOT = Path("/root/autodl-tmp/outputs_probe_desat")
DEFAULT_DATA_ROOTS = {
    "cifar10": Path("/root/autodl-tmp/datasets"),
    "organamnist": Path("/root/autodl-tmp/datasets_new"),
    "organcmnist": Path("/root/autodl-tmp/datasets_new"),
    "organsmnist": Path("/root/autodl-tmp/datasets_new"),
    "eurosat": Path("/root/autodl-tmp/datasets_new"),
}

DATASETS = ("cifar10", "organamnist", "organcmnist", "organsmnist", "eurosat")
EXP_ID = "GEO_SG1"
GRID = 32
RELIANCE = "R_high"
TRAINING_SEED = 0
PROBE_D_MODELS = (512, 384, 128, 64)
EXISTING_D_MODEL = 256
TAIL_START = 80
TAIL_END = 100
FORBIDDEN_OUTPUT_ROOTS = (
    Path("/root/autodl-tmp/outputs_main"),
    Path("/root/autodl-tmp/outputs_aug16"),
    Path("/root/autodl-tmp/outputs_p0b_backup"),
)
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "protocol",
        "dataset",
        "augmentation",
        "backbone",
        "exp_id",
        "reliance",
        "grid",
        "training_seed",
        "parameter_count",
        "architecture_signature",
        "training_config",
        "validation_history",
    }
)


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _metadata_matches(
    metadata: Mapping[str, Any],
    *,
    dataset: str,
    d_model: int,
    augmentation: str,
) -> bool:
    training_config = metadata.get("training_config")
    if not isinstance(training_config, Mapping):
        return False
    return (
        metadata.get("dataset") == dataset
        and metadata.get("backbone") == "mamba"
        and metadata.get("augmentation") == augmentation
        and metadata.get("exp_id") == EXP_ID
        and metadata.get("reliance") == RELIANCE
        and metadata.get("grid") == GRID
        and metadata.get("training_seed") == TRAINING_SEED
        and training_config.get("d_model") == d_model
    )


def _read_train_tail(metadata_path: Path) -> float:
    """Read only train_accuracy from the stored epoch history."""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read metadata: {metadata_path}: {error}") from error
    metadata = _as_mapping(payload, str(metadata_path))
    history = metadata.get("validation_history")
    if not isinstance(history, list):
        raise ValueError(f"{metadata_path}: epoch history is missing")
    values: list[float] = []
    for row in history:
        row_mapping = _as_mapping(row, f"{metadata_path} epoch row")
        if row_mapping.get("epoch") in range(TAIL_START, TAIL_END + 1):
            value = row_mapping.get("train_accuracy")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{metadata_path}: train_accuracy is not numeric")
            values.append(float(value))
    if len(values) != TAIL_END - TAIL_START + 1:
        raise ValueError(f"{metadata_path}: train-accuracy tail is incomplete")
    return fmean(values)


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _complete_target(run_directory: Path, *, dataset: str, d_model: int, augmentation: str) -> bool:
    metadata_path = run_directory / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = _load_json(metadata_path)
    if metadata is None:
        return False
    training_config = metadata.get("training_config")
    history = metadata.get("validation_history")
    history_complete = (
        isinstance(history, list)
        and len(history) == 100
        and all(
            isinstance(row, Mapping) and row.get("epoch") == epoch
            for epoch, row in enumerate(history, start=1)
        )
    )
    return (
        REQUIRED_METADATA_FIELDS <= set(metadata)
        and metadata.get("protocol") == "P0B"
        and metadata.get("dataset") == dataset
        and metadata.get("augmentation") == augmentation
        and metadata.get("backbone") == "mamba"
        and metadata.get("exp_id") == EXP_ID
        and metadata.get("reliance") == RELIANCE
        and metadata.get("grid") == GRID
        and metadata.get("training_seed") == TRAINING_SEED
        and isinstance(training_config, Mapping)
        and training_config.get("d_model") == d_model
        and history_complete
    )


def _find_existing_d256(
    main_root: Path,
    *,
    dataset: str,
    augmentation: str,
) -> tuple[Path, float]:
    if not main_root.is_dir():
        raise FileNotFoundError(f"main output root does not exist: {main_root}")
    matches: list[Path] = []
    for metadata_path in sorted(main_root.rglob("metadata.json")):
        metadata = _load_json(metadata_path)
        if metadata is not None and _metadata_matches(
            metadata,
            dataset=dataset,
            d_model=EXISTING_D_MODEL,
            augmentation=augmentation,
        ):
            matches.append(metadata_path)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one d_model=256 metadata file for {dataset}, found {len(matches)}"
        )
    return matches[0], _read_train_tail(matches[0])


def _resolve_repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _assert_probe_root(run_root: Path) -> None:
    resolved = run_root.resolve()
    for forbidden in FORBIDDEN_OUTPUT_ROOTS:
        forbidden_resolved = forbidden.resolve()
        if resolved == forbidden_resolved or forbidden_resolved in resolved.parents:
            raise ValueError(f"probe run root overlaps protected output namespace: {run_root}")


def _run_directory(run_root: Path, dataset: str, d_model: int, augmentation: str) -> Path:
    name = (
        f"p0b_{dataset}_{augmentation}_mamba_{EXP_ID}_"
        f"{RELIANCE}_seed{TRAINING_SEED}_d{d_model}"
    )
    return run_root / name


def _run_probe_cell(
    *,
    data_root: Path,
    run_root: Path,
    dataset: str,
    d_model: int,
    augmentation: str,
) -> tuple[Path, float]:
    run_directory = _run_directory(run_root, dataset, d_model, augmentation)
    if not _complete_target(
        run_directory,
        dataset=dataset,
        d_model=d_model,
        augmentation=augmentation,
    ):
        command = [
            sys.executable,
            "-m",
            "mamba_scan_study.experiments.run_p0b_feasibility",
            "--exp-id",
            EXP_ID,
            "--grid",
            str(GRID),
            "--training-seed",
            str(TRAINING_SEED),
            "--dataset",
            dataset,
            "--augmentation",
            augmentation,
            "--backbone",
            "mamba",
            "--d-model",
            str(d_model),
            "--data-root",
            str(data_root),
            "--no-download",
            "--run-root",
            str(run_root),
            "--execute",
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not _complete_target(
        run_directory,
        dataset=dataset,
        d_model=d_model,
        augmentation=augmentation,
    ):
        raise RuntimeError(f"runner did not produce complete metadata: {run_directory}")
    return run_directory / "metadata.json", _read_train_tail(run_directory / "metadata.json")


def _run(args: argparse.Namespace) -> list[tuple[str, int, Path, float]]:
    args.data_root = (
        _resolve_repo_relative(args.data_root) if args.data_root is not None else None
    )
    args.main_root = _resolve_repo_relative(args.main_root)
    args.run_root = _resolve_repo_relative(args.run_root)
    _assert_probe_root(args.run_root)
    results: list[tuple[str, int, Path, float]] = []
    for dataset in args.datasets:
        metadata_path, tail_mean = _find_existing_d256(
            args.main_root,
            dataset=dataset,
            augmentation=args.augmentation,
        )
        results.append((dataset, EXISTING_D_MODEL, metadata_path, tail_mean))
        for d_model in args.d_models:
            data_root = args.data_root or DEFAULT_DATA_ROOTS[dataset]
            metadata_path, tail_mean = _run_probe_cell(
                data_root=data_root,
                run_root=args.run_root,
                dataset=dataset,
                d_model=d_model,
                augmentation=args.augmentation,
            )
            results.append((dataset, d_model, metadata_path, tail_mean))
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="optional local override applied to every dataset",
    )
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_PROBE_ROOT)
    parser.add_argument("--augmentation", choices=("main_uniform",), default="main_uniform")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--d-models", nargs="+", type=int, choices=PROBE_D_MODELS, default=list(PROBE_D_MODELS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = _run(args)
    print("dataset\td_model\ttail_train_accuracy_mean\tsource")
    for dataset, d_model, source, tail_mean in rows:
        print(f"{dataset}\t{d_model}\t{tail_mean:.6f}\t{source}")


if __name__ == "__main__":
    main()
