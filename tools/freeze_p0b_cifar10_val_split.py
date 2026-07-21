"""Freeze or verify the P0-B CIFAR-10 train/validation split.

The frozen index arrays, not this generation routine, are the runtime source
of truth for future P0-B runs.  This tool never constructs CIFAR10(train=False).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10


DATA_ROOT = Path("data")
FROZEN_JSON = Path("P0B_CIFAR10_VAL_SPLIT_FROZEN.json")
REPORT = Path("REPORT_B4A_CIFAR10_VAL_SPLIT_FREEZE.md")
CONFIG = Path("docs/P0B_CONFIG_TABLE.md")
EXPECTED_CONFIG_SHA256 = (
    "790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e"
)
SPLIT_SEED = 20260720
EXPECTED_JSON_SHA256 = "ab1936e87df3826acb246602bbadecab9ae7e773219acf6a95cb84c1cc21c644"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def int64_c_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return sha256_bytes(values.tobytes(order="C"))


def source_files_sha256(data_root: Path) -> dict[str, str]:
    batch_dir = data_root / "cifar-10-batches-py"
    names = [*(f"data_batch_{index}" for index in range(1, 6)), "batches.meta"]
    return {
        name: sha256_file(batch_dir / name)
        for name in names
        if (batch_dir / name).is_file()
    }


def load_train_dataset() -> CIFAR10:
    # Intentionally the only CIFAR10 constructor in this file: train=False is
    # never constructed, read, or evaluated by B4A.
    return CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    )


def make_indices(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # One generator is used continuously in class order 0 through 9.
    rng = np.random.Generator(np.random.PCG64(20260720))
    train_members: list[np.ndarray] = []
    validation_members: list[np.ndarray] = []
    for class_id in range(10):
        class_indices = np.flatnonzero(targets == class_id).astype(np.int64)
        assert len(class_indices) == 5000
        rng.shuffle(class_indices)
        validation_members.append(class_indices[:500])
        train_members.append(class_indices[500:])
    train_indices = np.sort(np.concatenate(train_members)).astype(np.int64)
    validation_indices = np.sort(np.concatenate(validation_members)).astype(np.int64)
    return train_indices, validation_indices


def class_counts(targets: np.ndarray, indices: np.ndarray) -> list[int]:
    return [int(np.sum(targets[indices] == class_id)) for class_id in range(10)]


def validation_results(
    payload: dict[str, object], targets: np.ndarray, images: np.ndarray, frozen_json_sha256: str
) -> list[tuple[str, bool]]:
    train_indices = np.asarray(payload["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(payload["validation_indices"], dtype=np.int64)
    replay_train, replay_validation = make_indices(targets)
    recovered = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version": "1.0",
        "status": "FROZEN_FOR_P0B_FEASIBILITY",
        "dataset": "CIFAR10",
        "dataset_train_flag": True,
        "dataset_length": 50000,
        "split_seed": SPLIT_SEED,
        "bit_generator": "numpy.random.PCG64",
        "class_iteration_order": list(range(10)),
        "validation_per_class": 500,
        "train_per_class": 4500,
    }
    return [
        ("01 train length is 45000", len(train_indices) == 45000),
        ("02 validation length is 5000", len(validation_indices) == 5000),
        (
            "03 frozen array dtype semantics are int64",
            all(
                np.asarray(payload[name], dtype=np.int64).dtype == np.int64
                for name in ("train_indices", "validation_indices")
            ),
        ),
        (
            "04 all indices are in [0, 49999]",
            bool(
                np.all((train_indices >= 0) & (train_indices < 50000))
                and np.all((validation_indices >= 0) & (validation_indices < 50000))
            ),
        ),
        ("05 train has no duplicates", len(np.unique(train_indices)) == len(train_indices)),
        (
            "06 validation has no duplicates",
            len(np.unique(validation_indices)) == len(validation_indices),
        ),
        (
            "07 train and validation are disjoint",
            len(np.intersect1d(train_indices, validation_indices)) == 0,
        ),
        (
            "08 their union is exactly 0..49999",
            np.array_equal(
                np.sort(np.concatenate((train_indices, validation_indices))),
                np.arange(50000, dtype=np.int64),
            ),
        ),
        ("09 each full class count is 5000", class_counts(targets, np.arange(50000)) == [5000] * 10),
        ("10 each train class count is 4500", class_counts(targets, train_indices) == [4500] * 10),
        (
            "11 each validation class count is 500",
            class_counts(targets, validation_indices) == [500] * 10,
        ),
        (
            "12 final arrays are strictly increasing",
            bool(np.all(np.diff(train_indices) > 0) and np.all(np.diff(validation_indices) > 0)),
        ),
        (
            "13 independent PCG64 replay is elementwise identical",
            np.array_equal(train_indices, replay_train)
            and np.array_equal(validation_indices, replay_validation),
        ),
        (
            "14 embedded index hashes match arrays",
            payload["train_indices_sha256"] == int64_c_sha256(train_indices)
            and payload["validation_indices_sha256"] == int64_c_sha256(validation_indices),
        ),
        (
            "15 embedded image/target hashes match dataset",
            payload["images_uint8_c_sha256"] == sha256_bytes(images.tobytes(order="C"))
            and payload["targets_int64_c_sha256"] == sha256_bytes(targets.tobytes(order="C")),
        ),
        (
            "16 persisted JSON reloads and required fields match",
            recovered == payload
            and all(payload.get(key) == value for key, value in expected_fields.items()),
        ),
        (
            "17 JSON SHA is consistent with report and preregistration",
            frozen_json_sha256
            in Path("P0B_PREREG_FREEZE_CIFAR10_VAL_SPLIT.md").read_text(encoding="utf-8"),
        ),
    ]


def write_report(
    payload: dict[str, object],
    results: list[tuple[str, bool]],
    frozen_json_sha256: str,
    wall_seconds: float,
    cpu_seconds: float,
) -> None:
    train_indices = np.asarray(payload["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(payload["validation_indices"], dtype=np.int64)
    count_rows = "\n".join(
        f"| {class_id} | {payload['class_counts_full'][class_id]} | "
        f"{payload['class_counts_train'][class_id]} | "
        f"{payload['class_counts_validation'][class_id]} |"
        for class_id in range(10)
    )
    result_rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in results
    )
    REPORT.write_text(
        "# B4A CIFAR-10 Validation Split Freeze\n\n"
        "Status: `FROZEN_FOR_P0B_FEASIBILITY`\n\n"
        "## Execution\n\n"
        "- Freeze command: `conda run -n mair python -B tools/freeze_p0b_cifar10_val_split.py --freeze`\n"
        "- Verification command: `conda run -n mair python -B tools/freeze_p0b_cifar10_val_split.py --verify`\n"
        f"- Python: `{sys.executable}`\n"
        f"- NumPy: `{payload['numpy_version']}`; torchvision: `{payload['torchvision_version']}`\n"
        f"- Verification wall time: `{wall_seconds:.6f} s`; CPU time: `{cpu_seconds:.6f} s`\n"
        f"- Data root: `{payload['data_root']}`\n\n"
        "## Frozen Algorithm\n\n"
        "The official `CIFAR10(train=True)` population of 50,000 examples is split with "
        "`rng = np.random.Generator(np.random.PCG64(20260720))`. The one generator is "
        "used continuously for class IDs 0 through 9. Each class supplies the first 500 "
        "shuffled members to validation and the remaining 4,500 to train. Both final arrays "
        "are sorted `int64` arrays. The embedded arrays are the sole future runtime source; "
        "they must not be regenerated by P0-B performance code.\n\n"
        "| Class | Full | Train | Validation |\n| --- | ---: | ---: | ---: |\n"
        f"{count_rows}\n\n"
        "## Integrity and Freeze Hashes\n\n"
        f"- images_uint8_c_sha256: `{payload['images_uint8_c_sha256']}`\n"
        f"- targets_int64_c_sha256: `{payload['targets_int64_c_sha256']}`\n"
        f"- train_indices_sha256: `{payload['train_indices_sha256']}`\n"
        f"- validation_indices_sha256: `{payload['validation_indices_sha256']}`\n"
        f"- frozen JSON SHA-256: `{frozen_json_sha256}`\n"
        f"- train: length `{len(train_indices)}`, min `{train_indices.min()}`, max `{train_indices.max()}`\n"
        f"- validation: length `{len(validation_indices)}`, min `{validation_indices.min()}`, max `{validation_indices.max()}`\n"
        f"- docs/P0B_CONFIG_TABLE.md SHA-256: `{sha256_file(CONFIG)}`\n"
        "- Source file SHA-256 (official training files plus metadata):\n"
        + "\n".join(
            f"  - `{name}`: `{digest}`"
            for name, digest in payload["source_files_sha256"].items()
        )
        + "\n\n## Required Validations\n\n| Check | Result |\n| --- | --- |\n"
        + result_rows
        + "\n\n## Scope Boundary\n\n"
        "Only `CIFAR10(train=True)` was constructed. No `train=False` dataset was constructed, "
        "read, or evaluated. A torchvision download archive can physically contain test files; "
        "this does not mean B4A constructed or accessed a test dataset. No `outputs/` path was "
        "accessed, and no model, training, inference, or GPU task was run.\n"
    )


def make_payload(dataset: CIFAR10, train_indices: np.ndarray, validation_indices: np.ndarray) -> dict[str, object]:
    targets = np.ascontiguousarray(np.asarray(dataset.targets, dtype=np.int64))
    images = np.ascontiguousarray(dataset.data, dtype=np.uint8)
    return {
        "schema_version": "1.0",
        "status": "FROZEN_FOR_P0B_FEASIBILITY",
        "freeze_date": "2026-07-20",
        "freeze_scope": "P0-B feasibility pilot",
        "decision_record": "P0B_PREREG_FREEZE_CIFAR10_VAL_SPLIT.md",
        "dataset": "CIFAR10",
        "dataset_train_flag": True,
        "dataset_length": 50000,
        "split_seed": SPLIT_SEED,
        "bit_generator": "numpy.random.PCG64",
        "class_iteration_order": list(range(10)),
        "validation_per_class": 500,
        "train_per_class": 4500,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "train_indices_sha256": int64_c_sha256(train_indices),
        "validation_indices_sha256": int64_c_sha256(validation_indices),
        "targets_int64_c_sha256": sha256_bytes(targets.tobytes(order="C")),
        "images_uint8_c_sha256": sha256_bytes(images.tobytes(order="C")),
        "class_counts_full": class_counts(targets, np.arange(50000)),
        "class_counts_train": class_counts(targets, train_indices),
        "class_counts_validation": class_counts(targets, validation_indices),
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "data_root": str(DATA_ROOT.resolve()),
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "class_names": list(dataset.classes),
        "source_files_sha256": source_files_sha256(DATA_ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if sha256_file(CONFIG) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("docs/P0B_CONFIG_TABLE.md SHA-256 gate failed")
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    dataset = load_train_dataset()
    targets = np.ascontiguousarray(np.asarray(dataset.targets, dtype=np.int64))
    images = np.ascontiguousarray(dataset.data, dtype=np.uint8)
    if args.freeze:
        train_indices, validation_indices = make_indices(targets)
        payload = make_payload(dataset, train_indices, validation_indices)
        # This explicit B4A finalization replaces only the earlier incomplete
        # provenance payload; its deterministic index arrays must be identical.
        if FROZEN_JSON.exists() and sha256_file(FROZEN_JSON) != EXPECTED_JSON_SHA256:
            raise RuntimeError("refusing to overwrite an unexpected frozen split JSON")
        FROZEN_JSON.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        payload = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    frozen_json_sha256 = sha256_file(FROZEN_JSON)
    # The freeze invocation only creates the final artifact.  `--verify` is
    # subsequently run after the preregistration cites that final SHA.
    if args.freeze:
        print(frozen_json_sha256)
        return
    results = validation_results(payload, targets, images, frozen_json_sha256)
    if not all(passed for _, passed in results):
        raise RuntimeError("one or more B4A validations failed")
    write_report(
        payload,
        results,
        frozen_json_sha256,
        time.perf_counter() - start_wall,
        time.process_time() - start_cpu,
    )
    if frozen_json_sha256 not in REPORT.read_text(encoding="utf-8"):
        raise RuntimeError("report did not preserve the frozen JSON SHA-256")
    print(frozen_json_sha256)


if __name__ == "__main__":
    main()
