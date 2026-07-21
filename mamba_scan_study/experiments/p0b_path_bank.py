"""Resolve the immutable P0-B path banks into absolute row-major orders.

This module deliberately contains no path-generation logic.  It accepts only
source files whose bytes match the four P0-B freeze hashes, then reads their
already-frozen arrays.  `order[t] = cell` always refers to a row-major cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_SHA256 = {
    "lmto": "93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7",
    "random": "2f7b8a6fd3cfbbae9897b4ef4dc9dcfd1bf7744619d5818ceaca7604d565aee3",
    "validation_split": "e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09",
    "config": "790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e",
}
P0B_EXP_IDS = (
    "GEO_SG1",
    "GEO_SG2",
    "GEO_SG3",
    "GEO_SG4",
    "GEO_DIV",
    "RND_S1",
    "RND_S2",
    "RND_S3",
    "RND_D1",
    "RND_D2",
    "RND_D3",
    "LOC_S",
    "LOC_D",
)
P0B_GRIDS = (8, 32)
P0B_TRAINING_SEEDS = (0, 1, 2, 3)


@dataclass(frozen=True)
class P0BSourcePaths:
    """Locations are configurable, but only byte-identical frozen files pass."""

    lmto: Path
    random: Path
    validation_split: Path
    config: Path


@dataclass(frozen=True)
class P0BPathResolution:
    channel_orders: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    channel_path_ids: tuple[str, str, str, str]
    channel_order_sha256: tuple[str, str, str, str]
    channel_inverse_order_sha256: tuple[str, str, str, str]
    source_sha256: Mapping[str, str]
    latin_square_rotation: int | None
    exp_id: str
    grid: int
    training_seed: int
    path_family: str
    single_or_diverse: str


@dataclass(frozen=True)
class _ValidatedPath:
    path_id: str
    order: np.ndarray
    order_sha256: str
    inverse_order_sha256: str


def default_source_paths() -> P0BSourcePaths:
    return P0BSourcePaths(
        lmto=REPO_ROOT / "P0B_L_PATH_BANK_FROZEN.json",
        random=REPO_ROOT / "P0B_R_PATH_BANK_FROZEN.json",
        validation_split=REPO_ROOT / "P0B_CIFAR10_VAL_SPLIT_FROZEN.json",
        config=REPO_ROOT / "docs" / "P0B_CONFIG_TABLE.md",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _int64_c_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def verify_source_hashes(source_paths: P0BSourcePaths) -> dict[str, str]:
    """Hash all four sources before parsing either frozen path bank."""
    observed: dict[str, str] = {}
    for source_name, source_path in (
        ("lmto", source_paths.lmto),
        ("random", source_paths.random),
        ("validation_split", source_paths.validation_split),
        ("config", source_paths.config),
    ):
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"missing P0-B frozen source: {source_path}")
        digest = _sha256_file(source_path)
        expected = EXPECTED_SOURCE_SHA256[source_name]
        if digest != expected:
            raise ValueError(
                f"P0-B {source_name} SHA-256 mismatch: expected {expected}, got {digest}"
            )
        observed[source_name] = digest
    return observed


def _grid_entry(payload: dict, grid: int, source_name: str) -> dict:
    matches = [entry for entry in payload.get("grids", []) if entry.get("n") == grid]
    if len(matches) != 1:
        raise ValueError(f"{source_name} must contain exactly one grid n={grid}")
    entry = matches[0]
    expected_tokens = grid * grid
    if "N" in entry and entry["N"] != expected_tokens:
        raise ValueError(f"{source_name} n={grid} has inconsistent N={entry['N']}")
    return entry


def _validate_order(
    *,
    path_id: str,
    order_values: object,
    grid: int,
    declared_order_sha256: str,
    declared_inverse_sha256: str,
) -> _ValidatedPath:
    order = np.ascontiguousarray(np.asarray(order_values, dtype=np.int64))
    token_count = grid * grid
    if order.ndim != 1 or len(order) != token_count:
        raise ValueError(f"{path_id}: expected order length {token_count}, got shape {order.shape}")
    if np.any(order < 0) or np.any(order >= token_count):
        raise ValueError(f"{path_id}: order values must be in [0, {token_count - 1}]")
    if len(np.unique(order)) != token_count:
        raise ValueError(f"{path_id}: order is not a permutation")
    order_sha256 = _int64_c_sha256(order)
    if order_sha256 != declared_order_sha256:
        raise ValueError(f"{path_id}: order_sha256 does not match frozen order")
    inverse = np.empty(token_count, dtype=np.int64)
    inverse[order] = np.arange(token_count, dtype=np.int64)
    if not np.array_equal(inverse[order], np.arange(token_count, dtype=np.int64)):
        raise ValueError(f"{path_id}: inverse[order] != arange(N)")
    inverse_sha256 = _int64_c_sha256(inverse)
    if inverse_sha256 != declared_inverse_sha256:
        raise ValueError(f"{path_id}: inverse_order_sha256 does not match frozen inverse")
    return _ValidatedPath(path_id, order, order_sha256, inverse_sha256)


def _geometric_path(path_id: str, grid: int) -> _ValidatedPath:
    token_count = grid * grid
    if path_id == "G1":
        order = np.arange(token_count, dtype=np.int64)
    elif path_id == "G2":
        order = np.arange(token_count - 1, -1, -1, dtype=np.int64)
    elif path_id == "G3":
        order = np.asarray(
            [row * grid + column for column in range(grid) for row in range(grid)],
            dtype=np.int64,
        )
    elif path_id == "G4":
        order = _geometric_path("G3", grid).order[::-1].copy()
    else:
        raise ValueError(f"unknown geometric path {path_id!r}")
    inverse = np.empty(token_count, dtype=np.int64)
    inverse[order] = np.arange(token_count, dtype=np.int64)
    return _ValidatedPath(
        path_id,
        order,
        _int64_c_sha256(order),
        _int64_c_sha256(inverse),
    )


def _lmto_paths(payload: dict, grid: int) -> dict[str, _ValidatedPath]:
    entry = _grid_entry(payload, grid, "LMTO path bank")
    paths = entry.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("LMTO path bank must contain a paths mapping")
    result: dict[str, _ValidatedPath] = {}
    for path_id in ("L1", "L2", "L3", "L4"):
        record = paths.get(path_id)
        if not isinstance(record, dict) or record.get("label") != path_id:
            raise ValueError(f"LMTO path bank has inconsistent path_id {path_id}")
        result[path_id] = _validate_order(
            path_id=path_id,
            order_values=record.get("order"),
            grid=grid,
            declared_order_sha256=record.get("order_sha256"),
            declared_inverse_sha256=record.get("inverse_order_sha256"),
        )
    return result


def _random_paths(payload: dict, grid: int) -> dict[str, _ValidatedPath]:
    entry = _grid_entry(payload, grid, "random path bank")
    sets = entry.get("sets")
    if not isinstance(sets, dict):
        raise ValueError("random path bank must contain a sets mapping")
    result: dict[str, _ValidatedPath] = {}
    for set_number in (1, 2, 3):
        set_id = f"S{set_number}"
        set_record = sets.get(set_id)
        if not isinstance(set_record, dict) or not isinstance(set_record.get("paths"), dict):
            raise ValueError(f"random path bank is missing {set_id}")
        for path_number in (1, 2, 3, 4):
            path_id = f"R{set_number}_{path_number}"
            record = set_record["paths"].get(path_id)
            expected_seed = 17071 + 1000 * set_number + path_number
            if not isinstance(record, dict):
                raise ValueError(f"random path bank is missing {path_id}")
            if record.get("path_id") != path_id:
                raise ValueError(f"random path bank path_id mismatch for {path_id}")
            if record.get("set_id") != set_id:
                raise ValueError(f"random path bank set_id mismatch for {path_id}")
            if record.get("seed") != expected_seed:
                raise ValueError(f"random path bank seed mismatch for {path_id}")
            result[path_id] = _validate_order(
                path_id=path_id,
                order_values=record.get("order"),
                grid=grid,
                declared_order_sha256=record.get("order_sha256"),
                declared_inverse_sha256=record.get("inverse_order_sha256"),
            )
    return result


def _latin_square(path_ids: tuple[str, str, str, str], training_seed: int) -> tuple[str, str, str, str]:
    return path_ids[training_seed:] + path_ids[:training_seed]


def _selection(exp_id: str, training_seed: int) -> tuple[tuple[str, str, str, str], str, str, int | None]:
    if exp_id.startswith("GEO_SG"):
        path_id = f"G{exp_id[-1]}"
        return (path_id,) * 4, "GEO", "single", None
    if exp_id == "GEO_DIV":
        return _latin_square(("G1", "G2", "G3", "G4"), training_seed), "GEO", "diverse", training_seed
    if exp_id.startswith("RND_S"):
        set_number = exp_id[-1]
        path_id = f"R{set_number}_{training_seed + 1}"
        return (path_id,) * 4, "RND", "single", None
    if exp_id.startswith("RND_D"):
        set_number = exp_id[-1]
        path_ids = tuple(f"R{set_number}_{index}" for index in range(1, 5))
        return _latin_square(path_ids, training_seed), "RND", "diverse", training_seed
    if exp_id == "LOC_S":
        path_id = f"L{training_seed + 1}"
        return (path_id,) * 4, "LMTO", "single", None
    if exp_id == "LOC_D":
        return _latin_square(("L1", "L2", "L3", "L4"), training_seed), "LMTO", "diverse", training_seed
    raise ValueError(f"unknown P0-B exp_id={exp_id!r}")


def resolve_p0b_paths(
    exp_id: str,
    grid: int,
    training_seed: int,
    source_paths: P0BSourcePaths | None = None,
) -> P0BPathResolution:
    """Resolve one frozen P0-B design cell without generating any path."""
    if exp_id not in P0B_EXP_IDS:
        raise ValueError(f"unknown P0-B exp_id={exp_id!r}")
    if grid not in P0B_GRIDS:
        raise ValueError(f"P0-B grid must be one of {P0B_GRIDS}, got {grid}")
    if training_seed not in P0B_TRAINING_SEEDS:
        raise ValueError(f"P0-B training_seed must be one of {P0B_TRAINING_SEEDS}")
    source_paths = default_source_paths() if source_paths is None else source_paths
    source_sha256 = verify_source_hashes(source_paths)

    # Only after all byte-level source gates pass do we parse frozen path arrays.
    lmto_payload = json.loads(Path(source_paths.lmto).read_text(encoding="utf-8"))
    random_payload = json.loads(Path(source_paths.random).read_text(encoding="utf-8"))
    validated_paths: dict[str, _ValidatedPath] = {
        **{path_id: _geometric_path(path_id, grid) for path_id in ("G1", "G2", "G3", "G4")},
        **_random_paths(random_payload, grid),
        **_lmto_paths(lmto_payload, grid),
    }
    path_ids, path_family, single_or_diverse, rotation = _selection(exp_id, training_seed)
    paths = tuple(validated_paths[path_id] for path_id in path_ids)
    channel_orders = tuple(
        torch.from_numpy(path.order.copy()).to(dtype=torch.long, device="cpu") for path in paths
    )
    if len(channel_orders) != 4:
        raise AssertionError("P0-B resolver must return exactly four channel orders")
    return P0BPathResolution(
        channel_orders=channel_orders,
        channel_path_ids=path_ids,
        channel_order_sha256=tuple(path.order_sha256 for path in paths),
        channel_inverse_order_sha256=tuple(path.inverse_order_sha256 for path in paths),
        source_sha256=source_sha256,
        latin_square_rotation=rotation,
        exp_id=exp_id,
        grid=grid,
        training_seed=training_seed,
        path_family=path_family,
        single_or_diverse=single_or_diverse,
    )


def iter_p0b_design_cells() -> Iterator[tuple[str, int, int]]:
    for exp_id in P0B_EXP_IDS:
        for grid in P0B_GRIDS:
            for training_seed in P0B_TRAINING_SEEDS:
                yield exp_id, grid, training_seed
