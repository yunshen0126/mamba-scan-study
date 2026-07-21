#!/usr/bin/env python3
"""CPU-only B0 audit for locality-matched path controls.

Formal source read before this implementation:
``C:/Users/DELL/Desktop/一些md/P0A_B0_FORMAL_DEFINITIONS.md``.

This script is deliberately isolated from all model, data-loader, checkpoint,
and output code.  It imports only the Python standard library and NumPy.
It does not define an AUC and it does not select a final L generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import platform
import sys
import time
from typing import Iterable

import numpy as np


# Frozen audit specification.  Do not expose these as runtime tuning options.
GRID_SIZES = (8, 32)
BLOCKS_PER_AXIS = (2, 4)
BLOCK_AUDIT_SEEDS = {
    (8, 2): 2026072001,
    (8, 4): 2026072002,
    (32, 2): 2026072003,
    (32, 4): 2026072004,
}
SEARCH_SEEDS = {
    (8, "G1"): 2026072101,
    (8, "G2"): 2026072102,
    (8, "G3"): 2026072103,
    (8, "G4"): 2026072104,
    (32, "G1"): 2026072201,
    (32, "G2"): 2026072202,
    (32, "G3"): 2026072203,
    (32, "G4"): 2026072204,
}
BLOCK_SAMPLES = 20_000
BURN_IN_PROPOSALS = 100_000
THINNING_PROPOSALS = 25_000
MAX_PROPOSALS = 1_000_000
VERIFY_EVERY_PROPOSALS = 10_000
THRESHOLDS = (0.01, 0.05, 0.10, 0.20)
REPORT_PATH = Path("REPORT_B0_locality_control_feasibility.md")

# The eight orientations are precisely main axis x starting corner.  The
# starting corner also determines the first inner-axis direction; there is no
# separate serpentine-direction random variable.
ORIENTATION_LABELS = (
    "row_TL",
    "row_TR",
    "row_BL",
    "row_BR",
    "col_TL",
    "col_TR",
    "col_BL",
    "col_BR",
)


@dataclass(frozen=True)
class Grid:
    n: int
    horizontal_u: np.ndarray
    horizontal_v: np.ndarray
    vertical_u: np.ndarray
    vertical_v: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    incident: tuple[tuple[int, ...], ...]

    @property
    def edge_count(self) -> int:
        return int(self.edge_u.size)


@dataclass(frozen=True)
class LocalityMetrics:
    mean: float
    p50: float
    p90: float
    p95: float
    maximum: int
    dx: float
    dy: float
    axis_bias: float


@dataclass(frozen=True)
class DiagnosticPoint:
    proposal: int
    accepted_total: int
    accepted_after_burn: int
    acceptance_rate: float
    nearest_g: str
    hamming: float
    kendall_tau: float
    edge_jaccard: float


@dataclass(frozen=True)
class Candidate:
    n: int
    source_g: str
    seed: int
    proposal: int
    accepted_total: int
    accepted_after_burn: int
    metrics: LocalityMetrics
    nearest_g: str
    hamming: float
    kendall_tau: float
    edge_jaccard: float
    cdir: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class ChainResult:
    n: int
    source_g: str
    seed: int
    proposal_counts: tuple[int, int, int]
    accepted_total: int
    accepted_after_burn: int
    candidate: Candidate | None
    diagnostics: tuple[DiagnosticPoint, ...]
    checks: int


class Fenwick:
    """Integer order-statistics tree over d_seq bins 1..N-1."""

    def __init__(self, counts: np.ndarray) -> None:
        if counts.ndim != 1 or counts.size < 2:
            raise AssertionError("Fenwick counts must contain bins 0..N-1")
        self.size = int(counts.size - 1)
        self.tree = counts.astype(np.int64, copy=True)
        self.tree[0] = 0
        for index in range(1, self.size + 1):
            parent = index + (index & -index)
            if parent <= self.size:
                self.tree[parent] += self.tree[index]

    def add(self, index: int, delta: int) -> None:
        if not 1 <= index <= self.size:
            raise AssertionError(f"invalid d_seq bin {index}")
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def total(self) -> int:
        index = self.size
        value = 0
        while index:
            value += int(self.tree[index])
            index -= index & -index
        return value

    def kth(self, k_one_based: int) -> int:
        """Return the smallest d whose cumulative count is at least k."""
        total = self.total()
        if not 1 <= k_one_based <= total:
            raise AssertionError(f"k={k_one_based} is outside 1..{total}")
        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        remaining = k_one_based
        while bit:
            candidate = index + bit
            if candidate <= self.size and int(self.tree[candidate]) < remaining:
                index = candidate
                remaining -= int(self.tree[candidate])
            bit >>= 1
        return index + 1


def make_grid(n: int) -> Grid:
    cells = np.arange(n * n, dtype=np.int64).reshape(n, n)
    horizontal_u = cells[:, :-1].ravel()
    horizontal_v = cells[:, 1:].ravel()
    vertical_u = cells[:-1, :].ravel()
    vertical_v = cells[1:, :].ravel()
    edge_u = np.concatenate((horizontal_u, vertical_u))
    edge_v = np.concatenate((horizontal_v, vertical_v))
    incident_lists: list[list[int]] = [[] for _ in range(n * n)]
    for edge_id, (u, v) in enumerate(zip(edge_u.tolist(), edge_v.tolist())):
        incident_lists[u].append(edge_id)
        incident_lists[v].append(edge_id)
    return Grid(
        n=n,
        horizontal_u=horizontal_u,
        horizontal_v=horizontal_v,
        vertical_u=vertical_u,
        vertical_v=vertical_v,
        edge_u=edge_u,
        edge_v=edge_v,
        incident=tuple(tuple(items) for items in incident_lists),
    )


def inverse_order(order: np.ndarray) -> np.ndarray:
    pi = np.empty(order.size, dtype=np.int64)
    pi[order] = np.arange(order.size, dtype=np.int64)
    return pi


def validate_order(order: np.ndarray) -> np.ndarray:
    n_cells = int(order.size)
    if order.ndim != 1:
        raise AssertionError("order must be one-dimensional")
    if not np.array_equal(np.sort(order), np.arange(n_cells, dtype=np.int64)):
        raise AssertionError("order is not a permutation of all cells")
    pi = inverse_order(order)
    if not np.array_equal(pi[order], np.arange(n_cells, dtype=np.int64)):
        raise AssertionError("pi[order] != arange(N)")
    if not np.array_equal(pi, np.argsort(order)):
        raise AssertionError("pi != argsort(order)")
    return pi


def named_g_orders(n: int) -> dict[str, np.ndarray]:
    n_cells = n * n
    g1 = np.arange(n_cells, dtype=np.int64)
    g2 = g1[::-1].copy()
    g3 = np.arange(n_cells, dtype=np.int64).reshape(n, n).T.ravel().copy()
    g4 = g3[::-1].copy()
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4}


def distances(pi: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.abs(pi[u] - pi[v]).astype(np.int64, copy=False)


def numpy_linear_percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="linear"))


def metrics_from_pi(pi: np.ndarray, grid: Grid) -> tuple[LocalityMetrics, np.ndarray, np.ndarray]:
    d_horizontal = distances(pi, grid.horizontal_u, grid.horizontal_v)
    d_vertical = distances(pi, grid.vertical_u, grid.vertical_v)
    d_all = np.concatenate((d_horizontal, d_vertical))
    dx = float(np.mean(d_horizontal))
    dy = float(np.mean(d_vertical))
    metrics = LocalityMetrics(
        mean=float(np.mean(d_all)),
        p50=numpy_linear_percentile(d_all, 50),
        p90=numpy_linear_percentile(d_all, 90),
        p95=numpy_linear_percentile(d_all, 95),
        maximum=int(np.max(d_all)),
        dx=dx,
        dy=dy,
        axis_bias=float(math.log(dx / dy)),
    )
    return metrics, d_all, np.bincount(d_all, minlength=grid.n * grid.n)


def linear_percentile_from_fenwick(tree: Fenwick, total: int, q: float) -> float:
    """Exactly mirror numpy.percentile(values, q, method='linear').

    With zero-based sorted positions, h=(M-1)*q/100.  The result is the
    linear interpolation of the order statistics at floor(h) and ceil(h).
    For an even M at q=50, this therefore averages the two middle values;
    it is not an integer-median shortcut.
    """
    h = (total - 1) * (q / 100.0)
    low = math.floor(h)
    high = math.ceil(h)
    lower_value = tree.kth(low + 1)
    upper_value = tree.kth(high + 1)
    return float(lower_value + (h - low) * (upper_value - lower_value))


def c5_bounds(n: int) -> dict[str, tuple[float, float]]:
    target = (n + 1) / 2.0
    return {
        "mean": (0.9 * target, 1.1 * target),
        "p50": (0.9 * target, 1.1 * target),
        "p90": (0.9 * n, 1.1 * n),
    }


def meets_c5(metrics: LocalityMetrics, n: int) -> bool:
    bounds = c5_bounds(n)
    return (
        bounds["mean"][0] <= metrics.mean <= bounds["mean"][1]
        and bounds["p50"][0] <= metrics.p50 <= bounds["p50"][1]
        and bounds["p90"][0] <= metrics.p90 <= bounds["p90"][1]
    )


def assert_g_regression(grids: dict[int, Grid]) -> dict[int, dict[str, LocalityMetrics]]:
    results: dict[int, dict[str, LocalityMetrics]] = {}
    for n, grid in grids.items():
        expected = ((n + 1) / 2.0, (n + 1) / 2.0, float(n))
        order_map = named_g_orders(n)
        metric_map: dict[str, LocalityMetrics] = {}
        d_all_map: dict[str, np.ndarray] = {}
        for label, order in order_map.items():
            pi = validate_order(order)
            metrics, d_all, _ = metrics_from_pi(pi, grid)
            metric_map[label] = metrics
            d_all_map[label] = d_all
            observed = (metrics.mean, metrics.p50, metrics.p90)
            if observed != expected:
                raise AssertionError(
                    f"{label}, n={n} regression {observed} != {expected}"
                )
        if not np.array_equal(d_all_map["G1"], d_all_map["G2"]):
            raise AssertionError(f"G1/G2 d_seq mismatch at n={n}")
        if not np.array_equal(np.sort(d_all_map["G1"]), np.sort(d_all_map["G3"])):
            raise AssertionError(f"G1/G3 aggregate d_seq distribution mismatch at n={n}")
        if not math.isclose(metric_map["G1"].axis_bias, -metric_map["G3"].axis_bias):
            raise AssertionError(f"G1/G3 AxisBias sign mismatch at n={n}")
        if not math.isclose(metric_map["G2"].axis_bias, metric_map["G1"].axis_bias):
            raise AssertionError(f"G1/G2 AxisBias mismatch at n={n}")
        if not math.isclose(metric_map["G4"].axis_bias, metric_map["G3"].axis_bias):
            raise AssertionError(f"G3/G4 AxisBias mismatch at n={n}")
        results[n] = metric_map
    return results


def local_orientation_orders(m: int) -> dict[str, np.ndarray]:
    if m < 2:
        raise AssertionError("each block must contain at least a 2x2 grid")
    orders: dict[str, np.ndarray] = {}
    for label in ORIENTATION_LABELS:
        axis, corner = label.split("_")
        top = corner[0] == "T"
        left = corner[1] == "L"
        cells: list[int] = []
        if axis == "row":
            rows: Iterable[int] = range(m) if top else range(m - 1, -1, -1)
            first_forward = left
            for step, row in enumerate(rows):
                forward = first_forward if step % 2 == 0 else not first_forward
                columns: Iterable[int] = range(m) if forward else range(m - 1, -1, -1)
                cells.extend(row * m + column for column in columns)
        elif axis == "col":
            columns = range(m) if left else range(m - 1, -1, -1)
            first_forward = top
            for step, column in enumerate(columns):
                forward = first_forward if step % 2 == 0 else not first_forward
                rows = range(m) if forward else range(m - 1, -1, -1)
                cells.extend(row * m + column for row in rows)
        else:
            raise AssertionError(f"unknown orientation axis {axis}")
        orders[label] = np.asarray(cells, dtype=np.int64)
    return orders


def assert_orientation_family(m: int) -> dict[str, np.ndarray]:
    local_orders = local_orientation_orders(m)
    expected_cells = np.arange(m * m, dtype=np.int64)
    signatures: set[tuple[int, ...]] = set()
    for label, order in local_orders.items():
        if len(order) != m * m:
            raise AssertionError(f"{label}: len(block_order) != (n/b)^2")
        if not np.array_equal(np.sort(order), expected_cells):
            raise AssertionError(f"{label}: block order has duplicates or omissions")
        rows, columns = divmod(order, m)
        manhattan = np.abs(np.diff(rows)) + np.abs(np.diff(columns))
        if not np.all(manhattan == 1):
            raise AssertionError(f"{label}: block order is not four-neighbor continuous")
        signature = tuple(order.tolist())
        if signature in signatures:
            raise AssertionError(f"{label}: duplicate orientation order")
        signatures.add(signature)
    if len(signatures) != 8:
        raise AssertionError(f"expected 8 unique orientations, got {len(signatures)}")
    return local_orders


def block_candidate_order(
    n: int,
    b: int,
    rng: np.random.Generator,
    local_orders: dict[str, np.ndarray],
) -> np.ndarray:
    if n % b:
        raise AssertionError(f"n={n} is not divisible by b={b}")
    m = n // b
    if any(len(order) != m * m for order in local_orders.values()):
        raise AssertionError("block order length is not (n/b)^2")
    parts: list[np.ndarray] = []
    block_ids = rng.permutation(b * b)
    orientation_ids = rng.integers(0, len(ORIENTATION_LABELS), size=b * b)
    for visit, block_id in enumerate(block_ids.tolist()):
        block_row, block_column = divmod(block_id, b)
        local = local_orders[ORIENTATION_LABELS[int(orientation_ids[visit])]]
        local_row, local_column = divmod(local, m)
        global_cells = (block_row * m + local_row) * n + (block_column * m + local_column)
        parts.append(global_cells)
    order = np.concatenate(parts)
    validate_order(order)
    return order


def block_audit(grids: dict[int, Grid]) -> dict[tuple[int, int], dict[str, object]]:
    results: dict[tuple[int, int], dict[str, object]] = {}
    for n in GRID_SIZES:
        grid = grids[n]
        for b in BLOCKS_PER_AXIS:
            if n % b:
                raise AssertionError(f"n={n}, b={b} is invalid")
            m = n // b
            local_orders = assert_orientation_family(m)
            rng = np.random.default_rng(BLOCK_AUDIT_SEEDS[(n, b)])
            values = np.empty((BLOCK_SAMPLES, 3), dtype=np.float64)
            best_index = -1
            best_score = math.inf
            best_metrics: LocalityMetrics | None = None
            for sample_index in range(BLOCK_SAMPLES):
                order = block_candidate_order(n, b, rng, local_orders)
                metrics, _, _ = metrics_from_pi(inverse_order(order), grid)
                values[sample_index] = (metrics.mean, metrics.p50, metrics.p90)
                target = np.array(((n + 1) / 2.0, (n + 1) / 2.0, float(n)))
                score = float(np.sum(((values[sample_index] - target) / target) ** 2))
                if score < best_score:
                    best_index = sample_index
                    best_score = score
                    best_metrics = metrics
            bounds = c5_bounds(n)
            pass_mask = (
                (values[:, 0] >= bounds["mean"][0])
                & (values[:, 0] <= bounds["mean"][1])
                & (values[:, 1] >= bounds["p50"][0])
                & (values[:, 1] <= bounds["p50"][1])
                & (values[:, 2] >= bounds["p90"][0])
                & (values[:, 2] <= bounds["p90"][1])
            )
            results[(n, b)] = {
                "m": m,
                "sample_count": BLOCK_SAMPLES,
                "seed": BLOCK_AUDIT_SEEDS[(n, b)],
                "ranges": {
                    name: tuple(float(x) for x in np.percentile(values[:, index], (0, 5, 50, 95, 100), method="linear"))
                    for index, name in enumerate(("mean", "p50", "p90"))
                },
                "c5_count": int(np.sum(pass_mask)),
                "p50_is_one_count": int(np.sum(values[:, 1] == 1.0)),
                "best_index": best_index,
                "best_score": best_score,
                "best_metrics": best_metrics,
            }
    return results


class IncrementalState:
    """A current order with incremental d_seq histogram, sum, and Fenwick tree."""

    def __init__(self, order: np.ndarray, grid: Grid) -> None:
        self.grid = grid
        self.order = order.copy()
        self.pi = validate_order(self.order)
        metrics, d_all, histogram = metrics_from_pi(self.pi, grid)
        self.histogram = histogram.astype(np.int64, copy=True)
        self.total_distance = int(np.sum(d_all))
        self.fenwick = Fenwick(self.histogram)
        self.verify("initialization", expected_metrics=metrics)

    def metrics(self) -> LocalityMetrics:
        total = self.grid.edge_count
        mean = self.total_distance / total
        p50 = linear_percentile_from_fenwick(self.fenwick, total, 50)
        p90 = linear_percentile_from_fenwick(self.fenwick, total, 90)
        p95 = linear_percentile_from_fenwick(self.fenwick, total, 95)
        # Axis-specific means are not maintained incrementally because C5 does
        # not use them.  Recompute them only for a saved candidate/report row.
        d_horizontal = distances(self.pi, self.grid.horizontal_u, self.grid.horizontal_v)
        d_vertical = distances(self.pi, self.grid.vertical_u, self.grid.vertical_v)
        dx = float(np.mean(d_horizontal))
        dy = float(np.mean(d_vertical))
        return LocalityMetrics(
            mean=float(mean),
            p50=p50,
            p90=p90,
            p95=p95,
            maximum=self.fenwick.kth(total),
            dx=dx,
            dy=dy,
            axis_bias=float(math.log(dx / dy)),
        )

    def c5(self) -> bool:
        total = self.grid.edge_count
        mean = self.total_distance / total
        bounds = c5_bounds(self.grid.n)
        if not bounds["mean"][0] <= mean <= bounds["mean"][1]:
            return False
        p50 = linear_percentile_from_fenwick(self.fenwick, total, 50)
        if not bounds["p50"][0] <= p50 <= bounds["p50"][1]:
            return False
        p90 = linear_percentile_from_fenwick(self.fenwick, total, 90)
        return bounds["p90"][0] <= p90 <= bounds["p90"][1]

    def _affected_edges(self, u: int, v: int) -> list[int]:
        edge_ids = list(self.grid.incident[u])
        for edge_id in self.grid.incident[v]:
            if edge_id not in edge_ids:
                edge_ids.append(edge_id)
        return edge_ids

    def apply_swap(self, position_a: int, position_b: int) -> None:
        """Apply a position swap and update the locality state exactly."""
        if position_a == position_b:
            raise AssertionError("proposal selected the same sequence position")
        u = int(self.order[position_a])
        v = int(self.order[position_b])
        edge_ids = self._affected_edges(u, v)
        old_distances = [
            abs(int(self.pi[int(self.grid.edge_u[edge_id])]) - int(self.pi[int(self.grid.edge_v[edge_id])]))
            for edge_id in edge_ids
        ]
        self.order[position_a], self.order[position_b] = v, u
        self.pi[u], self.pi[v] = position_b, position_a
        new_distances = [
            abs(int(self.pi[int(self.grid.edge_u[edge_id])]) - int(self.pi[int(self.grid.edge_v[edge_id])]))
            for edge_id in edge_ids
        ]
        deltas: dict[int, int] = {}
        for distance in old_distances:
            deltas[distance] = deltas.get(distance, 0) - 1
        for distance in new_distances:
            deltas[distance] = deltas.get(distance, 0) + 1
        for distance, delta in deltas.items():
            if delta:
                self.histogram[distance] += delta
                self.fenwick.add(distance, delta)
                self.total_distance += distance * delta
        if self.histogram[0] != 0 or np.any(self.histogram[1:] < 0):
            raise AssertionError("incremental histogram became invalid")

    def verify(self, label: str, expected_metrics: LocalityMetrics | None = None) -> LocalityMetrics:
        """Full cross-check: d_seq histogram, mean, p50, and p90 must match."""
        validate_order(self.order)
        full_metrics, d_all, full_histogram = metrics_from_pi(self.pi, self.grid)
        if not np.array_equal(full_histogram, self.histogram):
            raise AssertionError(f"{label}: incremental d_seq histogram mismatch")
        if int(np.sum(d_all)) != self.total_distance:
            raise AssertionError(f"{label}: incremental distance sum mismatch")
        if self.fenwick.total() != self.grid.edge_count:
            raise AssertionError(f"{label}: Fenwick total mismatch")
        incremental_metrics = self.metrics()
        for name in ("mean", "p50", "p90", "p95"):
            if getattr(full_metrics, name) != getattr(incremental_metrics, name):
                raise AssertionError(f"{label}: incremental {name} mismatch")
        if full_metrics.maximum != incremental_metrics.maximum:
            raise AssertionError(f"{label}: incremental max mismatch")
        if expected_metrics is not None:
            for name in ("mean", "p50", "p90", "p95", "maximum", "dx", "dy", "axis_bias"):
                if getattr(full_metrics, name) != getattr(expected_metrics, name):
                    raise AssertionError(f"{label}: initialization {name} mismatch")
        return full_metrics


def local_pairs(n_cells: int, max_distance: int) -> np.ndarray:
    pairs = [
        (left, right)
        for distance in range(1, max_distance + 1)
        for left in range(0, n_cells - distance)
        for right in (left + distance,)
    ]
    return np.asarray(pairs, dtype=np.int64)


def propose_swap(
    rng: np.random.Generator,
    n_cells: int,
    local_pair_table: np.ndarray,
) -> tuple[int, int, int]:
    """Use exactly 1/3 adjacent, local-position, and global-position proposals."""
    proposal_type = int(rng.integers(0, 3))
    if proposal_type == 0:
        left = int(rng.integers(0, n_cells - 1))
        return proposal_type, left, left + 1
    if proposal_type == 1:
        pair = local_pair_table[int(rng.integers(0, local_pair_table.shape[0]))]
        return proposal_type, int(pair[0]), int(pair[1])
    left = int(rng.integers(0, n_cells))
    other = int(rng.integers(0, n_cells - 1))
    right = other if other < left else other + 1
    return proposal_type, left, right


def normalized_kendall_tau(order: np.ndarray, reference_pi: np.ndarray) -> float:
    ranks = reference_pi[order]
    size = int(ranks.size)
    tree = Fenwick(np.zeros(size + 1, dtype=np.int64))
    inversions = 0
    seen = 0
    for rank in ranks.tolist():
        index = int(rank) + 1
        less_or_equal = 0
        query_index = index
        while query_index:
            less_or_equal += int(tree.tree[query_index])
            query_index -= query_index & -query_index
        inversions += seen - less_or_equal
        tree.add(index, 1)
        seen += 1
    return inversions / (size * (size - 1) / 2)


def continuous_edge_set(order: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in zip(order[:-1], order[1:])
    }


def nearest_g_distances(
    order: np.ndarray,
    g_orders: dict[str, np.ndarray],
    g_pis: dict[str, np.ndarray],
    g_edge_sets: dict[str, set[tuple[int, int]]],
) -> tuple[str, float, float, float]:
    """Nearest G is frozen as minimum order-position Hamming, ties by G label."""
    candidate_edges = continuous_edge_set(order)
    choices: list[tuple[float, str, float, float]] = []
    for label in ("G1", "G2", "G3", "G4"):
        hamming = float(np.mean(order != g_orders[label]))
        kendall = normalized_kendall_tau(order, g_pis[label])
        reference_edges = g_edge_sets[label]
        union = len(candidate_edges | reference_edges)
        jaccard = len(candidate_edges & reference_edges) / union
        choices.append((hamming, label, kendall, jaccard))
    hamming, label, kendall, jaccard = min(choices, key=lambda item: (item[0], item[1]))
    return label, hamming, kendall, jaccard


def directional_coverage(pi: np.ndarray, grid: Grid) -> dict[str, tuple[float, ...]]:
    """C_dir for one candidate path, i.e. P={pi}; no AUC is computed."""
    direction_pairs = {
        "RIGHT": (grid.horizontal_u, grid.horizontal_v),
        "LEFT": (grid.horizontal_v, grid.horizontal_u),
        "DOWN": (grid.vertical_u, grid.vertical_v),
        "UP": (grid.vertical_v, grid.vertical_u),
    }
    tau_values = tuple(threshold * (pi.size - 1) for threshold in THRESHOLDS)
    coverage: dict[str, tuple[float, ...]] = {}
    for label, (u, v) in direction_pairs.items():
        forward_distance = pi[v] - pi[u]
        coverage[label] = tuple(
            float(np.mean((forward_distance > 0) & (forward_distance <= tau)))
            for tau in tau_values
        )
    return coverage


def run_chain(
    n: int,
    source_g: str,
    grid: Grid,
    g_orders: dict[str, np.ndarray],
    g_pis: dict[str, np.ndarray],
    g_edge_sets: dict[str, set[tuple[int, int]]],
) -> ChainResult:
    seed = SEARCH_SEEDS[(n, source_g)]
    rng = np.random.default_rng(seed)
    state = IncrementalState(g_orders[source_g], grid)
    if not state.c5():
        raise AssertionError(f"initial {source_g}, n={n} does not meet C5")
    pair_table = local_pairs(n * n, n)
    proposal_counts = [0, 0, 0]
    accepted_total = 0
    accepted_after_burn = 0
    candidate: Candidate | None = None
    diagnostics: list[DiagnosticPoint] = []
    checks = 1

    for proposal in range(1, MAX_PROPOSALS + 1):
        proposal_type, position_a, position_b = propose_swap(rng, n * n, pair_table)
        if position_a == position_b:
            raise AssertionError("proposal sampled the same position")
        proposal_counts[proposal_type] += 1
        state.apply_swap(position_a, position_b)
        if state.c5():
            accepted_total += 1
            if proposal > BURN_IN_PROPOSALS:
                accepted_after_burn += 1
        else:
            state.apply_swap(position_a, position_b)

        if proposal % VERIFY_EVERY_PROPOSALS == 0:
            state.verify(f"n={n} {source_g} proposal={proposal}")
            checks += 1

        if proposal > BURN_IN_PROPOSALS and (proposal - BURN_IN_PROPOSALS) % THINNING_PROPOSALS == 0:
            nearest_g, hamming, kendall, jaccard = nearest_g_distances(
                state.order, g_orders, g_pis, g_edge_sets
            )
            diagnostics.append(
                DiagnosticPoint(
                    proposal=proposal,
                    accepted_total=accepted_total,
                    accepted_after_burn=accepted_after_burn,
                    acceptance_rate=accepted_total / proposal,
                    nearest_g=nearest_g,
                    hamming=hamming,
                    kendall_tau=kendall,
                    edge_jaccard=jaccard,
                )
            )
            # Candidate selection is frozen: the first thinning checkpoint that
            # is C5-valid and differs from every G is saved.  The chain still
            # runs to the full proposal budget; later, farther states cannot
            # replace this official audit candidate.
            if candidate is None and not any(
                np.array_equal(state.order, g_order) for g_order in g_orders.values()
            ):
                metrics = state.verify(
                    f"n={n} {source_g} candidate proposal={proposal}"
                )
                checks += 1
                candidate = Candidate(
                    n=n,
                    source_g=source_g,
                    seed=seed,
                    proposal=proposal,
                    accepted_total=accepted_total,
                    accepted_after_burn=accepted_after_burn,
                    metrics=metrics,
                    nearest_g=nearest_g,
                    hamming=hamming,
                    kendall_tau=kendall,
                    edge_jaccard=jaccard,
                    cdir=directional_coverage(state.pi, grid),
                )

    state.verify(f"n={n} {source_g} final")
    checks += 1
    return ChainResult(
        n=n,
        source_g=source_g,
        seed=seed,
        proposal_counts=tuple(proposal_counts),
        accepted_total=accepted_total,
        accepted_after_burn=accepted_after_burn,
        candidate=candidate,
        diagnostics=tuple(diagnostics),
        checks=checks,
    )


def fmt(value: float | int, digits: int = 6) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def metric_triplet(metrics: LocalityMetrics) -> str:
    return f"mean={fmt(metrics.mean)}, p50={fmt(metrics.p50)}, p90={fmt(metrics.p90)}"


def range_text(values: Iterable[float]) -> str:
    items = list(values)
    return f"[{fmt(min(items))}, {fmt(max(items))}]"


def render_report(
    elapsed_seconds: float,
    g_metrics: dict[int, dict[str, LocalityMetrics]],
    block_results: dict[tuple[int, int], dict[str, object]],
    chain_results: list[ChainResult],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# B0 Locality-Control Feasibility Audit",
            "",
            "## Scope And Frozen Definitions",
            "",
            "- Formal source read before execution: `C:\\Users\\DELL\\Desktop\\一些md\\P0A_B0_FORMAL_DEFINITIONS.md`.",
            "- CPU-only script: Python standard library plus NumPy; no model, checkpoint, data loader, GPU, training, or inference imports.",
            "- `order[t]` is the row-major cell visited at step `t`; `pi[u]` is its visit step. Every path passed `pi[order] = arange(N)` and `pi = argsort(order)`.",
            "- All locality statistics use all undirected horizontal and vertical four-neighbor edges and `d_seq=abs(pi[u]-pi[v])`.",
            "- All percentiles call `numpy.percentile(..., method=\"linear\")`. The incremental Fenwick calculation uses the identical zero-based linear interpolation rule.",
            "- B0 reports the four frozen `C_dir` nodes only. No AUC is defined or reported.",
            "",
            "## Execution",
            "",
            f"- Command: `python tools/audit_p0a_locality_control.py`",
            f"- Python: `{sys.version.split()[0]}`; NumPy: `{np.__version__}`; platform: `{platform.platform()}`.",
            f"- Actual elapsed wall time: `{elapsed_seconds:.3f}` s.",
            f"- Block audit: {BLOCK_SAMPLES} samples for each (n,b), with independent frozen seeds.",
            f"- Constraint walk: {MAX_PROPOSALS:,} proposals per chain, burn-in {BURN_IN_PROPOSALS:,}, thinning {THINNING_PROPOSALS:,}; all counts are proposals, not accepted moves.",
            "- Proposal selection is exactly 1/3 each for adjacent sequence positions, uniformly sampled unordered local sequence-position pairs with distance `1..n`, and uniformly sampled ordered global distinct-position pairs. Same-position proposals are prohibited.",
            "",
            "## 1. G Exact Regression",
            "",
            "| n | G | mean | p50 | p90 | p95 | max | d_x | d_y | AxisBias | Status |",
            "|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|",
        ]
    )
    for n in GRID_SIZES:
        for label in ("G1", "G2", "G3", "G4"):
            metric = g_metrics[n][label]
            lines.append(
                f"| {n} | {label} | {fmt(metric.mean)} | {fmt(metric.p50)} | {fmt(metric.p90)} | {fmt(metric.p95)} | {metric.maximum} | {fmt(metric.dx)} | {fmt(metric.dy)} | {fmt(metric.axis_bias)} | PASS |"
            )
    lines.extend(
        [
            "",
            "All G paths are legal. For both grids, all four paths equal the frozen mean/p50/p90 regression values; G1/G2 have identical edgewise `d_seq`; G1/G3 have identical aggregate distributions and opposite AxisBias. Status: **PASS**.",
            "",
            "## 2. Old Blocked Serpentine Audit",
            "",
            "`b` is blocks per axis, total blocks are `b^2`, block side is `m=n/b`, and every block order has length `m^2=(n/b)^2`. Every (n,b) passed the no-duplicate/no-omission, intra-block four-neighbor continuity, and eight-unique-orientation assertions.",
            "",
            "| n | b | m | seed | samples | mean [min,p5,p50,p95,max] | p50 [min,p5,p50,p95,max] | p90 [min,p5,p50,p95,max] | C5 count | C5 rate | p50=1 count | p50=1 rate | closest sample | closest metrics | Status |",
            "|---:|---:|---:|---:|---:|:--|:--|:--|---:|---:|---:|---:|---:|:--|:--|",
        ]
    )
    for (n, b), result in sorted(block_results.items()):
        ranges = result["ranges"]
        best_metrics = result["best_metrics"]
        sample_count = int(result["sample_count"])
        c5_count = int(result["c5_count"])
        p50_one_count = int(result["p50_is_one_count"])
        def format_range(name: str) -> str:
            return ", ".join(fmt(value) for value in ranges[name])
        lines.append(
            f"| {n} | {b} | {result['m']} | {result['seed']} | {sample_count} | {format_range('mean')} | {format_range('p50')} | {format_range('p90')} | {c5_count} | {c5_count / sample_count:.6f} | {p50_one_count} | {p50_one_count / sample_count:.6f} | {result['best_index']} | {metric_triplet(best_metrics)} | PASS |"
        )
    lines.extend(
        [
            "",
            "This is a diagnostic of the old candidate family only. C5 was not relaxed, and no sampled path is promoted to a final L generator.",
            "",
            "## 3. General Constrained-Permutation Existence Audit",
            "",
            "Each chain begins at its named G path. A proposal is accepted only when its current mean, p50, and p90 remain within the frozen C5 intervals. The first post-burn thinning checkpoint that is C5-valid and differs from all four G permutations is retained; all chains continue to the fixed 1,000,000-proposal budget. No later state replaces a retained candidate based on distance.",
            "",
            "| n | source | seed | adjacent/local/global proposals | accepted | accepted after burn | acceptance rate | full checks | first candidate proposal | candidate C5 | nearest G (Hamming rule) | Hamming | norm. Kendall tau | edge Jaccard | Status |",
            "|---:|:--|---:|:--|---:|---:|---:|---:|---:|:--|:--|---:|---:|---:|:--|",
        ]
    )
    candidates_by_n: dict[int, int] = {n: 0 for n in GRID_SIZES}
    for result in chain_results:
        candidate = result.candidate
        if candidate is None:
            candidate_proposal = "-"
            candidate_c5 = "-"
            nearest = "-"
            hamming = kendall = jaccard = "-"
            status = "FAIL: no non-G sampled candidate"
        else:
            candidates_by_n[result.n] += 1
            candidate_proposal = str(candidate.proposal)
            candidate_c5 = "PASS"
            nearest = candidate.nearest_g
            hamming = fmt(candidate.hamming)
            kendall = fmt(candidate.kendall_tau)
            jaccard = fmt(candidate.edge_jaccard)
            status = "PASS"
        counts = "/".join(str(value) for value in result.proposal_counts)
        lines.append(
            f"| {result.n} | {result.source_g} | {result.seed} | {counts} | {result.accepted_total} | {result.accepted_after_burn} | {result.accepted_total / MAX_PROPOSALS:.6f} | {result.checks} | {candidate_proposal} | {candidate_c5} | {nearest} | {hamming} | {kendall} | {jaccard} | {status} |"
        )

    existence = "EXISTENCE_PASS" if all(candidates_by_n[n] == 4 for n in GRID_SIZES) else (
        "EXISTENCE_PARTIAL" if any(candidates_by_n[n] == 4 for n in GRID_SIZES) else "EXISTENCE_FAIL"
    )
    lines.extend(
        [
            "",
            f"Candidates found: n=8: {candidates_by_n[8]}/4; n=32: {candidates_by_n[32]}/4. **{existence}**.",
            "",
            "`EXISTENCE_PASS` only means that the frozen budget found four non-G permutations satisfying C5 at each scale. It does not approve this random walk as the final L generator and does not establish that the candidates are sufficiently independent of G.",
            "",
            "### Candidate Locality And Directional Coverage",
            "",
            "| n | source | mean | p50 | p90 | p95 | max | d_x | d_y | AxisBias | C5 |",
            "|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|",
        ]
    )
    for result in chain_results:
        if result.candidate is None:
            continue
        candidate = result.candidate
        metric = candidate.metrics
        lines.append(
            f"| {candidate.n} | {candidate.source_g} | {fmt(metric.mean)} | {fmt(metric.p50)} | {fmt(metric.p90)} | {fmt(metric.p95)} | {metric.maximum} | {fmt(metric.dx)} | {fmt(metric.dy)} | {fmt(metric.axis_bias)} | {'PASS' if meets_c5(metric, candidate.n) else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "| n | source | tau_tilde | tau=tau_tilde*(N-1) | RIGHT | LEFT | DOWN | UP |",
            "|---:|:--|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in chain_results:
        if result.candidate is None:
            continue
        candidate = result.candidate
        for index, threshold in enumerate(THRESHOLDS):
            tau = threshold * (candidate.n * candidate.n - 1)
            lines.append(
                f"| {candidate.n} | {candidate.source_g} | {threshold:.2f} | {fmt(tau)} | {fmt(candidate.cdir['RIGHT'][index])} | {fmt(candidate.cdir['LEFT'][index])} | {fmt(candidate.cdir['DOWN'][index])} | {fmt(candidate.cdir['UP'][index])} |"
            )

    lines.extend(
        [
            "",
            "### Post-Burn Chain Diagnostics",
            "",
            "Nearest G is pre-defined as the reference with minimum order-position Hamming distance, with G-label tie breaking. At every fixed thinning checkpoint the report records Hamming, normalized Kendall tau, and continuous-sequence-edge Jaccard against that reference; the table gives their observed ranges. No 'local neighborhood' distance threshold is frozen, so no binary long-term-residence conclusion is drawn from these diagnostics.",
            "",
            "| n | source | checkpoints | nearest G labels observed | Hamming range | norm. Kendall tau range | edge Jaccard range | accepted-after-burn range | Status |",
            "|---:|:--|---:|:--|:--|:--|:--|:--|:--|",
        ]
    )
    for result in chain_results:
        diagnostics = result.diagnostics
        if not diagnostics:
            lines.append(f"| {result.n} | {result.source_g} | 0 | - | - | - | - | - | FAIL |")
            continue
        labels = ",".join(sorted({point.nearest_g for point in diagnostics}))
        lines.append(
            f"| {result.n} | {result.source_g} | {len(diagnostics)} | {labels} | {range_text(point.hamming for point in diagnostics)} | {range_text(point.kendall_tau for point in diagnostics)} | {range_text(point.edge_jaccard for point in diagnostics)} | [{min(point.accepted_after_burn for point in diagnostics)}, {max(point.accepted_after_burn for point in diagnostics)}] | PASS |"
        )

    lines.extend(
        [
            "",
            "## 4. Final-Generator Candidate Comparison (Not Implemented Or Selected)",
            "",
            "| Candidate family | Reproducible | Risk of implicit G bias | Multiple instances | Needs new frozen distance threshold | CPU cost | Changes C5 | Decision |",
            "|:--|:--|:--|:--|:--|:--|:--|:--|",
            "| Constrained random walk used here | Yes, with frozen proposals/seeds/budget | Initialization at G can retain local ancestry | Yes | No for B0 existence, but needed before any independence claim | Moderate | No | Not selected |",
            "| Simulated annealing/direct optimization | Yes if objective, schedule, and seeds are frozen | Depends on objective construction | Yes | Potentially | Moderate to high | No, if C5 remains hard | Not selected |",
            "| Statistics-preserving local rewiring | Yes if move set and stopping rule are frozen | Depends on permitted rewires | Yes | Potentially | Moderate | No | Not selected |",
            "| Other explicitly specified generator | Only after proposal/seeds/stopping are frozen | Must be audited | Depends on design | Depends on design | Unknown | Must not | Not selected |",
            "",
            "## 5. Still Requires Researcher Freezing",
            "",
            "- Final L generator and its proposal/stopping specification.",
            "- Any requirement that L be sufficiently distant or independent from G.",
            "- Any AxisBias, polarity, or C_dir matching requirement beyond C5.",
            "- Any AUC definition or aggregation of C_dir nodes.",
            "",
            "## Boundary Declaration",
            "",
            "- Existing model/training/analysis/config files modified: no.",
            "- `outputs/` accessed: no.",
            "- Training, model inference, or GPU task run: no.",
            "- Commit or push performed: no.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    if tuple(int(part) for part in np.__version__.split(".")[:2]) < (1, 22):
        raise RuntimeError("NumPy >= 1.22 is required for method='linear'")
    try:
        grids = {n: make_grid(n) for n in GRID_SIZES}
        g_metrics = assert_g_regression(grids)
        block_results = block_audit(grids)
        chain_results: list[ChainResult] = []
        for n in GRID_SIZES:
            g_orders = named_g_orders(n)
            g_pis = {label: validate_order(order) for label, order in g_orders.items()}
            g_edge_sets = {label: continuous_edge_set(order) for label, order in g_orders.items()}
            for source_g in ("G1", "G2", "G3", "G4"):
                chain_results.append(
                    run_chain(n, source_g, grids[n], g_orders, g_pis, g_edge_sets)
                )
        report = render_report(time.perf_counter() - started, g_metrics, block_results, chain_results)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"Wrote {REPORT_PATH}")
    except (AssertionError, RuntimeError) as error:
        failure = "\n".join(
            [
                "# B0 Locality-Control Feasibility Audit",
                "",
                "## Status",
                "",
                f"**FAIL: audit stopped immediately.** `{type(error).__name__}: {error}`",
                "",
                "No threshold, seed, budget, or generator rule was adjusted after this failure.",
                "",
                "## Boundary Declaration",
                "",
                "- Existing model/training/analysis/config files modified: no.",
                "- `outputs/` accessed: no.",
                "- Training, model inference, or GPU task run: no.",
                "- Commit or push performed: no.",
                "",
            ]
        )
        REPORT_PATH.write_text(failure, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
