#!/usr/bin/env python3
"""B1 CPU-only extraction of the pre-specified B0 G1-chain candidates.

This script intentionally imports the completed B0 geometry audit rather than
copying its path, proposal, locality, or percentile logic.  It never imports
model, data, checkpoint, or output modules.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

import audit_p0a_locality_control as b0


FORMAL_PATH = Path("docs/P0A_B0_FORMAL_DEFINITIONS.md")
B0_SCRIPT_PATH = Path("tools/audit_p0a_locality_control.py")
B0_REPORT_PATH = Path("REPORT_B0_locality_control_feasibility.md")
JSON_PATH = Path("P0B_L_PATH_BANK_CANDIDATE.json")
REPORT_PATH = Path("REPORT_B1_L_PATH_BANK_CANDIDATE.md")
SOURCE_G = "G1"
SOURCE_SEEDS = {8: 2026072101, 32: 2026072201}
SOURCE_PROPOSAL = 125_000
STATUS_READY = "B1_CANDIDATE_READY"


class RecoveryFailure(AssertionError):
    """A frozen G1 recovery regression did not reproduce B0."""


# Every value below was transcribed from the completed B0 report.  B1 compares
# the re-generated aggregate after formatting to six decimals, exactly as B0
# displayed it; no B0 file is modified if a mismatch is found.
B0_BLOCK_EXPECTED = {
    (8, 2): {
        "mean": (4.714286, 5.142857, 6.000000, 7.142857, 7.571429),
        "p50": (1.000000, 1.000000, 1.000000, 1.000000, 1.000000),
        "p90": (7.000000, 7.000000, 16.900000, 30.100000, 38.200000),
        "c5_count": 0,
        "p50_one_count": 20_000,
        "best_index": 450,
        "best": (4.714286, 1.000000, 8.800000),
    },
    (8, 4): {
        "mean": (6.678571, 8.839286, 10.571429, 12.357143, 14.982143),
        "p50": (3.000000, 3.000000, 3.000000, 3.000000, 3.000000),
        "p90": (17.000000, 26.000000, 33.800000, 40.800000, 49.000000),
        "c5_count": 0,
        "p50_one_count": 0,
        "best_index": 10_479,
        "best": (6.946429, 3.000000, 17.800000),
    },
    (32, 2): {
        "mean": (16.741935, 18.677419, 22.548387, 26.677419, 28.612903),
        "p50": (1.000000, 1.000000, 1.000000, 1.000000, 1.000000),
        "p90": (27.000000, 27.000000, 27.000000, 27.000000, 27.000000),
        "c5_count": 0,
        "p50_one_count": 20_000,
        "best_index": 39,
        "best": (16.741935, 1.000000, 27.000000),
    },
    (32, 4): {
        "mean": (24.854839, 32.822581, 39.112903, 45.580645, 56.258065),
        "p50": (1.000000, 1.000000, 1.000000, 1.000000, 1.000000),
        "p90": (15.000000, 15.000000, 15.000000, 15.000000, 15.000000),
        "c5_count": 0,
        "p50_one_count": 20_000,
        "best_index": 4_487,
        "best": (24.854839, 1.000000, 15.000000),
    },
}


B0_G1_RECOVERY_EXPECTED = {
    8: {
        "proposal": 125_000,
        "mean": 4.946429,
        "p50": 4.500000,
        "p90": 8.000000,
        "p95": 12.000000,
        "maximum": 17,
        "dx": 2.553571,
        "dy": 7.339286,
        "axis_bias": -1.055749,
        "hamming": 0.781250,
        "kendall": 0.047123,
        "edge_jaccard": 0.235294,
    },
    32: {
        "proposal": 125_000,
        "mean": 18.140625,
        "p50": 15.000000,
        "p90": 35.000000,
        "p95": 41.000000,
        "maximum": 66,
        "dx": 4.760081,
        "dy": 31.521169,
        "axis_bias": -1.890395,
        "hamming": 0.919922,
        "kendall": 0.006411,
        "edge_jaccard": 0.111957,
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def order_sha256(order: np.ndarray) -> str:
    return sha256_bytes(np.asarray(order, dtype=np.int64).tobytes(order="C"))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def assert_six(actual: float, expected: float, label: str) -> None:
    if fmt(actual) != fmt(expected):
        raise RecoveryFailure(f"{label}: {fmt(actual)} != {fmt(expected)}")


def metrics_record(metrics: b0.LocalityMetrics) -> dict[str, float | int]:
    return {
        "mean": metrics.mean,
        "p50": metrics.p50,
        "p90": metrics.p90,
        "p95": metrics.p95,
        "max": metrics.maximum,
        "d_x": metrics.dx,
        "d_y": metrics.dy,
        "axis_bias": metrics.axis_bias,
    }


def assert_sources() -> tuple[str, str]:
    for path in (FORMAL_PATH, B0_SCRIPT_PATH, B0_REPORT_PATH):
        if not path.is_file():
            raise RuntimeError(f"required B1 source is missing: {path}")
    formal_hash = sha256_file(FORMAL_PATH)
    if formal_hash.upper() != "3E79D2F8C941F7C54F11EAEE21332265D9D064A9FB9971169FA18A6295D3CC8C":
        raise RuntimeError("formal definition SHA-256 does not match the approved source")
    report_text = B0_REPORT_PATH.read_text(encoding="utf-8")
    required_b0_text = (
        "Candidates found: n=8: 4/4; n=32: 4/4. **EXISTENCE_PASS**.",
        "| 8 | G1 | 2026072101 |",
        "| 32 | G1 | 2026072201 |",
        "| 8 | 2 | 4 | 2026072001 | 20000 |",
        "| 32 | 4 | 8 | 2026072004 | 20000 |",
    )
    for expected_text in required_b0_text:
        if expected_text not in report_text:
            raise RuntimeError(f"B0 report lacks frozen source text: {expected_text}")
    return formal_hash, sha256_file(B0_REPORT_PATH)


def block_summary(values: np.ndarray, n: int) -> dict[str, object]:
    bounds = b0.c5_bounds(n)
    c5 = (
        (values[:, 0] >= bounds["mean"][0])
        & (values[:, 0] <= bounds["mean"][1])
        & (values[:, 1] >= bounds["p50"][0])
        & (values[:, 1] <= bounds["p50"][1])
        & (values[:, 2] >= bounds["p90"][0])
        & (values[:, 2] <= bounds["p90"][1])
    )
    return {
        name: tuple(
            float(item)
            for item in np.percentile(values[:, column], (0, 5, 50, 95, 100), method="linear")
        )
        for column, name in enumerate(("mean", "p50", "p90"))
    } | {
        "c5_count": int(np.sum(c5)),
        "c5_rate": float(np.mean(c5)),
        "p50_one_count": int(np.sum(values[:, 1] == 1.0)),
        "p50_one_rate": float(np.mean(values[:, 1] == 1.0)),
    }


def assert_block_aggregate(
    n: int,
    b: int,
    values: np.ndarray,
    best_index: int,
    best_triplet: tuple[float, float, float],
) -> dict[str, object]:
    expected = B0_BLOCK_EXPECTED[(n, b)]
    observed = block_summary(values, n)
    for name in ("mean", "p50", "p90"):
        for index, (actual, target) in enumerate(zip(observed[name], expected[name])):
            if fmt(actual) != fmt(target):
                raise RuntimeError(f"B0 aggregate mismatch {(n, b)} {name}[{index}]")
    if observed["c5_count"] != expected["c5_count"]:
        raise RuntimeError(f"B0 aggregate C5 mismatch for {(n, b)}")
    if observed["p50_one_count"] != expected["p50_one_count"]:
        raise RuntimeError(f"B0 aggregate p50==1 mismatch for {(n, b)}")
    if best_index != expected["best_index"]:
        raise RuntimeError(f"B0 aggregate closest-sample mismatch for {(n, b)}")
    for actual, target in zip(best_triplet, expected["best"]):
        if fmt(actual) != fmt(target):
            raise RuntimeError(f"B0 aggregate closest-metrics mismatch for {(n, b)}")
    return observed


def regenerate_block_stability(grids: dict[int, b0.Grid]) -> dict[tuple[int, int], dict[str, object]]:
    results: dict[tuple[int, int], dict[str, object]] = {}
    for n in b0.GRID_SIZES:
        for b in b0.BLOCKS_PER_AXIS:
            m = n // b
            local_orders = b0.assert_orientation_family(m)
            rng = np.random.default_rng(b0.BLOCK_AUDIT_SEEDS[(n, b)])
            values = np.empty((b0.BLOCK_SAMPLES, 3), dtype=np.float64)
            target = np.array(((n + 1) / 2.0, (n + 1) / 2.0, float(n)))
            best_index = -1
            best_score = math.inf
            best_triplet = (math.nan, math.nan, math.nan)
            for sample_index in range(b0.BLOCK_SAMPLES):
                order = b0.block_candidate_order(n, b, rng, local_orders)
                metrics, _, _ = b0.metrics_from_pi(b0.inverse_order(order), grids[n])
                triplet = (metrics.mean, metrics.p50, metrics.p90)
                values[sample_index] = triplet
                score = float(np.sum(((values[sample_index] - target) / target) ** 2))
                if score < best_score:
                    best_score = score
                    best_index = sample_index
                    best_triplet = triplet
            aggregate = assert_block_aggregate(n, b, values, best_index, best_triplet)
            chunks = []
            for start in range(0, b0.BLOCK_SAMPLES, 2_000):
                chunk = block_summary(values[start : start + 2_000], n)
                chunks.append({"start": start + 1, "end": start + 2_000, **chunk})
            results[(n, b)] = {
                "n": n,
                "b": b,
                "m": m,
                "seed": b0.BLOCK_AUDIT_SEEDS[(n, b)],
                "aggregate": aggregate,
                "chunks": chunks,
            }
    return results


def recover_q(n: int, grid: b0.Grid) -> tuple[np.ndarray, b0.LocalityMetrics, tuple[str, float, float, float]]:
    """Replay only the frozen G1 chain to its first post-burn thinning point."""
    if SOURCE_SEEDS[n] != b0.SEARCH_SEEDS[(n, SOURCE_G)]:
        raise RecoveryFailure(f"n={n}: source seed differs from B0 G1 seed")
    g_orders = b0.named_g_orders(n)
    g_pis = {label: b0.validate_order(order) for label, order in g_orders.items()}
    g_edge_sets = {label: b0.continuous_edge_set(order) for label, order in g_orders.items()}
    rng = np.random.default_rng(SOURCE_SEEDS[n])
    state = b0.IncrementalState(g_orders[SOURCE_G], grid)
    pair_table = b0.local_pairs(n * n, n)
    for proposal in range(1, SOURCE_PROPOSAL + 1):
        proposal_type, position_a, position_b = b0.propose_swap(rng, n * n, pair_table)
        if proposal_type not in (0, 1, 2) or position_a == position_b:
            raise RecoveryFailure(f"n={n}: invalid frozen B0 proposal")
        state.apply_swap(position_a, position_b)
        if not state.c5():
            state.apply_swap(position_a, position_b)
    if SOURCE_PROPOSAL <= b0.BURN_IN_PROPOSALS:
        raise RecoveryFailure("source proposal is not after burn-in")
    if (SOURCE_PROPOSAL - b0.BURN_IN_PROPOSALS) % b0.THINNING_PROPOSALS:
        raise RecoveryFailure("source proposal is not a frozen thinning checkpoint")
    if any(np.array_equal(state.order, item) for item in g_orders.values()):
        raise RecoveryFailure(f"n={n}: recovered state equals a G path")
    metrics = state.verify(f"B1 recovery n={n} proposal={SOURCE_PROPOSAL}")
    nearest = b0.nearest_g_distances(state.order, g_orders, g_pis, g_edge_sets)
    expected = B0_G1_RECOVERY_EXPECTED[n]
    if SOURCE_PROPOSAL != expected["proposal"]:
        raise RecoveryFailure(f"n={n}: proposal regression mismatch")
    if metrics.maximum != expected["maximum"]:
        raise RecoveryFailure(f"n={n}: max {metrics.maximum} != {expected['maximum']}")
    for name, actual in (
        ("mean", metrics.mean),
        ("p50", metrics.p50),
        ("p90", metrics.p90),
        ("p95", metrics.p95),
        ("dx", metrics.dx),
        ("dy", metrics.dy),
        ("axis_bias", metrics.axis_bias),
        ("hamming", nearest[1]),
        ("kendall", nearest[2]),
        ("edge_jaccard", nearest[3]),
    ):
        assert_six(actual, expected[name], f"n={n} {name}")
    if nearest[0] != SOURCE_G:
        raise RecoveryFailure(f"n={n}: nearest G is {nearest[0]}, not G1")
    return state.order.copy(), metrics, nearest


def transpose_cells(order: np.ndarray, n: int) -> np.ndarray:
    rows, columns = divmod(order, n)
    return (columns * n + rows).astype(np.int64, copy=False)


def make_orbit(q: np.ndarray, n: int) -> dict[str, np.ndarray]:
    orbit = {
        "L1": q.copy(),
        "L2": q[::-1].copy(),
        "L3": transpose_cells(q, n),
        "L4": transpose_cells(q, n)[::-1].copy(),
    }
    if not np.array_equal(orbit["L2"], orbit["L1"][::-1]):
        raise AssertionError(f"n={n}: L2 reversal identity failed")
    if not np.array_equal(orbit["L3"], transpose_cells(orbit["L1"], n)):
        raise AssertionError(f"n={n}: L3 transpose identity failed")
    if not np.array_equal(orbit["L4"], orbit["L3"][::-1]):
        raise AssertionError(f"n={n}: L4 reversal identity failed")
    for label, order in orbit.items():
        b0.validate_order(order)
        if not b0.meets_c5(b0.metrics_from_pi(b0.inverse_order(order), b0.make_grid(n))[0], n):
            raise AssertionError(f"n={n}: {label} fails C5")
    if len({tuple(order.tolist()) for order in orbit.values()}) != 4:
        raise AssertionError(f"n={n}: orbit paths are not pairwise distinct")
    return orbit


def single_cdir(pi: np.ndarray, grid: b0.Grid) -> dict[str, tuple[float, ...]]:
    return b0.directional_coverage(pi, grid)


def collection_cdir(pis: list[np.ndarray], grid: b0.Grid) -> dict[str, tuple[float, ...]]:
    directions = {
        "RIGHT": (grid.horizontal_u, grid.horizontal_v),
        "LEFT": (grid.horizontal_v, grid.horizontal_u),
        "DOWN": (grid.vertical_u, grid.vertical_v),
        "UP": (grid.vertical_v, grid.vertical_u),
    }
    tau_values = tuple(threshold * (grid.n * grid.n - 1) for threshold in b0.THRESHOLDS)
    result: dict[str, tuple[float, ...]] = {}
    for label, (u, v) in directions.items():
        minimum = np.full(u.size, np.inf, dtype=np.float64)
        for pi in pis:
            forward = pi[v] - pi[u]
            valid = forward > 0
            minimum[valid] = np.minimum(minimum[valid], forward[valid])
        result[label] = tuple(float(np.mean(minimum <= tau)) for tau in tau_values)
    return result


def pairwise_matrices(orbit: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    labels = ("L1", "L2", "L3", "L4")
    edge_sets = {label: b0.continuous_edge_set(orbit[label]) for label in labels}
    hamming: list[list[float]] = []
    kendall: list[list[float]] = []
    jaccard: list[list[float]] = []
    for left in labels:
        hamming_row = []
        kendall_row = []
        jaccard_row = []
        for right in labels:
            hamming_row.append(float(np.mean(orbit[left] != orbit[right])))
            kendall_row.append(b0.normalized_kendall_tau(orbit[left], b0.inverse_order(orbit[right])))
            union = len(edge_sets[left] | edge_sets[right])
            jaccard_row.append(len(edge_sets[left] & edge_sets[right]) / union)
        hamming.append(hamming_row)
        kendall.append(kendall_row)
        jaccard.append(jaccard_row)
    return {"labels": list(labels), "hamming": hamming, "kendall": kendall, "edge_jaccard": jaccard}


def validate_orbit_symmetries(
    orbit: dict[str, np.ndarray], grid: b0.Grid
) -> tuple[dict[str, b0.LocalityMetrics], dict[str, np.ndarray]]:
    metrics = {}
    pis = {}
    distances_all = {}
    for label, order in orbit.items():
        pi = b0.validate_order(order)
        metric, d_all, _ = b0.metrics_from_pi(pi, grid)
        if not b0.meets_c5(metric, grid.n):
            raise AssertionError(f"n={grid.n}: {label} violates C5")
        metrics[label] = metric
        pis[label] = pi
        distances_all[label] = d_all
    if not np.array_equal(distances_all["L1"], distances_all["L2"]):
        raise AssertionError(f"n={grid.n}: L1/L2 edgewise d_seq mismatch")
    if not np.array_equal(np.sort(distances_all["L1"]), np.sort(distances_all["L3"])):
        raise AssertionError(f"n={grid.n}: L1/L3 aggregate d_seq mismatch")
    if not math.isclose(metrics["L2"].axis_bias, metrics["L1"].axis_bias):
        raise AssertionError(f"n={grid.n}: L1/L2 AxisBias mismatch")
    if not math.isclose(metrics["L3"].axis_bias, -metrics["L1"].axis_bias):
        raise AssertionError(f"n={grid.n}: L1/L3 AxisBias sign mismatch")
    if not math.isclose(metrics["L4"].axis_bias, metrics["L3"].axis_bias):
        raise AssertionError(f"n={grid.n}: L3/L4 AxisBias mismatch")
    return metrics, pis


def sequence_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_edges = b0.continuous_edge_set(left)
    right_edges = b0.continuous_edge_set(right)
    overlap = len(left_edges & right_edges) / (left.size - 1)
    jaccard = len(left_edges & right_edges) / len(left_edges | right_edges)
    return overlap, jaccard


def serialise_path(
    label: str,
    construction: str,
    order: np.ndarray,
    metrics: b0.LocalityMetrics,
) -> dict[str, object]:
    inverse = b0.inverse_order(order)
    return {
        "label": label,
        "construction": construction,
        "order": [int(value) for value in order.tolist()],
        "order_sha256": order_sha256(order),
        "inverse_order_sha256": order_sha256(inverse),
        "locality_metrics": metrics_record(metrics),
        "axis_bias": metrics.axis_bias,
    }


def matrix_markdown(name: str, matrix: list[list[float]], labels: list[str]) -> list[str]:
    lines = [f"#### {name}", "", "| | " + " | ".join(labels) + " |", "|:--|" + "|".join([":--:"] * len(labels)) + "|"]
    for label, row in zip(labels, matrix):
        lines.append("| " + label + " | " + " | ".join(fmt(value) for value in row) + " |")
    return lines + [""]


def render_report(
    wall_seconds: float,
    cpu_seconds: float,
    formal_hash: str,
    b0_report_hash: str,
    block_results: dict[tuple[int, int], dict[str, object]],
    grids_data: list[dict[str, object]],
    json_hash: str,
) -> str:
    lines = [
        "# B1 L Symmetry-Orbit Candidate Extraction",
        "",
        "## Status",
        "",
        f"**{STATUS_READY}**",
        "",
        "B0 `EXISTENCE_PASS` is retained. This report does not approve a final L generator.",
        "",
        "## Execution",
        "",
        "- Command: `python tools/extract_p0b_l_candidate.py`",
        f"- Wall time: `{wall_seconds:.3f}` s; process CPU time: `{cpu_seconds:.3f}` s.",
        f"- NumPy: `{np.__version__}`.",
        f"- Formal definition SHA-256: `{formal_hash}`.",
        f"- Source B0 report SHA-256: `{b0_report_hash}`.",
        f"- Candidate JSON SHA-256: `{json_hash}`.",
        "- Only the frozen G1 chains were replayed: n=8 seed 2026072101 and n=32 seed 2026072201, stopping at proposal 125000.",
        "",
        "## 1. Blocked-Serpentine 2,000-Sample Stability",
        "",
        "All four aggregate B0 regressions passed, including the frozen `0/20000` C5 count. Rows show the five-number summary `[min, p5, p50, p95, max]` for each 2,000-sample interval.",
        "",
        "| n | b | m | samples | C5 aggregate | p50=1 aggregate | Status |",
        "|---:|---:|---:|---:|---:|---:|:--|",
    ]
    for (n, b), result in sorted(block_results.items()):
        aggregate = result["aggregate"]
        lines.append(
            f"| {n} | {b} | {result['m']} | 20000 | {aggregate['c5_count']}/20000 | {aggregate['p50_one_count']}/20000 | PASS |"
        )
    lines.extend(
        [
            "",
            "| n | b | interval | mean [min,p5,p50,p95,max] | p50 [min,p5,p50,p95,max] | p90 [min,p5,p50,p95,max] | C5 count/rate | p50=1 count/rate |",
            "|---:|---:|:--|:--|:--|:--|:--|:--|",
        ]
    )
    for (n, b), result in sorted(block_results.items()):
        for chunk in result["chunks"]:
            def summary_text(name: str) -> str:
                return ", ".join(fmt(value) for value in chunk[name])
            lines.append(
                f"| {n} | {b} | {chunk['start']}-{chunk['end']} | {summary_text('mean')} | {summary_text('p50')} | {summary_text('p90')} | {chunk['c5_count']}/2000 ({chunk['c5_rate']:.6f}) | {chunk['p50_one_count']}/2000 ({chunk['p50_one_rate']:.6f}) |"
            )

    lines.extend(
        [
            "",
            "## 2. Frozen G1 Candidate Recovery",
            "",
            "| n | seed | proposal | mean | p50 | p90 | p95 | max | d_x | d_y | AxisBias | nearest G | Hamming | Kendall | edge Jaccard | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|:--|",
        ]
    )
    for data in grids_data:
        recovery = data["recovery"]
        metrics = recovery["metrics"]
        nearest = recovery["nearest"]
        lines.append(
            f"| {data['n']} | {data['source_seed']} | {data['source_proposal']} | {fmt(metrics.mean)} | {fmt(metrics.p50)} | {fmt(metrics.p90)} | {fmt(metrics.p95)} | {metrics.maximum} | {fmt(metrics.dx)} | {fmt(metrics.dy)} | {fmt(metrics.axis_bias)} | {nearest[0]} | {fmt(nearest[1])} | {fmt(nearest[2])} | {fmt(nearest[3])} | PASS |"
        )

    lines.extend(
        [
            "",
            "## 3. Single-Path Locality And C_dir",
            "",
            "| n | path | mean | p50 | p90 | p95 | max | d_x | d_y | AxisBias | nearest G | Hamming | Kendall | edge Jaccard | C5 |",
            "|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|:--|",
        ]
    )
    for data in grids_data:
        for label in ("G1", "G2", "G3", "G4", "L1", "L2", "L3", "L4"):
            metrics = data["path_metrics"][label]
            nearest = data["nearest"][label]
            lines.append(
                f"| {data['n']} | {label} | {fmt(metrics.mean)} | {fmt(metrics.p50)} | {fmt(metrics.p90)} | {fmt(metrics.p95)} | {metrics.maximum} | {fmt(metrics.dx)} | {fmt(metrics.dy)} | {fmt(metrics.axis_bias)} | {nearest[0]} | {fmt(nearest[1])} | {fmt(nearest[2])} | {fmt(nearest[3])} | {'PASS' if label.startswith('G') or b0.meets_c5(metrics, data['n']) else 'FAIL'} |"
            )
        lines.extend(["", "| n | path | tau_tilde | tau | RIGHT | LEFT | DOWN | UP |", "|---:|:--|---:|---:|---:|---:|---:|---:|"])
        for label in ("G1", "G2", "G3", "G4", "L1", "L2", "L3", "L4"):
            coverage = data["single_cdir"][label]
            for index, threshold in enumerate(b0.THRESHOLDS):
                tau = threshold * (data["n"] * data["n"] - 1)
                lines.append(
                    f"| {data['n']} | {label} | {threshold:.2f} | {fmt(tau)} | {fmt(coverage['RIGHT'][index])} | {fmt(coverage['LEFT'][index])} | {fmt(coverage['DOWN'][index])} | {fmt(coverage['UP'][index])} |"
                )
        lines.append("")

    lines.extend(
        [
            "## 4. Four-Path Collection C_dir",
            "",
            "`P_G={G1,G2,G3,G4}` and `P_L={L1,L2,L3,L4}` use the formal minimum positive forward distance over paths. No AUC is computed.",
            "",
            "| n | tau_tilde | tau | P_G RIGHT | P_G LEFT | P_G DOWN | P_G UP | P_L RIGHT | P_L LEFT | P_L DOWN | P_L UP | Delta RIGHT | Delta LEFT | Delta DOWN | Delta UP |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for data in grids_data:
        pg = data["collection_cdir"]["P_G"]
        pl = data["collection_cdir"]["P_L"]
        for index, threshold in enumerate(b0.THRESHOLDS):
            tau = threshold * (data["n"] * data["n"] - 1)
            values_g = [pg[direction][index] for direction in ("RIGHT", "LEFT", "DOWN", "UP")]
            values_l = [pl[direction][index] for direction in ("RIGHT", "LEFT", "DOWN", "UP")]
            deltas = [left - right for left, right in zip(values_l, values_g)]
            fields = " | ".join(fmt(value) for value in values_g + values_l + deltas)
            lines.append(f"| {data['n']} | {threshold:.2f} | {fmt(tau)} | {fields} |")

    lines.extend(["", "## 5. L-Orbit Structure", ""])
    for data in grids_data:
        lines.extend([f"### n={data['n']}", ""])
        matrices = data["matrices"]
        lines.extend(matrix_markdown("Pairwise Hamming", matrices["hamming"], matrices["labels"]))
        lines.extend(matrix_markdown("Pairwise Normalized Kendall Tau", matrices["kendall"], matrices["labels"]))
        lines.extend(matrix_markdown("Pairwise Sequence-Edge Jaccard", matrices["edge_jaccard"], matrices["labels"]))
        lines.extend(
            [
                "| Corresponding G/L | sequence-edge overlap fraction | sequence-edge Jaccard |",
                "|:--|---:|---:|",
            ]
        )
        for label, values in data["corresponding_overlap"].items():
            lines.append(f"| {label} | {fmt(values[0])} | {fmt(values[1])} |")
        lines.append("")

    lines.extend(["## 6. Path Hashes", "", "| n | path | order SHA-256 | inverse-order SHA-256 |", "|---:|:--|:--|:--|"])
    for data in grids_data:
        for label in ("L1", "L2", "L3", "L4"):
            path = data["json_paths"][label]
            lines.append(f"| {data['n']} | {label} | `{path['order_sha256']}` | `{path['inverse_order_sha256']}` |")
        lines.append(f"| {data['n']} | Q/source L1 | `{data['source_order_sha256']}` | `{data['json_paths']['L1']['inverse_order_sha256']}` |")

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- B0 established that non-G permutations satisfying the original C5 exist.",
            "- This four-path bank is the reversal x transpose orbit of the pre-specified G1-chain candidate, not a post-hoc selection among G2/G3/G4 chains.",
            "- The construction matches reversal/transpose set structure but can retain G1 wide-scale order; n=32 has the B0 G1 candidate's small normalized Kendall distance and must not be described as globally order-randomized.",
            "- C5 matches only mean/p50/p90 within 10 percent. It does not match the full distance distribution, p95/max, AxisBias, polarity, or C_dir.",
            "- Any future P_G versus P_L comparison is only canonical G symmetry orbit versus a preselected three-statistic-locality-matched topology-perturbed symmetry orbit. It cannot by itself attribute all effects to locality or rule out axis, polarity, or coverage.",
            "",
            "## Boundary Declaration",
            "",
            "- Existing B0/model/training/analysis/config files modified: no.",
            "- `outputs/` accessed: no.",
            "- Model, training, inference, or GPU task run: no.",
            "- Commit or push performed: no.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_failure(status: str, error: BaseException) -> None:
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# B1 L Symmetry-Orbit Candidate Extraction",
                "",
                "## Status",
                "",
                f"**{status}**: `{type(error).__name__}: {error}`",
                "",
                "Recovery stopped without changing seed, checkpoint, threshold, proposal budget, or candidate source.",
                "",
                "## Boundary Declaration",
                "",
                "- Existing B0/model/training/analysis/config files modified: no.",
                "- `outputs/` accessed: no.",
                "- Model, training, inference, or GPU task run: no.",
                "- Commit or push performed: no.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        if tuple(int(part) for part in np.__version__.split(".")[:2]) < (1, 22):
            raise RuntimeError("NumPy >= 1.22 is required for method='linear'")
        formal_hash, b0_report_hash = assert_sources()
        grids = {n: b0.make_grid(n) for n in b0.GRID_SIZES}
        block_results = regenerate_block_stability(grids)
        grids_data: list[dict[str, object]] = []
        json_grids = []
        for n in b0.GRID_SIZES:
            q, recovery_metrics, recovery_nearest = recover_q(n, grids[n])
            orbit = make_orbit(q, n)
            orbit_metrics, orbit_pis = validate_orbit_symmetries(orbit, grids[n])
            g_orders = b0.named_g_orders(n)
            g_pis = {label: b0.validate_order(order) for label, order in g_orders.items()}
            g_edge_sets = {label: b0.continuous_edge_set(order) for label, order in g_orders.items()}
            path_orders = {**g_orders, **orbit}
            path_pis = {**g_pis, **orbit_pis}
            path_metrics = {
                label: b0.metrics_from_pi(path_pis[label], grids[n])[0]
                for label in path_orders
            }
            nearest = {
                label: b0.nearest_g_distances(path_orders[label], g_orders, g_pis, g_edge_sets)
                for label in path_orders
            }
            single_coverages = {
                label: single_cdir(path_pis[label], grids[n])
                for label in path_orders
            }
            collection_coverages = {
                "P_G": collection_cdir([g_pis[label] for label in ("G1", "G2", "G3", "G4")], grids[n]),
                "P_L": collection_cdir([orbit_pis[label] for label in ("L1", "L2", "L3", "L4")], grids[n]),
            }
            matrices = pairwise_matrices(orbit)
            corresponding_overlap = {
                f"G{index}/L{index}": sequence_overlap(g_orders[f"G{index}"], orbit[f"L{index}"])
                for index in range(1, 5)
            }
            constructions = {
                "L1": "Q recovered from the frozen B0 G1 chain",
                "L2": "reverse(L1)",
                "L3": "transpose_cells(L1)",
                "L4": "reverse(transpose_cells(L1))",
            }
            json_paths = {
                label: serialise_path(label, constructions[label], orbit[label], orbit_metrics[label])
                for label in ("L1", "L2", "L3", "L4")
            }
            source_hash = order_sha256(q)
            grids_data.append(
                {
                    "n": n,
                    "source_seed": SOURCE_SEEDS[n],
                    "source_proposal": SOURCE_PROPOSAL,
                    "source_order_sha256": source_hash,
                    "recovery": {"metrics": recovery_metrics, "nearest": recovery_nearest},
                    "path_metrics": path_metrics,
                    "nearest": nearest,
                    "single_cdir": single_coverages,
                    "collection_cdir": collection_coverages,
                    "matrices": matrices,
                    "corresponding_overlap": corresponding_overlap,
                    "json_paths": json_paths,
                }
            )
            json_grids.append(
                {
                    "n": n,
                    "source_g": SOURCE_G,
                    "source_seed": SOURCE_SEEDS[n],
                    "source_proposal": SOURCE_PROPOSAL,
                    "source_order_sha256": source_hash,
                    "paths": json_paths,
                }
            )
        payload = {
            "schema_version": "1.0",
            "status": "CANDIDATE_NOT_FROZEN",
            "formal_definition_path": FORMAL_PATH.as_posix(),
            "formal_definition_sha256": formal_hash,
            "source_audit_script": B0_SCRIPT_PATH.as_posix(),
            "source_audit_report": B0_REPORT_PATH.as_posix(),
            "generator_name": "preselected_g1_reversal_transpose_orbit",
            "generator_description": "B0 G1 first post-burn thinning candidate Q, expanded by reversal and cell transpose; not a final approved L generator.",
            "numpy_version": np.__version__,
            "grids": json_grids,
        }
        JSON_PATH.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        json_hash = sha256_file(JSON_PATH)
        REPORT_PATH.write_text(
            render_report(
                time.perf_counter() - wall_started,
                time.process_time() - cpu_started,
                formal_hash,
                b0_report_hash,
                block_results,
                grids_data,
                json_hash,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {JSON_PATH} and {REPORT_PATH}")
    except RecoveryFailure as error:
        write_failure("RECOVERY_FAIL", error)
        raise
    except (AssertionError, RuntimeError, OSError, ValueError) as error:
        write_failure("B1_FAIL", error)
        raise


if __name__ == "__main__":
    main()
