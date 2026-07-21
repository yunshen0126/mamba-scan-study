"""CPU-only structural preflight for frozen P0-B paths and protocol."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from mamba_scan_study.experiments.p0b_path_bank import P0B_EXP_IDS, P0B_GRIDS, iter_p0b_design_cells, resolve_p0b_paths
from mamba_scan_study.experiments.run_p0b_feasibility import (
    FORMAL_CONFIG,
    FORMAL_ACCUM_STEPS,
    FORMAL_MICRO_BATCH,
    LEDGER_FILENAME,
    REPO_ROOT,
    architecture_operator_signature,
    nominal_flops_equality_signature,
    patch_size_for_grid,
    verify_formal_config,
    verify_ledger,
)
from mamba_scan_study.models.backbone import ChannelSplitBackbone


THRESHOLDS = (0.01, 0.05, 0.10, 0.20)


def _order_array(order: torch.Tensor) -> np.ndarray:
    array = np.ascontiguousarray(order.detach().cpu().numpy(), dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("P0-B order must be one-dimensional")
    return array


def _pi(order: torch.Tensor) -> np.ndarray:
    array = _order_array(order)
    positions = np.empty(array.size, dtype=np.int64)
    positions[array] = np.arange(array.size, dtype=np.int64)
    return positions


def _edge_pairs(grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells = np.arange(grid * grid, dtype=np.int64).reshape(grid, grid)
    return cells[:, :-1].ravel(), cells[:, 1:].ravel(), cells[:-1, :].ravel(), cells[1:, :].ravel()


def _undirected_distances(order: torch.Tensor, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = _pi(order)
    horizontal_u, horizontal_v, vertical_u, vertical_v = _edge_pairs(grid)
    horizontal = np.abs(positions[horizontal_u] - positions[horizontal_v])
    vertical = np.abs(positions[vertical_u] - positions[vertical_v])
    return np.concatenate((horizontal, vertical)), horizontal, vertical


def _path_statistics(order: torch.Tensor, grid: int) -> dict[str, float]:
    distances, horizontal, vertical = _undirected_distances(order, grid)
    dx = float(np.mean(horizontal))
    dy = float(np.mean(vertical))
    return {
        "mean": float(np.mean(distances)),
        "p50": float(np.percentile(distances, 50, method="linear")),
        "p90": float(np.percentile(distances, 90, method="linear")),
        "axis_bias": float(math.log(dx / dy)),
    }


def _directional_coverage(orders: tuple[torch.Tensor, ...], grid: int) -> dict[str, tuple[float, ...]]:
    positions = [_pi(order) for order in orders]
    horizontal_u, horizontal_v, vertical_u, vertical_v = _edge_pairs(grid)
    pairs = {
        "RIGHT": (horizontal_u, horizontal_v),
        "LEFT": (horizontal_v, horizontal_u),
        "DOWN": (vertical_u, vertical_v),
        "UP": (vertical_v, vertical_u),
    }
    coverage: dict[str, tuple[float, ...]] = {}
    for direction, (source, target) in pairs.items():
        nearest = np.full(source.size, np.inf)
        for position in positions:
            distance = position[target] - position[source]
            forward = distance > 0
            nearest[forward] = np.minimum(nearest[forward], distance[forward])
        coverage[direction] = tuple(
            float(np.mean(nearest <= threshold * (grid * grid - 1))) for threshold in THRESHOLDS
        )
    return coverage


def _trapezoid(values: list[float], nodes: list[float]) -> float:
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = getattr(np, "trapz", None)
    if implementation is None:
        raise RuntimeError("NumPy has neither trapezoid nor trapz")
    return float(implementation(values, nodes))


def _auc_macro(orders: tuple[torch.Tensor, ...], grid: int) -> tuple[dict[str, float], float]:
    coverage = _directional_coverage(orders, grid)
    auc_by_direction = {
        direction: _trapezoid([0.0, *values], [0.0, *THRESHOLDS]) / 0.20
        for direction, values in coverage.items()
    }
    return auc_by_direction, float(np.mean(list(auc_by_direction.values())))


def check_c2_edgewise(
    grid: int,
    g1_order: torch.Tensor,
    g2_order: torch.Tensor,
    g3_order: torch.Tensor,
    g4_order: torch.Tensor,
) -> None:
    """C2 requires both reversal pairs to preserve every undirected d_seq edge."""
    g1_distances, _, _ = _undirected_distances(g1_order, grid)
    g2_distances, _, _ = _undirected_distances(g2_order, grid)
    g3_distances, _, _ = _undirected_distances(g3_order, grid)
    g4_distances, _, _ = _undirected_distances(g4_order, grid)
    if not np.array_equal(g1_distances, g2_distances):
        raise AssertionError("C2 G1/G2 edgewise d_seq mismatch")
    if not np.array_equal(g3_distances, g4_distances):
        raise AssertionError("C2 G3/G4 edgewise d_seq mismatch")


def check_c1_to_c5() -> dict[str, object]:
    """Validate frozen path legality, geometry regressions, AUC, and LMTO C5."""
    for exp_id, grid, seed in iter_p0b_design_cells():
        resolution = resolve_p0b_paths(exp_id, grid, seed)
        for order in resolution.channel_orders:
            array = _order_array(order)
            if array.size != grid * grid or not np.array_equal(np.sort(array), np.arange(grid * grid)):
                raise AssertionError("C1 path legality failed")

    results: dict[str, object] = {}
    for grid in P0B_GRIDS:
        g_orders = {
            f"G{index}": resolve_p0b_paths(f"GEO_SG{index}", grid, 0).channel_orders[0]
            for index in range(1, 5)
        }
        check_c2_edgewise(
            grid,
            g_orders["G1"],
            g_orders["G2"],
            g_orders["G3"],
            g_orders["G4"],
        )
        statistics = {name: _path_statistics(order, grid) for name, order in g_orders.items()}
        if not math.isclose(statistics["G1"]["axis_bias"], -statistics["G3"]["axis_bias"], abs_tol=1e-12):
            raise AssertionError("C3 G1/G3 AxisBias sign mismatch")
        if not math.isclose(statistics["G2"]["axis_bias"], statistics["G1"]["axis_bias"], abs_tol=1e-12):
            raise AssertionError("C3 G1/G2 AxisBias mismatch")
        if not math.isclose(statistics["G4"]["axis_bias"], statistics["G3"]["axis_bias"], abs_tol=1e-12):
            raise AssertionError("C3 G3/G4 AxisBias mismatch")

        four_auc, four_macro = _auc_macro(tuple(g_orders[f"G{index}"] for index in range(1, 5)), grid)
        two_auc, two_macro = _auc_macro((g_orders["G1"], g_orders["G3"]), grid)
        expected_four = 0.85 if grid == 8 else 0.975
        expected_two = 0.425 if grid == 8 else 0.4875
        if not math.isclose(four_macro, expected_four, abs_tol=1e-12):
            raise AssertionError("C4 four-G AUC regression mismatch")
        if not math.isclose(two_macro, expected_two, abs_tol=1e-12) or not four_macro > two_macro:
            raise AssertionError("C4 polarity-coverage comparison failed")

        lmto_orders = resolve_p0b_paths("LOC_D", grid, 0).channel_orders
        lmto_auc, lmto_macro = _auc_macro(lmto_orders, grid)
        expected_lmto = 0.7991071428571429 if grid == 8 else 0.9645413306451613
        if not math.isclose(lmto_macro, expected_lmto, abs_tol=1e-12):
            raise AssertionError("LMTO AUC regression mismatch")
        target = statistics["G1"]
        for index, order in enumerate(lmto_orders, start=1):
            lmto = _path_statistics(order, grid)
            for key in ("mean", "p50", "p90"):
                if not 0.9 * target[key] <= lmto[key] <= 1.1 * target[key]:
                    raise AssertionError(f"C5 L{index} {key} is outside inclusive +/-10% interval")
        results[f"grid{grid}"] = {
            "four_g_auc": four_auc,
            "two_g_auc": two_auc,
            "lmto_auc": lmto_auc,
            "four_g_macro": four_macro,
            "two_g_macro": two_macro,
            "lmto_macro": lmto_macro,
        }
    return results


def _cpu_structure_model(grid: int, channel_orders: tuple[torch.Tensor, ...]) -> ChannelSplitBackbone:
    return ChannelSplitBackbone(
        img_size=FORMAL_CONFIG.img_size,
        patch_size=patch_size_for_grid(grid),
        in_chans=3,
        d_model=FORMAL_CONFIG.d_model,
        n_layers=2,
        block_type="gru",
        n_classes=10,
        variant="channel_same_row_4",
        pos_mode=FORMAL_CONFIG.pos_mode,
        channel_orders=channel_orders,
    )


def _group_block_schema(model: ChannelSplitBackbone) -> tuple[tuple[tuple[object, ...], ...], ...]:
    return tuple(
        tuple(
            (
                block.__class__.__qualname__,
                tuple((name, tuple(parameter.shape)) for name, parameter in block.named_parameters()),
                tuple((name, tuple(buffer.shape)) for name, buffer in block.named_buffers()),
            )
            for block in group
        )
        for group in model.group_blocks
    )


def _formal_training_plan_signature() -> tuple[tuple[str, object], ...]:
    return (
        ("epochs", FORMAL_CONFIG.epochs),
        ("warmup_epochs", FORMAL_CONFIG.warmup_epochs),
        ("optimizer", FORMAL_CONFIG.optimizer),
        ("base_lr", FORMAL_CONFIG.base_lr),
        ("weight_decay", FORMAL_CONFIG.weight_decay),
        ("grad_clip", FORMAL_CONFIG.grad_clip),
        ("effective_batch", FORMAL_CONFIG.effective_batch),
        ("amp", FORMAL_CONFIG.amp),
        ("micro_batch", FORMAL_MICRO_BATCH),
        ("accum_steps", FORMAL_ACCUM_STEPS),
    )


def check_c6_by_grid() -> dict[str, object]:
    """Check only within-grid equality; grid8 and grid32 intentionally differ."""
    results: dict[str, object] = {}
    for grid in P0B_GRIDS:
        reference = None
        for exp_id in P0B_EXP_IDS:
            resolution = resolve_p0b_paths(exp_id, grid, 0)
            model = _cpu_structure_model(grid, resolution.channel_orders)
            schema = {
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "parameter_schema": [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()],
                "buffer_schema": [(name, tuple(buffer.shape)) for name, buffer in model.named_buffers()],
                "group_count": len(model.group_blocks),
                "blocks_per_group": tuple(len(group) for group in model.group_blocks),
                "group_block_schema": _group_block_schema(model),
                "branch_dirs": tuple(model.branch_dirs),
                "explicit_channel_orders": model.explicit_channel_orders,
                "operator_signature": architecture_operator_signature(grid),
                "formal_training_plan_signature": _formal_training_plan_signature(),
                "nominal_flops_equality_signature": nominal_flops_equality_signature(grid),
            }
            if reference is None:
                reference = schema
            elif schema != reference:
                raise AssertionError(f"C6 equality failed within grid{grid}")
        if reference is None:
            raise AssertionError("missing C6 reference")
        results[f"grid{grid}"] = reference
    return results


def run_preflight(ledger_path: str | Path | None = None) -> dict[str, object]:
    source_sha256 = verify_formal_config()
    ledger = Path(ledger_path) if ledger_path is not None else REPO_ROOT / LEDGER_FILENAME
    ledger_sha256 = verify_ledger(ledger)
    return {
        "source_sha256": source_sha256,
        "ledger_sha256": ledger_sha256,
        "c1_to_c5": check_c1_to_c5(),
        "c6_by_grid": check_c6_by_grid(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only P0-B frozen preflight.")
    parser.add_argument("--ledger", default=str(REPO_ROOT / LEDGER_FILENAME))
    return parser.parse_args(argv)


def main() -> None:
    result = run_preflight(parse_args().ledger)
    print(f"PASS: P0-B sources, ledger {result['ledger_sha256']}, C1-C6 CPU preflight", flush=True)


if __name__ == "__main__":
    main()
