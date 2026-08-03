"""Render the preregistered Figure 1 forest plot from Main-624 metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from analyze_main624 import (
    AUGMENTATIONS,
    Summary,
    _format_display,
    _validated_group,
    ceiling_rows,
    contrast_summaries,
    load_records,
)


DATASET_ORDER = ("organamnist", "organcmnist", "organsmnist", "cifar10", "eurosat")
DISPLAY_NAMES = {
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
    "cifar10": "CIFAR-10",
    "eurosat": "EuroSAT",
}


def _summary_rows(records, augmentation: str) -> list[tuple[str, Summary | None, bool]]:
    rows: list[tuple[str, Summary | None, bool]] = []
    for dataset in DATASET_ORDER:
        try:
            cells = _validated_group(records, dataset, "mamba", augmentation)
        except ValueError:
            rows.append((dataset, None, False))
            continue
        summary = contrast_summaries(cells)["R_high"]["2"]
        ceiling_flagged = ceiling_rows(cells)["R_high"][1]
        rows.append((dataset, summary, ceiling_flagged))
    return rows


def _x_limits(rows: Sequence[tuple[str, Summary | None, bool]]) -> tuple[float, float]:
    available = [summary for _, summary, _ in rows if summary is not None]
    if not available:
        return -1.0, 1.0
    low = min(summary.lower_pp for summary in available)
    high = max(summary.upper_pp for summary in available)
    if low > 0.0:
        low = 0.0
        return low, high + (high - low) * 0.05
    if high < 0.0:
        high = 0.0
        return low - (high - low) * 0.05, high
    margin = (high - low) * 0.05
    return low - margin, high + margin


def _make_figure(rows: Sequence[tuple[str, Summary | None, bool]]):
    low, high = _x_limits(rows)
    figure, axis = plt.subplots(figsize=(3.4, 0.42 * len(rows) + 0.75), constrained_layout=True)
    y_positions = list(range(len(rows)))
    labels: list[str] = []
    annotations = []
    data_width = high - low
    label_offset = data_width * 0.0125
    for y, (dataset, summary, ceiling_flagged) in zip(y_positions, rows):
        display_name = DISPLAY_NAMES[dataset]
        labels.append(f"{display_name} *" if ceiling_flagged else display_name)
        if summary is None:
            continue
        axis.hlines(y, summary.lower_pp, summary.upper_pp, color="black", linewidth=1.2, zorder=2)
        axis.plot(
            summary.mean_pp,
            y,
            marker="o",
            markersize=5.5,
            markeredgecolor="black",
            markerfacecolor="white" if ceiling_flagged else "black",
            color="black",
            linestyle="None",
            zorder=3,
        )
        annotations.append(
            axis.text(
                summary.upper_pp + label_offset,
                y,
                _format_display(summary.mean_pp, signed=True),
                va="center",
                ha="left",
                fontsize=6.5,
                color="black",
                zorder=4,
            )
        )

    axis.axvline(0.0, color="0.5", linewidth=0.8, zorder=1)
    axis.set_xlim(low, high + data_width * 0.22)
    locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    ticks = [
        round(float(tick), 10)
        for tick in locator.tick_values(low, high)
        if low - 1e-9 <= tick <= high + 1e-9
    ]
    axis.set_xticks(ticks)
    axis.set_yticks(y_positions, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Geometric minus arbitrary\nmulti-path gain (pp)", fontsize=8)
    axis.tick_params(axis="x", labelsize=7)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return figure, axis, annotations


def plot_forest(records, augmentation: str, output: Path) -> None:
    """Plot unrounded R_high contrast-2 summaries supplied by analyze_main624."""
    figure, _, _ = _make_figure(_summary_rows(records, augmentation))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)


def verify_multiline_layout() -> None:
    """Check five in-memory rows for overlapping tick labels or point annotations."""
    rows = [
        (dataset, Summary(1.0 + index, 0.6 + index, 1.4 + index, (1.0, 1.0, 1.0, 1.0)), index == 1)
        for index, dataset in enumerate(DATASET_ORDER)
    ]
    figure, axis, annotations = _make_figure(rows)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_bounds = figure.bbox
    for text_group in (axis.get_yticklabels(), annotations):
        bounds = [text.get_window_extent(renderer) for text in text_group]
        for left, right in zip(bounds, bounds[1:]):
            if left.overlaps(right):
                raise AssertionError("forest-plot row labels overlap")
        for bound in bounds:
            if (
                bound.x0 < figure_bounds.x0
                or bound.x1 > figure_bounds.x1
                or bound.y0 < figure_bounds.y0
                or bound.y1 > figure_bounds.y1
            ):
                raise AssertionError("forest-plot text is clipped outside the figure")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--augmentation", choices=AUGMENTATIONS, default="main_uniform")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("figure1_forest.pdf"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plot_forest(load_records(args.runs_root), args.augmentation, args.output)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
