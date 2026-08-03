"""Regression-check the Main-624 analysis against the 104-run P0-B backup."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from analyze_main624 import analyze, load_records


EXPECTED = {
    "R_low": {
        "1": (3.66, 3.26, 4.06), "2": (0.86, 0.20, 1.52), "3": (-0.09, -0.48, 0.29), "4": (-0.03, -0.67, 0.62), "5": (0.45, -0.21, 1.11),
        "P_G": (0.85, 0.55, 1.15), "P_R": (-0.01, -0.52, 0.50), "P_LMTO": (0.40, -0.18, 0.98),
    },
    "R_high": {
        "1": (11.02, 10.55, 11.49), "2": (4.16, 3.84, 4.47), "3": (0.30, -0.66, 1.27), "4": (2.96, 1.97, 3.96), "5": (1.97, 0.30, 3.65),
        "P_G": (4.23, 4.06, 4.40), "P_R": (0.07, -0.24, 0.39), "P_LMTO": (2.26, 0.44, 4.07),
    },
}
TOLERANCE_PP = 0.01


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_p0b_backup"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summaries, _ = analyze(
        load_records(args.runs_root), ("mamba",), ("cifar10",), augmentation="p0b_legacy"
    )
    observed = summaries[("cifar10", "mamba")]
    failures: list[str] = []
    for reliance, expected_rows in EXPECTED.items():
        for name, expected in expected_rows.items():
            actual = observed[reliance][name]
            actual_values = (actual.mean_pp, actual.lower_pp, actual.upper_pp)
            for label, actual_value, expected_value in zip(("mean", "lower", "upper"), actual_values, expected):
                difference = actual_value - expected_value
                if abs(difference) > TOLERANCE_PP:
                    failures.append(f"{name} {reliance} {label}: actual={actual_value:+.4f}, expected={expected_value:+.4f}, diff={difference:+.4f} pp")
    if failures:
        raise AssertionError("P0-B regression failed:\n" + "\n".join(failures))
    print("P0-B regression passed: 8 metrics x 2 reliances x 3 statistics within +/-0.01 pp")


if __name__ == "__main__":
    main()
