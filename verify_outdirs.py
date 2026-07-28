"""Verify that main-experiment output-directory names are collision-free."""

from __future__ import annotations

import re

from mamba_scan_study.experiments.p0b_data import DATASET_CLASS_COUNTS
from mamba_scan_study.experiments.p0b_path_bank import P0B_EXP_IDS, P0B_GRIDS, P0B_TRAINING_SEEDS
from mamba_scan_study.experiments.run_p0b_feasibility import _run_directory


LEGACY_NAME_PATTERN = re.compile(r"^p0b_(GEO|RND|LOC)\w*_R_(low|high)_seed[0-3]$")


def _main_experiment_directories() -> list[str]:
    directories = [
        _run_directory(exp_id, grid, seed, "formal", None, dataset, "main_uniform", "mamba").name
        for dataset in DATASET_CLASS_COUNTS
        for exp_id in P0B_EXP_IDS
        for grid in P0B_GRIDS
        for seed in P0B_TRAINING_SEEDS
    ]
    directories.extend(
        _run_directory(exp_id, grid, seed, "formal", None, "cifar10", "main_uniform", "gru").name
        for exp_id in P0B_EXP_IDS
        for grid in P0B_GRIDS
        for seed in P0B_TRAINING_SEEDS
    )
    return directories


def main() -> None:
    directories = _main_experiment_directories()
    if len(directories) != 624 or len(set(directories)) != 624:
        raise AssertionError("main-experiment output directories are not a unique set of 624")
    legacy_names = [name for name in directories if LEGACY_NAME_PATTERN.match(name)]
    if legacy_names:
        raise AssertionError(f"main-experiment directories overlap the legacy namespace: {legacy_names}")
    print("\n".join(directories[:10]))
    print("\n".join(directories[-10:]))


if __name__ == "__main__":
    main()
