# Mamba Scan Direction Study

This package isolates two questions:

1. Does one-way row scanning cause measurable loss of vertical or non-row information?
2. If four-direction scanning improves accuracy, is the gain from direction information or from a multi-branch ensemble effect?

Current status: Stage 0 implementation only. It adds generic directional scanning, multi-branch backbones, synthetic carry datasets, and a regression runner that saves JSON and CSV outputs.

For synthetic carry tasks, Q1-eligible runs must use target-token readout. Mean pooling is still available as a debugging baseline, but those outputs are marked `q1_eligible=false`.

## Stage 0 Smoke Test

Run from the original repository root:

```bash
python -m mamba_scan_study.experiments.run_stage0_regression --smoke
```

Single-direction examples:

```bash
python -m mamba_scan_study.experiments.run_stage0_regression \
  --dataset synthetic --task-dir vertical --signal-strength single_patch \
  --branch-dirs row --block-type gru --pos-mode xy_learned --epochs 5 --seeds 0

python -m mamba_scan_study.experiments.run_stage0_regression \
  --dataset synthetic --task-dir vertical --signal-strength single_patch \
  --branch-dirs col --block-type gru --pos-mode xy_learned --epochs 5 --seeds 0
```

Four-direction branch example:

```bash
python -m mamba_scan_study.experiments.run_stage0_regression \
  --dataset synthetic --branch-dirs 4dir --epochs 5 --seeds 0
```

Outputs are written to `mamba_scan_study/outputs/stage0/stage0_results.json` and `.csv`.

## Implemented Components

- `models/scan_utils.py`: row, column, diagonal, and anti-diagonal flatten/restore utilities.
- `models/backbone.py`: `ScanBackbone`, compatibility `RowScanBackbone`, `MultiDirBackbone`, target-token classification, and `none`/`seq_learned`/`xy_learned`/`xy_sincos` position modes.
- `data/synthetic.py`: `VerticalCarry`, `HorizontalCarry`, and `DiagonalCarry` generators with `line`, `single_patch`, and `single_pixel` signal strengths. Each sample returns image, label, target coordinates, and source coordinates.
- `data/real_datasets.py`: CIFAR-10 and Tiny-ImageNet loader switches.
- `experiments/run_stage0_regression.py`: minimal training regression script with seed reset, parameter logging, rough FLOPs proxy, JSON and CSV persistence.
