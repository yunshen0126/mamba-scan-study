# Stage1C CIFAR-10 Direction Ablation Summary

Date: 2026-07-10

## Files

- `results.csv`
- `summary.csv`
- `results.json`
- `terminal.log`

## Completion Check

- Total result rows: 24
- Skipped rows: 0
- Blocks: `gru`, `mamba`
- Variants: `row`, `col`, `real_4dir`, `same_row_4`
- Seeds for every block/variant: `0, 1, 2`

## Main Results

Accuracy is reported as mean percent plus population standard deviation over seeds 0, 1, and 2.

| block | row | col | real_4dir | same_row_4 |
|---|---:|---:|---:|---:|
| GRU | 71.91 ± 0.31 | 71.65 ± 0.17 | 75.22 ± 0.21 | 75.20 ± 0.47 |
| Mamba | 75.77 ± 0.15 | 74.76 ± 0.19 | 78.49 ± 0.54 | 77.85 ± 0.24 |

## Deltas

| block | real_4dir vs row | real_4dir vs same_row_4 | row vs col |
|---|---:|---:|---:|
| GRU | +3.32 pp | +0.03 pp | +0.25 pp |
| Mamba | +2.72 pp | +0.64 pp | +1.01 pp |

## Interpretation

GRU and Mamba both ran successfully.

The four-branch variants are clearly better than the single-branch `row` baseline. However, `real_4dir` is not clearly better than `same_row_4`: the GRU difference is only +0.03 percentage points, and the Mamba difference is +0.64 percentage points. Therefore the CIFAR-10 result should not be written as strong evidence that direction information itself is effective.

The safer conclusion is that four-branch models improve CIFAR-10 accuracy, but the improvement likely includes multi-branch capacity, parameter count, and ensemble-like effects. The current Stage1C downstream classification result does not cleanly isolate directional information as the cause.

Single-direction `row` and `col` are close on CIFAR-10. The gap is about 0.25 percentage points for GRU and 1.01 percentage points for Mamba, which suggests the downstream classification task may not strongly use the information-transfer asymmetry observed in the synthetic target-token tasks.

## Caveats

- `real_4dir` and `same_row_4` are full-branch four-path variants.
- They are not GroupMamba-style channel split models.
- Mamba used the `mamba_ssm` unfused path because the current fast path is incompatible with the installed `causal_conv1d` signature.
- Model depth, width, scan directions, readout mode, and experiment design were not changed for this run.
- Stage2 attribution was not started.
