# Convergence Probe

CIFAR-10 row variant, seed 0, effective batch 128, 60 epochs. All grids within a block use the micro-batch selected by the timing pilot.

| block | grid | micro | best acc | best epoch | plateau status | plateau epoch |
|---|---:|---:|---:|---:|---|---:|
| gru | 8 | 128 | 74.75% | 58 | not_converged_by_probe_end |  |
| gru | 32 | 128 | 71.93% | 47 | plateau_observed | 50 |
| mamba | 8 | 128 | 77.91% | 60 | not_converged_by_probe_end |  |
| mamba | 32 | 64 | 73.53% | 60 | not_converged_by_probe_end |  |

## Epoch recommendation

Status: `extend_probe`. Do not choose a final epoch count yet; extend the probe to **100 epochs**. This is a confirmation target, not a claim that convergence at 100 epochs has already been observed.

## 150-run estimate

| scenario | epochs/run | GPU-hours | one-GPU days |
|---|---:|---:|---:|
| measured lower bound | 60 | 55.6 | 2.3 |
| 100-epoch planning | 100 | 92.7 | 3.9 |

The 60-epoch row is a measured lower bound. The 100-epoch row is a linear planning scenario pending convergence confirmation.

| block | grid | micro | single h/run | four h/run | cell GPU-h |
|---|---:|---:|---:|---:|---:|
| gru | 8 | 128 | 0.19 | 0.37 | 6.5 |
| gru | 16 | 128 | 0.20 | 0.49 | 7.9 |
| gru | 32 | 128 | 0.41 | 1.53 | 21.4 |
| mamba | 8 | 128 | 0.20 | 0.47 | 7.6 |
| mamba | 16 | 128 | 0.28 | 0.79 | 12.2 |
| mamba | 32 | 64 | 0.72 | 2.63 | 37.1 |

100-epoch planning total: **150 runs**, **92.7 GPU-hours** (**3.9 one-GPU days**). Grid16 single-branch timing is estimated from the prior full row/micro8 run scaled by the measured real_4dir batch speedup; grid8 and grid32 single-branch timings come directly from the 60-epoch probes.
