# Timing Pilot Report

Environment: `2.0.1+cu118`, CUDA `11.8`, GPU `NVIDIA GeForce RTX 3060 Laptop GPU`. Pilot uses row scan, 1 full epochs, micro-batch 8, gradient accumulation 16, AMP=True. No CNN stem was added.

## Measured cells

| dataset | block | grid | L | epoch 1 (min) | epoch 2 (min) | row peak MiB | 4dir peak MiB | 4dir train ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | gru | 8 | 64 | 0.02 | 0.02 | 28 | 34 | 4.02 |
| cifar10 | gru | 16 | 256 | 0.00 | 0.00 | 37 | 59 | 3.01 |
| cifar10 | gru | 32 | 1024 | 0.00 | 0.00 | 148 | 227 | 5.30 |
| cifar10 | mamba | 8 | 64 | 0.01 | 0.01 | 21 | 30 | 3.33 |
| cifar10 | mamba | 16 | 256 | 0.00 | 0.00 | 29 | 61 | 3.28 |
| cifar10 | mamba | 32 | 1024 | 0.00 | 0.00 | 148 | 264 | 2.99 |

## Matrix estimate

The original matrix has 360 runs (2 single-branch + 2 four-branch variants, 5 seeds). Estimated total: **0.8 GPU-hours**, local electricity **CNY 0.0** at 0.60 CNY/kWh.

Adding mandatory `shuffle_row` makes the uncut matrix 450 runs (3 single-branch + 2 four-branch variants, 5 seeds). Estimated total: **0.9 GPU-hours**, local electricity **CNY 0.0**.

Electricity is a hardware-only estimate based on sampled GPU board power. It excludes the rest of the laptop, cooling, failed runs, setup time, and any cloud rental price.
