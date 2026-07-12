# Timing Pilot Report

Environment: `2.0.1+cu118`, CUDA `11.8`, GPU `NVIDIA GeForce RTX 3060 Laptop GPU`. Pilot uses row scan, 2 full epochs, micro-batch 8, gradient accumulation 16, AMP=True. No CNN stem was added.

## Measured cells

| dataset | block | grid | L | epoch 1 (min) | epoch 2 (min) | row peak allocated MiB | 4dir peak allocated MiB | 4dir train ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | gru | 8 | 64 | 0.89 | 0.90 | 28 | 35 | 2.90 |
| cifar10 | gru | 16 | 256 | 1.07 | 1.17 | 37 | 61 | 2.92 |
| cifar10 | gru | 32 | 1024 | 1.95 | 1.98 | 149 | 228 | 3.08 |
| cifar10 | mamba | 8 | 64 | 1.25 | 1.12 | 22 | 32 | 3.29 |
| cifar10 | mamba | 16 | 256 | 1.04 | 0.97 | 30 | 63 | 2.60 |
| cifar10 | mamba | 32 | 1024 | 1.07 | 1.17 | 149 | 266 | 2.82 |
| cifar10_up64 | gru | 8 | 64 | 0.90 | 0.92 | 29 | 37 | 3.46 |
| cifar10_up64 | gru | 16 | 256 | 1.08 | 1.06 | 37 | 62 | 2.78 |
| cifar10_up64 | gru | 32 | 1024 | 1.97 | 2.01 | 70 | 155 | 2.81 |
| cifar10_up64 | mamba | 8 | 64 | 1.15 | 1.32 | 22 | 34 | 2.72 |
| cifar10_up64 | mamba | 16 | 256 | 1.15 | 1.04 | 30 | 64 | 2.48 |
| cifar10_up64 | mamba | 32 | 1024 | 1.15 | 1.07 | 65 | 189 | 3.19 |
| tiny_imagenet | gru | 8 | 64 | 1.54 | 1.58 | 29 | 37 | 2.76 |
| tiny_imagenet | gru | 16 | 256 | 1.93 | 1.89 | 38 | 62 | 2.69 |
| tiny_imagenet | gru | 32 | 1024 | 3.60 | 3.56 | 70 | 155 | 2.84 |
| tiny_imagenet | mamba | 8 | 64 | 1.97 | 1.84 | 22 | 35 | 2.53 |
| tiny_imagenet | mamba | 16 | 256 | 2.06 | 2.00 | 30 | 64 | 2.41 |
| tiny_imagenet | mamba | 32 | 1024 | 2.08 | 2.11 | 65 | 189 | 2.77 |

## Matrix estimate

| matrix | epochs/run | runs | GPU-hours | one-GPU days | electricity CNY |
|---|---:|---:|---:|---:|---:|
| original_360 | 30 | 360 | 532.0 | 22.2 | 11.4 |
| with_shuffle_row_450 | 30 | 450 | 601.4 | 25.1 | 12.8 |
| original_360 | 100 | 360 | 1773.2 | 73.9 | 37.9 |
| with_shuffle_row_450 | 100 | 450 | 2004.8 | 83.5 | 42.8 |

The original matrix has 2 single-branch and 2 four-branch variants. Adding mandatory `shuffle_row` changes this to 3 single-branch and 2 four-branch variants. Both use 5 seeds per cell. CSV files include compute-price scenarios at 1/2/3 CNY per GPU-hour.

Electricity is a hardware-only estimate based on sampled GPU board power. It excludes the rest of the laptop, cooling, failed runs, setup time, and any cloud rental price.
