You are taking over my `mamba_scan_study` project.

Do not jump to Stage 2. The current goal is still Stage 1A: determine whether directional scan order produces measurable target-token information-transfer gaps in synthetic carry tasks.

Start by reading:

- `HANDOFF_FOR_NEW_CODEX.md`
- `mamba_scan_study/experiments/run_gap_sweep.py`
- `mamba_scan_study/data/synthetic.py`
- `mamba_scan_study/models/backbone.py`

Important constraints:

1. Synthetic carry tasks must use target-token readout, not mean pooling.
2. Do not change the model structure unless explicitly asked.
3. Keep seed fairness: reset seed for every run and use a fixed DataLoader generator.
4. Save raw CSV/JSON results. Do not only print terminal output.
5. If `mamba_ssm` is not installed, try installing it only if the machine supports CUDA / nvcc. If installation fails, record the error and continue GRU analysis.
6. Do not implement probing or Stage 2 until Stage 1A full sweep is stable.

First task:

Run the full GRU Stage 1A sweep:

```bash
python -m mamba_scan_study.experiments.run_gap_sweep \
  --block-types gru \
  --task-dirs vertical horizontal diagonal \
  --branch-dirs row col diag anti_diag \
  --grid-sizes 8 16 24 32 \
  --signal-strengths single_patch single_pixel \
  --pos-modes none xy_learned \
  --seeds 0 1 2 \
  --epochs 5 \
  --train-samples 4096 \
  --test-samples 1024 \
  --amplitude 4.0 \
  --noise-std 0.5 \
  --outdir mamba_scan_study/outputs/stage1_gap_sweep/gru_full
```

Then summarize:

- vertical: `gap_col_minus_row`
- horizontal: `gap_row_minus_col`
- diagonal: `gap_diag_minus_row` and `gap_diag_minus_col`
- stability across grid size, signal strength, position mode, and seeds.

