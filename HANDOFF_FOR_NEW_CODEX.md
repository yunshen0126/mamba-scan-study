# mamba_scan_study handoff

Date: 2026-07-09

This archive contains the current `cifar_rowmask` / `mamba_scan_study` project and a compressed record of the research plan, implementation status, experimental results, and next actions.

## Research Question

Q1 existence:

Does one-way image row scanning cause a measurable information-transfer loss for non-row spatial structure, especially vertical information?

Q2 attribution, only if Q1 holds:

If four-direction scanning improves performance, is the gain caused by real direction information, or mainly by multi-path ensemble / extra-parameter effects?

The novelty is the causal isolation. Existing GroupMamba / LocalMamba / 2DMamba-style work generally reports "add directional module -> downstream task improves", but does not cleanly isolate:

- direction information vs ensemble effect;
- scan order vs position encoding;
- source-token visibility vs target-token information transfer.

## Core Design Insight

The synthetic task must be a target-token readout task.

Earlier mean-pooling classification was flawed: the model could inspect the whole image and classify from the source patch directly. That measured "is there a positive/negative patch somewhere", not "did source information arrive at the target token".

Current synthetic carry datasets return:

```python
image, label, target_row, target_col, source_row, source_col
```

For synthetic tasks, the classifier must use only:

```python
feat2d = model.forward_features(images)
logits = model.classify_from_target(feat2d, target_row, target_col)
```

This makes the task test information transfer into the target token.

## Current Implemented Structure

Important files:

- `mamba_scan_study/data/synthetic.py`
  - `VerticalCarry`, `HorizontalCarry`, `DiagonalCarry`
  - signal strength: `line`, `single_patch`, `single_pixel`
  - `amplitude`, `noise_std`
  - `shuffle_source` sanity control
  - returns image, label, target/source coordinates

- `mamba_scan_study/models/scan_utils.py`
  - scan directions: `row`, `col`, `diag`, `anti_diag`
  - flatten and restore helpers

- `mamba_scan_study/models/backbone.py`
  - `ScanBackbone`
  - `MultiDirBackbone`
  - compatibility `RowScanBackbone`
  - target-token classification: `classify_from_target`
  - position modes:
    - `none`
    - `seq_learned`
    - `xy_learned`
    - `xy_sincos`
  - Important: `xy_*` position encodings are defined on 2D coordinates and then flattened according to scan direction.

- `mamba_scan_study/experiments/run_stage0_regression.py`
  - Stage 0 regression runner
  - synthetic uses target-token readout by default
  - real datasets use mean pooling
  - fixed seed and DataLoader generator
  - saves JSON/CSV

- `mamba_scan_study/experiments/run_gap_sweep.py`
  - Stage 1A gap sweep runner
  - implemented; probing and Stage 2 are not implemented
  - supports GRU and Mamba, but skips Mamba if `mamba_ssm` is unavailable
  - saves `results.csv`, `summary.csv`, `results.json`

## Stage 0 Result

The target-token synthetic task now behaves correctly on GRU:

- vertical carry:
  - row scan struggles
  - col scan learns
- source-label shuffle:
  - col scan falls back near random
- mean pooling:
  - marked `q1_eligible=false`

This confirmed the old mean-pooling version was not a clean test, and the new target-token readout exposes the intended directional gap.

## Stage 1A Small Formal GRU Run

Command run from `mamba_scan_study`:

```bash
PYTHONPATH=.. python experiments/run_gap_sweep.py \
  --block-types gru \
  --task-dirs vertical horizontal \
  --branch-dirs row col \
  --grid-sizes 8 16 \
  --signal-strengths single_patch \
  --pos-modes xy_learned \
  --seeds 0 1 2 \
  --epochs 5 \
  --train-samples 2048 \
  --test-samples 512 \
  --amplitude 4.0 \
  --noise-std 0.5 \
  --outdir outputs/stage1_gap_sweep/gru_vh_small
```

Equivalent command from repository root `cifar_rowmask`:

```bash
python -m mamba_scan_study.experiments.run_gap_sweep \
  --block-types gru \
  --task-dirs vertical horizontal \
  --branch-dirs row col \
  --grid-sizes 8 16 \
  --signal-strengths single_patch \
  --pos-modes xy_learned \
  --seeds 0 1 2 \
  --epochs 5 \
  --train-samples 2048 \
  --test-samples 512 \
  --amplitude 4.0 \
  --noise-std 0.5 \
  --outdir mamba_scan_study/outputs/stage1_gap_sweep/gru_vh_small
```

Core GRU results:

| task | grid | row mean | col mean | gap |
|---|---:|---:|---:|---:|
| vertical | 8 | 0.5475 | 1.0000 | col-row = 0.4525 |
| vertical | 16 | 0.5098 | 0.9655 | col-row = 0.4557 |
| horizontal | 8 | 1.0000 | 0.5182 | row-col = 0.4818 |
| horizontal | 16 | 0.9876 | 0.5026 | row-col = 0.4850 |

Interpretation:

- vertical: `col > row`, as expected.
- horizontal: `row > col`, as expected.
- This supports moving to a larger Stage 1A sweep for GRU.
- This is not yet a paper conclusion.

Relevant output files:

- `mamba_scan_study/outputs/stage1_gap_sweep/gru_vh_small/results.csv`
- `mamba_scan_study/outputs/stage1_gap_sweep/gru_vh_small/summary.csv`
- `mamba_scan_study/outputs/stage1_gap_sweep/gru_vh_small/results.json`
- `mamba_scan_study/outputs/stage1_gap_sweep/gru_vh_small_terminal.log`

## Mamba Status

`mamba_ssm` is not installed in the original Mac environment.

Environment that failed:

- macOS arm64
- Python 3.13.5
- PyTorch 2.7.1
- CUDA unavailable
- MPS available

Attempted installs:

```bash
pip install causal-conv1d mamba-ssm
python -m pip install --force-reinstall --no-cache-dir --no-build-isolation causal-conv1d mamba-ssm -i https://pypi.org/simple
python -m pip install --force-reinstall --no-cache-dir --no-build-isolation causal-conv1d==1.2.2.post1 mamba-ssm==1.2.2 -i https://pypi.org/simple
```

Failure reason:

- No available binary wheel for macOS arm64 / Python 3.13.
- Source build requires CUDA `nvcc`.
- Error included:

```text
causal_conv1d was requested, but nvcc was not found
NameError: name 'bare_metal_version' is not defined
```

Mamba logs:

- `mamba_scan_study/outputs/stage1_gap_sweep/mamba_install.log`
- `mamba_scan_study/outputs/stage1_gap_sweep/mamba_install_no_build_isolation.log`
- `mamba_scan_study/outputs/stage1_gap_sweep/mamba_reinstall_current.log`
- `mamba_scan_study/outputs/stage1_gap_sweep/mamba_reinstall_old_versions.log`
- `mamba_scan_study/outputs/stage1_gap_sweep/mamba_vh_small_status.txt`

Practical recommendation:

Run Mamba experiments on Linux with NVIDIA CUDA and `nvcc`. Windows native installation may be difficult. If using Windows, WSL2 + CUDA is more realistic than native Windows Python for `mamba-ssm`.

## Stage 1A Full Sweep To Run Next

Do not implement probing or Stage 2 yet.

Run a larger GRU sweep first:

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

If `mamba_ssm` works on the target machine, run the corresponding Mamba sweep:

```bash
python -m mamba_scan_study.experiments.run_gap_sweep \
  --block-types mamba \
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
  --outdir mamba_scan_study/outputs/stage1_gap_sweep/mamba_full
```

## Later Work, Not Yet Implemented

Stage 1B probing:

- freeze backbone;
- extract token hidden states;
- train linear probes;
- predict source / previous-position information;
- plot decay curves by distance;
- compare row, col, bidirectional, four-direction.

Stage 1C real-data direction ablation:

- CIFAR-10 first;
- Tiny-ImageNet later, because official Tiny-ImageNet validation is not directly ImageFolder-compatible unless preprocessed.

Stage 2 attribution:

- single row baseline;
- real four-direction;
- same-direction ensemble;
- same-direction dropout ensemble;
- shuffled controls;
- full-branch vs channel-split design if aiming to compare to GroupMamba-like architectures.

## Known Caveats

1. `MultiDirBackbone` is currently full-branch, not GroupMamba channel split.
   It is good for isolating direction vs ensemble effects, but do not describe it as GroupMamba-equivalent.

2. Tiny-ImageNet loader may be wrong for raw official validation layout.
   Official validation has `val/images` and `val_annotations.txt`, not class folders. Use CIFAR and synthetic first.

3. `shuffle_order` currently means random scan path where token content and position encoding travel together.
   Later Stage 2 should distinguish content shuffle, position shuffle, and both.

4. The root old `run.py` still has fairness issues if multiple variants run under one seed without resetting before each variant. The new `mamba_scan_study` Stage0/Stage1A runners reset seeds per run.

## Windows Notes

From repository root:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision numpy
python -m mamba_scan_study.experiments.run_gap_sweep --smoke
```

For Mamba:

- Native Windows installation of `mamba-ssm` is often difficult.
- Prefer WSL2 Ubuntu + NVIDIA CUDA toolkit + PyTorch CUDA.
- Verify:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
import mamba_ssm
print("mamba_ssm available")
PY
```

