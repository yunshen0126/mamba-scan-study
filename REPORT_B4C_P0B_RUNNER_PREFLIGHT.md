# B4C P0-B Runner And CPU Preflight

Status: PASS after B4C repair. This stage implements the frozen data helper,
single-cell runner, completed-run validation/skip, deterministic 104-row
ledger, and CPU-only preflight. It does not launch a P0-B performance run.

## Files

Added:

- `mamba_scan_study/experiments/p0b_data.py`
- `mamba_scan_study/experiments/run_p0b_feasibility.py`
- `mamba_scan_study/experiments/run_p0b_preflight.py`
- `mamba_scan_study/analysis/test_p0b_runner_cpu.py`
- `P0B_RUN_LEDGER_104.csv`
- `run_p0b_feasibility.sh`
- `REPORT_B4C_P0B_RUNNER_PREFLIGHT.md`

No existing Stage-1, real-data, Batch C/D launcher, frozen artifact, L/R/split,
or configuration source file was modified.

## Source Gates

| Source | SHA-256 | Result |
| --- | --- | --- |
| `P0B_L_PATH_BANK_FROZEN.json` | `93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7` | PASS |
| `P0B_R_PATH_BANK_FROZEN.json` | `2f7b8a6fd3cfbbae9897b4ef4dc9dcfd1bf7744619d5818ceaca7604d565aee3` | PASS |
| `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` | `e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09` | PASS |
| `docs/P0B_CONFIG_TABLE.md` | `790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e` | PASS |

The runner and data helper verify source bytes before parsing frozen arrays or
constructing a model/loader. Source mismatch fails closed.

## Frozen Training Configuration

| Field | Value |
| --- | --- |
| Dataset / architecture / block | CIFAR-10 / channel-split / Mamba |
| `d_model` / `n_layers` / position mode | 256 / 2 / `xy_learned` |
| Epochs / warmup | 100 / 5 |
| Optimizer / base learning rate | AdamW / 0.001 |
| Weight decay / gradient clipping | 0.05 / 1.0 |
| Effective batch / AMP / workers | 128 / enabled / 4 |
| Formal micro-batch / accumulation steps | 128 / 1 |
| grid8 / grid32 patch size | 4 / 1 |
| Training seeds | 0, 1, 2, 3 |

Formal fields have no ordinary CLI override. Debug mode requires a separate
debug root and cannot write a formal run directory. The launcher only accepts
one explicit design cell or a manifest-only dry run; it never starts all 104
runs by default. The training seed is set before requested-model construction
and reset before loader construction. Formal mode accepts only micro-batch 128;
debug mode may explicitly select a positive divisor of 128. Dry-run reports the
resolved micro-batch and accumulation steps.

## Data And Ledger

`p0b_data.py` uses two `CIFAR10(train=True)` instances only. It validates the
frozen split SHA, image/target hashes, train/validation index hashes, exact
45,000/5,000 sizes, and 4,500/500 class counts. Train sampling uses an explicit
training-seed generator and seeded workers; validation is sequential.

The CPU test uses a temporary mocked CIFAR-10 source and rejects any
`train=False` construction. The real official archive was not read in B4C;
the production helper will validate its raw arrays before loader construction.

`P0B_RUN_LEDGER_104.csv` is code-generated and byte-verified after reread:

```text
rows: 104
SHA-256: 906f6af2f8a695b443b01ac9ff89e29f24b4cea85fb4717252404f58145bfe25
```

Each row contains the unique design key, grid/patch/reliance, path family,
single/diverse status, four path IDs, four order/inverse hashes, Latin rotation,
and all four source hashes. The runner validates both ledger bytes and content
before a formal model is created.

## Checkpoint And Metadata

Completed-run skip resolves requested paths and constructs the requested model
before reading a final checkpoint. It compares protocol, design cell, four
source hashes, path/order/inverse hashes, complete frozen training configuration,
metadata, and permutation buffers before strict state loading. Partial progress
is never skipped or resumed.

Final checkpoints and metadata are atomically written through a temporary file,
flush/close, and rename. The completed marker is written only after both final
artifacts succeed. Metadata is validation-only and includes the required path,
ledger, architecture/operator, nominal-computation-equality, Git, and validation
history fields. No `test_*` metric field exists.

Formal completed checkpoints additionally require a complete 100-row validation
history. Epochs must be exact integers 1 through 100; every row has exactly
`epoch`, `learning_rate`, `train_loss`, `train_accuracy`, `validation_loss`, and
`validation_accuracy`; all values are finite and accuracy lies in `[0, 1]`.
The validator also requires `training_config.epochs=100`,
`micro_batch=128`, and `accum_steps=1` before any strict state load.

## Preflight

The CPU-only preflight passed C1-C5 over the frozen paths and C6 separately
within grid8 and grid32. C2 checks both `d_seq(G1)==d_seq(G2)` and
`d_seq(G3)==d_seq(G4)` edge by edge. C6 uses `n_layers=2` and `block_type="gru"`
without a forward call. Within each grid it compares parameter count, trainable
parameter schema, four groups, two blocks per group, block module schema,
buffer schema, branch dirs, explicit control flow, formal training-plan
signature, micro-batch 128, accumulation steps 1, and the path-independent
nominal-computation equality signature.

No absolute Mamba FLOPs number is reported. grid8 and grid32 are intentionally
not compared for FLOPs equality because their sequence lengths differ. The GRU
structure check does not claim its absolute parameter count equals formal Mamba
parameter count; formal Mamba parameter recording awaits an authorized CUDA
smoke stage.

`run_p0b_feasibility.sh` is a single-cell validation wrapper, not a formal
training launcher. It has no one-command 104-run launch behavior.

## CPU Checks

Command:

```bash
conda run -n mair python -B -m mamba_scan_study.analysis.test_p0b_runner_cpu
conda run -n mair python -B -m mamba_scan_study.experiments.run_p0b_preflight
```

Results: both PASS. The test suite covers split/hash failures, official-test
rejection, ledger integrity, CLI guards, metadata naming, completed/partial
checkpoint behavior, strict-load ordering, atomic writes, runtime path/split
generation guards, C1-C6, and unchanged Stage-1 files. B4C repair tests add
formal micro-batch 128 acceptance, formal 64/1 rejection, debug 64 acceptance,
formal metadata 128/1 checks, corrupted-G4 C2 failure, and history rejection
for empty, 99-row, duplicate-epoch, NaN, Inf, and forbidden `test_acc` cases.

Environment: Python 3.10, NumPy 1.26.4, PyTorch 2.0.1+cu118, torchvision
0.15.2+cu118. CPU test wall time was 109.8 seconds. Preflight wall time was
25.03 seconds; CPU user/system time was 9.58/3.52 seconds.

Git commit: `2f3606d`; worktree is dirty from existing B0-B4B work and this
uncommitted B4C stage. Windows/WSL CRLF handling was checked by normalized
content comparison for the two protected old files; B4C files have no trailing
whitespace.

No existing `outputs/` path was accessed. No formal training, model inference,
GPU operation, Mamba smoke test, commit, or push was performed.
