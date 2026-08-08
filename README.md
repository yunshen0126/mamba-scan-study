# Separating path geometry from branch heterogeneity in Vision Mamba

Code, preregistrations and analysis for a controlled study of scan order in
vision state-space models.

**The question.** Multi-direction scanning is credited with improving access to
two-dimensional structure. The comparison behind that credit — one scan path
against `k` paths at fixed capacity — moves two things at once: geometric
complementarity, and the mere fact that the branches differ. This repository
holds an experiment that separates them, by adding the control arm the
literature we reviewed does not run: `k` distinct **arbitrary** permutations,
matched to the geometric set in path count and in capacity, carrying no spatial
structure.

**The headline.** The arbitrary multi-path gain is not resolved from zero in any
of the ten dataset-by-load cells, with point estimates inside `[-0.16, +0.17]`
percentage points. The difference between the two gains is resolved on one
dataset of five, so the registered proposition fails its own decision rule. The
contribution is a missing comparison and a reporting norm, not an effect.

---

## Read this first if you are reviewing the paper

| You want | Go to |
|---|---|
| The registered criteria, fixed before any run | [`MAIN_PREREG_01.md`](MAIN_PREREG_01.md) |
| What we committed to say for each possible outcome | [`MAIN_PREREG_ADDENDUM_03_CONTINGENCY.md`](MAIN_PREREG_ADDENDUM_03_CONTINGENCY.md) |
| The statistical plan | [`P0B_PREREG_ANALYSIS_PLAN.md`](P0B_PREREG_ANALYSIS_PLAN.md) |
| Every erratum found during the study, including the ones that reflect badly on us | [`docs/03_EVIDENCE_LEDGER.md`](docs/03_EVIDENCE_LEDGER.md) |
| External timestamps of the criteria hashes | [`yunshen0126/prereg-timestamps`](https://github.com/yunshen0126/prereg-timestamps) |
| The capacity arm that failed its own registered gate | [`CAP01_RESULTS.md`](CAP01_RESULTS.md) |

**Preregistration is the point of this repository, not a formality.** Criteria,
path banks and validation splits were content-hashed and pushed to an
independent repository before the runs began; GitHub's server-side timestamps on
those pushes are what make "fixed in advance" checkable by someone who does not
trust us. The evidence ledger records where we fell short of our own protocol.

---

## Repository layout

```
mamba_scan_study/
  experiments/          runners and batch launchers with preflight checks
  analysis/             every table and figure in the paper is produced here
  models/  data/        backbone, channel-split apparatus, path handling
docs/
  03_EVIDENCE_LEDGER.md the errata record
  P0B_CONFIG_TABLE.md   the frozen configuration, field by field
  prefill_snapshot/     pre-result snapshots of two documents, with diffs
cap01/                  reports from the capacity arm
*.md                    preregistrations, freeze records and stage reports
P0B_*_FROZEN.json       the frozen path banks and validation splits
```

### Legacy files at the repository root

`config.py`, `data.py`, `masking.py`, `model.py`, `run.py`, `train.py`,
`verify_outdirs.py`, `run_batch_*.sh`, `requirements.txt` and `tools/` belong to
an **earlier, unrelated experiment** on row-masking reconstruction, from which
this repository grew. **They are not used by any result in the paper.** They are
kept rather than deleted because the evidence ledger refers to commits that
contain them and removing them would make that record harder to follow. Nothing
outside `mamba_scan_study/`, `docs/`, `cap01/` and the preregistration documents
is part of this study.

---

## Reproducing the results

Model checkpoints are **not** distributed. What the released metadata archive
supports, and what it does not, is worth stating precisely:

- Every figure, every supplementary table, and the equivalence and capacity-arm
  analyses run from the metadata archive alone.
- `analyze_main624.py`, which emits the main-text tables, additionally verifies
  each run against its completion marker **and** the presence of
  `final_checkpoint.pt`. It will report runs as incomplete if you have only the
  archive. `make_supplementary_tables.py` performs the same statistics without
  that one check — it imports `analyze_main624.py` rather than reimplementing
  it — and is the entry point to use with the archive.

### 1. Get the run metadata

Per-run metadata for all 728 runs — 624 in the main experiment, 104 in the
capacity arm — including the full hundred-epoch history of every run:

> **`seed_level_metadata_v2.tar.gz`** — see Releases. SHA-256 is recorded in
> `docs/03_EVIDENCE_LEDGER.md`.

Unpack it. You should see `outputs_main/`, `outputs_cap512/` and `main_launch/`.

### 2. Regenerate the tables

```bash
# Main results tables, in LaTeX
python mamba_scan_study/analysis/analyze_main624.py \
    --runs-root outputs_main --augmentation main_uniform --emit latex

# Supplementary tables, from metadata alone (no checkpoints needed)
python mamba_scan_study/analysis/make_supplementary_tables.py \
    --runs-root outputs_main --cap01-root outputs_cap512 \
    --analyze mamba_scan_study/analysis/analyze_main624.py \
    --out supplementary_tables.tex
```

`analyze_main624.py` verifies each run against its completion marker and the
SHA-256 of its own metadata, and by default also requires `final_checkpoint.pt`
to be present. Pass `--metadata-only` to relax that check to
`completed.json` + metadata SHA-256, which is what the released archive
supports; no statistical convention changes, and the two invocations agree
byte-for-byte on every emitted table. `make_supplementary_tables.py` likewise
runs from metadata alone and imports `analyze_main624.py` rather than
reimplementing its statistics.

**`--runs-root` must point at `outputs_main`, not at the archive root.** The
capacity arm in `outputs_cap512` shares design-cell keys with the main
experiment and is excluded from every registered judgement by
`PREREG_CAP_01` section 0. Pointing at the archive root aborts with
`duplicate metadata for design cell` rather than silently mixing the two.

### 3. Regenerate the figures

Figures are not stored in this repository. They regenerate from the released
metadata archive alone; no checkpoints are required.

```bash
mkdir -p figures
python mamba_scan_study/analysis/plot_forest.py       --runs-root outputs_main --output figures/figure1_forest.pdf
python mamba_scan_study/analysis/plot_components.py   --runs-root outputs_main --output figures/figure_components.pdf
python mamba_scan_study/analysis/plot_load_gating.py  --runs-root outputs_main --output figures/figure4_load_gating.pdf
python mamba_scan_study/analysis/plot_ceiling.py      --runs-root outputs_main --output figures/figure5_ceiling.pdf
python mamba_scan_study/analysis/plot_paths.py         --grid 8
python mamba_scan_study/analysis/plot_distance_dist.py --grid 32
```

### 4. The checks that will stop you if something is wrong

Several scripts abort rather than emit a wrong number:

- `plot_ceiling.py` cross-checks 70 values against the frozen table and refuses
  to draw on any mismatch.
- `equivalence_PR.py` cross-checks all ten `P_R` intervals against the
  main-text table.
- `cap01_judge.py --selftest` reproduces 24 published values at
  `d_model = 256`, bit for bit, before it will judge the capacity arm.
- `plot_distance_dist.py` verifies eight path-bank statistics against
  `P0B_PREREG_FREEZE_L_AUC.md`.
- `inertness_check_16.py` compares sixteen cells against archived values from an
  earlier configuration.

If one of these fails on your machine, the discrepancy is real and we would like
to hear about it.

---

## Environment

The runs were executed on a single NVIDIA GeForce RTX 4090, driver 580.105.08,
Linux 5.15.0, PyTorch 2.0.1 with CUDA 11.8. A full package lock
(`requirements-lock.txt`) and the environment record (`env.txt`) are in Releases
alongside the metadata archive.

Analysis and plotting need only Python, NumPy and Matplotlib; no GPU.

---

## What this study does not establish

Stated here because the paper states it and a repository should not read more
confidently than its paper.

- The apparatus keeps four scan paths in **disjoint channel groups**. This
  isolates path identity, but it is not the fused multi-directional block of
  standard Vision Mamba, and every result is conditional on that architecture.
- All runs are 32×32 classification. The disagreement that motivates the study
  is in segmentation, where locality enters the loss at every token.
- The central quantity is identified as a **path-family by diversity
  interaction**. Calling it geometry-specific adds an assumption this design does
  not test: arbitrary and geometric paths also differ in locality, in axis bias
  and in single-path accuracy.
- Four seeds per cell move training randomness, the representative single path
  and the channel assignment together, so the intervals do not support
  generalisation over path draws.
- A capacity arm at `d_model = 512` failed its own registered gate and returned
  no measurement. The objection that the null on the arbitrary gain reflects
  insufficient width therefore remains open.

---

## Licence and archive

Source code is released under the MIT licence; see [`LICENSE`](LICENSE). The
preregistration documents, the evidence ledger and the per-run metadata archive
are released under CC BY 4.0. The analysed datasets belong to their providers
and are not redistributed here.

The submission archive is deposited on Zenodo with a version DOI, so that the
record cited in the paper does not change as this branch does:

> **DOI: [10.5281/zenodo.XXXXXXX]** — fill in after the first Zenodo release

## Citation

```bibtex
@article{tian2026separating,
  author  = {Tian, Zhongyu and Jin, Guozhe},
  title   = {Separating path geometry from branch heterogeneity in Vision Mamba:
             a matched arbitrary-permutation control},
  journal = {under review},
  year    = {2026}
}
```

## Contact

Zhongyu Tian — <yunshen0126@outlook.com>
Guozhe Jin — <jinguozhe@ybu.edu.cn>
Department of Artificial Intelligence, School of Engineering, Yanbian University
