# Multi-path scan gains in Vision Mamba

Code, frozen path banks, per-run records and analysis scripts for the study of
what a multi-path scan buys in a vision state-space model at fixed capacity.

A state-space model applied to an image has to serialise the patch grid, and the
ordering costs nothing: parameter count and nominal arithmetic are unchanged.
Multi-direction scanning is justified in the literature by the claim that one
direction cannot reach structure lying across the other spatial axis, and the
evidence offered for it compares `k` copies of one path against `k` distinct
geometric paths. That comparison changes the diversity of the paths and the
spatial axes they cover at the same time. This repository contains the
comparison that separates them: `k` distinct **arbitrary** permutations, matched
in path count, parameters and nominal arithmetic.

---

## What is here

```
P0B_R_PATH_BANK_FROZEN.json     three frozen banks of four arbitrary permutations
P0B_L_PATH_BANK_FROZEN.json     the locality-matched orbit L1..L4
P0B_EUROSAT_SPLIT_FROZEN.json   the frozen EuroSAT validation split
P0B_RUN_LEDGER_104.csv          condition-to-path assignment for the main design

mamba_scan_study/               training pipeline for the main design
run_fused_scan.py               the fused-block arm, standalone
launch_fused.sh                 launcher for the fused arm

scan_geometry.py                locality and axis-spread measures for a path set
export_all_results.py           all three batches -> one tidy CSV
export_main_cells.py            the ten cells of the main design
compute_2x2.py                  P_G, P_R, P_L and the interaction, per cell
make_fig_28.py                  Fig. 2 of the Letter
mkbbl.py                        refs.bib -> a static IEEE-style bibliography

results/results_all.csv         one row per run: the table every number comes from
```

---

## The three batches

They are analysed separately and never pooled. The differences are the reason.

| | main design | earlier batch | fused arm |
|---|---|---|---|
| runs | 520 (+104 recurrent) | 360 | 40 |
| module | channel-split, four 64-d branches | channel-split | fused, one shared 256-d operator |
| combination | concatenate before the classifier | concatenate | sum inside the block |
| datasets | CIFAR-10, three MedMNIST organ planes, EuroSAT | CIFAR-10, CIFAR-100 | CIFAR-10, CIFAR-100 |
| granularities | L = 64, 1024 | L = 64, 256, 1024 | L = 1024 |
| widths | d = 256 | d = 64, 256 | d = 256 |
| operators | Mamba (+ a GRU arm on CIFAR-10) | Mamba, GRU | Mamba |
| seeds | 4 | 5 | 4 |
| augmentation | none | random crop and horizontal flip | none |
| endpoint | frozen validation split, mean of epochs 80–100 | CIFAR test split, mean of the final 20 | frozen validation split, mean of the final 20 |
| arbitrary paths | the frozen banks above | drawn per run from a run seed | the frozen banks above |
| precision | fp16 mixed | fp16 mixed | bf16 mixed |

The fused arm uses bf16 because the four scans are summed inside the block
before being averaged, and the intermediate sum overflows fp16 on some seeds.
All forty of its runs use the same setting.

---

## Conditions

Within each path family, the repeated condition places one path in all four scan
slots and the distinct condition places four different paths in one model.
Parameter count and nominal arithmetic are identical across every condition
compared.

| group | main design | fused arm | paths |
|---|---|---|---|
| `GEO_S` | `GEO_SG1`–`GEO_SG4` | `GEO_SG1` | one canonical raster, four copies |
| `GEO_DIV` | `GEO_DIV` | `GEO_DIV` | G1 G2 G3 G4 |
| `RND_S` | `RND_S1`–`RND_S3` | `RND_S1` | one arbitrary permutation, four copies |
| `RND_D` | `RND_D1`–`RND_D3` | `RND_D1` | four distinct arbitrary permutations |
| `LOC_S` | `LOC_S` | `LOC_S1` | one locality-matched path, four copies |
| `LOC_D` | `LOC_D` | `LOC_D1` | L1 L2 L3 L4 |

The quantities reported are, per seed and then averaged:

```
P_F  = accuracy(F_distinct) - accuracy(F_repeated)     for F in {G, R, L}
D    = P_G - P_R                                        the interaction
S    = accuracy(G_repeated) - accuracy(R_repeated)      single-path structure
```

`P_R` is measured inside a family carrying no canonical spatial order, so a gain
there is attributable to the paths differing and to nothing else.

---

## Reproducing the numbers

Every table and figure in the Letter is regenerated from the released records
with no GPU time.

```bash
# one row per run, across all three batches
python3 export_all_results.py \
    --main    <path>/outputs_main \
    --earlier <path>/outputs \
    --fused   <path>/outputs_fused \
    --out results/results_all.csv

# Table I: the ten cells of the main design
python3 export_main_cells.py <path>/outputs_main --tail 21

# Table II and the fused arm
python3 compute_2x2.py <path>/outputs   --tail 20    # earlier batch
python3 compute_2x2.py <path>/outputs_fused --tail 20

# Fig. 2
python3 make_fig_28.py

# geometry of the three path families
python3 -c "
from scan_geometry import gen_G, gen_R, describe, adjacent_pairs
p = adjacent_pairs(32)
print(describe('G', gen_G(32), 32, p))
print(describe('R', gen_R(32, 17071), 32, p))"
```

The endpoint window differs between batches: the main design terminates its
schedule at zero, so epochs 99 and 100 are identical and the window is epochs
80–100; the other two batches use the final twenty epochs. `--tail 21` and
`--tail 20` reproduce the published values exactly.

---

## Running the fused arm

```bash
bash launch_fused.sh selftest   # index round trip, fusion equivalence, capacity match
bash launch_fused.sh gpu        # interpreter, torch build, visible devices
bash launch_fused.sh one        # ten epochs on the hardest condition
bash launch_fused.sh all        # the forty runs
```

`selftest` checks four things before any GPU time is spent: that every path and
its inverse satisfy `pi[order] == arange(N)`; that gather followed by scatter is
the identity in all four slots; that four copies of one path, averaged, reproduce
a single scan along that path exactly; and that parameter count does not depend
on how many of the four paths are distinct. All four must pass.

A run that has not learned by epoch 8 is aborted rather than written out, and a
non-finite loss stops the run immediately.

---

## Path representation

Every permutation is stored as an index vector `order`, where `order[t]` is the
row-major index of the cell visited at step `t`, together with its inverse `pi`,
where `pi[u]` is the step at which cell `u` is visited. The two satisfy

```
pi[order] == arange(N)
```

and this is asserted wherever a path is constructed or loaded. Getting the
inverse wrong is the failure mode this guards against: a scan gathers along
`order` and scatters along `pi`, and swapping them silently produces a different
model rather than an error.

---

## License and contact

MIT. Questions to yunshen0126@outlook.com.
