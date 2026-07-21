# B0 Locality-Control Feasibility Audit

## Scope And Frozen Definitions

- Formal source read before execution: `C:\Users\DELL\Desktop\一些md\P0A_B0_FORMAL_DEFINITIONS.md`.
- CPU-only script: Python standard library plus NumPy; no model, checkpoint, data loader, GPU, training, or inference imports.
- `order[t]` is the row-major cell visited at step `t`; `pi[u]` is its visit step. Every path passed `pi[order] = arange(N)` and `pi = argsort(order)`.
- All locality statistics use all undirected horizontal and vertical four-neighbor edges and `d_seq=abs(pi[u]-pi[v])`.
- All percentiles call `numpy.percentile(..., method="linear")`. The incremental Fenwick calculation uses the identical zero-based linear interpolation rule.
- B0 reports the four frozen `C_dir` nodes only. No AUC is defined or reported.

## Execution

- Command: `python tools/audit_p0a_locality_control.py`
- Python: `3.13.11`; NumPy: `2.4.1`; platform: `Windows-11-10.0.26200-SP0`.
- Actual elapsed wall time: `868.156` s.
- Block audit: 20000 samples for each (n,b), with independent frozen seeds.
- Constraint walk: 1,000,000 proposals per chain, burn-in 100,000, thinning 25,000; all counts are proposals, not accepted moves.
- Proposal selection is exactly 1/3 each for adjacent sequence positions, uniformly sampled unordered local sequence-position pairs with distance `1..n`, and uniformly sampled ordered global distinct-position pairs. Same-position proposals are prohibited.

## 1. G Exact Regression

| n | G | mean | p50 | p90 | p95 | max | d_x | d_y | AxisBias | Status |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 8 | G1 | 4.500000 | 4.500000 | 8.000000 | 8.000000 | 8 | 1.000000 | 8.000000 | -2.079442 | PASS |
| 8 | G2 | 4.500000 | 4.500000 | 8.000000 | 8.000000 | 8 | 1.000000 | 8.000000 | -2.079442 | PASS |
| 8 | G3 | 4.500000 | 4.500000 | 8.000000 | 8.000000 | 8 | 8.000000 | 1.000000 | 2.079442 | PASS |
| 8 | G4 | 4.500000 | 4.500000 | 8.000000 | 8.000000 | 8 | 8.000000 | 1.000000 | 2.079442 | PASS |
| 32 | G1 | 16.500000 | 16.500000 | 32.000000 | 32.000000 | 32 | 1.000000 | 32.000000 | -3.465736 | PASS |
| 32 | G2 | 16.500000 | 16.500000 | 32.000000 | 32.000000 | 32 | 1.000000 | 32.000000 | -3.465736 | PASS |
| 32 | G3 | 16.500000 | 16.500000 | 32.000000 | 32.000000 | 32 | 32.000000 | 1.000000 | 3.465736 | PASS |
| 32 | G4 | 16.500000 | 16.500000 | 32.000000 | 32.000000 | 32 | 32.000000 | 1.000000 | 3.465736 | PASS |

All G paths are legal. For both grids, all four paths equal the frozen mean/p50/p90 regression values; G1/G2 have identical edgewise `d_seq`; G1/G3 have identical aggregate distributions and opposite AxisBias. Status: **PASS**.

## 2. Old Blocked Serpentine Audit

`b` is blocks per axis, total blocks are `b^2`, block side is `m=n/b`, and every block order has length `m^2=(n/b)^2`. Every (n,b) passed the no-duplicate/no-omission, intra-block four-neighbor continuity, and eight-unique-orientation assertions.

| n | b | m | seed | samples | mean [min,p5,p50,p95,max] | p50 [min,p5,p50,p95,max] | p90 [min,p5,p50,p95,max] | C5 count | C5 rate | p50=1 count | p50=1 rate | closest sample | closest metrics | Status |
|---:|---:|---:|---:|---:|:--|:--|:--|---:|---:|---:|---:|---:|:--|:--|
| 8 | 2 | 4 | 2026072001 | 20000 | 4.714286, 5.142857, 6.000000, 7.142857, 7.571429 | 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 | 7.000000, 7.000000, 16.900000, 30.100000, 38.200000 | 0 | 0.000000 | 20000 | 1.000000 | 450 | mean=4.714286, p50=1.000000, p90=8.800000 | PASS |
| 8 | 4 | 2 | 2026072002 | 20000 | 6.678571, 8.839286, 10.571429, 12.357143, 14.982143 | 3.000000, 3.000000, 3.000000, 3.000000, 3.000000 | 17.000000, 26.000000, 33.800000, 40.800000, 49.000000 | 0 | 0.000000 | 0 | 0.000000 | 10479 | mean=6.946429, p50=3.000000, p90=17.800000 | PASS |
| 32 | 2 | 16 | 2026072003 | 20000 | 16.741935, 18.677419, 22.548387, 26.677419, 28.612903 | 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 | 27.000000, 27.000000, 27.000000, 27.000000, 27.000000 | 0 | 0.000000 | 20000 | 1.000000 | 39 | mean=16.741935, p50=1.000000, p90=27.000000 | PASS |
| 32 | 4 | 8 | 2026072004 | 20000 | 24.854839, 32.822581, 39.112903, 45.580645, 56.258065 | 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 | 15.000000, 15.000000, 15.000000, 15.000000, 15.000000 | 0 | 0.000000 | 20000 | 1.000000 | 4487 | mean=24.854839, p50=1.000000, p90=15.000000 | PASS |

This is a diagnostic of the old candidate family only. C5 was not relaxed, and no sampled path is promoted to a final L generator.

## 3. General Constrained-Permutation Existence Audit

Each chain begins at its named G path. A proposal is accepted only when its current mean, p50, and p90 remain within the frozen C5 intervals. The first post-burn thinning checkpoint that is C5-valid and differs from all four G permutations is retained; all chains continue to the fixed 1,000,000-proposal budget. No later state replaces a retained candidate based on distance.

| n | source | seed | adjacent/local/global proposals | accepted | accepted after burn | acceptance rate | full checks | first candidate proposal | candidate C5 | nearest G (Hamming rule) | Hamming | norm. Kendall tau | edge Jaccard | Status |
|---:|:--|---:|:--|---:|---:|---:|---:|---:|:--|:--|---:|---:|---:|:--|
| 8 | G1 | 2026072101 | 333406/333947/332647 | 105429 | 94166 | 0.105429 | 103 | 125000 | PASS | G1 | 0.781250 | 0.047123 | 0.235294 | PASS |
| 8 | G2 | 2026072102 | 333145/333358/333497 | 109805 | 99103 | 0.109805 | 103 | 125000 | PASS | G2 | 0.937500 | 0.071429 | 0.200000 | PASS |
| 8 | G3 | 2026072103 | 332463/333277/334260 | 108758 | 98166 | 0.108758 | 103 | 125000 | PASS | G3 | 0.734375 | 0.039683 | 0.211538 | PASS |
| 8 | G4 | 2026072104 | 332856/332888/334256 | 107255 | 96454 | 0.107255 | 103 | 125000 | PASS | G4 | 0.843750 | 0.081349 | 0.211538 | PASS |
| 32 | G1 | 2026072201 | 333127/333591/333282 | 288537 | 259598 | 0.288537 | 103 | 125000 | PASS | G1 | 0.919922 | 0.006411 | 0.111957 | PASS |
| 32 | G2 | 2026072202 | 333217/333063/333720 | 287074 | 258373 | 0.287074 | 103 | 125000 | PASS | G2 | 0.924805 | 0.006547 | 0.113166 | PASS |
| 32 | G3 | 2026072203 | 333007/334236/332757 | 290478 | 261347 | 0.290478 | 103 | 125000 | PASS | G3 | 0.922852 | 0.007161 | 0.108342 | PASS |
| 32 | G4 | 2026072204 | 332958/333172/333870 | 289515 | 261160 | 0.289515 | 103 | 125000 | PASS | G4 | 0.910156 | 0.006636 | 0.110749 | PASS |

Candidates found: n=8: 4/4; n=32: 4/4. **EXISTENCE_PASS**.

`EXISTENCE_PASS` only means that the frozen budget found four non-G permutations satisfying C5 at each scale. It does not approve this random walk as the final L generator and does not establish that the candidates are sufficiently independent of G.

### Candidate Locality And Directional Coverage

| n | source | mean | p50 | p90 | p95 | max | d_x | d_y | AxisBias | C5 |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 8 | G1 | 4.946429 | 4.500000 | 8.000000 | 12.000000 | 17 | 2.553571 | 7.339286 | -1.055749 | PASS |
| 8 | G2 | 4.928571 | 4.500000 | 8.000000 | 13.900000 | 21 | 2.857143 | 7.000000 | -0.896088 | PASS |
| 8 | G3 | 4.937500 | 4.500000 | 8.000000 | 11.000000 | 19 | 7.446429 | 2.428571 | 1.120431 | PASS |
| 8 | G4 | 4.946429 | 4.500000 | 8.000000 | 14.900000 | 19 | 6.821429 | 3.071429 | 0.797926 | PASS |
| 32 | G1 | 18.140625 | 15.000000 | 35.000000 | 41.000000 | 66 | 4.760081 | 31.521169 | -1.890395 | PASS |
| 32 | G2 | 18.144657 | 15.000000 | 35.000000 | 42.000000 | 76 | 4.763105 | 31.526210 | -1.889920 | PASS |
| 32 | G3 | 18.146673 | 15.000000 | 35.000000 | 43.000000 | 77 | 31.451613 | 4.841734 | 1.871177 | PASS |
| 32 | G4 | 18.148690 | 15.000000 | 35.000000 | 42.000000 | 81 | 31.443548 | 4.853831 | 1.868426 | PASS |

| n | source | tau_tilde | tau=tau_tilde*(N-1) | RIGHT | LEFT | DOWN | UP |
|---:|:--|---:|---:|---:|---:|---:|---:|
| 8 | G1 | 0.01 | 0.630000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | G1 | 0.05 | 3.150000 | 0.589286 | 0.160714 | 0.089286 | 0.035714 |
| 8 | G1 | 0.10 | 6.300000 | 0.785714 | 0.160714 | 0.321429 | 0.035714 |
| 8 | G1 | 0.20 | 12.600000 | 0.821429 | 0.178571 | 0.875000 | 0.035714 |
| 8 | G2 | 0.01 | 0.630000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | G2 | 0.05 | 3.150000 | 0.303571 | 0.428571 | 0.000000 | 0.196429 |
| 8 | G2 | 0.10 | 6.300000 | 0.321429 | 0.642857 | 0.000000 | 0.482143 |
| 8 | G2 | 0.20 | 12.600000 | 0.321429 | 0.678571 | 0.000000 | 0.857143 |
| 8 | G3 | 0.01 | 0.630000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | G3 | 0.05 | 3.150000 | 0.053571 | 0.035714 | 0.589286 | 0.232143 |
| 8 | G3 | 0.10 | 6.300000 | 0.321429 | 0.035714 | 0.732143 | 0.250000 |
| 8 | G3 | 0.20 | 12.600000 | 0.892857 | 0.035714 | 0.750000 | 0.250000 |
| 8 | G4 | 0.01 | 0.630000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | G4 | 0.05 | 3.150000 | 0.071429 | 0.160714 | 0.196429 | 0.410714 |
| 8 | G4 | 0.10 | 6.300000 | 0.071429 | 0.410714 | 0.214286 | 0.678571 |
| 8 | G4 | 0.20 | 12.600000 | 0.071429 | 0.785714 | 0.214286 | 0.785714 |
| 32 | G1 | 0.01 | 10.230000 | 0.496976 | 0.416331 | 0.015121 | 0.002016 |
| 32 | G1 | 0.05 | 51.150000 | 0.554435 | 0.445565 | 0.976815 | 0.002016 |
| 32 | G1 | 0.10 | 102.300000 | 0.554435 | 0.445565 | 0.997984 | 0.002016 |
| 32 | G1 | 0.20 | 204.600000 | 0.554435 | 0.445565 | 0.997984 | 0.002016 |
| 32 | G2 | 0.01 | 10.230000 | 0.385081 | 0.522177 | 0.002016 | 0.027218 |
| 32 | G2 | 0.05 | 51.150000 | 0.420363 | 0.579637 | 0.002016 | 0.962702 |
| 32 | G2 | 0.10 | 102.300000 | 0.420363 | 0.579637 | 0.002016 | 0.997984 |
| 32 | G2 | 0.20 | 204.600000 | 0.420363 | 0.579637 | 0.002016 | 0.997984 |
| 32 | G3 | 0.01 | 10.230000 | 0.026210 | 0.003024 | 0.514113 | 0.378024 |
| 32 | G3 | 0.05 | 51.150000 | 0.954637 | 0.003024 | 0.579637 | 0.420363 |
| 32 | G3 | 0.10 | 102.300000 | 0.996976 | 0.003024 | 0.579637 | 0.420363 |
| 32 | G3 | 0.20 | 204.600000 | 0.996976 | 0.003024 | 0.579637 | 0.420363 |
| 32 | G4 | 0.01 | 10.230000 | 0.003024 | 0.023185 | 0.379032 | 0.529234 |
| 32 | G4 | 0.05 | 51.150000 | 0.003024 | 0.966734 | 0.411290 | 0.588710 |
| 32 | G4 | 0.10 | 102.300000 | 0.003024 | 0.996976 | 0.411290 | 0.588710 |
| 32 | G4 | 0.20 | 204.600000 | 0.003024 | 0.996976 | 0.411290 | 0.588710 |

### Post-Burn Chain Diagnostics

Nearest G is pre-defined as the reference with minimum order-position Hamming distance, with G-label tie breaking. At every fixed thinning checkpoint the report records Hamming, normalized Kendall tau, and continuous-sequence-edge Jaccard against that reference; the table gives their observed ranges. No 'local neighborhood' distance threshold is frozen, so no binary long-term-residence conclusion is drawn from these diagnostics.

| n | source | checkpoints | nearest G labels observed | Hamming range | norm. Kendall tau range | edge Jaccard range | accepted-after-burn range | Status |
|---:|:--|---:|:--|:--|:--|:--|:--|:--|
| 8 | G1 | 36 | G1,G3 | [0.781250, 0.953125] | [0.047123, 0.343254] | [0.024390, 0.285714] | [2956, 94166] | PASS |
| 8 | G2 | 36 | G1,G2 | [0.781250, 0.968750] | [0.062996, 0.921131] | [0.095652, 0.223301] | [2712, 99103] | PASS |
| 8 | G3 | 36 | G3,G4 | [0.734375, 0.968750] | [0.039683, 0.928075] | [0.145455, 0.298969] | [2540, 98166] | PASS |
| 8 | G4 | 36 | G4 | [0.781250, 0.953125] | [0.059028, 0.107143] | [0.125000, 0.260000] | [2395, 96454] | PASS |
| 32 | G1 | 36 | G1 | [0.918945, 0.964844] | [0.006411, 0.013596] | [0.085987, 0.120482] | [7419, 259598] | PASS |
| 32 | G2 | 36 | G2 | [0.913086, 0.958008] | [0.006547, 0.012723] | [0.085987, 0.124794] | [7092, 258373] | PASS |
| 32 | G3 | 36 | G3 | [0.916016, 0.964844] | [0.007161, 0.013601] | [0.079683, 0.114379] | [7334, 261347] | PASS |
| 32 | G4 | 36 | G4 | [0.910156, 0.969727] | [0.006636, 0.013771] | [0.083686, 0.116203] | [7197, 261160] | PASS |

## 4. Final-Generator Candidate Comparison (Not Implemented Or Selected)

| Candidate family | Reproducible | Risk of implicit G bias | Multiple instances | Needs new frozen distance threshold | CPU cost | Changes C5 | Decision |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Constrained random walk used here | Yes, with frozen proposals/seeds/budget | Initialization at G can retain local ancestry | Yes | No for B0 existence, but needed before any independence claim | Moderate | No | Not selected |
| Simulated annealing/direct optimization | Yes if objective, schedule, and seeds are frozen | Depends on objective construction | Yes | Potentially | Moderate to high | No, if C5 remains hard | Not selected |
| Statistics-preserving local rewiring | Yes if move set and stopping rule are frozen | Depends on permitted rewires | Yes | Potentially | Moderate | No | Not selected |
| Other explicitly specified generator | Only after proposal/seeds/stopping are frozen | Must be audited | Depends on design | Depends on design | Unknown | Must not | Not selected |

## 5. Still Requires Researcher Freezing

- Final L generator and its proposal/stopping specification.
- Any requirement that L be sufficiently distant or independent from G.
- Any AxisBias, polarity, or C_dir matching requirement beyond C5.
- Any AUC definition or aggregation of C_dir nodes.

## Boundary Declaration

- Existing model/training/analysis/config files modified: no.
- `outputs/` accessed: no.
- Training, model inference, or GPU task run: no.
- Commit or push performed: no.
