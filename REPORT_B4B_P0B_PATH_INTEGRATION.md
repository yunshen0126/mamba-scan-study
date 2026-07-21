# B4B P0-B Path Integration

Status: PASS. This report covers only the resolver and `ChannelSplitBackbone`
explicit-path integration. It does not implement a data helper, runner,
launcher, ledger, checkpoint metadata, training, or GPU smoke test.

## Source Gates

All four source files were hashed before either frozen path bank was parsed.

| Source | SHA-256 | Result |
| --- | --- | --- |
| `P0B_L_PATH_BANK_FROZEN.json` | `93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7` | PASS |
| `P0B_R_PATH_BANK_FROZEN.json` | `2f7b8a6fd3cfbbae9897b4ef4dc9dcfd1bf7744619d5818ceaca7604d565aee3` | PASS |
| `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` | `e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09` | PASS |
| `docs/P0B_CONFIG_TABLE.md` | `790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e` | PASS |

The resolver accepts alternate file locations only through `P0BSourcePaths`.
Those files must have these exact bytes and SHA-256 values. It does not call
`torch.randperm`, B0/B1 search or recovery code, or validation-split generation.

## Resolver Results

`13 exp_id x 2 grid x 4 training seed = 104` unique
`(exp_id, grid, training_seed)` design cells resolved: **104/104 PASS**.
Every result contains four CPU `torch.long` absolute row-major orders, four path
IDs, and four paired order/inverse SHA-256 values. Every actual order is checked
for grid `n`, available `N`, length, range, uniqueness, `order_sha256`,
`inverse[order] = arange(N)`, and `inverse_order_sha256` before return.

| Example | seed | ch0..ch3 |
| --- | ---: | --- |
| `GEO_SG3` | 2 | `G3,G3,G3,G3` |
| `GEO_DIV` | 1 | `G2,G3,G4,G1` |
| `RND_S2` | 3 | `R2_4,R2_4,R2_4,R2_4` |
| `RND_D3` | 3 | `R3_4,R3_1,R3_2,R3_3` |
| `LOC_S` | 2 | `L3,L3,L3,L3` |
| `LOC_D` | 2 | `L3,L4,L1,L2` |

The R artifact is read using its actual `sets: S1,S2,S3` structure. All 24
frozen R records were checked across grid8/grid32: each has the requested
`path_id`, `set_id`, frozen seed `17071 + 1000*s + i`, order hash, and inverse
hash. Tests reject each of the four source SHA gates independently, invalid
length/duplicate/out-of-range orders, bad order and inverse hashes, and bad R
`set_id` or seed metadata.

## Backbone Interface

`ChannelSplitBackbone(..., channel_orders=None)` preserves the old path exactly.
With `channel_orders is not None`, only `variant="channel_same_row_4"` is
accepted, `branch_dirs` is exactly `("row", "row", "row", "row")`, and each
absolute `order[t] = row_major_cell` is copied as CPU `torch.long` into the
existing persistent `(4, L)` buffers:

```text
channel_permutations
channel_inverse_permutations
```

The inverse is constructed as `inverse[order] = arange(L)`. No trainable
parameters, branch count, group width, blocks, fusion, norm, or head changed.
`explicit_channel_orders` is an ordinary Python boolean and is absent from the
state dict.

In explicit mode all groups always apply token permutation, available position
permutation, available mask permutation, and post-block inverse selection. This
includes identity G1. Legacy mode retains the identity shortcut. There is no
path-content-specific forward branch.

## CPU Tests

Command:

```bash
conda run -n mair python -B -m mamba_scan_study.analysis.test_p0b_path_integration_cpu
```

Environment: `/home/tuling/miniconda3/envs/mair/bin/python`, PyTorch
`2.0.1+cu118`. `/usr/bin/time -p` recorded wall `14.55 s`, user `6.97 s`,
system `3.10 s`, CPU `10.07 s`.

| Check | Result |
| --- | --- |
| Four source gates; 104/104 resolver cells; all 24 R records | PASS |
| G1/G2/G3/G4/R/LMTO actual pre-block order equals absolute input order | PASS |
| G3 is column raster once, with no branch-local second reordering | PASS |
| Position/mask share the order; inverse restores row-major cells | PASS |
| Explicit G1 identity executes token/position/mask/inverse `index_select` | PASS |
| Legacy identity retains its shortcut | PASS |
| Legacy four-variant permutations match an independent generator reference | PASS |
| Buffer names/dtype/shape/persistence and strict legacy-like state load | PASS |
| Fixed-input legacy output equals explicit identity after strict state load | PASS |
| G/R/LMTO single/diverse six-class parameters, buffers, operator graph, and output shapes | PASS |

The tests use CPU GRU construction, `n_layers=0`, and identity capture modules;
they do not execute a Mamba CPU kernel. No `outputs/` path was accessed, and no
training, inference, GPU operation, commit, or push was performed.
