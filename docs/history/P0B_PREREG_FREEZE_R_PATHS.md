# P0-B R Path Freeze

Before any P0-B performance run, 24 R paths were frozen in `P0B_R_PATH_BANK_FROZEN.json`.
For each `n in {8,32}`, `s in {1,2,3}`, and `i in {1,2,3,4}`, a new CPU `torch.Generator` was seeded with `17071 + 1000*s + i`, then used once for `torch.randperm(n*n, generator=g, device="cpu", dtype=torch.int64)`.

The frozen full arrays and NumPy int64 C-byte order/inverse hashes are the only runtime source of truth. No training-time redraw, seed offset, or post-hoc selection is permitted. R is a fixed path blocking factor, not a random-effect sample from a path population.

For NumPy < 2.0, `np.trapz` is used only as the compatibility API for the frozen trapezoidal integral: it does not alter the nodes, zero anchor, integration interval, normalization, or mathematical AUC definition.

`RND_Ss` maps training seed 0/1/2/3 to `R^s_1/R^s_2/R^s_3/R^s_4`, repeated on all four channels. `RND_Ds` uses the fixed four paths with the existing Latin-square channel rotation. This preserves the original exp_id, 13 conditions, 2 reliance levels, 4 training seeds, and 104 runs.
