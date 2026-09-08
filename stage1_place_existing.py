"""
stage1_place_existing.py

Place the existing conditions on the (lambda, alpha) plane. Reads the frozen
path banks; no GPU, no training, no writes to any frozen artefact.

    python stage1_place_existing.py --grid 32 \
        --r-bank P0B_R_PATH_BANK_FROZEN.json \
        --l-bank P0B_L_PATH_BANK_FROZEN.json

The banks' internal schema is discovered rather than assumed. Because the
stored arrays may be `order` or `pi`, every bank is reported under both
readings; pick the reading whose G-side control matches the known reference
(d_mean = 16.5 at grid 32).
"""

import argparse
import numpy as np
from scan_geometry import (adjacent_pairs, describe, load_bank, to_order,
                           gen_G, gen_A, gen_B, HDR, fmt, locality)


def chunk4(items):
    """group the discovered arrays into sets of four, in file order"""
    return [items[i:i + 4] for i in range(0, len(items) - 3, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--r-bank", default=None)
    ap.add_argument("--l-bank", default=None)
    ap.add_argument("--a-w", type=int, default=8)
    args = ap.parse_args()

    n = args.grid
    pairs = adjacent_pairs(n)

    print("=" * 108)
    print(f"grid {n}x{n}   N = {n*n}")
    print("=" * 108)
    print(HDR)

    G = gen_G(n)
    dG = describe("G  (analytic)", G, n, pairs)
    print(fmt(dG))

    for label, path in (("R", args.r_bank), ("L", args.l_bank)):
        if path is None:
            continue
        found = load_bank(path, n)
        print(f"\n-- {label} bank: {path}")
        print(f"   {len(found)} permutations discovered; keys: {[k for k, _ in found][:8]}")
        for reading in ("as order", "as pi"):
            groups = chunk4([a if reading == "as order" else to_order(a) for _, a in found])
            for gi, grp in enumerate(groups):
                print(fmt(describe(f"{label}{gi+1} {reading}", grp, n, pairs)))
        print()

    print("-- candidate new conditions")
    print(fmt(describe(f"A  (w={args.a_w})", gen_A(n, 43000, w=args.a_w), n, pairs)))
    print(fmt(describe("B", gen_B(n, 62000), n, pairs)))

    print()
    print("=" * 108)
    print("legacy d_seq matching criterion, checked on the reference path")
    print("=" * 108)
    st = locality(G[0], n, pairs)
    pi = np.empty(n * n, dtype=np.int64)
    pi[G[0]] = np.arange(n * n)
    d = np.abs(pi[pairs[:, 0]] - pi[pairs[:, 1]])
    vals, cnts = np.unique(d, return_counts=True)
    print(f"G1  d_mean={st['d_mean']:.2f}  p50={st['d_p50']:.2f}  p90={st['d_p90']:.2f}")
    print(f"G1  distinct d_seq values {vals.tolist()} with counts {cnts.tolist()}")
    print("p50 falls between the two mass points, so a +/-10% band on it is not a")
    print("band on anything; report d_mean and lambda instead.")


if __name__ == "__main__":
    main()
