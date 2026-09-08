#!/usr/bin/env python3
"""
compute_2x2.py  (v2)

Fixes over v1:
  - endpoints auto-scaled to percentage points when stored as fractions
  - arch and d_model are part of the run key, so channel_split/full_branch and
    d64/d256 runs no longer overwrite one another
  - three decimals, and a compatibility bound printed for P_R

    python3 compute_2x2.py ~/mamba-scan-study --tail 10

Condition mapping:
    channel_same_row_4   -> GEO_S     channel_real_4dir   -> GEO_DIV
    channel_same_perm_4  -> RND_S     channel_rand_perm_4 -> RND_D
    P_G = GEO_DIV - GEO_S,  P_R = RND_D - RND_S,  interaction = P_G - P_R.
Reads only.
"""

import json
import math
import os
import sys
from collections import defaultdict

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "wandb"}
COND = {
    # earlier channel-split batch
    "channel_same_row_4": "GEO_S", "channel_real_4dir": "GEO_DIV",
    "channel_same_perm_4": "RND_S", "channel_rand_perm_4": "RND_D",
    # fused-scan arm and the main design
    "GEO_SG1": "GEO_S", "GEO_DIV": "GEO_DIV",
    "RND_S1": "RND_S", "RND_D1": "RND_D",
    "LOC_S1": "LOC_S", "LOC_D1": "LOC_D",
    "LOC_S": "LOC_S", "LOC_D": "LOC_D",
}
ORDER = ("GEO_S", "GEO_DIV", "RND_S", "RND_D")
OPTIONAL = ("LOC_S", "LOC_D")
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
ACC_HINTS = ("val_acc", "valid_acc", "val_accuracy", "validation_accuracy",
             "test_acc", "acc")


def find_series(node, depth=0):
    if depth > 5:
        return None
    if isinstance(node, list):
        if node and isinstance(node[0], dict):
            for h in ACC_HINTS:
                if h in node[0]:
                    v = [r[h] for r in node if isinstance(r.get(h), (int, float))]
                    if v:
                        return v
        if node and all(isinstance(x, (int, float)) for x in node):
            return list(node)
        for v in node:
            r = find_series(v, depth + 1)
            if r:
                return r
    if isinstance(node, dict):
        for h in ACC_HINTS:
            v = node.get(h)
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                return list(v)
        for k in ("history", "epochs", "log", "curve", "metrics"):
            if k in node:
                r = find_series(node[k], depth + 1)
                if r:
                    return r
        for v in node.values():
            if isinstance(v, (dict, list)):
                r = find_series(v, depth + 1)
                if r:
                    return r
    return None


def collect(node, out, depth=0):
    if depth > 6:
        return
    if isinstance(node, dict):
        vid = node.get("variant") or node.get("exp_id")
        if vid in COND and "dataset" in node:
            s = find_series(node)
            if s:
                out.append((node, s))
        for v in node.values():
            if isinstance(v, (dict, list)):
                collect(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                collect(v, out, depth + 1)


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = TCRIT.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, m - h, m + h


def fmt(m, lo, hi):
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{m:+7.3f} [{lo:+7.3f},{hi:+7.3f}] {star}"


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else ".")
    tail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 10

    recs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(dirpath, fn), encoding="utf-8",
                          errors="replace") as f:
                    obj = json.load(f)
            except Exception:
                continue
            collect(obj, recs)

    table, meta = {}, defaultdict(set)
    for node, series in recs:
        cell = (str(node.get("dataset")), str(node.get("arch")),
                str(node.get("d_model")), str(node.get("block_type")),
                str(node.get("grid")))
        vid = node.get("variant") or node.get("exp_id")
        key = cell + (COND[vid], str(node.get("seed")))
        use = series[-tail:] if len(series) >= tail else series
        table[key] = sum(use) / len(use)
        for f in ("param_count", "epochs", "hflip", "horizontal_flip", "augment",
                  "augmentation", "shuffle_order", "sequence_length", "patch_size"):
            if f in node:
                meta[cell + (f,)].add(str(node[f]))

    scale = 100.0 if (table and max(table.values()) <= 1.5) else 1.0
    print(f"root: {root}")
    print(f"{len(table)} unique runs, endpoint = mean of last {tail} epochs"
          f"{', rescaled to percentage points' if scale == 100 else ''}\n")

    for cell in sorted({k[:5] for k in table}):
        ds, arch, dm, blk, grid = cell
        seeds = sorted({k[6] for k in table if k[:5] == cell})
        common = [s for s in seeds if all(cell + (c, s) in table for c in ORDER)]
        print("=" * 84)
        print(f"{ds}  arch={arch}  d_model={dm}  block={blk}  grid={grid}"
              f"   n={len(common)}")
        print("=" * 84)
        if len(common) < 2:
            counts = {c: sum(1 for s in seeds if cell + (c, s) in table) for c in ORDER}
            print(f"  incomplete: {counts}\n")
            continue

        present = list(ORDER) + [c for c in OPTIONAL
                                 if all(cell + (c, s) in table for s in common)]
        for c in present:
            v = [table[cell + (c, s)] * scale for s in common]
            print(f"  {c:<9} {sum(v)/len(v):7.3f}")

        def get(c, s):
            return table[cell + (c, s)] * scale

        pg = [get("GEO_DIV", s) - get("GEO_S", s) for s in common]
        pr = [get("RND_D", s) - get("RND_S", s) for s in common]
        print()
        print(f"  P_G          {fmt(*ci(pg))}")
        print(f"  P_R          {fmt(*ci(pr))}")
        m, lo, hi = ci(pr)
        print(f"               |P_R| compatible with <= {max(abs(lo), abs(hi)):.3f} pp")
        print(f"  P_G - P_R    {fmt(*ci([a - b for a, b in zip(pg, pr)]))}")
        if "LOC_S" in present and "LOC_D" in present:
            pl = [get("LOC_D", s) - get("LOC_S", s) for s in common]
            print(f"  P_L          {fmt(*ci(pl))}")
            print(f"  P_G - P_L    {fmt(*ci([a - b for a, b in zip(pg, pl)]))}")
        print(f"  structure    {fmt(*ci([get('GEO_S', s) - get('RND_S', s) for s in common]))}"
              f"  (GEO_S - RND_S)")

        pc = sorted({v for k, vs in meta.items() if k[:5] == cell
                     and k[5] == "param_count" for v in vs})
        if pc:
            print(f"  param_count  {pc}"
                  f"{'' if len(pc) == 1 else '   <-- NOT MATCHED'}")
        for f in ("epochs", "hflip", "horizontal_flip", "augment", "augmentation",
                  "sequence_length", "patch_size"):
            vs = sorted({v for k, val in meta.items() if k[:5] == cell
                         and k[5] == f for v in val})
            if vs:
                print(f"  {f:<12} {vs}")
        print()

    print("* marks an interval excluding zero.")


if __name__ == "__main__":
    main()
