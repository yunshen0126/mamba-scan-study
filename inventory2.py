#!/usr/bin/env python3
"""
inventory2.py  --  what runs exist, by dataset / block / grid / variant.

    python3 inventory2.py ~/mamba-scan-study

Reads every json in full (the previous probe truncated at 200 kB, which is why
the big stage1_results.json files looked corrupt), walks nested structures for
any record carrying a `dataset` field, and reports the condition vocabulary and
seed coverage. Reads only.
"""

import json
import os
import sys
from collections import defaultdict

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", "wandb"}
ID_KEYS = ("variant", "exp_id", "condition", "key")
INTEREST = ("dataset", "arch", "d_model", "block_type", "variant", "exp_id",
            "condition", "branch_dirs", "grid", "patch_size", "sequence_length",
            "seed", "param_count", "status")


def walk_records(node, out, depth=0):
    """collect every dict that has a 'dataset' field and some identifier"""
    if depth > 6:
        return
    if isinstance(node, dict):
        if "dataset" in node and any(k in node for k in ID_KEYS):
            out.append({k: node.get(k) for k in INTEREST if k in node})
        for v in node.values():
            if isinstance(v, (dict, list)):
                walk_records(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                walk_records(v, out, depth + 1)


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else ".")
    recs = []
    bad = 0
    files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
            except Exception:
                bad += 1
                continue
            files += 1
            before = len(recs)
            walk_records(obj, recs)
            for r in recs[before:]:
                r["_file"] = os.path.relpath(p, root)

    print(f"root: {root}")
    print(f"{files} json parsed, {bad} unparseable, {len(recs)} run records found\n")

    def ident(r):
        for k in ID_KEYS:
            if r.get(k) not in (None, ""):
                return str(r[k])
        return "?"

    by_ds = defaultdict(list)
    for r in recs:
        by_ds[str(r.get("dataset", "?")).lower()].append(r)

    print("records per dataset:")
    for ds, rs in sorted(by_ds.items(), key=lambda x: -len(x[1])):
        print(f"  {ds:<16} {len(rs)}")
    print()

    for ds in ("cifar100", "cifar10"):
        rs = by_ds.get(ds)
        print("=" * 92)
        print(f"{ds}")
        print("=" * 92)
        if not rs:
            print("  none\n")
            continue

        # what apparatus variants exist at all
        for field in ("arch", "block_type", "d_model", "param_count", "grid",
                      "patch_size", "status"):
            vals = sorted({str(r.get(field)) for r in rs if field in r})
            if vals:
                print(f"  {field:<14} {vals[:12]}{' ...' if len(vals) > 12 else ''}")
        print()

        vocab = sorted({ident(r) for r in rs})
        print(f"  condition labels ({len(vocab)}):")
        for v in vocab:
            print(f"    {v}")
        print()

        # coverage: (block_type, grid, variant) -> seeds
        cov = defaultdict(set)
        for r in rs:
            cov[(str(r.get("block_type", "?")), str(r.get("grid", "?")),
                 ident(r))].add(str(r.get("seed", "?")))
        blocks = sorted({k[0] for k in cov})
        grids = sorted({k[1] for k in cov})
        for b in blocks:
            for g in grids:
                keys = [k for k in cov if k[0] == b and k[1] == g]
                if not keys:
                    continue
                print(f"  block={b}  grid={g}")
                for k in sorted(keys, key=lambda x: x[2]):
                    seeds = sorted(cov[k])
                    print(f"    {k[2]:<52} {len(seeds)} seeds {seeds}")
                print()

    print("=" * 92)
    print("files contributing cifar100 records")
    print("=" * 92)
    for f in sorted({r["_file"] for r in by_ds.get("cifar100", [])})[:20]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
