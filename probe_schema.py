#!/usr/bin/env python3
"""
probe_schema.py  --  show me what your result files actually look like.

    python3 probe_schema.py ~/mamba-scan-study

Finds every csv/json whose text mentions cifar100 or cifar10, and prints its
column names and two sample records. Output is capped so it stays pasteable.
Reads only.
"""

import csv
import io
import json
import os
import sys

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", "wandb"}
MAX_FILES = 14
MAX_VAL = 60


def trunc(v):
    s = str(v).replace("\n", " ")
    return s if len(s) <= MAX_VAL else s[:MAX_VAL] + "..."


def head_text(path, n=200000):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(n)
    except Exception:
        return ""


def show_csv(path, text):
    try:
        rd = csv.reader(io.StringIO(text))
        rows = []
        for i, r in enumerate(rd):
            rows.append(r)
            if i >= 3:
                break
    except Exception as e:
        print(f"    unreadable as csv: {e}")
        return
    if not rows:
        print("    empty")
        return
    print(f"    columns ({len(rows[0])}): {rows[0]}")
    for r in rows[1:3]:
        print(f"    row: {[trunc(x) for x in r]}")


def flat(obj, out, prefix="", depth=0):
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                flat(v, out, f"{prefix}{k}.", depth + 1)
            else:
                out[f"{prefix}{k}"] = v
    elif isinstance(obj, list) and obj and not isinstance(obj[0], (dict, list)):
        out[prefix.rstrip(".")] = f"[list of {len(obj)}]"
    return out


def show_json(path, text):
    try:
        obj = json.loads(text)
    except Exception as e:
        print(f"    unreadable as json: {e}")
        return
    if isinstance(obj, list):
        print(f"    top level: list of {len(obj)}")
        recs = [x for x in obj[:2] if isinstance(x, dict)]
    elif isinstance(obj, dict):
        print(f"    top level: dict, keys = {list(obj)[:12]}")
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals) and len(vals) > 1:
            recs = vals[:2]
        else:
            recs = [obj]
    else:
        print(f"    top level: {type(obj).__name__}")
        return
    for r in recs:
        d = flat(r, {})
        items = list(d.items())[:18]
        print("    record keys/values:")
        for k, v in items:
            print(f"      {k} = {trunc(v)}")
        if len(d) > 18:
            print(f"      ... {len(d) - 18} more keys")


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else ".")
    hits100, hits10 = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.lower().endswith((".csv", ".json", ".jsonl")):
                continue
            p = os.path.join(dirpath, fn)
            t = head_text(p)
            low = t.lower()
            if "cifar100" in low or "cifar-100" in low:
                hits100.append((p, t))
            elif "cifar10" in low or "cifar-10" in low:
                hits10.append((p, t))

    print(f"root: {root}")
    print(f"files mentioning cifar100: {len(hits100)}")
    print(f"files mentioning cifar10 only: {len(hits10)}")
    print()

    shown = 0
    for label, hits in (("CIFAR-100", hits100), ("CIFAR-10", hits10)):
        if not hits:
            continue
        print("=" * 76)
        print(f"{label}  --  showing up to {MAX_FILES // 2} files, largest first")
        print("=" * 76)
        hits.sort(key=lambda x: -os.path.getsize(x[0]))
        for p, t in hits[: MAX_FILES // 2]:
            print(f"\n  {os.path.relpath(p, root)}  ({os.path.getsize(p)} bytes)")
            if p.lower().endswith(".csv"):
                show_csv(p, t)
            else:
                show_json(p, t)
            shown += 1
        print()

    if not shown:
        print("no csv/json mentions cifar at all. Try a wider root, e.g.")
        print("    python3 probe_schema.py /root")


if __name__ == "__main__":
    main()
