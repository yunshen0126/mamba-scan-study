#!/usr/bin/env python3
"""
export_all_results.py

Walk the three batches and write one tidy CSV with one row per run. This is the
table every number in the paper is computed from, and it is small enough to keep
in the repository alongside the code.

    python3 export_all_results.py \
        --main   /root/autodl-tmp/outputs_main \
        --earlier /root/mamba-scan-study/mamba_scan_study/outputs \
        --fused  /root/autodl-tmp/outputs_fused \
        --out    results/results_all.csv

Columns
    batch          main | earlier | fused
    dataset        cifar10, cifar100, organamnist, organcmnist, organsmnist, eurosat
    arch           channel_split | fused_scan
    block          mamba | gru
    d_model        backbone width
    grid           tokens per side
    seq_len        grid squared
    exp_id         the condition as recorded by the runner
    group          GEO_S | GEO_DIV | RND_S | RND_D | LOC_S | LOC_D
    seed           training seed
    epochs         epochs completed
    tail_val_acc   mean validation accuracy over the final `--tail` epochs, in %
    tail_train_acc mean training accuracy over the same window, in %
    final_val_acc  last-epoch validation accuracy, in %
    param_count    parameters, where the run recorded it
    source         path of the file the row came from

Reads only.
"""

import argparse
import csv
import json
import os
import re

SKIP = {".git", "__pycache__", ".ipynb_checkpoints", "wandb", "checkpoints",
        "predictions"}

GROUP = {
    "GEO_SG1": "GEO_S", "GEO_SG2": "GEO_S", "GEO_SG3": "GEO_S", "GEO_SG4": "GEO_S",
    "GEO_DIV": "GEO_DIV",
    "RND_S1": "RND_S", "RND_S2": "RND_S", "RND_S3": "RND_S",
    "RND_D1": "RND_D", "RND_D2": "RND_D", "RND_D3": "RND_D",
    "LOC_S": "LOC_S", "LOC_D": "LOC_D", "LOC_S1": "LOC_S", "LOC_D1": "LOC_D",
    "channel_same_row_4": "GEO_S", "channel_real_4dir": "GEO_DIV",
    "channel_same_perm_4": "RND_S", "channel_rand_perm_4": "RND_D",
}

VAL_KEYS = ("validation_accuracy", "val_accuracy", "val_acc", "test_accuracy",
            "test_acc")
TRAIN_KEYS = ("train_accuracy", "train_acc")
HIST_KEYS = ("validation_history", "history", "epochs", "log", "metrics", "curve")

RUNDIR = re.compile(
    r"^p0b_(?P<ds>[a-z0-9]+)_.*?_(?P<block>mamba|gru)_"
    r"(?P<exp>GEO_SG\d|GEO_DIV|RND_S\d|RND_D\d|LOC_S1?|LOC_D1?)_"
    r"R_(?P<rel>low|high)_seed(?P<seed>\d+)$", re.I)


def history_of(rec):
    for hk in HIST_KEYS:
        h = rec.get(hk)
        if isinstance(h, list) and h and isinstance(h[0], dict):
            if any(k in h[0] for k in VAL_KEYS):
                return h
    return None


def series(hist, keys):
    k = next((k for k in keys if k in hist[0]), None)
    if k is None:
        return []
    return [e[k] for e in hist if isinstance(e.get(k), (int, float))]


def pct(vals):
    if not vals:
        return ""
    m = sum(vals) / len(vals)
    return round(m * 100 if m <= 1.5 else m, 4)


def collect(node, out, depth=0):
    if depth > 6:
        return
    if isinstance(node, dict):
        if "dataset" in node and any(k in node for k in ("exp_id", "variant")):
            out.append(node)
        for v in node.values():
            if isinstance(v, (dict, list)):
                collect(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                collect(v, out, depth + 1)


def rows_from(root, batch, tail):
    rows, seen = [], set()
    if not root or not os.path.isdir(root):
        return rows
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP]
        for f in fn:
            if not f.endswith(".json"):
                continue
            path = os.path.join(dp, f)
            try:
                obj = json.load(open(path, encoding="utf-8", errors="replace"))
            except Exception:
                continue
            recs = []
            collect(obj, recs)
            meta = RUNDIR.match(os.path.basename(dp))
            for r in recs:
                hist = history_of(r)
                if not hist:
                    continue
                exp = str(r.get("exp_id") or r.get("variant") or "")
                if exp not in GROUP:
                    continue
                ds = str(r.get("dataset", meta.group("ds") if meta else "")).lower()
                seed = str(r.get("training_seed", r.get("seed", "")))
                grid = r.get("grid", "")
                block = str(r.get("block_type") or r.get("backbone") or
                            (meta.group("block") if meta else "")).lower()
                dm = r.get("d_model") or (r.get("training_config") or {}).get("d_model", "")
                key = (batch, ds, block, str(dm), str(grid), exp, seed)
                if key in seen:
                    continue
                seen.add(key)
                v = series(hist, VAL_KEYS)
                t = series(hist, TRAIN_KEYS)
                rows.append({
                    "batch": batch, "dataset": ds,
                    "arch": r.get("arch", "channel_split"),
                    "block": block, "d_model": dm, "grid": grid,
                    "seq_len": r.get("sequence_length",
                                     int(grid) ** 2 if str(grid).isdigit() else ""),
                    "exp_id": exp, "group": GROUP[exp], "seed": seed,
                    "epochs": len(hist),
                    "tail_val_acc": pct(v[-tail:]),
                    "tail_train_acc": pct(t[-tail:]),
                    "final_val_acc": pct(v[-1:]),
                    "param_count": r.get("parameter_count", r.get("param_count", "")),
                    "source": os.path.relpath(path, root),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=None)
    ap.add_argument("--earlier", default=None)
    ap.add_argument("--fused", default=None)
    ap.add_argument("--tail", type=int, default=21)
    ap.add_argument("--fused-tail", type=int, default=20)
    ap.add_argument("--out", default="results/results_all.csv")
    a = ap.parse_args()

    rows = []
    rows += rows_from(a.main, "main", a.tail)
    rows += rows_from(a.earlier, "earlier", a.fused_tail)
    rows += rows_from(a.fused, "fused", a.fused_tail)
    if not rows:
        raise SystemExit("no runs found; check the paths")

    rows.sort(key=lambda r: (r["batch"], r["dataset"], r["block"], str(r["d_model"]),
                             str(r["grid"]), r["group"], r["exp_id"], r["seed"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["batch", "dataset", "arch", "block", "d_model", "grid", "seq_len",
            "exp_id", "group", "seed", "epochs", "tail_val_acc",
            "tail_train_acc", "final_val_acc", "param_count", "source"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {a.out}: {len(rows)} runs")
    for b in ("main", "earlier", "fused"):
        sub = [r for r in rows if r["batch"] == b]
        if not sub:
            continue
        cells = {(r["dataset"], r["block"], str(r["d_model"]), str(r["grid"]))
                 for r in sub}
        print(f"  {b:<8} {len(sub):>4} runs, {len(cells)} cells, "
              f"datasets {sorted({r['dataset'] for r in sub})}")
    short = [r for r in rows if r["epochs"] < 50]
    if short:
        print(f"\n  {len(short)} runs have fewer than 50 epochs; check them:")
        for r in short[:8]:
            print(f"    {r['batch']} {r['dataset']} {r['exp_id']} seed{r['seed']} "
                  f"({r['epochs']} epochs)")


if __name__ == "__main__":
    main()
