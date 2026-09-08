#!/usr/bin/env python3
"""
export_main_cells.py

Produce the ten-cell table for the main design: P_R, P_G, P_G-P_R (and P_L)
per dataset per patch granularity, as seed-paired means with 95% t intervals.

Two modes.

    python3 export_main_cells.py --probe /root/mamba-scan-study
        Locate the main batch and print the schema of whatever it finds, so the
        column names can be pinned. Run this first.

    python3 export_main_cells.py /root/mamba-scan-study --tail 20
        Compute and print the table, plus a LaTeX version ready to paste.

Condition grouping in the main design:
    GEO_S   = mean of GEO_SG1..GEO_SG4     (four single canonical paths)
    GEO_DIV = the four-canonical-path condition
    RND_S   = mean of RND_S1..RND_S3       (three single arbitrary paths)
    RND_D   = mean of RND_D1..RND_D3       (three arbitrary four-path sets)
    LOC_S, LOC_D = the locality-matched pair
so  P_G = GEO_DIV - GEO_S,  P_R = RND_D - RND_S,  P_L = LOC_D - LOC_S,
each computed per seed and then averaged.
Reads only; writes nothing.
"""

import csv
import io
import json
import math
import os
import re
import sys
from collections import defaultdict

SKIP = {".git", "__pycache__", ".ipynb_checkpoints", "wandb"}

GROUPS = {
    "GEO_S":   re.compile(r"^GEO[_-]?SG?\d+$", re.I),
    "GEO_DIV": re.compile(r"^GEO[_-]?(DIV|D)$", re.I),
    "RND_S":   re.compile(r"^RND[_-]?S\d+$", re.I),
    "RND_D":   re.compile(r"^RND[_-]?D\d+$", re.I),
    "LOC_S":   re.compile(r"^LOC[_-]?S$", re.I),
    "LOC_D":   re.compile(r"^LOC[_-]?D$", re.I),
}
NEEDED = ("GEO_S", "GEO_DIV", "RND_S", "RND_D")
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}

FIELDS = {
    "dataset": ["dataset", "data", "dataset_name", "ds"],
    "exp_id":  ["exp_id", "expid", "condition", "cond", "variant", "arm", "name"],
    "grid":    ["grid", "granularity", "load", "patch_size", "patch", "reliance",
                "sequence_length", "seq_len"],
    "seed":    ["seed", "training_seed", "train_seed"],
    "acc":     ["val_acc", "valid_acc", "val_accuracy", "validation_accuracy",
                "tail_val_acc", "val_acc_tail", "endpoint", "accuracy", "acc",
                "test_acc", "top1", "val_top1"],
    "epoch":   ["epoch", "ep"],
}
MARKERS = ("geo_sg", "geo_div", "rnd_s", "rnd_d", "loc_d", "organamnist",
           "organcmnist", "organsmnist", "eurosat", "geo_s", "rnd_", "loc_s")
MAXBYTES = 40 * 1024 * 1024


BAD_ACC = ("accum", "accur_steps", "acceleration")


def pick(headers, kind):
    low = {h.lower().strip(): h for h in headers}
    for c in FIELDS[kind]:
        if c in low:
            return low[c]
    if kind == "acc":
        # never fuzzy-match accuracy; "accum_steps" is a real column in this study
        for h in headers:
            hl = h.lower()
            if any(b in hl for b in BAD_ACC):
                continue
            if ("acc" in hl or "top1" in hl) and ("val" in hl or "test" in hl
                                                  or hl in ("acc", "accuracy")):
                return h
        return None
    for c in FIELDS[kind]:
        for h in headers:
            if c in h.lower():
                return h
    return None


RUNDIR = re.compile(
    r"^p0b_(?P<ds>[a-z0-9]+)_.*?_(?P<block>mamba|gru)_"
    r"(?P<exp>GEO_SG\d|GEO_DIV|RND_S\d|RND_D\d|LOC_S|LOC_D)_"
    r"R_(?P<rel>low|high)_seed(?P<seed>\d+)$", re.I)


def from_dirname(path):
    """recover dataset / condition / granularity / seed from the run folder name"""
    for part in reversed(os.path.abspath(path).split(os.sep)):
        m = RUNDIR.match(part)
        if m:
            d = m.groupdict()
            return {"dataset": d["ds"].lower(), "exp_id": d["exp"].upper(),
                    "grid": "coarse" if d["rel"].lower() == "low" else "fine",
                    "seed": d["seed"], "block": d["block"].lower()}
    return None


def group_of(label):
    lab = str(label).strip()
    for name, rx in GROUPS.items():
        if rx.match(lab):
            return name
    return None


def norm_grid(v):
    s = str(v).strip().lower()
    if s in ("1024", "32", "fine", "high", "1"):
        return "fine"
    if s in ("64", "8", "coarse", "low", "4"):
        return "coarse"
    return s


def read_csv(path):
    try:
        with io.open(path, newline="", encoding="utf-8", errors="replace") as f:
            rd = csv.DictReader(f)
            if not rd.fieldnames:
                return None, []
            return rd.fieldnames, list(rd)
    except Exception:
        return None, []


def flat(o, out, pre="", d=0):
    if d > 4:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                flat(v, out, pre + k + ".", d + 1)
            else:
                out[pre + k] = v
    return out


HIST_KEYS = ("validation_history", "history", "epochs", "log", "metrics",
             "curve", "val_history")
EPOCH_KEYS = ("epoch", "ep", "step")
VALACC_KEYS = ("validation_accuracy", "val_accuracy", "val_acc", "valid_accuracy",
               "validation_acc", "test_accuracy", "test_acc")


def _history_rows(rec):
    """If a run record carries a per-epoch history, expand it into one row per
    epoch, carrying the record's scalar fields along."""
    for hk in HIST_KEYS:
        h = rec.get(hk)
        if not (isinstance(h, list) and h and isinstance(h[0], dict)):
            continue
        ak = next((k for k in VALACC_KEYS if k in h[0]), None)
        if ak is None:
            ak = next((k for k in h[0]
                       if "acc" in k.lower() and "train" not in k.lower()), None)
        if ak is None:
            continue
        ek = next((k for k in EPOCH_KEYS if k in h[0]), None)
        scal = {k: v for k, v in rec.items()
                if not isinstance(v, (dict, list))}
        rows = []
        for e in h:
            if not isinstance(e, dict) or ak not in e:
                continue
            r = dict(scal)
            r["val_acc"] = e[ak]
            r["epoch"] = e.get(ek, len(rows) + 1)
            rows.append(r)
        if rows:
            return sorted({k for r in rows for k in r}), rows
    return None, []


def read_json(path):
    try:
        obj = json.load(io.open(path, encoding="utf-8", errors="replace"))
    except Exception:
        return None, []
    found = []

    def walk(n, d=0):
        if d > 6:
            return
        if isinstance(n, dict):
            if "dataset" in n and any(k in n for k in ("exp_id", "condition", "variant")):
                found.append(n)
            for v in n.values():
                if isinstance(v, (dict, list)):
                    walk(v, d + 1)
        elif isinstance(n, list):
            for v in n:
                if isinstance(v, (dict, list)):
                    walk(v, d + 1)

    walk(obj)
    if not found:
        return None, []

    rows = []
    for rec in found:
        hdr, hr = _history_rows(rec)
        if hr:
            rows.extend(hr)
        else:
            rows.append(flat(rec, {}))
    if not rows:
        return None, []
    return sorted({k for r in rows for k in r}), rows


def scan(root):
    """files whose content carries a design label, plus every csv/json that sits
    inside a run directory (those hold the metrics and carry no label text)."""
    hits = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP]
        if RUNDIR.match(os.path.basename(dp)):
            hits.extend(os.path.join(dp, f) for f in fn
                        if f.lower().endswith((".csv", ".json")))
            continue
        for f in fn:
            if not f.lower().endswith((".csv", ".json")):
                continue
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > MAXBYTES:
                    continue
                head = io.open(p, encoding="utf-8", errors="replace").read().lower()
            except Exception:
                continue
            if any(m in head for m in MARKERS):
                hits.append(p)
    return hits


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = TCRIT.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, m - h, m + h


def cell(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]"


def tex(m, lo, hi):
    s = f"${m:+.2f}$ $[{lo:+.2f},{hi:+.2f}]$"
    return "\\bcell{" + s + "}" if (lo > 0 or hi < 0) else s


def main():
    probe = "--probe" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = os.path.expanduser(args[0] if args else ".")
    tail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 20

    files = scan(root)
    print(f"root: {root}\n{len(files)} candidate files mention the main-design labels\n")
    if not files:
        print("nothing found. Try a wider root, or grep manually:")
        print(f"  grep -rl 'GEO_SG1' {root} | head")
        return

    if probe:
        rundirs = []
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in SKIP]
            if RUNDIR.match(os.path.basename(dp)):
                rundirs.append((dp, sorted(fn)))
            if len(rundirs) >= 3:
                break
        if rundirs:
            print("run directories detected; contents of the first few:")
            for dp, fn in rundirs:
                print(f"  {os.path.basename(dp)}")
                print(f"     parsed: {from_dirname(dp)}")
                print(f"     files:  {fn}")
                for f in fn:
                    fp = os.path.join(dp, f)
                    if f.endswith(".json"):
                        try:
                            o = json.load(io.open(fp, encoding="utf-8", errors="replace"))
                        except Exception:
                            continue
                        if isinstance(o, dict):
                            print(f"     {f} ALL keys ({len(o)}): {sorted(o)}")
                            for kk, vv in o.items():
                                if isinstance(vv, list) and vv and isinstance(vv[0], (dict, int, float)):
                                    print(f"        {kk}: list[{len(vv)}], first = "
                                          f"{str(vv[0])[:120]}")
                                elif isinstance(vv, dict) and any(
                                        "acc" in str(x).lower() for x in vv):
                                    print(f"        {kk}: dict keys {sorted(vv)[:12]}")
                        else:
                            print(f"     {f}: {type(o).__name__}")
                    elif f.endswith((".pt", ".pth")):
                        try:
                            import torch
                        except ImportError:
                            print(f"     {f}: torch unavailable, cannot inspect")
                            continue
                        try:
                            ck = torch.load(fp, map_location="cpu", weights_only=False)
                        except Exception as e:
                            print(f"     {f}: unreadable ({e})")
                            continue
                        if isinstance(ck, dict):
                            print(f"     {f} keys: {sorted(ck)}")
                            for kk, vv in ck.items():
                                if kk == "model_state":
                                    continue
                                if isinstance(vv, list) and vv:
                                    print(f"        {kk}: list[{len(vv)}], first = "
                                          f"{str(vv[0])[:140]}")
                                elif isinstance(vv, dict):
                                    print(f"        {kk}: dict keys {sorted(vv)[:16]}")
                    elif f.endswith(".csv"):
                        h, r = read_csv(fp)
                        print(f"     {f} columns: {h[:16] if h else 'unparsed'}")
                        if r:
                            print(f"     {f} last row: {{k: str(r[-1][k])[:18] for k in (h or [])[:8]}}".replace("{k: str(r[-1][k])[:18] for k in (h or [])[:8]}", str({k: str(r[-1][k])[:18] for k in (h or [])[:8]})))
                print()
            print("If a column above holds validation accuracy, rerun without --probe.")
            return
        for p in sorted(files, key=lambda x: -os.path.getsize(x))[:8]:
            print(f"  {os.path.relpath(p, root)}  ({os.path.getsize(p)} B)")
            hdr, rows = (read_csv(p) if p.endswith(".csv") else read_json(p))
            if not hdr:
                print("     not parseable as a record table")
                continue
            print(f"     columns: {hdr[:20]}")
            for r in rows[:2]:
                print("     row:", {k: str(r[k])[:24] for k in hdr[:9] if k in r})
            print()
        print("If the columns look right, rerun without --probe.")
        return

    # dataset -> grid -> group -> seed -> [values]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    used, skipped = set(), []
    for p in files:
        hdr, rows = (read_csv(p) if p.endswith(".csv") else read_json(p))
        if not hdr:
            continue
        c_ds, c_id = pick(hdr, "dataset"), pick(hdr, "exp_id")
        c_gr, c_sd = pick(hdr, "grid"), pick(hdr, "seed")
        c_ac, c_ep = pick(hdr, "acc"), pick(hdr, "epoch")
        meta = from_dirname(os.path.dirname(p))
        if meta and c_ac:
            for r in rows:
                r.setdefault("__ds", meta["dataset"])
                r.setdefault("__id", meta["exp_id"])
                r.setdefault("__gr", meta["grid"])
                r.setdefault("__sd", meta["seed"])
                r.setdefault("__blk", meta["block"])
            c_ds, c_id, c_gr, c_sd = "__ds", "__id", "__gr", "__sd"
            if any(r.get("__blk") != "mamba" for r in rows):
                continue
        if not all((c_ds, c_id, c_gr, c_sd, c_ac)):
            missing = [n for n, v in (("dataset", c_ds), ("exp_id", c_id),
                                      ("grid", c_gr), ("seed", c_sd),
                                      ("accuracy", c_ac)) if not v]
            skipped.append((os.path.relpath(p, root), missing, hdr[:14]))
            continue
        # per-epoch tables: keep only the tail window
        by_run = defaultdict(list)
        for r in rows:
            g = group_of(r.get(c_id, ""))
            if g is None:
                continue
            try:
                v = float(r[c_ac])
            except (TypeError, ValueError, KeyError):
                continue
            key = (str(r.get(c_ds)).lower(), norm_grid(r.get(c_gr)), g,
                   str(r.get(c_sd)), str(r.get(c_id)))
            ep = None
            if c_ep:
                try:
                    ep = int(float(r[c_ep]))
                except (TypeError, ValueError, KeyError):
                    ep = None
            by_run[key].append((ep, v))
        if by_run:
            used.add(os.path.relpath(p, root))
        for (ds, gr, g, sd, cond), pts in by_run.items():
            if len(pts) > 1 and all(e is not None for e, _ in pts):
                pts.sort(key=lambda t: t[0])
                vals = [v for _, v in pts[-tail:]]
            else:
                vals = [v for _, v in pts]
            acc[ds][gr][g][sd].append(sum(vals) / len(vals))

    print("files used:")
    for u in sorted(used)[:10]:
        print("  ", u)
    if not used:
        print("  (none)")
    if skipped:
        print("\nfiles that look relevant but were skipped, and why:")
        for f, miss, cols in skipped[:12]:
            print(f"  {f}\n     missing column(s): {miss}\n     has: {cols}")
        print("\nIf one of these IS the results table, tell me its accuracy")
        print("column name and I will pin it.")
    print()

    scale = None
    for ds in acc:
        for gr in acc[ds]:
            for g in acc[ds][gr]:
                for sd in acc[ds][gr][g]:
                    for v in acc[ds][gr][g][sd]:
                        scale = 100.0 if (scale is None and v <= 1.5) else (scale or 1.0)

    rows_out = []
    for ds in sorted(acc):
        for gr in ("coarse", "fine"):
            if gr not in acc[ds]:
                continue
            tab = acc[ds][gr]
            if not all(k in tab for k in NEEDED):
                print(f"  SKIP {ds}/{gr}: missing {[k for k in NEEDED if k not in tab]}")
                continue
            seeds = sorted(set.intersection(*[set(tab[k]) for k in NEEDED]))
            if len(seeds) < 2:
                print(f"  SKIP {ds}/{gr}: {len(seeds)} common seeds")
                continue
            mean = lambda g, s: sum(tab[g][s]) / len(tab[g][s]) * scale
            pg = [mean("GEO_DIV", s) - mean("GEO_S", s) for s in seeds]
            pr = [mean("RND_D", s) - mean("RND_S", s) for s in seeds]
            it = [a - b for a, b in zip(pg, pr)]
            pl = None
            if "LOC_S" in tab and "LOC_D" in tab:
                ls = [s for s in seeds if s in tab["LOC_S"] and s in tab["LOC_D"]]
                if len(ls) >= 2:
                    pl = [mean("LOC_D", s) - mean("LOC_S", s) for s in ls]
            rows_out.append((ds, gr, len(seeds), ci(pr), ci(pg), ci(it),
                             ci(pl) if pl else None))

    print(f"\n{'dataset':<14}{'gran':<8}{'n':<3}{'P_R':<24}{'P_G':<24}{'P_G-P_R':<24}{'P_L'}")
    for ds, gr, n, r, g, i, l in rows_out:
        print(f"{ds:<14}{gr:<8}{n:<3}{cell(*r):<24}{cell(*g):<24}{cell(*i):<24}"
              f"{cell(*l) if l else '-'}")

    print("\n% ---- LaTeX body, paste into the table in main.tex ----")
    for ds, gr, n, r, g, i, l in rows_out:
        print(f"{ds} & {gr} & {tex(*r)} & {tex(*g)} & {tex(*i)} \\\\")


if __name__ == "__main__":
    main()
