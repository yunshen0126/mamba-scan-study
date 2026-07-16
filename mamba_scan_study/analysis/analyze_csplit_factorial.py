#!/usr/bin/env python3
"""Channel-split 2x2 factorial analysis (structure x diversity).

Reads stage1_history.csv, groups by (seed, block, grid), takes the
tail-window mean of train_acc/test_acc per variant, and computes the
2x2 factorial main effects + interaction.

Axes:
  structure : {real, same_row} = structured row-order
              {rand_perm, same_perm} = random shuffle
  diversity : {real, rand_perm} = 4 distinct patterns
              {same_row, same_perm} = 1 pattern x4

  structure_effect = mean(structured) - mean(shuffled)
  diversity_effect = mean(diverse)    - mean(single)
  interaction      = (real - same_row) - (rand_perm - same_perm)
                     i.e. geometric-diversity gain - shuffle-diversity gain

Primary metric: train_acc (capacity thermometer). test_acc reported too.

Pure stdlib. Usage:
  python analyze_csplit_factorial.py <history.csv> [--tail-lo 80] [--tail-hi 100]
"""
import csv
import sys
import argparse
from collections import defaultdict

VARIANTS = ("channel_real_4dir", "channel_same_row_4",
            "channel_rand_perm_4", "channel_same_perm_4")
SHORT = {"channel_real_4dir": "real", "channel_same_row_4": "same_row",
         "channel_rand_perm_4": "rand_perm", "channel_same_perm_4": "same_perm"}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history_csv")
    ap.add_argument("--tail-lo", type=int, default=80)
    ap.add_argument("--tail-hi", type=int, default=100)
    args = ap.parse_args()

    # group[(seed, block, grid)][variant][metric] = list of tail values
    group = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(args.history_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            ep = int(row["epoch"])
            if not (args.tail_lo <= ep <= args.tail_hi):
                continue
            key = (row["seed"], row["block_type"], int(row["grid"]))
            v = row["variant"]
            group[key][v]["train"].append(float(row["train_acc"]))
            group[key][v]["test"].append(float(row["test_acc"]))

    for key in sorted(group, key=lambda k: (k[1], k[2], k[0])):
        seed, block, grid = key
        vardata = group[key]
        print(f"\n===== seed={seed} block={block} grid={grid} "
              f"(tail {args.tail_lo}-{args.tail_hi}) =====")

        # per-variant tail means
        tr, te = {}, {}
        for v in VARIANTS:
            if v in vardata:
                tr[v] = mean(vardata[v]["train"])
                te[v] = mean(vardata[v]["test"])
                print(f"  {SHORT[v]:<10} train={tr[v]:.4f}  test={te[v]:.4f}")
            else:
                print(f"  {SHORT[v]:<10} MISSING")

        if not all(v in tr for v in VARIANTS):
            present = sorted(SHORT[v] for v in vardata)
            print(f"  -> INCOMPLETE (have: {present}); factorial skipped")
            continue

        for label, d in (("TRAIN", tr), ("TEST", te)):
            R, SR = d["channel_real_4dir"], d["channel_same_row_4"]
            RP, SP = d["channel_rand_perm_4"], d["channel_same_perm_4"]
            structure = mean([R, SR]) - mean([RP, SP])
            diversity = mean([R, RP]) - mean([SR, SP])
            interaction = (R - SR) - (RP - SP)
            div_struct = R - SR      # diversity gain within structured row
            div_shuffle = RP - SP    # diversity gain within shuffle
            print(f"  [{label}] structure(struct-shuffle) = {structure:+.4f}")
            print(f"  [{label}] diversity(diverse-single)  = {diversity:+.4f}")
            print(f"  [{label}] interaction (geom_div - shuffle_div) = {interaction:+.4f}")
            print(f"           ({div_struct:+.4f} within-structured  vs "
                  f"{div_shuffle:+.4f} within-shuffle)")


if __name__ == "__main__":
    main()
