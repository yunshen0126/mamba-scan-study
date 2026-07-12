import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict
from itertools import product

import torch

from mamba_scan_study.experiments.run_stage0_regression import (
    Stage0Config,
    build_loaders,
    count_params,
    evaluate,
    set_seed,
    train_one_epoch,
)
from mamba_scan_study.models.backbone import HAS_MAMBA, MultiDirBackbone


SCAN_DIRS = ("row", "col", "diag", "anti_diag")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-types", nargs="+", default=["gru", "mamba"], choices=["gru", "mamba"])
    p.add_argument("--task-dirs", nargs="+", default=["vertical", "horizontal", "diagonal"],
                   choices=["vertical", "horizontal", "diagonal"])
    p.add_argument("--branch-dirs", nargs="+", default=list(SCAN_DIRS), choices=list(SCAN_DIRS))
    p.add_argument("--grid-sizes", nargs="+", type=int, default=[8, 16, 24, 32])
    p.add_argument("--signal-strengths", nargs="+", default=["single_patch", "single_pixel"],
                   choices=["line", "single_patch", "single_pixel"])
    p.add_argument("--pos-modes", nargs="+", default=["none", "xy_learned"],
                   choices=["none", "seq_learned", "xy_learned", "xy_sincos"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--amplitude", type=float, default=2.5)
    p.add_argument("--noise-std", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="mamba_scan_study/outputs/stage1_gap_sweep")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def make_cfg(args, block_type, task_dir, branch_dir, grid_size, signal_strength, pos_mode):
    return Stage0Config(
        dataset="synthetic",
        task_dir=task_dir,
        signal_strength=signal_strength,
        amplitude=args.amplitude,
        noise_std=args.noise_std,
        grid_size=grid_size,
        img_size=grid_size * args.patch_size,
        patch_size=args.patch_size,
        n_classes=2,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        block_type=block_type,
        branch_dirs=branch_dir,
        d_model=args.d_model,
        n_layers=args.n_layers,
        pos_mode=pos_mode,
        readout_mode="target",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seeds=args.seeds,
        outdir=args.outdir,
    )


def blank_result_row(cfg, seed, skipped, reason):
    return {
        "block_type": cfg.block_type,
        "task_dir": cfg.task_dir,
        "branch_dirs": cfg.branch_dirs,
        "grid_size": cfg.grid_size,
        "signal_strength": cfg.signal_strength,
        "pos_mode": cfg.pos_mode,
        "seed": seed,
        "best_acc": "",
        "last_acc": "",
        "param_count": "",
        "elapsed_sec": 0.0,
        "skipped": skipped,
        "skip_reason": reason,
    }


def run_one(cfg, seed, device):
    if cfg.block_type == "mamba" and not HAS_MAMBA:
        return blank_result_row(cfg, seed, True, "mamba_ssm_not_installed")

    set_seed(seed)
    train_loader, test_loader = build_loaders(cfg, seed)
    set_seed(seed)
    model = MultiDirBackbone(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        block_type=cfg.block_type,
        bidirectional=cfg.bidirectional,
        n_classes=cfg.n_classes,
        branch_dirs=cfg.branch_dirs,
        dropout=cfg.dropout,
        shuffle_order=cfg.shuffle_order,
        pos_mode=cfg.pos_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = 0.0
    last_acc = 0.0
    t0 = time.time()
    for _epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, optimizer, device, cfg)
        test_metrics = evaluate(model, test_loader, device, cfg)
        last_acc = test_metrics["acc"]
        best_acc = max(best_acc, last_acc)

    return {
        "block_type": cfg.block_type,
        "task_dir": cfg.task_dir,
        "branch_dirs": cfg.branch_dirs,
        "grid_size": cfg.grid_size,
        "signal_strength": cfg.signal_strength,
        "pos_mode": cfg.pos_mode,
        "seed": seed,
        "best_acc": best_acc,
        "last_acc": last_acc,
        "param_count": count_params(model),
        "elapsed_sec": time.time() - t0,
        "skipped": False,
        "skip_reason": "",
    }


def mean_std(values):
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize(rows):
    grouped = {}
    for row in rows:
        if row["skipped"]:
            continue
        key = (
            row["block_type"],
            row["task_dir"],
            row["grid_size"],
            row["signal_strength"],
            row["pos_mode"],
        )
        grouped.setdefault(key, {}).setdefault(row["branch_dirs"], []).append(float(row["best_acc"]))

    summaries = []
    for key, by_dir in sorted(grouped.items()):
        block_type, task_dir, grid_size, signal_strength, pos_mode = key
        summary = {
            "block_type": block_type,
            "task_dir": task_dir,
            "grid_size": grid_size,
            "signal_strength": signal_strength,
            "pos_mode": pos_mode,
            "gap_col_minus_row": "",
            "gap_row_minus_col": "",
            "gap_diag_minus_row": "",
            "gap_diag_minus_col": "",
        }
        means = {}
        for scan_dir in SCAN_DIRS:
            mean, std = mean_std(by_dir.get(scan_dir, []))
            summary[f"{scan_dir}_mean"] = mean
            summary[f"{scan_dir}_std"] = std
            means[scan_dir] = mean

        if task_dir == "vertical" and means.get("col") != "" and means.get("row") != "":
            summary["gap_col_minus_row"] = means["col"] - means["row"]
        if task_dir == "horizontal" and means.get("row") != "" and means.get("col") != "":
            summary["gap_row_minus_col"] = means["row"] - means["col"]
        if task_dir == "diagonal":
            if means.get("diag") != "" and means.get("row") != "":
                summary["gap_diag_minus_row"] = means["diag"] - means["row"]
            if means.get("diag") != "" and means.get("col") != "":
                summary["gap_diag_minus_col"] = means["diag"] - means["col"]
        summaries.append(summary)
    return summaries


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.smoke:
        args.block_types = ["gru", "mamba"]
        args.task_dirs = ["vertical"]
        args.branch_dirs = ["row", "col"]
        args.grid_sizes = [8]
        args.signal_strengths = ["single_patch"]
        args.pos_modes = ["xy_learned"]
        args.seeds = [0]
        args.epochs = 1
        args.train_samples = 256
        args.test_samples = 128
        args.batch_size = 64
        args.d_model = 32
        args.n_layers = 1
        args.outdir = os.path.join(args.outdir, "smoke")

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} HAS_MAMBA={HAS_MAMBA} outdir={args.outdir}")

    rows = []
    combos = product(
        args.block_types,
        args.task_dirs,
        args.branch_dirs,
        args.grid_sizes,
        args.signal_strengths,
        args.pos_modes,
        args.seeds,
    )
    for block_type, task_dir, branch_dir, grid_size, signal_strength, pos_mode, seed in combos:
        cfg = make_cfg(args, block_type, task_dir, branch_dir, grid_size, signal_strength, pos_mode)
        row = run_one(cfg, seed, device)
        rows.append(row)
        status = "SKIP" if row["skipped"] else f"best={row['best_acc']:.4f}"
        print(
            f"{status} block={block_type} task={task_dir} scan={branch_dir} "
            f"grid={grid_size} signal={signal_strength} pos={pos_mode} seed={seed}"
        )

    results_csv = os.path.join(args.outdir, "results.csv")
    summary_csv = os.path.join(args.outdir, "summary.csv")
    results_json = os.path.join(args.outdir, "results.json")
    result_fields = [
        "block_type",
        "task_dir",
        "branch_dirs",
        "grid_size",
        "signal_strength",
        "pos_mode",
        "seed",
        "best_acc",
        "last_acc",
        "param_count",
        "elapsed_sec",
        "skipped",
        "skip_reason",
    ]
    summary_fields = [
        "block_type",
        "task_dir",
        "grid_size",
        "signal_strength",
        "pos_mode",
        "row_mean",
        "row_std",
        "col_mean",
        "col_std",
        "diag_mean",
        "diag_std",
        "anti_diag_mean",
        "anti_diag_std",
        "gap_col_minus_row",
        "gap_row_minus_col",
        "gap_diag_minus_row",
        "gap_diag_minus_col",
    ]
    summaries = summarize(rows)
    write_csv(results_csv, rows, result_fields)
    write_csv(summary_csv, summaries, summary_fields)
    with open(results_json, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "has_mamba": HAS_MAMBA,
                "stage0_config_schema": asdict(Stage0Config()),
                "rows": rows,
                "summary": summaries,
            },
            f,
            indent=2,
        )
    print(f"saved results: {results_csv}")
    print(f"saved summary: {summary_csv}")
    print(f"saved json   : {results_json}")


if __name__ == "__main__":
    main()
