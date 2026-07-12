import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict
from itertools import product

import torch
import torch.nn.functional as F

from mamba_scan_study.experiments.run_stage0_regression import (
    Stage0Config,
    build_loaders,
    count_params,
    evaluate,
    set_seed,
    train_one_epoch,
)
from mamba_scan_study.models.backbone import HAS_MAMBA, MultiDirBackbone
from mamba_scan_study.models.scan_utils import SCAN_DIRS, scan_indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-types", nargs="+", default=["gru", "mamba"], choices=["gru", "mamba"])
    p.add_argument("--task-dirs", nargs="+", default=["vertical", "horizontal"],
                   choices=["vertical", "horizontal", "diagonal"])
    p.add_argument("--branch-dirs", nargs="+", default=["row", "col"], choices=list(SCAN_DIRS))
    p.add_argument("--grid-sizes", nargs="+", type=int, default=[8, 16, 24, 32])
    p.add_argument("--signal-strengths", nargs="+", default=["single_patch"],
                   choices=["line", "single_patch", "single_pixel"])
    p.add_argument("--pos-modes", nargs="+", default=["xy_learned"],
                   choices=["none", "seq_learned", "xy_learned", "xy_sincos"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--probe-epochs", type=int, default=5)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--amplitude", type=float, default=4.0)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="mamba_scan_study/outputs/stage1_probing")
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
        "backbone_best_acc": "",
        "backbone_last_acc": "",
        "probe_best_acc": "",
        "probe_last_acc": "",
        "row_distance": "",
        "col_distance": "",
        "scan_distance": "",
        "param_count": "",
        "probe_param_count": "",
        "elapsed_sec": 0.0,
        "skipped": skipped,
        "skip_reason": reason,
    }


def target_features(backbone, batch, device):
    images, labels, target_row, target_col, _source_row, _source_col = batch
    images = images.to(device)
    labels = labels.to(device)
    feat2d = backbone.forward_features(images)
    batch_idx = torch.arange(feat2d.shape[0], device=device)
    features = feat2d[
        batch_idx,
        target_row.to(device),
        target_col.to(device),
    ]
    return features, labels


def train_probe_one_epoch(backbone, probe, loader, optimizer, device, cfg):
    backbone.eval()
    probe.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features, labels = target_features(backbone, batch, device)
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
        optimizer.step()
        total += labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def evaluate_probe(backbone, probe, loader, device):
    backbone.eval()
    probe.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        features, labels = target_features(backbone, batch, device)
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        total += labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def distance_stats(loader, grid_size, scan_dir):
    scan_pos = {}
    for pos, row_major_idx in enumerate(scan_indices(grid_size, grid_size, scan_dir)):
        r = row_major_idx // grid_size
        c = row_major_idx % grid_size
        scan_pos[(r, c)] = pos

    total = 0
    row_sum = 0.0
    col_sum = 0.0
    scan_sum = 0.0
    for batch in loader:
        _images, _labels, target_row, target_col, source_row, source_col = batch
        for tr, tc, sr, sc in zip(target_row, target_col, source_row, source_col):
            tr = int(tr)
            tc = int(tc)
            sr = int(sr)
            sc = int(sc)
            row_sum += abs(tr - sr)
            col_sum += abs(tc - sc)
            scan_sum += abs(scan_pos[(tr, tc)] - scan_pos[(sr, sc)])
            total += 1
    return {
        "row_distance": row_sum / total,
        "col_distance": col_sum / total,
        "scan_distance": scan_sum / total,
    }


def run_one(cfg, seed, device, args):
    if cfg.block_type == "mamba" and not HAS_MAMBA:
        return blank_result_row(cfg, seed, True, "mamba_ssm_not_installed")

    set_seed(seed)
    train_loader, test_loader = build_loaders(cfg, seed)
    set_seed(seed)
    backbone = MultiDirBackbone(
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
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    backbone_best_acc = 0.0
    backbone_last_acc = 0.0
    t0 = time.time()
    for _epoch in range(cfg.epochs):
        train_one_epoch(backbone, train_loader, optimizer, device, cfg)
        test_metrics = evaluate(backbone, test_loader, device, cfg)
        backbone_last_acc = test_metrics["acc"]
        backbone_best_acc = max(backbone_best_acc, backbone_last_acc)

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    set_seed(seed)
    probe = torch.nn.Linear(cfg.d_model, cfg.n_classes).to(device)
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )

    probe_best_acc = 0.0
    probe_last_acc = 0.0
    for _epoch in range(args.probe_epochs):
        train_probe_one_epoch(backbone, probe, train_loader, probe_optimizer, device, cfg)
        probe_metrics = evaluate_probe(backbone, probe, test_loader, device)
        probe_last_acc = probe_metrics["acc"]
        probe_best_acc = max(probe_best_acc, probe_last_acc)

    distances = distance_stats(test_loader, cfg.grid_size, cfg.branch_dirs)
    return {
        "block_type": cfg.block_type,
        "task_dir": cfg.task_dir,
        "branch_dirs": cfg.branch_dirs,
        "grid_size": cfg.grid_size,
        "signal_strength": cfg.signal_strength,
        "pos_mode": cfg.pos_mode,
        "seed": seed,
        "backbone_best_acc": backbone_best_acc,
        "backbone_last_acc": backbone_last_acc,
        "probe_best_acc": probe_best_acc,
        "probe_last_acc": probe_last_acc,
        "row_distance": distances["row_distance"],
        "col_distance": distances["col_distance"],
        "scan_distance": distances["scan_distance"],
        "param_count": count_params(backbone),
        "probe_param_count": count_params(probe),
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
        grouped.setdefault(key, {}).setdefault(row["branch_dirs"], []).append(float(row["probe_best_acc"]))

    summaries = []
    for key, by_dir in sorted(grouped.items()):
        block_type, task_dir, grid_size, signal_strength, pos_mode = key
        summary = {
            "block_type": block_type,
            "task_dir": task_dir,
            "grid_size": grid_size,
            "signal_strength": signal_strength,
            "pos_mode": pos_mode,
            "probe_gap_col_minus_row": "",
            "probe_gap_row_minus_col": "",
        }
        means = {}
        for scan_dir in SCAN_DIRS:
            mean, std = mean_std(by_dir.get(scan_dir, []))
            summary[f"{scan_dir}_probe_mean"] = mean
            summary[f"{scan_dir}_probe_std"] = std
            means[scan_dir] = mean

        if task_dir == "vertical" and means.get("col") != "" and means.get("row") != "":
            summary["probe_gap_col_minus_row"] = means["col"] - means["row"]
        if task_dir == "horizontal" and means.get("row") != "" and means.get("col") != "":
            summary["probe_gap_row_minus_col"] = means["row"] - means["col"]
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
        args.probe_epochs = 1
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
        row = run_one(cfg, seed, device, args)
        rows.append(row)
        status = "SKIP" if row["skipped"] else (
            f"backbone={row['backbone_best_acc']:.4f} probe={row['probe_best_acc']:.4f}"
        )
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
        "backbone_best_acc",
        "backbone_last_acc",
        "probe_best_acc",
        "probe_last_acc",
        "row_distance",
        "col_distance",
        "scan_distance",
        "param_count",
        "probe_param_count",
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
        "row_probe_mean",
        "row_probe_std",
        "col_probe_mean",
        "col_probe_std",
        "diag_probe_mean",
        "diag_probe_std",
        "anti_diag_probe_mean",
        "anti_diag_probe_std",
        "probe_gap_col_minus_row",
        "probe_gap_row_minus_col",
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
