import argparse
import csv
import json
import os
import signal
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


VARIANT_BRANCH_DIRS = {
    "row": "row",
    "col": "col",
    "diag": "diag",
    "anti_diag": "anti_diag",
    "real_4dir": "row,col,diag,anti_diag",
    "same_row_4": "row,row,row,row",
}
SKIPPED_VARIANTS = {
    "bidir_row": "reverse_scan_not_implemented",
}
VARIANTS = tuple(VARIANT_BRANCH_DIRS) + tuple(SKIPPED_VARIANTS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--block-types", nargs="+", default=["gru", "mamba"], choices=["gru", "mamba"])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["row", "col", "real_4dir", "same_row_4"],
        choices=list(VARIANTS),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--pos-mode", default="xy_learned",
                   choices=["none", "seq_learned", "xy_learned", "xy_sincos"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--setup-timeout-sec", type=int, default=900)
    p.add_argument("--outdir", type=str, default="mamba_scan_study/outputs/stage1_direction_ablation")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--checkpoint-dir", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def make_cfg(args, block_type, variant):
    return Stage0Config(
        dataset=args.dataset,
        data_root=args.data_root,
        img_size=32,
        patch_size=args.patch_size,
        in_chans=3,
        n_classes=10,
        block_type=block_type,
        branch_dirs=VARIANT_BRANCH_DIRS.get(variant, ""),
        d_model=args.d_model,
        n_layers=args.n_layers,
        pos_mode=args.pos_mode,
        readout_mode="mean_pool",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seeds=args.seeds,
        outdir=args.outdir,
    )


def blank_result_row(args, block_type, variant, seed, skipped, reason):
    return {
        "dataset": args.dataset,
        "block_type": block_type,
        "variant": variant,
        "branch_dirs": VARIANT_BRANCH_DIRS.get(variant, ""),
        "pos_mode": args.pos_mode,
        "seed": seed,
        "best_acc": "",
        "last_acc": "",
        "param_count": "",
        "elapsed_sec": 0.0,
        "skipped": skipped,
        "skip_reason": reason,
    }


class SetupTimeout(Exception):
    pass


class setup_timeout:
    def __init__(self, seconds):
        self.seconds = seconds
        self.previous_handler = None

    def __enter__(self):
        if self.seconds <= 0 or not hasattr(signal, "SIGALRM"):
            return
        self.previous_handler = signal.getsignal(signal.SIGALRM)

        def _handle_timeout(_signum, _frame):
            raise SetupTimeout(f"setup timed out after {self.seconds} seconds")

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, _exc_type, _exc, _tb):
        if self.seconds <= 0 or not hasattr(signal, "SIGALRM"):
            return
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous_handler)


def run_one(args, block_type, variant, seed, device):
    if variant in SKIPPED_VARIANTS:
        return blank_result_row(args, block_type, variant, seed, True, SKIPPED_VARIANTS[variant])
    if block_type == "mamba" and not HAS_MAMBA:
        return blank_result_row(args, block_type, variant, seed, True, "mamba_ssm_not_installed")

    cfg = make_cfg(args, block_type, variant)
    t0 = time.time()
    try:
        set_seed(seed)
        with setup_timeout(args.setup_timeout_sec):
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
    except Exception as exc:
        row = blank_result_row(args, block_type, variant, seed, True, f"setup_error: {exc}")
        row["elapsed_sec"] = time.time() - t0
        return row

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_acc = 0.0
    last_acc = 0.0
    best_epoch = -1
    try:
        for epoch in range(cfg.epochs):
            train_one_epoch(model, train_loader, optimizer, device, cfg)
            test_metrics = evaluate(model, test_loader, device, cfg)
            last_acc = test_metrics["acc"]
            if last_acc > best_acc:
                best_acc = last_acc
                best_epoch = epoch
                if args.save_checkpoints:
                    ckpt_dir = args.checkpoint_dir or os.path.join(args.outdir, "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"{block_type}_{variant}_seed{seed}.pt")
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "cfg": asdict(cfg),
                            "args": vars(args),
                            "block_type": block_type,
                            "variant": variant,
                            "seed": seed,
                            "best_acc": best_acc,
                            "best_epoch": best_epoch,
                            "branch_dirs": cfg.branch_dirs,
                        },
                        ckpt_path,
                    )
    except Exception as exc:
        row = blank_result_row(args, block_type, variant, seed, True, f"train_error: {exc}")
        row["param_count"] = count_params(model)
        row["elapsed_sec"] = time.time() - t0
        return row

    return {
        "dataset": args.dataset,
        "block_type": block_type,
        "variant": variant,
        "branch_dirs": cfg.branch_dirs,
        "pos_mode": cfg.pos_mode,
        "seed": seed,
        "best_acc": best_acc,
        "last_acc": last_acc,
        "param_count": count_params(model),
        "elapsed_sec": time.time() - t0,
        "best_epoch": best_epoch,
        "skipped": False,
        "skip_reason": "",
    }


def preflight_dataset(args):
    cfg = make_cfg(args, args.block_types[0], "row")
    with setup_timeout(args.setup_timeout_sec):
        build_loaders(cfg, args.seeds[0])


def mean_std(values):
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize(rows):
    grouped = {}
    branch_dirs = {}
    for row in rows:
        if row["skipped"]:
            continue
        key = (row["dataset"], row["block_type"], row["pos_mode"])
        grouped.setdefault(key, {}).setdefault(row["variant"], []).append(float(row["best_acc"]))
        branch_dirs[row["variant"]] = row["branch_dirs"]

    summaries = []
    for key, by_variant in sorted(grouped.items()):
        dataset, block_type, pos_mode = key
        means = {}
        for variant, values in by_variant.items():
            mean, std = mean_std(values)
            means[variant] = mean
            same_row_mean = means.get("same_row_4")
            row_mean = means.get("row")
            summaries.append(
                {
                    "dataset": dataset,
                    "block_type": block_type,
                    "variant": variant,
                    "branch_dirs": branch_dirs.get(variant, VARIANT_BRANCH_DIRS.get(variant, "")),
                    "pos_mode": pos_mode,
                    "mean": mean,
                    "std": std,
                    "accs": json.dumps(values),
                    "delta_vs_row": "" if row_mean in ("", None) else mean - row_mean,
                    "delta_vs_same_row_4": ""
                    if same_row_mean in ("", None)
                    else mean - same_row_mean,
                }
            )

        row_mean = means.get("row")
        same_row_mean = means.get("same_row_4")
        for summary in summaries:
            if (
                summary["dataset"] == dataset
                and summary["block_type"] == block_type
                and summary["pos_mode"] == pos_mode
            ):
                mean = summary["mean"]
                summary["delta_vs_row"] = "" if row_mean in ("", None) else mean - row_mean
                summary["delta_vs_same_row_4"] = (
                    "" if same_row_mean in ("", None) else mean - same_row_mean
                )
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
        args.variants = ["row", "col", "real_4dir", "same_row_4", "bidir_row"]
        args.seeds = [0]
        args.epochs = 1
        args.batch_size = 256
        args.d_model = 32
        args.n_layers = 1
        args.setup_timeout_sec = min(args.setup_timeout_sec, 120)
        args.outdir = os.path.join(args.outdir, "smoke")

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} HAS_MAMBA={HAS_MAMBA} dataset={args.dataset} outdir={args.outdir}")
    print("note=real_4dir and same_row_4 are full-branch multi-path variants, not channel-split GroupMamba")

    rows = []
    dataset_error = ""
    try:
        preflight_dataset(args)
    except Exception as exc:
        dataset_error = f"dataset_setup_error: {exc}"
        print(f"SKIP_ALL reason={dataset_error}")

    for block_type, variant, seed in product(args.block_types, args.variants, args.seeds):
        if dataset_error:
            row = blank_result_row(args, block_type, variant, seed, True, dataset_error)
        else:
            row = run_one(args, block_type, variant, seed, device)
        rows.append(row)
        status = "SKIP" if row["skipped"] else f"best={row['best_acc']:.4f}"
        print(
            f"{status} dataset={args.dataset} block={block_type} variant={variant} "
            f"branches={row['branch_dirs']} pos={args.pos_mode} seed={seed} "
            f"reason={row['skip_reason']}"
        )

    results_csv = os.path.join(args.outdir, "results.csv")
    summary_csv = os.path.join(args.outdir, "summary.csv")
    results_json = os.path.join(args.outdir, "results.json")
    result_fields = [
        "dataset",
        "block_type",
        "variant",
        "branch_dirs",
        "pos_mode",
        "seed",
        "best_acc",
        "last_acc",
        "param_count",
        "elapsed_sec",
        "best_epoch",
        "skipped",
        "skip_reason",
    ]
    summary_fields = [
        "dataset",
        "block_type",
        "variant",
        "branch_dirs",
        "pos_mode",
        "mean",
        "std",
        "accs",
        "delta_vs_row",
        "delta_vs_same_row_4",
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
                "note": "real_4dir and same_row_4 are full-branch multi-path variants, not channel-split GroupMamba",
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
