import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mamba_scan_study.data.real_datasets import build_real_loaders
from mamba_scan_study.models.backbone import ChannelSplitBackbone, HAS_MAMBA, MultiDirBackbone


VARIANTS = {
    "row": {"branch_dirs": "row", "shuffle_order": False},
    "col": {"branch_dirs": "col", "shuffle_order": False},
    "real_4dir": {"branch_dirs": "row,col,diag,anti_diag", "shuffle_order": False},
    "same_row_4": {"branch_dirs": "row,row,row,row", "shuffle_order": False},
    "shuffle_row": {"branch_dirs": "row", "shuffle_order": True},
}
CHANNEL_VARIANTS = (
    "channel_real_4dir",
    "channel_same_row_4",
    "channel_same_perm_4",
    "channel_rand_perm_4",
)
BLOCKS = ("gru", "mamba")
GRIDS = (8, 16, 32)


@dataclass
class Config:
    dataset: str
    img_size: int
    arch: str
    shuffle_seed: int
    data_root: str
    outdir: str
    microbatch_csv: str
    epochs: int
    warmup_epochs: int
    effective_batch: int
    d_model: int
    n_layers: int
    pos_mode: str
    base_lr: float
    weight_decay: float
    grad_clip: float
    num_workers: int
    seed: int
    amp: bool
    consistency_target: float
    consistency_tolerance: float
    consistency_min_positive: float


def parse_args():
    parser = argparse.ArgumentParser(description="Exact Stage 1 seed-0 CIFAR-10 sweep.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar10_up64"])
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--arch", default="full_branch", choices=["full_branch", "channel_split"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--outdir", default="mamba_scan_study/outputs/stage1_seed0")
    parser.add_argument(
        "--microbatch-csv",
        default="mamba_scan_study/outputs/microbatch_pilot/best_microbatch.csv",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--pos-mode", default="xy_learned")
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_microbatches(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["block_type"], int(row["grid"])): int(row["fastest_micro_batch"])
        for row in rows
    }


def lr_scale(epoch, epochs, warmup_epochs):
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_epoch_lr(optimizer, base_lr, scale):
    value = base_lr * scale
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def shuffle_seed(seed, grid):
    return 1_000_000 + seed * 10_000 + grid


def make_model(cfg, block_type, grid, variant, device):
    if cfg.arch == "channel_split":
        return ChannelSplitBackbone(
            img_size=cfg.img_size,
            patch_size=cfg.img_size // grid,
            in_chans=3,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            block_type=block_type,
            n_classes=10,
            variant=variant,
            shuffle_seed=cfg.shuffle_seed + grid,
            pos_mode=cfg.pos_mode,
        ).to(device)
    spec = VARIANTS[variant]
    return MultiDirBackbone(
        img_size=cfg.img_size,
        patch_size=cfg.img_size // grid,
        in_chans=3,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        block_type=block_type,
        n_classes=10,
        branch_dirs=spec["branch_dirs"],
        shuffle_order=spec["shuffle_order"],
        shuffle_seed=cfg.shuffle_seed + grid if spec["shuffle_order"] else None,
        pos_mode=cfg.pos_mode,
    ).to(device)


def make_loaders(cfg, micro_batch):
    generator = torch.Generator().manual_seed(cfg.seed)
    return build_real_loaders(
        cfg.dataset,
        cfg.data_root,
        micro_batch,
        num_workers=cfg.num_workers,
        img_size=cfg.img_size,
        download=True,
        generator=generator,
    )


def train_epoch(model, loader, optimizer, scaler, cfg, accum_steps, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = correct = 0
    loss_sum = 0.0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits, _ = model(images)
            raw_loss = F.cross_entropy(logits, labels)
            loss = raw_loss / accum_steps
        scaler.scale(loss).backward()
        if batch_index % accum_steps == 0 or batch_index == len(loader):
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += raw_loss.item() * labels.numel()
    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, cfg, device, collect=False):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    logits_parts = []
    labels_parts = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
        total += labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
        if collect:
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels.cpu())
    metrics = {"loss": loss_sum / total, "acc": correct / total}
    if not collect:
        return metrics, None
    logits = torch.cat(logits_parts).numpy()
    labels = torch.cat(labels_parts).numpy()
    return metrics, {
        "sample_index": np.arange(labels.shape[0], dtype=np.int64),
        "labels": labels,
        "predictions": logits.argmax(1),
        "logits": logits,
    }


def run_key(dataset, arch, d_model, block_type, grid, variant):
    return f"{dataset}|{arch}|d{d_model}|{block_type}|{grid}|{variant}"


def artifact_stem(cfg, block_type, grid, variant):
    stem = f"{block_type}_{variant}_grid{grid}_seed{cfg.seed}"
    if cfg.arch != "full_branch" or cfg.d_model != 64:
        stem = f"{cfg.arch}_d{cfg.d_model}_{stem}"
    return stem


def run_one(cfg, block_type, grid, variant, micro_batch, device):
    set_seed(cfg.seed)
    train_loader, test_loader = make_loaders(cfg, micro_batch)
    set_seed(cfg.seed)
    model = make_model(cfg, block_type, grid, variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    accum_steps = cfg.effective_batch // micro_batch
    history = []
    arrays = None
    progress_dir = os.path.join(cfg.outdir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    progress_path = os.path.join(progress_dir, f"{artifact_stem(cfg, block_type, grid, variant)}.json")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        scale = lr_scale(epoch, cfg.epochs, cfg.warmup_epochs)
        lr = set_epoch_lr(optimizer, cfg.base_lr, scale)
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        train_start = epoch_start
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, cfg, accum_steps, device
        )
        torch.cuda.synchronize()
        train_sec = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        test_metrics, collected = evaluate(
            model, test_loader, cfg, device, collect=epoch == cfg.epochs
        )
        torch.cuda.synchronize()
        eval_sec = time.perf_counter() - eval_start
        if collected is not None:
            arrays = collected
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "lr_scale": scale,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "train_sec": train_sec,
                "eval_sec": eval_sec,
                "epoch_wall_sec": time.perf_counter() - epoch_start,
            }
        )
        with open(progress_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "running",
                    "dataset": cfg.dataset,
                    "arch": cfg.arch,
                    "d_model": cfg.d_model,
                    "block_type": block_type,
                    "grid": grid,
                    "variant": variant,
                    "seed": cfg.seed,
                    "micro_batch": micro_batch,
                    "accum_steps": accum_steps,
                    "history": history,
                },
                handle,
                indent=2,
            )
        print(
            f"STAGE1 block={block_type} grid={grid} variant={variant} "
            f"epoch={epoch}/{cfg.epochs} lr={lr:.7f} "
            f"train={train_metrics['acc']:.4f} test={test_metrics['acc']:.4f} "
            f"sec={history[-1]['epoch_wall_sec']:.1f}",
            flush=True,
        )
    permutation = None
    if cfg.arch == "full_branch" and variant == "shuffle_row":
        permutation = model.branches[0].shuffle_perm.detach().cpu().tolist()
    elif cfg.arch == "channel_split" and variant in (
        "channel_same_perm_4",
        "channel_rand_perm_4",
    ):
        permutation = model.channel_permutations.detach().cpu().tolist()
    branch_dirs = (
        VARIANTS[variant]["branch_dirs"]
        if cfg.arch == "full_branch"
        else ",".join(model.branch_dirs)
    )
    run_shuffle_seed = None
    if variant in ("shuffle_row", "channel_same_perm_4", "channel_rand_perm_4"):
        run_shuffle_seed = cfg.shuffle_seed + grid
    result = {
        "key": run_key(cfg.dataset, cfg.arch, cfg.d_model, block_type, grid, variant),
        "dataset": cfg.dataset,
        "arch": cfg.arch,
        "d_model": cfg.d_model,
        "block_type": block_type,
        "variant": variant,
        "branch_dirs": branch_dirs,
        "grid": grid,
        "sequence_length": grid * grid,
        "patch_size": cfg.img_size // grid,
        "seed": cfg.seed,
        "micro_batch": micro_batch,
        "accum_steps": accum_steps,
        "effective_batch": cfg.effective_batch,
        "shuffle_order": variant
        in ("shuffle_row", "channel_same_perm_4", "channel_rand_perm_4"),
        "shuffle_seed": run_shuffle_seed,
        "shuffle_perm": permutation,
        "param_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        "elapsed_sec": time.perf_counter() - run_start,
        "history": history,
    }
    with open(progress_path, "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", **result}, handle, indent=2)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "run": {key: value for key, value in result.items() if key != "history"},
        "history": history,
    }
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result, arrays, checkpoint


def matrix_order(dataset="cifar10", arch="full_branch"):
    if arch == "channel_split":
        grids = (32,) if dataset == "cifar10_up64" else GRIDS
        return [
            (block_type, grid, variant)
            for block_type in BLOCKS
            for grid in grids
            for variant in CHANNEL_VARIANTS
        ]
    if dataset == "cifar10_up64":
        variants = ("row", "shuffle_row", "same_row_4", "real_4dir")
        return [(block_type, 32, variant) for block_type in BLOCKS for variant in variants]
    gate = [
        ("mamba", 8, "same_row_4"),
        ("mamba", 8, "real_4dir"),
    ]
    remaining = []
    variant_order = ("row", "shuffle_row", "col", "same_row_4", "real_4dir")
    for block_type in BLOCKS:
        for grid in GRIDS:
            for variant in variant_order:
                item = (block_type, grid, variant)
                if item not in gate:
                    remaining.append(item)
    return gate + remaining


def grouped_results(results):
    return {
        (row["block_type"], row["grid"], row["variant"]): row for row in results
    }


def paired_rows(results, arch):
    grouped = grouped_results(results)
    variant_names = tuple(VARIANTS) if arch == "full_branch" else CHANNEL_VARIANTS
    real_name = "real_4dir" if arch == "full_branch" else "channel_real_4dir"
    same_name = "same_row_4" if arch == "full_branch" else "channel_same_row_4"
    rows = []
    for block_type in BLOCKS:
        for grid in GRIDS:
            variants = {
                variant: grouped.get((block_type, grid, variant)) for variant in variant_names
            }
            available = [result for result in variants.values() if result is not None]
            if not available:
                continue
            epochs = len(available[0]["history"])
            for epoch_index in range(epochs):
                accuracies = {
                    variant: result["history"][epoch_index]["test_acc"]
                    if result is not None
                    else None
                    for variant, result in variants.items()
                }
                delta = None
                if accuracies[real_name] is not None and accuracies[same_name] is not None:
                    delta = accuracies[real_name] - accuracies[same_name]
                order = None
                if arch == "full_branch" and accuracies["row"] is not None and accuracies["shuffle_row"] is not None:
                    order = accuracies["row"] - accuracies["shuffle_row"]
                rows.append(
                    {
                        "block_type": block_type,
                        "grid": grid,
                        "epoch": epoch_index + 1,
                        **{f"acc_{variant}": accuracies[variant] for variant in variant_names},
                        "delta_direction": delta,
                        "order_utilization": order,
                    }
                )
    return rows


def tail_summary(paired):
    rows = []
    for block_type in BLOCKS:
        for grid in GRIDS:
            group = [
                row
                for row in paired
                if row["block_type"] == block_type
                and row["grid"] == grid
                and 80 <= row["epoch"] <= 100
            ]
            if not group:
                continue
            output = {"block_type": block_type, "grid": grid, "epochs": "80-100"}
            for metric in ("delta_direction", "order_utilization"):
                values = [float(row[metric]) for row in group if row[metric] is not None]
                if values:
                    output.update(
                        {
                            f"{metric}_mean": statistics.mean(values),
                            f"{metric}_std": statistics.pstdev(values),
                            f"{metric}_min": min(values),
                            f"{metric}_max": max(values),
                            f"{metric}_range": max(values) - min(values),
                            f"{metric}_last": values[-1],
                        }
                    )
            rows.append(output)
    return rows


def consistency_check(results, cfg):
    if cfg.arch != "full_branch":
        return {"status": "not_applicable"}
    grouped = grouped_results(results)
    real = grouped.get(("mamba", 8, "real_4dir"))
    same = grouped.get(("mamba", 8, "same_row_4"))
    if real is None or same is None:
        return {"status": "pending"}
    if len(real["history"]) < 100 or len(same["history"]) < 100:
        return {"status": "pending", "note": "Consistency gate requires epochs 80-100."}
    values = [
        real["history"][index]["test_acc"] - same["history"][index]["test_acc"]
        for index in range(79, 100)
    ]
    mean = statistics.mean(values)
    lower = max(cfg.consistency_min_positive, cfg.consistency_target - cfg.consistency_tolerance)
    upper = cfg.consistency_target + cfg.consistency_tolerance
    passed = lower <= mean <= upper
    return {
        "status": "pass" if passed else "fail",
        "reference_stage1c_delta": cfg.consistency_target,
        "accepted_tail_mean_interval": [lower, upper],
        "tail_epoch_range": [80, 100],
        "tail_mean": mean,
        "tail_std": statistics.pstdev(values),
        "tail_min": min(values),
        "tail_max": max(values),
        "epoch100_delta": values[-1],
        "note": "Failure stops Stage 1 before the remaining runs.",
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def history_rows(results):
    rows = []
    for result in results:
        for epoch in result["history"]:
            rows.append(
                {
                    "arch": result.get("arch", "full_branch"),
                    "d_model": result.get("d_model", 64),
                    "block_type": result["block_type"],
                    "grid": result["grid"],
                    "variant": result["variant"],
                    "seed": result["seed"],
                    "micro_batch": result["micro_batch"],
                    "accum_steps": result["accum_steps"],
                    **epoch,
                }
            )
    return rows


def plot_metrics(path, paired):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for column, block_type in enumerate(BLOCKS):
        for grid in GRIDS:
            group = [
                row for row in paired if row["block_type"] == block_type and row["grid"] == grid
            ]
            if not group:
                continue
            epochs = [row["epoch"] for row in group]
            for axis, metric in zip(axes[:, column], ("delta_direction", "order_utilization")):
                values = [row[metric] for row in group]
                if all(value is not None for value in values):
                    axis.plot(epochs, np.asarray(values) * 100, label=f"grid{grid}")
        axes[0, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[0, column].set_title(block_type.upper())
        axes[1, column].set_xlabel("Epoch")
        axes[0, column].legend()
        axes[1, column].legend()
    axes[0, 0].set_ylabel("delta_direction (pp)")
    axes[1, 0].set_ylabel("order_utilization (pp)")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def save_all(cfg, results, final=False):
    os.makedirs(cfg.outdir, exist_ok=True)
    paired = paired_rows(results, cfg.arch)
    summary = tail_summary(paired)
    consistency = consistency_check(results, cfg)
    payload = {
        "config": asdict(cfg),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "mamba_fast_path": False,
        },
        "matrix": [
            {
                "dataset": cfg.dataset,
                "arch": cfg.arch,
                "d_model": cfg.d_model,
                "block_type": block,
                "grid": grid,
                "variant": variant,
                "seed": cfg.seed,
                "shuffle_seed": cfg.shuffle_seed + grid
                if variant in ("shuffle_row", "channel_same_perm_4", "channel_rand_perm_4")
                else None,
            }
            for block, grid, variant in matrix_order(cfg.dataset, cfg.arch)
        ],
        "consistency_check": consistency,
        "results": results,
        "tail_summary": summary,
    }
    with open(os.path.join(cfg.outdir, "stage1_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(os.path.join(cfg.outdir, "stage1_history.csv"), history_rows(results))
    write_csv(os.path.join(cfg.outdir, "paired_metrics_per_epoch.csv"), paired)
    write_csv(os.path.join(cfg.outdir, "tail_80_100_summary.csv"), summary)
    with open(os.path.join(cfg.outdir, "consistency_check.json"), "w", encoding="utf-8") as handle:
        json.dump(consistency, handle, indent=2)
    if final:
        plot_metrics(os.path.join(cfg.outdir, "paired_metrics_curves.png"), paired)


def load_results(cfg):
    path = os.path.join(cfg.outdir, "stage1_results.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("results", [])


def main():
    args = parse_args()
    cfg = Config(
        dataset=args.dataset,
        img_size=args.img_size,
        arch=args.arch,
        shuffle_seed=shuffle_seed(args.seed, 0),
        data_root=args.data_root,
        outdir=args.outdir,
        microbatch_csv=args.microbatch_csv,
        epochs=2 if args.smoke else args.epochs,
        warmup_epochs=1 if args.smoke else args.warmup_epochs,
        effective_batch=args.effective_batch,
        d_model=args.d_model,
        n_layers=args.n_layers,
        pos_mode=args.pos_mode,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=0 if args.smoke else args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        consistency_target=0.0064,
        consistency_tolerance=0.0064,
        consistency_min_positive=0.001,
    )
    # seed guard lifted 2026-07-14: batch A (seeds 0-4) authorized
    expected_img_size = {"cifar10": 32, "cifar10_up64": 64}[cfg.dataset]
    if cfg.img_size != expected_img_size:
        raise ValueError(f"dataset={cfg.dataset} requires img_size={expected_img_size}")
    for grid in {item[1] for item in matrix_order(cfg.dataset, cfg.arch)}:
        if cfg.img_size % grid:
            raise ValueError(f"img_size={cfg.img_size} is not divisible by grid={grid}")
    if not args.smoke and (cfg.epochs != 100 or cfg.warmup_epochs != 5):
        raise ValueError("Formal Stage 1 requires 100 epochs and 5 warmup epochs")
    if not torch.cuda.is_available() or not HAS_MAMBA:
        raise RuntimeError("CUDA and mamba_ssm are required")
    microbatches = read_microbatches(cfg.microbatch_csv)
    if set(microbatches) != {(block, grid) for block in BLOCKS for grid in GRIDS}:
        raise ValueError("best_microbatch.csv does not contain all six block/grid cells")
    for value in microbatches.values():
        if cfg.effective_batch % value:
            raise ValueError("Selected micro-batch must divide effective batch")
    work = matrix_order(cfg.dataset, cfg.arch)
    expected_runs = {
        ("cifar10", "full_branch"): 30,
        ("cifar10_up64", "full_branch"): 8,
        ("cifar10", "channel_split"): 24,
        ("cifar10_up64", "channel_split"): 8,
    }[(cfg.dataset, cfg.arch)]
    if len(work) != expected_runs:
        raise RuntimeError(f"Stage 1 matrix must contain exactly {expected_runs} runs")

    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "checkpoints"), exist_ok=True)
    results = [] if args.no_resume else load_results(cfg)
    completed = {
        run_key(
            row.get("dataset", "cifar10"),
            row.get("arch", "full_branch"),
            row.get("d_model", 64),
            row["block_type"],
            row["grid"],
            row["variant"],
        )
        for row in results
    }
    gate = consistency_check(results, cfg)
    if gate["status"] == "fail":
        raise SystemExit("Stage1C consistency gate previously failed; refusing to continue")

    device = torch.device("cuda")
    for block_type, grid, variant in work:
        key = run_key(cfg.dataset, cfg.arch, cfg.d_model, block_type, grid, variant)
        if key in completed:
            print(f"RESUME skip {key}", flush=True)
            continue
        micro_batch = microbatches[(block_type, grid)]
        print(
            f"START {key} micro={micro_batch} accum={cfg.effective_batch // micro_batch}",
            flush=True,
        )
        result, arrays, checkpoint = run_one(
            cfg, block_type, grid, variant, micro_batch, device
        )
        stem = artifact_stem(cfg, block_type, grid, variant)
        prediction_path = os.path.join(cfg.outdir, "predictions", f"{stem}.npz")
        checkpoint_path = os.path.join(cfg.outdir, "checkpoints", f"{stem}.pt")
        np.savez_compressed(prediction_path, **arrays)
        torch.save(checkpoint, checkpoint_path)
        result["prediction_npz"] = prediction_path
        result["checkpoint"] = checkpoint_path
        results.append(result)
        completed.add(key)
        save_all(cfg, results)
        print(f"DONE {key}", flush=True)

        if (
            cfg.arch == "full_branch"
            and not args.smoke
            and block_type == "mamba"
            and grid == 8
            and variant == "real_4dir"
        ):
            gate = consistency_check(results, cfg)
            print(f"CONSISTENCY {json.dumps(gate)}", flush=True)
            if gate["status"] != "pass":
                save_all(cfg, results)
                raise SystemExit("Stage1C consistency gate failed; stopping before remaining runs")

    final = len(results) == expected_runs
    save_all(cfg, results, final=final)
    print(f"saved Stage 1 to {cfg.outdir}; completed={len(results)}/{expected_runs}", flush=True)


if __name__ == "__main__":
    main()
