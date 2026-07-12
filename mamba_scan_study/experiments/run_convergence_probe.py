import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mamba_scan_study.data.real_datasets import build_real_loaders
from mamba_scan_study.models.backbone import HAS_MAMBA, MultiDirBackbone


@dataclass
class Config:
    data_root: str
    outdir: str
    microbatch_csv: str
    prior_timing_json: str
    block_types: tuple[str, ...]
    grids: tuple[int, ...]
    effective_batch: int
    epochs: int
    d_model: int
    n_layers: int
    pos_mode: str
    lr: float
    weight_decay: float
    grad_clip: float
    num_workers: int
    seed: int
    amp: bool
    plateau_window: int
    plateau_tolerance: float
    plateau_confirmation: int


def parse_args():
    parser = argparse.ArgumentParser(description="60-epoch grid convergence probe.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--outdir", default="mamba_scan_study/outputs/convergence_probe")
    parser.add_argument(
        "--microbatch-csv",
        default="mamba_scan_study/outputs/microbatch_pilot/best_microbatch.csv",
    )
    parser.add_argument(
        "--prior-timing-json",
        default="mamba_scan_study/outputs/timing_pilot/pilot_results.json",
    )
    parser.add_argument("--block-types", nargs="+", default=["gru", "mamba"])
    parser.add_argument("--grids", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--pos-mode", default="xy_learned")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--plateau-tolerance", type=float, default=0.002)
    parser.add_argument("--plateau-confirmation", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize():
    torch.cuda.synchronize()


def read_best_microbatches(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row["block_type"], int(row["grid"])): int(row["fastest_micro_batch"])
            for row in csv.DictReader(handle)
        }


def make_loaders(cfg, micro_batch):
    generator = torch.Generator().manual_seed(cfg.seed)
    return build_real_loaders(
        "cifar10",
        cfg.data_root,
        micro_batch,
        num_workers=cfg.num_workers,
        img_size=32,
        download=True,
        generator=generator,
    )


def make_model(cfg, block_type, grid):
    return MultiDirBackbone(
        img_size=32,
        patch_size=32 // grid,
        in_chans=3,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        block_type=block_type,
        n_classes=10,
        branch_dirs="row",
        pos_mode=cfg.pos_mode,
    ).cuda()


def train_epoch(model, loader, optimizer, scaler, cfg, accum_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = correct = 0
    loss_sum = 0.0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
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
def evaluate(model, loader, cfg, collect=False):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    logits_parts = []
    labels_parts = []
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
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


def run_key(block_type, grid):
    return f"{block_type}|{grid}"


def run_probe(cfg, block_type, grid, micro_batch):
    set_seed(cfg.seed)
    train_loader, test_loader = make_loaders(cfg, micro_batch)
    set_seed(cfg.seed)
    model = make_model(cfg, block_type, grid)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    accum_steps = cfg.effective_batch // micro_batch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    history = []
    final_arrays = None
    for epoch in range(1, cfg.epochs + 1):
        synchronize()
        epoch_start = time.perf_counter()
        train_start = epoch_start
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, cfg, accum_steps)
        synchronize()
        train_sec = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        test_metrics, arrays = evaluate(
            model, test_loader, cfg, collect=epoch == cfg.epochs
        )
        synchronize()
        eval_sec = time.perf_counter() - eval_start
        if arrays is not None:
            final_arrays = arrays
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "train_sec": train_sec,
                "eval_sec": eval_sec,
                "epoch_wall_sec": time.perf_counter() - epoch_start,
            }
        )
        print(
            f"PROBE block={block_type} grid={grid} micro={micro_batch} "
            f"epoch={epoch}/{cfg.epochs} train={train_metrics['acc']:.4f} "
            f"test={test_metrics['acc']:.4f} sec={history[-1]['epoch_wall_sec']:.1f}",
            flush=True,
        )
    result = {
        "key": run_key(block_type, grid),
        "dataset": "cifar10",
        "block_type": block_type,
        "variant": "row",
        "grid": grid,
        "sequence_length": grid * grid,
        "micro_batch": micro_batch,
        "accum_steps": accum_steps,
        "effective_batch": cfg.effective_batch,
        "seed": cfg.seed,
        "history": history,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result, final_arrays


def rolling_mean(values, window):
    output = [None] * len(values)
    for index in range(window - 1, len(values)):
        output[index] = float(np.mean(values[index - window + 1 : index + 1]))
    return output


def find_plateau(history, window, tolerance, confirmation):
    accuracy = [row["test_acc"] for row in history]
    smooth = rolling_mean(accuracy, window)
    last_candidate = len(accuracy) - confirmation
    for epoch_index in range(max(window - 1, 9), last_candidate):
        current = smooth[epoch_index]
        future = [value for value in smooth[epoch_index + 1 :] if value is not None]
        if future and max(future) - current <= tolerance:
            return {
                "status": "plateau_observed",
                "plateau_epoch": epoch_index + 1,
                "smoothed_acc": current,
                "future_gain": max(future) - current,
            }
    return {
        "status": "not_converged_by_probe_end",
        "plateau_epoch": None,
        "smoothed_acc": smooth[-1],
        "future_gain": None,
    }


def curve_rows(results, cfg):
    rows = []
    for result in results:
        smooth = rolling_mean(
            [epoch["test_acc"] for epoch in result["history"]], cfg.plateau_window
        )
        for epoch, smoothed in zip(result["history"], smooth):
            rows.append(
                {
                    "block_type": result["block_type"],
                    "grid": result["grid"],
                    "micro_batch": result["micro_batch"],
                    **epoch,
                    "test_acc_rolling": "" if smoothed is None else smoothed,
                }
            )
    return rows


def analyze(results, cfg):
    analyses = []
    for result in results:
        plateau = find_plateau(
            result["history"],
            cfg.plateau_window,
            cfg.plateau_tolerance,
            cfg.plateau_confirmation,
        )
        analyses.append(
            {
                "block_type": result["block_type"],
                "grid": result["grid"],
                "micro_batch": result["micro_batch"],
                "best_test_acc": max(row["test_acc"] for row in result["history"]),
                "best_epoch": max(
                    result["history"], key=lambda row: row["test_acc"]
                )["epoch"],
                **plateau,
            }
        )
    observed = [row["plateau_epoch"] for row in analyses if row["plateau_epoch"]]
    all_observed = len(observed) == len(analyses)
    if all_observed:
        recommended = min(
            cfg.epochs,
            int(math.ceil((max(observed) + cfg.plateau_confirmation) / 5.0) * 5),
        )
        recommendation = {
            "status": "recommend_epoch_count",
            "recommended_epochs": recommended,
            "rule": "latest plateau + confirmation window, rounded up to 5",
        }
    else:
        recommendation = {
            "status": "extend_probe",
            "recommended_epochs": None,
            "next_probe_target_epochs": 100,
            "rule": "at least one grid did not show a confirmed plateau by epoch 60",
        }
    return analyses, recommendation


def read_microbatch_timings(path):
    directory = os.path.dirname(path)
    results_path = os.path.join(directory, "microbatch_results.json")
    with open(results_path, encoding="utf-8") as handle:
        return json.load(handle)["results"]


def old_row_micro8_timings(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)["results"]
    return {
        (row["block_type"], row["grid"]): row["mean_epoch_sec"]
        for row in rows
        if row["dataset"] == "cifar10"
    }


def cost_estimate(results, cfg, epochs):
    micro_results = read_microbatch_timings(cfg.microbatch_csv)
    old_row = old_row_micro8_timings(cfg.prior_timing_json)
    selected = read_best_microbatches(cfg.microbatch_csv)
    four = {
        (row["block_type"], row["grid"]): row
        for row in micro_results
        if row["status"] == "ok"
        and row["micro_batch"] == selected[(row["block_type"], row["grid"])]
    }
    single_actual = {
        (row["block_type"], row["grid"]): float(
            np.mean([epoch["epoch_wall_sec"] for epoch in row["history"]])
        )
        for row in results
    }
    rows = []
    for block_type in cfg.block_types:
        for grid in (8, 16, 32):
            four_row = four[(block_type, grid)]
            if grid in cfg.grids:
                single_sec = single_actual[(block_type, grid)]
                single_source = "60_epoch_probe_mean"
            else:
                baseline_four = next(
                    row
                    for row in micro_results
                    if row["status"] == "ok"
                    and row["block_type"] == block_type
                    and row["grid"] == grid
                    and row["micro_batch"] == 8
                )
                speedup = baseline_four["mean_epoch_sec"] / four_row["mean_epoch_sec"]
                single_sec = old_row[(block_type, grid)] / speedup
                single_source = "prior_row_micro8_scaled_by_real4_speedup"
            cell_hours = epochs * (15 * single_sec + 10 * four_row["mean_epoch_sec"]) / 3600
            rows.append(
                {
                    "block_type": block_type,
                    "grid": grid,
                    "recommended_micro_batch": selected[(block_type, grid)],
                    "epochs_per_run": epochs,
                    "single_branch_runs": 15,
                    "four_branch_runs": 10,
                    "total_runs": 25,
                    "single_epoch_sec": single_sec,
                    "single_timing_source": single_source,
                    "four_epoch_sec": four_row["mean_epoch_sec"],
                    "single_hours_per_run": epochs * single_sec / 3600,
                    "four_hours_per_run": epochs * four_row["mean_epoch_sec"] / 3600,
                    "cell_gpu_hours": cell_hours,
                }
            )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path, results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for axis, block_type in zip(axes, ("gru", "mamba")):
        for result in results:
            if result["block_type"] != block_type:
                continue
            epochs = [row["epoch"] for row in result["history"]]
            axis.plot(
                epochs,
                [row["test_acc"] * 100 for row in result["history"]],
                label=f"grid{result['grid']}",
                linewidth=1.6,
            )
        axis.set_title(block_type.upper())
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Test accuracy (%)")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_report(path, analyses, recommendation, cost_sets, cfg):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Convergence Probe\n\n")
        handle.write(
            "CIFAR-10 row variant, seed 0, effective batch 128, 60 epochs. All grids "
            "within a block use the micro-batch selected by the timing pilot.\n\n"
        )
        handle.write("| block | grid | micro | best acc | best epoch | plateau status | plateau epoch |\n")
        handle.write("|---|---:|---:|---:|---:|---|---:|\n")
        for row in analyses:
            handle.write(
                f"| {row['block_type']} | {row['grid']} | {row['micro_batch']} | "
                f"{row['best_test_acc'] * 100:.2f}% | {row['best_epoch']} | "
                f"{row['status']} | {row['plateau_epoch'] or ''} |\n"
            )
        handle.write("\n## Epoch recommendation\n\n")
        handle.write(f"Status: `{recommendation['status']}`. ")
        if recommendation["recommended_epochs"]:
            handle.write(f"Use **{recommendation['recommended_epochs']} epochs** for every grid.\n\n")
        else:
            handle.write(
                "Do not choose a final epoch count yet; extend the probe to "
                f"**{recommendation['next_probe_target_epochs']} epochs**. This is a confirmation "
                "target, not a claim that convergence at 100 epochs has already been observed.\n\n"
            )
        handle.write("## 150-run estimate\n\n")
        handle.write("| scenario | epochs/run | GPU-hours | one-GPU days |\n")
        handle.write("|---|---:|---:|---:|\n")
        for label, costs in cost_sets:
            total = sum(row["cell_gpu_hours"] for row in costs)
            handle.write(
                f"| {label} | {costs[0]['epochs_per_run']} | {total:.1f} | {total / 24:.1f} |\n"
            )
        handle.write("\nThe 60-epoch row is a measured lower bound. The 100-epoch row is a linear "
                     "planning scenario pending convergence confirmation.\n\n")
        handle.write(
            "| block | grid | micro | single h/run | four h/run | cell GPU-h |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        detail_costs = cost_sets[-1][1]
        for row in detail_costs:
            handle.write(
                f"| {row['block_type']} | {row['grid']} | "
                f"{row['recommended_micro_batch']} | {row['single_hours_per_run']:.2f} | "
                f"{row['four_hours_per_run']:.2f} | {row['cell_gpu_hours']:.1f} |\n"
            )
        total = sum(row["cell_gpu_hours"] for row in detail_costs)
        handle.write(
            f"\n100-epoch planning total: **150 runs**, **{total:.1f} GPU-hours** "
            f"(**{total / 24:.1f} one-GPU days**). Grid16 single-branch timing is estimated "
            "from the prior full row/micro8 run scaled by the measured real_4dir batch speedup; "
            "grid8 and grid32 single-branch timings come directly from the 60-epoch probes.\n"
        )


def save_all(cfg, results):
    analyses, recommendation = analyze(results, cfg)
    costs_60 = cost_estimate(results, cfg, 60) if len(results) == 4 else []
    costs_100 = cost_estimate(results, cfg, 100) if len(results) == 4 else []
    payload = {
        "config": asdict(cfg),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "mamba_fast_path": False,
        },
        "results": results,
        "analyses": analyses,
        "recommendation": recommendation,
        "cost_estimate_60": costs_60,
        "cost_estimate_100": costs_100,
    }
    with open(os.path.join(cfg.outdir, "convergence_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(os.path.join(cfg.outdir, "convergence_curves.csv"), curve_rows(results, cfg))
    write_csv(os.path.join(cfg.outdir, "plateau_analysis.csv"), analyses)
    if costs_60:
        write_csv(os.path.join(cfg.outdir, "main_sweep_150_cost.csv"), costs_60)
        write_csv(os.path.join(cfg.outdir, "main_sweep_150_cost_epochs60.csv"), costs_60)
        write_csv(os.path.join(cfg.outdir, "main_sweep_150_cost_epochs100.csv"), costs_100)
        plot_curves(os.path.join(cfg.outdir, "convergence_curves.png"), results)
        write_report(
            os.path.join(cfg.outdir, "CONVERGENCE_AND_COST_REPORT.md"),
            analyses,
            recommendation,
            [("measured lower bound", costs_60), ("100-epoch planning", costs_100)],
            cfg,
        )


def load_results(cfg):
    path = os.path.join(cfg.outdir, "convergence_results.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("results", [])


def main():
    args = parse_args()
    cfg = Config(
        data_root=args.data_root,
        outdir=args.outdir,
        microbatch_csv=args.microbatch_csv,
        prior_timing_json=args.prior_timing_json,
        block_types=tuple(args.block_types),
        grids=tuple(args.grids),
        effective_batch=args.effective_batch,
        epochs=args.epochs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        pos_mode=args.pos_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        plateau_window=args.plateau_window,
        plateau_tolerance=args.plateau_tolerance,
        plateau_confirmation=args.plateau_confirmation,
    )
    if cfg.epochs != 60:
        raise ValueError("Formal convergence probe requires 60 epochs")
    if not torch.cuda.is_available() or not HAS_MAMBA:
        raise RuntimeError("CUDA and mamba_ssm are required")
    best_micro = read_best_microbatches(cfg.microbatch_csv)
    os.makedirs(cfg.outdir, exist_ok=True)
    results = [] if args.no_resume else load_results(cfg)
    completed = {row["key"] for row in results}
    for block_type in cfg.block_types:
        for grid in cfg.grids:
            key = run_key(block_type, grid)
            if key in completed:
                print(f"RESUME skip {key}", flush=True)
                continue
            micro_batch = best_micro[(block_type, grid)]
            if cfg.effective_batch % micro_batch:
                raise ValueError("Selected micro-batch does not divide effective batch")
            print(f"START {key} micro={micro_batch}", flush=True)
            result, arrays = run_probe(cfg, block_type, grid, micro_batch)
            prediction_path = os.path.join(
                cfg.outdir, f"predictions_{block_type}_grid{grid}_seed{cfg.seed}.npz"
            )
            np.savez_compressed(prediction_path, **arrays)
            result["prediction_npz"] = prediction_path
            results.append(result)
            completed.add(key)
            save_all(cfg, results)
            print(f"DONE {key}", flush=True)
    save_all(cfg, results)
    print(f"saved convergence probe to {cfg.outdir}", flush=True)


if __name__ == "__main__":
    main()
