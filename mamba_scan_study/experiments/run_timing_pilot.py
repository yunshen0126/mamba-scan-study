import argparse
import csv
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mamba_scan_study.data.real_datasets import build_real_loaders
from mamba_scan_study.models.backbone import HAS_MAMBA, MultiDirBackbone


DATASET_META = {
    "cifar10": {"img_size": 32, "n_classes": 10},
    "cifar10_up64": {"img_size": 64, "n_classes": 10},
    "tiny_imagenet": {"img_size": 64, "n_classes": 200},
}
BRANCHES = {
    "row": "row",
    "real_4dir": "row,col,diag,anti_diag",
}


@dataclass
class PilotConfig:
    datasets: tuple[str, ...]
    block_types: tuple[str, ...]
    grids: tuple[int, ...]
    data_root: str
    outdir: str
    epochs: int
    batch_size: int
    accum_steps: int
    d_model: int
    n_layers: int
    pos_mode: str
    lr: float
    weight_decay: float
    grad_clip: float
    num_workers: int
    seed: int
    amp: bool
    calibration_warmup: int
    calibration_steps: int
    max_train_batches: int
    max_test_batches: int
    projection_epochs: tuple[int, ...]
    electricity_cny_per_kwh: float


def parse_args():
    parser = argparse.ArgumentParser(description="Two-epoch timing pilot; never launches a sweep.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_META),
        choices=list(DATASET_META),
    )
    parser.add_argument(
        "--block-types", nargs="+", default=["gru", "mamba"], choices=["gru", "mamba"]
    )
    parser.add_argument("--grids", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--outdir", default="mamba_scan_study/outputs/timing_pilot")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--pos-mode", default="xy_learned", choices=["none", "xy_learned"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--calibration-warmup", type=int, default=2)
    parser.add_argument("--calibration-steps", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--projection-epochs", nargs="+", type=int, default=[30, 100])
    parser.add_argument("--full-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--electricity-cny-per-kwh", type=float, default=0.60)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PowerSampler:
    def __init__(self, interval_sec=2.0):
        self.interval_sec = interval_sec
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                value = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).splitlines()[0]
                self.samples.append(float(value.strip()))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.stop_event.wait(self.interval_sec)

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)

    @property
    def mean_watts(self):
        return float(np.mean(self.samples)) if self.samples else None


def build_loaders(dataset, data_root, batch_size, num_workers, seed):
    meta = DATASET_META[dataset]
    generator = torch.Generator().manual_seed(seed)
    loader_name = "tiny-imagenet" if dataset == "tiny_imagenet" else dataset
    return build_real_loaders(
        loader_name,
        data_root,
        batch_size,
        num_workers=num_workers,
        img_size=meta["img_size"],
        download=True,
        generator=generator,
    )


def build_model(dataset, block_type, grid, variant, cfg, device):
    meta = DATASET_META[dataset]
    if meta["img_size"] % grid:
        raise ValueError(f"img_size={meta['img_size']} is not divisible by grid={grid}")
    patch_size = meta["img_size"] // grid
    return MultiDirBackbone(
        img_size=meta["img_size"],
        patch_size=patch_size,
        in_chans=3,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        block_type=block_type,
        n_classes=meta["n_classes"],
        branch_dirs=BRANCHES[variant],
        pos_mode=cfg.pos_mode,
    ).to(device)


def limited_batches(loader, maximum):
    for batch_index, batch in enumerate(loader):
        if maximum and batch_index >= maximum:
            break
        yield batch_index, batch


def train_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0
    correct = 0
    loss_sum = 0.0
    batch_count = 0
    for batch_index, (images, labels) in limited_batches(loader, cfg.max_train_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            logits, _ = model(images)
            raw_loss = F.cross_entropy(logits, labels)
            loss = raw_loss / cfg.accum_steps
        scaler.scale(loss).backward()
        batch_count = batch_index + 1
        is_boundary = batch_count % cfg.accum_steps == 0
        is_last = batch_count == len(loader) or (
            cfg.max_train_batches and batch_count == cfg.max_train_batches
        )
        if is_boundary or is_last:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        loss_sum += raw_loss.item() * labels.numel()
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "samples": total,
        "batches": batch_count,
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg, collect=False):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    logits_parts = []
    labels_parts = []
    batch_count = 0
    for batch_index, (images, labels) in limited_batches(loader, cfg.max_test_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
        total += labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
        batch_count = batch_index + 1
        if collect:
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels.cpu())
    metrics = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "samples": total,
        "batches": batch_count,
    }
    if not collect:
        return metrics, None
    logits_array = torch.cat(logits_parts).numpy()
    labels_array = torch.cat(labels_parts).numpy()
    predictions = logits_array.argmax(axis=1)
    arrays = {
        "sample_index": np.arange(labels_array.shape[0], dtype=np.int64),
        "labels": labels_array,
        "predictions": predictions,
        "logits": logits_array,
    }
    return metrics, arrays


def cached_batch(loader, device):
    images, labels = next(iter(loader))
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def benchmark_variant(dataset, block_type, grid, variant, train_batch, test_batch, cfg, device):
    set_seed(cfg.seed + 1000)
    model = build_model(dataset, block_type, grid, variant, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    images, labels = train_batch
    test_images, _test_labels = test_batch

    def train_step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    @torch.no_grad()
    def eval_step():
        model.eval()
        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            model(test_images)

    for _ in range(cfg.calibration_warmup):
        train_step()
        eval_step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(cfg.calibration_steps):
        train_step()
    synchronize(device)
    train_sec = (time.perf_counter() - t0) / cfg.calibration_steps
    t0 = time.perf_counter()
    for _ in range(cfg.calibration_steps):
        eval_step()
    synchronize(device)
    eval_sec = (time.perf_counter() - t0) / cfg.calibration_steps
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "train_step_sec": train_sec,
        "eval_step_sec": eval_sec,
        "peak_memory_mb": peak_mb,
        "param_count": params,
    }


def cell_key(dataset, block_type, grid):
    return f"{dataset}|{block_type}|{grid}"


def run_cell(dataset, block_type, grid, train_loader, test_loader, cfg, device):
    if block_type == "mamba" and not HAS_MAMBA:
        raise RuntimeError("mamba_ssm is unavailable in the selected Python environment")
    set_seed(cfg.seed)
    model = build_model(dataset, block_type, grid, "row", cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    epochs = []
    final_arrays = None
    with PowerSampler() as power_sampler:
        for epoch in range(cfg.epochs):
            synchronize(device)
            epoch_start = time.perf_counter()
            train_start = epoch_start
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, cfg)
            synchronize(device)
            train_sec = time.perf_counter() - train_start
            eval_start = time.perf_counter()
            test_metrics, arrays = evaluate(
                model, test_loader, device, cfg, collect=epoch == cfg.epochs - 1
            )
            synchronize(device)
            eval_sec = time.perf_counter() - eval_start
            epoch_sec = time.perf_counter() - epoch_start
            if arrays is not None:
                final_arrays = arrays
            epoch_row = {
                "epoch": epoch + 1,
                "train_sec": train_sec,
                "eval_sec": eval_sec,
                "epoch_wall_sec": epoch_sec,
                "train": train_metrics,
                "test": test_metrics,
            }
            epochs.append(epoch_row)
            print(
                f"CELL dataset={dataset} block={block_type} grid={grid} "
                f"epoch={epoch + 1}/{cfg.epochs} train_sec={train_sec:.1f} "
                f"eval_sec={eval_sec:.1f} peak_mb="
                f"{torch.cuda.max_memory_allocated(device) / (1024**2):.0f}",
                flush=True,
            )

    row_peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    row_params = sum(parameter.numel() for parameter in model.parameters())
    mean_power_w = power_sampler.mean_watts
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_batch = cached_batch(train_loader, device)
    test_batch = cached_batch(test_loader, device)
    calibration = {}
    for variant in ("row", "real_4dir"):
        calibration[variant] = benchmark_variant(
            dataset, block_type, grid, variant, train_batch, test_batch, cfg, device
        )
    row_cal = calibration["row"]
    four_cal = calibration["real_4dir"]
    train_ratio = four_cal["train_step_sec"] / row_cal["train_step_sec"]
    eval_ratio = four_cal["eval_step_sec"] / row_cal["eval_step_sec"]
    return {
        "key": cell_key(dataset, block_type, grid),
        "dataset": dataset,
        "block_type": block_type,
        "grid": grid,
        "sequence_length": grid * grid,
        "img_size": DATASET_META[dataset]["img_size"],
        "patch_size": DATASET_META[dataset]["img_size"] // grid,
        "pilot_variant": "row",
        "seed": cfg.seed,
        "epochs": epochs,
        "mean_epoch_sec": float(np.mean([row["epoch_wall_sec"] for row in epochs])),
        "mean_train_sec": float(np.mean([row["train_sec"] for row in epochs])),
        "mean_eval_sec": float(np.mean([row["eval_sec"] for row in epochs])),
        "row_peak_memory_mb": row_peak_mb,
        "row_param_count": row_params,
        "mean_power_w": mean_power_w,
        "power_samples": len(power_sampler.samples),
        "calibration": calibration,
        "four_to_single_train_ratio": train_ratio,
        "four_to_single_eval_ratio": eval_ratio,
        "four_peak_memory_mb": four_cal["peak_memory_mb"],
        "predictions": final_arrays,
    }


def serializable_result(result):
    return {key: value for key, value in result.items() if key != "predictions"}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_csv_rows(results):
    rows = []
    for result in results:
        row = {
            "dataset": result["dataset"],
            "block_type": result["block_type"],
            "grid": result["grid"],
            "sequence_length": result["sequence_length"],
            "patch_size": result["patch_size"],
            "mean_epoch_sec": result["mean_epoch_sec"],
            "epoch1_sec": result["epochs"][0]["epoch_wall_sec"],
            "epoch2_sec": result["epochs"][-1]["epoch_wall_sec"],
            "mean_train_sec": result["mean_train_sec"],
            "mean_eval_sec": result["mean_eval_sec"],
            "row_peak_memory_mb": result["row_peak_memory_mb"],
            "four_peak_memory_mb": result["four_peak_memory_mb"],
            "four_to_single_train_ratio": result["four_to_single_train_ratio"],
            "four_to_single_eval_ratio": result["four_to_single_eval_ratio"],
            "mean_power_w": result["mean_power_w"],
        }
        rows.append(row)
    return rows


def estimate_cost_rows(
    results, cfg, single_variants, four_variants, seeds, matrix_name, full_epochs
):
    rows = []
    for result in results:
        single_epoch_sec = result["mean_epoch_sec"]
        four_epoch_sec = (
            result["mean_train_sec"] * result["four_to_single_train_ratio"]
            + result["mean_eval_sec"] * result["four_to_single_eval_ratio"]
        )
        single_runs = single_variants * seeds
        four_runs = four_variants * seeds
        cell_hours = full_epochs * (
            single_runs * single_epoch_sec + four_runs * four_epoch_sec
        ) / 3600.0
        mean_power_w = result["mean_power_w"] or 0.0
        energy_kwh = cell_hours * mean_power_w / 1000.0
        rows.append(
            {
                "matrix": matrix_name,
                "dataset": result["dataset"],
                "block_type": result["block_type"],
                "grid": result["grid"],
                "sequence_length": result["sequence_length"],
                "single_branch_runs": single_runs,
                "four_branch_runs": four_runs,
                "total_runs": single_runs + four_runs,
                "full_epochs_per_run": full_epochs,
                "single_branch_hours_per_run": full_epochs * single_epoch_sec / 3600.0,
                "four_branch_hours_per_run": full_epochs * four_epoch_sec / 3600.0,
                "cell_gpu_hours": cell_hours,
                "estimated_energy_kwh": energy_kwh,
                "electricity_cny": energy_kwh * cfg.electricity_cny_per_kwh,
                "compute_cny_at_1_per_gpu_hour": cell_hours,
                "compute_cny_at_2_per_gpu_hour": cell_hours * 2.0,
                "compute_cny_at_3_per_gpu_hour": cell_hours * 3.0,
            }
        )
    return rows


def write_report(path, results, cost_sets, cfg):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Timing Pilot Report\n\n")
        handle.write(
            f"Environment: `{torch.__version__}`, CUDA `{torch.version.cuda}`, "
            f"GPU `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}`. "
            f"Pilot uses row scan, {cfg.epochs} full epochs, micro-batch {cfg.batch_size}, "
            f"gradient accumulation {cfg.accum_steps}, AMP={cfg.amp}. No CNN stem was added.\n\n"
        )
        handle.write("## Measured cells\n\n")
        handle.write(
            "| dataset | block | grid | L | epoch 1 (min) | epoch 2 (min) | "
            "row peak allocated MiB | 4dir peak allocated MiB | 4dir train ratio |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for result in results:
            handle.write(
                f"| {result['dataset']} | {result['block_type']} | {result['grid']} | "
                f"{result['sequence_length']} | {result['epochs'][0]['epoch_wall_sec'] / 60:.2f} | "
                f"{result['epochs'][-1]['epoch_wall_sec'] / 60:.2f} | "
                f"{result['row_peak_memory_mb']:.0f} | {result['four_peak_memory_mb']:.0f} | "
                f"{result['four_to_single_train_ratio']:.2f} |\n"
            )
        handle.write("\n## Matrix estimate\n\n")
        handle.write("| matrix | epochs/run | runs | GPU-hours | one-GPU days | electricity CNY |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for matrix_name, full_epochs, rows in cost_sets:
            hours = sum(row["cell_gpu_hours"] for row in rows)
            electricity = sum(row["electricity_cny"] for row in rows)
            runs = sum(row["total_runs"] for row in rows)
            handle.write(
                f"| {matrix_name} | {full_epochs} | {runs} | {hours:.1f} | "
                f"{hours / 24:.1f} | {electricity:.1f} |\n"
            )
        handle.write("\n")
        handle.write(
            "The original matrix has 2 single-branch and 2 four-branch variants. Adding mandatory "
            "`shuffle_row` changes this to 3 single-branch and 2 four-branch variants. Both use "
            "5 seeds per cell. CSV files include compute-price scenarios at 1/2/3 CNY per GPU-hour.\n\n"
        )
        handle.write(
            "Electricity is a hardware-only estimate based on sampled GPU board power. It excludes "
            "the rest of the laptop, cooling, failed runs, setup time, and any cloud rental price.\n"
        )


def save_all(results, cfg):
    os.makedirs(cfg.outdir, exist_ok=True)
    clean_results = [serializable_result(result) for result in results]
    with open(os.path.join(cfg.outdir, "pilot_results.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(cfg),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "has_mamba": HAS_MAMBA,
                "results": clean_results,
            },
            handle,
            indent=2,
        )
    write_csv(os.path.join(cfg.outdir, "pilot_results.csv"), result_csv_rows(results))
    cost_sets = []
    for full_epochs in cfg.projection_epochs:
        original_rows = estimate_cost_rows(
            results, cfg, 2, 2, 5, "original_360", full_epochs
        )
        revised_rows = estimate_cost_rows(
            results, cfg, 3, 2, 5, "with_shuffle_row_450", full_epochs
        )
        write_csv(
            os.path.join(cfg.outdir, f"matrix_cost_original_360_epochs{full_epochs}.csv"),
            original_rows,
        )
        write_csv(
            os.path.join(
                cfg.outdir, f"matrix_cost_with_shuffle_row_450_epochs{full_epochs}.csv"
            ),
            revised_rows,
        )
        if full_epochs == 30:
            write_csv(os.path.join(cfg.outdir, "matrix_cost_original_360.csv"), original_rows)
            write_csv(
                os.path.join(cfg.outdir, "matrix_cost_with_shuffle_row_450.csv"),
                revised_rows,
            )
        cost_sets.extend(
            [
                ("original_360", full_epochs, original_rows),
                ("with_shuffle_row_450", full_epochs, revised_rows),
            ]
        )
    if len(results) == len(cfg.datasets) * len(cfg.block_types) * len(cfg.grids):
        write_report(
            os.path.join(cfg.outdir, "TIMING_PILOT_REPORT.md"),
            results,
            cost_sets,
            cfg,
        )


def load_completed(cfg):
    path = os.path.join(cfg.outdir, "pilot_results.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("results", [])


def main():
    args = parse_args()
    cfg = PilotConfig(
        datasets=tuple(args.datasets),
        block_types=tuple(args.block_types),
        grids=tuple(args.grids),
        data_root=args.data_root,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        d_model=args.d_model,
        n_layers=args.n_layers,
        pos_mode=args.pos_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        calibration_warmup=args.calibration_warmup,
        calibration_steps=args.calibration_steps,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
        projection_epochs=tuple(
            [args.full_epochs] if args.full_epochs is not None else args.projection_epochs
        ),
        electricity_cny_per_kwh=args.electricity_cny_per_kwh,
    )
    if cfg.epochs != 2 and not (cfg.max_train_batches or cfg.max_test_batches):
        raise ValueError("The formal timing pilot must use exactly 2 epochs")
    if cfg.batch_size != 8 or cfg.accum_steps != 16:
        print("WARNING: non-protocol batch/accum settings are intended only for smoke tests")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Timing pilot requires CUDA")
    os.makedirs(cfg.outdir, exist_ok=True)
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"cuda={torch.version.cuda} HAS_MAMBA={HAS_MAMBA}",
        flush=True,
    )
    results = [] if args.no_resume else load_completed(cfg)
    completed = {result["key"] for result in results}
    for dataset in cfg.datasets:
        train_loader, test_loader = build_loaders(
            dataset, cfg.data_root, cfg.batch_size, cfg.num_workers, cfg.seed
        )
        print(
            f"dataset={dataset} train={len(train_loader.dataset)} test={len(test_loader.dataset)}",
            flush=True,
        )
        for block_type in cfg.block_types:
            for grid in cfg.grids:
                key = cell_key(dataset, block_type, grid)
                if key in completed:
                    print(f"RESUME skip completed {key}", flush=True)
                    continue
                print(f"START {key}", flush=True)
                result = run_cell(
                    dataset, block_type, grid, train_loader, test_loader, cfg, device
                )
                arrays = result.pop("predictions")
                prediction_path = os.path.join(
                    cfg.outdir, f"predictions_{dataset}_{block_type}_grid{grid}_seed{cfg.seed}.npz"
                )
                np.savez_compressed(prediction_path, **arrays)
                result["prediction_npz"] = prediction_path
                results.append(result)
                completed.add(key)
                save_all(results, cfg)
                print(f"DONE {key} predictions={prediction_path}", flush=True)
    save_all(results, cfg)
    print(f"saved timing pilot to {cfg.outdir}", flush=True)


if __name__ == "__main__":
    main()
