import argparse
import csv
import gc
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


BRANCH_DIRS = "row,col,diag,anti_diag"


@dataclass
class Config:
    data_root: str
    outdir: str
    block_types: tuple[str, ...]
    grids: tuple[int, ...]
    micro_batches: tuple[int, ...]
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
    max_train_batches: int
    max_test_batches: int


def parse_args():
    parser = argparse.ArgumentParser(description="Real-4dir micro-batch timing pilot.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--outdir", default="mamba_scan_study/outputs/microbatch_pilot")
    parser.add_argument("--block-types", nargs="+", default=["gru", "mamba"])
    parser.add_argument("--grids", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--micro-batches", nargs="+", type=int, default=[8, 32, 64, 128])
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--pos-mode", default="xy_learned")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
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


class GpuSampler:
    def __init__(self, interval_sec=1.0):
        self.interval_sec = interval_sec
        self.power_watts = []
        self.memory_used_mb = []
        self.stop_event = threading.Event()
        self.thread = None

    def _run(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).splitlines()[0]
                power, memory = [float(part.strip()) for part in output.split(",")]
                self.power_watts.append(power)
                self.memory_used_mb.append(memory)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.stop_event.wait(self.interval_sec)

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)

    def summary(self):
        return {
            "mean_power_w": float(np.mean(self.power_watts)) if self.power_watts else None,
            "peak_nvidia_memory_mb": max(self.memory_used_mb) if self.memory_used_mb else None,
            "gpu_samples": len(self.memory_used_mb),
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


def make_model(cfg, block_type, grid, device):
    return MultiDirBackbone(
        img_size=32,
        patch_size=32 // grid,
        in_chans=3,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        block_type=block_type,
        n_classes=10,
        branch_dirs=BRANCH_DIRS,
        pos_mode=cfg.pos_mode,
    ).to(device)


def iter_limited(loader, maximum):
    for index, batch in enumerate(loader):
        if maximum and index >= maximum:
            break
        yield index, batch


def train_epoch(model, loader, optimizer, scaler, device, cfg, accum_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = correct = batch_count = 0
    loss_sum = 0.0
    loader_batches = len(loader)
    for batch_index, (images, labels) in iter_limited(loader, cfg.max_train_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits, _ = model(images)
            raw_loss = F.cross_entropy(logits, labels)
            loss = raw_loss / accum_steps
        scaler.scale(loss).backward()
        batch_count = batch_index + 1
        final_batch = batch_count == loader_batches or (
            cfg.max_train_batches and batch_count == cfg.max_train_batches
        )
        if batch_count % accum_steps == 0 or final_batch:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += raw_loss.item() * labels.numel()
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "samples": total,
        "micro_steps": batch_count,
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg, collect=False):
    model.eval()
    total = correct = batch_count = 0
    loss_sum = 0.0
    logits_parts = []
    labels_parts = []
    for batch_index, (images, labels) in iter_limited(loader, cfg.max_test_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
        total += labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
        batch_count = batch_index + 1
        if collect:
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels.cpu())
    metrics = {
        "loss": loss_sum / total,
        "acc": correct / total,
        "samples": total,
        "micro_steps": batch_count,
    }
    if not collect:
        return metrics, None
    logits = torch.cat(logits_parts).numpy()
    labels = torch.cat(labels_parts).numpy()
    arrays = {
        "sample_index": np.arange(labels.shape[0], dtype=np.int64),
        "labels": labels,
        "predictions": logits.argmax(1),
        "logits": logits,
    }
    return metrics, arrays


def cell_key(block_type, grid, micro_batch):
    return f"{block_type}|{grid}|{micro_batch}"


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def blank_result(cfg, block_type, grid, micro_batch):
    return {
        "key": cell_key(block_type, grid, micro_batch),
        "dataset": "cifar10",
        "block_type": block_type,
        "variant": "real_4dir",
        "grid": grid,
        "sequence_length": grid * grid,
        "micro_batch": micro_batch,
        "accum_steps": cfg.effective_batch // micro_batch,
        "effective_batch": cfg.effective_batch,
        "seed": cfg.seed,
        "status": "pending",
        "error": "",
        "epochs": [],
    }


def run_cell(cfg, block_type, grid, micro_batch, device):
    result = blank_result(cfg, block_type, grid, micro_batch)
    train_loader, test_loader = make_loaders(cfg, micro_batch)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = make_model(cfg, block_type, grid, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    final_arrays = None
    with GpuSampler() as gpu_sampler:
        for epoch in range(cfg.epochs):
            synchronize(device)
            epoch_start = time.perf_counter()
            train_start = epoch_start
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                cfg,
                result["accum_steps"],
            )
            synchronize(device)
            train_sec = time.perf_counter() - train_start
            eval_start = time.perf_counter()
            test_metrics, arrays = evaluate(
                model, test_loader, device, cfg, collect=epoch == cfg.epochs - 1
            )
            synchronize(device)
            eval_sec = time.perf_counter() - eval_start
            if arrays is not None:
                final_arrays = arrays
            result["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_sec": train_sec,
                    "eval_sec": eval_sec,
                    "epoch_wall_sec": time.perf_counter() - epoch_start,
                    "train": train_metrics,
                    "test": test_metrics,
                }
            )
            print(
                f"CELL block={block_type} grid={grid} micro={micro_batch} "
                f"epoch={epoch + 1}/{cfg.epochs} train_sec={train_sec:.1f} "
                f"eval_sec={eval_sec:.1f}",
                flush=True,
            )
    result.update(gpu_sampler.summary())
    result["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    result["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2)
    result["param_count"] = sum(parameter.numel() for parameter in model.parameters())
    result["mean_epoch_sec"] = float(
        np.mean([epoch["epoch_wall_sec"] for epoch in result["epochs"]])
    )
    result["status"] = "ok"
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result, final_arrays


def load_results(cfg):
    path = os.path.join(cfg.outdir, "microbatch_results.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("results", [])


def summary_rows(results):
    baselines = {
        (row["block_type"], row["grid"]): row["mean_epoch_sec"]
        for row in results
        if row["status"] == "ok" and row["micro_batch"] == 8
    }
    rows = []
    for result in results:
        baseline = baselines.get((result["block_type"], result["grid"]))
        mean_epoch = result.get("mean_epoch_sec")
        rows.append(
            {
                "dataset": result["dataset"],
                "block_type": result["block_type"],
                "variant": result["variant"],
                "grid": result["grid"],
                "sequence_length": result["sequence_length"],
                "micro_batch": result["micro_batch"],
                "accum_steps": result["accum_steps"],
                "effective_batch": result["effective_batch"],
                "status": result["status"],
                "epoch1_sec": result["epochs"][0]["epoch_wall_sec"]
                if result["epochs"]
                else "",
                "epoch2_sec": result["epochs"][-1]["epoch_wall_sec"]
                if result["epochs"]
                else "",
                "mean_epoch_sec": mean_epoch if mean_epoch is not None else "",
                "speedup_vs_micro8": baseline / mean_epoch
                if baseline and mean_epoch
                else "",
                "peak_allocated_mb": result.get("peak_allocated_mb", ""),
                "peak_reserved_mb": result.get("peak_reserved_mb", ""),
                "peak_nvidia_memory_mb": result.get("peak_nvidia_memory_mb", ""),
                "mean_power_w": result.get("mean_power_w", ""),
                "error": result.get("error", ""),
            }
        )
    return rows


def best_rows(results):
    rows = []
    for block_type in sorted({row["block_type"] for row in results}):
        for grid in sorted({row["grid"] for row in results}):
            candidates = [
                row
                for row in results
                if row["block_type"] == block_type
                and row["grid"] == grid
                and row["status"] == "ok"
            ]
            if not candidates:
                continue
            baseline = next((row for row in candidates if row["micro_batch"] == 8), None)
            largest = max(candidates, key=lambda row: row["micro_batch"])
            fastest = min(candidates, key=lambda row: row["mean_epoch_sec"])
            rows.append(
                {
                    "block_type": block_type,
                    "grid": grid,
                    "sequence_length": grid * grid,
                    "max_feasible_micro_batch": largest["micro_batch"],
                    "max_feasible_accum_steps": largest["accum_steps"],
                    "max_feasible_peak_nvidia_mb": largest.get("peak_nvidia_memory_mb"),
                    "fastest_micro_batch": fastest["micro_batch"],
                    "fastest_accum_steps": fastest["accum_steps"],
                    "fastest_epoch_sec": fastest["mean_epoch_sec"],
                    "speedup_vs_micro8": baseline["mean_epoch_sec"] / fastest["mean_epoch_sec"]
                    if baseline
                    else None,
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


def write_report(path, rows, best):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Micro-batch Timing Pilot\n\n")
        handle.write(
            "CIFAR-10, real_4dir, two full epochs per cell, effective batch 128. "
            "OOM cells are recorded without retry. Mamba uses the unfused path.\n\n"
        )
        handle.write(
            "| block | grid | micro | accum | status | epoch mean (s) | speedup | "
            "peak allocated MiB | peak reserved MiB | nvidia-smi peak MiB |\n"
        )
        handle.write("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            def number(name, digits=1):
                value = row[name]
                return "" if value == "" or value is None else f"{float(value):.{digits}f}"

            handle.write(
                f"| {row['block_type']} | {row['grid']} | {row['micro_batch']} | "
                f"{row['accum_steps']} | {row['status']} | {number('mean_epoch_sec')} | "
                f"{number('speedup_vs_micro8', 2)} | {number('peak_allocated_mb')} | "
                f"{number('peak_reserved_mb')} | {number('peak_nvidia_memory_mb')} |\n"
            )
        handle.write("\n## Selected batch sizes\n\n")
        handle.write(
            "| block | grid | max feasible micro | fastest micro | fastest accum | speedup |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in best:
            handle.write(
                f"| {row['block_type']} | {row['grid']} | "
                f"{row['max_feasible_micro_batch']} | {row['fastest_micro_batch']} | "
                f"{row['fastest_accum_steps']} | {row['speedup_vs_micro8']:.2f} |\n"
            )


def save_all(cfg, results):
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "microbatch_results.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(cfg),
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "has_mamba": HAS_MAMBA,
                    "mamba_fast_path": False,
                },
                "results": results,
            },
            handle,
            indent=2,
        )
    rows = summary_rows(results)
    best = best_rows(results)
    write_csv(os.path.join(cfg.outdir, "microbatch_results.csv"), rows)
    write_csv(os.path.join(cfg.outdir, "best_microbatch.csv"), best)
    expected = len(cfg.block_types) * len(cfg.grids) * len(cfg.micro_batches)
    if len(results) == expected:
        write_report(os.path.join(cfg.outdir, "MICROBATCH_REPORT.md"), rows, best)


def main():
    args = parse_args()
    cfg = Config(
        data_root=args.data_root,
        outdir=args.outdir,
        block_types=tuple(args.block_types),
        grids=tuple(args.grids),
        micro_batches=tuple(args.micro_batches),
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
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
    )
    for micro_batch in cfg.micro_batches:
        if cfg.effective_batch % micro_batch:
            raise ValueError("Every micro-batch must divide effective_batch")
    if cfg.epochs != 2 and not (cfg.max_train_batches or cfg.max_test_batches):
        raise ValueError("Formal micro-batch pilot requires exactly two epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if "mamba" in cfg.block_types and not HAS_MAMBA:
        raise RuntimeError("mamba_ssm is unavailable")
    os.makedirs(cfg.outdir, exist_ok=True)
    results = [] if args.no_resume else load_results(cfg)
    completed = {row["key"] for row in results}
    print(
        f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"effective_batch={cfg.effective_batch} fast_path=False",
        flush=True,
    )
    for block_type in cfg.block_types:
        for grid in cfg.grids:
            for micro_batch in cfg.micro_batches:
                key = cell_key(block_type, grid, micro_batch)
                if key in completed:
                    print(f"RESUME skip {key}", flush=True)
                    continue
                print(f"START {key}", flush=True)
                try:
                    result, arrays = run_cell(cfg, block_type, grid, micro_batch, torch.device("cuda"))
                    prediction_path = os.path.join(
                        cfg.outdir,
                        f"predictions_{block_type}_grid{grid}_micro{micro_batch}_seed{cfg.seed}.npz",
                    )
                    np.savez_compressed(prediction_path, **arrays)
                    result["prediction_npz"] = prediction_path
                except Exception as exc:
                    if not is_oom(exc):
                        raise
                    result = blank_result(cfg, block_type, grid, micro_batch)
                    result["status"] = "oom"
                    result["error"] = str(exc)
                    result["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
                    result["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024**2)
                    print(f"OOM {key}: {exc}", flush=True)
                    torch.cuda.empty_cache()
                results.append(result)
                completed.add(key)
                save_all(cfg, results)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"DONE {key} status={result['status']}", flush=True)
    save_all(cfg, results)
    print(f"saved micro-batch pilot to {cfg.outdir}", flush=True)


if __name__ == "__main__":
    main()
