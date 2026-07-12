import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mamba_scan_study.data.real_datasets import build_real_loaders
from mamba_scan_study.data.synthetic import build_synthetic_dataset
from mamba_scan_study.models.backbone import HAS_MAMBA, MultiDirBackbone


@dataclass
class Stage0Config:
    dataset: str = "synthetic"
    data_root: str = "data"
    task_dir: str = "vertical"
    signal_strength: str = "single_patch"
    amplitude: float = 2.5
    noise_std: float = 1.0
    shuffle_source: bool = False
    grid_size: int = 8
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    n_classes: int = 2
    train_samples: int = 4096
    test_samples: int = 1024
    block_type: str = "gru"
    branch_dirs: str = "row"
    n_branches: int | None = None
    bidirectional: bool = False
    d_model: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    shuffle_order: bool = False
    pos_mode: str = "seq_learned"
    readout_mode: str = "target"
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 0
    seeds: list[int] = field(default_factory=lambda: [0])
    outdir: str = "mamba_scan_study/outputs/stage0"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "cifar10", "tiny-imagenet"], default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--task-dir", choices=["vertical", "horizontal", "diagonal"], default=None)
    p.add_argument("--signal-strength", choices=["line", "single_patch", "single_pixel"], default=None)
    p.add_argument("--amplitude", type=float, default=None)
    p.add_argument("--noise-std", type=float, default=None)
    p.add_argument("--shuffle-source", action="store_true")
    p.add_argument("--grid-size", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--block-type", choices=["gru", "mamba"], default=None)
    p.add_argument("--branch-dirs", type=str, default=None)
    p.add_argument("--n-branches", type=int, default=None)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--shuffle-order", action="store_true")
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--pos-mode", choices=["none", "seq_learned", "xy_learned", "xy_sincos"], default=None)
    p.add_argument("--readout-mode", choices=["target", "mean_pool"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--train-samples", type=int, default=None)
    p.add_argument("--test-samples", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def build_config(args):
    cfg = Stage0Config()
    for name, value in vars(args).items():
        if name == "smoke" or value is None:
            continue
        setattr(cfg, name, value)
    if args.smoke:
        cfg.epochs = 1
        cfg.train_samples = 256
        cfg.test_samples = 128
        cfg.batch_size = 64
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.seeds = [0]
    if cfg.dataset == "synthetic":
        cfg.img_size = cfg.grid_size * cfg.patch_size
        cfg.n_classes = 2
    elif cfg.dataset == "cifar10":
        cfg.img_size = 32
        cfg.in_chans = 3
        cfg.n_classes = 10
        cfg.readout_mode = "mean_pool"
    elif cfg.dataset == "tiny-imagenet":
        cfg.img_size = 64
        cfg.in_chans = 3
        cfg.n_classes = 200
        cfg.readout_mode = "mean_pool"
    return cfg


def build_loaders(cfg, seed):
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    if cfg.dataset == "synthetic":
        common = dict(
            grid_size=cfg.grid_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            task_dir=cfg.task_dir,
            signal_strength=cfg.signal_strength,
            amplitude=cfg.amplitude,
            noise_std=cfg.noise_std,
            shuffle_source=cfg.shuffle_source,
        )
        train_set = build_synthetic_dataset(
            "train", num_samples=cfg.train_samples, seed=seed, **common
        )
        test_set = build_synthetic_dataset(
            "test", num_samples=cfg.test_samples, seed=seed, **common
        )
        return (
            DataLoader(
                train_set,
                batch_size=cfg.batch_size,
                shuffle=True,
                drop_last=True,
                generator=train_generator,
            ),
            DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False),
        )
    return build_real_loaders(
        cfg.dataset,
        cfg.data_root,
        cfg.batch_size,
        cfg.num_workers,
        img_size=cfg.img_size,
        download=True,
        generator=train_generator,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model):
    # Lightweight proxy for logging. Exact FLOPs can be added later with a profiler.
    return {
        "note": "rough proxy, not profiler FLOPs",
        "params_times_tokens": count_params(model) * model.L,
    }


def batch_logits(model, batch, cfg, device):
    if cfg.dataset == "synthetic":
        images, labels, target_row, target_col, _source_row, _source_col = batch
        images = images.to(device)
        labels = labels.to(device)
        if cfg.readout_mode == "target":
            feat2d = model.forward_features(images)
            logits = model.classify_from_target(feat2d, target_row, target_col)
            return logits, labels
        if cfg.readout_mode == "mean_pool":
            logits, _ = model(images)
            return logits, labels
        raise ValueError(f"unknown readout_mode={cfg.readout_mode!r}")

    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    logits, _ = model(images)
    return logits, labels


def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, labels = batch_logits(model, batch, cfg, device)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total += labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        logits, labels = batch_logits(model, batch, cfg, device)
        loss = F.cross_entropy(logits, labels)
        total += labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
    return {"loss": loss_sum / total, "acc": correct / total}


def run_seed(cfg, seed, device):
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
        n_branches=cfg.n_branches,
        dropout=cfg.dropout,
        shuffle_order=cfg.shuffle_order,
        pos_mode=cfg.pos_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    result = {
        "seed": seed,
        "branch_dirs": model.branch_dirs,
        "pos_mode": cfg.pos_mode,
        "readout_mode": cfg.readout_mode,
        "amplitude": cfg.amplitude,
        "noise_std": cfg.noise_std,
        "shuffle_source": cfg.shuffle_source,
        "q1_eligible": cfg.dataset == "synthetic"
        and cfg.readout_mode == "target"
        and not cfg.shuffle_source,
        "param_count": count_params(model),
        "flops": estimate_flops(model),
        "history": [],
    }
    best_acc = 0.0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg)
        test_metrics = evaluate(model, test_loader, device, cfg)
        best_acc = max(best_acc, test_metrics["acc"])
        row = {"epoch": epoch + 1, "train": train_metrics, "test": test_metrics}
        result["history"].append(row)
        print(
            f"seed={seed} ep={epoch + 1}/{cfg.epochs} "
            f"train={train_metrics['acc']:.4f} test={test_metrics['acc']:.4f}"
        )
    result["best_acc"] = best_acc
    result["elapsed_sec"] = time.time() - t0
    return result


def save_results(cfg, results):
    os.makedirs(cfg.outdir, exist_ok=True)
    payload = {"config": asdict(cfg), "results": results, "has_mamba": HAS_MAMBA}
    json_path = os.path.join(cfg.outdir, "stage0_results.json")
    csv_path = os.path.join(cfg.outdir, "stage0_results.csv")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "branch_dirs",
                "pos_mode",
                "readout_mode",
                "amplitude",
                "noise_std",
                "shuffle_source",
                "q1_eligible",
                "param_count",
                "best_acc",
                "elapsed_sec",
                "last_test_acc",
                "last_test_loss",
            ],
        )
        writer.writeheader()
        for res in results:
            last = res["history"][-1]["test"]
            writer.writerow(
                {
                    "seed": res["seed"],
                    "branch_dirs": ",".join(res["branch_dirs"]),
                    "pos_mode": res["pos_mode"],
                    "readout_mode": res["readout_mode"],
                    "amplitude": res["amplitude"],
                    "noise_std": res["noise_std"],
                    "shuffle_source": res["shuffle_source"],
                    "q1_eligible": res["q1_eligible"],
                    "param_count": res["param_count"],
                    "best_acc": res["best_acc"],
                    "elapsed_sec": res["elapsed_sec"],
                    "last_test_acc": last["acc"],
                    "last_test_loss": last["loss"],
                }
            )
    return json_path, csv_path


def main():
    cfg = build_config(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} block={cfg.block_type} HAS_MAMBA={HAS_MAMBA}")
    print(
        f"dataset={cfg.dataset} branch_dirs={cfg.branch_dirs} "
        f"readout={cfg.readout_mode} pos={cfg.pos_mode} seeds={cfg.seeds}"
    )
    results = [run_seed(cfg, seed, device) for seed in cfg.seeds]
    json_path, csv_path = save_results(cfg, results)
    print(f"saved json: {json_path}")
    print(f"saved csv : {csv_path}")


if __name__ == "__main__":
    main()
