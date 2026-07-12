"""
主入口：跑完整对照实验。

用法（命令行可覆盖默认配置）：
  python run.py                         # 用 config.py 的默认设置跑全部
  python run.py --epochs 50 --seeds 0   # 快速试跑
  python run.py --block-type mamba      # 用真实 Mamba（需先装 mamba-ssm）
  python run.py --smoke                 # 30 秒冒烟测试，确认能跑通

Windows 注意：入口已加 if __name__ == '__main__' 保护。
"""
import os
import json
import argparse
import statistics
from dataclasses import asdict

import torch

from config import Config
from data import build_cifar10_loaders
from model import RowScanBackbone, ReconHead, HAS_MAMBA
from train import (
    train_one_epoch, evaluate, measure_inference_speed, lr_lambda_fn,
)


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--block-type", type=str, default=None, choices=["gru", "mamba"])
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--aux-lambda", type=float, default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--variants", type=str, nargs="+", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="冒烟测试：极小设置，确认能跑通")
    return p.parse_args()


def build_config(args):
    cfg = Config()
    if args.data_root is not None: cfg.data_root = args.data_root
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.d_model is not None: cfg.d_model = args.d_model
    if args.n_layers is not None: cfg.n_layers = args.n_layers
    if args.block_type is not None: cfg.block_type = args.block_type
    if args.bidirectional: cfg.bidirectional = True
    if args.aux_lambda is not None: cfg.aux_lambda = args.aux_lambda
    if args.mask_ratio is not None: cfg.mask_ratio = args.mask_ratio
    if args.seeds is not None: cfg.seeds = args.seeds
    if args.variants is not None: cfg.variants = args.variants
    if args.no_amp: cfg.use_amp = False
    if args.outdir is not None: cfg.outdir = args.outdir

    if args.smoke:
        cfg.epochs = 2
        cfg.seeds = [0]
        cfg.batch_size = 64
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.lambda_warmup_epochs = 1
    return cfg


def train_one_variant(variant, cfg, train_loader, test_loader, device):
    backbone = RowScanBackbone(
        img_size=cfg.img_size, patch_size=cfg.patch_size, in_chans=cfg.in_chans,
        d_model=cfg.d_model, n_layers=cfg.n_layers, block_type=cfg.block_type,
        bidirectional=cfg.bidirectional, n_classes=cfg.n_classes,
    ).to(device)

    use_aux = variant in ("row_aux", "col_aux")
    recon_head = None
    if use_aux:
        recon_head = ReconHead(cfg.d_model, cfg.patch_size, cfg.in_chans).to(device)

    params = list(backbone.parameters())
    if recon_head is not None:
        params += list(recon_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: lr_lambda_fn(e, cfg.warmup_epochs, cfg.epochs)
    )
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    cfg.use_amp = use_amp  # 关掉非 CUDA 上的 amp

    best_acc = 0.0
    history = []
    for epoch in range(cfg.epochs):
        tr_acc, lam = train_one_epoch(
            backbone, recon_head, train_loader, optimizer, scaler,
            cfg, device, variant, epoch,
        )
        te_acc = evaluate(backbone, test_loader, device)
        scheduler.step()
        best_acc = max(best_acc, te_acc)
        history.append(te_acc)
        print(f"    [{variant:9s}] ep {epoch+1:3d}/{cfg.epochs}  "
              f"train={tr_acc:.4f}  test={te_acc:.4f}  λ={lam:.3f}")

    speed = measure_inference_speed(backbone, test_loader, device)
    return {
        "variant": variant,
        "best_acc": best_acc,
        "last5_acc": sum(history[-5:]) / len(history[-5:]),
        "history": history,
        "infer_imgs_per_sec": speed,
    }


def main():
    args = parse_args()
    cfg = build_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  CIFAR-10  Row-Scan + Row-Mask Reconstruction Auxiliary")
    print("=" * 70)
    print(f"  device={device}  block={cfg.block_type}  "
          f"bidir={cfg.bidirectional}  HAS_MAMBA={HAS_MAMBA}")
    print(f"  epochs={cfg.epochs}  batch={cfg.batch_size}  d_model={cfg.d_model}  "
          f"layers={cfg.n_layers}  λ={cfg.aux_lambda}  mask_ratio={cfg.mask_ratio}")
    print(f"  seeds={cfg.seeds}  variants={cfg.variants}")
    print("=" * 70)

    train_loader, test_loader = build_cifar10_loaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers, download=True
    )

    os.makedirs(cfg.outdir, exist_ok=True)
    all_results = {}  # variant -> list of best_acc across seeds

    for seed in cfg.seeds:
        print(f"\n########## SEED {seed} ##########")
        set_seed(seed)
        for variant in cfg.variants:
            print(f"\n  ==> training variant: {variant}")
            res = train_one_variant(variant, cfg, train_loader, test_loader, device)
            all_results.setdefault(variant, []).append(res["best_acc"])
            # 存单次结果
            out = os.path.join(cfg.outdir, f"seed{seed}_{variant}.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  ==> {variant}: best={res['best_acc']:.4f}  "
                  f"speed={res['infer_imgs_per_sec']:.0f} img/s")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("  SUMMARY  (mean ± std over seeds)")
    print("=" * 70)
    summary = {}
    for variant, accs in all_results.items():
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        summary[variant] = {"mean": mean, "std": std, "accs": accs}
        print(f"  {variant:12s}  {mean*100:.2f} ± {std*100:.2f}   (seeds: "
              + ", ".join(f"{a*100:.2f}" for a in accs) + ")")

    # 关键对比
    if "baseline" in summary and "row_aux" in summary:
        delta = summary["row_aux"]["mean"] - summary["baseline"]["mean"]
        print(f"\n  >> row_aux 相对 baseline: {delta*100:+.2f} 个百分点")
    if "row_aux" in summary and "col_aux" in summary:
        delta2 = summary["row_aux"]["mean"] - summary["col_aux"]["mean"]
        print(f"  >> row_aux 相对 col_aux : {delta2*100:+.2f} 个百分点 "
              f"(正值=垂直专属性成立)")

    with open(os.path.join(cfg.outdir, "summary.json"), "w") as f:
        json.dump({"config": asdict(cfg), "summary": summary}, f, indent=2)
    print(f"\n  结果已保存到 {cfg.outdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
