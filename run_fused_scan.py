#!/usr/bin/env python3
"""
run_fused_scan.py

The fused-block arm. Every cell measured so far uses a channel-split module
whose four branches never exchange information inside the backbone, so the null
on arbitrary path diversity can be attributed to the apparatus. This script runs
the same four conditions in a block where the four scans are summed *inside* the
block, over the full channel width, through one shared operator.

    # 1. verify the indexing and the fusion, no GPU time spent
    python3 run_fused_scan.py --selftest \
        --bank /root/mamba-scan-study/P0B_R_PATH_BANK_FROZEN.json

    # 2. one run
    python3 run_fused_scan.py \
        --config-from /root/autodl-tmp/outputs_main/p0b_cifar10_main_uniform_mamba_GEO_DIV_R_high_seed0/metadata.json \
        --bank /root/mamba-scan-study/P0B_R_PATH_BANK_FROZEN.json \
        --dataset cifar10 --exp-id GEO_DIV --seed 0 \
        --out /root/autodl-tmp/outputs_fused

Conditions, all at four scan slots and identical parameter count:
    GEO_SG1  G1 in all four slots
    GEO_DIV  G1 G2 G3 G4
    RND_S1   one frozen arbitrary permutation in all four slots
    RND_D1   four distinct frozen arbitrary permutations
    LOC_S1   one frozen locality-matched permutation in all four slots
    LOC_D1   the four members of the locality-matched orbit

Output goes to <out>/p0b_<dataset>_fused_scan_<block>_<exp_id>_R_high_seed<n>/
as metadata.json with a validation_history field, matching the layout the
existing analysis scripts already read.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def canonical_orders(n):
    """G1 row-major, G2 its reversal, G3 column-major, G4 its reversal."""
    N = n * n
    t = torch.arange(N)
    g1 = t.clone()
    g2 = torch.flip(g1, dims=[0]).contiguous()
    g3 = (t % n) * n + (t // n)
    g4 = torch.flip(g3, dims=[0]).contiguous()
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4}


def load_bank(path, n, count=4):
    """Pull `count` permutations of length n*n out of the frozen bank, whatever
    the file's internal shape. Returns them in file order."""
    N = n * n
    found = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            if len(node) == N and all(isinstance(x, int) for x in node):
                a = torch.tensor(node, dtype=torch.long)
                if torch.equal(torch.sort(a).values, torch.arange(N)):
                    found.append(a)
                    return
            for v in node:
                walk(v)

    with open(path) as f:
        walk(json.load(f))
    if len(found) < count:
        raise SystemExit(f"bank {path}: found {len(found)} permutations of "
                         f"length {N}, need {count}")
    return found[:count]


def inverse(order):
    pi = torch.empty_like(order)
    pi[order] = torch.arange(order.numel(), dtype=order.dtype)
    return pi


EXP_IDS = ("GEO_SG1", "GEO_DIV", "RND_S1", "RND_D1", "LOC_S1", "LOC_D1")


def build_orders(exp_id, n, bank_path, lbank_path=None):
    G = canonical_orders(n)
    if exp_id == "GEO_SG1":
        picks = [G["G1"]] * 4
    elif exp_id == "GEO_DIV":
        picks = [G["G1"], G["G2"], G["G3"], G["G4"]]
    elif exp_id in ("RND_S1", "RND_D1"):
        R = load_bank(bank_path, n, 4)
        picks = [R[0]] * 4 if exp_id == "RND_S1" else R
    elif exp_id in ("LOC_S1", "LOC_D1"):
        if not lbank_path:
            raise SystemExit(f"{exp_id} needs --lbank")
        L = load_bank(lbank_path, n, 4)
        picks = [L[0]] * 4 if exp_id == "LOC_S1" else L
    else:
        raise SystemExit(f"unknown exp_id {exp_id}")
    orders = torch.stack(picks)
    pis = torch.stack([inverse(o) for o in picks])
    for k in range(4):
        assert torch.equal(pis[k][orders[k]], torch.arange(n * n)), \
            f"order/inverse mismatch in slot {k}"
    return orders, pis


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class GRUScan(nn.Module):
    """Fallback sequence operator, used by --selftest and by --block gru."""

    def __init__(self, d_model):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


def make_operator(block, d_model):
    if block == "gru":
        return GRUScan(d_model)
    from mamba_ssm import Mamba
    return Mamba(d_model=d_model)


class FusedScanBlock(nn.Module):
    """Four scans of the full-width input through ONE shared operator, summed
    inside the block. This is the only structural difference from the
    channel-split apparatus: nothing is split, and the combination happens here
    rather than at the classifier.

    Parameter count does not depend on how many of the four orders are distinct,
    so the four conditions are matched by construction.
    """

    def __init__(self, d_model, orders, pis, block="mamba"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.op = make_operator(block, d_model)
        self.register_buffer("orders", orders, persistent=True)
        self.register_buffer("pis", pis, persistent=True)

    def forward(self, x):                      # x: (B, L, D)
        h = self.norm(x)
        acc = 0
        for k in range(self.orders.shape[0]):
            xs = h.index_select(1, self.orders[k])
            ys = self.op(xs)
            acc = acc + ys.index_select(1, self.pis[k])
        return x + acc / self.orders.shape[0]


class FusedScanNet(nn.Module):
    def __init__(self, orders, pis, n, d_model=256, depth=2,
                 num_classes=10, patch=1, in_ch=3, block="mamba"):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n * n, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [FusedScanBlock(d_model, orders, pis, block) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, img):
        x = self.embed(img).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x).mean(dim=1))


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def selftest(args):
    n = args.grid
    print(f"grid {n}x{n}, L = {n*n}\n")
    ok = True

    exps = [e for e in EXP_IDS if args.lbank or not e.startswith("LOC")]
    if not args.lbank:
        print("  (--lbank not given, skipping the locality-matched conditions)\n")
    for exp in exps:
        orders, pis = build_orders(exp, n, args.bank, args.lbank)
        distinct = len({tuple(o.tolist()) for o in orders})
        print(f"  {exp:<8} order/inverse consistent, {distinct} distinct path(s)")

    print("\n  round trip: gather then scatter must be the identity")
    orders, pis = build_orders("RND_D1", n, args.bank, args.lbank)
    x = torch.randn(2, n * n, 8)
    for k in range(4):
        back = x.index_select(1, orders[k]).index_select(1, pis[k])
        assert torch.allclose(back, x), f"round trip failed in slot {k}"
    print("     passed for all four slots")

    print("\n  fusion equivalence: four copies of one path, averaged, must equal")
    print("  a single scan along that path")
    torch.manual_seed(0)
    o1, p1 = build_orders("GEO_SG1", n, args.bank, args.lbank)
    blk = FusedScanBlock(8, o1, p1, block="gru").eval()
    x = torch.randn(2, n * n, 8)
    with torch.no_grad():
        fused = blk(x)
        h = blk.norm(x)
        single = x + blk.op(h.index_select(1, o1[0])).index_select(1, p1[0])
    same = torch.allclose(fused, single, atol=1e-5)
    print(f"     {'passed' if same else 'FAILED'} "
          f"(max abs diff {(fused - single).abs().max():.2e})")
    ok &= same

    print("\n  capacity match: parameter count must not depend on the condition")
    counts = {}
    for exp in exps:
        torch.manual_seed(0)
        o, p = build_orders(exp, n, args.bank, args.lbank)
        m = FusedScanNet(o, p, n, d_model=64, depth=2, block="gru")
        counts[exp] = sum(q.numel() for q in m.parameters())
    print("    ", counts)
    matched = len(set(counts.values())) == 1
    print(f"     {'passed' if matched else 'FAILED'}")
    ok &= matched

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED -- do not run"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def loaders(dataset, root, batch, workers, val_size=5000):
    from torchvision import datasets, transforms as T
    mean = (0.4914, 0.4822, 0.4465) if dataset == "cifar10" else (0.5071, 0.4865, 0.4409)
    std = (0.2470, 0.2435, 0.2616) if dataset == "cifar10" else (0.2673, 0.2564, 0.2762)
    # augmentation disabled, matching the main design's uniform policy
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    full = cls(root=root, train=True, download=True, transform=tf)
    g = torch.Generator().manual_seed(12345)     # split is fixed across seeds
    perm = torch.randperm(len(full), generator=g)
    val_idx, tr_idx = perm[:val_size].tolist(), perm[val_size:].tolist()
    tr = torch.utils.data.Subset(full, tr_idx)
    va = torch.utils.data.Subset(full, val_idx)
    mk = lambda d, sh: torch.utils.data.DataLoader(
        d, batch_size=batch, shuffle=sh, num_workers=workers, pin_memory=True,
        drop_last=False)
    return mk(tr, True), mk(va, False), (10 if dataset == "cifar10" else 100)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def evaluate(model, loader, dev):
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            hit += (model(x).argmax(1) == y).sum().item()
            tot += y.numel()
    return hit / max(tot, 1)


def train(args):
    if not torch.cuda.is_available() and not args.allow_cpu:
        print("CUDA is not available to this interpreter, so training would run "
              "on CPU and be killed for memory.\n"
              f"  python           : {sys.executable}\n"
              f"  torch            : {torch.__version__}\n"
              f"  built with CUDA  : {torch.version.cuda}\n"
              f"  devices visible  : {torch.cuda.device_count()}\n"
              "Check `nvidia-smi`, `conda env list`, and CUDA_VISIBLE_DEVICES. "
              "Pass --allow-cpu only for a tiny --limit-batches smoke test.")
        return 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    n = args.grid
    orders, pis = build_orders(args.exp_id, n, args.bank, args.lbank)
    tr, va, ncls = loaders(args.dataset, args.data_root, args.micro_batch,
                           args.workers)
    model = FusedScanNet(orders.to(dev), pis.to(dev), n, d_model=args.d_model,
                         depth=args.depth, num_classes=ncls, patch=args.patch,
                         block=args.block).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"{args.dataset} {args.exp_id} seed{args.seed} | {nparam} params | {dev}")
    if dev == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  gpu {torch.cuda.get_device_name(0)} | "
              f"{free/2**30:.1f} GiB free of {total/2**30:.1f}")
    print(f"  L={n*n} d_model={args.d_model} micro_batch={args.micro_batch} "
          f"epochs={args.epochs} lr={args.base_lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr,
                            weight_decay=args.weight_decay)
    steps = len(tr)
    warm = args.warmup_epochs * steps
    total = args.epochs * steps

    def lr_at(step):
        if step < warm:
            return args.base_lr * (step + 1) / max(warm, 1)
        prog = (step - warm) / max(total - warm, 1)
        return args.base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and dev == "cuda")
    hist, step = [], 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        hit = tot = 0
        for bi, (x, y) in enumerate(tr):
            if args.limit_batches and bi >= args.limit_batches:
                break
            lr = lr_at(step)
            for gp in opt.param_groups:
                gp["lr"] = lr
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and dev == "cuda"):
                out = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            hit += (out.argmax(1) == y).sum().item()
            tot += y.numel()
            step += 1
        vacc = evaluate(model, va, dev)
        hist.append({"epoch": ep, "learning_rate": lr,
                     "train_accuracy": hit / tot, "validation_accuracy": vacc})
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  train {hit/tot:.4f}  val {vacc:.4f}  "
                  f"lr {lr:.2e}  {time.time()-t0:.0f}s")

    name = (f"p0b_{args.dataset}_fused_scan_{args.block}_{args.exp_id}"
            f"_R_high_seed{args.seed}")
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    meta = {
        "protocol": "FUSED-01", "dataset": args.dataset, "backbone": args.block,
        "block_type": args.block, "arch": "fused_scan", "exp_id": args.exp_id,
        "variant": args.exp_id, "reliance": "R_high", "grid": n,
        "patch_size": args.patch, "sequence_length": n * n,
        "training_seed": args.seed, "seed": args.seed,
        "d_model": args.d_model, "parameter_count": nparam,
        "augmentation": "none",
        "channel_path_ids": {"GEO_SG1": ["G1"] * 4,
                             "GEO_DIV": ["G1", "G2", "G3", "G4"],
                             "RND_S1": ["R1"] * 4,
                             "RND_D1": ["R1", "R2", "R3", "R4"],
                             "LOC_S1": ["L1"] * 4,
                             "LOC_D1": ["L1", "L2", "L3", "L4"]}[args.exp_id],
        "training_config": {"epochs": args.epochs, "base_lr": args.base_lr,
                            "micro_batch": args.micro_batch, "amp": args.amp,
                            "grad_clip": args.grad_clip,
                            "weight_decay": args.weight_decay,
                            "warmup_epochs": args.warmup_epochs},
        "validation_history": hist,
    }
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    tail = [h["validation_accuracy"] for h in hist[-20:]]
    print(f"  wrote {d}\n  tail-20 val accuracy {sum(tail)/len(tail)*100:.3f}")
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config-from", default=None,
                    help="a main-batch metadata.json to copy hyperparameters from")
    ap.add_argument("--bank", required=True,
                    help="frozen arbitrary path bank (R)")
    ap.add_argument("--lbank", default=None,
                    help="frozen locality-matched path bank (L); required for LOC_*")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--exp-id", default="GEO_DIV", choices=list(EXP_IDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./outputs_fused")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--block", default="mamba", choices=["mamba", "gru"])
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--patch", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU training; only sensible with --limit-batches")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="cap training batches per epoch (0 = no cap)")
    args = ap.parse_args()

    if args.config_from:
        cfg = json.load(open(args.config_from)).get("training_config", {})
        for k, dest in (("epochs", "epochs"), ("base_lr", "base_lr"),
                        ("micro_batch", "micro_batch"), ("d_model", "d_model"),
                        ("grad_clip", "grad_clip"), ("amp", "amp")):
            if k in cfg and cfg[k] is not None:
                setattr(args, dest, cfg[k])
        print(f"hyperparameters copied from {args.config_from}: "
              f"epochs={args.epochs} lr={args.base_lr} batch={args.micro_batch} "
              f"d_model={args.d_model}")

    return selftest(args) if args.selftest else train(args)


if __name__ == "__main__":
    sys.exit(main())
