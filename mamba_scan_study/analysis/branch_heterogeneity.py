#!/usr/bin/env python3
"""
branch_heterogeneity.py -- 实验 A：分支异质性的推理级测量。

**只读推理。不训练, 不写任何 run 目录, 不进任何判据。探索性、事后。**

要回答的问题（审稿意见"核心构念从未被测量"）:
  论文的中心构念是 branch heterogeneity, 但全文只测了它的**结果差** (P_R),
  从未验证 arbitrary-diverse 条件是否真的产生了彼此不同的分支。若四个 channel
  group 训练后收敛到近似相同的函数, 那 P_R 的零与异质性无关。

================================================================================
分解的正确性（重要, 请连同结果一起报告）
================================================================================
`ChannelSplitBackbone.forward_features` 的最后一步是

    self.norm(torch.cat(outputs, dim=-1))        # LayerNorm 跨全部 d_model 通道

所以四组**通过 LayerNorm 的均值与方差互相耦合**, logits **不是**四个独立
per-group readout 之和。本脚本因此不做那个假设。

它做两件各自精确的事:

  (A) 归一化**之前**的每组特征 outputs[g]。这是纯粹的每组表示, 完全不受
      LayerNorm 耦合影响。组间用**线性 CKA** 比较: 逐元素相关是错的, 因为两组
      的通道索引之间没有任何对应关系 (第 3 通道对第 3 通道是任意配对), 那样算
      出来必然接近零, 连"四组走同一条路径"的条件也接近零。CKA 对通道置换与
      各向同性缩放不变, 正是此处需要的性质。

  (B) 已实现 logits 的精确线性分解。head 是 nn.Linear, 故
          logits = W z + b = sum_g (W_g z_g) + b
      按通道分块是恒等式。**但 z_g 本身受 LN 耦合**, 所以 W_g z_g 是"该组对
      已实现 logits 的贡献", 不是"该组单独会输出什么"的反事实。这一区别必须
      在论文中写明, 不得表述为 per-group 独立预测。

用法（云端）:
  python3 branch_heterogeneity.py \
      --runs-root /root/autodl-tmp/outputs_main \
      --dataset cifar10 --reliance R_high \
      --data-root /root/autodl-tmp/datasets \
      --out /root/fig7_work/branch_het.tsv
"""

from __future__ import annotations

import argparse
import itertools
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

import torch

CONDITIONS = ("GEO_SG1", "GEO_DIV", "RND_S1", "RND_D1", "RND_D2", "RND_D3", "LOC_D")
SEEDS = (0, 1, 2, 3)
GRID = {"R_low": 8, "R_high": 32}


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """线性 CKA。对通道置换与各向同性缩放不变, 故可跨 group 比较表示。

    X, Y: (n_samples, n_features)。列中心化后
        CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X.double() - X.double().mean(0, keepdim=True)
    Y = Y.double() - Y.double().mean(0, keepdim=True)
    xty = (X.T @ Y).pow(2).sum()
    xtx = (X.T @ X).pow(2).sum().sqrt()
    yty = (Y.T @ Y).pow(2).sum().sqrt()
    denom = xtx * yty
    return float(xty / denom) if denom > 0 else float("nan")


@torch.no_grad()
def measure(model, loader, device: str) -> dict:
    from mamba_scan_study.models.scan_utils import flatten_scan, restore_scan

    model.eval().to(device)
    n_groups = 4
    W = model.head.weight
    b = model.head.bias
    gw = W.shape[1] // n_groups

    contrib = [[] for _ in range(n_groups)]
    prenorm = [[] for _ in range(n_groups)]
    total, labels = [], []

    for images, target in loader:
        images = images.to(device, non_blocking=True)

        x = model.patch_embed(images).permute(0, 2, 3, 1)
        groups_in = x.chunk(n_groups, dim=-1)
        outs = []
        for g, (gx, blocks) in enumerate(zip(groups_in, model.group_blocks)):
            tokens = flatten_scan(gx, model.branch_dirs[g])
            pos = model._position(g, tokens.shape[0], tokens.device)
            perm = model.channel_permutations[g]
            inv = model.channel_inverse_permutations[g]
            should = model.explicit_channel_orders or not torch.equal(
                perm, torch.arange(model.L, device=perm.device))
            if should:
                tokens = tokens.index_select(1, perm)
                if pos is not None:
                    pos = pos.index_select(1, perm)
            if pos is not None:
                tokens = tokens + pos
            for blk in blocks:
                tokens = blk(tokens)
            if should:
                tokens = tokens.index_select(1, inv)
            out_g = restore_scan(tokens, model.H, model.W, model.branch_dirs[g])
            outs.append(out_g)
            prenorm[g].append(out_g.mean(dim=(1, 2)).float().cpu())

        z = model.norm(torch.cat(outs, dim=-1)).mean(dim=(1, 2))
        for g in range(n_groups):
            zg = z[:, g * gw:(g + 1) * gw]
            Wg = W[:, g * gw:(g + 1) * gw]
            contrib[g].append((zg @ Wg.T).float().cpu())
        total.append((z @ W.T + b).float().cpu())
        labels.append(target)

    y = torch.cat(labels)
    C = [torch.cat(c) for c in contrib]
    P = [torch.cat(p) for p in prenorm]
    L = torch.cat(total)

    combined = (L.argmax(1) == y).float().mean().item() * 100.0
    acc = [(c.argmax(1) == y).float().mean().item() * 100.0 for c in C]
    preds = [c.argmax(1) for c in C]
    pairs = list(itertools.combinations(range(n_groups), 2))

    disagree = st.mean((preds[i] != preds[j]).float().mean().item() * 100.0
                       for i, j in pairs)
    logit_corr = st.mean(
        torch.corrcoef(torch.stack([C[i].flatten(), C[j].flatten()]))[0, 1].item()
        for i, j in pairs)
    feat_cka = st.mean(linear_cka(P[i], P[j]) for i, j in pairs)
    logit_cka = st.mean(linear_cka(C[i], C[j]) for i, j in pairs)

    return {"combined_acc": combined,
            "group_acc_mean": st.mean(acc),
            "group_acc_spread": max(acc) - min(acc),
            "combination_gain": combined - st.mean(acc),
            "pair_disagree_pct": disagree,
            "pair_logit_corr": logit_corr,
            "pair_prenorm_feat_cka": feat_cka,
            "pair_logit_cka": logit_cka}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_main"))
    ap.add_argument("--repo", type=Path, default=Path("/root/mamba-scan-study"))
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--reliance", default="R_high", choices=tuple(GRID))
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets")
    ap.add_argument("--augmentation", default="main_uniform")
    ap.add_argument("--out", type=Path, default=Path("/root/fig7_work/branch_het.tsv"))
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    from mamba_scan_study.experiments.run_p0b_feasibility import (
        FORMAL_CONFIG, construct_requested_model)
    from mamba_scan_study.experiments.p0b_path_bank import resolve_p0b_paths
    from mamba_scan_study.experiments.p0b_data import build_p0b_loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = GRID[a.reliance]
    cfg = replace(FORMAL_CONFIG, dataset=a.dataset, block_type="mamba", num_workers=2)

    print("实验 A：分支异质性的推理级测量  **探索性, 事后, 不进任何判据**")
    print(f"  {a.dataset} / {a.reliance} (grid={grid}) / {a.augmentation} / {device}")
    print("  注: logits 非四组之和 (LayerNorm 跨通道耦合); 见脚本首部说明。")
    print()

    cols = ("condition", "seed", "combined_acc", "group_acc_mean", "group_acc_spread",
            "combination_gain", "pair_disagree_pct", "pair_logit_corr",
            "pair_prenorm_feat_cka", "pair_logit_cka")
    rows = []
    for exp in CONDITIONS:
        for seed in SEEDS:
            d = a.runs_root / (f"p0b_{a.dataset}_{a.augmentation}_mamba_{exp}_"
                               f"{a.reliance}_seed{seed}")
            ck = d / "final_checkpoint.pt"
            if not ck.is_file():
                print(f"  [缺] {d.name}")
                continue
            res = resolve_p0b_paths(exp, grid, seed)
            model = construct_requested_model(res, torch.device(device), a.dataset, cfg)
            model.load_state_dict(torch.load(ck, map_location=device)["model_state"],
                                  strict=True)
            loaders = build_p0b_loaders(a.data_root, 128, seed,
                                        num_workers=cfg.num_workers, download=False,
                                        dataset=a.dataset, augmentation=a.augmentation)
            m = measure(model, loaders.validation, device)
            rows.append((exp, seed, *[m[c] for c in cols[2:]]))
            print(f"  {exp:8s} seed{seed}  acc {m['combined_acc']:6.2f}  "
                  f"group {m['group_acc_mean']:6.2f}  gain {m['combination_gain']:+5.2f}  "
                  f"disagree {m['pair_disagree_pct']:5.2f}%  "
                  f"logit_r {m['pair_logit_corr']:+.3f}  "
                  f"featCKA {m['pair_prenorm_feat_cka']:.3f}  "
                  f"logitCKA {m['pair_logit_cka']:.3f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        print("没有可用的 checkpoint。", file=sys.stderr)
        return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(repr(v) if isinstance(v, float) else str(v)
                               for v in r) + "\n")
    print(f"\n已写出 {a.out}")

    print("\n=== 条件均值 ===")
    for exp in CONDITIONS:
        sub = [r for r in rows if r[0] == exp]
        if not sub:
            continue
        print(f"  {exp:8s} disagree {st.mean(x[6] for x in sub):5.2f}%   "
              f"logit_r {st.mean(x[7] for x in sub):+.3f}   "
              f"featCKA {st.mean(x[8] for x in sub):.3f}   "
              f"logitCKA {st.mean(x[9] for x in sub):.3f}   "
              f"gain {st.mean(x[5] for x in sub):+5.2f}")

    print()
    print("读法（探索性, 事后, 只能进 Discussion）:")
    print("  以 RND_S1 (四组同一条路径) 为基准读 RND_D*: 若两者的 CKA 与分歧率")
    print("    相近, 则让路径互异这一操纵并未改变分支异质性, P_R 的零随之得解释。")
    print("  RND_D* 分歧率高、CKA 低 -> 分支确实异质。此时 P_R 仍为零, 说明零不是")
    print("    因为异质性没产生, 而是因为它不兑现 —— 这加强论文的 null。")
    print("  RND_D* 分歧率低、相关高 -> 分支趋同。P_R 的零与异质性无关,")
    print("    论文对 P_R 的解释须相应削弱, 且必须如实报告。")
    print("  两种结果都要报告。本测量不进任何判据, 不修改任何已报告结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
