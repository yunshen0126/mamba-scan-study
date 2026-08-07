#!/usr/bin/env python3
"""
fp32_recheck.py -- 实验 B：fp16 评估噪声的抽查。

**只读推理。不训练, 不写任何 run 目录, 不进任何判据。探索性、事后。**

背景（审稿意见弱点 ⑨）:
  §4.4 承认验证准确率在 autocast 下计算, 即 fp16。文中的辩护是"每个条件一致,
  故不能区分它们"——这对**偏倚**成立, 对**方差**不成立。fp16 评估噪声会加宽
  区间, 而更宽的区间使 M2（P_R 含零）更容易满足。这是一个方向性地偏向本文
  头条 null 的技术选择, 因此需要一个量级估计。

本脚本对同一批 checkpoint 各做两次验证集前向 —— 一次 autocast(fp16), 一次
纯 fp32 —— 报告二者之差。**它不能修正任何已报告的数字**（尾窗是训练时逐 epoch
记录的, 无法追溯重算）, 只给出"该噪声有多大"的量级。

用法（云端）:
  python3 fp32_recheck.py \
      --runs-root /root/autodl-tmp/outputs_main \
      --data-root /root/autodl-tmp/datasets \
      --out /root/fig7_work/fp32_recheck.tsv
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

import torch

# 抽查覆盖两个家族、两个负载档、四个 seed。共 16 个 run。
CELLS = (("GEO_SG1", "R_high"), ("GEO_DIV", "R_high"),
         ("RND_S1", "R_high"), ("RND_D1", "R_high"))
SEEDS = (0, 1, 2, 3)
GRID = {"R_low": 8, "R_high": 32}


@torch.no_grad()
def accuracy(model, loader, device: str, amp: bool) -> float:
    model.eval().to(device)
    correct = total = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            logits, _ = model(images)
        correct += (logits.float().argmax(1) == target).sum().item()
        total += target.numel()
    return 100.0 * correct / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_main"))
    ap.add_argument("--repo", type=Path, default=Path("/root/mamba-scan-study"))
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets")
    ap.add_argument("--augmentation", default="main_uniform")
    ap.add_argument("--out", type=Path, default=Path("/root/fig7_work/fp32_recheck.tsv"))
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    from mamba_scan_study.experiments.run_p0b_feasibility import (
        FORMAL_CONFIG, construct_requested_model)
    from mamba_scan_study.experiments.p0b_path_bank import resolve_p0b_paths
    from mamba_scan_study.experiments.p0b_data import build_p0b_loaders

    if not torch.cuda.is_available():
        print("需要 CUDA：autocast 的行为是本测量的对象。", file=sys.stderr)
        return 2
    device = "cuda"

    print("实验 B：fp32 重评估抽查  **探索性, 事后, 不进任何判据**")
    print(f"  {a.dataset} / {a.augmentation}")
    print("  报告同一 checkpoint 在 autocast(fp16) 与纯 fp32 下的验证准确率之差。")
    print("  本脚本不修正任何已报告数字；尾窗是训练时逐 epoch 记录的，无法追溯重算。")
    print()

    rows = []
    for exp, rel in CELLS:
        grid = GRID[rel]
        cfg = replace(FORMAL_CONFIG, dataset=a.dataset, block_type="mamba", num_workers=2)
        for seed in SEEDS:
            d = a.runs_root / (f"p0b_{a.dataset}_{a.augmentation}_mamba_{exp}_"
                               f"{rel}_seed{seed}")
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
            a16 = accuracy(model, loaders.validation, device, amp=True)
            a32 = accuracy(model, loaders.validation, device, amp=False)
            rows.append((exp, rel, seed, a16, a32, a32 - a16))
            print(f"  {exp:8s} {rel:7s} seed{seed}   fp16 {a16:6.2f}   fp32 {a32:6.2f}   "
                  f"diff {a32 - a16:+.3f}")
            del model
            torch.cuda.empty_cache()

    if not rows:
        print("没有可用的 checkpoint。", file=sys.stderr)
        return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        fh.write("condition\treliance\tseed\tacc_fp16\tacc_fp32\tdiff\n")
        for r in rows:
            fh.write("\t".join(repr(v) if isinstance(v, float) else str(v)
                               for v in r) + "\n")
    print(f"\n已写出 {a.out}")

    diffs = [r[5] for r in rows]
    print()
    print("=== 汇总 ===")
    print(f"  n = {len(diffs)}")
    print(f"  fp32 - fp16 均值   {st.mean(diffs):+.4f} pp")
    print(f"  绝对值最大         {max(abs(x) for x in diffs):.4f} pp")
    print(f"  标准差             {st.stdev(diffs) if len(diffs) > 1 else 0.0:.4f} pp")
    print()
    print("读法（探索性, 事后, 只能进 Discussion 或 Limitations）:")
    print("  差值远小于 P_R 的区间半宽（约 0.3 pp）-> fp16 评估噪声不足以解释")
    print("    M2 的满足, §4.4 的技术选择对头条 null 的影响可忽略。")
    print("  差值与该半宽同量级或更大 -> 必须如实报告, 且 M2 的证据地位相应削弱。")
    print("  两种结果都要报告。本测量不修正任何已报告数字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
