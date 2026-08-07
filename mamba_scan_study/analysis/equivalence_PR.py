#!/usr/bin/env python3
"""
equivalence_PR.py -- P_R 的等效性分析 (TOST), 十个 dataset x load 格。

**只读, 纯探索性。** 不改任何预注册判据, 不改 M2 的判定, 不重跑任何 run。

背景 (审稿意见 R1-1):
  M2 以"P_R 的 95% 区间跨零"为满足条件。该方向的问题在于: 区间越宽越容易
  跨零, 即装置分辨力越差越容易"满足 M2"。因此 M2 只能支持"四个 seed 未能把
  P_R 与零分开"这一窄读法, 不能给出效应的上界。

  TOST 把方向倒过来: 等效性检验中分辨力越差越**难**通过。

口径:
  与 P0B_PREREG_ANALYSIS_PLAN 一致 -- 尾窗 epoch 80-100 含端点、按 run 算术
  平均、按 seed 配对后再聚合、单位 pp。
  Delta_min = |mean| + t(3, 0.95) * s / sqrt(4)，t(3,0.95) = 2.353
  即: 在 alpha = 0.05 下两个单侧检验同时拒绝"差异 >= Delta"的最小 Delta,
  等价于 90% 区间的外沿。

**不得**用本脚本的输出重述、替换或加强 M2 的判定。等效界是事后选定的,
未经预注册; 论文中必须标注为探索性 (sec res_equivalence)。

自检: 同时打印 95% 区间, 与论文 Table 6 的 P_R 行逐位比对, 不符即中止。

用法:
  python3 equivalence_PR.py --runs-root /path/to/outputs_main
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

DATASETS = ("cifar10", "organamnist", "organcmnist", "organsmnist", "eurosat")
RELIANCES = ("R_low", "R_high")
RND_SINGLE = ("RND_S1", "RND_S2", "RND_S3")
RND_DIVERSE = ("RND_D1", "RND_D2", "RND_D3")
SEEDS = (0, 1, 2, 3)
T_975 = 3.182      # t(3), two-sided 95%
T_95 = 2.353       # t(3), one-sided 95% -> TOST at alpha = 0.05
BOUNDS = (0.5, 1.0)

# 论文 Table 6 的 P_R 行 (low, high), 逐字抄自论文, 非本脚本回填。
FROZEN = {
    "cifar10":     ((-0.06, -0.37, +0.25), (-0.04, -0.35, +0.27)),
    "organamnist": ((+0.05, -0.27, +0.37), (+0.17, -0.20, +0.55)),
    "organcmnist": ((-0.02, -0.18, +0.14), (-0.16, -0.49, +0.16)),
    "organsmnist": ((-0.09, -0.58, +0.40), (+0.08, -0.46, +0.63)),
    "eurosat":     ((-0.07, -0.37, +0.22), (-0.01, -0.37, +0.36)),
}


def q2(x: float) -> float:
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def tail_val(path: Path) -> float:
    hist = json.loads(path.read_text())["validation_history"]
    if len(hist) != 100:
        raise ValueError(f"{path}: history length {len(hist)}")
    window = hist[79:100]
    if len(window) != 21:
        raise ValueError(f"{path}: tail window is not 21 epochs")
    for row in window:
        v = row["validation_accuracy"]
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{path}: validation_accuracy={v} outside [0,1]")
    return st.mean(r["validation_accuracy"] for r in window) * 100.0


def p_r(root: Path, dataset: str, reliance: str) -> list[float]:
    def gm(exps, seed):
        return st.mean(
            tail_val(root / f"p0b_{dataset}_main_uniform_mamba_{e}_{reliance}_seed{seed}"
                     / "metadata.json") for e in exps)
    return [gm(RND_DIVERSE, s) - gm(RND_SINGLE, s) for s in SEEDS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("outputs_main"))
    args = ap.parse_args()

    print("P_R 等效性分析 (TOST, alpha = 0.05)   **探索性, 事后, 不改 M2**")
    print()
    head = (f"{'dataset':13s}{'load':8s}{'P_R':>8s}{'95% CI':>19s}{'D_min':>8s}"
            + "".join(f"{'±'+str(b):>8s}" for b in BOUNDS))
    print(head)
    print("-" * len(head))

    bad, dmins = [], []
    for ds in DATASETS:
        for i, rel in enumerate(RELIANCES):
            vals = p_r(args.runs_root, ds, rel)
            m = st.mean(vals)
            se = st.stdev(vals) / math.sqrt(len(vals))
            lo, hi = m - T_975 * se, m + T_975 * se
            dmin = abs(m) + T_95 * se
            dmins.append((dmin, ds, rel))

            exp = FROZEN[ds][i]
            got = (q2(m), q2(lo), q2(hi))
            if got != exp:
                bad.append(f"{ds}/{rel}: 实得 {got} != Table 6 {exp}")

            print(f"{ds:13s}{rel:8s}{q2(m):>+8.2f}  [{q2(lo):>+6.2f},{q2(hi):>+6.2f}]"
                  f"{q2(dmin):>8.2f}"
                  + "".join(f"{('yes' if dmin <= b else 'NO'):>8s}" for b in BOUNDS))

    print()
    if bad:
        for b in bad:
            print(f"  [失配] {b}", file=sys.stderr)
        print("\n交叉核对失败: 与论文 Table 6 的 P_R 不一致。", file=sys.stderr)
        return 1
    print("交叉核对通过: 十格的 P_R 点估计与 95% 区间与论文 Table 6 逐位一致。")

    worst, wds, wrel = max(dmins)
    print(f"最大 D_min = {q2(worst)} pp  ({wds}, {wrel})")
    for b in BOUNDS:
        n = sum(1 for d, _, _ in dmins if d <= b)
        print(f"±{b} pp 等效: {n}/10 格")
    print()
    print("表述边界: D_min 描述本数据能排除什么, 不是实践上可忽略的阈值;")
    print("等效界事后选定、未预注册; 排除的是本装置与本路径库下的**均值**效应,")
    print("不是任一具体路径对的效应。不得用以重述 M2。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
