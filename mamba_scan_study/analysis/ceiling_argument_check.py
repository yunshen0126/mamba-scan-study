#!/usr/bin/env python3
"""
ceiling_argument_check.py -- 核实"训练侧天花板压制 ②"这一说法的可证伪性。

只读脚本。不写入任何 run 目录，不修改任何冻结产物。

要回答的问题:
  M1 失败的四个数据集都处于天花板标记格。若训练侧天花板确实压缩了泛化侧的
  可测空间, 那么在同一批 run、同一个格里, 装置应当同样测不到别的效应。
  本脚本检验该推论: 报告每个数据集在 R_high 下

    (a) 结构组尾窗 train accuracy 中位数   -> 天花板判据量
    (b) 十三个路径条件的尾窗 val accuracy 跨度 -> 泛化端是否也触顶
    (c) 结构对比 (1) = mean(GEO_S*) - mean(RND_S*) -> 装置在该格的测量能力
    (d) 对比 (2) = P_G - P_R                       -> 被判为零的那个量

  若 (b) 显示泛化端未触顶, 且 (c) 的区间排除零而 (d) 跨零,
  则"天花板压制效应"必须额外解释它为何放过 (1) 而只压制 (2)。

统计口径 (P0B_PREREG_ANALYSIS_PLAN 2/3):
  尾窗 epoch 80-100 含端点, 按 run 取算术平均
  区间 mean +- 3.182 * s / sqrt(4), t(3,0.975), ddof=1, 单位 pp
  对比按 seed 配对后再聚合, 聚合层不做中间舍入, 仅显示层舍入

用法:
  python3 ceiling_argument_check.py
  python3 ceiling_argument_check.py --runs-root /root/autodl-tmp/outputs_main
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt
from pathlib import Path

DATASETS = ("cifar10", "organamnist", "organcmnist", "organsmnist", "eurosat")
GEO_SINGLE = ("GEO_SG1", "GEO_SG2", "GEO_SG3", "GEO_SG4")
RND_SINGLE = ("RND_S1", "RND_S2", "RND_S3")
RND_DIVERSE = ("RND_D1", "RND_D2", "RND_D3")
GEO_DIVERSE = ("GEO_DIV",)
# 天花板判据的"结构组"含 GEO_DIV，与 analyze_main624.py:40 的
# STRUCTURE_EXP_IDS = GEO_SINGLE + ("GEO_DIV",) 一致。
# 对比 (1) 用的仍是 GEO_SINGLE，两者不是同一个集合。
STRUCTURE_EXP_IDS = GEO_SINGLE + GEO_DIVERSE
ALL_EXPS = GEO_SINGLE + GEO_DIVERSE + RND_SINGLE + RND_DIVERSE + ("LOC_S", "LOC_D")
SEEDS = (0, 1, 2, 3)
RELIANCE = "R_high"
TAIL_LO, TAIL_HI = 79, 100          # epoch 80-100 含端点 -> 0-based
T_CRIT = 3.182
CEILING_THRESHOLD = 95.0


def q2(x: float, signed: bool = True) -> str:
    d = Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{d:+.2f}" if signed else f"{d}"


def tail(meta: dict, key: str) -> float:
    hist = meta["validation_history"]
    if len(hist) != 100:
        raise ValueError(f"epoch history length {len(hist)}, expected 100")
    window = hist[TAIL_LO:TAIL_HI]
    if len(window) != 21:
        raise ValueError("tail window is not 21 epochs")
    return st.mean(row[key] for row in window) * 100.0


def ci(values) -> tuple[float, float, float]:
    m = st.mean(values)
    half = T_CRIT * st.stdev(values) / sqrt(len(values))
    return m, m - half, m + half


def load(root: Path, dataset: str, exp: str, seed: int) -> dict:
    p = (root / f"p0b_{dataset}_main_uniform_mamba_{exp}_{RELIANCE}_seed{seed}"
         / "metadata.json")
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open() as fh:
        return json.load(fh)


def group_mean(cells: dict, exps, seed: int, key: str) -> float:
    """组内先按 seed 取该 seed 下各条件的均值。"""
    return st.mean(cells[(e, seed)][key] for e in exps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="/root/autodl-tmp/outputs_main", type=Path)
    args = ap.parse_args()

    print("=" * 78)
    print("天花板论证核验  (R_high, mamba, main_uniform)")
    print("=" * 78)
    print()

    for ds in DATASETS:
        cells = {}
        try:
            for exp in ALL_EXPS:
                for s in SEEDS:
                    m = load(args.runs_root, ds, exp, s)
                    cells[(exp, s)] = {
                        "train": tail(m, "train_accuracy"),
                        "val": tail(m, "validation_accuracy"),
                    }
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"{ds}: 读取失败 -> {exc}", file=sys.stderr)
            return 2

        # (a) 结构组尾窗 train 中位数 (GEO_SG1-4 + GEO_DIV，共 20 值)
        struct_train = [cells[(e, s)]["train"] for e in STRUCTURE_EXP_IDS for s in SEEDS]
        med_train = st.median(struct_train)
        flagged = med_train > CEILING_THRESHOLD

        # (b) 十三条件的 val 跨度 (按条件取四 seed 均值后比较)
        per_exp_val = {e: st.mean(cells[(e, s)]["val"] for s in SEEDS) for e in ALL_EXPS}
        lo_e = min(per_exp_val, key=per_exp_val.get)
        hi_e = max(per_exp_val, key=per_exp_val.get)
        spread = per_exp_val[hi_e] - per_exp_val[lo_e]

        # (c) 结构对比 (1): mean(GEO_S*) - mean(RND_S*), 按 seed 配对
        c1 = [group_mean(cells, GEO_SINGLE, s, "val") - group_mean(cells, RND_SINGLE, s, "val")
              for s in SEEDS]
        c1m, c1l, c1h = ci(c1)

        # (d) 对比 (2): P_G - P_R, 按 seed 配对
        pg = [group_mean(cells, GEO_DIVERSE, s, "val") - group_mean(cells, GEO_SINGLE, s, "val")
              for s in SEEDS]
        pr = [group_mean(cells, RND_DIVERSE, s, "val") - group_mean(cells, RND_SINGLE, s, "val")
              for s in SEEDS]
        c2 = [a - b for a, b in zip(pg, pr)]
        c2m, c2l, c2h = ci(c2)

        excl = lambda lo, hi: lo > 0 or hi < 0
        print(f"--- {ds} ---")
        print(f"  (a) 结构组 train 中位数     {q2(med_train, False)}%"
              f"   {'[天花板标记]' if flagged else '[未标记]'}")
        print(f"  (b) 十三条件 val 跨度       {q2(spread, False)} pp"
              f"   ({lo_e} {q2(per_exp_val[lo_e], False)}"
              f" .. {hi_e} {q2(per_exp_val[hi_e], False)})")
        print(f"  (c) 结构对比 (1)            {q2(c1m)} [{q2(c1l)}, {q2(c1h)}]"
              f"   {'区间排除零' if excl(c1l, c1h) else '跨零'}")
        print(f"  (d) 对比 (2) = P_G - P_R    {q2(c2m)} [{q2(c2l)}, {q2(c2h)}]"
              f"   {'区间排除零' if excl(c2l, c2h) else '跨零'}")
        if flagged and excl(c1l, c1h) and not excl(c2l, c2h):
            print(f"      -> 该格被标记为饱和, 但装置在同格测出 (1) = {q2(c1m)} pp;")
            print(f"         天花板压制 (2) 的说法需额外解释它为何放过 (1)。")
        print()

    print("说明: (c) 与 (d) 的口径与 analyze_main624.py 一致, 可交叉核对 Table 6。")
    print("      (b) 为描述性, 不设阈值, 仅用于判断泛化端是否亦触顶。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
