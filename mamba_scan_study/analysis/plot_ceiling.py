#!/usr/bin/env python3
"""
plot_ceiling.py -- Figure 5: 天花板论证。

只读脚本。从 outputs_main 的 metadata 直接重算, 不写入任何 run 目录。

要呈现的事实 (论文 sec lim_saturation / sec disc_task):
  M1 失败的四个数据集在高负载档全部触发天花板标记 (结构组尾窗 train acc
  中位数 > 95%)。若"训练侧天花板压制了泛化侧效应"成立, 装置在这些格里
  应当什么都测不到。但实际是:
    (a) 泛化端未触顶  -- 十三条件的 val acc 跨度 1.94 到 5.90 pp
    (b) 装置能测出 (1) -- 结构对比区间全部排除零
    (c) 却测不到 (2)   -- P_G - P_R 区间全部跨零
  故该说法须额外解释它为何放过 (1) 而只压制 (2)。

左图: 每个数据集的 train acc 中位数 (结构组, 含 GEO_DIV, 与 analyze_main624
      的 STRUCTURE_EXP_IDS 一致) 与 val acc 跨度, 双轴条形
右图: 同一批格里 (1) 与 (2) 的点估计与区间

用法:
  python3 plot_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = (("cifar10", "CIFAR-10"), ("organamnist", "OrganAMNIST"),
            ("organcmnist", "OrganCMNIST"), ("organsmnist", "OrganSMNIST"),
            ("eurosat", "EuroSAT"))
GEO_SINGLE = ("GEO_SG1", "GEO_SG2", "GEO_SG3", "GEO_SG4")
GEO_DIVERSE = ("GEO_DIV",)
RND_SINGLE = ("RND_S1", "RND_S2", "RND_S3")
RND_DIVERSE = ("RND_D1", "RND_D2", "RND_D3")
STRUCTURE_EXP_IDS = GEO_SINGLE + GEO_DIVERSE      # analyze_main624.py:40
ALL_EXPS = GEO_SINGLE + GEO_DIVERSE + RND_SINGLE + RND_DIVERSE + ("LOC_S", "LOC_D")
SEEDS = (0, 1, 2, 3)
RELIANCE = "R_high"
TAIL_LO, TAIL_HI = 79, 100
T_CRIT = 3.182
CEILING = 95.0

C_STRUCT = "#3B5FA8"
C_GEOSPEC = "#C4553B"
C_TRAIN = "#B0A8C8"
C_SPREAD = "#4A8A6F"
C_GRID = "#D9D9E0"
C_TEXT = "#2A2A32"


def tails(p: Path) -> tuple[float, float]:
    with p.open() as fh:
        m = json.load(fh)
    h = m["validation_history"]
    if len(h) != 100:
        raise ValueError(f"{p}: history length {len(h)}")
    w = h[TAIL_LO:TAIL_HI]
    return (st.mean(r["train_accuracy"] for r in w) * 100.0,
            st.mean(r["validation_accuracy"] for r in w) * 100.0)


def load(root: Path, dataset: str) -> dict:
    out = {}
    for exp in ALL_EXPS:
        for s in SEEDS:
            p = root / f"p0b_{dataset}_main_uniform_mamba_{exp}_{RELIANCE}_seed{s}" / "metadata.json"
            if not p.is_file():
                raise FileNotFoundError(p)
            out[(exp, s)] = tails(p)
    return out


def ci(v):
    m = st.mean(v)
    h = T_CRIT * st.stdev(v) / sqrt(len(v))
    return m, m - h, m + h


def gm(cells, exps, s, idx):
    return st.mean(cells[(e, s)][idx] for e in exps)


def summarise(cells) -> dict:
    med_train = st.median([cells[(e, s)][0] for e in STRUCTURE_EXP_IDS for s in SEEDS])
    per_exp_val = {e: st.mean(cells[(e, s)][1] for s in SEEDS) for e in ALL_EXPS}
    spread = max(per_exp_val.values()) - min(per_exp_val.values())
    c1 = ci([gm(cells, GEO_SINGLE, s, 1) - gm(cells, RND_SINGLE, s, 1) for s in SEEDS])
    pg = [gm(cells, GEO_DIVERSE, s, 1) - gm(cells, GEO_SINGLE, s, 1) for s in SEEDS]
    pr = [gm(cells, RND_DIVERSE, s, 1) - gm(cells, RND_SINGLE, s, 1) for s in SEEDS]
    c2 = ci([a - b for a, b in zip(pg, pr)])
    return {"med_train": med_train, "flagged": med_train > CEILING,
            "spread": spread, "c1": c1, "c2": c2}


def excl(t) -> bool:
    return t[1] > 0 or t[2] < 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_main"))
    ap.add_argument("--output", type=Path, default=Path("figure5_ceiling.pdf"))
    args = ap.parse_args()

    d = {}
    for key, _ in DATASETS:
        try:
            d[key] = summarise(load(args.runs_root, key))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"FAIL {key}: {e}", file=sys.stderr)
            return 2

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.90, bottom=0.20)

    y = np.arange(len(DATASETS))[::-1].astype(float)
    labs = [lab for _, lab in DATASETS]

    ax.axvline(0, color="#8A8A96", lw=1.0, zorder=2)

    for i, (key, _) in enumerate(DATASETS):
        x = d[key]
        # (a) 训练侧: 超出 95% 标记线的量
        ax.barh(y[i] + 0.26, x["med_train"] - CEILING, height=0.20,
                color=C_TRAIN, zorder=3)
        ax.text(x["med_train"] - CEILING + 0.18, y[i] + 0.26,
                f"{x['med_train']:.2f}%", va="center", fontsize=7.2, color=C_TEXT)
        # (b) 泛化侧: 十三条件的跨度
        ax.barh(y[i] + 0.06, x["spread"], height=0.20, color=C_SPREAD, zorder=3)
        ax.text(x["spread"] + 0.18, y[i] + 0.06, f"{x['spread']:.2f}",
                va="center", fontsize=7.2, color=C_TEXT)
        # (c)(d) 同格内实际解析出的两个对比
        for k, colr, off in (("c1", C_STRUCT, -0.16), ("c2", C_GEOSPEC, -0.34)):
            m, lo, hi = x[k]
            solid = excl((m, lo, hi))
            ax.errorbar(m, y[i] + off, xerr=[[m - lo], [hi - m]], fmt="o", ms=4.6,
                        color=colr, ecolor=colr, elinewidth=1.2, capsize=2.4,
                        capthick=1.2, zorder=4,
                        markerfacecolor=colr if solid else "white",
                        markeredgewidth=1.2)

    # 数据集之间的浅分隔
    for i in range(len(DATASETS) - 1):
        ax.axhline(y[i] - 0.5, color="#ECECF0", lw=0.8, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9.5, color=C_TEXT)
    ax.set_xlabel("percentage points  (all four quantities on one scale)",
                  fontsize=9.5, color=C_TEXT)
    ax.set_title("Every high-load cell is ceiling-flagged, yet the validation "
                 "endpoint has headroom\nand the apparatus resolves one contrast "
                 "but not the other",
                 fontsize=10, color=C_TEXT, pad=8)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_edgecolor("#B8B8C4")
    ax.tick_params(labelsize=9, colors=C_TEXT)
    ax.set_ylim(y[-1] - 0.62, y[0] + 0.56)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=C_TRAIN, label="train acc median, amount above the 95% flag"),
        Patch(facecolor=C_SPREAD, label="val acc spread across the 13 conditions"),
        Line2D([], [], color=C_STRUCT, marker="o", ms=4.6, lw=1.2,
               label="structure contrast"),
        Line2D([], [], color=C_GEOSPEC, marker="o", ms=4.6, lw=1.2,
               markerfacecolor="white", markeredgewidth=1.2,
               label=r"$P_G - P_R$  (open: interval covers zero)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.5, -0.13), handletextpad=0.55,
               columnspacing=1.6)

    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"已写出 {args.output} 与 {args.output.with_suffix('.png')}")

    print("\n=== 数值 (与 Table 6 / Table 8 交叉核对) ===")
    print(f"{'dataset':13s}{'train med':>11s}{'flag':>6s}{'val spread':>12s}"
          f"{'structure':>24s}{'P_G-P_R':>24s}")
    for key, lab in DATASETS:
        x = d[key]
        print(f"{lab:13s}{x['med_train']:>11.2f}{'yes' if x['flagged'] else 'no':>6s}"
              f"{x['spread']:>12.2f}"
              f"{x['c1'][0]:>+9.2f} [{x['c1'][1]:+6.2f},{x['c1'][2]:+6.2f}]"
              f"{x['c2'][0]:>+9.2f} [{x['c2'][1]:+6.2f},{x['c2'][2]:+6.2f}]")

    print("\n=== 论证核心 ===")
    for key, lab in DATASETS:
        x = d[key]
        if x["flagged"] and excl(x["c1"]) and not excl(x["c2"]):
            print(f"  {lab}: 饱和标记, 但同格测出 structure = {x['c1'][0]:+.2f} pp, "
                  f"而 P_G-P_R 跨零")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
