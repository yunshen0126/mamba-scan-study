#!/usr/bin/env python3
"""
plot_components.py -- Figure: 多路径增益的两个分量, 分开画。

只读脚本。从 outputs_main 的 metadata 直接重算, 不写入任何 run 目录。

为什么需要这张图:
  论文的头条发现是分支异质性成分 P_R 在全部十个 dataset x load 格中区间跨零,
  点估计落在 [-0.16, +0.17] pp 以内。但森林图画的是差值 (2) = P_G - P_R,
  读者需要在脑中做减法才能看出 P_R 本身有多接近零。本图把两个分量分开呈现。

  P_G = mean(GEO_DIV) - mean(GEO_S*)     几何族的多路径增益
  P_R = mean(RND_D*)  - mean(RND_S*)     任意族的多路径增益
  (2) = P_G - P_R                        几何专属成分

统计口径 (P0B_PREREG_ANALYSIS_PLAN 2/3):
  尾窗 epoch 80-100 含端点, 按 run 取算术平均
  区间 mean +- 3.182 * s / sqrt(4), t(3,0.975), ddof=1, 单位 pp
  对比按 seed 配对后再聚合, 聚合层不做中间舍入

用法:
  python3 plot_components.py
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
SEEDS = (0, 1, 2, 3)
RELIANCES = (("R_low", "Low load  ($L=64$)"), ("R_high", "High load  ($L=1024$)"))
TAIL_LO, TAIL_HI = 79, 100
T_CRIT = 3.182

C_PG = "#3B5FA8"      # 几何族, 与 Fig 2 同色
C_PR = "#C4553B"      # 任意族, 与 Fig 2 同色
C_GRID = "#D9D9E0"
C_TEXT = "#2A2A32"


def tail(p: Path) -> float:
    with p.open() as fh:
        m = json.load(fh)
    h = m["validation_history"]
    if len(h) != 100:
        raise ValueError(f"{p}: history length {len(h)}")
    w = h[TAIL_LO:TAIL_HI]
    if len(w) != 21:
        raise ValueError(f"{p}: tail window {len(w)}")
    return st.mean(r["validation_accuracy"] for r in w) * 100.0


def load(root: Path, dataset: str) -> dict:
    out = {}
    for exp in GEO_SINGLE + GEO_DIVERSE + RND_SINGLE + RND_DIVERSE:
        for rel, _ in RELIANCES:
            for s in SEEDS:
                p = root / f"p0b_{dataset}_main_uniform_mamba_{exp}_{rel}_seed{s}" / "metadata.json"
                if not p.is_file():
                    raise FileNotFoundError(p)
                out[(exp, rel, s)] = tail(p)
    return out


def gm(cells, exps, rel, s):
    return st.mean(cells[(e, rel, s)] for e in exps)


def ci(v):
    m = st.mean(v)
    h = T_CRIT * st.stdev(v) / sqrt(len(v))
    return m, m - h, m + h


def components(cells) -> dict:
    out = {}
    for rel, _ in RELIANCES:
        pg = [gm(cells, GEO_DIVERSE, rel, s) - gm(cells, GEO_SINGLE, rel, s) for s in SEEDS]
        pr = [gm(cells, RND_DIVERSE, rel, s) - gm(cells, RND_SINGLE, rel, s) for s in SEEDS]
        out[("P_G", rel)] = ci(pg)
        out[("P_R", rel)] = ci(pr)
        out[("c2", rel)] = ci([a - b for a, b in zip(pg, pr)])
    return out


def excl(t) -> bool:
    return t[1] > 0 or t[2] < 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_main"))
    ap.add_argument("--output", type=Path, default=Path("figure_components.pdf"))
    args = ap.parse_args()

    d = {}
    for key, _ in DATASETS:
        try:
            d[key] = components(load(args.runs_root, key))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"FAIL {key}: {e}", file=sys.stderr)
            return 2

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.865, bottom=0.175, wspace=0.09)

    y = np.arange(len(DATASETS))[::-1].astype(float)

    for ax, (rel, title) in zip(axes, RELIANCES):
        ax.axvline(0, color="#8A8A96", lw=1.0, zorder=2)
        for i, (key, _) in enumerate(DATASETS):
            for k, colr, off in (("P_G", C_PG, 0.17), ("P_R", C_PR, -0.17)):
                m, lo, hi = d[key][(k, rel)]
                solid = excl((m, lo, hi))
                ax.errorbar(m, y[i] + off, xerr=[[m - lo], [hi - m]], fmt="o", ms=5.2,
                            color=colr, ecolor=colr, elinewidth=1.3, capsize=2.6,
                            capthick=1.3, zorder=3,
                            markerfacecolor=colr if solid else "white",
                            markeredgewidth=1.3)
        for i in range(len(DATASETS) - 1):
            ax.axhline(y[i] - 0.5, color="#ECECF0", lw=0.8, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in DATASETS], fontsize=9.5, color=C_TEXT)
        ax.set_title(title, fontsize=10.5, color=C_TEXT, pad=7)
        ax.set_xlabel("percentage points", fontsize=9.5, color=C_TEXT)
        ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_edgecolor("#B8B8C4")
        ax.tick_params(labelsize=9, colors=C_TEXT)
        ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)

    # 标出 P_R 的十格总包络
    allpr = [d[k][("P_R", r)] for k, _ in DATASETS for r, _ in RELIANCES]
    lo_pt, hi_pt = min(t[0] for t in allpr), max(t[0] for t in allpr)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=C_PG, marker="o", ms=5.2, lw=1.3,
               label=r"$P_G$  geometric multi-path gain"),
        Line2D([], [], color=C_PR, marker="o", ms=5.2, lw=1.3,
               markerfacecolor="white", markeredgewidth=1.3,
               label=r"$P_R$  arbitrary multi-path gain"),
        Line2D([], [], color="#8A8A96", marker="o", ms=5.2, lw=0,
               markerfacecolor="white", markeredgewidth=1.3,
               label="open marker: interval covers zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.5, -0.06), handletextpad=0.5,
               columnspacing=1.4)

    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"已写出 {args.output} 与 {args.output.with_suffix('.png')}")

    print("\n=== 与 Table 6 交叉核对 ===")
    print(f"{'dataset':13s}{'qty':6s}{'low load':>23s}{'high load':>23s}")
    for key, lab in DATASETS:
        for k in ("P_G", "P_R", "c2"):
            row = "".join(f"{d[key][(k,r)][0]:>+9.2f} [{d[key][(k,r)][1]:+6.2f},"
                          f"{d[key][(k,r)][2]:+6.2f}]" for r, _ in RELIANCES)
            print(f"{lab:13s}{k:6s}{row}")

    n_excl = sum(excl(t) for t in allpr)
    print(f"\n=== P_R 的十格总览 ===")
    print(f"  点估计范围  [{lo_pt:+.2f}, {hi_pt:+.2f}] pp")
    print(f"  区间排除零的格数  {n_excl} / 10"
          f"{'   (与论文一致)' if n_excl == 0 else '   ★ 与论文不符'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
