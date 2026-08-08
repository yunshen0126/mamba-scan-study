#!/usr/bin/env python3
"""
plot_load_gating.py -- Figure 4: 负载门控在两个对比上的不对称。

只读脚本。从 outputs_main 的 metadata 直接重算, 不写入任何 run 目录。

要呈现的事实 (论文 sec 6.3):
  contrast (1) structure  = mean(GEO_S*) - mean(RND_S*)
      其负载交互项在五个数据集上全部区间排除零
  contrast (2) P_G - P_R  = path-family x diversity interaction
      其负载交互项只在 CIFAR-10 上区间排除零

即: 扫描负载调节"排序质量"的价值远比它调节"排序多样性"的价值来得广泛。

统计口径 (P0B_PREREG_ANALYSIS_PLAN 2/3):
  尾窗 epoch 80-100 含端点, 按 run 取算术平均
  区间 mean +- 3.182 * s / sqrt(4), t(3,0.975), ddof=1, 单位 pp
  对比按 seed 配对后再聚合, 聚合层不做中间舍入

用法:
  python3 plot_load_gating.py
  python3 plot_load_gating.py --runs-root /root/autodl-tmp/outputs_main
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
RELIANCES = ("R_low", "R_high")
TAIL_LO, TAIL_HI = 79, 100
T_CRIT = 3.182

C_STRUCT = "#3B5FA8"     # contrast (1), 与 Fig 2 几何族同色
C_GEOSPEC = "#C4553B"    # contrast (2)
C_GRID = "#D9D9E0"
C_TEXT = "#2A2A32"


def tail(meta_path: Path) -> float:
    with meta_path.open() as fh:
        m = json.load(fh)
    h = m["validation_history"]
    if len(h) != 100:
        raise ValueError(f"{meta_path}: history length {len(h)}")
    w = h[TAIL_LO:TAIL_HI]
    if len(w) != 21:
        raise ValueError(f"{meta_path}: tail window {len(w)}")
    return st.mean(r["validation_accuracy"] for r in w) * 100.0


def load_cells(root: Path, dataset: str) -> dict:
    cells = {}
    for exp in GEO_SINGLE + GEO_DIVERSE + RND_SINGLE + RND_DIVERSE:
        for rel in RELIANCES:
            for s in SEEDS:
                p = root / f"p0b_{dataset}_main_uniform_mamba_{exp}_{rel}_seed{s}" / "metadata.json"
                if not p.is_file():
                    raise FileNotFoundError(p)
                cells[(exp, rel, s)] = tail(p)
    return cells


def gmean(cells, exps, rel, seed):
    return st.mean(cells[(e, rel, seed)] for e in exps)


def ci(vals):
    m = st.mean(vals)
    h = T_CRIT * st.stdev(vals) / sqrt(len(vals))
    return m, m - h, m + h


def contrasts(cells) -> dict:
    """返回 {(name, reliance): (mean, lo, hi)} 与 {(name,'pair'): ...}"""
    out = {}
    per_seed = {}
    for rel in RELIANCES:
        c1 = [gmean(cells, GEO_SINGLE, rel, s) - gmean(cells, RND_SINGLE, rel, s)
              for s in SEEDS]
        pg = [gmean(cells, GEO_DIVERSE, rel, s) - gmean(cells, GEO_SINGLE, rel, s)
              for s in SEEDS]
        pr = [gmean(cells, RND_DIVERSE, rel, s) - gmean(cells, RND_SINGLE, rel, s)
              for s in SEEDS]
        c2 = [a - b for a, b in zip(pg, pr)]
        per_seed[("c1", rel)] = c1
        per_seed[("c2", rel)] = c2
        out[("c1", rel)] = ci(c1)
        out[("c2", rel)] = ci(c2)
    for k in ("c1", "c2"):
        paired = [hi - lo for hi, lo in zip(per_seed[(k, "R_high")], per_seed[(k, "R_low")])]
        out[(k, "pair")] = ci(paired)
    return out


def excl(t) -> bool:
    return t[1] > 0 or t[2] < 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_main"))
    ap.add_argument("--output", type=Path, default=Path("figure4_load_gating.pdf"))
    args = ap.parse_args()

    data = {}
    for key, _ in DATASETS:
        try:
            data[key] = contrasts(load_cells(args.runs_root, key))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"FAIL {key}: {e}", file=sys.stderr)
            return 2

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.86, bottom=0.155, wspace=0.10)

    ypos = np.arange(len(DATASETS))[::-1]

    for ax, (col, title) in zip(axes, (("R_low", "Low load  ($L=64$)"),
                                       ("R_high", "High load  ($L=1024$)"))):
        ax.axvline(0, color="#8A8A96", lw=1.0, zorder=1)
        for i, (key, _) in enumerate(DATASETS):
            y = ypos[i]
            for k, colr, off in (("c1", C_STRUCT, 0.16), ("c2", C_GEOSPEC, -0.16)):
                m, lo, hi = data[key][(k, col)]
                solid = excl((m, lo, hi))
                ax.errorbar(m, y + off, xerr=[[m - lo], [hi - m]], fmt="o",
                            ms=5.2, color=colr, ecolor=colr, elinewidth=1.3,
                            capsize=2.6, capthick=1.3, zorder=3,
                            markerfacecolor=colr if solid else "white",
                            markeredgewidth=1.3)
        ax.set_yticks(ypos)
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

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=C_STRUCT, marker="o", ms=5.2, lw=1.3,
               label=r"structure  mean($G$ single) $-$ mean($R$ single)"),
        Line2D([], [], color=C_GEOSPEC, marker="o", ms=5.2, lw=1.3,
               label=r"interaction (2)  $P_G - P_R$"),
        Line2D([], [], color="#8A8A96", marker="o", ms=5.2, lw=0,
               markerfacecolor="white", markeredgewidth=1.3,
               label="open marker: interval covers zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.5, -0.055), handletextpad=0.5,
               columnspacing=1.4)

    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"已写出 {args.output} 与 {args.output.with_suffix('.png')}")

    print("\n=== 与 Table 6 交叉核对 (应逐位一致) ===")
    print(f"{'dataset':13s}{'quantity':10s}{'low load':>22s}{'high load':>22s}{'paired':>22s}")
    for key, lab in DATASETS:
        for k, nm in (("c1", "structure"), ("c2", "P_G-P_R")):
            row = "".join(
                f"{data[key][(k,c)][0]:+8.2f} [{data[key][(k,c)][1]:+6.2f},{data[key][(k,c)][2]:+6.2f}]"
                for c in ("R_low", "R_high", "pair"))
            print(f"{lab:13s}{nm:10s}{row}")
    print("\n=== 负载交互项区间排除零的数据集 ===")
    for k, nm in (("c1", "structure  "), ("c2", "P_G - P_R  ")):
        hit = [lab for key, lab in DATASETS if excl(data[key][(k, "pair")])]
        print(f"  {nm}: {', '.join(hit) if hit else '无'}  ({len(hit)}/5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
