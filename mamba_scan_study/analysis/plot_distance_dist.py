#!/usr/bin/env python3
"""
plot_distance_dist.py -- Figure: 三个路径族的序列距离分布。

只读脚本。经 resolve_p0b_paths 取路径（含 SHA 校验），不写入任何 run 目录。

为什么需要这张图 (论文 sec lim_lmto):
  辅助族 L 与几何族 G 的匹配只在序列距离分布的三个汇总统计量上成立 ——
  均值、中位数、九十分位 —— 其余一律未匹配: 完整分布、上尾、轴向偏置、
  极性、有向覆盖。该限制目前只以文字陈述。本图把它画出来:
  三条曲线在三个标记的统计量处几乎重合, 而分布形状明显不同。

定义 (取自 P0B_PREREG_FREEZE_L_AUC.md 4 与 REPORT_B1 3):
  对每一对四邻接的 token (u, v), 计算二者在扫描序列中的位置差
      gap(u, v) = | pi^{-1}(u) - pi^{-1}(v) |
  分布取自全部 2n(n-1) 对。这是空间填充曲线的标准局部性度量:
  "空间上相邻的两个 token, 在序列里隔了多远"。

  d_x = 横向邻居对的均值,  d_y = 纵向邻居对的均值
  AxisBias = ln(d_x / d_y)

  冻结的 C5 判据: 每条 L 在每个 grid 下, mean / p50 / p90 均须落在
  对应 G 目标的 +-10% 内。冻结值 (n=32): mean 18.140625, p50 15.0,
  p90 35.0, p95 41.0, max 66, d_x 4.760081, d_y 31.521169,
  AxisBias -1.890395。本脚本重算这些值并与之比对。

用法:
  python3 plot_distance_dist.py
  python3 plot_distance_dist.py --grid 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mamba_scan_study.experiments.p0b_path_bank import resolve_p0b_paths

FAMILIES = (
    ("GEO_DIV", "Geometric", "#3B5FA8"),
    ("RND_D1",  "Arbitrary", "#C4553B"),
    ("LOC_D",   "Auxiliary", "#4A8A6F"),
)
C_GRID = "#D9D9E0"
C_TEXT = "#2A2A32"


def neighbour_gaps(order: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """对每一对四邻接 token, 返回其在扫描序列中的位置差。

    order[t] = 第 t 步访问的 token 索引; 需要的是它的逆置换:
    pos[token] = 该 token 在序列中的位置。
    返回 (横向邻居对的 gap, 纵向邻居对的 gap)。
    """
    pos = np.empty(grid * grid, dtype=np.int64)
    pos[order] = np.arange(grid * grid, dtype=np.int64)
    p = pos.reshape(grid, grid)              # p[row, col]
    gx = np.abs(p[:, 1:] - p[:, :-1]).ravel()   # 横向邻居 (同一行, 相邻列)
    gy = np.abs(p[1:, :] - p[:-1, :]).ravel()   # 纵向邻居 (同一列, 相邻行)
    return gx, gy


def family_distances(exp_id: str, grid: int, seed: int):
    res = resolve_p0b_paths(exp_id, grid, seed)
    ids = list(res.channel_path_ids)
    per_path = []
    for o in res.channel_orders:
        gx, gy = neighbour_gaps(np.asarray(o).astype(np.int64), grid)
        per_path.append({
            "all": np.concatenate([gx, gy]),
            "d_x": float(np.mean(gx)),
            "d_y": float(np.mean(gy)),
        })
    allg = np.concatenate([q["all"] for q in per_path])
    return ids, allg, per_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=32, choices=(8, 32))
    ap.add_argument("--seed", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or Path(f"figure_distance_dist_grid{args.grid}.pdf")

    data = {}
    for exp_id, label, color in FAMILIES:
        try:
            ids, d, per_path = family_distances(exp_id, args.grid, args.seed)
        except Exception as e:
            print(f"FAIL {exp_id}: {e}", file=sys.stderr)
            return 2
        data[label] = {"ids": ids, "d": d, "color": color, "per_path": per_path,
                       "mean": float(np.mean(d)),
                       "median": float(np.median(d)),
                       "p90": float(np.percentile(d, 90)),
                       "p95": float(np.percentile(d, 95)),
                       "max": float(np.max(d)),
                       "d_x": float(np.mean([q["d_x"] for q in per_path])),
                       "d_y": float(np.mean([q["d_y"] for q in per_path]))}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                   gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.subplots_adjust(left=0.095, right=0.985, top=0.885, bottom=0.20, wspace=0.30)

    # --- 左: 互补累积分布, 双对数; 三族跨两个数量级, 线性轴会把 G/L 压在原点 ---
    for label, v in data.items():
        d = np.sort(v["d"])
        xs = np.unique(d)
        ccdf = 1.0 - np.searchsorted(d, xs, side="right") / len(d)
        keep = ccdf > 0
        axA.step(xs[keep], ccdf[keep], where="post", color=v["color"], lw=1.7,
                 label=label, zorder=3)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(0.8, 1400)
    axA.set_xlabel("sequence gap between spatially adjacent tokens",
                   fontsize=9.5, color=C_TEXT)
    axA.set_ylabel("fraction of pairs exceeding $x$", fontsize=9.5, color=C_TEXT)
    axA.set_title("Full distribution (complementary CDF)",
                  fontsize=10, color=C_TEXT, pad=7)
    axA.grid(which="both", color=C_GRID, lw=0.5, zorder=0)
    axA.set_axisbelow(True)
    axA.legend(frameon=False, fontsize=8.6, loc="lower left")

    # 标出三族的最大间隔
    for label, v in data.items():
        axA.annotate(f"{v['max']:.0f}", xy=(v["max"], 1.0 / len(v["d"])),
                     xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=7.4, color=v["color"])

    # --- 右: 只画 G 与 L; 任意族不是匹配目标, 放进来会毁掉尺度 ---
    stats = (("mean", "mean"), ("median", "median"), ("p90", "90th pct"))
    xpos = np.arange(len(stats))
    width = 0.34
    pair = [(lab, data[lab]) for lab in ("Geometric", "Auxiliary")]
    for j, (label, v) in enumerate(pair):
        axB.bar(xpos + (j - 0.5) * width, [v[k] for k, _ in stats],
                width=width * 0.9, color=v["color"], zorder=3, label=label)
    g, l = data["Geometric"], data["Auxiliary"]
    for i, (k, _) in enumerate(stats):
        rel = (l[k] - g[k]) / g[k] * 100
        axB.text(xpos[i], max(g[k], l[k]) * 1.09, f"{rel:+.1f}%",
                 ha="center", fontsize=7.8, color=C_TEXT)
    axB.axhline(0, color="#B8B8C4", lw=0.8)
    axB.set_ylim(0, max(max(g[k], l[k]) for k, _ in stats) * 1.26)
    axB.set_xticks(xpos)
    axB.set_xticklabels([lab for _, lab in stats], fontsize=9, color=C_TEXT)
    axB.set_ylabel("sequence gap", fontsize=9.5, color=C_TEXT)
    axB.set_title("The three matched statistics  ($\\pm10\\%$ tolerance)",
                  fontsize=10, color=C_TEXT, pad=7)
    axB.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)
    axB.set_axisbelow(True)
    axB.legend(frameon=False, fontsize=8.4, loc="upper left")

    for ax in (axA, axB):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_edgecolor("#B8B8C4")
        ax.tick_params(labelsize=9, colors=C_TEXT)

    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"已写出 {out} 与 {out.with_suffix('.png')}")

    print(f"\n=== 三个族的邻接 token 序列间隔 (grid={args.grid}, seed={args.seed}) ===")
    print("  mean/p50/p90/p95/max: 四条合并;  d_x/d_y/AxisBias: 第一条单独")
    print(f"{'family':12s}{'paths':22s}{'mean':>9s}{'p50':>7s}{'p90':>7s}"
          f"{'p95':>7s}{'max':>6s}{'d_x':>9s}{'d_y':>9s}{'AxisBias':>10s}")
    for label, v in data.items():
        import math
        q = v["per_path"][0]
        ab = math.log(q["d_x"] / q["d_y"]) if q["d_y"] > 0 else float("nan")
        print(f"{label:12s}{','.join(v['ids']):22s}{v['mean']:>9.3f}{v['median']:>7.1f}"
              f"{v['p90']:>7.1f}{v['p95']:>7.1f}{v['max']:>6.0f}"
              f"{q['d_x']:>9.3f}{q['d_y']:>9.3f}{ab:>10.4f}")

    if args.grid == 32:
        frozen = {"mean": 18.140625, "p50": 15.0, "p90": 35.0, "p95": 41.0,
                  "max": 66.0, "d_x": 4.760081, "d_y": 31.521169,
                  "AxisBias": -1.890395}
        aux = data["Auxiliary"]
        import math
        # 冻结表的 d_x/d_y/AxisBias 是单条路径 (L1) 的值; 四条平均会互相抵消,
        # 因为 L3/L4 是 L1/L2 的转置。分布统计量 mean/p50/p90 则是四条合并。
        p1 = aux["per_path"][0]
        got = {"mean": aux["mean"], "p50": aux["median"], "p90": aux["p90"],
               "p95": aux["p95"], "max": aux["max"], "d_x": p1["d_x"],
               "d_y": p1["d_y"], "AxisBias": math.log(p1["d_x"] / p1["d_y"])}
        print("\n=== 与 P0B_PREREG_FREEZE_L_AUC.md 的冻结值比对 (辅助族) ===")
        ok = True
        for k, exp in frozen.items():
            g = got[k]
            hit = abs(g - exp) < 5e-4
            ok &= hit
            print(f"  {k:10s} 冻结 {exp:>12.6f}   实得 {g:>12.6f}   "
                  f"{'一致' if hit else '★ 不一致'}")
        print(f"\n  {'距离定义已核实, 本图可用。' if ok else '★ 定义仍不一致, 本图不得进入论文。'}")

    print("\n=== C5 判据: 辅助族的 mean/p50/p90 须在几何族的 +-10% 内 ===")
    g, l = data["Geometric"], data["Auxiliary"]
    for k, nm in (("mean", "mean"), ("median", "p50"), ("p90", "p90")):
        rel = (l[k] - g[k]) / g[k] * 100 if g[k] else float("nan")
        print(f"  {nm:6s}  G {g[k]:9.4f}   L {l[k]:9.4f}   相对差 {rel:+7.2f}%   "
              f"{'在 ±10% 内' if abs(rel) <= 10 else '★ 超出 ±10%'}")
    print("\n注: C5 只约束这三项。完整分布、上尾、轴向偏置、极性、有向覆盖均未匹配,")
    print("    左图的尾部差异即为其直接体现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
