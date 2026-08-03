#!/usr/bin/env python3
"""
plot_paths.py -- Figure 2: 三个冻结路径族的实际遍历轨迹。

只读脚本。经 resolve_p0b_paths 取路径（含 SHA 校验），不触碰 JSON 内部结构，
不写入任何 run 目录。

布局: 3 行 x 4 列
  行 1  G1..G4   几何族 (行主序 / 其反转 / 列主序 / 其反转)
  行 2  R1..R4   任意族 (冻结的随机排列)
  行 3  L1..L4   辅助族 (三统计量局部性匹配、拓扑扰动)

每格在 n x n 的 token 网格上画出遍历折线, 颜色沿路径由浅到深表示先后次序。
默认 grid=8 (低负载档): 64 个 token 能看清转折。grid=32 是 1024 个点, 画出来
是一团黑, 仅在需要时用 --grid 32 生成附录版。

用法:
  python3 plot_paths.py
  python3 plot_paths.py --grid 32 --output figure2_paths_grid32.pdf
  python3 plot_paths.py --seed 1          # R_S 族随 seed 变化, 见 training_mapping
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
from matplotlib.collections import LineCollection

from mamba_scan_study.experiments.p0b_path_bank import resolve_p0b_paths

# 每行取一个 diverse 条件, 它一次给出该族的四条路径
ROWS = (
    ("GEO_DIV", "Geometric family", "#3B5FA8"),
    ("RND_D1",  "Arbitrary family", "#C4553B"),
    ("LOC_D",   "Auxiliary family", "#4A8A6F"),
)


def orders_for(exp_id: str, grid: int, seed: int):
    """返回 (path_ids, [order arrays])。order[i] = 第 i 步访问的 token 索引。"""
    r = resolve_p0b_paths(exp_id, grid, seed)
    return r.channel_path_ids, [np.asarray(o).astype(np.int64) for o in r.channel_orders]


def draw_path(ax, order: np.ndarray, grid: int, base_color: str, title: str) -> None:
    """在 grid x grid 网格上画遍历折线, 颜色沿路径加深。"""
    # token 索引 -> (row, col); row-major 的 patch 展平约定
    rows, cols = np.divmod(order, grid)
    x, y = cols.astype(float), rows.astype(float)

    # 底层点阵
    gx, gy = np.meshgrid(np.arange(grid), np.arange(grid))
    ax.scatter(gx, gy, s=2.5 if grid <= 8 else 0.6, c="#D9D9E0", zorder=1,
               linewidths=0)

    # 折线, 分段着色表示先后
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = _shaded_cmap(base_color)
    lc = LineCollection(segs, cmap=cmap, linewidths=1.4 if grid <= 8 else 0.35,
                        capstyle="round", joinstyle="round", zorder=2)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)

    # 起点终点
    ax.scatter([x[0]], [y[0]], s=26 if grid <= 8 else 10, c="white",
               edgecolors=base_color, linewidths=1.4, zorder=3)
    ax.scatter([x[-1]], [y[-1]], s=26 if grid <= 8 else 10, c=base_color,
               zorder=3, linewidths=0)

    m = 0.6
    ax.set_xlim(-m, grid - 1 + m)
    ax.set_ylim(grid - 1 + m, -m)          # 反转 y, 让 (0,0) 在左上
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#C8C8D0"); s.set_linewidth(0.8)
    ax.set_title(title, fontsize=10, pad=4, color="#2A2A32")


def _shaded_cmap(hex_color: str):
    """由基色生成 浅->深 的两端渐变。"""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    r, g, b = to_rgb(hex_color)
    light = (min(1.0, r + (1 - r) * 0.70),
             min(1.0, g + (1 - g) * 0.70),
             min(1.0, b + (1 - b) * 0.70))
    dark = (r * 0.65, g * 0.65, b * 0.65)
    return LinearSegmentedColormap.from_list("shade", [light, hex_color, dark])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=8, choices=(8, 32))
    ap.add_argument("--seed", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    out = args.output or Path(f"figure2_paths_grid{args.grid}.pdf")

    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.7))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.945, bottom=0.055,
                        wspace=0.16, hspace=0.30)

    for i, (exp_id, row_label, color) in enumerate(ROWS):
        try:
            path_ids, orders = orders_for(exp_id, args.grid, args.seed)
        except Exception as exc:
            print(f"FAIL {exp_id}: {exc}", file=sys.stderr)
            return 2
        if len(orders) != 4:
            print(f"FAIL {exp_id}: 期望 4 条路径, 实得 {len(orders)}", file=sys.stderr)
            return 2
        for j in range(4):
            draw_path(axes[i][j], orders[j], args.grid, color, path_ids[j])
        # 行标签
        axes[i][0].text(-0.34, 0.5, row_label, transform=axes[i][0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=10.5, color=color, weight="semibold")

    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"已写出 {out} 与 {out.with_suffix('.png')}")

    # 描述性核对: 每条路径必须是 0..n^2-1 的一个排列
    print("\n=== 完整性核对 ===")
    n = args.grid * args.grid
    for exp_id, label, _ in ROWS:
        ids, orders = orders_for(exp_id, args.grid, args.seed)
        for pid, o in zip(ids, orders):
            ok = (len(o) == n and np.array_equal(np.sort(o), np.arange(n)))
            print(f"  {label:18s} {pid:3s}  长度 {len(o):5d}  "
                  f"{'是 0..n-1 的排列' if ok else '★ 不是合法排列'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
