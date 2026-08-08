#!/usr/bin/env python3
"""coverage_nodes.py -- 补充材料 5.2: 四方向 C_dir 在四个冻结节点上的值。

只读脚本。经 resolve_p0b_paths 取路径（含 SHA 校验），不写入任何 run 目录。

定义取自 P0B_PREREG_FREEZE_L_AUC.md 7:
  delta(u->v) = min_{pi in Pi} { pos_pi[v] - pos_pi[u] : pos_pi[v] > pos_pi[u] }
                +inf 若集合中无路径在 u 之后访问 v
  C_dir(x)    = population 中 delta <= x*(N-1) 的比例
  节点 x in {0.01, 0.05, 0.10, 0.20}, 加数学锚点 C_dir(0)=0
  AUC_dir     = numpy.trapezoid(y_nodes, x_nodes) / 0.20
  AUC_macro   = mean over RIGHT/LEFT/DOWN/UP

用法:
  python3 coverage_nodes.py            # 两个 grid 都算, 打印表并输出 LaTeX
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from mamba_scan_study.experiments.p0b_path_bank import resolve_p0b_paths

NODES = (0.01, 0.05, 0.10, 0.20)
DIRS = ("RIGHT", "LEFT", "DOWN", "UP")

# P0B_PREREG_FREEZE_L_AUC.md 7 的冻结 AUC_macro。不符即中止。
FROZEN = {
    (8,  "G4"):   0.85,
    (8,  "G2"):   0.425,
    (8,  "LMTO"): 0.7991071428571429,
    (32, "G4"):   0.975,
    (32, "G2"):   0.4875,
    (32, "LMTO"): 0.9645413306451613,
}


def populations(n):
    idx = np.arange(n * n)
    r, c = np.divmod(idx, n)
    return {
        "RIGHT": (idx[c < n - 1], idx[c < n - 1] + 1),
        "LEFT":  (idx[c > 0],     idx[c > 0] - 1),
        "DOWN":  (idx[r < n - 1], idx[r < n - 1] + n),
        "UP":    (idx[r > 0],     idx[r > 0] - n),
    }


def c_dir(orders, n):
    """orders: 每条路径的 order 数组 (order[t] = 第 t 步访问的 token)。"""
    N = n * n
    positions = []
    for o in orders:
        p = np.empty(N, dtype=np.int64)
        p[np.asarray(o).astype(np.int64)] = np.arange(N, dtype=np.int64)
        positions.append(p)
    out = {}
    for d, (u, v) in populations(n).items():
        best = np.full(len(u), np.inf)
        for p in positions:
            step = (p[v] - p[u]).astype(float)
            step[step <= 0] = np.inf
            best = np.minimum(best, step)
        out[d] = [float(np.mean(best <= x * (N - 1))) for x in NODES]
    return out


# P0B_PREREG_FREEZE_L_AUC.md 7 指定 numpy.trapezoid; 该函数在 numpy 2.0 前名为
# trapz, 两者为同一实现的改名, 数值逐位相同。此处按可用者绑定。
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def auc(y):
    return float(_TRAPZ([0.0] + list(y), [0.0] + list(NODES)) / 0.20)


def sets_for(n):
    """返回 {label: [orders]}。G 族取 GEO_DIV, LMTO 取 LOC_D, 均在 seed 0。"""
    g = resolve_p0b_paths("GEO_DIV", n, 0)
    by_id = dict(zip(g.channel_path_ids, g.channel_orders))
    missing = {"G1", "G3"} - set(by_id)
    if missing:
        sys.exit(f"FAIL: GEO_DIV 未返回 {missing}, 实得 {list(by_id)}")
    l = resolve_p0b_paths("LOC_D", n, 0)
    return {
        "G4":   list(g.channel_orders),
        "G2":   [by_id["G1"], by_id["G3"]],
        "LMTO": list(l.channel_orders),
    }


def main() -> int:
    rows, ok = [], True
    for n in (8, 32):
        for label, orders in sets_for(n).items():
            cd = c_dir(orders, n)
            aucs = {d: auc(cd[d]) for d in DIRS}
            macro = float(np.mean([aucs[d] for d in DIRS]))
            exp = FROZEN[(n, label)]
            hit = abs(macro - exp) < 1e-9
            ok &= hit
            print(f"\nn={n}  {label}   AUC_macro 冻结 {exp:.16f}  "
                  f"实得 {macro:.16f}  {'一致' if hit else '★ 不一致'}")
            for d in DIRS:
                print(f"   {d:6s} " +
                      "  ".join(f"C({x:.2f})={v:.4f}" for x, v in zip(NODES, cd[d])) +
                      f"   AUC={aucs[d]:.10f}")
            rows.append((n, label, cd, aucs, macro))
    if not ok:
        print("\n★ 与冻结值不符, 结果不得进入论文。", file=sys.stderr)
        return 2

    print("\n=== LaTeX (supplementary 5.2) ===")
    name = {"G4": r"$\{G_1,G_2,G_3,G_4\}$", "G2": r"$\{G_1,G_3\}$",
            "LMTO": r"$\mathcal{L}$"}
    for n, label, cd, aucs, macro in rows:
        for d in DIRS:
            cells = " & ".join(f"{v:.4f}" for v in cd[d])
            print(f"${n}$ & {name[label]} & \\textsc{{{d.lower()}}} & {cells} "
                  f"& {aucs[d]:.4f} \\\\")
        print(r"\midrule")
    print("\n八项冻结值全部一致, 本表可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())