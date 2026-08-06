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

版式 (2026-08-04 修订, 应审稿意见):
  左 panel  饱和诊断  -- train acc 超出 95% 标记线的量, 与十三条件的 val acc
                        跨度。两根条形, 无区间, 无零线, 单位是 accuracy points。
  右 panel  注册对比  -- (1) 与 (2) 的点估计与 95% 区间, 有零线,
                        单位是 percentage points of validation accuracy。
  两个 panel 横轴独立。旧版把四个量画在同一根横轴上, 版式本身在诱导读者
  做不可比的相减, 尽管图注写了不可相减。拆分后该诱导消失。
  数值口径与旧版逐位相同, 本次改动只涉及 axes 布局与样式。

用法:
  python3 plot_ceiling.py --output /root/fig7_work/figure5_ceiling.pdf
  python3 plot_ceiling.py --dump-values /root/fig7_work/values_new.tsv
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
    if len(w) != 21:
        raise ValueError(f"{p}: tail window is not 21 epochs")
    for r in w:
        for k in ("train_accuracy", "validation_accuracy"):
            if not 0.0 <= r[k] <= 1.0:
                raise ValueError(f"{p}: {k}={r[k]} outside [0,1]; "
                                 "the *100 rescaling below would be wrong")
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
            "spread": spread, "c1": c1, "c2": c2,
            "pg": ci(pg), "pr": ci(pr)}


def excl(t) -> bool:
    return t[1] > 0 or t[2] < 0


# ---------------------------------------------------------------- 版式


def render(d: dict, output: Path) -> None:
    """两个 panel, 横轴独立。d 的结构见 summarise()。"""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.4, 3.5), sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.07})
    fig.subplots_adjust(left=0.150, right=0.985, top=0.88, bottom=0.28)

    y = np.arange(len(DATASETS))[::-1].astype(float)
    labs = [lab for _, lab in DATASETS]

    axR.axvline(0, color="#8A8A96", lw=1.0, zorder=2)

    for i, (key, _) in enumerate(DATASETS):
        x = d[key]

        # 左: (a) 训练侧超出 95% 标记线的量
        axL.barh(y[i] + 0.16, x["med_train"] - CEILING, height=0.26,
                 color=C_TRAIN, zorder=3)
        axL.text(x["med_train"] - CEILING + 0.22, y[i] + 0.16,
                 f"{x['med_train']:.2f}%", va="center", fontsize=7.2, color=C_TEXT)
        # 左: (b) 十三条件的泛化端跨度
        axL.barh(y[i] - 0.16, x["spread"], height=0.26, color=C_SPREAD, zorder=3)
        axL.text(x["spread"] + 0.22, y[i] - 0.16, f"{x['spread']:.2f}",
                 va="center", fontsize=7.2, color=C_TEXT)

        # 右: (c)(d) 同格内实际解析出的两个对比
        for k, colr, off in (("c1", C_STRUCT, 0.16), ("c2", C_GEOSPEC, -0.16)):
            m, lo, hi = x[k]
            solid = excl((m, lo, hi))
            axR.errorbar(m, y[i] + off, xerr=[[m - lo], [hi - m]], fmt="o", ms=4.6,
                         color=colr, ecolor=colr, elinewidth=1.2, capsize=2.4,
                         capthick=1.2, zorder=4,
                         markerfacecolor=colr if solid else "white",
                         markeredgewidth=1.2)

    for ax in (axL, axR):
        for i in range(len(DATASETS) - 1):
            ax.axhline(y[i] - 0.5, color="#ECECF0", lw=0.8, zorder=0)
        ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_edgecolor("#B8B8C4")
        ax.tick_params(labelsize=9, colors=C_TEXT)
        ax.set_ylim(y[-1] - 0.55, y[0] + 0.55)

    axL.set_yticks(y)
    axL.set_yticklabels(labs, fontsize=9.5, color=C_TEXT)
    axR.tick_params(labelleft=False)

    # 数值标签写在条形末端, 给右侧留出位置, 否则最长的一根会被裁掉。
    _wid = max(max(x["med_train"] - CEILING, x["spread"]) for x in d.values())
    axL.set_xlim(0, _wid * 1.16)
    axL.set_xlabel("accuracy points\n(diagnostic quantities, not effect sizes)",
                   fontsize=8.8, color=C_TEXT)
    axR.set_xlabel("percentage points of validation accuracy\n(95% CI)",
                   fontsize=8.8, color=C_TEXT)
    axL.set_title("Saturation diagnostics", fontsize=9.6, color=C_TEXT, pad=6)
    axR.set_title("Registered contrasts", fontsize=9.6, color=C_TEXT, pad=6)

    axL.legend(handles=[
        Patch(facecolor=C_TRAIN, label="train acc median, amount above the 95% flag"),
        Patch(facecolor=C_SPREAD, label="val acc spread across the 13 conditions"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.26), frameon=False,
        fontsize=7.8, handlelength=1.4, handletextpad=0.5)

    axR.legend(handles=[
        Line2D([], [], color=C_STRUCT, marker="o", ms=4.6, lw=1.2,
               label="structure contrast"),
        Line2D([], [], color=C_GEOSPEC, marker="o", ms=4.6, lw=1.2,
               markerfacecolor="white", markeredgewidth=1.2,
               label=r"$P_G - P_R$  (open: interval covers zero)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.26), frameon=False,
        fontsize=7.8, handlelength=1.4, handletextpad=0.5)

    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"已写出 {output} 与 {output.with_suffix('.png')}")


# ---------------------------------------------------------------- 数值


def dump_values(d: dict, path: Path) -> None:
    cols = ("dataset", "med_train", "flagged", "spread",
            "c1_m", "c1_lo", "c1_hi", "c2_m", "c2_lo", "c2_hi",
            "pg_m", "pg_lo", "pg_hi", "pr_m", "pr_lo", "pr_hi")
    with path.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for key, _ in DATASETS:
            x = d[key]
            row = [key, repr(x["med_train"]), str(x["flagged"]), repr(x["spread"])]
            for k in ("c1", "c2", "pg", "pr"):
                row += [repr(v) for v in x[k]]
            fh.write("\t".join(row) + "\n")
    print(f"已写出 {path}  (全精度, 供改动前后 diff)")


def report(d: dict) -> None:
    print("\n=== 数值 (与 Table 6 / Table 8 交叉核对) ===")
    print(f"{'dataset':13s}{'train med':>11s}{'flag':>6s}{'val spread':>12s}"
          f"{'structure':>24s}{'P_G-P_R':>24s}")
    for key, lab in DATASETS:
        x = d[key]
        print(f"{lab:13s}{x['med_train']:>11.2f}{'yes' if x['flagged'] else 'no':>6s}"
              f"{x['spread']:>12.2f}"
              f"{x['c1'][0]:>+9.2f} [{x['c1'][1]:+6.2f},{x['c1'][2]:+6.2f}]"
              f"{x['c2'][0]:>+9.2f} [{x['c2'][1]:+6.2f},{x['c2'][2]:+6.2f}]")

    # ADDENDUM_03 sec.3: 报告 (2) 时必须同时报告 P_G 与 P_R 两个分量。
    print("\n=== (2) 的两个分量 (ADDENDUM_03 sec.3 要求与 (2) 同时报告) ===")
    print(f"{'dataset':13s}{'P_G':>24s}{'P_R':>24s}")
    for key, lab in DATASETS:
        x = d[key]
        print(f"{lab:13s}"
              f"{x['pg'][0]:>+9.2f} [{x['pg'][1]:+6.2f},{x['pg'][2]:+6.2f}]"
              f"{x['pr'][0]:>+9.2f} [{x['pr'][1]:+6.2f},{x['pr'][2]:+6.2f}]")

    print("\n=== 论证核心 ===")
    for key, lab in DATASETS:
        x = d[key]
        if x["flagged"] and excl(x["c1"]) and not excl(x["c2"]):
            print(f"  {lab}: 饱和标记, 但同格测出 structure = {x['c1'][0]:+.2f} pp, "
                  f"而 P_G-P_R 跨零")


# ---------------------------------------------------------------- 交叉核对

# 冻结值, 逐字读自论文的冻结表, 不是从本脚本的输出回填的 --
# 否则核对就是循环的。来源标在每行注释里。
# 与本脚本的重算结果比对, 不符即中止, 不出图。显示层两位舍入后比较。
FROZEN = {
    # c1/c2/pg/pr: Table 6 的 high load 列。med_train/spread: sec res_ceiling。
    "cifar10":     {"med_train": 95.90, "spread": 14.51,
                    "c1": (+9.94, +9.51, +10.37), "c2": (+4.35, +3.52, +5.18),
                    "pg": (+4.31, +3.45, +5.18),  "pr": (-0.04, -0.35, +0.27)},
    "organamnist": {"med_train": 99.99, "spread": 2.61,
                    "c1": (+1.88, +1.42, +2.34),  "c2": (-0.12, -0.67, +0.43),
                    "pg": (+0.06, -0.17, +0.28),  "pr": (+0.17, -0.20, +0.55)},
    "organcmnist": {"med_train": 99.91, "spread": 2.75,
                    "c1": (+1.99, +0.84, +3.14),  "c2": (+0.47, -0.42, +1.37),
                    "pg": (+0.31, -0.64, +1.26),  "pr": (-0.16, -0.49, +0.16)},
    "organsmnist": {"med_train": 97.45, "spread": 5.90,
                    "c1": (+4.53, +3.13, +5.93),  "c2": (+0.23, -1.31, +1.76),
                    "pg": (+0.31, -0.77, +1.39),  "pr": (+0.08, -0.46, +0.63)},
    "eurosat":     {"med_train": 99.83, "spread": 1.94,
                    "c1": (+1.34, +0.88, +1.81),  "c2": (+0.40, -0.04, +0.85),
                    "pg": (+0.40, -0.33, +1.13),  "pr": (-0.01, -0.37, +0.36)},
}


def crosscheck(d: dict) -> None:
    from decimal import Decimal, ROUND_HALF_UP

    def r2(x):
        return float(Decimal(repr(x)).quantize(Decimal("0.01"),
                                               rounding=ROUND_HALF_UP))

    bad = []
    for key, exp in FROZEN.items():
        got = d[key]
        for field in ("med_train", "spread"):
            if r2(got[field]) != exp[field]:
                bad.append(f"{key}.{field}: recomputed {r2(got[field])} "
                           f"!= frozen {exp[field]}")
        for q in ("c1", "c2", "pg", "pr"):
            for j, nm in enumerate(("point", "lo", "hi")):
                if r2(got[q][j]) != exp[q][j]:
                    bad.append(f"{key}.{q}.{nm}: recomputed {r2(got[q][j])} "
                               f"!= frozen {exp[q][j]}")
    if bad:
        raise SystemExit("交叉核对失败, 未出图:\n  " + "\n  ".join(bad))
    print("交叉核对通过: 70 项与冻结值一致 "
          "(5 x [train med, spread, (1), (2), P_G, P_R])。")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_main"))
    ap.add_argument("--output", type=Path, default=Path("figure5_ceiling.pdf"))
    ap.add_argument("--dump-values", type=Path, default=None,
                    help="把全精度数值写成 TSV, 供改动前后 diff")
    ap.add_argument("--no-crosscheck", action="store_true",
                    help="跳过与冻结值的交叉核对 (仅用于诊断)")
    args = ap.parse_args()

    d = {}
    for key, _ in DATASETS:
        try:
            d[key] = summarise(load(args.runs_root, key))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"FAIL {key}: {e}", file=sys.stderr)
            return 2

    if args.dump_values is not None:
        dump_values(d, args.dump_values)

    if not args.no_crosscheck:
        crosscheck(d)

    render(d, args.output)
    report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
