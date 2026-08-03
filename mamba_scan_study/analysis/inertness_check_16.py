#!/usr/bin/env python3
"""
inertness_check_16.py -- CODE_DELTA_68dff0b_32edce6.md 5.1 的 16 格代码惰性检验。

只读脚本。不写入任何 run 目录，不修改任何冻结产物，不调用训练代码。

比较对象:
  recomputed  main experiment, cifar10 / mamba / main_uniform / R_high  (outputs_main)
  archived    augmentation sensitivity check, same 16 cells             (outputs_aug16)

两侧目录命名不同, 这本身是 68dff0b..32edce6 的改动之一 (CODE_DELTA 2.2):
  recomputed  p0b_cifar10_main_uniform_mamba_{exp}_R_high_seed{s}   (含 backbone 段)
  archived    p0b_cifar10_main_uniform_{exp}_R_high_seed{s}         (无 backbone 段)

判据 (CODE_DELTA 5.1, 于见到复算值之前冻结):
  A  复算 P_G' 点估计落在 [2.14, 4.86] 内            -> 主判据
  B  复算 P_R' 的 95% CI 含零                        -> 辅助判据
  C  逐格报告 16 个尾窗 val acc 与旧值之差, 不设阈值  -> 描述性

A 或 B 任一不过 -> CODE_DELTA 7 的数值惰性主张撤回。
C 不得事后补设阈值 (5.1 明文)。

统计口径 (P0B_PREREG_ANALYSIS_PLAN 2/3):
  尾窗 epoch 80-100 含端点, 21 个 epoch, 按 run 取算术平均
  区间 mean +- 3.182 * s / sqrt(4), t(3, 0.975), ddof=1, 单位 pp
  对比按 seed 配对后再聚合, 聚合层不做任何中间舍入
  显示层 Decimal(repr(x)).quantize('0.01', ROUND_HALF_UP)

用法:
  python3 inertness_check_16.py
  python3 inertness_check_16.py --main-root /root/autodl-tmp/outputs_main \
                                --archive-root /root/autodl-tmp/outputs_aug16
  python3 inertness_check_16.py --emit latex
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt
from pathlib import Path

EXPS = ("GEO_DIV", "GEO_SG1", "RND_D1", "RND_S1")
SEEDS = (0, 1, 2, 3)

# 尾窗: epoch 80-100 含端点 => 0-based 索引 79..99
TAIL_LO, TAIL_HI = 79, 100

T_CRIT = 3.182          # t(3, 0.975)
RECORDED_PG_INTERVAL = (2.14, 4.86)   # CODE_DELTA 5.1 记录值区间, 来自 MAIN_PREREG_01 5.2
RECORDED_PG_POINT = 3.50              # 同上, 记录点估计


def q2(x: float) -> str:
    """显示层舍入, 只在输出时使用。"""
    d = Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{d:+.2f}"


def q2u(x: float) -> str:
    """无符号显示, 用于准确率。"""
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def tail_mean(meta_path: Path, key: str) -> float:
    """读取 validation_history, 返回尾窗内 key 的算术平均, 单位 pp。"""
    with meta_path.open() as fh:
        meta = json.load(fh)
    hist = meta.get("validation_history")
    if hist is None:
        raise KeyError(f"{meta_path}: 缺少 validation_history")
    if len(hist) != 100:
        raise ValueError(f"{meta_path}: validation_history 长度为 {len(hist)}, 应为 100")
    window = hist[TAIL_LO:TAIL_HI]
    if len(window) != 21:
        raise ValueError(f"{meta_path}: 尾窗长度为 {len(window)}, 应为 21")
    vals = []
    for row in window:
        if key not in row:
            raise KeyError(f"{meta_path}: 尾窗某 epoch 缺少字段 {key}")
        vals.append(row[key])
    return st.mean(vals) * 100.0


def ci(values) -> tuple[float, float, float]:
    """返回 (mean, lo, hi)。不做任何中间舍入。"""
    n = len(values)
    if n != 4:
        raise ValueError(f"每格应为 4 个观测, 实得 {n}")
    m = st.mean(values)
    half = T_CRIT * st.stdev(values) / sqrt(n)
    return m, m - half, m + half


def resolve(root: Path, exp: str, seed: int, with_backbone: bool) -> Path:
    name = (
        f"p0b_cifar10_main_uniform_mamba_{exp}_R_high_seed{seed}"
        if with_backbone
        else f"p0b_cifar10_main_uniform_{exp}_R_high_seed{seed}"
    )
    path = root / name / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    return path


def collect(root: Path, with_backbone: bool, key: str) -> dict:
    return {
        (exp, seed): tail_mean(resolve(root, exp, seed, with_backbone), key)
        for exp in EXPS
        for seed in SEEDS
    }


def paired(cells: dict, minuend: str, subtrahend: str) -> list:
    """按 seed 配对后再聚合 (ANALYSIS_PLAN 3)。"""
    return [cells[(minuend, s)] - cells[(subtrahend, s)] for s in SEEDS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-root", default="/root/autodl-tmp/outputs_main",
                    help="复算侧: 主实验产物根目录")
    ap.add_argument("--archive-root", default="/root/autodl-tmp/outputs_aug16",
                    help="归档侧: 增强敏感性检查产物根目录")
    ap.add_argument("--emit", choices=("text", "latex"), default="text")
    args = ap.parse_args()

    main_root = Path(args.main_root)
    arch_root = Path(args.archive_root)

    try:
        new_val = collect(main_root, with_backbone=True, key="validation_accuracy")
        old_val = collect(arch_root, with_backbone=False, key="validation_accuracy")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    new_pg, new_pr = paired(new_val, "GEO_DIV", "GEO_SG1"), paired(new_val, "RND_D1", "RND_S1")
    old_pg, old_pr = paired(old_val, "GEO_DIV", "GEO_SG1"), paired(old_val, "RND_D1", "RND_S1")

    n_pg, n_pr = ci(new_pg), ci(new_pr)
    o_pg, o_pr = ci(old_pg), ci(old_pr)

    crit_a = RECORDED_PG_INTERVAL[0] <= n_pg[0] <= RECORDED_PG_INTERVAL[1]
    crit_b = n_pr[1] <= 0.0 <= n_pr[2]
    # 归档侧自洽性: 归档 P_G' 应重现 CODE_DELTA 5.1 的记录值
    self_check = abs(o_pg[0] - RECORDED_PG_POINT) < 0.02

    if args.emit == "latex":
        print("% ==== TABLE: inertness ====")
        print("% source: inertness_check_16.py")
        print("condition & quantity & seed 0 & seed 1 & seed 2 & seed 3 \\\\")
        for exp in EXPS:
            esc = exp.replace("_", r"\_")
            row = lambda lbl, f: print(
                f"\\texttt{{{esc}}} & {lbl} & "
                + " & ".join(f(s) for s in SEEDS) + r" \\")
            row("recomputed", lambda s: q2u(new_val[(exp, s)]))
            row("archived", lambda s: q2u(old_val[(exp, s)]))
            row("difference", lambda s: f"${q2(new_val[(exp, s)] - old_val[(exp, s)])}$")
        return 0 if (crit_a and crit_b) else 1

    w = 74
    print("=" * w)
    print("CODE_DELTA 5.1  16-cell inertness check")
    print("=" * w)
    print(f"  recomputed : {main_root}")
    print(f"  archived   : {arch_root}")
    print(f"  endpoint   : validation accuracy, tail window epoch 80-100 inclusive")
    print()

    print("--- 判据 C: 逐格尾窗 validation accuracy (%), 描述性, 无阈值 ---")
    print(f"  {'condition':10s} {'quantity':11s} " + " ".join(f"{'seed '+str(s):>8s}" for s in SEEDS))
    diffs = []
    for exp in EXPS:
        print(f"  {exp:10s} {'recomputed':11s} " + " ".join(f"{q2u(new_val[(exp,s)]):>8s}" for s in SEEDS))
        print(f"  {'':10s} {'archived':11s} " + " ".join(f"{q2u(old_val[(exp,s)]):>8s}" for s in SEEDS))
        d = [new_val[(exp, s)] - old_val[(exp, s)] for s in SEEDS]
        diffs += d
        print(f"  {'':10s} {'difference':11s} " + " ".join(f"{q2(x):>8s}" for x in d))
    print(f"\n  n=16, 跨度 [{q2(min(diffs))}, {q2(max(diffs))}], "
          f"绝对值 >= 0.10 的格数: {sum(1 for x in diffs if abs(x) >= 0.10)}")
    print("  依 CODE_DELTA 5.1, 本项不设阈值, 且不得事后补设。")
    print()

    print("--- 受限对比 ---")
    for lbl, (m, lo, hi) in (("P_G' recomputed", n_pg), ("P_G' archived  ", o_pg),
                             ("P_R' recomputed", n_pr), ("P_R' archived  ", o_pr)):
        print(f"  {lbl}  {q2(m)} [{q2(lo)}, {q2(hi)}]")
    print()

    print("--- 归档侧自洽性校验 ---")
    print(f"  归档 P_G' 应重现 CODE_DELTA 5.1 记录值 {RECORDED_PG_POINT:+.2f} "
          f"{list(RECORDED_PG_INTERVAL)}")
    print(f"  实得 {q2(o_pg[0])} [{q2(o_pg[1])}, {q2(o_pg[2])}]  ->  "
          + ("吻合" if self_check else "★ 不吻合: 归档目录可能不是判据注册时所指的那批"))
    print()

    print("--- 判定 ---")
    print(f"  A  P_G' 点估计 {q2(n_pg[0])} 落在 {list(RECORDED_PG_INTERVAL)} 内"
          f"  ->  {'通过' if crit_a else '不通过'}")
    print(f"  B  P_R' 区间 [{q2(n_pr[1])}, {q2(n_pr[2])}] 含零"
          f"  ->  {'通过' if crit_b else '不通过'}")
    print(f"  C  描述性, 无判定")
    print()

    print("--- 功效边界 (CODE_DELTA 5.1 要求与结果同时报告) ---")
    width = RECORDED_PG_INTERVAL[1] - RECORDED_PG_INTERVAL[0]
    print(f"  记录值区间宽 {width:.2f} pp, 判据 A 对小幅系统性偏移检出力很低;")
    print(f"  通过 A 只排除大幅偏移, 不排除 0.5 pp 量级的系统平移。")
    print(f"  本研究无 '纯 GPU 非确定性下重跑同一配置的差异幅度' 基线测量,")
    print(f"  故判据 C 的差值无法拆分为代码导致与非确定性导致的部分。")
    print()

    if not self_check:
        print("★ 归档侧自洽性校验未通过, 判定结果不可采信。")
        return 2
    if crit_a and crit_b:
        print("结论: A 与 B 均通过, CODE_DELTA 7 的数值惰性主张不撤回。")
        return 0
    print("结论: A 或 B 不通过, 依 CODE_DELTA 5.1, "
          "7 的数值惰性主张须撤回, 差异须调查并如实报告。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
