#!/usr/bin/env python3
"""
cap01_judge.py -- CAP-01 容量稳健性臂的 C0-C3 判定。

**只读。** 不写入任何 run 目录, 不写入 git 仓库, 不 commit。
唯一写动作是 --report 指定的报告文件, 且拒绝写入仓库内。

================================================================================
本脚本的权限边界 (PREREG_CAP_01 sec.4.5 / sec.7 第 4 条)
================================================================================
  * 只输出 C0 / C1 / C2 / C3
  * **不判命题 A** -- 已由 MAIN-01 判定并固定为"不成立 (1/5)"
  * **不判 M3**     -- 需五个数据集, 本臂不涉及
  * **不投票**
  * 不读、不实例化、不报告任何 test split
  * 不计算负载交互项 -- 本臂只跑 R_high, 结构上算不了 (sec.2.3 已如实记录)

判据在见到本臂任何数据之前已冻结, 并于 2026-08-03T08:25:30Z 由 GitHub 观测到
推送至 yunshen0126/prereg-timestamps (commit f46a42e), 所记冻结文件 SHA-256:
  36cc44c7394e2b23a810a5bc0d4d62bec290259935392317a0fb37458ca9b029
本脚本不得修改任何判据、不得调整任何阈值。

================================================================================
统计口径
================================================================================
全部复用 analyze_main624.py 的函数, 不重新实现:
  _record_from_metadata  尾窗 epoch 80-100 含端点, 按 run 算术平均, epoch 连续性校验
  _summary               mean +- 3.182 * s / sqrt(4), t(3,0.975), ddof=1, 单位 pp
  _format_display        Decimal(repr(x)) + ROUND_HALF_UP, 仅显示层舍入
  _completion_status     completed.json + final_checkpoint.pt + metadata SHA 三重校验
  _expected_run_directory  d_model 宽度后缀的目录名解析

用法:
  # 冒烟测试: 在 outputs_main (d256) 上验证流水线与论文 Table 6 一致
  python3 cap01_judge.py --selftest

  # 正式判定: CAP-01 跑完之后
  python3 cap01_judge.py --runs-root /root/autodl-tmp/outputs_cap512 --d-model 512
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics as st
import sys
from pathlib import Path

BACKBONE = "mamba"
AUGMENTATION = "main_uniform"
RELIANCE = "R_high"
SEEDS = (0, 1, 2, 3)

CAP01_DATASETS = ("cifar10", "organamnist")

# PREREG_CAP_01 sec.4.0: 该阈值在见到本臂任何数据之前冻结, 不得调整。
# 依据为主实验四个飘红数据集的实测跨度最小值 (EuroSAT 1.94 pp) 的一半。
C0_SPREAD_THRESHOLD_PP = 1.0

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def rule(title: str) -> None:
    say()
    say("=" * 76)
    say(title)
    say("=" * 76)


def load_analyze_module(path: Path, d_model: int):
    """按文件路径 import analyze_main624, 不依赖 PYTHONPATH (analysis 不是 package)。"""
    if not path.is_file():
        raise SystemExit(f"找不到 analyze_main624.py: {path}")
    spec = importlib.util.spec_from_file_location("analyze_main624", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_main624"] = mod
    spec.loader.exec_module(mod)
    mod._D_MODEL = d_model          # 目录名的宽度后缀, 不进入任何统计计算
    return mod


# ----------------------------------------------------------------- 装载


def load_cells(mod, runs_root: Path, dataset: str) -> dict:
    """返回 {(exp_id, seed): RunRecord}。任何一格不完整即中止。"""
    records_by_metadata = {}
    dirs = {}
    for exp_id in mod.EXP_IDS:
        for seed in SEEDS:
            d = mod._expected_run_directory(
                runs_root, dataset, BACKBONE, AUGMENTATION, exp_id, RELIANCE, seed)
            dirs[(exp_id, seed)] = d
            mp = d / "metadata.json"
            if mp.is_file():
                try:
                    records_by_metadata[mp] = mod._record_from_metadata(mp)
                except ValueError:
                    pass

    cells, bad = {}, []
    for (exp_id, seed), d in dirs.items():
        status, record = mod._completion_status(
            d, records_by_metadata, dataset, BACKBONE, AUGMENTATION,
            exp_id, RELIANCE, seed)
        if status != "completed" or record is None:
            bad.append(f"{dataset} {exp_id} seed{seed}: {status}  ({d.name})")
        else:
            cells[(exp_id, seed)] = record
    if bad:
        say(f"[{dataset}] 设计矩阵不完整, {len(bad)} 格不可用:")
        for b in bad:
            say(f"  {b}")
        raise SystemExit(
            f"\n{dataset} 的 {len(mod.EXP_IDS) * len(SEEDS)} 格中有 {len(bad)} 格不可用。\n"
            "判定中止。PREREG_CAP_01 未规定不完整批次的判定方式, 不得就残缺数据判定。")
    return cells


# ----------------------------------------------------------------- 计算


def quantities(mod, cells: dict) -> dict:
    """全部量。seed 配对后再聚合。"""
    def gm(exps, seed):
        return st.mean(cells[(e, seed)].tail_validation_pp for e in exps)

    med_train = st.median(
        cells[(e, s)].tail_train_pp for e in mod.STRUCTURE_EXP_IDS for s in SEEDS)

    per_exp_val = {e: st.mean(cells[(e, s)].tail_validation_pp for s in SEEDS)
                   for e in mod.EXP_IDS}
    lo_e = min(per_exp_val, key=per_exp_val.get)
    hi_e = max(per_exp_val, key=per_exp_val.get)
    spread = per_exp_val[hi_e] - per_exp_val[lo_e]

    c1 = [gm(mod.GEO_SINGLE, s) - gm(mod.RND_SINGLE, s) for s in SEEDS]
    pg = [cells[("GEO_DIV", s)].tail_validation_pp - gm(mod.GEO_SINGLE, s) for s in SEEDS]
    pr = [gm(mod.RND_DIVERSE, s) - gm(mod.RND_SINGLE, s) for s in SEEDS]
    c2 = [a - b for a, b in zip(pg, pr)]

    return {
        "med_train": med_train,
        "flagged": med_train > 95.0,
        "spread": spread,
        "spread_lo": (lo_e, per_exp_val[lo_e]),
        "spread_hi": (hi_e, per_exp_val[hi_e]),
        "c1": mod._summary(c1),
        "P_G": mod._summary(pg),
        "P_R": mod._summary(pr),
        "c2": mod._summary(c2),
    }


def excl(s) -> bool:
    """区间是否排除零。"""
    return s.lower_pp > 0.0 or s.upper_pp < 0.0


def fmt(mod, s) -> str:
    f = mod._format_display
    return f"{f(s.mean_pp, signed=True)} [{f(s.lower_pp, signed=True)}, {f(s.upper_pp, signed=True)}]"


# ----------------------------------------------------------------- 判定


def judge(mod, q: dict) -> dict:
    """C0-C3。判定由脚本执行, 不得由人工读表判断 (PREREG_CAP_01 sec.3)。"""
    c0_ok = excl(q["c1"]) and q["spread"] > C0_SPREAD_THRESHOLD_PP
    return {
        "C0": c0_ok,
        "C1_PR_covers_zero": not excl(q["P_R"]),
        "C2_lower_gt_zero": q["c2"].lower_pp > 0.0,
        "C3_c2_covers_zero": not excl(q["c2"]),
    }


def report_dataset(mod, name: str, q: dict, v: dict) -> None:
    rule(f"{name}   (d_model = {mod._D_MODEL}, {RELIANCE}, {AUGMENTATION}, {BACKBONE})")

    say("--- C0 前置: 本臂在饱和条件下是否仍有测量能力 (sec.4.0) ---")
    say(f"  1. 结构组尾窗 train 中位数   {mod._format_display(q['med_train'])}%"
        f"   {'[天花板标记]' if q['flagged'] else '[未标记]'}")
    say(f"  2. 十三条件 val 跨度         {mod._format_display(q['spread'])} pp"
        f"   ({q['spread_lo'][0]} {mod._format_display(q['spread_lo'][1])}"
        f" .. {q['spread_hi'][0]} {mod._format_display(q['spread_hi'][1])})")
    say(f"     冻结阈值 {C0_SPREAD_THRESHOLD_PP:.1f} pp  ->  "
        f"{'跨度 > 阈值' if q['spread'] > C0_SPREAD_THRESHOLD_PP else '跨度 <= 阈值'}")
    say(f"  3. 结构对比 (1)              {fmt(mod, q['c1'])}"
        f"   {'区间排除零' if excl(q['c1']) else '跨零'}")
    say()
    if v["C0"]:
        say("  >>> C0 通过: 装置在本容量下仍有测量能力。C1/C2/C3 的判定**有效**。")
    else:
        say("  >>> C0 **不通过**: 本臂在该数据集上测量能力不足。")
        say("      该数据集的 C1/C2/C3 结果**如实报告但不予解释**,")
        say("      且**不得用于回应容量意见**。(sec.4.0 表格第二行)")

    say()
    say("--- 各量 (报告 (2) 时必须同时报告 P_G 与 P_R, ADDENDUM_03 sec.3) ---")
    say(f"  P_G                          {fmt(mod, q['P_G'])}")
    say(f"  P_R                          {fmt(mod, q['P_R'])}"
        f"   {'区间排除零' if excl(q['P_R']) else '跨零'}")
    say(f"  (2) = P_G - P_R              {fmt(mod, q['c2'])}"
        f"   {'区间排除零' if excl(q['c2']) else '跨零'}")

    say()
    say("--- 天花板诊断 (sec.4.4, 描述性, 不设通过条件) ---")
    say(f"  结构组 train 中位数 {mod._format_display(q['med_train'])}%, "
        f"标记状态 {'是' if q['flagged'] else '否'}。")
    say("  依 ADDENDUM_03 sec.6, 本项目不预设饱和会压缩或放大效应; 提高容量预计")
    say("  会提高训练准确率, 该方向已知, 故本项仅作记录。")


def report_criteria(mod, res: dict) -> None:
    """C1-C3 的跨数据集结论, 按 PREREG_CAP_01 事前固定的表述输出。"""
    rule("C1  P_R 的零是否跨容量成立  (sec.4.1, 本臂的核心问题)")
    for ds in CAP01_DATASETS:
        q, v = res[ds]
        say(f"  {ds:14s} P_R = {fmt(mod, q['P_R'])}   "
            f"{'含零' if v['C1_PR_covers_zero'] else '**不含零**'}"
            f"{'' if v['C0'] else '   [C0 未通过, 不予解释]'}")
    both = all(res[d][1]["C1_PR_covers_zero"] for d in CAP01_DATASETS)
    say()
    if both:
        say("  >>> 两个都含零。事前固定的结论表述:")
        say("      分支异质性成分为零这一结果在 2 倍容量范围内稳定。")
        say("      论文可写\"该零在两个容量点上均成立\",")
        say("      **不得**写成\"对任意容量成立\"。")
    else:
        say("  >>> 一个或两个不含零。事前固定的结论表述:")
        say("      **这是对 MAIN-01 头条发现的实质限制。**")
        say("      必须如实写入正文与结论: 分支异质性为零不跨容量成立,")
        say("      其适用范围以本臂测得的容量点为界。")
        say("      该情形下论文的核心论证须相应削弱, **不得只在附录提及**。")

    rule("C2  几何成分是否跨容量成立  (sec.4.2, CIFAR-10)")
    q, v = res["cifar10"]
    say(f"  (2) = {fmt(mod, q['c2'])}   下界 "
        f"{'> 0' if v['C2_lower_gt_zero'] else '<= 0 (区间跨零)'}"
        f"{'' if v['C0'] else '   [C0 未通过, 不予解释]'}")
    say()
    if v["C2_lower_gt_zero"]:
        say("  >>> 下界 > 0。事前固定的结论表述:")
        say("      几何专属多路径增益在 2 倍容量下存活。")
        say("      可与 MAIN-01 的 +4.35 并列报告,")
        say("      **不得**表述为\"确认\"或\"复现\"—— 容量不同, 是不同条件下的")
        say("      一致观察。")
    else:
        say("  >>> 区间跨零。事前固定的结论表述:")
        say("      如实报告。**不得**据此声称效应是小容量特有的")
        say("      (单数据集、四 seed 不足以支撑该主张),")
        say("      亦**不得**用于重新解释 MAIN-01 的结果。")

    rule("C3  OrganAMNIST 的零是否跨容量成立  (sec.4.3)")
    q, v = res["organamnist"]
    say(f"  (2) = {fmt(mod, q['c2'])}   "
        f"{'含零' if v['C3_c2_covers_zero'] else '**不含零**'}"
        f"{'' if v['C0'] else '   [C0 未通过, 不予解释]'}")
    say()
    if v["C3_c2_covers_zero"]:
        say("  >>> 含零: MAIN-01 的对应零结果在 2 倍容量下稳定。")
    else:
        say("  >>> 不含零: 如实报告。该情形连同 C1 的第二种结果,")
        say("      共同构成对 MAIN-01 结论的容量依赖性证据。")

    rule("表述承诺 (sec.5) —— 写论文时逐条对照")
    for i, t in enumerate((
        "不得用本臂的任何结果修改、软化或重新表述 MAIN-01 对命题 A 与 M3 的判定。",
        "不得把本臂结果外推至未测容量。最强表述是\"在 d_model ∈ {256, 512} 这一"
        "二倍范围内稳定\", 不得写成\"与容量无关\"或\"对大模型成立\"。",
        "本臂与 MAIN-01 一致时, 不得表述为\"确认\"或\"复现\"。",
        "本臂的事后动机 (sec.0) 必须在论文中**每一处**报告本臂结果的位置同时标注。",
        "不得因本臂结果不利而更换 d_model 后重跑并只报告后一次。",
        "本臂**不回应**\"分辨率过小\"这一意见, 也不得被表述为回应了它。"
        "全部 run 仍为 32x32 输入。",
    ), 1):
        say(f"  {i}. {t}")

    say()
    say("本臂不设投票、不判命题 A、不判 M3 (sec.4.5)。本脚本亦未计算它们。")


# ----------------------------------------------------------------- 自检

# 论文 Table 6 (high load 列) 与 sec res_ceiling 的冻结值, 逐字抄自论文,
# 不是从本脚本输出回填 —— 否则自检是循环的。
SELFTEST_FROZEN = {
    "cifar10": {
        "med_train": "95.90", "spread": "14.51",
        "c1": ("+9.94", "+9.51", "+10.37"), "c2": ("+4.35", "+3.52", "+5.18"),
        "P_G": ("+4.31", "+3.45", "+5.18"), "P_R": ("-0.04", "-0.35", "+0.27"),
    },
    "organamnist": {
        "med_train": "99.99", "spread": "2.61",
        "c1": ("+1.88", "+1.42", "+2.34"), "c2": ("-0.12", "-0.67", "+0.43"),
        "P_G": ("+0.06", "-0.17", "+0.28"), "P_R": ("+0.17", "-0.20", "+0.55"),
    },
}


def selftest(mod, runs_root: Path) -> int:
    rule("自检: 在 outputs_main (d_model = 256) 上验证流水线与论文 Table 6 一致")
    say("本模式不产生任何 CAP-01 结论, 只验证判定流水线的口径。")
    bad = []
    for ds in CAP01_DATASETS:
        cells = load_cells(mod, runs_root, ds)
        q = quantities(mod, cells)
        exp = SELFTEST_FROZEN[ds]
        got = {
            "med_train": mod._format_display(q["med_train"]),
            "spread": mod._format_display(q["spread"]),
        }
        for k in ("med_train", "spread"):
            if got[k] != exp[k]:
                bad.append(f"{ds}.{k}: 实得 {got[k]} != 冻结 {exp[k]}")
        for k in ("c1", "c2", "P_G", "P_R"):
            s = q[k]
            trip = tuple(mod._format_display(x, signed=True)
                         for x in (s.mean_pp, s.lower_pp, s.upper_pp))
            if trip != exp[k]:
                bad.append(f"{ds}.{k}: 实得 {trip} != 冻结 {exp[k]}")
        say(f"  {ds:14s} train med {got['med_train']}%  spread {got['spread']} pp  "
            f"(1) {fmt(mod, q['c1'])}  (2) {fmt(mod, q['c2'])}")
        say(f"  {'':14s} P_G {fmt(mod, q['P_G'])}   P_R {fmt(mod, q['P_R'])}")

    say()
    if bad:
        for b in bad:
            say(f"  [失配] {b}")
        say()
        say(">>> **自检失败。** 判定流水线与产出论文表格的算法不一致, ")
        say("    在查清之前不得用本脚本判定 CAP-01。")
        return 1
    say(">>> 自检通过: 24 项与论文 Table 6 逐位一致。")
    say("    判定流水线与产出论文表格的算法同源。")
    return 0


# -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--repo", type=Path, default=Path("/root/mamba-scan-study"))
    ap.add_argument("--analyze", type=Path, default=None,
                    help="analyze_main624.py 的路径; 默认由 --repo 推出")
    ap.add_argument("--report", type=Path,
                    default=Path("/root/fig7_work/cap01_judge_report.txt"))
    ap.add_argument("--selftest", action="store_true",
                    help="在 outputs_main (d256) 上验证流水线, 不判定 CAP-01")
    args = ap.parse_args()

    if args.selftest:
        runs_root = args.runs_root or Path("/root/autodl-tmp/outputs_main")
        d_model = 256
        args.report = args.report.with_name("cap01_judge_selftest.txt")
    else:
        runs_root = args.runs_root or Path("/root/autodl-tmp/outputs_cap512")
        d_model = args.d_model

    if args.report.resolve().is_relative_to(args.repo.resolve()):
        print(f"拒绝把报告写进仓库 ({args.report})，会弄脏工作树。", file=sys.stderr)
        return 2
    if not runs_root.is_dir():
        print(f"产物根不存在: {runs_root}", file=sys.stderr)
        return 2

    analyze_path = args.analyze or (args.repo / "mamba_scan_study" / "analysis"
                                    / "analyze_main624.py")
    mod = load_analyze_module(analyze_path, d_model)

    say("CAP-01 判定  cap01_judge.py")
    say(f"  产物根          {runs_root}")
    say(f"  d_model         {d_model}")
    say(f"  口径来源        {analyze_path}  (import, 未重新实现)")
    say(f"  冻结判据 SHA    36cc44c7394e2b23a810a5bc0d4d62bec290259935392317a0fb37458ca9b029")
    say(f"  带外时间戳      f46a42e @ 2026-08-03T08:25:30Z (GitHub PushEvent)")

    if args.selftest:
        rc = selftest(mod, runs_root)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(OUT) + "\n")
        print(f"\n报告已写出: {args.report}")
        return rc

    res = {}
    for ds in CAP01_DATASETS:
        cells = load_cells(mod, runs_root, ds)
        q = quantities(mod, cells)
        v = judge(mod, q)
        res[ds] = (q, v)
        report_dataset(mod, ds, q, v)

    report_criteria(mod, res)

    rule("提醒")
    say("本臂**不是结果盲的** (PREREG_CAP_01 sec.0): 判据在见到本臂数据前冻结,")
    say("但起草于 MAIN-01 全部结果已知之后。论文中每一处报告本臂结果的位置")
    say("**必须同时标注**其动机为事后的。")
    say()
    say("判定结果写入 CAP01_RESULTS.md, **不追加进 PREREG_CAP_01.md**")
    say("(ledger 8k (e): 追加会使该文件 SHA 偏离带外时间戳所记的")
    say(" 36cc44c7..., 断裂目前唯一的\"自冻结后零改动\"直接证据)。")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(OUT) + "\n")
    print(f"\n报告已写出: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
