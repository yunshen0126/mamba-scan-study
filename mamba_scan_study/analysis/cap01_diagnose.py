#!/usr/bin/env python3
"""
cap01_diagnose.py -- CAP-01 的逐条件逐 seed 诊断。

**只读，纯描述。** 不判定、不改判据、不排除任何 run、不建议排除任何 run。

为什么需要它:
  C0 在两个数据集上都不通过 —— 结构对比 (1) 的区间跨零, 而十三条件的 val
  跨度高达 56.18 / 24.78 pp。这两件事同时成立, 只能是 seed 间方差被少数
  异常 run 撑爆。本脚本把每一格摊开, 使"装置失去测量能力"这句话有具体内容
  可写, 而不是一句断言。

  与 MAIN-01 (d_model=256) 的同格并列, 以便看出是哪些条件在加宽后改变了行为。

**本脚本的输出不得用于:**
  * 排除任何 run 或任何 seed (ADDENDUM_03 §9 / PREREG_CAP_01 §6)
  * 事后修改 C0-C3 的任何阈值 (CODE_DELTA §5.1)
  * 重跑本臂或更换 d_model 后只报告后一次 (PREREG_CAP_01 §5 第 5 条)
  * 论证 C0 "其实应该通过"
  判定已由 cap01_judge.py 完成并固定: C0 不通过, C1/C2/C3 不予解释。

统计口径复用 analyze_main624.py 的 _record_from_metadata (尾窗 epoch 80-100)。

用法:
  python3 cap01_diagnose.py
  python3 cap01_diagnose.py --no-baseline      # 不与 d256 并列
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
DATASETS = ("cifar10", "organamnist")
CHANCE = {"cifar10": 10.0, "organamnist": 100.0 / 11.0}   # 10 类 / 11 类

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def rule(t: str) -> None:
    say()
    say("=" * 96)
    say(t)
    say("=" * 96)


def load_mod(path: Path, d_model: int):
    name = f"analyze_main624_d{d_model}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses 在 @dataclass 处理期间会通过 cls.__module__ 回查 sys.modules,
    # 所以必须在 exec_module 之前注册, 否则 Python 3.10 会抛
    # AttributeError: 'NoneType' object has no attribute '__dict__'。
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod._D_MODEL = d_model
    return mod


def collect(mod, root: Path, ds: str) -> dict:
    out = {}
    for e in mod.EXP_IDS:
        for s in SEEDS:
            d = mod._expected_run_directory(root, ds, BACKBONE, AUGMENTATION, e, RELIANCE, s)
            mp = d / "metadata.json"
            if not mp.is_file():
                continue
            try:
                r = mod._record_from_metadata(mp)
            except ValueError as exc:
                say(f"  [读取失败] {d.name}: {exc}")
                continue
            out[(e, s)] = (r.tail_train_pp, r.tail_validation_pp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/outputs_cap512"))
    ap.add_argument("--baseline-root", type=Path, default=Path("/root/autodl-tmp/outputs_main"))
    ap.add_argument("--repo", type=Path, default=Path("/root/mamba-scan-study"))
    ap.add_argument("--report", type=Path, default=Path("/root/fig7_work/cap01_diagnose.txt"))
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    if args.report.resolve().is_relative_to(args.repo.resolve()):
        print("拒绝把报告写进仓库。", file=sys.stderr)
        return 2

    ana = args.repo / "mamba_scan_study" / "analysis" / "analyze_main624.py"
    m512 = load_mod(ana, 512)
    m256 = None if args.no_baseline else load_mod(ana, 256)

    say("CAP-01 逐格诊断  cap01_diagnose.py   (只读, 纯描述, 不判定)")
    say(f"  d512 产物根  {args.runs_root}")
    if m256:
        say(f"  d256 对照根  {args.baseline_root}")

    for ds in DATASETS:
        cur = collect(m512, args.runs_root, ds)
        base = collect(m256, args.baseline_root, ds) if m256 else {}

        rule(f"{ds}   逐条件 x 逐 seed 尾窗准确率 (train / val, pp)")
        say(f"  随机猜测基线 {CHANCE[ds]:.2f} pp")
        say()
        head = f"  {'condition':10s}"
        for s in SEEDS:
            head += f"{'seed'+str(s):>17s}"
        head += f"{'val mean':>10s}{'val sd':>8s}"
        if base:
            head += f"{'d256 val':>10s}{'delta':>8s}"
        say(head)
        say("  " + "-" * (len(head) - 2))

        for e in m512.EXP_IDS:
            vals = [cur[(e, s)][1] for s in SEEDS if (e, s) in cur]
            row = f"  {e:10s}"
            for s in SEEDS:
                if (e, s) in cur:
                    tr, va = cur[(e, s)]
                    row += f"{tr:>8.2f}/{va:<8.2f}"
                else:
                    row += f"{'--':>17s}"
            if vals:
                row += f"{st.mean(vals):>10.2f}"
                row += f"{(st.stdev(vals) if len(vals) > 1 else 0.0):>8.2f}"
                if base:
                    bv = [base[(e, s)][1] for s in SEEDS if (e, s) in base]
                    if bv:
                        row += f"{st.mean(bv):>10.2f}{st.mean(vals) - st.mean(bv):>+8.2f}"
            say(row)

        # 逼近随机猜测的格
        near = [(e, s, cur[(e, s)][0], cur[(e, s)][1]) for e in m512.EXP_IDS for s in SEEDS
                if (e, s) in cur and cur[(e, s)][1] < CHANCE[ds] * 3.0]
        say()
        say(f"  val < 3x 随机猜测 ({CHANCE[ds] * 3.0:.1f} pp) 的格: {len(near)}")
        for e, s, tr, va in near:
            kind = ("train 亦低 -> 未学到" if tr < CHANCE[ds] * 3.0
                    else "train 高 val 低 -> 泛化崩塌")
            say(f"    {e} seed{s}   train {tr:.2f}  val {va:.2f}   {kind}")

        # 组内 seed 离散度
        say()
        say("  各条件的 seed 间标准差, 由大到小 (前 6):")
        sds = sorted(((st.stdev([cur[(e, s)][1] for s in SEEDS if (e, s) in cur]), e)
                      for e in m512.EXP_IDS
                      if len([1 for s in SEEDS if (e, s) in cur]) > 1), reverse=True)
        for sd, e in sds[:6]:
            say(f"    {e:10s} {sd:8.2f} pp")

    rule("如何使用本报告")
    say("可以: 在论文中如实描述 d512 下发生了什么, 作为 C0 不通过的具体内容。")
    say("      任何关于成因的说法都是事后的, 只能出现在 Discussion")
    say("      (ADDENDUM_03 §2), 不得进摘要、结论或任何主张句。")
    say()
    say("不可以: 排除任何 run 或 seed; 事后调整 C0-C3 的阈值;")
    say("        论证 C0 '其实应该通过'; 更换 d_model 重跑并只报告后一次。")
    say("        判定已固定: C0 不通过, C1/C2/C3 如实报告但不予解释,")
    say("        且不得用于回应容量意见。")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(OUT) + "\n")
    print(f"\n报告已写出: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
