#!/usr/bin/env python3
"""
make_supplementary_tables.py -- 从 seed-level metadata 生成补充材料的表格。

**只读。** 不需要 GPU，不需要 checkpoint，不需要云端。在本机跑。

为什么不能直接用 analyze_main624.py:
  它的 _completion_status 要求每个 run 目录同时有 metadata.json、completed.json
  和 final_checkpoint.pt。你下载的 seed_level_metadata_v2.tar.gz 只含前两者
  (checkpoint 共 1.1 GB, 未下载)。所以 analyze_main624.py 会把每个 run 判为
  failed, 输出空表。

  本脚本复用 analyze_main624.py 的统计口径 (_record_from_metadata / _summary /
  _format_display / contrast_summaries / interaction_summaries / ceiling_rows),
  只跳过 checkpoint 那一项完成性校验。统计一个字没改。

自检: 输出前先核对 CIFAR-10 高负载的 P_G / P_R / (2) 与论文 Table 6 是否逐位
一致, 不符即中止不出表。

输出: 五段 LaTeX, 直接替换 supplementary.tex 里的 [table: ...] 占位符。

用法 (Windows PowerShell 或 cmd 均可):
  python make_supplementary_tables.py ^
      --runs-root  D:\\path\\to\\outputs_main ^
      --cap01-root D:\\path\\to\\outputs_cap512 ^
      --analyze    D:\\path\\to\\analyze_main624.py ^
      --out        supplementary_tables.tex
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics as st
import sys
from pathlib import Path

BACKBONE = "mamba"
AUG = "main_uniform"
DATASETS = ("cifar10", "organamnist", "organcmnist", "organsmnist", "eurosat")
SEEDS = (0, 1, 2, 3)
RELIANCES = ("R_low", "R_high")

ROWS = (
    (r"\ding{172} structure", "1"),
    (r"\ding{174} polarity", "3"),
    (r"\ding{175} axis", "4"),
    (r"\ding{176} $P_G - P_L$", "5"),
    (r"\quad $P_L$", "P_LMTO"),
)

FROZEN = {"P_G": ("+4.31", "+3.45", "+5.18"),
          "P_R": ("-0.04", "-0.35", "+0.27"),
          "2":   ("+4.35", "+3.52", "+5.18")}


def load_mod(path: Path, d_model: int = 256):
    name = f"analyze_main624_d{d_model}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses 需要先注册
    spec.loader.exec_module(mod)
    mod._D_MODEL = d_model
    return mod


def cells(mod, root: Path, dataset: str, backbone: str, reliances=RELIANCES,
          suffix: str = "") -> dict:
    """按目录名直接读 metadata，跳过 checkpoint 校验。"""
    out, missing = {}, []
    for exp in mod.EXP_IDS:
        for rel in reliances:
            for s in SEEDS:
                d = root / (f"p0b_{dataset}_{AUG}_{backbone}_{exp}_{rel}_seed{s}{suffix}")
                p = d / "metadata.json"
                if not p.is_file():
                    missing.append(d.name); continue
                try:
                    out[(exp, rel, s)] = mod._record_from_metadata(p)
                except ValueError as e:
                    missing.append(f"{d.name} ({e})")
    if missing:
        print(f"[{dataset}/{backbone}] 缺 {len(missing)} 格:", file=sys.stderr)
        for x in missing[:8]:
            print("   ", x, file=sys.stderr)
        raise SystemExit("设计矩阵不完整，未出表。检查 --runs-root 是否指向解包后的 outputs_main。")
    return out


def fmt(mod, s) -> str:
    f = mod._format_display
    return (f"${f(s.mean_pp, signed=True)}$ "
            f"$[{f(s.lower_pp, signed=True)}, {f(s.upper_pp, signed=True)}]$")


def selftest(mod, c) -> None:
    su = mod.contrast_summaries(c)["R_high"]
    bad = []
    for k, exp in FROZEN.items():
        s = su[k]
        got = tuple(mod._format_display(v, signed=True)
                    for v in (s.mean_pp, s.lower_pp, s.upper_pp))
        if got != exp:
            bad.append(f"cifar10 R_high {k}: 实得 {got} != Table 6 {exp}")
    if bad:
        for b in bad:
            print("  [失配]", b, file=sys.stderr)
        raise SystemExit("与论文 Table 6 不一致，未出表。")
    print("自检通过: CIFAR-10 高负载的 P_G / P_R / (2) 与论文 Table 6 逐位一致。")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--cap01-root", type=Path, default=None)
    ap.add_argument("--analyze", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("supplementary_tables.tex"))
    a = ap.parse_args()

    mod = load_mod(a.analyze)
    L = []

    # ---- S1: 探索性对比, 五数据集 x 两负载 ----
    all_c = {ds: cells(mod, a.runs_root, ds, BACKBONE) for ds in DATASETS}
    selftest(mod, all_c["cifar10"])

    L += [r"% ==== supplementary section 1: exploratory contrasts ====",
          r"\begin{center}\small", r"\begin{tabular}{llccc}", r"\toprule",
          r"dataset & quantity & low load & high load & paired difference \\",
          r"\midrule"]
    for ds in DATASETS:
        su = mod.contrast_summaries(all_c[ds])
        inter = mod.interaction_summaries(su)
        for lab, key in ROWS:
            L.append(f"{ds} & {lab} & {fmt(mod, su['R_low'][key])} & "
                     f"{fmt(mod, su['R_high'][key])} & {fmt(mod, inter[key])} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{center}", ""]

    # ---- S2: GRU 臂 ----
    try:
        g = cells(mod, a.runs_root, "cifar10", "gru")
        su = mod.contrast_summaries(g); inter = mod.interaction_summaries(su)
        L += [r"% ==== supplementary section 2: recurrent backbone ====",
              r"\begin{center}\small", r"\begin{tabular}{lccc}", r"\toprule",
              r"quantity & low load & high load & paired difference \\", r"\midrule"]
        for lab, key in ((r"$P_G$", "P_G"), (r"$P_R$", "P_R"),
                         (r"\ding{173} $P_G-P_R$", "2")) + ROWS:
            L.append(f"{lab} & {fmt(mod, su['R_low'][key])} & "
                     f"{fmt(mod, su['R_high'][key])} & {fmt(mod, inter[key])} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    except SystemExit:
        print("GRU 臂未找到，跳过第 2 节。", file=sys.stderr)

    # ---- S3: 天花板诊断 ----
    L += [r"% ==== supplementary: ceiling diagnostics ====",
          r"\begin{center}\small", r"\begin{tabular}{lcccc}", r"\toprule",
          r"dataset & low load & flagged & high load & flagged \\", r"\midrule"]
    for ds in DATASETS:
        rows = mod.ceiling_rows(all_c[ds])
        lo, hi = rows["R_low"], rows["R_high"]
        L.append(f"{ds} & {mod._format_display(lo[0])} & {'yes' if lo[1] else 'no'} & "
                 f"{mod._format_display(hi[0])} & {'yes' if hi[1] else 'no'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]

    # ---- S4: 容量臂逐条件 ----
    if a.cap01_root and a.cap01_root.is_dir():
        L += [r"% ==== supplementary: capacity arm, per condition ====",
              r"\begin{center}\small", r"\begin{tabular}{llcc}", r"\toprule",
              r"dataset & condition & tail train (\%) & tail val (\%) \\", r"\midrule"]
        for ds in ("cifar10", "organamnist"):
            c = cells(mod, a.cap01_root, ds, BACKBONE,
                      reliances=("R_high",), suffix="_d512")
            for exp in mod.EXP_IDS:
                tr = st.mean(c[(exp, "R_high", s)].tail_train_pp for s in SEEDS)
                va = st.mean(c[(exp, "R_high", s)].tail_validation_pp for s in SEEDS)
                L.append(f"{ds} & \\texttt{{{exp.replace('_', chr(92)+'_')}}} & "
                         f"{mod._format_display(tr)} & {mod._format_display(va)} \\\\")
            L.append(r"\midrule")
        L[-1] = r"\bottomrule"
        L += [r"\end{tabular}", r"\end{center}", ""]

    a.out.write_text("\n".join(L), encoding="utf-8")
    print(f"已写出 {a.out}  ({len(L)} 行)")
    print("把其中各段贴到 supplementary.tex 对应的 [table: ...] 占位符处。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
