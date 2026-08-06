#!/usr/bin/env python3
"""
paper_lint.py -- 论文对 HANDOFF §6 九条不可违反约束的自动检查。

只读。不改任何文件。

**这是筛子不是判官。** 它抓的是词面模式，抓不到语义。零命中不等于合规，
命中也不等于违规——每条命中都要人读上下文判断。它的价值在于：改稿之后
跑一遍，确保没有把已经改对的东西改回去（第二轮审稿的头号意见就是
exact-zero 表述，全文改过一次，最怕回潮）。

用法:
  python3 paper_lint.py                      # 默认当前目录
  python3 paper_lint.py --root /path/to/paper
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (编号, 说明, 正则, 是否致命)
CHECKS = [
    ("§6.1", "exact-zero 表述回潮：区间含零 != 效应为零",
     r"(effect|gain|contrast|component|remainder)\s+(is|was|are|were)\s+(exactly\s+)?zero"
     r"|(is|was)\s+zero\b"
     r"|no\s+(measurable\s+)?(effect|gain)\s+(exists|is present)"
     r"|P_R\s*=\s*0", True),

    ("§6.2", "\"部分支持\"：命题 A 是整体判定，1/5 就是不成立",
     r"partial(ly)?\s+(support|confirm|hold)|partial\s+evidence|in part supports", True),

    ("§6.5", "GRU：不得写成\"两种骨干均确认\"",
     r"both\s+backbones?|across\s+(the\s+)?two\s+backbones?|confirmed\s+(in|on|by)\s+both", True),

    ("§6.6", "术语禁令：不得称多分支效应为 ensemble",
     r"\bensembl", True),

    ("§6.7", "L 族不得用于 locality 推断",
     r"(auxiliary|L[_ ]?family|P_L|LMTO)[^.]{0,120}\blocalit", True),

    ("§6.4", "事后解释只能在 Discussion（此项按小节位置另查，见下）",
     r"(?!x)x", False),

    ("A2", "术语统一：contrast ② 的第三个名字不得回潮",
     r"geometry-specific\s+(remainder|effect|component)", True),

    ("A4", "M3 机制化语言不得回潮",
     r"M3[^.]{0,80}(mechanis|mechanistic)|mechanis\w*\s+test", True),

    ("格式", "残留的 \\todo",
     r"\\todo", False),

    ("格式", "残留的占位符",
     r"<[^>]{0,40}待[^>]{0,40}>|TODO|FIXME|XXX|\bplaceholder\b", False),
]

# 事后解释的标志词；只允许出现在 Discussion / Limitations
POSTHOC = r"\b(we speculate|one explanation|plausibly|presumably|may be because|"       \
          r"a possible reason|this suggests that the|we conjecture)\b"
POSTHOC_ALLOWED = ("discussion.tex", "limitations.tex")


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


# 否定语境：命中的是禁令的否定句 (论文正确地声明了不那么说), 不是违规。
# 例如 "not that the effect is zero" / "we deliberately avoid the word ensemble"
# / "is not confirmation across two backbones"。
NEGATION = re.compile(
    r"\bnot\s+(a\s+|an\s+|the\s+)?(that|demonstration|confirmation|claim|"
    r"equivalence|evidence|proof)?"
    r"|\bis not\b|\bare not\b|\bwas not\b|\bcannot\b|\bnever\b"
    r"|\bavoid(s|ed)?\s+(the\s+)?word"
    r"|\bdeliberately avoid|\bwe avoid|\bdistinguishes .{0,40}from"
    r"|\brather than\b|\bnot\s+a\s+demonstration", re.I)


def in_negation(lines: list[str], idx: int) -> bool:
    """看命中行及其前后一行, 是否处于否定语境。"""
    window = " ".join(lines[max(0, idx - 1): idx + 2])
    return bool(NEGATION.search(window))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    files = sorted(list(args.root.glob("*.tex")) + list((args.root / "sections").glob("*.tex")))
    if not files:
        print(f"在 {args.root} 下找不到 .tex", file=sys.stderr)
        return 2

    total = 0
    for tag, desc, pattern, fatal in CHECKS:
        rx = re.compile(pattern, re.I)
        hits = []
        benign = []
        for f in files:
            body = strip_comments(f.read_text(encoding="utf-8", errors="replace"))
            lines = body.split("\n")
            for i, line in enumerate(lines):
                if rx.search(line):
                    rec = (f.name, i + 1, line.strip()[:150])
                    (benign if in_negation(lines, i) else hits).append(rec)
        mark = "!!" if (hits and fatal) else ("? " if hits else "OK")
        extra = f"  (另 {len(benign)} 处在否定语境, 已排除)" if benign else ""
        print(f"[{mark}] {tag:5s} {desc}   命中 {len(hits)}{extra}")
        for name, ln, txt in hits:
            print(f"        {name}:{ln}  {txt}")
        for name, ln, txt in benign:
            print(f"   [否定] {name}:{ln}  {txt}")
        total += len(hits) if fatal else 0

    # 事后解释的位置检查
    rx = re.compile(POSTHOC, re.I)
    bad = []
    for f in files:
        if f.name in POSTHOC_ALLOWED:
            continue
        body = strip_comments(f.read_text(encoding="utf-8", errors="replace"))
        for line_no, line in enumerate(body.split("\n"), 1):
            if rx.search(line):
                bad.append((f.name, line_no, line.strip()[:150]))
    print(f"[{'!!' if bad else 'OK'}] §6.4  事后解释出现在 Discussion/Limitations 之外   命中 {len(bad)}")
    for name, ln, txt in bad:
        print(f"        {name}:{ln}  {txt}")
    total += len(bad)

    # 摘要词数
    main_tex = args.root / "main.tex"
    if main_tex.is_file():
        t = main_tex.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S)
        if m:
            a = re.sub(r"\\[a-zA-Z]+", "", m.group(1))
            n = len([x for x in a.split() if re.search(r"[A-Za-z0-9]", x)])
            print(f"[{'!!' if n > 250 else 'OK'}] 摘要  词数 {n} / 250")
            if n > 250:
                total += 1

    print()
    print("零命中不等于合规。本工具只查词面模式，语义须人读。"
          if total == 0 else f">>> {total} 处致命项待人工判读。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
