#!/usr/bin/env python3
"""
cap01_finalize.py -- CAP-01 收尾取证。

**只读。** 不写入任何 run 目录, 不写入 git 仓库, 不 commit, 不删除任何文件。
唯一的写动作是 --report 指定的报告文件, 默认落在 /root/fig7_work/ (仓库外)。

在 104 run 全部完成后跑一次, 输出 ledger 第十一条需要回填的全部事实:

  1. 完成度与完整性  -- 104 个目录是否齐全, 设计矩阵是否满格, 每个 run 是否
                        真的跑满 100 epoch (缺一即视为未完成, 不是"基本完成")
  2. provenance 均匀性 -- git_commit / git_dirty 是否全部一致。若不一致,
                        ledger (b) 的措辞必须改写, 因为"均匀的 True"这一
                        处置理由将不再成立
  3. 时间重建        -- 由目录 mtime 与 cap01.log 反推起止时刻
  4. 未跟踪文件清单   -- 连同 SHA-256, 作为清理前的存证
  5. 预注册文件 SHA   -- 核对 PREREG_CAP_01.md 的自指 SHA 与带外时间戳是否已回填

用法:
  python3 cap01_finalize.py
  python3 cap01_finalize.py --runs-root /root/autodl-tmp/outputs_cap512 \
                            --report /root/fig7_work/cap01_finalize_report.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

DATASETS = ("cifar10", "organamnist")
CONDITIONS = ("GEO_SG1", "GEO_SG2", "GEO_SG3", "GEO_SG4", "GEO_DIV",
              "RND_S1", "RND_S2", "RND_S3", "RND_D1", "RND_D2", "RND_D3",
              "LOC_S", "LOC_D")
SEEDS = (0, 1, 2, 3)
EXPECTED = len(DATASETS) * len(CONDITIONS) * len(SEEDS)      # 104
SUFFIX = "_d512"
EPOCHS = 100

OUT = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def h(title: str) -> None:
    say()
    say("=" * 74)
    say(title)
    say("=" * 74)


def sha256(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def git(repo: Path, *args) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return r.stdout.rstrip("\n")


def run_dir_name(ds: str, cond: str, seed: int) -> str:
    return f"p0b_{ds}_main_uniform_mamba_{cond}_R_high_seed{seed}{SUFFIX}"


# ------------------------------------------------------------------ 1


def check_completeness(root: Path) -> dict:
    h("1. 完成度与完整性")
    found, missing, short = {}, [], []
    for ds in DATASETS:
        for cond in CONDITIONS:
            for s in SEEDS:
                name = run_dir_name(ds, cond, s)
                mp = root / name / "metadata.json"
                if not mp.is_file():
                    missing.append(name)
                    continue
                try:
                    m = json.loads(mp.read_text())
                except Exception as e:                      # noqa: BLE001
                    missing.append(f"{name} (metadata 不可解析: {e})")
                    continue
                hist = m.get("validation_history")
                if not isinstance(hist, list) or len(hist) != EPOCHS:
                    short.append(f"{name} (history={len(hist) if isinstance(hist, list) else '缺失'})")
                    continue
                found[name] = (m, mp)

    stray = sorted(p.name for p in root.glob("p0b_*")
                   if p.is_dir() and p.name not in
                   {run_dir_name(d, c, s) for d in DATASETS
                    for c in CONDITIONS for s in SEEDS})

    say(f"设计矩阵      {len(DATASETS)} 数据集 x {len(CONDITIONS)} 条件 x "
        f"{len(SEEDS)} seed = {EXPECTED} run")
    say(f"完整可用      {len(found)} / {EXPECTED}")
    say(f"缺失          {len(missing)}")
    say(f"epoch 不足    {len(short)}")
    say(f"矩阵外目录    {len(stray)}")
    for lst, tag in ((missing, "缺失"), (short, "epoch 不足"), (stray, "矩阵外")):
        for n in lst:
            say(f"  [{tag}] {n}")

    if len(found) == EXPECTED and not missing and not short and not stray:
        say()
        say(">>> 本臂完成: 104/104, 零失败, 无矩阵外产物。")
    else:
        say()
        say(">>> 本臂**未**满足完成条件。ledger 与 PREREG_CAP_01 §10 不得写"
            "'全部完成'; 判定脚本不得在此状态下运行。")
    return found


# ------------------------------------------------------------------ 2


def check_provenance(found: dict) -> None:
    h("2. provenance 均匀性  (决定 ledger (b) 的措辞)")
    commits = Counter()
    dirty = Counter()
    for name, (m, _) in found.items():
        commits[str(m.get("git_commit", "缺失"))[:7]] += 1
        dirty[str(m.get("git_dirty", "缺失"))] += 1

    say("git_commit 分布:")
    for k, v in commits.most_common():
        say(f"  {k:>10s}  {v:>4d}")
    say("git_dirty 分布:")
    for k, v in dirty.most_common():
        say(f"  {k:>10s}  {v:>4d}")

    say()
    if len(commits) == 1 and len(dirty) == 1:
        c = next(iter(commits))
        d = next(iter(dirty))
        say(f">>> 均匀。全部 {len(found)} 个 run: git_commit={c}, git_dirty={d}")
        say(f">>> ledger 回填: <<<N_RUNS>>> = {len(found)}")
        say(f">>> ledger 回填: <<<UNIFORMITY>>> = "
            f"全部 {len(found)} 个 run 的 git_commit 与 git_dirty 完全一致 "
            f"({c} / {d})")
    else:
        say(">>> **不均匀**。ledger (b) 的处置理由 (第 3 条'均匀的 True 好过"
            "半 True 半 False') 不再成立, 该段必须改写为如实描述实际分布, "
            "并说明分界点与成因。不得沿用草稿措辞。")
        say(f">>> ledger 回填: <<<N_RUNS>>> = {len(found)}")
        say(">>> ledger 回填: <<<UNIFORMITY>>> = 见上方分布表, 不均匀")


# ------------------------------------------------------------------ 3


def check_timing(root: Path, found: dict, log: Path) -> None:
    h("3. 时间重建  (metadata 无时间字段, 只能反推)")
    tf = sorted({k for m, _ in found.values() for k in m
                 if any(t in k.lower() for t in
                        ("time", "start", "end", "stamp", "date", "elapsed",
                         "duration", "wall"))})
    say(f"metadata 中的时间类字段: {tf if tf else '无 (与 2026-08-04 的核实一致)'}")

    mtimes = sorted((Path(p).parent.stat().st_mtime, name)
                    for name, (_, p) in found.items())
    if not mtimes:
        say("无可用目录, 跳过。")
        return
    first_t, first_n = mtimes[0]
    last_t, last_n = mtimes[-1]
    fmt = "%Y-%m-%d %H:%M:%S"
    say(f"最早完成      {datetime.fromtimestamp(first_t).strftime(fmt)}  {first_n}")
    say(f"最晚完成      {datetime.fromtimestamp(last_t).strftime(fmt)}  {last_n}")

    wall = None
    if log.is_file():
        txt = log.read_text(errors="replace")
        m = re.search(r"\[COMPLETED\].*?\s(\d+)s", txt)
        if m:
            wall = int(m.group(1))
    if wall:
        start = datetime.fromtimestamp(first_t) - timedelta(seconds=wall)
        say(f"首个 run 墙钟 {wall}s")
        say(f">>> 重建起跑    {start.strftime(fmt)}   "
            f"(最早完成 - 首个 run 墙钟)")
        say(f">>> 重建总时长  {(datetime.fromtimestamp(last_t) - start)}")
        say(">>> 注: mtime 会被任何后续访问改写, 本值为下界重建, 非直接记录。")
    else:
        say("未能从 cap01.log 取到首个 run 的墙钟, 起跑时刻无法重建。")


# ------------------------------------------------------------------ 4


def check_worktree(repo: Path) -> None:
    h("4. 未跟踪文件清单  (清理前存证)")
    say(f"HEAD                {git(repo, 'rev-parse', '--short', 'HEAD')}")
    tracked = git(repo, "status", "--porcelain", "--untracked-files=no")
    say(f"已跟踪文件的改动    {'无' if not tracked else '有 —— 见下'}")
    if tracked:
        for line in tracked.splitlines():
            say(f"  {line}")
        say("  >>> 这与 2026-08-04 的核实不同 (当时为空)。须查清何时、由何产生。")

    full = git(repo, "status", "--porcelain")
    untracked = [ln[3:] for ln in full.splitlines() if ln.startswith("??")]
    say(f"未跟踪文件          {len(untracked)} 个")
    for rel in sorted(untracked):
        p = repo / rel
        if p.is_file():
            st = p.stat()
            say(f"  {sha256(p)}  {st.st_size:>9d}  "
                f"{datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}  {rel}")
        else:
            say(f"  {'(目录)':<64s}  {rel}")
    say()
    say(">>> 上表即 ledger (b) 所指的未跟踪文件存证。清理动作须在本表落盘之后。")


# ------------------------------------------------------------------ 5


def check_prereg(repo: Path) -> None:
    h("5. 预注册文件 SHA 与带外时间戳回填状态")
    for name in ("PREREG_CAP_01.md", "MAIN_PREREG_01.md",
                 "MAIN_PREREG_ADDENDUM_03_CONTINGENCY.md"):
        p = repo / name
        if not p.is_file():
            say(f"{name:44s}  **不存在**")
            continue
        say(f"{name:44s}  {sha256(p)}")

    cap = repo / "PREREG_CAP_01.md"
    if cap.is_file():
        tail = cap.read_text(errors="replace")[-600:]
        say()
        say("PREREG_CAP_01.md 末尾:")
        for line in tail.splitlines()[-6:]:
            say(f"  {line}")
        if "<冻结后回填>" in tail or "<冻结后提交" in tail:
            say()
            say(">>> **占位符未回填。** 带外时间戳是本件效力的支点 "
                "(PREREG_CAP_01 §9 末段)。在追加 §10 之前必须先查清 "
                "prereg-timestamps 里到底提交了哪个 SHA, 并回填。")
            say(">>> 若从未提交过, 那是比 ledger 第十一条严重得多的问题, "
                "须单独记一条勘误, 且本臂的'判据事前冻结'主张须相应削弱。")
        else:
            say()
            say(">>> 占位符已回填。追加 §10 后本文件 SHA 将改变: "
                "**保留原冻结 SHA 的记录, 新 SHA 另行标注为'追加结果后'**, "
                "不得用新 SHA 覆盖旧的 —— 覆盖会毁掉时间戳锚点。")


# ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/root/autodl-tmp/outputs_cap512"))
    ap.add_argument("--repo", type=Path, default=Path("/root/mamba-scan-study"))
    ap.add_argument("--log", type=Path, default=Path("/root/cap01.log"))
    ap.add_argument("--report", type=Path,
                    default=Path("/root/fig7_work/cap01_finalize_report.txt"))
    args = ap.parse_args()

    if not args.runs_root.is_dir():
        print(f"产物根不存在: {args.runs_root}", file=sys.stderr)
        return 2
    if args.report.resolve().is_relative_to(args.repo.resolve()):
        print(f"拒绝把报告写进仓库 ({args.report})，会弄脏工作树。", file=sys.stderr)
        return 2

    say(f"CAP-01 收尾取证   生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"产物根 {args.runs_root}")
    say(f"仓库   {args.repo}")

    found = check_completeness(args.runs_root)
    if found:
        check_provenance(found)
        check_timing(args.runs_root, found, args.log)
    check_worktree(args.repo)
    check_prereg(args.repo)

    h("回填清单")
    say("ledger_8k_draft.md 中的 <<< >>> 标记, 按第 2 节的 >>> 行填写。")
    say("其余数字已于 2026-08-04 核实闭合, 不需要改。")
    say("若第 1 节报告未完成、或第 2 节报告不均匀、或第 5 节报告占位符未回填,")
    say("**先停下来**, 不要继续走 §9 的落盘流程。")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(OUT) + "\n")
    print(f"\n报告已写出: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
