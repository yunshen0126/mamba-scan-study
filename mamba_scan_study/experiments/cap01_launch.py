#!/usr/bin/env python3
"""
cap01_launch.py -- CAP-01 容量稳健性臂发车器 (d_model=512)

设计矩阵:
  2 数据集 (cifar10, organamnist) x 13 路径条件 x R_high(grid=32) x 4 seed = 104 run

与 main624_launch.py 同构的保护:
  - preflight: HEAD 干净、产物根隔离、数据根可解析、d_model 参数存在
  - 每个 run 独立子进程, 失败不影响其他
  - 断点续跑: 目标目录已有完整 metadata 则跳过
  - 记录 run 台账 tsv (含 git_commit, 耗时, 退出码)

用法:
  python3 cap01_launch.py --dry-run           # 只列计划, 不执行
  python3 cap01_launch.py --preflight-only    # 只跑 preflight
  python3 cap01_launch.py --workers 3         # 实跑
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path("/root/mamba-scan-study")
RUN_ROOT = Path("/root/autodl-tmp/outputs_cap512")
LEDGER = RUN_ROOT / "cap01_runs.tsv"

DATASETS = ("cifar10", "organamnist")
DATA_ROOT = {
    "cifar10": "/root/autodl-tmp/datasets",
    "organamnist": "/root/autodl-tmp/datasets_new",
}
EXP_IDS = ("GEO_SG1", "GEO_SG2", "GEO_SG3", "GEO_SG4", "GEO_DIV",
           "RND_S1", "RND_S2", "RND_S3", "RND_D1", "RND_D2", "RND_D3",
           "LOC_S", "LOC_D")
SEEDS = (0, 1, 2, 3)
GRID = 32                 # -> R_high
D_MODEL = 512
AUGMENTATION = "main_uniform"
BACKBONE = "mamba"

FORBIDDEN_ROOTS = ("outputs_main", "outputs_aug16", "outputs_p0b_backup",
                   "outputs_probe_desat")
CST = timezone(timedelta(hours=8))


def now() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def run_dir(dataset: str, exp_id: str, seed: int) -> Path:
    # 与 run_p0b_feasibility.py:623 的 width_suffix 一致
    suffix = "" if D_MODEL == 256 else f"_d{D_MODEL}"
    return RUN_ROOT / (
        f"p0b_{dataset}_{AUGMENTATION}_{BACKBONE}_{exp_id}_R_high_seed{seed}{suffix}")


def is_complete(d: Path) -> bool:
    p = d / "metadata.json"
    if not p.is_file():
        return False
    try:
        m = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    h = m.get("validation_history")
    return (isinstance(h, list) and len(h) == 100
            and m.get("training_config", {}).get("d_model") == D_MODEL)


def preflight() -> bool:
    ok = True

    def chk(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'OK ' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    print("===== preflight =====")

    # 1. 产物根不得与受保护命名空间重叠
    resolved = RUN_ROOT.resolve()
    clash = [f for f in FORBIDDEN_ROOTS if f in resolved.parts]
    chk("产物根与受保护目录零交集", not clash, str(RUN_ROOT))

    # 2. git 状态
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        chk("HEAD 已解析", True, head)
        chk("工作区干净 (metadata 的 git_dirty 将为 False)", dirty == "",
            "有未提交改动:\n      " + dirty.replace("\n", "\n      ") if dirty else "")
    except subprocess.CalledProcessError as e:
        chk("git 可用", False, str(e)); head = "?"

    # 3. runner 支持所需参数
    try:
        h = subprocess.run([sys.executable, "-m",
                            "mamba_scan_study.experiments.run_p0b_feasibility", "--help"],
                           cwd=REPO_ROOT, capture_output=True, text=True, timeout=120).stdout
        for flag in ("--d-model", "--run-root", "--no-download"):
            chk(f"runner 支持 {flag}", flag in h)
    except Exception as e:
        chk("runner --help 可执行", False, str(e))

    # 4. 数据根可解析
    for ds, root in DATA_ROOT.items():
        chk(f"数据根存在 {ds}", Path(root).is_dir(), root)

    # 5. 路径条件数
    chk("路径条件数为 13", len(EXP_IDS) == 13, f"{len(EXP_IDS)}")
    chk("设计矩阵为 104", len(DATASETS) * len(EXP_IDS) * len(SEEDS) == 104)

    print(f"===== preflight {'通过' if ok else '未通过'} =====\n")
    return ok


def execute(job: tuple[str, str, int]) -> dict:
    dataset, exp_id, seed = job
    d = run_dir(dataset, exp_id, seed)
    if is_complete(d):
        return {"dataset": dataset, "exp_id": exp_id, "seed": seed,
                "status": "SKIP", "duration_s": 0, "exit_code": 0, "outdir": d.name}
    cmd = [sys.executable, "-m", "mamba_scan_study.experiments.run_p0b_feasibility",
           "--exp-id", exp_id, "--grid", str(GRID), "--training-seed", str(seed),
           "--dataset", dataset, "--augmentation", AUGMENTATION,
           "--backbone", BACKBONE, "--d-model", str(D_MODEL),
           "--data-root", DATA_ROOT[dataset], "--run-root", str(RUN_ROOT),
           "--no-download", "--execute"]
    t0 = time.time()
    log = RUN_ROOT / "logs" / f"{dataset}_{exp_id}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        r = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    dur = int(time.time() - t0)
    status = "COMPLETED" if (r.returncode == 0 and is_complete(d)) else "FAILED"
    print(f"  [{status}] {dataset} {exp_id} seed{seed}  {dur}s", flush=True)
    return {"dataset": dataset, "exp_id": exp_id, "seed": seed, "status": status,
            "duration_s": dur, "exit_code": r.returncode, "outdir": d.name}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="preflight 未过仍执行 (不建议)")
    args = ap.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    ok = preflight()
    if args.preflight_only:
        return 0 if ok else 1
    if not ok and not args.force:
        print("preflight 未通过, 已中止。确需执行请加 --force。")
        return 1

    jobs = [(ds, e, s) for ds in DATASETS for e in EXP_IDS for s in SEEDS]
    todo = [j for j in jobs if not is_complete(run_dir(*j))]
    print(f"计划 {len(jobs)} run, 已完成 {len(jobs)-len(todo)}, 待跑 {len(todo)}, "
          f"并行度 {args.workers}")

    if args.dry_run:
        for j in todo[:8]:
            print("  ", run_dir(*j).name)
        if len(todo) > 8:
            print(f"   ... 其余 {len(todo)-8} 个")
        return 0

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(execute, j): j for j in todo}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            done = sum(1 for r in rows if r["status"] in ("COMPLETED", "SKIP"))
            el = time.time() - t0
            eta = el / i * (len(todo) - i) if i else 0
            print(f"    进度 {i}/{len(todo)}  成功 {done}  "
                  f"已用 {el/3600:.1f}h  预计剩余 {eta/3600:.1f}h", flush=True)

    new = not LEDGER.exists()
    with LEDGER.open("a") as fh:
        if new:
            fh.write("ts\tdataset\texp_id\tseed\tstatus\tduration_s\texit_code"
                     "\tgit_commit\td_model\toutdir\n")
        for r in rows:
            fh.write(f"{now()}\t{r['dataset']}\t{r['exp_id']}\t{r['seed']}\t{r['status']}"
                     f"\t{r['duration_s']}\t{r['exit_code']}\t{head}\t{D_MODEL}"
                     f"\t{r['outdir']}\n")

    failed = [r for r in rows if r["status"] == "FAILED"]
    print(f"\n===== 完成 =====")
    print(f"  总用时 {(time.time()-t0)/3600:.1f} h")
    print(f"  成功 {len(rows)-len(failed)} / {len(rows)}")
    if failed:
        print("  失败:")
        for r in failed:
            print(f"    {r['dataset']} {r['exp_id']} seed{r['seed']} rc={r['exit_code']}")
        print("  重跑本脚本会自动跳过已完成的, 只补失败的。")
    print(f"  台账: {LEDGER}")
    print(f"\n分析: python3 mamba_scan_study/analysis/analyze_main624.py \\")
    print(f"        --runs-root {RUN_ROOT} --augmentation main_uniform \\")
    print(f"        --d-model {D_MODEL} --emit latex")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
