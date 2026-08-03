#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main624_launch.py — MAIN-01 主实验发车编排器

依据: MAIN_PREREG_01.md §2 (设计矩阵) / §12.2 (发车协议)
      P0B_PREREG_ADDENDUM_01.md §5 (冻结产物完整性)
      docs/03_EVIDENCE_LEDGER.md §8e.1 (git_dirty 缺陷) / §8e.8 (成本实测)

设计原则: 全部 preflight 断言 fail-closed。任何一条不过则不发车,不给 override。
          脚本自身不做任何 git 写操作。跑批期间不 commit 由 preflight 冻结 HEAD 保证。

用法:
  python -u main624_launch.py --preflight-only --expected-head <40位SHA>
  python -u main624_launch.py --dry-run       --expected-head <40位SHA>
  python -u main624_launch.py                 --expected-head <40位SHA>
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ============================================================================
# 【需要你核对/修改的唯一区块】runner 接口
# 发车前必须: python main624_launch.py --dry-run  然后把输出的目录与命令
#             跟一次真实 run 的实际产物逐字对比。对不上就改这里,别改别处。
# ============================================================================

RUNNER_MODULE = "mamba_scan_study.experiments.run_p0b_feasibility"
PYTHON_BIN = sys.executable

# runner 没有 --output-dir: 目录由 runner 自己算,发车脚本只能"预测"。
# 因此 outdir_for() 必须与 run_p0b_feasibility.py:553 的拼接逐字一致。
# 训练超参全部硬编码在 FORMAL_CONFIG 内,不经 CLI 传 —— 不要再往 FIXED_ARGS 塞超参。
FIXED_ARGS = [
    "--augmentation", "main_uniform",   # MAIN_PREREG_01 §5.1 五数据集统一禁 HFlip
    "--mode", "formal",
    "--execute",                        # 不加则 runner 只做 dry-run,不训练
]

BACKBONE_FLAG = "--backbone"            # 待 C2 实现;实现后必须与此一致
RELIANCE_GRID = {"R_low": 8, "R_high": 32}   # patch 由 grid 推出,无独立 flag

# --data-root 默认是相对路径 "data",一个参数覆盖不了云端两个根 —— 按数据集分别传
DATA_ROOT = {
    "cifar10":     "/root/autodl-tmp/datasets",
    "organamnist": "/root/autodl-tmp/datasets_new",
    "organcmnist": "/root/autodl-tmp/datasets_new",
    "organsmnist": "/root/autodl-tmp/datasets_new",
    "eurosat":     "/root/autodl-tmp/datasets_new",
}
# runner 解析出的实际数据路径,preflight 用来断言存在
DATA_PROBE = {
    "cifar10":     "cifar-10-batches-py",
    "organamnist": "organamnist.npz",
    "organcmnist": "organcmnist.npz",
    "organsmnist": "organsmnist.npz",
    "eurosat":     "eurosat/2750",
}
# MAIN_PREREG_01 §4.1 写死的 train/val 样本数。Organ 三个的划分来源不受 SHA 门
# 保护(用 MedMNIST 官方划分),样本数断言是唯一能挡住"数据集悄悄换版本"的东西。
EXPECTED_SPLIT_COUNTS = {
    "cifar10":     (45000, 5000),
    "organamnist": (34561, 6491),
    "organcmnist": (12975, 2392),
    "organsmnist": (13932, 2452),
    "eurosat":     (22000, 2500),
}

# P0-B legacy 命名(104 个已完成 run),主实验绝不能产生这个形状
LEGACY_RE = r"^p0b_(GEO_SG[1-4]|GEO_DIV|RND_[SD][1-3]|LOC_[SD])_R_(low|high)_seed[0-3]$"


def outdir_for(r):
    """镜像 run_p0b_feasibility.py:553。主实验 augmentation 恒为 main_uniform,
    故一律走 else 分支。backbone 段依 C2 加入(mamba 也带,保持对称)。"""
    aug = "main_uniform"
    return (f"outputs/p0b_{r['dataset']}_{aug}_{r['backbone']}"
            f"_{r['exp_id']}_{r['reliance']}_seed{r['seed']}")


def build_cmd(r, outdir):
    return [
        PYTHON_BIN, "-u", "-m", RUNNER_MODULE,
        "--dataset", r["dataset"],
        "--data-root", DATA_ROOT[r["dataset"]],
        BACKBONE_FLAG, r["backbone"],
        "--exp-id", r["exp_id"],
        "--grid", str(RELIANCE_GRID[r["reliance"]]),
        "--training-seed", str(r["seed"]),
    ] + FIXED_ARGS


# ============================================================================
# 冻结产物 (MAIN_PREREG_01 §3, 六项)
# ============================================================================

FROZEN_SHAS = {
    "docs/P0B_CONFIG_TABLE.md":
        "790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e",
    "P0B_RUN_LEDGER_104.csv":
        "906f6af2f8a695b443b01ac9ff89e29f24b4cea85fb4717252404f58145bfe25",
    "P0B_L_PATH_BANK_FROZEN.json":
        "93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7",
    "P0B_R_PATH_BANK_FROZEN.json":
        "2f7b8a6fd3cfbbae9897b4ef4dc9dcfd1bf7744619d5818ceaca7604d565aee3",
    "P0B_CIFAR10_VAL_SPLIT_FROZEN.json":
        "e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09",
    "P0B_EUROSAT_SPLIT_FROZEN.json":
        "f5ddb2db3f8ffc74efb77295e0fac17d34df85179bcd78de3f4e638b685c4117",
}

# runner 的 SHA 门只比对这四项 (P0B_PREREG_ANALYSIS_PLAN §7)
RUNNER_GATED = [
    "docs/P0B_CONFIG_TABLE.md",
    "P0B_L_PATH_BANK_FROZEN.json",
    "P0B_R_PATH_BANK_FROZEN.json",
    "P0B_CIFAR10_VAL_SPLIT_FROZEN.json",
]

# metadata.json 的真实 SHA 字段名 (C1 实测)。全数据集通用的四项。
METADATA_SHA_FIELDS = {
    "config_source_sha256": "docs/P0B_CONFIG_TABLE.md",
    "ledger_sha256":        "P0B_RUN_LEDGER_104.csv",
    "lmto_source_sha256":   "P0B_L_PATH_BANK_FROZEN.json",
    "random_source_sha256": "P0B_R_PATH_BANK_FROZEN.json",
}

# 自指 SHA 的预注册文件: (路径, 对应 commit, 该 commit 下的内容 SHA)
# 工作区版本含回填行,直接 sha256sum 必然不匹配 —— 只能用 git show 校验
SELF_REF_PREREG = [
    ("MAIN_PREREG_01.md", "68dff0b",
     "1ccd6245d2583e563086e09379b1e20944110a109c7374d7b8576b62a66cb7ff"),
    ("P0B_PREREG_ANALYSIS_PLAN.md", "a63d9e9",
     "841d103a1566040a505a22f514f44db816fbd876ed4910cf54104852278b5d15"),
]

# ============================================================================
# 设计矩阵 (MAIN_PREREG_01 §2)
# ============================================================================

DATASETS = ["cifar10", "organamnist", "organcmnist", "organsmnist", "eurosat"]
EXP_IDS = ["GEO_SG1", "GEO_SG2", "GEO_SG3", "GEO_SG4", "GEO_DIV",
           "RND_S1", "RND_S2", "RND_S3", "RND_D1", "RND_D2", "RND_D3",
           "LOC_S", "LOC_D"]
RELIANCES = ["R_low", "R_high"]
SEEDS = [0, 1, 2, 3]

# canary: 与 P0-B 一致 (GEO_SG1 / R_low / seed0)。R_low 最便宜,且已覆盖
# 新数据集的全部新增风险面 (resize / 灰度 repeat / normalize 常数)。
CANARY_KEY = ("GEO_SG1", "R_low", 0)

# (dataset, backbone) 为 canary 单位 —— 6 组,不是 5 组。
# MAIN_PREREG_01 §12.2 写"每个数据集",但 cifar10-GRU 是独立代码路径。加严不变更。
GROUPS = [(d, "mamba") for d in DATASETS] + [("cifar10", "gru")]

EXPECTED_TOTAL = 624

# 成本模型 (ledger §8e.8 实测): grid8 1690s / grid32 6995s, 按 train 集规模折算
TRAIN_SIZES = {"cifar10": 45000, "organamnist": 34561, "organcmnist": 12975,
               "organsmnist": 13932, "eurosat": 22000}

# GRU 相对 Mamba 的耗时系数。0.70 是 MAIN_PREREG_01 §12.1 的 ~88 h 隐含值,
# 但【无实测支撑】: P0-B 是 Mamba-only,批次 C 的 GRU 用的是 real_4dir 实现,
# 没有 explicit permutation 的 index_select 开销。预算按 1.0 留余量更稳。
GRU_TIME_FACTOR = 0.70

TSV_COLUMNS = [
    "run_uid", "group", "dataset", "backbone", "exp_id", "reliance", "seed",
    "is_canary", "outdir", "status", "start_iso", "end_iso", "duration_s",
    "exit_code", "host", "git_commit", "git_dirty", "launcher_sha", "log_path",
]


# ============================================================================
# 工具
# ============================================================================

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo)] + list(args),
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


# ============================================================================
# Preflight —— 全部 fail-closed
# ============================================================================

def preflight(args, plan):
    repo = Path(args.repo).resolve()
    ok = []

    def p(msg):
        ok.append(msg)
        print(f"  [PASS] {msg}")

    print("== PREFLIGHT ==")

    # --- 1. 仓库位置 ---
    rc, top, _ = run_git(repo, "rev-parse", "--show-toplevel")
    check(rc == 0, "不是 git 仓库: %s" % repo)
    check(Path(top).resolve() == repo, f"--repo 不是仓库根: {repo} vs {top}")
    p(f"仓库根 = {repo}")

    # --- 2. 工作区干净 (MAIN_PREREG_01 §12.2 第 2 条) ---
    rc, out, _ = run_git(repo, "status", "--porcelain")
    check(rc == 0 and out == "",
          "git status --short 非空,发车前必须清空:\n" + out)
    p("git status --short 为空")

    # --- 3. HEAD 锁定 ---
    rc, head, _ = run_git(repo, "rev-parse", "HEAD")
    check(rc == 0, "无法读取 HEAD")
    check(head == args.expected_head,
          f"HEAD 不匹配: 实际 {head} != 预期 {args.expected_head}")
    p(f"HEAD = {head[:7]} (与 --expected-head 一致)")

    # --- 4. 已 push (ADDENDUM §5.2「未 push 不算冻结」) ---
    if not args.skip_fetch:
        rc, _, err = run_git(repo, "fetch", args.remote, "--quiet")
        check(rc == 0, f"git fetch {args.remote} 失败 (SSH over 443?): {err}")
    rc, remote_head, _ = run_git(repo, "rev-parse", f"{args.remote}/{args.branch}")
    check(rc == 0, f"无法读取 {args.remote}/{args.branch}")
    check(remote_head == head,
          "HEAD 未 push。P0B_PREREG_ADDENDUM_01.md §5.2: 文件只有在已 track 且已 push\n"
          "  到 origin 之后其 SHA 才构成有效冻结。当前预注册尚未生效,不得发车。\n"
          f"  HEAD={head[:7]}  {args.remote}/{args.branch}={remote_head[:7]}")
    p(f"{args.remote}/{args.branch} == HEAD (预注册已生效)")

    # --- 5. 六个冻结产物的工作区 SHA ---
    for rel, want in FROZEN_SHAS.items():
        f = repo / rel
        check(f.exists(), f"冻结产物缺失: {rel}")
        got = sha256_file(f)
        check(got == want, f"SHA 不匹配 {rel}\n    实际 {got}\n    预期 {want}")
    p(f"六个冻结产物 SHA-256 全部匹配")

    # --- 6. .gitattributes 的 -text 条目 (ADDENDUM §5.3 行尾符不变量) ---
    ga = repo / ".gitattributes"
    check(ga.exists(), ".gitattributes 不存在")
    ga_txt = ga.read_text(encoding="utf-8", errors="replace")
    missing = [rel for rel in FROZEN_SHAS
               if not any(Path(rel).name in ln and "-text" in ln
                          for ln in ga_txt.splitlines())]
    check(not missing,
          "以下冻结产物在 .gitattributes 中没有 -text 条目,跨平台 checkout 后 SHA\n"
          "  必然不匹配且与篡改不可区分 (ADDENDUM §5.3): " + ", ".join(missing))
    p(".gitattributes 覆盖全部冻结产物 (-text)")

    # --- 7. 自指 SHA 的预注册文件 (只能用 git show 校验) ---
    for rel, commit, want in SELF_REF_PREREG:
        rc, blob, err = run_git(repo, "show", f"{commit}:{rel}")
        check(rc == 0, f"git show {commit}:{rel} 失败: {err}")
        got = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        if got != want:
            r2 = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"],
                                capture_output=True)
            got = hashlib.sha256(r2.stdout).hexdigest()
        check(got == want,
              f"预注册自指 SHA 不匹配 {rel}@{commit}\n    实际 {got}\n    预期 {want}")
    p("预注册文件自指 SHA 匹配 (MAIN_PREREG_01 @ 68dff0b)")

    # --- 8. 624 个产物目录必须互异 ---
    dirs = [r["outdir"] for r in plan]
    check(len(plan) == EXPECTED_TOTAL,
          f"计划 run 数 {len(plan)} != {EXPECTED_TOTAL}")
    dup = len(dirs) - len(set(dirs))
    check(dup == 0,
          f"产物目录碰撞 {dup} 处 —— outdir_for() 未覆盖 dataset/backbone 维度。\n"
          "  若沿用 P0-B 的 outputs/p0b_{exp_id}_{reliance}_seed{S}/,624 个 run\n"
          "  会写进 104 个目录并互相覆盖,且续跑逻辑会静默 SKIP 掉后 520 个。")
    import re
    clash = [d for d in dirs if re.match(LEGACY_RE, Path(d).name)]
    check(not clash,
          "以下目录命中 P0-B legacy 形状,会覆盖已完成的 104 个 run "
          "(通常是 --augmentation main_uniform 漏传): " + ", ".join(clash[:5]))
    p(f"{len(dirs)} 个产物目录互异,且不落入 P0-B legacy 命名空间")

    # --- 8b. 已完成的 104 个 P0-B run 必须完好 ---
    legacy = sorted((repo / "outputs").glob("p0b_*_R_*_seed?")) if (repo / "outputs").exists() else []
    legacy = [d for d in legacy if re.match(LEGACY_RE, d.name)]
    if legacy:
        check(len(legacy) == 104, f"P0-B legacy 目录 {len(legacy)} 个,应为 104")
        bad = [d.name for d in legacy
               if not (d / "metadata.json").exists()
               or not str((json.loads((d / "metadata.json").read_text(encoding="utf-8"))
                           ).get("git_commit", "")).startswith("34edddb")]
        check(not bad, f"P0-B legacy run 已受损或 git_commit 非 34edddb: {bad[:5]}")
        p("104 个 P0-B legacy run 完好 (git_commit=34edddb)")
    else:
        p("未发现 P0-B legacy 目录 (云端首次 checkout 时正常)")

    # --- 9. outputs 被 git 忽略 (ledger §8e.1 的 git_dirty 缺陷) ---
    rc, _, _ = run_git(repo, "check-ignore", "-q", "outputs")
    check(rc == 0,
          "outputs/ 未被 .gitignore 忽略。跑批期间产物目录会让每个 run 记成\n"
          "  git_dirty=true (ledger §8e.1 已发生过一次),而跑批期间禁止 commit,无法补救。")
    p("outputs/ 已被 git 忽略")

    # --- 10. 日志目录在仓库外 ---
    ld = Path(args.launch_dir).resolve()
    check(repo not in ld.parents and ld != repo,
          f"--launch-dir 在仓库内 ({ld})。TSV/日志会污染工作区 → git_dirty。")
    ld.mkdir(parents=True, exist_ok=True)
    p(f"发车目录在仓库外: {ld}")

    # --- 11. 环境同质性 (ledger §10.1: num_workers 差异即可移动 0.33pp) ---
    import platform
    env = {"python": platform.python_version(), "host": socket.gethostname()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        raise Fail(f"import torch 失败: {e}")
    try:
        import numpy, torchvision
        env["numpy"] = numpy.__version__
        env["torchvision"] = torchvision.__version__
    except Exception as e:
        raise Fail(f"import numpy/torchvision 失败: {e}")
    check(env["gpu"] is not None, "CUDA 不可用")
    if args.expect_torch:
        check(env["torch"] == args.expect_torch,
              f"torch 版本 {env['torch']} != 预期 {args.expect_torch}。\n"
              "  全批必须在同一环境完成 (ledger §10.1)。")
    p(f"环境: torch {env['torch']} / {env['gpu']}")

    # --- 12. 磁盘 ---
    outroot = (repo / "outputs")
    probe = outroot if outroot.exists() else repo
    free_gb = shutil.disk_usage(probe).free / 2**30
    check(free_gb >= args.min_free_gb,
          f"产物盘剩余 {free_gb:.1f} GB < 要求 {args.min_free_gb} GB")
    p(f"产物盘剩余 {free_gb:.1f} GB"
      + ("  (提示: outputs 建议 symlink 到 /root/autodl-tmp)" if not outroot.is_symlink() else ""))

    # --- 12b. 五个数据集在各自 data-root 下可解析 ---
    missing_ds = []
    for ds, root in DATA_ROOT.items():
        probe = Path(root) / DATA_PROBE[ds]
        if not probe.exists():
            missing_ds.append(f"{ds} -> {probe}")
    check(not missing_ds,
          "以下数据集在 --data-root 下找不到:\n    " + "\n    ".join(missing_ds))
    p("五个数据集在各自 data-root 下均可解析")

    print(f"== PREFLIGHT 全部通过 ({len(ok)} 项) ==\n")
    return {"env": env, "head": head, "repo": str(repo)}


# ============================================================================
# 计划构造
# ============================================================================

def est_seconds(dataset, reliance, backbone="mamba"):
    base = 1690.0 if reliance == "R_low" else 6995.0
    f = GRU_TIME_FACTOR if backbone == "gru" else 1.0
    return base * f * TRAIN_SIZES[dataset] / TRAIN_SIZES["cifar10"]


def build_plan():
    plan = []
    for gi, (ds, bk) in enumerate(GROUPS):
        gname = f"{ds}_{bk}"
        runs = []
        for eid in EXP_IDS:
            for rel in RELIANCES:
                for s in SEEDS:
                    r = {"group": gname, "group_index": gi, "dataset": ds,
                         "backbone": bk, "exp_id": eid, "reliance": rel, "seed": s}
                    r["is_canary"] = (eid, rel, s) == CANARY_KEY
                    r["est_s"] = est_seconds(ds, rel, bk)
                    r["outdir"] = outdir_for(r)
                    r["run_uid"] = f"{gname}_{eid}_{rel}_seed{s}"
                    runs.append(r)
        canary = [r for r in runs if r["is_canary"]]
        rest = [r for r in runs if not r["is_canary"]]
        assert len(canary) == 1, f"{gname}: canary 不唯一"
        assert len(runs) == 104, f"{gname}: {len(runs)} != 104"
        # 组内 LPT: 长作业优先,减少尾部空转
        rest.sort(key=lambda r: (-r["est_s"], r["exp_id"], r["seed"]))
        plan.extend(canary + rest)
    return plan


# ============================================================================
# 完成判定 / 隔离
# ============================================================================

def load_metadata(repo, outdir):
    f = Path(repo) / outdir / "metadata.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_history(repo, outdir, md):
    if isinstance(md, dict):
        for k in ("validation_history", "history"):
            if isinstance(md.get(k), list):
                return md[k]
    for name in ("validation_history.json", "history.json"):
        f = Path(repo) / outdir / name
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def completion_status(repo, outdir, r, head):
    """返回 (是否完成, 原因)。fail-closed: 任何一项不确定都算未完成。"""
    d = Path(repo) / outdir
    if not d.exists():
        return False, "absent"
    md = load_metadata(repo, outdir)
    if md is None:
        return False, "no_metadata"
    hist = get_history(repo, outdir, md)
    if not isinstance(hist, list) or len(hist) != 100:
        return False, f"history_len={0 if hist is None else len(hist)}"
    epochs = [h.get("epoch") for h in hist]
    if epochs != list(range(1, 101)):
        return False, "epochs_not_1_100"
    if any(("test" in str(k).lower()) for h in hist for k in h.keys()):
        return False, "test_field_present"
    gc = str(md.get("git_commit", ""))
    if not gc or not head.startswith(gc[:7]):
        return False, f"git_commit={gc[:7]}"
    for key, want in (("dataset", r["dataset"]), ("exp_id", r["exp_id"]),
                      ("training_seed", r["seed"])):
        if key in md and str(md[key]) != str(want):
            return False, f"{key}_mismatch"
    # metadata 的真实字段名(C1 实测),不存在 source_shas/frozen_shas。
    # 缺字段一律判未完成 —— 不得 fail-open。
    for field, rel in METADATA_SHA_FIELDS.items():
        got = md.get(field)
        if not got:
            return False, f"missing:{field}"
        if got != FROZEN_SHAS[rel]:
            return False, f"sha_mismatch:{field}"
    # split_source_sha256 对全部数据集都写四源门里的 CIFAR-10 划分 SHA(runner 现状,
    # MAIN_PREREG_01 §3 要求四源门不变,故不动它)。Organ 三个的真实划分来源因此
    # 不受 SHA 保护 —— 改用 §4.1 写死的样本数作为断言。
    # 若最终决定不实现 data_split_provenance,删掉下面这块即可。
    prov = md.get("data_split_provenance")
    if not isinstance(prov, dict):
        return False, "missing:data_split_provenance"
    want_tr, want_va = EXPECTED_SPLIT_COUNTS[r["dataset"]]
    if (prov.get("train_n"), prov.get("val_n")) != (want_tr, want_va):
        return False, (f"split_counts={prov.get('train_n')}/{prov.get('val_n')}"
                       f" != {want_tr}/{want_va}")
    if md.get("augmentation") != "main_uniform":
        return False, f"augmentation={md.get('augmentation')}"
    if "backbone" in md and str(md["backbone"]) != r["backbone"]:
        return False, "backbone_mismatch"
    return True, "ok"


def quarantine(repo, outdir, launch_dir, reason):
    src = Path(repo) / outdir
    if not src.exists():
        return None
    q = Path(launch_dir) / "quarantine" / f"{time.strftime('%Y%m%d_%H%M%S')}_{Path(outdir).name}_{reason}"
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(q))
    return str(q)


def verify_canary(repo, outdir, r, head, strict_dirty):
    md = load_metadata(repo, outdir)
    check(md is not None, f"canary 无 metadata.json: {outdir}")
    tc = md.get("training_config", md)
    for key, want in (("epochs", 100), ("num_workers", 4)):
        if key in tc:
            check(int(tc[key]) == want, f"canary {key}={tc[key]} != {want}")
    for key, want in (("micro_batch", 128), ("accum_steps", 1)):
        if key in md:
            check(int(md[key]) == want, f"canary {key}={md[key]} != {want}")
    if "augmentation" in md:
        check(md["augmentation"] == "main_uniform",
              f"canary augmentation={md['augmentation']} != main_uniform (§5.1)")
    if "backbone" in md:
        check(str(md["backbone"]) == r["backbone"], "canary backbone 不匹配")
    for field, rel in METADATA_SHA_FIELDS.items():
        check(md.get(field) == FROZEN_SHAS[rel],
              f"canary {field} = {md.get(field)}\n    应为 {FROZEN_SHAS[rel]} ({rel})")
    if strict_dirty and "git_dirty" in md:
        check(md["git_dirty"] in (False, "false", 0),
              "canary git_dirty=true。ledger §8e.1 的缺陷复现: 工作区有未跟踪文件,\n"
              "  而跑批期间禁止 commit,后续 623 个 run 的溯源记录会全部带此缺陷。")
    ok, why = completion_status(repo, outdir, r, head)
    check(ok, f"canary 完成度校验失败: {why}")
    return md


# ============================================================================
# 主循环
# ============================================================================

class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        new = not self.path.exists()
        self.f = open(self.path, "a", encoding="utf-8", newline="")
        if new:
            self.f.write("\t".join(TSV_COLUMNS) + "\n")
            self._flush()

    def _flush(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def write(self, row):
        self.f.write("\t".join(str(row.get(c, "")) for c in TSV_COLUMNS) + "\n")
        self._flush()


def free_vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/mamba-scan-study")
    ap.add_argument("--launch-dir", default="/root/autodl-tmp/main_launch")
    ap.add_argument("--expected-head", required=True)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--min-free-mib", type=int, default=4500)
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--expect-torch", default="")
    ap.add_argument("--max-consecutive-failures", type=int, default=3)
    ap.add_argument("--max-group-failures", type=int, default=5)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-strict-canary-dirty", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    launch_dir = Path(args.launch_dir).resolve()
    plan = build_plan()

    if args.dry_run:
        print(f"== DRY RUN: {len(plan)} runs ==")
        tot = sum(r["est_s"] for r in plan) / 3600
        gru_h = sum(r["est_s"] for r in plan if r["backbone"] == "gru") / 3600
        tot_c = tot - gru_h + gru_h / GRU_TIME_FACTOR   # GRU 系数取 1.0 的保守上界
        print(f"预估 {tot:.0f} process-h (GRU 系数 {GRU_TIME_FACTOR}, 与预注册 §12.1 一致); "
              f"保守上界 {tot_c:.0f} process-h")
        print(f"{args.jobs} 并行按 P0-B 实测加速比 4.64x (125.4 process-h / 27 h 墙钟, "
              f"非 5.0x):")
        print(f"  墙钟 {tot/4.64:.0f}–{tot_c/4.64:.0f} h + 6 次 canary 串行 ≈ +3 h "
              f"→ {tot/4.64/24+0.13:.1f}–{tot_c/4.64/24+0.13:.1f} 天\n")
        seen = {}
        for r in plan:
            seen.setdefault(r["outdir"], []).append(r["run_uid"])
            tag = "CANARY" if r["is_canary"] else "      "
            print(f"[{tag}] {r['outdir']}")
            print(f"         {' '.join(shlex.quote(x) for x in build_cmd(r, r['outdir']))}")
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        print(f"\n唯一目录 {len(seen)} / 计划 {len(plan)}")
        if dups:
            print("!! 目录碰撞 !!")
            for k, v in list(dups.items())[:10]:
                print(f"   {k} <- {v}")
            return 2
        return 0

    try:
        info = preflight(args, plan)
    except Fail as e:
        print(f"\n[FAIL-CLOSED] {e}\n发车中止。", file=sys.stderr)
        return 1
    head = info["head"]

    launcher_sha = sha256_file(Path(__file__))[:16]
    manifest = launch_dir / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps({
        "started": now_iso(), "head": head, "launcher_sha256_16": launcher_sha,
        "jobs": args.jobs, "env": info["env"], "n_runs": len(plan),
        "frozen_shas": FROZEN_SHAS, "argv": sys.argv,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.preflight_only:
        print("--preflight-only: 未发车。")
        return 0

    ledger = Ledger(launch_dir / "main624_runs.tsv")
    (launch_dir / "logs").mkdir(parents=True, exist_ok=True)
    stop_file = launch_dir / "STOP"

    # --- 信号: 优雅排空,第二次才硬停 ---
    state = {"draining": False, "hard": False}

    def on_sig(signum, frame):
        if state["draining"]:
            state["hard"] = True
            print("\n[SIGNAL] 第二次信号: 硬停,杀死在跑进程。", flush=True)
        else:
            state["draining"] = True
            print("\n[SIGNAL] 停止派发新 run,等待在跑的 run 结束。再按一次强制终止。", flush=True)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    # --- 预扫: SKIP / 隔离残缺 ---
    print("== 扫描已有产物 ==")
    queue, n_skip, n_quar = [], 0, 0
    for r in plan:
        ok, why = completion_status(repo, r["outdir"], r, head)
        if ok:
            n_skip += 1
            ledger.write({**r, "status": "SKIP", "start_iso": now_iso(),
                          "end_iso": now_iso(), "duration_s": 0, "exit_code": 0,
                          "host": socket.gethostname(), "git_commit": head[:7],
                          "launcher_sha": launcher_sha})
            continue
        if why != "absent":
            qp = quarantine(repo, r["outdir"], launch_dir, why)
            n_quar += 1
            print(f"  [QUARANTINE] {r['run_uid']} ({why}) -> {qp}")
        queue.append(r)
    print(f"  SKIP {n_skip} / 隔离 {n_quar} / 待跑 {len(queue)}\n")

    canary_state = {g: "pending" for g in {r["group"] for r in plan}}
    for g in list(canary_state):
        c = next(r for r in plan if r["group"] == g and r["is_canary"])
        okc, _ = completion_status(repo, c["outdir"], c, head)
        if okc:
            canary_state[g] = "ok"
    group_fail = {g: 0 for g in canary_state}

    running, consec_fail, done = [], 0, 0
    t0 = time.time()

    def dispatchable(r):
        if r["is_canary"]:
            # canary 独占其组;且不与同组任何 run 并跑
            return canary_state[r["group"]] == "pending" and \
                   not any(x["r"]["group"] == r["group"] for x in running)
        return canary_state[r["group"]] == "ok"

    def launch(r):
        outdir = r["outdir"]
        (repo / outdir).parent.mkdir(parents=True, exist_ok=True)
        log = launch_dir / "logs" / f"{r['run_uid']}.log"
        fh = open(log, "w", encoding="utf-8")
        cmd = build_cmd(r, outdir)
        fh.write(f"# {now_iso()}\n# {' '.join(shlex.quote(x) for x in cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(repo), stdout=fh,
                                stderr=subprocess.STDOUT)
        print(f"[{now_iso()}] START  {r['run_uid']}"
              + ("  (CANARY)" if r["is_canary"] else ""), flush=True)
        return {"r": r, "proc": proc, "fh": fh, "log": str(log),
                "t0": time.time(), "start_iso": now_iso()}

    while (queue or running) and not state["hard"]:
        if stop_file.exists() and not state["draining"]:
            state["draining"] = True
            print(f"[{now_iso()}] 检测到 STOP 文件,停止派发。", flush=True)

        # 派发
        while (not state["draining"] and len(running) < args.jobs
               and consec_fail < args.max_consecutive_failures):
            cand = next((r for r in queue if dispatchable(r)
                         and group_fail[r["group"]] < args.max_group_failures), None)
            if cand is None:
                break
            fv = free_vram_mib()
            if fv is not None and running and fv < args.min_free_mib:
                print(f"[{now_iso()}] 显存空闲 {fv} MiB < {args.min_free_mib},暂缓派发",
                      flush=True)
                break
            queue.remove(cand)
            running.append(launch(cand))

        if not running:
            if queue and not state["draining"]:
                print(f"[{now_iso()}] 无可派发 run 但队列非空 "
                      f"(canary 阻塞或失败上限),终止。", flush=True)
            break

        time.sleep(5)

        # 收割
        for job in list(running):
            rc = job["proc"].poll()
            if rc is None:
                continue
            running.remove(job)
            job["fh"].close()
            r = job["r"]
            dur = int(time.time() - job["t0"])
            okc, why = completion_status(repo, r["outdir"], r, head)
            status = "COMPLETED" if (rc == 0 and okc) else "FAILED"
            md = load_metadata(repo, r["outdir"]) or {}
            ledger.write({**r, "status": status, "start_iso": job["start_iso"],
                          "end_iso": now_iso(), "duration_s": dur, "exit_code": rc,
                          "host": socket.gethostname(),
                          "git_commit": str(md.get("git_commit", ""))[:7],
                          "git_dirty": md.get("git_dirty", ""),
                          "launcher_sha": launcher_sha, "log_path": job["log"]})
            done += 1
            elapsed = (time.time() - t0) / 3600
            print(f"[{now_iso()}] {status:9s} {r['run_uid']}  {dur}s  rc={rc}"
                  + ("" if okc else f"  ({why})")
                  + f"   [{done} done, {len(queue)} queued, {elapsed:.1f}h]", flush=True)

            if status == "COMPLETED":
                consec_fail = 0
                if r["is_canary"]:
                    try:
                        verify_canary(repo, r["outdir"], r, head,
                                      strict_dirty=(not args.no_strict_canary_dirty))
                        canary_state[r["group"]] = "ok"
                        print(f"[{now_iso()}] CANARY PASS: {r['group']} "
                              f"→ 放行剩余 103 个", flush=True)
                    except Fail as e:
                        canary_state[r["group"]] = "blocked"
                        print(f"[{now_iso()}] CANARY VERIFY FAIL: {r['group']}\n"
                              f"  {e}\n  该组 103 个 run 已封锁。", flush=True)
            else:
                consec_fail += 1
                group_fail[r["group"]] += 1
                if r["is_canary"]:
                    canary_state[r["group"]] = "blocked"
                    print(f"[{now_iso()}] CANARY FAILED: {r['group']} 全组封锁。",
                          flush=True)
                if consec_fail >= args.max_consecutive_failures:
                    print(f"[{now_iso()}] 连续 {consec_fail} 次失败,停止派发。",
                          flush=True)

    if state["hard"]:
        for job in running:
            job["proc"].kill()
            job["fh"].close()

    # 汇总
    print("\n== 汇总 ==")
    print(f"墙钟 {(time.time()-t0)/3600:.2f} h; 本轮完成 {done}; 剩余队列 {len(queue)}")
    for g, st in sorted(canary_state.items()):
        n_ok = sum(1 for r in plan
                   if r["group"] == g and completion_status(repo, r["outdir"], r, head)[0])
        print(f"  {g:24s} canary={st:8s}  完成 {n_ok}/104")
    print(f"TSV: {ledger.path}")
    total_ok = sum(1 for r in plan if completion_status(repo, r["outdir"], r, head)[0])
    print(f"全局 {total_ok}/{EXPECTED_TOTAL}")
    if total_ok < EXPECTED_TOTAL:
        print("未满 624:修掉失败原因后重跑同一条命令即可补跑 (已完成的会 SKIP)。")
        return 3
    print("624/624 完成。立即执行备份 (MAIN_PREREG_01 §12.2 第 5 条)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
