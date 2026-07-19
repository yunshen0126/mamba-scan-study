# 批次 D 预注册:CIFAR-100 channel-split 2×2 析因

**撰写时间:2026-07-19,发车前。本文件在任何 CIFAR-100 结果产生之前写定,提交后不得修改判据部分。**

## 1. 目的

批次 C 已在 CIFAR-10 上确立三分量负载门控(structure / diversity / interaction)。批次 D 在 CIFAR-100 上复制同一析因,目的是**削弱"结论仅限单一数据集"这一 limitation**,并检验三分量模式是否跨数据集稳健。

本批次**不**用于检验因果结构轴(那需要换任务域,属分割实验的职责)。CIFAR-100 与 CIFAR-10 同为自然图像分类,预期因果结构相近,属**同因果结构、不同任务难度**的复制。

## 2. 配置

- 数据集:CIFAR-100(100 类),32×32,标准 train/test 划分
- 架构:channel_split,四变体 real_4dir / same_row_4 / same_perm_4 / rand_perm_4
- 尺度:**仅 d256**(d64 在 100 类下预期严重欠拟合,信息量低;时间受限故只跑 d256)
- grid ∈ {8, 16, 32},block ∈ {gru, mamba},seed ∈ {0,1,2,3,4}
- 超参与批次 C **完全一致**:`--n-layers 2 --effective-batch 128 --epochs 100 --warmup-epochs 5 --base-lr 0.001 --weight-decay 0.05 --grad-clip 1.0 --pos-mode xy_learned --num-workers 4`
- 唯一允许的差异:`--dataset cifar100` 及随之而来的 `num_classes=100` 分类头。**任何其他超参改动都会使批次 D 与批次 C 不可比,不得调参。**

## 3. 预注册判据(与批次 C 同构,口径为 train_acc 尾窗 80-100,5-seed t-CI)

- **判据 D1(structure 单调):** structure 效应随 grid 8→16→32 单调递增,两个 block 均成立。
- **判据 D2(diversity 门控):** grid32 的 diversity 5-seed CI 下界 > 0,且 grid8 的 diversity CI 跨 0。
- **判据 D3(interaction 门控):** grid32 的 interaction 5-seed CI 下界 > 0,且 grid8 的 interaction CI 跨 0。
- **判据 D4(几何性):** grid32 的 interaction 分解中,within-structured 多样性收益显著为正,within-shuffle 多样性收益 CI 跨 0。

**同时记录 test 口径全部指标。** 鉴于批次 C 在 d256 出现 train 侧饱和、效应迁移至泛化侧的现象(见 ledger §8b.5),CIFAR-100 的 100 类任务更难、train_acc 预期更低,该饱和可能不出现。**无论出现与否,判据一律按 train 口径判定,test 口径作为预先声明的次要终点(secondary endpoint)报告。** 此处 test 口径的次要终点地位是在发车前声明的,因此在批次 D 中不属事后探索。

## 4. 预期结果与判读分支(发车前写定)

- **分支 A — 三判据全过、模式与 CIFAR-10 同构:** 最有利结果。结论升级为"三分量负载门控在两个自然图像分类数据集上复制",limitation 中"单一数据集"一条削除。
- **分支 B — structure 过、diversity/interaction 门槛位置移动(例如在 grid16 即点亮):** 仍支持门控框架,但说明阈值与任务难度有关。这是**有信息量的结果**,应写入正文而非藏起来:阈值随任务难度移动本身可被两因素账解释(更难的任务对每单位扫描负载的利用更充分)。
- **分支 C — grid32 的 diversity/interaction 在 CIFAR-100 上不显著:** 对框架不利。此时**不得**归因于"CIFAR-100 更难所以噪声大"而搁置,应如实记录,并将 claim 限定回 CIFAR-10,同时将此作为分割实验的重点检验对象。
- **分支 D — train_acc 未饱和(< 85%)且 train/test 结论一致:** 顺带反证批次 C 的 ceiling 解释是特定于 d256+CIFAR-10 的,加强 §8b.5 的可信度。

**以上四个分支在发车前全部写下,任何一个都不构成"实验失败"。** 记录此点是为防止事后按结果挑选叙事。

## 5. 分析口径

复用 `mamba_scan_study/analysis/analyze_csplit_factorial.py`,分组维度为 (seed, block, grid),自变量为 variant。**禁止使用 `tail_80_100_summary.csv`**(该文件按 block×grid 汇总,与 channel-split 的自变量不匹配,见 AGENTS.md 与 ledger §8c)。

CI 计算:5-seed,t₄,.₉₇₅ = 2.776,mean ± t·s/√5。

## 6. 与批次 C 的可比性声明

批次 D 与批次 C(d256)除数据集与分类头外配置完全相同,seed 序列相同,workers 同为 4,因此两批次的效应量可直接并列比较。若实际执行中出现任何配置偏离(例如 microbatch 因显存被迫调整),必须在此文件追加记录,并在论文中明确该批次不可与批次 C 直接并列。
