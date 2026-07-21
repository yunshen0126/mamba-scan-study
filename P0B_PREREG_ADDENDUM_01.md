# P0-B 预注册补充件 01：对比共线性、效应量外推限制与冻结产物完整性

**日期：2026-07-21**
**状态：在任何 P0-B 性能运行之前冻结。**
**适用范围：仅 P0-B feasibility pilot。**

---

## 0. 本件的地位

本件**不修改** `docs/P0B_CONFIG_TABLE.md`（SHA `790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e`），
也**不修改** `P0B_PREREG_FREEZE_L_AUC.md`、`P0B_PREREG_FREEZE_R_PATHS.md`、
`P0B_PREREG_FREEZE_CIFAR10_VAL_SPLIT.md`。

上述文件保持逐字节不变，其 SHA 继续有效。本件以独立文件的形式补充**结果解释约束**与**冻结产物完整性要求**，落盘后自身亦须记录 SHA。

**本件不增加任何实验。** 13 条件 × 2 reliance × 4 training seed = 104 runs 不变。

---

## 1. 主对比 ②③④ 的共线性限制

### 1.1 事实依据

`P0B_PREREG_FREEZE_L_AUC.md` §7 的几何回归值（n=32）：

| 路径集合 | RIGHT | LEFT | DOWN | UP | AUC_macro |
|---|---:|---:|---:|---:|---:|
| `{G1,G3}` | 0.975 | 0 | 0.975 | 0 | 0.4875 |
| `{G1,G2,G3,G4}` | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 |

四路径集合相对两路径集合 `{G1,G3}` 的**全部**覆盖增益都落在 LEFT 与 UP 两个总体上；RIGHT 与 DOWN 完全相等。

其结构原因是冻结定义本身：`G2 = N−1−pi_G1`、`G4 = N−1−pi_G3` 为整序列严格反转，因而逐对满足

```text
|pi_G2[u] − pi_G2[v]| = |pi_G1[u] − pi_G1[v]|
```

即 `C_undirected(G1,G2,G3,G4) = C_undirected(G1,G3)`。

### 1.2 由此产生的限制

`GEO_DIV` 相对四个单一几何条件所增加的多样性，可分解为两个正交成分：

```text
轴向多样性:  G1 vs G3   （AxisBias 符号相反）
极性多样性:  G1 vs G2, G3 vs G4   （无向 d_seq 逐对相同，仅传播方向相反）
```

其中**极性成分对无向覆盖的贡献恒为零**，其全部作用体现在有向覆盖上。

而 P0-B 的主对比③与④恰好分别隔离这两个成分：

```text
③ traversal polarity:  GEO_SG1 − GEO_SG2
④ scan axis:           GEO_SG1 − GEO_SG3
```

**因此主对比②的多样性成分，在本设计中可分解为③与④各自隔离的两个维度。②③④不是三条独立证据。**

报告与论文中：

- 不得把 ②、③、④ 同时显著并列计数为三项独立支持；
- 报告 ② 时须同时报告 ③ 与 ④，并说明 ② 的效应可归因于哪一（或哪些）成分；
- 若 ② 显著而 ③④ 均不显著，须作为内部不一致明确指出，而非只报告 ②。

### 1.3 与 C4 判定口径的关系

`P0B_PREREG_FREEZE_L_AUC.md` §7 已据同一事实修正了 C4 的判定口径
（改用 `AUC_macro`，并禁止写成"四路径集合在每个已覆盖方向上都严格更高"）。
本节是该修正在**性能侧**的对应约束；两者依据相同，缺一不可。

---

## 2. 禁止使用批次 C 的效应量对 P0-B 做功效外推

批次 C 的 `channel_real_4dir` 使用的路径集合为：

```text
(row, col, diag, anti_diag)
```

该集合含 diag 与 anti_diag，具有**真实的无向覆盖多样性**。其测得的 geom-div 交互为 2.1–3.7 pp。

P0-B 的 `GEO_DIV` 使用：

```text
(G1, G2, G3, G4) = (raster_lr, raster_reverse, column_raster, column_reverse)
```

由 §1.1，该集合的无向覆盖等于 `{G1,G3}`。**两者不是同一个操纵。**

因此：

- **P0-B 主对比②的效应量与方差，只能由 P0-B 自身的 C8 估计给出。**
- 使用批次 C 的 2.1–3.7 pp 进行样本量或功效规划会系统性高估。
- 主实验（严格型 20 条件 vs 预算型 11 条件、seed 4 vs 8）的规模决策必须依据 P0-B 的 C8，不得依据批次 C。

本限制不涉及修改 `P0B_CONFIG_TABLE.md` §8.1 的冻结路径定义。禁止在主几何集合中混入 diag / serpentine / hilbert 的理由（会同时改变 topology、locality、axis、coverage 四个属性）依然成立。本节是记账与外推约束，不是设计变更。

---

## 3. `contrast_5` 的事前预期

### 3.1 事实依据

三项已记录的定量事实：

1. `docs/03_EVIDENCE_LEDGER.md` §11.1：旧 block-randomized serpentine 候选族在四个 `(n,b) ∈ {8,32}×{2,4}` 组合中均为 **0/20000** 条通过 C5，合计 8×10⁴ 次抽样零通过，C5 未放宽。
2. `P0B_PREREG_FREEZE_L_AUC.md` §4：n=32 的 LMTO 基路径 L1 相对 G1 的 normalized Kendall distance 约为 **0.006411**。
3. `P0B_PREREG_FREEZE_L_AUC.md` §7：n=32 下四路径 `AUC_macro` 分别为 LMTO **0.9645**、G **0.975**。

C5 的实测余量亦极窄（n=32）：

| 统计量 | G 目标 | LMTO 实测 | ±10% 上界 | 余量 |
|---|---:|---:|---:|---:|
| mean | 16.5 | 18.140625 | 18.15 | 0.009 |
| p90 | 32.0 | 35.0 | 35.2 | 0.2 |

### 3.2 事前声明

上述证据一致表明：**C5 的三统计量 ±10% 约束把可行集压缩到 G 的邻域附近。** 一个结构上确实不同的候选族在 8×10⁴ 次抽样中零通过，而实际冻结的 LMTO 与 G 在序关系与有向覆盖上都高度接近。

因此本件在任何 P0-B 性能结果产生之前声明：

> **`contrast_5 ≈ 0` 属于事前预期结果。**
>
> 该零结果既不支持也不否定"多路径收益来自 locality 而非轴向/极性组织"这一命题，也不构成对 canonical raster orbit 特殊性的否定。
>
> `contrast_5` 在 P0-B 中的作用限于两点：(i) 验证 LMTO 路径可被稳定训练；(ii) 提供该对比的效应量方差量级，供主实验规划使用。

本声明与 `P0B_PREREG_FREEZE_L_AUC.md` §6 的解释边界及 §8 的禁用描述清单一致，是其**定量补充**，不替代、不放宽。

### 3.3 不删除 LOC_S / LOC_D

尽管 `contrast_5` 的信息量受限，条件 12–13（16 runs，约 8 GPU·小时）予以保留。理由：

- P0-B 定位为 feasibility pilot，`P0B_CONFIG_TABLE.md` §3 明确"不要求效应显著非零"；
- LMTO 路径的可训练性本身是需要验证的工程事实；
- 删除条件将改变已冻结的 104-run 设计规模与 `P0B_RUN_LEDGER_104.csv`。

---

## 4. reliance 操纵有效性检查（新增预注册检查项）

### 4.1 背景

P0-B 的 reliance 操纵为 `R_low = grid8 (L=64)` 与 `R_high = grid32 (L=1024)`，架构为 d256、数据集为 CIFAR-10。

已有证据表明 `R_low` **不是零 reliance 锚点**：

- CIFAR-10 full-branch 的 `order_utilization` 为 grid8 (gru 5.32 / mamba 4.86) 对 grid32 (gru 10.57 / mamba 11.90)，比值约 1:2.4；
- 批次 D（CIFAR-100、d256）的 TEST 侧交互在 mamba 上三档全部显著为正（grid8 +1.17、grid16 +1.71、grid32 +4.69），呈梯度而非阈值。

此外，**CIFAR-10 × d256 × grid8 这一格从未运行过**：批次 C 的 d256 只有 grid32，grid8/grid16 的证据全部来自 d64。批次 D 的 d256 三档在 CIFAR-100 上。因此"数据集"与"模型宽度"两个候选解释在现有数据中不可分离，本件不就该模式作归因，也不为此增加实验。

### 4.2 检查项

> **操纵有效性检查：** 报告两档 reliance 下各自的跨路径性能跨度
> ```text
> S_order = max_j Perf_j − min_j Perf_j
> ```
> j 遍历 `GEO_SG1..GEO_SG4` 与 `RND_S1..RND_S3`（即全部单一路径条件），两档分别计算。
>
> reliance 操纵有效的前提是 `S_order(grid32) > S_order(grid8)`。
>
> 若两者接近，则两档之间的 reliance 对比被压缩，主对比①与②（均定义为"在两档 reliance 下的差"）的功效低于规划值。此情形须在结果中如实报告为**操纵强度不足**，不得解释为"reliance 无调节作用"的零效应证据。

### 4.3 表述规范

论文与报告中：

- 不得把 `R_low = grid8` 描述为"低 reliance"以外的任何更强表述（如"无 reliance"、"扫描顺序不起作用的对照"）；
- 该操纵是**中 vs 高**的对比，不是**无 vs 有**；
- 本检查使用 P0-B 已有的 104 个 run，不增加任何实验。

---

## 5. 冻结产物完整性：路径、机器与行尾符

### 5.1 冻结产物清单（三列）

原交接文档仅以文件名与 SHA 记录冻结产物，未记录路径与所在机器，导致 fail-closed 规则在实际操作中无从执行（不知校验哪个文件即无从匹配）。现补全：

| 文件 | 仓库相对路径 | SHA-256 |
|---|---|---|
| L 路径库 | `P0B_L_PATH_BANK_FROZEN.json` | `93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7` |
| R 路径库 | `P0B_R_PATH_BANK_FROZEN.json` | `2f7b8a6fd3cfbbae9897b4ef4dc9dcfd1bf7744619d5818ceaca7604d565aee3` |
| val split | `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` | `e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09` |
| 配置表 | `docs/P0B_CONFIG_TABLE.md` | `790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e` |
| run ledger | `P0B_RUN_LEDGER_104.csv` | `906f6af2f8a695b443b01ac9ff89e29f24b4cea85fb4717252404f58145bfe25` |

所在机器：自 commit `8055759` 起，五者均已 track 并 push，本地（Windows）与云端（AutoDL）两处一致。

**冻结依据文件亦纳入版本控制**（此前为 untracked，不受 fail-closed 保护）：

```text
P0B_PREREG_FREEZE_L_AUC.md
P0B_PREREG_FREEZE_R_PATHS.md
P0B_PREREG_FREEZE_CIFAR10_VAL_SPLIT.md
docs/P0A_B0_FORMAL_DEFINITIONS.md
P0B_L_PATH_BANK_CANDIDATE.json
```

### 5.2 「未 push 不算冻结」

> 一个文件只有在**已 track 且已 push 到 origin** 之后，其 SHA 才构成有效冻结。
> 仅存在于单台机器工作区的文件不受版本控制保护，无可回滚版本，其 SHA 不具备冻结效力。

### 5.3 行尾符不变量

冻结产物必须在 `.gitattributes` 中标注 `-text`，禁止任何行尾符转换。

实测依据：`P0B_L_PATH_BANK_FROZEN.json` 在磁盘上为 **CRLF**，而 `docs/P0B_CONFIG_TABLE.md` 为 **LF**（生成工具不同）。§10 的五个 SHA 是在各自当前形态上计算的。若允许 `core.autocrlf` 规范化，跨平台 checkout 后 SHA 必然不匹配，B4B 的 source gate 会 fail closed，**而失败原因是行尾符而非文件被篡改，两者在 fail-closed 规则下无法区分**。

跨平台验证已完成：commit `8055759` push 后，云端 `git pull` 所得五个 SHA 与本地逐字相同。

**新增冻结产物时，必须先在 `.gitattributes` 中加入 `-text` 条目，再 `git add`。**

---

## 6. 待办（非预注册内容，记入工作清单）

以下为发车前的工程项，不属于预注册约束：

1. **B4C 的阻塞清单应从四条扩为五条。** `REPORT_B4B_P0B_PATH_INTEGRATION.md` 的 CPU 测试使用 `n_layers=0`，此时 `group_blocks` 为空、模型绝大部分参数不存在，故其"G/R/LMTO 六类参数与缓冲区一致"一项的检验强度远低于字面。B4C 阻塞 #4 修复 C6 测试为 `n_layers=2, block_type=gru` 时，应将 B4B 的参数/缓冲区检查一并重跑。C6 要求的是"13 条件参数量、FLOPs、训练计划完全一致"。

2. **确认 runner 的目录与列命名不依赖 `variant`。** explicit 模式强制 `variant="channel_same_row_4"`，13 个条件的 `self.variant` 完全相同。配置表规定输出目录为 `outputs/p0b_{exp_id}_{reliance}_seed{S}/`（使用 `exp_id`），应确认 runner 与分析脚本均不以 `variant` 作为区分键，否则 13 个条件会互相覆盖。

3. **批次 D 的复现基线 commit 为 `02981d9`**，非 `8055759`。批次 D 在云端运行时，仓库处于 `02981d9`，其后四个 commit（`f8b4785`、`1a22e26`、`fe2e8f0`、`2f3606d`）均为文档与 PSI 分析代码，不含训练代码，故不影响批次 D 结果。此事实应记入批次 D 的 ledger 条目。

---

## 7. 落盘后

本件落盘后须：

1. 在 `.gitattributes` 中加入 `P0B_PREREG_ADDENDUM_01.md -text`
2. `git add` 并 commit
3. 计算并记录本件自身的 SHA-256，追加到冻结产物清单（§5.1）
4. 在 `docs/03_EVIDENCE_LEDGER.md` 追加一条指向本件的记录

本件在任何 P0-B 性能结果产生之前冻结。
