# 主实验预注册（MAIN-01）：扫描方向价值的负载门控与任务依赖

**状态**：在主实验任何一次性能运行之前冻结。
**日期**：2026-07-28
**上游依据**：`docs/03_EVIDENCE_LEDGER.md` §8b（批次 C）、§8d（批次 D）、§8e（P0-B）；`P0B_PREREG_ADDENDUM_01.md`；`P0B_PREREG_ANALYSIS_PLAN.md`。

---

## 0. 本件的地位

本件是**确认性实验**的预注册。P0-B 为可行性 pilot（`docs/P0B_CONFIG_TABLE.md` §1 已将其 13 个条件的"确认性"列全部标为否），其结果仅用于方差量级估计与工程验证，不构成本件任何主张的证据。

本件与既有预注册的关系：
- `P0B_PREREG_ANALYSIS_PLAN.md`（补充件 02）规定的主终点、尾窗、区间估计方法**整体沿用**，本件不修改。
- `P0B_PREREG_ADDENDUM_01.md` §1（对比 ②③④ 共线性）、§2（禁止用批次 C 效应量外推）、§3（`contrast_5 ≈ 0` 的事前预期）**继续生效**。
- 本件新增：数据集维度、增强策略、判据 M1–M3、跨数据集序判据、以及相应的 claim boundary。

---

## 1. 科学命题

三分量框架：**方向价值 =（结构 + 多样性 + 交互，各自被扫描负载门控）× 任务的方向性因果结构**。

本实验检验第二个乘子。核心可证伪命题有二：

**命题 A（几何专属性）**：多路径收益仅在路径携带几何结构时出现。四条互不相同的**随机**路径相对四条相同随机路径无收益；四条互不相同的**几何**路径则有收益。

**命题 B（任务依赖的序）**：几何专属的多路径收益，其大小随任务的方向性因果结构强度而变，且该序可事前预测。

命题 A 若成立，则将"多路径收益"与"集成效应"分离——这是既有扫描顺序文献（依赖 k=1 vs k=4 对比，缺少随机-多样对照格）无法区分的量。

---

## 2. 设计矩阵

### 2.1 主体（Mamba）

| 因子 | 水平 | 数 |
|---|---|---|
| dataset | cifar10, organamnist, organcmnist, organsmnist, eurosat | 5 |
| exp_id | GEO_SG1–SG4, GEO_DIV, RND_S1–S3, RND_D1–D3, LOC_S, LOC_D | 13 |
| reliance | R_low (grid8, patch4, L=64), R_high (grid32, patch1, L=1024) | 2 |
| training_seed | 0, 1, 2, 3 | 4 |

**合计 520 runs。**

### 2.2 GRU 稳健性子实验

仅在 `cifar10` 上重复全部 13 条件 × 2 reliance × 4 seed = **104 runs**。

只跑一个数据集但跑全 13 条件：残缺条件无法计算完整的 ②，稳健性声明会不完整。

**总计 624 runs。**

### 2.3 seed 数固定为 4 的理由

冻结路径库的基数即为 4：G 族 {G1..G4}、R 族每套 {R*_1..R*_4}、L 族 {L1..L4}。拉丁方设计为 4×4，seed s 取 `R*_{s+1}`、`L_{s+1}`，GEO_DIV 取 [G1,G2,G3,G4] 的第 s 次旋转。

第 5 个 seed 需要 `R1_5 / R2_5 / R3_5 / L5`，这些路径**不存在于冻结产物中**；生成它们将破坏 `P0B_R_PATH_BANK_FROZEN.json` 与 `P0B_L_PATH_BANK_FROZEN.json` 的 SHA，进而使既有 104 个 P0-B run 的溯源链断裂。

功效依据：P0-B 实测跨 seed 标准差多在 0.2–0.6 pp（ledger §8e.7）；增强敏感性检查显示禁用水平翻转后方差约翻倍（§9.2）。即便按翻倍后的方差，主要效应在 2–3 个 seed 下即达 80% 功效。4 seed 提供余量。

---

## 3. 冻结产物清单

| 产物 | SHA-256 |
|---|---|
| `docs/P0B_CONFIG_TABLE.md` | `790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e` |
| `P0B_RUN_LEDGER_104.csv` | `906f6af2f8a695b443b01ac9ff89e29f24b4cea85fb4717252404f58145bfe25` |
| `P0B_L_PATH_BANK_FROZEN.json` | `93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7` |
| `P0B_R_PATH_BANK_FROZEN.json` | `2f7b8a6fd3cfbbae9897b4ef4dc9dcfd1bf7744619d5818ceaca7604d565aee3` |
| `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` | `e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09` |
| `P0B_EUROSAT_SPLIT_FROZEN.json` | `f5ddb2db3f8ffc74efb77295e0fac17d34df85179bcd78de3f4e638b685c4117` |

ledger 定义的是 `(exp_id, grid, training_seed) → 路径分配`，与 dataset、augmentation、backbone **正交**，故五个数据集与 GRU 子实验共用同一 ledger，四源 SHA 门保持不变。

### 3.1 L 路径库的限制声明

`P0B_L_PATH_BANK_FROZEN.json` 的元数据自述为 *"B0 G1 first post-burn thinning candidate Q, expanded by reversal and cell transpose; **not a final approved L generator**"*。

本实验**沿用**该路径库，不重新生成，并声明：

- L 轨道的构造与 G 轨道**平行**：L1 为从 G1 链恢复的候选 Q，L2 = reverse(L1)，L3 = transpose_cells(L1)，L4 = reverse(transpose_cells(L1))；G2/G3/G4 同样由 G1 经反转与转置生成。故 L1–L4 与 G1–G4 的轨道结构同构，`mean(LOC)` 与 `mean(GEO_S*)` 的比较在口径上成立。
- L1/L2 的 axis_bias 为负、L3/L4 为正（grid32：∓1.8904），轨道覆盖两个轴向，与 G 轨道一致。
- 但该库为 P0-B 阶段冻结的**候选**生成器，未经最终审定。**凡依赖 L 族的结论（即 ⑤ 与 locality 回收比例）一律标注该限制，且不得作为确认性主张。**

---

## 4. 数据管线

### 4.1 数据集规格

| dataset | 类别 | 原始尺寸 | 通道 | train | validation | 划分来源 |
|---|---|---|---|---|---|---|
| cifar10 | 10 | 32×32 | 3 | 45,000 | 5,000 | 冻结（`e28719c9…`） |
| organamnist | 11 | 28×28 | 1 | 34,561 | 6,491 | MedMNIST 官方 |
| organcmnist | 11 | 28×28 | 1 | 12,975 | 2,392 | MedMNIST 官方 |
| organsmnist | 11 | 28×28 | 1 | 13,932 | 2,452 | MedMNIST 官方 |
| eurosat | 10 | 64×64 | 3 | 22,000 | 2,500 | 冻结（`f5ddb2db…`） |

三个 Organ 变体为**同一批腹部 CT 器官的三个解剖切面**（轴位 / 冠状位 / 矢状位），类别数与标注体系相同，构成域内的受控各向异性操纵。

official test 在本实验全程**不实例化、不评估、不报告**。EuroSAT 的 test split 虽已冻结，本实验亦不构造。

### 4.2 统一预处理

五个数据集一律输出 `(B, 3, 32, 32)` float32：

```
train: Resize((32,32), BILINEAR, 作用于 PIL)   [cifar10 跳过]
     → RandomCrop(32, padding=4)
     → ToTensor
     → RepeatGrayscaleChannels                 [仅 Organ]
     → Normalize(各数据集自身常数)

eval : Resize((32,32), BILINEAR, 作用于 PIL)   [cifar10 跳过]
     → ToTensor
     → RepeatGrayscaleChannels                 [仅 Organ]
     → Normalize(各数据集自身常数)
```

灰度经 repeat 转 3 通道而非色彩空间转换，使 patch embedding 在五个数据集上**完全相同**——任何跨数据集差异都不得归因于架构差异。

所有 resize 作用于 PIL Image 且显式指定 BILINEAR，以规避 torchvision 0.15 与 0.16 之间 `antialias` 默认行为的差异（本地 0.16.2、云端 0.15.2）。

### 4.3 归一化常数

均基于各数据集自身 train split、resize 之后计算，硬编码于源码：

| dataset | mean | std |
|---|---|---|
| cifar10 | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) |
| organamnist | (0.4681101025,)×3 | (0.2801411101,)×3 |
| organcmnist | (0.4942488707,)×3 | (0.2674806004,)×3 |
| organsmnist | (0.4954148361,)×3 | (0.2679301867,)×3 |
| eurosat | (0.3447923299, 0.3808058131, 0.4081652860) | (0.1978067505, 0.1315186118, 0.1098431517) |

EuroSAT 常数基于 `P0B_EUROSAT_SPLIT_FROZEN.json` 的 22,000 个 `train_indices` 计算。

---

## 5. 增强策略：五数据集统一禁用水平翻转

### 5.1 决定

`augmentation = "main_uniform"`，五个数据集**全部**禁用 `RandomHorizontalFlip`。`RandomCrop(32, padding=4)` 保留（平移在 x/y 上各向同性，不注入方向偏好）。

### 5.2 理由

**原则性理由**：水平翻转本身即为方向性干预，向数据分布注入左右反射对称性。本研究的对象正是方向性结构，用方向对称化的增强去测量方向的价值，在方法上自相矛盾。

**设计性理由**：主实验的核心命题之一是 ② 在数据集间的**排序**。若仅 Organ 禁用翻转（其解剖左右有语义），跨数据集的效应量差异将与增强策略差异混淆，且偏差方向有利于假设（anticonservative）。

**实证依据**：16-run 增强敏感性检查（CIFAR-10、grid32、GEO_SG1/GEO_DIV/RND_S1/RND_D1 × 4 seed，formal 模式），与 P0-B 同名条件对照：

| 量 | 带 HFlip（P0-B） | 无 HFlip |
|---|---|---|
| `P_G' = GEO_DIV − GEO_SG1` | +2.84 [2.23, 3.46] | **+3.50** [2.14, 4.86] |
| `P_R' = RND_D1 − RND_S1` | +0.15 [−0.58, 0.89] ✗ | **+0.11** [−1.35, 1.57] ✗ |
| 交互 `P_G' − P_R'` | +2.69 [1.91, 3.47] | **+3.39** [1.64, 5.14] |

几何多样性收益幸存且点估计增大 23%；随机多样性仍为零。

**代价（已知并接受）**：seed 间标准差约翻倍（`P_G'` 0.388 → 0.853，`P_R'` 0.461 → 0.918），结构组尾窗 train_accuracy 由 91.6–94.6% 升至 95.7–96.9%，validation 绝对值下降 1.2–1.9 pp。

### 5.3 该证据的外推限制

敏感性检查仅在 CIFAR-10 上执行。禁用水平翻转对 Organ（解剖左右有语义）与 EuroSAT（航拍影像近似反射不变）的影响未经直接检验。此项列入 limitation。

---

## 6. 主终点与统计口径

沿用 `P0B_PREREG_ANALYSIS_PLAN.md` §1–§3，不做修改：

- **主终点**：validation accuracy，来自各数据集的冻结/官方 validation split
- **尾窗**：epoch 80..100 含端点共 21 个，按 run 取算术平均
- **观测数**：每个 design cell 为 4（training_seed 0..3）
- **区间**：`mean ± t(3, 0.975) × s / sqrt(4)`，`t(3,0.975) = 3.182`，s 为样本标准差（ddof=1）
- **单位**：百分点（pp）
- 禁止 best-epoch 选取（构成对 validation 的多次查看）

训练侧指标全程记录，仅作诊断，不作判定依据。

### 6.1 对比定义

引用 `docs/P0B_CONFIG_TABLE.md` §1，原样复述、不修改配置表：

```
GEO_S* = {GEO_SG1, GEO_SG2, GEO_SG3, GEO_SG4}
RND_S* = {RND_S1, RND_S2, RND_S3}
RND_D* = {RND_D1, RND_D2, RND_D3}

P_G    = GEO_DIV − mean(GEO_S*)
P_R    = mean(RND_D*) − mean(RND_S*)
P_LMTO = LOC_D − LOC_S

①  mean(GEO_S*) − mean(RND_S*)      结构 × locality
②  P_G − P_R                        几何专属的多路径收益
③  GEO_SG1 − GEO_SG2                traversal polarity
④  GEO_SG1 − GEO_SG3                scan axis
⑤  P_G − P_LMTO                     canonical-orbit specificity
```

每个对比在 R_low、R_high 各算一次；交互 = 按 seed 配对后 R_high 值减 R_low 值。

---

## 7. 确认性判据

**确认性主张仅限以下三条。** 其余全部为探索性，不得事后升级。

### M1（几何多样性收益为正）

对每个数据集，② 在 R_high 上的 t₃ CI 下界 > 0。

### M2（随机多样性收益为零）

对每个数据集，`P_R` 在 R_low 与 R_high 两档的 t₃ CI **均跨零**。

M1 与 M2 合起来构成命题 A。**单独的 M1 不足以支撑"几何专属"**——必须同时观察到随机侧为零。

### M3（跨数据集的序）

**事前预测**：② 在 R_high 上满足

```
min(organamnist, organcmnist, organsmnist)  >  cifar10  >  eurosat
```

**三档判定，全部事前写死**：

| 观察 | 判定 |
|---|---|
| 点估计顺序成立，**且** Organ 最小值与 cifar10 的 t₃ CI 不重叠，**且** cifar10 与 eurosat 的 t₃ CI 不重叠 | **序成立** |
| 点估计顺序成立，但任一相邻对的 CI 重叠 | **方向一致、区分度不足**；不得声称序成立 |
| 点估计顺序不成立 | **序被证伪**；如实报告，并检讨第二个乘子的表述 |

### 7.1 多重比较处理

M1 与 M2 各在 5 个数据集上评估，共 10 个数据集内检验。不采用形式化的族误差率校正，改用**事前投票规则**：

> 命题 A 成立，当且仅当 M1 与 M2 在 **≥4 个数据集**上同时满足。

理由：五个数据集为独立证据来源而非同一假设的重复检验；投票规则避免了 n=4 下 Holm 校正导致的 t 临界值失控（df=3 时 α=0.005 对应 t≈7.45），同时保留了"不能靠单个数据集立论"的约束。

M3 为单一检验，不参与投票。

### 7.2 探索性项目（不得作为确认性主张）

- ① 结构效应及其负载门控交互
- ③ traversal polarity；④ scan axis
- ⑤ canonical-orbit specificity（另受 §3.1 的 L 路径库限制）
- `P_LMTO` 与 locality 回收比例
- GRU 稳健性复现
- Organ 三切面之间的比较
- 天花板诊断（§8）
- ③ 在禁用水平翻转后是否由零转为非零（P0-B 下该零被增强与设计双重决定，本实验可分离）

---

## 8. 天花板诊断

依 ledger §8d.5 与 §8e.6，训练侧终点在高拟合水平下会系统性低估效应。禁用水平翻转后，CIFAR-10 结构组的尾窗 train_accuracy 已达 95.7–96.9%。

**事前规则**：记录每个 run 的尾窗 train_accuracy。对每个 (dataset, reliance)，若结构组（GEO_S* 与 GEO_DIV）的尾窗 train_accuracy 中位数 **> 95%**，则在报告中标注该格处于强饱和区，并声明：

> 该数据集的效应量以 pp 为单位与其他数据集直接比较受限。

M3 的判定基于**点估计的序与 CI 重叠**，对各数据集的基线水平差异稳健；但绝对效应量的跨数据集数值比较受本规则限制。

organamnist 与 organcmnist 的探针准确率为 94.68% / 94.02%（15-epoch 小 CNN、32×32），落入较高区间，饱和风险高于 organsmnist（80.51%）与 eurosat（79.08%）。该风险事前已知并接受：即便饱和发生，其本身即为有信息量的结果，且 M3 的序判据对此稳健。

---

## 9. 事前预期与负值预案

三条写死：

1. 依 `P0B_PREREG_ADDENDUM_01.md` §3，`contrast_5 ≈ 0` 是事前预期。⑤ 已降级为探索性；**不得事后包装为"我们发现 locality 不解释效应"**。
2. 依 ADDENDUM §2，禁止用批次 C 或 P0-B 的效应量对本实验做功效外推。
3. 依 ledger §8d.4 与 §8e，低负载档（R_low）的 diversity/interaction 效应**可能显著为负**。负值是预期内的可能结果，**不得当作实现缺陷排查、不得触发重跑、不得调整 seed 或路径**。任何因出现负值而发起的代码改动必须先停止并提交理由。

---

## 10. 记录项

除 `docs/P0B_CONFIG_TABLE.md` §11.6 已规定的 metadata 外，每个 run 另记录：

- `dataset`、`augmentation`、`backbone`
- 尾窗 train_accuracy、尾窗 validation accuracy、二者之差
- epoch 窗 (10,20) 与 (90,100) 的 validation accuracy
- 起止时间戳（runner 自身不记录耗时，由发车脚本补充）

前三项进入 metadata；诊断量在分析阶段从 `validation_history` 导出。

---

## 11. Claim boundary

- 结论限于：五个 32×32 分类数据集、channel-split 架构、Mamba backbone（GRU 仅在 cifar10 上作稳健性检查）、d_model=256、两档 reliance（grid8/grid32）、本文冻结的 G/R/L 三族路径、以及禁用水平翻转的统一增强。
- **推断的非对称性**：对比 ① 与 ② 中，GEO 侧为固定的 G1–G4（不随 seed 变化），RND 侧每个 seed 抽取不同的随机路径实例。因此其 CI **对随机路径可泛化，对几何路径仅条件于这四条 canonical raster 路径**，不构成对"几何路径一般"的推断。
- **跨任务的动机缺口**：M3 的事前排序，其文献动机来自分割任务（Flatten Wisely 为脑 MRI 分割，报告 27 Dice 点跨度；Zhu 为遥感分割，报告无显著差异），而本实验为分类任务。分割是稠密预测，空间局部性直接进入损失；分类只需全局池化后的判别信息。因此本实验的表述限于：**本框架预测了一个跨域的序，该序若成立，则为解释上述两篇文献的分歧提供一个候选机制**——而非"我们解释了该矛盾"。
- ⑤ 与 locality 回收比例另受 §3.1 的 L 路径库限制。
- 未测尺度、未测架构、未测任务域（检测、分割、生成）均不外推。

---

## 12. 成本与发车协议

### 12.1 预算

按 P0-B 实测（grid8 1690 s/run、grid32 6995 s/run，5 进程并行）与各数据集训练集规模折算：

| 组 | runs | process-h |
|---|---|---|
| cifar10 (Mamba) | 104 | 125 |
| organamnist | 104 | 96 |
| organcmnist | 104 | 36 |
| organsmnist | 104 | 39 |
| eurosat | 104 | 61 |
| cifar10 (GRU) | 104 | ~88 |
| **合计** | **624** | **~446** |

5 进程并行 → 墙钟约 **89 小时（3.7 天）**。

### 12.2 发车协议

1. 每个数据集的第一个 run 作为 canary 单独执行，`COMPLETED` 后方启动该数据集剩余 103 个
2. 发车前 `git status --short` 必须为空；跑批期间**禁止任何 commit**（`validate_completed_run` 比对 `git_commit`，新 commit 将使已完成 run 无法 SKIP）
3. 发车脚本记录每个 run 的起止时间戳与退出码
4. 单个 run 失败不中止整批；失败者记入 TSV 并保留日志，重跑同一脚本即可补跑
5. 全部完成后立即打包备份至数据盘

---

## 13. 落盘后

在 `.gitattributes` 加入 `MAIN_PREREG_01.md -text`；commit；记录本文件自身的 SHA-256 并回填至文末，注明该 SHA 对应回填前那一版的 commit。

**本件冻结后，第 7 节的判据不得修改。** 任何设计变更须以新的补充件形式追加，并说明变更时点与已完成的 run 数。

---

**本件 SHA-256**：（回填）
**对应 commit**：（回填）
