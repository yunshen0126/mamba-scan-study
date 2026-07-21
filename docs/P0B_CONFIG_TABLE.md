# P0-B 实验配置表(确定版)

**日期:2026-07-20**
**性质:P0-B 为可行性研究(feasibility pilot),非确认性分析。**
**本表冻结后即进入预注册。表内任何字段变更须记入 ledger 并说明理由。**

---

## 0. 冻结的定义(表外不得改动)

### 0.1 主几何路径集合 G

```
G1 = raster_lr        π(r,c) = r·W + c
G2 = raster_reverse   π(u) = N − 1 − π_G1(u)      (整序列严格反转)
G3 = column_raster    π(r,c) = c·H + r
G4 = column_reverse   π(u) = N − 1 − π_G3(u)      (整序列严格反转)
```

### 0.2 随机路径集合 R^s(s = 1,2,3)

每套四条 `R^s_1..R^s_4` 的唯一运行时来源是 `P0B_R_PATH_BANK_FROZEN.json` 中的完整冻结 `order` 数组及 hash。生成 provenance 保留为独立 CPU generator 的 `seed(s,i) = 17071 + 1000·s + i`,但训练或模型代码不得重新调用 `torch.randperm` 恢复路径。

R 与训练 seed 分离:对 `RND_Ss`,训练 seed 0/1/2/3 分别使用 `R^s_1/R^s_2/R^s_3/R^s_4`,并在四通道重复;对 `RND_Ds`,固定使用 `R^s_1..R^s_4`,按原 Latin square 轮换到 ch0..ch3。R 是性能运行前固定的路径阻断因素,不是每个训练 seed 重抽的随机效应样本。

### 0.3 ★ locality-matched topology-perturbed symmetry orbit (LMTO)

P0-B 的辅助控制路径库为预先冻结的 **locality-matched topology-perturbed symmetry orbit (LMTO)**。每个 grid 的四条路径来自 `P0B_L_PATH_BANK_FROZEN.json`:

```text
L1 = Q (B0 预先指定的 G1 链候选)
L2 = reverse(Q)
L3 = transpose_cells(Q)
L4 = reverse(transpose_cells(Q))
```

每条 L 的 `d_seq` mean/p50/p90 位于对应 G 目标的 ±10% 内。完整分布、p95/max、AxisBias、polarity、coverage 和与 G 的距离均只作诊断,不是匹配约束。

**作废记录:** 上一版“locality 匹配随机 Hamilton 路径”及“若 `P_G − P_L` 不显著则收益主要来自 locality”的表述由本节和 §1 对比⑤替代,不得并列使用。LMTO 不是随机路径总体样本,也不是与 G 独立的随机样本。

### 0.4 ★ 有向覆盖与冻结 AUC(预注册冻结,不得事后挑选)

四个总体分别为 `RIGHT`, `LEFT`, `DOWN`, `UP`;节点固定为 `x=tau/(N-1) ∈ {0.01, 0.05, 0.10, 0.20}`。增加数学锚点 `C_dir(0)=0`。

对每个方向:

```text
x_nodes = [0, 0.01, 0.05, 0.10, 0.20]
y_nodes = [0, C_dir(0.01), C_dir(0.05), C_dir(0.10), C_dir(0.20)]
AUC_dir = trapezoidal_integral(y_nodes over x_nodes) / 0.20
```

实现等价于 `numpy.trapezoid(y_nodes, x_nodes) / 0.20`。主集合级标量为 `AUC_macro = mean(AUC_RIGHT, AUC_LEFT, AUC_DOWN, AUC_UP)`。必须同时报告四方向 AUC 与所有冻结节点,不得只报告 macro 或定义其他正式 AUC。

### 0.5 度量口径

- 主终点:**验证集** top-1(开发期与预注册主终点)
- 测试集冻结,待最终模型与分析方案定稿后一次性使用
- 尾窗:epoch 80–100 均值

### 0.6 通道轮换(Latin square)

diverse 条件下路径与通道的对应关系随 seed 循环轮换:

| seed | ch0 | ch1 | ch2 | ch3 |
|---|---|---|---|---|
| 0 | P1 | P2 | P3 | P4 |
| 1 | P2 | P3 | P4 | P1 |
| 2 | P3 | P4 | P1 | P2 |
| 3 | P4 | P1 | P2 | P3 |

single 条件下四通道使用同一路径,无需轮换;但**随机 single 的代表路径随 seed 轮换**(见表内 path instance 列),使每条 `R^s_i` 恰好出现一次。

---

## 1. P0-B 配置表

**固定参数(所有行相同):** `--arch channel_split --dataset cifar10 --d-model 256 --n-layers 2 --block mamba --effective-batch 128 --epochs 100 --warmup-epochs 5 --base-lr 0.001 --weight-decay 0.05 --grad-clip 1.0 --pos-mode xy_learned --num-workers 4`

**reliance 操纵:** `R_low = grid8 (patch4, L=64)`;`R_high = grid32 (patch1, L=1024)`
**block 类型:** 仅 mamba(GRU 留待主实验作 backbone 稳健性检查)
**seed:** 0,1,2,3

| # | exp_id | 条件类型 | 路径组合(ch0..ch3) | path instance(按 seed) | 通道轮换 | 主对比归属 | 确认性? |
|---|---|---|---|---|---|---|---|
| 1 | GEO_SG1 | 几何单一 | G1,G1,G1,G1 | 固定 | 否 | ①② | 否 |
| 2 | GEO_SG2 | 几何单一 | G2,G2,G2,G2 | 固定 | 否 | ①②③ | 否 |
| 3 | GEO_SG3 | 几何单一 | G3,G3,G3,G3 | 固定 | 否 | ①②④ | 否 |
| 4 | GEO_SG4 | 几何单一 | G4,G4,G4,G4 | 固定 | 否 | ①② | 否 |
| 5 | GEO_DIV | 几何多样 | G1,G2,G3,G4 | 固定 | **是** | ② | 否 |
| 6 | RND_S1 | 随机单一(套1) | R¹ᵢ ×4 | seed0→R¹₁, s1→R¹₂, s2→R¹₃, s3→R¹₄ | 否 | ①② | 否 |
| 7 | RND_S2 | 随机单一(套2) | R²ᵢ ×4 | 同上,套2 | 否 | ①② | 否 |
| 8 | RND_S3 | 随机单一(套3) | R³ᵢ ×4 | 同上,套3 | 否 | ①② | 否 |
| 9 | RND_D1 | 随机多样(套1) | R¹₁,R¹₂,R¹₃,R¹₄ | 固定 | **是** | ② | 否 |
| 10 | RND_D2 | 随机多样(套2) | R²₁..R²₄ | 固定 | **是** | ② | 否 |
| 11 | RND_D3 | 随机多样(套3) | R³₁..R³₄ | 固定 | **是** | ② | 否 |
| 12 | LOC_S | LMTO single | Lᵢ ×4 | seed0→L₁, s1→L₂, s2→L₃, s3→L₄ | 否 | ②⑤ | 否 |
| 13 | LOC_D | LMTO diverse | L₁,L₂,L₃,L₄ | 固定 | **是** | ②⑤ | 否 |

**主对比编号:**
- ① reliance × locality:`mean(GEO_S*) − mean(RND_S*)` 在两档 reliance 下的差
- ② reliance × geometric-vs-random performance gain:`P_G − P_R`,其中 `P_G = GEO_DIV − mean(GEO_S*)`,`P_R = mean(RND_D*) − mean(RND_S*)`
- ③ traversal polarity:`GEO_SG1 − GEO_SG2`(无向 d_seq 严格相同)
- ④ scan axis:`GEO_SG1 − GEO_SG3`(AxisBias 符号相反)
- ⑤ canonical-orbit specificity control:`contrast_5 = P_G − P_LMTO`,其中 `P_G = GEO_DIV − mean(GEO_SG1, GEO_SG2, GEO_SG3, GEO_SG4)`,`P_LMTO = LOC_D − LOC_S`。`contrast_5 > 0` 仅表示 canonical raster symmetry orbit 的多路径增益大于该预先冻结、三统计量 locality 匹配的 topology-perturbed symmetry orbit;`contrast_5 ≈ 0` 仅表示在该辅助控制下未发现额外优势。无论结果如何,均不能自动写成“全部收益来自 locality”,且不能单独排除完整 locality 分布、AxisBias 幅度、polarity 或 coverage 差异。

**运行数:** 13 条件 × 2 reliance × 4 seed = **104 runs**

**输出目录命名:** `outputs/p0b_{exp_id}_{reliance}_seed{S}/`

---

## 2. 成本估算

基于批次 C(d256, mamba, 100 epoch)的实测:24 组合 × 5 seed = 120 runs ≈ 1–1.3 天(5 进程并行,util ~98%)。

| 项 | grid8 (L=64) | grid32 (L=1024) |
|---|---|---|
| 单 run 约耗时 | ~20 min | ~40 min |
| run 数 | 52 | 52 |
| 小计 | ~17 GPU·h | ~35 GPU·h |

**合计 ≈ 52 GPU·小时;5 进程并行下墙钟约 1–1.5 天;AutoDL 4090 按量计费约 ¥150–250。**

**显存:** 批次 C d256 单进程 698 MiB,5 进程约 3.5 GB / 24 GB。grid32 会更高但远未触顶。

---

## 3. P0-B 的成功标准(不要求效应显著非零)

**这是工程与统计可行性检查,不是提前筛掉零结果。** 若 GEO_SG1 与 GEO_SG2 无差异,这可能是真实科学结果。

**检查项:**

| # | 检查 | 通过条件 |
|---|---|---|
| C1 | 路径合法性 | 全部路径:长度=H×W、无重复、无遗漏;声称连续者连续步为四邻域 |
| C2 | 极性对照前提 | G1 与 G2 的**无向** d_seq 逐对严格相同 |
| C3 | 轴向对照前提 | G1 与 G3 的 AxisBias 符号相反 |
| C4 | 有向覆盖拆分成立 | `AUC_macro({G1,G2,G3,G4}) > AUC_macro({G1,G3})`;不得要求每个已覆盖方向严格更高,RIGHT/DOWN 可相等,增益来自补足反向总体 |
| C5 | LMTO locality 三统计量匹配成立 | 每条 L 的 d_seq mean/p50/p90 落在 G 的 ±10% 内;不匹配完整分布、p95/max、AxisBias、polarity 或 coverage |
| C6 | 计算匹配 | 13 条件参数量、FLOPs、训练计划完全一致 |
| C7 | 训练稳定 | 无发散、无 NaN、尾窗曲线平稳 |
| C8 | 方差估计 | 输出五个主对比的效应量点估计与 CI 初步范围 |

**C8 不设"CI 宽度可接受"的硬门槛**——4 个观测对 CI 宽度的估计不稳定。其用途是**为完整实验的功效与预算规划提供方差量级**。

---

## 4. P0-B 的限制声明(写入预注册)

> P0-B 的随机单一路径使用预先随机抽样,仅用于估计运行方差与初步效应,**不作为组成路径严格平衡的确认性分析**。
>
> 4 个 seed 下,每条 `R^s_i` 仅对应一个训练 seed,因此 path instance 与 training seed 在随机单一条件内仍高度混合,**无法估计 path × seed 交互**。
>
> S=3 不足以估计 path-set 随机效应方差。path set 作**固定阻断因素**处理,结论限定为"在三套预定义随机路径集合中一致"。
>
> 仅一套几何集合 G,结论限定为"所研究的四条几何路径"。

> LMTO 是为 P0-B feasibility pilot 预先冻结的辅助控制,不是 random Hamilton locality control、perfect locality matching、axis/polarity/coverage-matched control,也不证明 locality 单独解释效应。

---

## 5. 前置检查(P0-B 发车前必须完成并记录)

| # | 检查项 | 若不通过的后果 |
|---|---|---|
| F1 | SSM 是单向还是双向 | 若已含双向处理,**主对比③(极性)天然无效** |
| F2 | 状态在哪里初始化/重置 | 影响路径语义 |
| F3 | 每行之间是否重置状态 | 若重置,raster 与"逐行独立扫描"是两个自变量 |
| F4 | 反向扫描后如何恢复空间位置 | 影响 G2/G4 的实现正确性 |
| F5 | 通道融合是否对方向对称 | 若不对称,通道轮换是必需的(已在 §0.6 处理) |

---

## 6. 不在 P0-B 内的项目(及原因)

| 项目 | 原因 | 何时做 |
|---|---|---|
| GRU backbone | 单 backbone 足以做可行性检查 | 主实验 |
| d64 尺度 | 批次 C 已证 d64 欠拟合 | 不做 |
| 拓扑分析集合(serpentine/hilbert/block_shuffle) | 独立实验,不进主析因 | P0-C |
| **第二种 reliance 操纵(λ 门控局部分支)** | 需先实现;见 §7 | **P0-C(必做)** |
| 严格确认性方案(5+5S=20 条件) | 待 P0-B 方差估计后决定 | 主实验 |
| P1-A 合成方向性任务 | 需先冻结标签规则等七项 | P1-A |
| 标准 VMamba 外部锚点 | 见 §8 | 主实验后期 |

---

## 7. ★ 第二种 reliance 操纵的实现规格(P0-C,应导师建议具体化)

**不改卷积核尺寸或层数**(会同时改变参数量、FLOPs、深度)。采用**固定计算的门控局部分支**:

```
z_λ = Norm[ (1−λ)·x + λ·L(x) ]
```

- `L` 为固定结构的局部混合算子(扫描前)
- **所有 λ 条件都实际计算 `L(x)`** → 参数量、名义 FLOPs、网络深度完全一致
- `λ` 在训练前固定,**不得根据结果调整档位**
- 融合后使用相同归一化,降低幅值混淆

**档位:** `λ ∈ {0, 0.5, 1}`

**预注册须同时规定:**
- λ 是否所有层相同(**答:是**)
- 是否只调扫描前局部分支(**答:是**)
- 是否根据结果调整档位(**答:否**)
- 如何检查它确实改变了扫描消融敏感性(**答:报告各 λ 下 scan branch 消融引起的性能下降,即 SRI**)

**表述规范:** λ 操纵与 patch-size 操纵是**两个独立的 operationalizations**,均不等于"纯 reliance"。**只有两者都得到相同的 `Reliance Manipulation × Scan Order` 交互,才构成 convergent evidence。**

---

## 8. ★ 标准 VMamba 外部锚点(应导师建议,从"可延后"升为最小核心)

**不复制完整矩阵。** 规格:

- 标准或公开的 VMamba 实现
- 数据:Tiny-ImageNet / ImageNet-100,或 CIFAR 分辨率适配版
- reliance:仅低/高两档
- 路径条件:仅 2–3 个最关键(建议 `GEO_DIV` / `GEO_SG1` / `RND_D`)
- seed:3
- 定位:**外部一致性验证**,非确认性分析

**理由:** 自制小模型 + CIFAR + 合成数据的组合可能被评价为玩具设定。对更强会议而言,标准架构的小规模复现比再增加十种扫描路径更重要。

---

## 9. ★ P1-A 合成任务的负对照与等难度要求(应导师建议)

除已列七项冻结(标签规则、α 数学含义、难度匹配、非方向捷径控制、canonical 主轴、固定生成种子与划分、旋转后标签是否不变)外,增加:

**负对照:** `α=0` 时,方向扫描之间的期望差异应为零。

**等难度要求:** 不同 α 条件下,使用旋转不变的 oracle 特征或非扫描模型时,整体任务难度应近似;或通过调节非方向噪声,使 baseline accuracy 落在相近范围。

**目标结果不是单纯 `D_canonical > 0`,而是有序交互:**
```
D_{α=0} ≈ 0,  D_{α=medium} > 0,  D_{α=high} > D_{α=medium}
且该趋势在高 reliance 下更陡
```

---

## 10. ★ PSI 处方性定位的三层表述(应导师建议)

| 层级 | 表述 | 需要的证据 |
|---|---|---|
| L1(主文已支持) | PSI 是 scan-reliance 的**内部描述指标** | P1–P3 |
| L2(部分验证后) | PSI 是扫描敏感性的**候选预测指标** | + P4b/P5 |
| L3(留出验证后) | PSI 是设计扫描路径前的**低成本诊断工具** | + 留出预测 |

**留出预测的操作化:**
```
S_order = max_j Perf_j − min_j Perf_j        (跨扫描路径的性能跨度)
检验:PSI 能否预测 S_order > δ  (δ 取有实际意义的阈值,如 1 pp)
```
**阈值必须在一部分模型配置上设定,在留出的模型/宽度/数据集上检验。不得在同一批配置上选阈值并报告效果。**

报告:相关性、留出预测误差、二分类 AUROC 或精确率、高/低 PSI 区间的平均 scan-order spread。

**若不做 P4b/P5 或留出预测,practical guidance 只能写成"基于实验观察得到的设计建议",不得把 PSI 升为已验证的处方工具。**

---

## 11. 文献补充(读三篇原文后新增)

### 11.1 ★ SF-Mamba Table 4:本文中心假设的未被识别实例

SF-Mamba(arXiv 2603.16423, 2026-03)Table 4 在两个宏架构上比较七种扫描设计:

| | MambaVision-T(含 Attention) | 同架构去掉 Attention |
|---|---|---|
| 扫描设计带来的 top-1 跨度 | **0.3 pp** | **0.9 pp**(含参数不匹配的 Vim block 行则为 1.5 pp) |

**当 Attention 可承担全局混合时,扫描设计几乎不重要;当 SSM 独自承担时,影响放大 3–5 倍。** 这正是 scan reliance 门控,但 SF-Mamba 将该表用于论证多方向扫描的效率劣势,**未识别、未检验、未给出理论**。

**必须引用并定位为:**
> An unremarked instance of this interaction appears in SF-Mamba's Table 4, where the spread across scan designs is roughly three times larger once attention is removed. We make this dependence explicit and test it under controlled reliance manipulation.

**不构成抢先的理由:** 仅二值对照(有/无 Attention),无连续 reliance 旋钮,无路径组成匹配,无 locality/axis/polarity/coverage 分解。

### 11.2 EQ-VMamba:数据集对称性与设计收益的关联

EQ-VMamba(arXiv 2603.09138v2, 2026-04)在 §IV-B 观察到:等变架构在遥感数据上收益远大于自然图像,归因于"自然图像通常遵循标准正立朝向、缺乏全局旋转对称",而遥感图像"天然具有更强旋转对称"。其结论节把**"建立数据集层面对称性的量化度量,以预测等变设计的潜在收益"列为 open problem**。

**与本文的关系:** 这是"标签相关各向异性"因素的另一侧面表述,且其 open problem 与本文的 PSI 诊断工具构想属同类问题。应在 discussion 中对话,不得声称本文首次提出数据依赖性。

### 11.3 VNCT:scan invariance vs direction invariance 的概念区分

VNCT(arXiv 2607.03589v2, 2026-07)§A.5 明确区分 **direction invariance**(对扫描朝向不敏感,可由多方向平均近似达成)与 **scan invariance**(对任意 token 置换不变,强得多)。

**与本文五概念表的关系:** 该区分与本文的 multi-path coverage / path topology 有交叉但不重合,应在概念定义节引用以划清边界。

---

## 12. 下一步

1. 完成 §5 前置检查 F1–F5,记录结果
2. 实现 G1–G4、R^s、L 路径 + §0.4 度量 + §0.6 通道轮换,通过 C1–C5
3. 本表定稿后写入 P0-B 预注册并 commit
4. 发车 P0-B(104 runs,约 1–1.5 天)
5. 依 C8 的方差估计决定主实验规模(严格型 20 条件 vs 预算型 11 条件;seed 4 vs 8)
