# 03 — 证据台账

**版本:** 2026-07-16(第四次:追加 §8b/§8c)
**用途:** 记录每一条证据、它的强度、以及**三次已发生的推翻**。
**规则:** 反例与负结果与正结果同等保存。任何被推翻的结论保留在此,不删除。

---

## 0. 强度分级

| 级别 | 含义 |
|---|---|
| **铁证** | 效应量巨大,不依赖统计检验,单 seed 即可确信 |
| **强** | 效应显著,但依赖统计检验;需多 seed 确认 |
| **中** | 方向明确,但机制解释尚有替代可能 |
| **弱 / 待定** | 单 seed,形状不稳,或存在已知混杂 |
| **已作废** | 曾被相信,已被推翻。保留以记录教训 |

---

## 1. 【已作废】"Mamba 长程 carry 足够,方向无信息损失"

**曾经的结论:** VerticalCarry 实验中 row = col = 100%,shuffle 塌到 52.7%。
据此认为 Mamba 太强,任何方向都能恢复信息,"补充方向信息"这一说不成立。

**为何作废:** 该实验使用 **mean-pooling readout**,存在捷径 ——
分类器可以直接在全图定位 source patch,**无需沿扫描序列把信息传到 target token**。
row = col = 100% 是捷径造成的假象,不是 carry 能力的证明。

**教训(已写入 AGENTS.md 第 0 节):**
> 任何 carry / 传递类实验,readout 必须是 target-token,绝不能用 mean-pooling。
> 一个"太漂亮"的对称结果(row = col = 100%)本身就是捷径的信号。

**注意:此禁令仅针对 synthetic carry 实验。真实图像分类使用 spatial mean-pool +
linear head 是标准做法,不受此限。**

---

## 2. 【铁证】方向信息缺口真实存在,且是灾难性的

修正 readout 为 target-token 后(Stage1A):

| Mamba, VerticalCarry, single_patch | col(匹配) | row(不匹配) |
|---|---|---|
| grid 16 | **1.000** | 0.507 |
| grid 24 | **1.000** | 0.511 |
| grid 32 | **1.000** | 0.519 |

**不匹配的扫描方向不是"略差",是完全失败,退化为随机猜测。**

**这与 §1 的旧结论完全相反。** 方向信息缺口不但存在,而且是彻底的。

**地位:** 这是全文的支点。**现有文献中没有任何等价实验。**

---

## 3. 【铁证】容量解释了 full-branch 增益的绝大部分

CIFAR-10, d_model=64, 参数量匹配对照:

| 对比 | GRU | Mamba |
|---|---|---|
| `row → real_4dir`(总增益,含 4× 参数) | +3.32pp | +2.72pp |
| `row → same_row_4`(纯容量/多路径) | +3.29pp | +2.07pp |
| **`same_row_4 → real_4dir`(纯方向)** | **+0.03pp** | **+0.65pp** |

**方向最多解释总增益的四分之一。**

**注意:`same_row_4` 与 `real_4dir` 参数量、分支数、融合方式完全相同,
唯一变量是扫描方向。因此这个差值按定义就是方向效应,不可能是别的。**

**术语纠正:不得称多分支效应为 "ensemble"。** 四个分支是**联合训练**、
在**特征层**融合的,不是独立训练 + 预测层平均。正确表述是
**"多分支容量与结构效应"**,不过度声称机制。

---

## 4. 【强】方向贡献非零,但很小

seed 0, 100 epoch + cosine, 尾窗 80–100:

| block | grid | delta_direction |
|---|---|---|
| GRU | 8 | +0.72 ± 0.11 pp |
| Mamba | 8 | +0.93 ± 0.12 pp |

McNemar 检验(seed 0, grid8, Mamba, Stage1C checkpoint):
769 修复 / 673 破坏,χ² = 6.39,**p = 0.0115** —— **名义显著,非零。**

**不得写成"方向无贡献"。** 正确表述:
> 方向贡献小但大概率非零(约 0.6–0.9pp),而容量贡献 2.07pp。
> 方向最多解释四分支增益的四分之一。

---

## 5. 【中】表征去相关假说被排除

CKA 分支相似度(seed 0,四个分支两两 off-diagonal CKA 均值):

| block | grid | CKA(real) | CKA(same) | 去相关增益 | delta_direction |
|---|---|---|---|---|---|
| GRU | 8 | 0.467 | 0.482 | +0.015 | +0.72pp |
| GRU | 16 | 0.463 | 0.481 | +0.017 | +0.11pp |
| GRU | 32 | 0.386 | 0.396 | **+0.010(最小)** | **+3.65pp(最大)** |
| Mamba | 8 | 0.327 | 0.345 | +0.017 | +0.93pp |
| Mamba | 16 | 0.285 | 0.345 | **+0.060(最大)** | **+0.52pp(几乎最小)** |
| Mamba | 32 | 0.271 | 0.300 | +0.029 | +3.93pp |

**去相关增益 vs delta_direction 的相关系数 r = −0.22(n=6)—— 不相关,且符号为负。**

两个极端直接互相打脸:
- `gru/grid32`:去相关增益最小,准确率增益最大
- `mamba/grid16`:去相关增益最大,准确率增益几乎最小

**结论:机制 B(去相关 → 平均降噪)被排除。机制 A(信息互补)是唯一还站着的解释。**

**附带发现:** `same_row_4` 的四个分支(扫描方向完全相同,仅随机初始化不同)
CKA 已达 0.48 / 0.34。换成四个真正不同的几何方向,CKA 只再降 0.015。
**扫描方向对"分支学到什么"的影响,远小于随机初始化带来的差异。**
但就是这一点点表征差异,在高扫描负载下买到了显著的准确率增益 ——
说明那点增益不是"多样性红利",而是特定的、任务关键的信息。

**保留:** CKA 是钝器,可能对"少量但关键的信息差异"不敏感。因此严格说:
**去相关假说被排除,但信息互补假说是"剩下的那个",不是被直接证实的。**
直接证实需要掩码因果探针(见 §8)。

---

## 6. 【弱 / 待定】方向贡献随扫描负载上升

seed 0, cifar10:

| 配置 | L | GRU order | GRU delta | MB order | MB delta |
|---|---|---|---|---|---|
| grid8 (patch4) | 64 | 5.32 | +0.72 | 4.86 | +0.93 |
| grid16 (patch2) | 256 | 7.80 | **+0.11** | 7.24 | **+0.52** |
| grid32 (patch1) | 1024 | 10.57 | **+3.65** | 11.90 | **+3.93** |

**问题一:形状不是单调的。** 两个模型都在 grid16 下凹,然后在 grid32 暴涨 7 倍。
数据更像一个**阈值**(order < 8 时 delta ≈ 0–1;order > 10 时跳到 ~3.9),
但 3 个点、1 个 seed,**分不清阈值、单调、还是噪声**。

**问题二:自变量不是"扫描长度"。** 图像尺寸固定 32×32 时,
`grid = 32 / patch_size` —— **扫描长度与局部聚合强度由同一个旋钮控制,物理上不可分离。**
论文中**绝不能写"扫描长度是因果变量"**,必须统一称为**"扫描负载"**
(空间混合在 patch embedding 与 SSM 之间的分工)。

**问题三:order_utilization 不是发现,是操作检验。**
`patch_size=1` 时每个 token 就是一个像素,打乱顺序等于把图像彻底粉碎;
`patch_size=4` 时每个 token 是 4×4 块,内部结构还在,打乱伤害小得多。
**order 随 patch 变小而上升几乎是数学必然。** 它的正确身份是 manipulation check
(验证 patch_size 这个旋钮确实转移了空间混合的负担),**不是结论。**

---

## 7. 【已作废的对照】up64 双线性上采样

**设计意图:** cifar10/grid32 用 `patch_size=1`,是 1×1 卷积。担心 +3.9pp 是
"退化 patch embed"的伪影。用 `cifar10_up64`(双线性上采样到 64)+ `patch_size=2`
做对照,预期保持相同的信息覆盖但用真实的 2×2 卷积核。

**预注册判定:** delta < +1.5pp → 判定为伪影,headline 作废。

**结果:** GRU +0.28pp, Mamba +1.65pp。**按字面判定:证伪。**

**但这个对照本身是无效的。** manipulation check 同时塌了:
order_utilization 从 ~11 掉到 ~7.7。

**原因:双线性插值会混合邻居像素。** 32→64 上采样后,每个 2×2 patch 覆盖的四个值
分别混合了原图相邻像素。**插值把局部聚合偷偷还给了卷积。**
这个对照没有控制住它声称要控制的变量。

**因此:该测试对原假设不提供信息(manipulation check 失败),既不证实也不证伪。**

**⚠️ 动机性推理警告:** 上述"对照无效"的辩护,正是一个有动机的研究者会用来
逃避证伪的说辞。必须记录反面证据:
- 关系不单调(GRU 从 order 5.3 到 7.8,delta 反而下降)
- 同样 order ≈7.7,GRU 是 +0.28,Mamba 是 +1.65,差 6 倍。"order 预测 delta"并不紧
- 全部单 seed

**辩护的依据(供读者自行判断):**
- `order_utilization` 是这个实验**之前**就定义好的 scan-load 度量,不是事后发明
- 双线性插值混合邻居是**数学事实**,不依赖任何实验数据
- 理论做出了新的可证伪预测(delta 跟着 order 走而非跟着 L 走),
  up64 那个点确实落在预测位置(order 7.5–8.0,与 cifar10/grid16 的 7.2–7.8 相当;
  delta 0.28/1.65 vs 0.11/0.52,同一量级)

**这个失败的对照留下了一个有价值的副产品:**
`up64_bl/grid32` 与 `cifar10/grid32` **序列长度完全相同(L=1024)、token 数相同**,
唯一区别是每个 token 含不含邻居信息。
**结果:order 掉 3pp,delta 掉 2–3pp。→ 序列长度不是驱动因素,局部信息才是。**
(此解读为事后分析,需 5-seed 与掩码探针验证,不得作为预注册结论使用。)

---

## 8. 【已排除的对照】最近邻上采样 —— 数学恒等式,不必跑

曾计划用最近邻上采样 + `patch_size=2` 作为"零邻居信息 + 真实卷积核"的对照。

**已证明该对照与 `cifar10/patch1` 数学等价:**

NN 上采样 2× 后,每个 2×2 patch 内是**同一像素的四份拷贝**。
2×2 卷积作用其上:

```
Σ_{a,b} W[o,c,a,b] · v[c] = (Σ_{a,b} W[o,c,a,b]) · v[c]
```

**即权重求和后的 1×1 卷积。** 数值验证:最大绝对差 2.4e-07(浮点误差)。
初始化分布也一致(kaiming 按 fan_in 缩放,四位置求和后方差恰好抵消)。

**推论:"1×1 卷积是退化架构"这个反驳本身不成立。**
`patch_size=1` 不是病态,它**就是"零局部聚合"这个处理条件本身**,是自变量而非伪影。
审稿人的这一质疑可以用两行数学直接答掉,不需要实验。

**教训(已写入 AGENTS.md 第 8 节):**
> 提出任何对照前,先检查它是否与已有条件数学等价。同义反复的对照跑了等于没跑。

---

# §8b 批次 C:channel-split 2×2 析因(定稿,d64+d256 双尺度 5-seed)

**版本:2026-07-19 定稿。取代此前的 GRU 单 seed pilot 数字。**
**本节 CI 全部由 2026-07-19 重算(t₄,.₉₇₅=2.776,尾窗 80-100,单位 pp),数据源为十个完整的 stage1_history.csv(各 2401 行 = 24 组合 × 100 epoch,已入库)。**
**作废说明:此前 §8b 的 d64 数字源自 csplit_d64_seed0 的一个残缺 pilot 文件(1401 行,仅 14/24 个组合完成),该文件后被完整重跑覆盖(经 diff 验证为干净重跑,旧行 1400/1400 全部不存在于新文件,无 RESUME 混批)。故旧 §8b 中所有 d64 数字(如 MAMBA32 interaction [2.01,2.20])一律作废,以本节为准([1.98,2.30])。**

## 8b.1 设计回顾

2×2 析因:{结构化行序, 随机乱序} × {单一模式×4, 多样×4},四变体 real_4dir / same_row_4 / same_perm_4 / rand_perm_4,恒参数、恒通道宽度(C/4)、恒序列长度,唯一变量是四路 SSM 的 token 顺序。三个析因量:

- structure = {real, same_row} − {rand_perm, same_perm}
- diversity = {real, rand_perm} − {same_row, same_perm}
- interaction = (real − same_row) − (rand_perm − same_perm)

预注册判据(§8b 旧版事先定死,口径为 train_acc):
- 判据1:structure 随 grid 8→16→32 单调
- 判据2:grid32 diversity 5-seed CI 下界 > 0
- 判据3:grid32 interaction 5-seed CI 下界 > 0

## 8b.2 拟合状态(判读前提)

| 尺度 | 形态 | real 变体 grid32 |
|---|---|---|
| d64 | 欠拟合(train≈test, gap −0.7~+0.9pp) | gru 69.1/68.5, mamba 72.1/71.2 |
| d256 | 过拟合侧(gap +4.8~+12.4pp) | gru 89.6/79.6, mamba 92.5/80.1 |

d256 train_acc 达 85–93%,**欠拟合疑虑排除**:d64 结果不是欠拟合伪影。但 d256 进入过拟合域后 train/test 两口径开始分家(见 8b.5),这是 d64 不存在的现象。

## 8b.3 主结果表(5-seed mean [95% CI], pp)

### d64 TRAIN(预注册主口径)

| | structure | diversity | interaction |
|---|---|---|---|
| gru 8 | +3.65 [3.31, 3.99] | −0.13 [−0.54, 0.27] | −0.85 [−1.58, −0.12] |
| gru 16 | +5.71 [5.50, 5.92] | −0.02 [−0.31, 0.27] | −0.07 [−0.90, 0.77] |
| gru 32 | +9.78 [9.50, 10.05] | +1.85 [1.69, 2.01] | +3.60 [2.97, 4.22] |
| mamba 8 | +4.30 [3.65, 4.94] | −0.27 [−0.91, 0.36] | −0.36 [−0.76, 0.04] |
| mamba 16 | +6.59 [6.19, 6.99] | +0.09 [−0.30, 0.48] | +0.04 [−0.52, 0.61] |
| mamba 32 | +12.97 [12.61, 13.32] | +1.00 [0.71, 1.29] | +2.14 [1.98, 2.30] |

### d64 TEST

| | structure | diversity | interaction |
|---|---|---|---|
| gru 8 | +3.56 [3.08, 4.04] | −0.01 [−0.51, 0.49] | −1.12 [−1.81, −0.43] |
| gru 16 | +5.43 [4.74, 6.11] | −0.36 [−0.72, 0.01] | −0.63 [−1.71, 0.46] |
| gru 32 | +9.30 [8.91, 9.69] | +1.34 [1.09, 1.60] | +2.67 [2.11, 3.23] |
| mamba 8 | +3.58 [2.74, 4.42] | −0.17 [−1.00, 0.66] | −0.07 [−0.59, 0.45] |
| mamba 16 | +6.00 [5.59, 6.41] | +0.01 [−0.67, 0.69] | +0.07 [−1.11, 1.26] |
| mamba 32 | +12.25 [11.79, 12.71] | +1.24 [0.90, 1.57] | +2.32 [1.52, 3.12] |

### d256 TRAIN(预注册主口径)

| | structure | diversity | interaction |
|---|---|---|---|
| gru 8 | +7.76 [7.51, 8.01] | +0.02 [−0.14, 0.18] | −0.60 [−1.32, 0.12] |
| gru 16 | +8.66 [7.99, 9.33] | +0.31 [0.03, 0.59] | +0.46 [−0.01, 0.94] |
| gru 32 | +11.90 [11.26, 12.54] | +1.97 [1.76, 2.17] | +4.10 [3.46, 4.74] |
| mamba 8 | +7.19 [6.87, 7.51] | +0.13 [−0.25, 0.51] | −0.21 [−0.82, 0.40] |
| mamba 16 | +9.41 [9.01, 9.81] | +0.21 [−0.09, 0.51] | +0.06 [−0.96, 1.09] |
| mamba 32 | +16.18 [14.54, 17.82] | +0.55 [0.10, 1.00] | **+0.43 [−0.53, 1.39]** |

### d256 TEST

| | structure | diversity | interaction |
|---|---|---|---|
| gru 8 | +4.89 [4.62, 5.17] | +0.31 [−0.01, 0.62] | −0.58 [−1.44, 0.27] |
| gru 16 | +6.26 [5.52, 7.00] | +0.18 [−0.28, 0.63] | −0.12 [−0.39, 0.15] |
| gru 32 | +11.94 [11.02, 12.86] | +1.60 [1.17, 2.02] | +3.04 [2.50, 3.59] |
| mamba 8 | +3.71 [2.94, 4.47] | +0.08 [−0.14, 0.29] | +0.50 [−0.13, 1.14] |
| mamba 16 | +6.39 [6.08, 6.71] | +0.36 [−0.00, 0.72] | +0.44 [−0.10, 0.97] |
| mamba 32 | +13.66 [12.93, 14.39] | +1.91 [1.08, 2.74] | +3.62 [3.14, 4.09] |

## 8b.4 预注册判据判定

| 判据 | d64 gru | d64 mamba | d256 gru | d256 mamba |
|---|---|---|---|---|
| 1. structure 单调 | ✅ | ✅ | ✅ | ✅ |
| 2. grid32 diversity 下界>0 | ✅ (1.69) | ✅ (0.71) | ✅ (1.76) | ✅ (0.10) |
| 3. grid32 interaction 下界>0 | ✅ (2.97) | ✅ (1.98) | ✅ (3.46) | **❌ (−0.53)** |

**判据3 在 4 个预注册格中通过 3 个。** 未通过的 d256 mamba train 一格,伴随两个事实:(a) train_acc 已饱和至 92.5%,real 与 same_row 在 train 上仅差 0.77pp(92.50 vs 91.73);(b) 同格 test 口径下同一效应为 +3.62 [3.14, 4.09],且强于 d64。归因为**口径饱和(ceiling)而非机制缺失**。此归因为事后解释,判据判定本身按预注册口径原样记录,不以口径切换回补。

## 8b.5 泛化侧证据(TEST 口径,非预注册,探索性)

grid32 interaction 在 2 尺度 × 2 backbone 的 test 口径下四格全部显著为正,且 d64→d256 增强:

| | d64 | d256 |
|---|---|---|
| gru | +2.67 [2.11, 3.23] | +3.04 [2.50, 3.59] |
| mamba | +2.32 [1.52, 3.12] | +3.62 [3.14, 4.09] |

结合 8b.4:d256 mamba 中,单方向×4 已能将训练集拟合至与真四方向几乎同水平,但泛化差 3.7pp。**几何方向多样性在高负载下购买的是泛化能力;容量受限(d64)时该差异同时表现在拟合上,容量充足(d256)时仅存于泛化。** 此现象本身与"容量门控机制表达"的两因素账一致,但属事后观察。

**【2026-07-26 修订,不删改以上原文】** 以上"口径饱和(ceiling)"的归因已被批次 D(CIFAR-100)证伪:mamba32 在 train_acc 仅 78.28%(远低于本节 92.5% 的饱和水平,亦低于预注册 85% 参考线)时,train interaction 仍从早窗 +2.47 衰减至尾窗 +1.12,而同批 gru32 在更低的 72.53% train_acc 下完全不衰减。可见 train_acc 绝对水平不能预测该衰减,衰减仅与 `block=mamba` 且 `d=256` 相关。详见 §8d.5,替代假说见 §8d.6。本段原文作为该判断在当时证据下的合理归因予以保留,不作删除。

## 8b.6 几何性判决(interaction 分解,grid32)

within-structured 多样性收益 vs within-shuffle 多样性收益:

| | geom-div | shuffle-div |
|---|---|---|
| d64 gru train | +3.65 [3.18, 4.11] | +0.05 [−0.12, 0.22] |
| d64 mamba train | +2.07 [1.73, 2.41] | −0.07 [−0.34, 0.20] |
| d64 gru test | +2.67 [2.14, 3.20] | +0.01 [−0.07, 0.08] |
| d64 mamba test | +2.40 [1.74, 3.06] | +0.08 [−0.25, 0.41] |
| d256 gru train | +4.01 [3.53, 4.50] | −0.08 [−0.32, 0.15] |
| d256 mamba train | +0.77 [−0.05, 1.59] | +0.34 [−0.11, 0.78] |
| d256 gru test | +3.12 [2.54, 3.70] | +0.07 [−0.35, 0.49] |
| d256 mamba test | +3.72 [3.00, 4.44] | +0.10 [−0.88, 1.09] |

**shuffle-div 在全部八格中均为 0.1pp 量级且无一显著。** 多样性红利只在真几何方向出现,随机乱序的"多样性"不产生价值。几何性主张在除 d256 mamba train(ceiling 格)外的所有测量中成立。

## 8b.7 附带观察:低负载下 interaction 轻微为负

gru grid8 interaction 在两尺度、两口径下四格全负(−0.85 / −1.12 / −0.60 / −0.58),其中 d64 双口径显著(train −0.85 [−1.58, −0.12];test −1.12 [−1.81, −0.43])。含义:低负载下几何方向多样性相对乱序多样性轻微有害,越过负载阈值后翻正并放大——符号翻转形态比单纯 0→正更支持门控机制。**但此为 GRU 独有现象:mamba grid8 四个测量符号混乱(−0.36 / −0.07 / −0.21 / +0.50),无一致方向。写作中只可表述为 "a GRU-specific observation, not replicated in Mamba",不得升格为独立结论,也不得写成 "directionally consistent across backbones"。**

## 8b.8 本节 claim boundary

- 判据3 的完整表述:几何性主张在 d64 预注册口径成立、在 d256 非预注册(test)口径成立、在 d256 预注册口径不成立(ceiling)。三句缺一不可,不得简写为"全部通过"。
- diversity 判据在 d256 mamba 下界仅 +0.10,属脆弱通过,不得单独引用该格作为强证据。
- 全部结论限于 CIFAR-10 + channel-split + d∈{64,256} + 2层,不外推。
- test 口径分析属事后探索,论文中必须标注 exploratory,与预注册结果分栏呈现。

## 8c. 【教训】channel-split 不能用主实验的 delta 汇总口径

`tail_80_100_summary.csv` 由主实验(cloud_seed)脚本生成,按 block×grid 算
delta_direction。**套到 channel-split 输出上是错的** —— 它算出的"mamba grid8 +0.0013"
是脚本乱套的产物,与主实验 +0.58pp 不可比。channel-split 要的是四变体两两析因
(structure/diversity/interaction),不是 row-vs-col 的 delta。
**channel-split 一律用 `mamba_scan_study/analysis/analyze_csplit_factorial.py`,
勿信 summary CSV 里的数字。**

**教训(建议写入 AGENTS.md):** 复用汇总脚本前,先确认它的分组维度与新实验的
自变量匹配。channel-split 的自变量是 variant,不是 block×grid。

---

# §8d 批次 D:CIFAR-100 channel-split 2×2 析因(定稿,d256 5-seed)

**版本:2026-07-26 定稿。数据源为五个完整的 `csplit_c100_d256_seed{0..4}/stage1_history.csv`(各 2400 行 = 24 组合 × 100 epoch),复算脚本为 `mamba_scan_study/analysis/analyze_csplit_factorial.py` 同构口径,独立重算全部通过,容差 0.02pp。**

## 8d.1 设计与可比性

CIFAR-100,d256 单尺度,其余配置与批次 C d256 完全一致:`--n-layers 2 --effective-batch 128 --epochs 100 --warmup-epochs 5 --base-lr 0.001 --weight-decay 0.05 --grad-clip 1.0 --pos-mode xy_learned --num-workers 4`,四变体 real_4dir / same_row_4 / same_perm_4 / rand_perm_4,seed ∈ {0,1,2,3,4}。

全批实测 `micro_batch=128 / accum_steps=1`,与批次 C d256 逐格相同,故按 `docs/PREREG_batch_D_cifar100.md` §6,两批次效应量可直接并列,无需追加偏离记录。复现基线 commit 为 `02981d9`(依据 `P0B_PREREG_ADDENDUM_01.md` §6 第 3 条;其后四个纯文档/PSI-分析 commit `f8b4785`/`1a22e26`/`fe2e8f0`/`2f3606d` 不含训练代码,不影响本批次结果)。

数据完整性:120 cell(2 block × 3 grid × 4 variant × 5 seed),每 cell epoch 1..100 全覆盖,无 NaN/Inf。**本节禁止使用 `tail_80_100_summary.csv`**(分组维度与 channel-split 不匹配,见 §8c)。

## 8d.2 主结果表(5-seed mean [95% CI], pp)

### TRAIN(train_acc,预注册主口径,尾窗 80-100)

| | structure | diversity | interaction |
|---|---|---|---|
| gru 8 | +8.03 [7.37, 8.68] | −0.18 [−0.60, 0.23] | −0.68 [−1.31, −0.05] |
| gru 16 | +8.86 [7.80, 9.92] | −0.28 [−0.67, 0.12] | −0.74 [−1.57, 0.09] |
| gru 32 | +12.74 [11.14, 14.35] | +2.16 [1.82, 2.50] | +4.20 [3.60, 4.80] |
| mamba 8 | +8.75 [7.98, 9.52] | −0.67 [−0.88, −0.46] | −1.55 [−3.15, 0.05] |
| mamba 16 | +12.54 [11.55, 13.54] | −0.33 [−1.03, 0.36] | −0.67 [−2.24, 0.90] |
| mamba 32 | +18.71 [18.16, 19.26] | +0.73 [0.11, 1.35] | +1.14 [−0.07, 2.34] |

### TEST(test_acc,次要终点)

依 `docs/PREREG_batch_D_cifar100.md` §3,test 口径在发车前已声明为**次要终点(secondary endpoint)**,非事后探索。

| | structure | diversity | interaction |
|---|---|---|---|
| gru 8 | +4.40 [3.89, 4.91] | +0.39 [−0.22, 1.00] | +0.17 [−0.15, 0.50] |
| gru 16 | +5.87 [5.17, 6.57] | +0.26 [−0.38, 0.90] | +0.40 [−0.67, 1.47] |
| gru 32 | +12.70 [12.40, 13.01] | +2.01 [1.75, 2.27] | +3.71 [3.19, 4.22] |
| mamba 8 | +4.45 [3.61, 5.29] | +0.21 [−0.12, 0.55] | +1.17 [0.69, 1.65] |
| mamba 16 | +7.24 [6.67, 7.81] | +0.86 [0.61, 1.11] | +1.71 [1.29, 2.12] |
| mamba 32 | +14.33 [13.53, 15.13] | +2.48 [2.12, 2.84] | +4.69 [3.82, 5.57] |

### D4 分解(grid32,interaction 分解为 geom-div / shuffle-div)

| 口径 | | geom-div | shuffle-div |
|---|---|---|---|
| TRAIN | gru32 | +4.26 [3.63, 4.89] | +0.06 [−0.06, 0.18] |
| TRAIN | mamba32 | +1.30 [0.17, 2.43] | +0.16 [−0.31, 0.63] |
| TEST | gru32 | +3.86 [3.37, 4.35] | +0.16 [0.01, 0.30] |
| TEST | mamba32 | +4.83 [4.08, 5.58] | +0.14 [−0.15, 0.43] |

## 8d.3 判据判定

逐条对照 `docs/PREREG_batch_D_cifar100.md` §3:

- **判据 D1(structure 单调):** 两个 backbone 均**通过**——gru 与 mamba 的 structure 均随 grid 8→16→32 单调递增。
- **判据 D2(diversity 门控,要求 grid32 CI 下界>0 且 grid8 CI 跨 0):** GRU **通过**。MAMBA 的 grid32 子句通过(+0.73,下界 0.11),但 **grid8 子句不通过**——要求"grid8 diversity CI 跨 0",实测 −0.67 [−0.88, −0.46] 显著为负,不是跨 0。判**判据 D2 在 MAMBA 上 FAIL**。
- **判据 D3(interaction 门控,要求 grid32 CI 下界>0 且 grid8 CI 跨 0):** **两个 backbone 均 FAIL**。GRU 的 grid32 子句通过(+4.20,下界 3.60),但 grid8 子句不通过(−0.68 [−1.31, −0.05] 显著为负);MAMBA 的 grid8 子句通过(interaction −1.55 [−3.15, 0.05] 跨 0),但**grid32 子句不通过**(+1.14 [−0.07, 2.34] 跨 0,未能建立下界>0)。
- **判据 D4(几何性):** 两个 backbone 均**通过**——grid32 的 geom-div 全部显著为正,shuffle-div 全部跨 0。

## 8d.4 新现象:低负载端由零转负

批次 C(§8b.7)中 grid8 的 diversity/interaction 只是"跨 0",本批次在 CIFAR-100 低负载端变为**显著为负**(gru8 interaction −0.68 [−1.31, −0.05];mamba8 diversity −0.67 [−0.88, −0.46])。即低负载档,几何多样性不是无效而是**有代价**:四通道各自分摊一个方向,不如四路同向、把全部容量押在单一几何方向上。

此项与 §8b.7 记录的"GRU grid8 interaction 轻微为负"同向,且在 CIFAR-100 上强度更大、统计上更确定(批次 C 仅 GRU 一侧观察到,批次 D 在 GRU interaction 与 MAMBA diversity 两处均观察到)。回指 §8b.7:该节当时只敢写"GRU 独有观察,不得升格为独立结论";本批次数据支持将"低负载端有代价"扩展为跨 backbone 的一般现象,但仍限于本节判据 D2/D3 涉及的具体子句,不外推为更强表述。

## 8d.5 ceiling 假设被本批次证伪

§8b.4 对 d256 mamba32 判据3(train interaction 未过)给出的归因是:**train_acc 已饱和至 92.5%,real 与 same_row 在 train 上仅差 0.77pp,故判据未过是口径饱和(ceiling)而非机制缺失**。批次 D 提供了直接检验这一假设的机会。

批次 D 的 mamba32 尾窗(90-100)train_acc 仅 **78.28%**,远低于批次 C d256 mamba32 的 92.5%,也低于 `docs/PREREG_batch_D_cifar100.md` §4 分支 D 设定的 85% 参考线。**若 ceiling 解释成立,train interaction 在此低 train_acc 下不应再衰减。但实测 TRAIN interaction 仍从早窗(epoch 10-20)+2.47 衰减至尾窗(epoch 90-100)+1.12,衰减幅度 −54%。** 作为对照,同批 gru32 在 72.53% 的更低 train_acc 下完全不衰减(早窗 +3.59 → 尾窗 +4.20,不降反升)。

三个数据集/尺度的完整轨迹:

**本表所有量取 epoch 90–100 窗口,与 §8d.2 主结果表的尾窗(epoch 80–100)不同,因轨迹分析需要末端窗口而非整个尾窗。**

| 数据集/尺度 | block | interaction (ep10–20) | interaction (ep90–100) | train_acc (ep90–100) |
|---|---|---|---|---|
| C10 d64 | gru | +2.83 | +3.58 | 69.21% |
| C10 d64 | mamba | +1.08 | +2.17 | 72.35% |
| C10 d256 | gru | +4.22 | +4.10 | 89.94% |
| C10 d256 | mamba | +2.60 | +0.37 | 92.91% |
| C100 d256 | gru | +3.59 | +4.20 | 72.53% |
| C100 d256 | mamba | +2.47 | +1.12 | 78.28% |

**结论:train_acc 水平不能预测该衰减。** 衰减仅出现在 `block=mamba` 且 `d=256` 的两格(C10 d256 mamba、C100 d256 mamba),跨两个数据集各一格,与 train_acc 绝对水平(92.91% 与 78.28%,相差 14.6pp)无关。**原 ceiling 解释(§8b.4)按本批次数据判为不成立。**

## 8d.6 替代解释(标注为假说,非结论)

【强度:中 / 假说】Mamba 的长程 carry 在容量充足(d256)时,为单方向扫描(same_row_4)提供了拟合训练集的替代路径,使其能够逼近 real_4dir 的训练集表现,从而压低 train interaction;GRU 缺少这一路径。d64 容量不足以支撑该路径被利用,故 d64 两个 backbone 均不衰减。

**该路径不迁移到泛化。** 批次 D mamba32 的 TEST interaction 为 +4.69 [3.82, 5.57],未衰减;real_4dir 的 generalization gap(train−test)+23.21pp 反而**小于** same_row_4 的 +26.74pp——即单方向路径在训练集上追近了 real_4dir,但代价是更严重的过拟合。

此假说与本项目起点的 VerticalCarry 受控实验一致(§1:row=col=100%、shuffle 塌至 52.7%),可表述为:**carry 足以拟合,不足以泛化。** 两格证据(C10 d256 mamba、C100 d256 mamba)不足以支撑更强表述,仅作方向性假说记录。

## 8d.7 分支判定

按 `docs/PREREG_batch_D_cifar100.md` §4,本批次最接近**分支 B**(框架成立、阈值随任务难度移动),但移动方向是低负载端(grid8)由跨 0 转为显著为负,而非预注册设想的"阈值提前点亮(如 grid16 即点亮)"。

分支 D 的前件半满足(mamba32 train_acc 78.28% < 85%)而后件不满足(train 侧 interaction 仍衰减,与 gru32 不一致,train/test 结论不一致)——该组合正是 §8d.5 证伪 ceiling 解释的依据,而非分支 D 所设想的"顺带反证 ceiling、加强 §8b.5 可信度"的干净结果。

明确记录:**非分支 A(判据 D2/D3 均有 FAIL 子句,非三判据全过),非分支 C(structure 与 D4 两判据仍稳健通过,并非全面不显著)。**

## 8d.8 claim boundary

结论限于 CIFAR-10 / CIFAR-100 两个自然图像分类数据集、channel-split 架构、d256、本文冻结的四条几何路径(real_4dir / same_row_4 / same_perm_4 / rand_perm_4)。不得外推至其他任务域、其他架构(non-channel-split)或未测尺度(CIFAR-100 d64 未跑)。§8d.6 为假说,不为结论,不得在正文中作为已证实机制引用。

# §8e P0-B 可行性 pilot:104 runs 定稿(Mamba-only, d256, 4 seed)

**版本:2026-07-28 定稿。数据源为 `outputs/p0b_{exp_id}_{reliance}_seed{S}/metadata.json`，独立复算严格遵循 `P0B_PREREG_ANALYSIS_PLAN.md` §1–§3；所有报告量均为尾窗 epoch 80–100 的 run 内算术均值、seed 配对后聚合，单位为百分点(pp)。**

## 8e.1 完整性与溯源

104/104 runs 完成，零失败。完整析因为 13 个 `exp_id` × 2 个 reliance × training seed 0..3；每条 `validation_history` 恰为连续 epoch 1..100 的 100 行，字段集严格为 `{epoch, learning_rate, train_loss, train_accuracy, validation_loss, validation_accuracy}`，不含任何 test 字段。全批 `micro_batch=128`、`accum_steps=1`、`training_config.epochs=100`、`training_config.num_workers=4`，`git_commit` 前七位均为 `34edddb`；四个冻结源 SHA 与 ledger SHA 均在全批一致。grid8 参数量均为 282122、grid32 均为 282890，且各 grid 内 `architecture_signature` 唯一。

`git_dirty=true` 出现在 103/104 个 run；唯一的 false 是 canary（GEO_SG1 / R_low / seed0）。原因是旧 `.gitignore` 未覆盖仓库根 `outputs/`：canary 构造 metadata 时产物目录尚未生成，其后每个 run 均观察到前序未跟踪目录。本轮已在 `.gitignore` 追加根锚定的 `/outputs/`。该记录缺陷**不影响实验结论**，因为全批 `git_commit` 一致。

## 8e.2 五个主对比

口径依 `P0B_PREREG_ANALYSIS_PLAN.md` §1–§3：主终点为 validation accuracy；official test 在 P0-B 全程未实例化。每格 4 个 seed，区间为 `mean ± 3.182 × s / sqrt(4)`，其中 `s` 为跨 seed 样本标准差(ddof=1)。

| 对比/分解 | R_low | R_high | 交互(R_high−R_low) |
|---|---|---|---|
| ① mean(GEO_S*) − mean(RND_S*) | +3.66 [+3.26, +4.06] | +11.02 [+10.55, +11.49] | +7.36 [+7.14, +7.58] |
| ② P_G − P_R | +0.86 [+0.20, +1.52] | +4.16 [+3.84, +4.47] | +3.29 [+2.83, +3.76] |
| ③ GEO_SG1 − GEO_SG2 | −0.09 [−0.48, +0.29] | +0.30 [−0.66, +1.27] | +0.40 [−0.47, +1.27] |
| ④ GEO_SG1 − GEO_SG3 | −0.03 [−0.67, +0.62] | +2.96 [+1.97, +3.96] | +2.99 [+2.50, +3.48] |
| ⑤ P_G − P_LMTO | +0.45 [−0.21, +1.11] | +1.97 [+0.30, +3.65] | +1.52 [+0.06, +2.98] |
| P_G = GEO_DIV − mean(GEO_S*) | +0.85 [+0.55, +1.15] | +4.23 [+4.06, +4.40] | +3.38 [+2.92, +3.83] |
| P_R = mean(RND_D*) − mean(RND_S*) | −0.01 [−0.52, +0.50] | +0.07 [−0.24, +0.39] | +0.08 [−0.26, +0.42] |
| P_LMTO = LOC_D − LOC_S | +0.40 [−0.18, +0.98] | +2.26 [+0.44, +4.07] | +1.86 [+0.16, +3.55] |

## 8e.3 核心发现：多样性收益是几何专属的

`P_R` 在两档 reliance 下均为零：R_low −0.01 [−0.52, +0.50]，R_high +0.07 [−0.24, +0.39]；相对地，`P_G` 在 R_high 为 +4.23 [+4.06, +4.40]。因此，四条互不相同的**随机**路径相对四条相同随机路径没有收益，而四条互不相同的**几何**路径存在大幅收益。这将“多路径收益”与“集成效应”分离，是依赖 k=1 vs k=4 的既有扫描顺序比较无法区分的量。

## 8e.4 零对照与负载门控

③ polarity 在两档均跨零，符合 GEO_SG1 与 GEO_SG2 无向 `d_seq` 严格相同的设计预期，构成装置未虚报效应的行为证据。④ scan axis 在 R_low 为零、R_high 为 +2.96 [+1.97, +3.96]，交互 +2.99：同一几何差异在低扫描负载下不存在、在高负载下达到约 3pp，是负载门控最直接的单点演示。

## 8e.5 ⑤ 的结果与事前预期的偏离

`P0B_PREREG_ADDENDUM_01.md` §3.2 事前声明 `contrast_5 ≈ 0` 为预期。实测 R_low +0.45 跨零，符合该预期；R_high 为 +1.97 [+0.30, +3.65]，下界仅勉强越零。**4 seed 下该下界脆弱，不足以支撑强主张。**

更具信息量的是分解：`P_LMTO` 在 R_high 为 +2.26，说明 locality 匹配的 topology-perturbed 轨道同样获得多路径收益，仅小于 canonical 轨道。R_high 单路径侧 `mean(RND_S*)=63.27`、`LOC_S=70.19`、`mean(GEO_S*)=74.29`，故回收比例 `(LOC_S−RND_S)/(GEO_S−RND_S)=62.8%`。按配置表 §1⑤ 的措辞约束，不得写成“全部收益来自 locality”，亦不得单独排除完整 locality 分布、AxisBias 幅度、polarity 或 coverage 差异。

## 8e.6 事前终点选择的验证

该终点选择在 P0-B 任何 run 执行之前冻结。将同一批数据改用 `train_accuracy`，其余口径不变，结果如下。

| 量 | R_low train_accuracy | R_high train_accuracy |
|---|---|---|
| ① | +7.11 [+6.85, +7.37] | +16.25 [+14.44, +18.07] |
| ② | +1.57 [+0.78, +2.36] | +2.23 [+1.69, +2.77] |
| P_G | +1.50 [+0.75, +2.24] | +2.48 [+1.85, +3.10] |
| ④ | −0.14 [−0.83, +0.55] | −0.37 [−1.26, +0.53] |

R_high 尾窗 train_accuracy 分别为：GEO_SG1 91.63、GEO_SG3 92.00、GEO_DIV 94.57、RND_S1 75.90、LOC_S 82.40、LOC_D 85.57。结构组达 91.6–94.6%，超过 §8d.5 记录的 88–90% 阈值。若沿用批次 C/D 的 train_acc 口径，`P_G` 被压缩约 41%，且 ④ 会由 validation 的 +2.96 反转为 −0.37（符号相反且跨零）；分析计划 §1 的事前决定避免了“扫描轴向无影响”的错误结论。

## 8e.7 C8 方差估计与主实验含义

| 对比 | R_low sd | R_high sd | 交互 sd |
|---|---:|---:|---:|
| ① | 0.252 | 0.294 | 0.136 |
| ② | 0.413 | 0.198 | 0.293 |
| ③ | 0.245 | 0.608 | 0.545 |
| ④ | 0.405 | 0.624 | 0.309 |
| ⑤ | 0.416 | 1.054 | 0.916 |

跨 seed 方差很小，多数对比 sd 在 0.2–0.6 pp；①、②、④的主要效应按此量级在 2–3 个 seed 下即可达到 80% 功效。方差最大的是⑤的 R_high（sd 1.054）。本条仅为方差量级估计：配置表 §1 已将 13 个条件的“确认性”列全部标为否，本批次不得用于提出确认性主张（依 `P0B_PREREG_ANALYSIS_PLAN.md` §4）。

## 8e.8 成本实测

grid8 单 run 约 1690 s（28.2 min），grid32 约 6995 s（116.6 min）；5 进程并行时，104 runs 总墙钟约 27 h。相对配置表 §2 的原估计（52 GPU·h；grid8 20 min、grid32 40 min），grid8 估计准确，**grid32 实际为估计的 2.9 倍**。原因为 explicit permutation 的 `index_select` 开销随序列长度线性增长，而原估计外推自批次 C 的 `real_4dir` 实现，后者没有该开销。B5 单进程实测峰值显存为 3595 MiB，较批次 C 同规格的 3379 MiB 高 6.4%。该成本修正记录于此，不修改受 SHA 门控的配置表 §2。

## 8e.9 claim boundary

结论限于 CIFAR-10、channel-split 架构、Mamba backbone、d256、两档 reliance（grid8/grid32）及本文冻结的 G/R/L 三族路径。P0-B 是可行性 pilot，非确认性检验；跨数据集、跨 backbone、跨尺度的外推均须由主实验支持。

## 9. 【未完成】待补证据

| 实验 | 状态 | 为什么必需 |
|---|---|---|
| **5-seed(cloud_seed0–4,150 run)** | 🔄 云端运行中 | **一切结论的统计基础。** 没有它,§4–§6 全是轶事 |
| **channel-split 2×2 析因** | ❌ 代码就绪,未跑 | 打到 GroupMamba 的真实结构;`rand_perm` / `same_perm` 是唯一能判决"方向是不是几何的"的设计 |
| **掩码因果探针** | ❌ 未实现 | 唯一能把机制 A 从"剩下的那个"变成"被证实的那个" |
| **`widened_row` 对照** | ❌ 未实现 | 把"容量"与"多路径结构"进一步拆开(便宜) |
| **channel-split d_model=256** | ❌ 未跑 | 容量稳健性:d=64 时每组仅宽 16,可能欠拟合,导致"方向无用"的结论不可解释 |

---

## 10. 已知混杂与保留

1. **`num_workers` 影响数值。** 本地(workers=4)mamba/grid8 delta = +0.93;
   云端(workers=8)= +0.60。差 0.33pp,恰在 seed 间标准差(~0.39pp)量级内。
   **规避:5 个 seed 全部在云端同一环境跑,环境同质。**
   本地 seed 0 降级为跨硬件复现性检查。

2. **本地 unfused / 云端可用 fast path。** 代码显式设置 `use_fast_path=False`,
   **不得更改** —— 改了就与已完成的 run 不可比。

3. **`grid16` 那个点整体可疑。** delta 下凹、CKA 去相关增益异常高(mamba +0.060)。
   必须靠 5-seed 判定是噪声还是真实。

4. **channel-split d_model=64 时每组宽度仅 16**,总参数 18,314,
   只有单个 full-width row 分支(约 54,700)的 1/3。
   **若跑出 `channel_real ≈ channel_same`,无法区分"方向无贡献"与"模型太小用不上"。**
   必须报告 train_acc;若明显欠拟合,补 d_model=256(每组宽 64,序列层参数与
   full-branch 匹配:220,682 vs 218,762)。

---

## 11. P0-B LMTO 与 C_dir AUC 冻结(2026-07-20,性能运行前)

### 11.1 可行性与路径库记录

- 旧 block-serpentine 候选族在四个 `(n,b)∈{8,32}×{2,4}` 组合中均为 **0/20000** 条通过 C5;C5 未放宽。
- B0 状态为 `EXISTENCE_PASS`:在冻结预算内,两个尺度均找到四条满足 C5 且非 G 的排列。这不批准最终 L 生成器。
- B1 状态为 `B1_CANDIDATE_READY`:只恢复预先指定的 G1 链候选(n=8 seed 2026072101、n=32 seed 2026072201,均为 proposal 125000),并构造 reversal × transpose 四路径轨道;未在 G2/G3/G4 候选间事后择优。
- 候选 JSON: `P0B_L_PATH_BANK_CANDIDATE.json`, SHA-256 `10d8db0354967e9850c3873c7d8d4b3d91bb0e1dfa1622e6dad6881b7f4ccd7f`。
- 冻结 JSON: `P0B_L_PATH_BANK_FROZEN.json`, SHA-256 `93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7`;状态 `FROZEN_FOR_P0B_FEASIBILITY`。
- 正式名称为 **locality-matched topology-perturbed symmetry orbit (LMTO)**。不得称为 random Hamilton paths、locality-matched random paths、uniform random paths 或与 G 独立的随机样本。

### 11.2 LMTO 边界

- n=32 的 B0 G1 候选相对 G1 normalized Kendall distance 仅约 `0.006411`,不得称为“全局顺序随机化”。
- 原 C5 仅匹配四邻域 `d_seq` 的 mean/p50/p90(相对 G 目标 ±10%);不匹配完整分布、p95/max、AxisBias 幅度、polarity、coverage 或与 G 的距离。
- 冻结仅授权 LMTO 作为 P0-B feasibility pilot 的辅助控制,不构成跨任务、跨尺度或最终论文版本的唯一最优 L 生成器。

### 11.3 对比⑤与 AUC 冻结

- 对比⑤正式名称为 **canonical-orbit specificity control**:
  `P_G = GEO_DIV − mean(GEO_SG1, GEO_SG2, GEO_SG3, GEO_SG4)`,
  `P_LMTO = LOC_D − LOC_S`,
  `contrast_5 = P_G − P_LMTO`。
- `contrast_5 > 0` 仅表示 canonical raster symmetry orbit 的多路径增益大于该预先冻结、三统计量 locality 匹配的 topology-perturbed symmetry orbit;`contrast_5 ≈ 0` 仅表示在该辅助控制下未发现额外优势。任一结果均不能自动写成“全部收益来自 locality”,也不能单独排除完整 locality 分布、AxisBias、polarity 或 coverage 差异。
- `C_dir` AUC 固定为四方向分别计算,节点 `x=[0,0.01,0.05,0.10,0.20]`,值 `y=[0,C_dir(0.01),C_dir(0.05),C_dir(0.10),C_dir(0.20)]`,使用零点锚定的梯形积分并除以 `0.20`: `AUC_dir = numpy.trapezoid(y,x)/0.20`;主集合标量 `AUC_macro = mean(AUC_RIGHT,AUC_LEFT,AUC_DOWN,AUC_UP)`。必须报告四方向 AUC 与全部节点,不得只报 macro 或另定义 AUC。
- C4 的准确判定为 `AUC_macro({G1,G2,G3,G4}) > AUC_macro({G1,G3})`;不得写成四路径集合在每个已覆盖方向都严格高于两路径集合,RIGHT/DOWN 可能相等,增益来自补足反向总体。

### 11.4 P0-B 规模保持

本次冻结发生在任何 P0-B 性能运行前。保留原 `exp_id`、13 条件、2 reliance levels、4 training seeds、104 runs 与 diverse 条件的 Latin square;未增加实验。

## 12. P0-B R 路径冻结(性能运行前)

- `P0B_R_PATH_BANK_FROZEN.json` 以独立 CPU `torch.Generator` 和 `seed=17071+1000*s+i` 冻结 n=8/32、S1–S3、R1–R4 共 24 条完整路径。
- 完整 order 和 NumPy int64 C-byte order/inverse hash 是唯一运行时 source of truth;旧的训练时 `torch.randperm` 做法由冻结数组替代。
- 不允许重抽、加 offset 或按距离/C_dir/AUC 筛选。R 是固定路径阻断因素,不是路径总体随机效应。
- 保留原 exp_id、13 条件、2 reliance、4 training seeds、Latin square 与 104 runs。

## 13. P0-B CIFAR-10 validation split 冻结(性能运行前)

- 旧 runner 每 epoch 评估官方 CIFAR-10 test,不适用于 P0-B 的 validation 主终点。
- P0-B 冻结官方 `train=True` 内 45,000/5,000 stratified train/validation split: seed `20260720`, `numpy.random.PCG64`,class 0→9 连续 RNG,每类 4,500/500。
- `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` SHA `e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09`;完整 indices 是后续唯一来源。
- P0-B 不实例化、读取或评估 official `train=False` dataset;配置表 `docs/P0B_CONFIG_TABLE.md` 未修改。

---

## 8f. MAIN-01 主实验：结果落地与两份结果前文档的归档

**记录时间：** 2026-08-02。**执行 commit：** `32edce6`（624/624，零失败，零缺格）。

### 8f.1 两份结果前文档的版本与 SHA

两份文件于 2026-07-29 撰写完成，其 SHA-256 于同日提交至独立仓库
`github.com/yunshen0126/prereg-timestamps` commit `d73beba`，GitHub 服务端时间戳可由第三方核验。

| 版本 | 说明 | ADDENDUM_03 | CODE_DELTA |
|---|---|---|---|
| V0 | 撰写完成版，**即被时间戳覆盖的版本** | `7f02f9ba8c0a8f02708cabb048dce12c6f9b001a6fe3758a6b3c7a02e64c2beb` | `9918623eaed4a1dcec7efe0acd03d78a67302681e5bf331c8f2589e9d79400d4` |
| V1 | 填充后、自指 SHA 回填前 | `9040e032e1b12264d34d9a400c821c96b12bfba679a56a3ec97795c5357ae8ca` | `00b8501b0ab22d74413972306e4de67e8afaa1a7e3ad2bc12f063fc73e582477` |
| V2 | 自指 SHA 回填后，**即本 commit 收录的版本** | `1e5a15c1d37b3650e55c4cbd243b9fba4e41c54f1c7bd804b1521c0d7c241509` | 同 V1（本件无自指 SHA 行） |
| V3 | CODE_DELTA §9 口径更正后（见 §8g.10） | 不适用 | `9f014f514e861e3d3bf92200c3f8f2ddb32749326fbe3d26e83f3cea48ca1ac0` |
V0 两份原件留存于 `docs/prefill_snapshot/`，可直接与时间戳记录比对。

**差异范围：** V0→V1 的全部改动由 `docs/prefill_snapshot/*.fill.diff` 两个 unified diff 界定。
ADDENDUM_03 的改动限于 §8 情形 S7 第 1 条（回填 `FORMAL_CONFIG` 具体数值，
并删除该条下要求回填的指令引用块）；CODE_DELTA 的改动为在文末追加 §5.2（纯追加，无既有行改动）。
判据、口径、表述约束无一处变动。

`MAIN_PREREG_ADDENDUM_03_CONTINGENCY.md` 的自指 SHA 记录值为 V1 的 SHA，
故工作区版本直接 `sha256sum` 必然不匹配，与 `MAIN_PREREG_01.md` 同理，
唯一有效的校验是 `git show <commit>:<file> | sha256sum` 并与 V1 值比对。

### 8f.2 两份文档长期位于版本控制之外

两份文件自 2026-07-29 撰写至本次 commit 期间，**仅存在于本地工作副本，从未进入 git 历史**
（`git log --all -- '*ADDENDUM_03*' '*CODE_DELTA*'` 为空，且不受 `.gitignore` 影响）。
期间其完整性由外部时间戳而非 git 保障。本次 commit 前已核验本地副本、云端副本与
时间戳记录三方 SHA 逐位一致。此事实解释了 commit 时间晚于时间戳时间三日。

### 8f.3 代码惰性检验（`CODE_DELTA` §5.1）的执行结果

判据 A 与 B 均通过，§7 的数值惰性主张不撤回。详见 `CODE_DELTA_68dff0b_32edce6.md` §9。
归档侧 16 格重算得 `P_G' = +3.50 [+2.14, +4.86]`，与 §5.1 记录值逐位相同。

---

## 8g. 勘误（erratum）九条

### 8g.1 §8e.2 两格由聚合前舍入产生

与 `P0B_PREREG_ANALYSIS_PLAN.md` §3 规定的口径（聚合层不做任何中间舍入）不符：

| 量 | 原记录 | 正确值 |
|---|---|---|
| `P_R` @ R_high | +0.07 | **+0.08** |
| ④ axis @ R_low | −0.03 | **−0.02** |

成因：逐 seed 值先舍到两位再平均。两格均属 P0-B，**不出现在论文中**。

### 8g.2 §12.1 GRU 成本的系统性低估

预注册假设 GRU 单 run 耗时为 Mamba 的 0.70×，**实测 1.32×**
（cifar10 R_high：GRU ~9,518 s vs Mamba ~7,200 s）。全批墙钟约 95–100 h，
预注册估 89 h，差额几乎全部来自此项。

### 8g.3 `MAIN_PREREG_01.md` §2.3 内的失效交叉引用

§2.3 所引「§9.2」应读作「§5.2」。该文件受 SHA 门控不得修改，
详见 `CODE_DELTA_68dff0b_32edce6.md` §4.3。

### 8g.4 16 个增强敏感性 run 已迁至 `outputs_aug16`

详见 `CODE_DELTA_68dff0b_32edce6.md` §4.1。该批目录为本次判据 C 的归档侧数据源。

### 8g.5 批次 D 的复现基线 commit

为 `02981d9`，**非 `8055759`**。详见 `CODE_DELTA_68dff0b_32edce6.md` §4.2。

### 8g.6 CIFAR-10 划分 SHA 的誊写错误

交接备忘 `HANDOFF.md` §6.2 将 `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` 的 SHA 尾部
写作 `…0c072b1f09`，run metadata 的 `split_source_sha256` 实为 `…c0f072b1f09`
（`c` 与 `0` 位置互换，属誊写时的数位转置）。前缀 `e28719c9` 一致。
冻结文件本身的完整 SHA 见本节下方校验记录；runner 按完整 SHA 门控且全批 `git_dirty = False`，
故为交接文档的誊写错误，非产物不符。

冻结文件实测 SHA-256：`e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09`

### 8g.7 分析脚本对 GRU 臂输出了投票行

`MAIN_PREREG_ADDENDUM_03_CONTINGENCY.md` §7 描述分析脚本「保留逐数据集 M1/M2
但不输出投票与 M3」，但 `--backbone gru` 的实际 latex 输出含
`Proposition A & \multicolumn{3}{l}{not evaluable (1/5)}` 一行。
该行是五数据集投票规则套用于单数据集臂产生的过滤器伪影。
**论文未采用该行**（`sections/results.tex` §6.8 只报逐数据集 M1/M2，不报投票、不报 M3）。
脚本行为与文档描述的偏差在此记录，脚本本身未修改（批次已结束，改动不影响已产出结果）。

### 8g.8 `_compare_metadata` 的 legacy 豁免未按 `git_commit` 收口

详见 `CODE_DELTA_68dff0b_32edce6.md` §3.4。不影响主实验 624 个 run
（其 `augmentation = main_uniform`，永不进入该分支），待后续收紧。

### 8g.9 `run_p0b_feasibility.py` 第 740–741 行 `set_seed` 连写两遍

无害重复调用，第二次调用不改变 RNG 状态相对第一次的结果。记录待清理。

---

## 8h. 论文侧的对应关系

| 论文位置 | 数据来源 |
|---|---|
| Table 5 completeness | `analyze_main624.py --emit latex`，commit `32edce6` |
| Table 6 contrasts、Figure 1 | 同上（Mamba） |
| Table 7 criteria、Table 8 ceiling、Table 9 exploratory | 同上 |
| Table 10 GRU | `analyze_main624.py --backbone gru --emit latex` |
| Table 11 inertness | `CODE_DELTA` §9 |

论文触发的结果情形（`ADDENDUM_03` §8.1）：S1、S3、S5、S9 触发；S2、S4、S6 一致性、S7、S8 未触发。

---

## 8i. 去饱和后续臂：探针执行与不予执行的决定

**记录时间：** 2026-08-02/03。**性质：事后的、未预注册的探索性检查。**

### 8i.1 动机（事后）

MAIN-01 中 M1 失败的四个数据集在高负载档全部触发天花板标记（§8f）。
由此产生一个假设：那四个零可能是训练侧饱和造成的测量失败，而非效应的真实缺席。
论文 `sections/discussion.tex` 已将该假设标注为事后的、未预注册的考量。

为把它变成可测量的，曾计划一个"去饱和后续臂"：降低 `d_model` 使训练准确率压至
95% 以下，重跑五个数据集的高负载档（260 run），并以 CIFAR-10 作阳性对照
（它是全实验唯一 ② 区间排除零的数据集，若降容量后其效应亦消失，则操纵毁掉了
测量能力，本臂作废）。该臂的预注册草稿已起草，**但未冻结、未时间戳、未执行**。

### 8i.2 探针结果

先执行探针以确定 `d_model` 取值：五个数据集 × `GEO_SG1` × `R_high` × seed0，
`d_model ∈ {128, 64}`，产物在 `/root/autodl-tmp/outputs_probe_desat`。
`d_model = 256` 档直接读取 `outputs_main` 既有 run。
探针经与主实验相同的 runner 执行（仅额外传 `--d-model` 与 `--run-root`）。

单格尾窗 train accuracy（%）：

| dataset | d256 | d128 | d64 | 降幅 d256→d128 |
|---|---:|---:|---:|---:|
| cifar10 | 96.41 | 80.56 | 70.12 | **−15.85** |
| organamnist | 99.99 | 99.90 | — | **−0.09** |
| organcmnist | 99.91 | 98.90 | — | −1.01 |
| organsmnist | 97.27 | 93.51 | — | −3.76 |
| eurosat | 99.78 | 97.68 | — | −2.10 |

（`d64` 仅跑了 cifar10；见 §8i.3 的决定。以上为 `GEO_SG1/seed0` 单格值，
**不是** §8f 天花板表所用的结构组中位数，两者不是同一个量。）

### 8i.3 探针结论：降容量是错误的杠杆

宽度砍半，OrganAMNIST 的训练准确率只掉 0.09 个点，而 CIFAR-10 掉了 15.85 个点。
器官数据集的饱和是**任务属性**（28×28 上采样的低分辨率灰度图，训练集
一万二至三万五）而非容量过剩。外推至 `d64`：CIFAR-10 已跌至 70.12（欠拟合），
器官数据集预计仍在 99% 以上。

因此**不存在一个 `d_model` 能同时满足**该臂预注册草稿的 D0（CIFAR-10 阳性对照存活）
与 D2（目标数据集训练准确率低于 95%）。降容量路径被探针自身证伪。

### 8i.4 不予执行的决定，及其替代

**该臂不予执行。** 决定的依据不是机时，而是主实验数据中已存在一个更强的反驳，
且该反驳不携带事后动机：

对四个 M1 失败的数据集，在高负载档、同一批 run、同一个格内：

| dataset | 十三条件 val 跨度 (pp) | 结构对比 ① | 对比 ② |
|---|---:|---|---|
| organamnist | 2.61 | +1.88 [+1.42, +2.34] | −0.12 [−0.67, +0.43] |
| organcmnist | 2.75 | +1.99 [+0.84, +3.14] | +0.47 [−0.42, +1.37] |
| organsmnist | 5.90 | +4.53 [+3.13, +5.93] | +0.23 [−1.31, +1.76] |
| eurosat | 1.94 | +1.34 [+0.88, +1.81] | +0.40 [−0.04, +0.85] |

泛化端未触顶（跨度 1.94–5.90 pp），且装置在同格测出了 ①（区间全部排除零）。
"训练侧天花板压制了 ②"这一说法，须额外解释它为何在同一批 run、同一负载档、
同一输入上放过 ①。

该反驳全部使用已注册、已冻结、已报告的数据，**不引入任何事后动机**，
强于 260 个事后动机的 run。论文 `sections/limitations.tex` §Saturation 与
`sections/discussion.tex` §Task structure 已据此改写：原先"本数据不可证伪"的
表述改为"部分可由本数据证伪，且证据不支持该假设"。

### 8i.5 留存

- 探针产物：`/root/autodl-tmp/outputs_probe_desat`（6 个 run）
- 探针脚本：`mamba_scan_study/analysis/probe_desat.py`
- 核验脚本：`mamba_scan_study/analysis/ceiling_argument_check.py`
- 该臂预注册草稿**未冻结、未时间戳、未进入版本控制**，仅留存于本地工作副本。
  若日后重启该臂，须以新的预注册重新起草并重新时间戳，不得沿用该草稿声称事前性。
- `run_p0b_feasibility.py` 因探针新增 `--d-model`、`--run-root`、`--no-download`
  三个参数。默认值下与改动前逐字等价，已实测：`--d-model 256` 时
  `parameter_count = 282890`、
  `nominal_flops_equality_signature = c5ca93463cbbaa7bb6df0eccfffa24b15bf719b6a820b328e7c19a387eeb6ae3`，
  与既有 624 个 run 相同，故不构成新的代码差异，无需新增 CODE_DELTA。

---

## 8g.10 CODE_DELTA §5.2 初版差值表的口径错误（勘误第十条）

commit `873e2bb` 收录的 `CODE_DELTA_68dff0b_32edce6.md` §5.2 中，
判据 C 的 16 格差值由**已舍入到两位的显示值相减**得出，与
`P0B_PREREG_ANALYSIS_PLAN` §3 规定的"聚合层不做中间舍入、仅显示层舍入"不符。

论文 `sections/results.tex` §6.9 与 Table 11 已同步更正。
该节在本次修正中一并改名为 §9（原编号 §5.2 排在 §8 之后，编号顺序有误）。

六格差 0.01：GEO_DIV seed3、GEO_SG1 seed0、RND_D1 seed0 与 seed3、
RND_S1 seed0 与 seed1。跨度由 `−0.23 ~ +0.08` 更正为 **`−0.24 ~ +0.09`**；
`RND_S1` seed1 由 `−0.23` 更正为 **`−0.24`**；
归档侧 `P_R'` 区间上界由 `+1.58` 更正为 **`+1.57`**。

**判据 A 与 B 的判定结论、`P_G'` 与 `P_R'` 的点估计均未改变。**

该错误由新增的可复现脚本 `mamba_scan_study/analysis/inertness_check_16.py` 发现。
论文 `sections/results.tex` §6.9 与 Table 11 已同步更正。

---

## 8j. 分析与发车脚本纳入版本控制

MAIN-01 的五张表、Figure 1 与代码惰性检验此前由仅存于数据盘的脚本生成，
不在版本控制内，构成论文"released in full"主张的实质缺口。现纳入仓库：

| 脚本 | 用途 | 原位置 |
|---|---|---|
| `mamba_scan_study/experiments/main624_launch.py` | 624 run 发车器（含 preflight） | `/root/autodl-tmp/main_launch/` |
| `mamba_scan_study/analysis/analyze_main624.py` | 五张表（Table 5–9） | `/root/autodl-tmp/analysis/` |
| `mamba_scan_study/analysis/plot_forest.py` | Figure 1 | 同上 |
| `mamba_scan_study/analysis/regress_main624_p0b.py` | P0-B 回归（48 断言） | 同上 |
| `mamba_scan_study/analysis/inertness_check_16.py` | 代码惰性检验（CODE_DELTA §5.1） | 新增 |
| `mamba_scan_study/analysis/probe_desat.py` | 去饱和探针（§8i） | 新增 |
| `mamba_scan_study/analysis/ceiling_argument_check.py` | 天花板论证核验（§8i.4） | 新增 |

纳入前已核验：`analyze_main624.py` 在当前 HEAD 下重跑，输出与
`tables_mamba.txt` 逐字节一致；`regress_main624_p0b.py` 48/48 断言通过。

分六件，性质不同，分开记。合并叙述会稀释其中最实质的一件。

---

## (a) `PREREG_CAP_01` §9 第 3 步的自指 SHA 条款不可执行，且正确地未被执行

本件 §9 第 3 步要求"回填自指 SHA"。该动作在数学上不可能：一旦把 SHA 写入文件，
文件内容即改变，该 SHA 随即失效。文末两处占位符

```
**本件 SHA：** `<冻结后回填>`
**带外时间戳：** `<冻结后提交至 prereg-timestamps 并回填 commit 与日期>`
```

至今未填。**这不是疏漏，未填恰恰保住了文件与带外时间戳的逐位一致。**

带外记录（`yunshen0126/prereg-timestamps` 的 `2026-08-03_cap01.txt`，
commit `f46a42e`，自报时刻 2026-08-03T08:25:27Z = 北京时间 16:25:27）所记：

```
SHA-256 of the preregistration:
  36cc44c7394e2b23a810a5bc0d4d62bec290259935392317a0fb37458ca9b029
    PREREG_CAP_01.md
```

2026-08-04 复算本仓库内 `PREREG_CAP_01.md` 得同一值，**逐位吻合**。
即本件自冻结至今零字节改动。

带外记录同时载明：

```
mamba-scan-study HEAD at freeze: 92cd470
mamba-scan-study commit carrying this document: 4a5f7ed
```

故本件 §7.3 所记 `92cd470` 作为"冻结时 HEAD"是准确的；带外记录已把冻结时 HEAD
与承载本件的 commit 分别标注，并声明后者本身不构成优先权证据。
唯一措辞不确之处在于 §7.3 的上下文（"本臂开跑前 HEAD 须干净"）会让读者以为
`92cd470` 是执行时的 commit，而实际执行时 HEAD 为 `4a5f7ed`（preflight 记录）
及其后的 `68fa05d`（见 (b)）。以本条勘误澄清，**不修改本件**。

带外记录另载明记录时零 CAP-01 run 已执行、输出根为空。文件系统证据与之相符：
`/root/autodl-tmp/outputs_cap512` 出生于 16:18:02，早于记录，但当时为空目录。

## (b) 本臂执行期间发生一次 commit，违反 `PREREG_CAP_01` §6

### 证据链

| 时刻 | 事件 | 距起跑 | 证据 |
|---|---|---|---|
| 16:18:59 | commit `92cd470` cap01 launcher | — | `git log` |
| 16:22:45 | commit `4a5f7ed` 冻结 CAP-01 预注册 | — | `git log` |
| **16:25:27** | **带外时间戳 commit `f46a42e` 自报时刻** | −14.9 s | GitHub API，见 (a) |
| **16:25:41.926** | **发车器进程启动**（`/root/cap01.log` 出生时刻） | 0 | `stat -c %w`，硬下界 |
| 16:25:41.9–16:25:44 | preflight 运行并通过 | +0–2 s | `cap01.log`，HEAD=`4a5f7ed`、工作区干净 |
| ≈16:25:44 | 首批 run 落地 | +2 s | 首批 run 墙钟 8294 s，倒推自完成时刻 18:43:58 |
| 17:16:35 | `figure2_paths_grid8.pdf` 写入仓库根目录，工作树变脏 | +50m51s | 文件 mtime |
| 17:24:35 | `figure4_load_gating.pdf` | +58m51s | 文件 mtime |
| 17:28:53 | `figure5_ceiling.pdf` | +1h03m09s | 文件 mtime |
| 17:44:18 | `figure_components.pdf` | +1h18m34s | 文件 mtime |
| 17:53:58 | `plot_distance_dist.py` 写入仓库根目录（见 (d)） | +1h28m14s | 文件 mtime |
| 17:55:28 | `figure_distance_dist_grid32.pdf` | +1h29m44s | 文件 mtime |
| **17:58:24** | **commit `68fa05d`** | **+1h32m40s** | `git log` |

起跑时刻由 `/root/cap01.log` 的**出生时刻**（`stat -c %w` = 16:25:41.926）直接给出，
不依赖反推——`nohup` 创建该日志即发车器进程启动，是"CAP-01 执行任何东西"的硬下界。

**须记一条方法上的更正：** run 产物目录的出生时刻与其 mtime 相差 32 ms
（`p0b_..._GEO_SG1_R_high_seed2_d512`：出生 18:43:58.153958，mtime 18:43:58.185937），
说明 **runner 在 run 结束时才创建产物目录**，而非开跑时。故产物目录的时间戳
只能定位 run 的**结束**，不能定位其开始。本条早先曾据此反推起跑，结论虽与
日志出生时刻一致（相差约 2 秒），但推理路径不成立，在此更正。

### 变更范围

`68fa05d` 单独包含的变更（`git show --name-status 68fa05d`）：

```
A       mamba_scan_study/analysis/plot_ceiling.py
A       mamba_scan_study/analysis/plot_components.py
A       mamba_scan_study/analysis/plot_distance_dist.py
A       mamba_scan_study/analysis/plot_load_gating.py
A       mamba_scan_study/analysis/plot_paths.py
```

五个文件状态全部为 `A`（新增），**无修改、无删除**。
`PREREG_CAP_01.md` 与 `.gitattributes` 属 `4a5f7ed`，不在本次 commit 内。

`mamba_scan_study/analysis/` 不是 package（无 `__init__.py`），
`mamba_scan_study/experiments/` 下无任何脚本 import 之
（`grep -rn "analysis" mamba_scan_study/experiments/*.py` 无 import 命中）。
故本臂执行路径逐位未变：操纵量 `d_model`、runner、模型定义、数据装载、
冻结路径库均未受影响。

### 记录后果

全部 `104` 个 run 的 metadata 记 `git_commit = 68fa05d`、
`git_dirty = True`，包括时间上早于该 commit 的那些 run——即 run 记录的 commit
不是其开跑时的 commit。均匀性由 `cap01_finalize.py` 核实：
`全部 104 个 run 的 git_commit 与 git_dirty 完全一致 (68fa05d / True)，由 cap01_finalize.py 于 2026-08-06 17:46 核实`。

主实验对照：`outputs_main` 的 run 记 `git_commit = 32edce6`、`git_dirty = False`。

### 处置

**不补救。** 理由三条：

1. `ADDENDUM_03` §9 禁止因此类事由发起重跑；
2. 跑批期间禁止 commit，`git_dirty` 无法回填；
3. **不清理未跟踪文件**——现清理将使其后的 run 记为 `False`，产生半 True
   半 False 的不均匀记录，劣于当前的均匀状态。清理留待本臂结束后。

### 成因与教训

本账本 §8e.1 已记录过同一缺陷。`main624_launch.py` 为此设有 canary
（第 324–328 行的 outputs 忽略检查、第 525–527 行的 canary `git_dirty` 断言），
主实验因而干净。`cap01_launch.py` 仅在第 97–100 行作一次性 preflight
（`git status --porcelain` 非空即中止），**未继承该 canary**，故缺陷复发。

**可推广的教训：发车器的 preflight 是一次性的，不覆盖执行期间。**
凡长批次，须有周期性复查或对产物目录的写入隔离；仅靠开跑前一次检查不足以
保证批次内 provenance 的一致。

---

## (c) 起跑时刻记录错误，且 metadata 无时间字段

`HANDOFF_2026-08-03` §5 记"8 月 3 日 18:35 起跑"，与 (b) 的证据不符。
实际起跑为 **16:25:41.926**（发车器进程启动），差 2 小时 09 分。
18:35 接近第一批 run 的完成时刻（18:43:58），推测系当时据日志首次出现输出而记。

CAP-01 的 metadata 不含任何时间字段
（`[k for k in m if 'time'/'start'/'stamp'/'date' in k.lower()]` 返回空列表），
故本臂的起止时刻只能由 `cap01.log` 与文件系统 mtime 重建，且 mtime 会被任何
后续访问改写。**后续实验的 runner 应记录 UTC 起止时刻**——本条的重建工作
本不必要。

---

## (d) 仓库根目录的 `plot_distance_dist.py` 为弃用草稿

两份副本在取值口径上不同，不是格式差异：

| 副本 | `d_x` / `d_y` / `AxisBias` 取自 |
|---|---|
| 仓库根目录（未跟踪，mtime 17:53:58） | `aux["d_x"]`，四条路径合并 |
| `mamba_scan_study/analysis/`（已跟踪，属 `68fa05d`） | `aux["per_path"][0]`，仅 L1 |

已跟踪版本于 2026-08-04 重跑，与 `P0B_PREREG_FREEZE_L_AUC.md` 的八项冻结值
**逐位一致**：

```
mean 18.140625  p50 15.000000  p90 35.000000  p95 41.000000  max 66.000000
d_x 4.760081    d_y 31.521169  AxisBias -1.890395
```

`AxisBias = -1.890395` 与论文所报 `-1.890`（`method.tex` §subsec:lmto、
`results.tex` §subsec:res_exploratory）相符。C5 判据三项亦复现：
mean +9.94%、p50 −9.09%、p90 +9.38%，全部在 ±10% 内，与论文所报三个 margin 一致。

四条合并口径不可能得到该值：L3/L4 为 L1/L2 的转置，轴向偏置相互抵消，
合并后 `AxisBias` 趋近 0（本次实测 Arbitrary 族的合并值为 −0.0238，可作量级参照）。

**论文所用为已跟踪版本。** 直接证据：论文 Figure 3 的产物
`figure_distance_dist_grid32.png` 与已跟踪版本 2026-08-04 重跑所得
`dd_check.png` 的 SHA-256 完全相同：

```
59bb554e206245e673376fbfcc366387f2ce11892f753efcc3bd99462439956a
```

根目录副本待本臂结束后删除。删除动作记于本条，不另立勘误。

---

## (e) 判定结果不追加进本件，另立 `CAP01_RESULTS.md`

本件 §9 第 5 步要求"本臂全部 run 完成后，判定结果追加至本件 §10"。
**该步骤不予执行，改以另立文件的方式履行其实质。**

理由：本件当前 SHA-256 `36cc44c7394e2b23a810a5bc0d4d62bec290259935392317a0fb37458ca9b029`
与带外时间戳所记逐位相同（见 (a)）。**向本件追加任何一个字节，该一致性即断裂**，
而这一致性是目前证明"判据自冻结后零改动"的唯一直接证据，也是本臂
"事前冻结判据"这一（本已弱于 MAIN-01 的）主张的全部依托。

处理：新建 `CAP01_RESULTS.md`，开头载明所依据的冻结文件及其 SHA
（`PREREG_CAP_01.md` @ `36cc44c7...`）与带外时间戳 commit（`f46a42e`），
随后写 C0–C3 判定。`PREREG_CAP_01.md` **永久保持零字节改动**，
文末两处占位符永久保留其占位状态，不填。

**这是对本件 §9.5 字面的偏离，如实记录于此。** 偏离方向为更严：
§9.5 的目的是使判定结果与判据可对照，另立文件同样达成该目的，
且额外保住了冻结文件与带外时间戳的逐位一致。

论文中每一处引用本臂结果的位置，除 §5 第 4 条要求的事后动机标注外，
引用对象为 `CAP01_RESULTS.md`，其中回指冻结文件的 SHA。

## (f) 带外时间戳与执行的先后：由第三方观测确立

| 事件 | 北京时间 | 证据性质 |
|---|---|---|
| 带外记录 commit `f46a42e` | 2026-08-03 16:25:27 | 提交者自报（本地时钟） |
| **GitHub 服务端观测到 PushEvent** | **2026-08-03 16:25:30** | **第三方观测** |
| 发车器进程启动（`/root/cap01.log` 出生） | 2026-08-03 16:25:41.926 | 文件系统 |

推送早于执行 **11.9 秒**。取证于 2026-08-04：

```
PushEvent 2026-08-03T08:25:30Z
PushEvent 2026-07-29T00:46:11Z
CreateEvent 2026-07-29T00:44:25Z
```

（GitHub Events API，`/repos/yunshen0126/prereg-timestamps/events`；
原始响应存于 `push_event_raw.json`，SHA-256 见同名 `.sha256`。
**GitHub 的公开事件保留期约 90 天，2026 年 11 月初之后该查询将返回空**，
故上述取证不可重现，以存档为准。）

commit 自报时刻与推送观测时刻相差 3 秒，二者自洽。

**先后关系由此不依赖提交者的时钟**：无论本地时钟是否准确，GitHub 在
2026-08-03T08:25:30Z 已持有该文件的哈希记录，而本机在其后 11.9 秒才启动发车器。

**但 11.9 秒的余量仍窄，本条不以余量为卖点。** 真正承重的是三项：
(i) 带外记录自陈"记录时零 CAP-01 run 已执行、输出根为空"，与文件系统证据相符；
(ii) 冻结文件 SHA 与记录所载逐位吻合，证明其后零改动；
(iii) 判据 C0–C3 全文在该文件内，故判据内容自推送时刻起即固定且可公开验证。

若审稿人以余量之窄质疑，正确的回应是指向 (i)(ii)(iii) 与第三方观测的性质，
而非争辩秒数。本条如实记录余量之窄，不作粉饰。

---

## (g) 本臂的判定结果与其记录位置

C0 在两个数据集上均不通过；C1/C2/C3 如实报告但不予解释，且不得用于回应容量意见。
判定结果不追加进 `PREREG_CAP_01.md`（见 (e)），载于 `CAP01_RESULTS.md`。

一并记录三件与本账本相关的事实：

1. **104 / 104，零失败。** "零失败"指全部 run 跑满 100 epoch 并通过
   `completed.json` + `final_checkpoint.pt` + metadata SHA 三重校验。它不表示
   全部 run 都学到了东西：15 个 run 训练未收敛，训练与验证准确率同时停在常数
   输出水平。该事实记于 `CAP01_RESULTS.md` §4，未作任何排除处置。

2. **`cap01_judge.py` 的一处设计疏忽。** 该脚本在 C0 不通过时仍打印 §4.1–§4.3
   为 C1/C2/C3 预设的结论措辞，仅在每行末尾挂 `[C0 未通过, 不予解释]` 标记。
   那些措辞以 C0 通过为前提，C0 不通过时全部无效。脚本本应整段抑制。
   记于此以防后续引用其输出时误抄。

3. **`cap01_finalize.py` 第 5 节的告警已过期。** 该节报告 `PREREG_CAP_01.md`
   的两处占位符未回填并建议停止流程。依 (a)，未回填是正确状态——自指 SHA
   无法回填，不填恰恰保住了与带外时间戳的逐位一致。该告警应忽略。

---

## 附：论文正文的对应处理

于 Limitations 或补充材料的可复现性段落加一句，**不进摘要、不进结论、
不进任何主张句**（`ADDENDUM_03` §2）：

> The capacity-robustness runs record a repository commit made after the batch
> had begun; that commit added read-only figure-generation scripts and left the
> execution path unchanged. See the evidence ledger, entry 11.

不展开、不解释、不辩护。指向 ledger 即可。
