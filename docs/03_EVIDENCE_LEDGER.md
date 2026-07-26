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
