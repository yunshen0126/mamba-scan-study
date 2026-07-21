# P0-B 预注册修订：L 路径库与 C_dir AUC 冻结

**日期：2026-07-20**  
**状态：在任何 P0-B 性能运行之前冻结。**  
**适用范围：仅 P0-B feasibility pilot。**

## 1. 决策

B0 已证明满足原 C5 的非 G 排列存在。B1 已按事先指定规则恢复 G1 链候选，并构造 reversal × transpose 四路径轨道。

现将该四路径库冻结为 P0-B 的辅助控制路径库，但其正式名称改为：

```text
locality-matched topology-perturbed symmetry orbit
```

简称：

```text
LMTO
```

不得再称为：

- locality-matched random paths；
- random Hamilton paths；
- uniform random paths；
- 与 G 独立的随机样本。

冻结只表示该精确路径库可用于 P0-B feasibility pilot，不表示它是跨任务、跨尺度或最终论文版本的唯一最优 L 生成器。

---

## 2. 来源与完整性

B1 候选源文件：

```text
P0B_L_PATH_BANK_CANDIDATE.json
```

候选文件 SHA-256：

```text
10d8db0354967e9850c3873c7d8d4b3d91bb0e1dfa1622e6dad6881b7f4ccd7f
```

正式定义源 SHA-256：

```text
3e79d2f8c941f7c54f11eaee21332265d9d064a9fb9971169fa18a6295d3cc8c
```

冻结时必须复制为：

```text
P0B_L_PATH_BANK_FROZEN.json
```

除顶层冻结元数据外，两个 grid 下 L1–L4 的完整 `order`、局部性指标和路径哈希不得改变。

冻结文件顶层状态：

```text
FROZEN_FOR_P0B_FEASIBILITY
```

冻结文件必须记录：

```text
source_candidate_file
source_candidate_sha256
freeze_date
freeze_scope
decision_record
```

---

## 3. 路径构造

每个 grid 的 L1 均来自 B0 中预先指定的 G1 链：

```text
n=8:
  seed = 2026072101
  proposal = 125000

n=32:
  seed = 2026072201
  proposal = 125000
```

四路径：

```text
L1 = Q
L2 = reverse(Q)
L3 = transpose_cells(Q)
L4 = reverse(transpose_cells(Q))
```

其中：

```text
transpose_cells(r*n+c) = c*n+r
```

### n=8 order SHA-256

```text
L1  3de8bce25dbb1a6d914df0de51bda80be5d7d618fc9fd50614688d0b077dca53
L2  23e52c218f568e3bb5b3aebd6cdf34ef0c95ac6265f2d59d377e85cf347bd46d
L3  3f51c455720a5c3f26b8ed54c0a8be3527409b7abb00a1042096ee93bfa28b5c
L4  24294a4a911ede8b8969765e58fd3ec49210ba921f7c03046dc87006cf264c20
```

### n=32 order SHA-256

```text
L1  559200199ddd125fb84fa4e717edf7d1276168a2ba3c022fccdbfba7ba906d21
L2  584d18f8dc56ec52a5a58dfabdccdb2497b5259b105d0c30fa6c14a0952ffcc2
L3  a8d13331184c4a4415fccd8ef42c1837f8545e84924b7de2ae21120a913e1dd1
L4  262703529def5f54f48dd7ac64d6d0f2ea072dd55a649fc0e39686ebe223c524
```

---

## 4. C5 与额外差异

原 C5 保持不变。每条 L 在每个 grid 下均必须满足：

```text
mean、p50、p90 分别位于对应 G 目标的 ±10% 内
```

冻结值：

```text
n=8:
  mean = 4.946428571428571
  p50  = 4.5
  p90  = 8.0
  p95  = 12.0
  max  = 17

n=32:
  mean = 18.140625
  p50  = 15.0
  p90  = 35.0
  p95  = 41.0
  max  = 66
```

以下内容仅作诊断，不是匹配约束：

- 完整 d_seq 分布；
- p95/max；
- AxisBias 的绝对幅度；
- 单路径或集合级 C_dir；
- 与 G 的 Hamming、Kendall 或 sequence-edge overlap。

特别是 n=32 的 L1 相对 G1 normalized Kendall distance 仅约 0.006411，不得称为“全局顺序随机化”。

---

## 5. P0-B 条件映射

保留原 exp_id，以避免改变 13 条件和 104 runs 的设计规模。

```text
LOC_S:
  正式条件名 = LMTO single
  seed0 -> L1 ×4
  seed1 -> L2 ×4
  seed2 -> L3 ×4
  seed3 -> L4 ×4

LOC_D:
  正式条件名 = LMTO diverse
  基础集合 = L1,L2,L3,L4
  按原 Latin square 随训练 seed 轮换到 ch0..ch3
```

路径实例只由冻结 JSON 与 grid 决定，不由训练 seed 生成。训练 seed 只决定 Latin-square 通道映射及训练随机性。

---

## 6. 对比⑤的正式改名与解释

原名称“地板效应控制”过强，改为：

```text
⑤ canonical-orbit specificity control
```

定义：

```text
P_G    = GEO_DIV - mean(GEO_SG1, GEO_SG2, GEO_SG3, GEO_SG4)
P_LMTO = LOC_D - LOC_S

contrast_5 = P_G - P_LMTO
```

解释边界：

- `contrast_5 > 0`：canonical raster symmetry orbit 的多路径增益大于该预先冻结、三统计量 locality 匹配的 topology-perturbed symmetry orbit。
- `contrast_5 ≈ 0`：在该辅助控制下，没有发现 canonical raster orbit 的额外多路径优势。
- 无论结果为何，都不能自动写成“全部收益来自 locality”。
- 该对比不能单独排除完整 locality 分布、AxisBias 幅度、polarity 或 coverage 差异。
- LMTO 是固定辅助控制，不是从某个随机路径总体中抽样得到的随机效应样本。

---

## 7. C_dir AUC 冻结

四个方向分别计算：

```text
RIGHT, LEFT, DOWN, UP
```

冻结节点：

```text
x = tau/(N-1) ∈ {0.01, 0.05, 0.10, 0.20}
```

增加数学锚点：

```text
C_dir(0) = 0
```

每个方向的归一化 AUC：

```text
x_nodes = [0, 0.01, 0.05, 0.10, 0.20]
y_nodes = [0, C_dir(0.01), C_dir(0.05), C_dir(0.10), C_dir(0.20)]

AUC_dir =
  trapezoidal_integral(y over x from 0 to 0.20) / 0.20
```

实现必须等价于：

```python
numpy.trapezoid(y_nodes, x_nodes) / 0.20
```

主集合级标量：

```text
AUC_macro = mean(AUC_RIGHT, AUC_LEFT, AUC_DOWN, AUC_UP)
```

必须同时报告四个方向 AUC 和所有冻结节点；不得只报告 macro 值。

### 几何回归值

四路径 G：

```text
n=8:
  AUC_dir   = 0.85（四方向相同）
  AUC_macro = 0.85

n=32:
  AUC_dir   = 0.975（四方向相同）
  AUC_macro = 0.975
```

两路径 `{G1,G3}`：

```text
n=8:
  RIGHT=0.85, LEFT=0, DOWN=0.85, UP=0
  AUC_macro=0.425

n=32:
  RIGHT=0.975, LEFT=0, DOWN=0.975, UP=0
  AUC_macro=0.4875
```

LMTO 四路径：

```text
n=8:
  AUC_dir   = 0.7991071428571429（四方向相同）
  AUC_macro = 0.7991071428571429

n=32:
  AUC_dir   = 0.9645413306451613（四方向相同）
  AUC_macro = 0.9645413306451613
```

因此 C4 的准确判定口径为：

```text
AUC_macro({G1,G2,G3,G4}) > AUC_macro({G1,G3})
```

不得写成四路径集合在每个已覆盖方向上都严格高于两路径集合；RIGHT/DOWN 上可能相等，增益来自补足反向总体。

---

## 8. 预注册限制声明

P0-B 是 feasibility pilot，LMTO 仅用于检查一个预先固定的辅助拓扑控制是否可实现、可训练以及效应方差量级。

论文或报告中允许的描述：

```text
a preselected three-statistic-locality-matched,
topology-perturbed reversal/transpose symmetry orbit
```

禁止描述：

```text
random Hamilton locality control
perfect locality matching
axis/polarity/coverage-matched control
proof that locality alone explains the effect
```

本修订在任何 P0-B 性能结果产生前冻结。
