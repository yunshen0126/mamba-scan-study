# CONTENT — P0-A/B0 正式定义源（路径与指标）

**性质：内容规格，不是任务指令。**  
**用途：仅作为 `INSTRUCTION_B0_locality_control_feasibility_audit.md` 第二阶段的正式定义来源。**  
**日期：2026-07-20**

本文件冻结 B0 审计所需的 G1–G4、AxisBias、四方向 `C_dir`、阈值和 C5。  
本文件**不冻结**最终 L 生成器，不冻结 AUC 定义，也不授权修改模型或运行训练。

---

## 1. 网格、索引与路径表示

仅审计：

```text
n ∈ {8, 32}
H = W = n
N = n²
```

cell 使用 row-major 索引：

```text
u = r·n + c
r,c ∈ {0,...,n−1}
```

每条路径同时表示为：

```text
order[t] = 第 t 步访问的 cell row-major 索引
pi[u]    = cell u 被访问的步序号
```

必须满足：

```text
pi[order] = arange(N)
pi = argsort(order)
```

---

## 2. G1–G4

### G1：raster_lr

```text
pi_G1[u] = u
order_G1[t] = t
```

### G2：raster_reverse

G1 的完整序列严格反转：

```text
pi_G2[u] = N − 1 − pi_G1[u]
order_G2[t] = N − 1 − t
```

### G3：column_raster

```text
pi_G3(r,c) = c·n + r
pi_G3[r·n+c] = c·n+r

order_G3[t] = (t mod n)·n + floor(t/n)
```

### G4：column_reverse

G3 的完整序列严格反转：

```text
pi_G4[u] = N − 1 − pi_G3[u]
order_G4 = flip(order_G3)
```

不得把“每行内部反转”代替完整序列反转。

---

## 3. 四邻域边

无向水平边：

```text
(r,c) — (r,c+1)
r=0,...,n−1
c=0,...,n−2
```

数量为 `n(n−1)`。

无向垂直边：

```text
(r,c) — (r+1,c)
r=0,...,n−2
c=0,...,n−1
```

数量为 `n(n−1)`。

合并边数：

```text
M = 2n(n−1)
```

---

## 4. 无向 locality 指标

对每条无向四邻域边 `{u,v}`：

```text
d_seq(u,v) = |pi[u] − pi[v]|
```

在水平边与垂直边合并后的完整数组上报告：

```text
mean
p50
p90
p95
max
```

分位数冻结为：

```python
numpy.percentile(values, q, method="linear")
```

G1–G4 的合并分布均由以下两部分等量组成：

```text
1，重复 n(n−1) 次
n，重复 n(n−1) 次
```

因此精确目标为：

```text
mean_G = p50_G = (n+1)/2
p90_G = n
```

回归值：

| n | mean_G | p50_G | p90_G |
|---|---:|---:|---:|
| 8 | 4.5 | 4.5 | 8.0 |
| 32 | 16.5 | 16.5 | 32.0 |

---

## 5. AxisBias

分别计算：

```text
d_x = mean(d_seq over all horizontal neighbor pairs)
d_y = mean(d_seq over all vertical neighbor pairs)

AxisBias = log(d_x / d_y)
```

自然对数。

回归要求：

```text
AxisBias(G1) = −AxisBias(G3)
AxisBias(G2) = AxisBias(G1)
AxisBias(G4) = AxisBias(G3)
```

允许浮点容差只用于 `log` 后的数值比较；底层 `d_x`、`d_y` 应精确。

---

## 6. 四方向有向覆盖节点

对路径集合 `P={pi_1,...,pi_C}` 和一个有序空间邻居 `u→v`：

```text
d_min_plus(u→v)
  = min over c satisfying pi_c[v] > pi_c[u]
      (pi_c[v] − pi_c[u])

若不存在满足方向条件的路径，记为 +∞。
```

覆盖节点：

```text
C_dir(tau) = proportion of ordered pairs with d_min_plus <= tau
```

四个总体必须分别计算，不得合并：

```text
RIGHT: u=(r,c),     v=(r,c+1)
LEFT:  u=(r,c+1),   v=(r,c)
DOWN:  u=(r,c),     v=(r+1,c)
UP:    u=(r+1,c),   v=(r,c)
```

归一化阈值冻结为：

```text
tau_tilde ∈ {0.01, 0.05, 0.10, 0.20}
tau = tau_tilde·(N−1)
```

使用浮点比较，不取整：

```text
d_min_plus <= tau
```

算术备注：n=8 时，`tau_tilde=0.01` 对应 `tau=0.63`，而有限正向距离至少为 1，因此该节点恒为 0。

B0 只报告上述四个节点。**AUC 尚未冻结，不得在 B0 中自行定义为正式指标。**

---

## 7. C5：locality 三统计量匹配

对每条候选 L 路径，在每个 `n∈{8,32}` 下分别要求：

```text
0.9·mean_G <= mean_L <= 1.1·mean_G
0.9·p50_G  <= p50_L  <= 1.1·p50_G
0.9·p90_G  <= p90_L  <= 1.1·p90_G
```

两个尺度都必须取得四条候选，才可记：

```text
EXISTENCE_PASS
```

该状态只表示在冻结搜索预算内找到满足 C5 的四条非 G 排列，不表示：

- 已批准最终 L 生成器；
- L 与 G 已足够独立；
- L 已匹配 AxisBias、极性或有向覆盖；
- 主对比⑤的机制解释已经成立。

---

## 8. 旧块化 serpentine 候选族中的 b

本节仅用于忠实审计旧提议。

`b` 表示**每个轴上的块数**：

```text
总块数 = b²
b ∈ {2,4}
```

每个块的边长为：

```text
m = n/b
```

每个块含有、且块内 order 必须覆盖：

```text
m² = (n/b)²
```

个 cell。

因此块内合法性断言必须是：

```text
len(block_order) == (n/b)²
```

而不是 `b²`。

每块的 8 个标签定义为：

```text
row-major / column-major
×
TL / TR / BL / BR
```

起始角已经决定外轴起点和首个内轴方向，不得额外加入独立的“serpentine direction”变量。

---

## 9. 冲突处理

若仓库中的其他文件与本文件在 B0 所需定义上冲突：

1. 停止执行；
2. 列出冲突文件、章节与两套定义；
3. 不自行选择；
4. 不生成路径或报告结果。

旧版 `INSTRUCTION_B` 和旧版 P0-B prereg amendments 已暂停，不能覆盖本文件。
