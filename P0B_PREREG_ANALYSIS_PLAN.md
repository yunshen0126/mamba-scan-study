# P0-B 分析口径预注册（补充件 02）

**撰写时间：2026-07-26。状态：在任何 P0-B 性能运行之前冻结。**

**与补充件 01 的关系：** `P0B_PREREG_ADDENDUM_01.md` 管对比共线性、效应量外推限制与冻结产物完整性；本文件（补充件 02）管**终点选择与统计口径**。两者互不覆盖、互不修改，均对 P0-B 生效。

---

## 1. 主终点

P0-B 的主终点为 **validation accuracy**，数据来自 `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` 冻结的 5,000 样本 validation 集。**official test 在 P0-B 全程冻结：不实例化、不评估、不报告。**

**依据（事前的，P0-B 尚未运行任何 run）：**

批次 C（CIFAR-10 d256）与批次 D（CIFAR-100 d256）两批独立数据均显示，在 `block=mamba, d_model=256, grid=32` 配置下，训练侧 interaction 效应随训练进程系统性衰减，而泛化侧效应稳定或增强——见 `docs/03_EVIDENCE_LEDGER.md` §8d.5 的六格轨迹表。

引用该表时需注意口径衔接：**§8d.5 的轨迹表取 epoch 90–100 窗口**（`interaction (ep90–100)` 与 `train_acc (ep90–100)` 两列），**与本文件 §2 规定的尾窗（epoch 80–100）定义不同**——§8d.5 的窗口是为了突出训练末端的衰减轨迹，而非整个尾窗的平均。**P0-B 自身的一切判定一律使用本文件 §2 的尾窗定义；§8d.5 的窗口仅为历史证据的记录口径，不迁移到 P0-B。**

P0-B 的固定边界（`docs/P0B_CONFIG_TABLE.md` §1：仅 mamba、`d-model 256`）恰为上述配置。因此训练侧终点在 P0-B 配置下会系统性低估效应，不适合作为主终点。

**训练侧指标（train_acc、train_loss）仍全程记录并作为诊断量报告，不作为判定依据。**

## 2. 尾窗与聚合

尾窗定义为 epoch 80..100（含端点），共 21 个 epoch，按 run 取算术平均。**禁止使用 best-epoch 选取**（会构成对 validation 的多次查看）。每个 design cell 的观测数为 4（training seed 0..3）。

## 3. 效应量与区间

五个主对比 ①–⑤ 的定义直接引用 `docs/P0B_CONFIG_TABLE.md` §1，在本文件中原样复述，不修改配置表：

- ① reliance × locality：`mean(GEO_S*) − mean(RND_S*)` 在两档 reliance 下的差
- ② reliance × geometric-vs-random performance gain：`P_G − P_R`，其中 `P_G = GEO_DIV − mean(GEO_S*)`，`P_R = mean(RND_D*) − mean(RND_S*)`
- ③ traversal polarity：`GEO_SG1 − GEO_SG2`
- ④ scan axis：`GEO_SG1 − GEO_SG3`
- ⑤ canonical-orbit specificity control：`contrast_5 = P_G − P_LMTO`，其中 `P_G = GEO_DIV − mean(GEO_SG1, GEO_SG2, GEO_SG3, GEO_SG4)`，`P_LMTO = LOC_D − LOC_S`

区间估计：`mean ± t(3, 0.975) × s / sqrt(4)`，`t(3,0.975) = 3.182`，`s` 为样本标准差（ddof=1）。单位为百分点。

**明确声明：4 个观测下的 CI 宽度估计不稳定，本 pilot 不设"CI 宽度可接受"的硬门槛。** 区间的用途是为主实验的功效与预算规划提供方差量级，与 `docs/P0B_CONFIG_TABLE.md` §2 的成本估算及其 C8 检查项一致。

## 4. 多重比较处理

五个主对比**不做家族误差率校正**，理由是 P0-B 为 feasibility pilot 而非确认性检验——`docs/P0B_CONFIG_TABLE.md` §1 已将 13 个条件的"确认性"列全部标为"否"。

**此点必须在报告与论文中显式声明，不得事后按"某个对比显著"提出确认性主张。**

## 5. 事前预期与负值预案

三条写死：

1. 依 `P0B_PREREG_ADDENDUM_01.md` §3，`contrast_5 ≈ 0` 是事前预期，不得事后包装为"我们发现 locality 不解释效应"。
2. 依 `P0B_PREREG_ADDENDUM_01.md` §2，禁止用批次 C 的效应量对 P0-B 做功效外推。
3. 依 `docs/03_EVIDENCE_LEDGER.md` §8d.4，低负载档（`R_low = grid8`）的 diversity/interaction 效应**可能显著为负**。负值是预期内的可能结果，**不得当作实现缺陷去排查、不得触发重跑、不得调整 seed 或路径**。任何因出现负值而发起的代码改动都必须先停止并提交理由。

## 6. 诊断记录项

每个 run 除现有 runner 已记录的字段外，另记录：

- 尾窗（epoch 80–100）train_acc
- 尾窗（epoch 80–100）validation accuracy
- 二者之差（generalization gap）
- epoch 窗 (10,20) 与 (90,100) 的 validation accuracy

用途是在 P0-B 内部复核 §1 所依据的训练侧衰减现象是否同样出现，属**诊断**不属**判定**。

## 7. 与冻结产物的关系

本文件**不修改**任何 SHA 门控文件：`docs/P0B_CONFIG_TABLE.md`（SHA `790e08faf1856d8307d56500e0143cdb36225ae10c3542287e33b8efd6c1a33e`）、`P0B_L_PATH_BANK_FROZEN.json`、`P0B_R_PATH_BANK_FROZEN.json`、`P0B_CIFAR10_VAL_SPLIT_FROZEN.json` 四个 source SHA 继续有效。本文件**不进入 runner 的 SHA 门**，仅在分析阶段生效。

## 8. 落盘后

1. 在 `.gitattributes` 加入 `P0B_PREREG_ANALYSIS_PLAN.md -text`
2. commit
3. 计算本文件自身的 SHA-256，回填到下方"本件 SHA"一行，第二次 commit

---

**本件 SHA：**（待第二次 commit 回填）
