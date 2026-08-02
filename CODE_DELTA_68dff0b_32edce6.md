# 代码差异说明：预注册冻结点 `68dff0b` 与主实验执行点 `32edce6`

**撰写时间：2026-07-29。状态：主实验运行期间撰写，在任何主实验结果产生之前落盘。**

**本件性质：** 记账与论证文件，**不是**预注册补充件。本件不新增、不修改、不放宽任何判据。
`MAIN_PREREG_01.md`（SHA `1ccd6245d2583e563086e09379b1e20944110a109c7374d7b8576b62a66cb7ff`，对应 commit `68dff0b`）、
`P0B_PREREG_ADDENDUM_01.md`、`P0B_PREREG_ANALYSIS_PLAN.md` 三件保持逐字节不变，其 SHA 继续有效。

---

## 0. 本件要回答的问题

主实验预注册冻结于 commit `68dff0b`。624 个 run 实际执行于 commit `32edce6`。
两者之间存在代码差异。本件逐条列出该差异，并对每一条论证：**它不改变任何 run 的数值结果。**

若不作此说明，论文的复现声明存在一个缺口：读者按预注册所述的冻结点 checkout 代码，
得到的 runner 与实际产出 624 个 run 的 runner 不是同一份。

---

## 1. commit 链与差异范围

```text
870d034  feat(data): extend P0-B data pipeline to five datasets
2d804f6  feat(runner): wire dataset and augmentation dimensions
68dff0b  prereg: freeze MAIN-01 main experiment preregistration   ← 预注册冻结点
37195ee  prereg: record MAIN-01 self SHA-256
32edce6  runner: backbone dimension, eurosat SHA gate + split      ← 624 runs 执行点
         binding, data split provenance
```

`68dff0b..32edce6` 共两个 commit：

| commit | 性质 | 是否含训练代码改动 |
|---|---|---|
| `37195ee` | 仅回填 `MAIN_PREREG_01.md` 的自指 SHA 一行 | **否**（纯文档） |
| `32edce6` | runner 与数据管线 | 是 |

因此**全部代码差异集中在单个 commit `32edce6`**，涉及三个文件：

```text
mamba_scan_study/experiments/run_p0b_feasibility.py
mamba_scan_study/experiments/p0b_data.py
.gitattributes
```
外加一个新增的独立校验脚本 `verify_outdirs.py`（不被 runner import，无运行期影响）。

### 1.1 自指 SHA 的校验口径

`MAIN_PREREG_01.md` 在 `68dff0b` 落盘、在 `37195ee` 回填自身 SHA。因此**工作区版本的
`sha256sum` 必然不等于记录值**，唯一有效的校验是：

```bash
git show 68dff0b:MAIN_PREREG_01.md | sha256sum
# 1ccd6245d2583e563086e09379b1e20944110a109c7374d7b8576b62a66cb7ff
```

该断言已在发车前的 preflight 中执行并通过（云端，2026-07-29 01:0x）。

---

## 2. 六条代码改动与数值惰性论证

### 2.1 新增 `--backbone {mamba,gru}`

**改动：** `run_p0b_feasibility.py` 增加 CLI 参数，默认 `mamba`；经
`replace(FORMAL_CONFIG, dataset=args.dataset, block_type=args.backbone)` 注入模型配置；
并进入 `architecture_operator_signature()` 与 `nominal_flops_equality_signature()`。

**为什么不改变数值：** 默认值为 `mamba`，与改动前 `FORMAL_CONFIG.block_type` 的硬编码值相同。
`replace()` 在 `backbone == "mamba"` 时产出与原 `FORMAL_CONFIG` 逐字段相等的对象。
`--backbone` 进入两个签名函数意味着 GRU 与 Mamba 的架构签名不会互相混淆，
但对 Mamba 路径而言签名的输入值未变，故签名值未变。

**依据：** 520 个 Mamba run 全部以 `--backbone mamba` 执行，走的是与 `68dff0b` 逐字等价的路径。

### 2.2 产物目录增加 backbone 段

**改动：** `_run_directory()` 的 else 分支由

```text
p0b_{dataset}_{augmentation}_{exp_id}_{reliance}_seed{seed}
```
改为
```text
p0b_{dataset}_{augmentation}_{backbone}_{exp_id}_{reliance}_seed{seed}
```

if 分支（`dataset == "cifar10" and augmentation == "p0b_legacy"`）一个字符未动。

**为什么不改变数值：** 目录名不进入任何训练路径，不参与 RNG 播种、不影响数据加载顺序、
不进入任何 SHA 门。它只决定产物写到哪里。

**为什么必须改：** 若不改，主实验 cifar10 的 104 个 GRU run 与 104 个 Mamba run
目录逐字相同，会互相覆盖。

**副作用（有利）：** 该改动使 16 个增强敏感性 run（命名为
`p0b_cifar10_main_uniform_{exp_id}_{reliance}_seed{s}`，缺 backbone 段）与主实验
cifar10-Mamba 的同格目录（含 `_mamba_` 段）不再重名。见 §4.1。

### 2.3 EuroSAT 划分文件加入 source SHA 门

**改动：** 新增 `_verify_eurosat_split_source()`，仅当 `dataset == "eurosat"` 时由
`verify_formal_config(dataset=args.dataset)` 调用，比对
`P0B_EUROSAT_SPLIT_FROZEN.json` 的 SHA-256 与常量
`f5ddb2db3f8ffc74efb77295e0fac17d34df85179bcd78de3f4e638b685c4117`。

**为什么不改变数值：** 纯新增校验，不改变已有四源门的映射与比对逻辑；
非 EuroSAT 路径不触发。校验通过则继续，不通过则抛异常终止——
不存在"通过但改变了行为"的分支。

**为什么必须改：** `MAIN_PREREG_01.md` §3 的冻结产物清单为六项，而改动前的门只覆盖四项，
EuroSAT 的 104 个 run 的划分来源不受 fail-closed 保护。这是遗漏，不是设计选择。

### 2.4 新增 `data_split_provenance` 与五数据集样本数断言

**改动：** `p0b_data.py` 新增 `DATASET_SPLIT_LENGTHS` 常量与 `_data_split_provenance()`；
在三条数据构造路径（Organ / EuroSAT / CIFAR-10）上，于构造完成后立即断言
`(train_n, val_n)` 与下表逐字相等，不等则抛 `ValueError`（退出码 1）：

| dataset | train | validation |
|---|---:|---:|
| cifar10 | 45,000 | 5,000 |
| organamnist | 34,561 | 6,491 |
| organcmnist | 12,975 | 2,392 |
| organsmnist | 13,932 | 2,452 |
| eurosat | 22,000 | 2,500 |

该表逐字取自 `MAIN_PREREG_01.md` §4.1。metadata 新增字段 `data_split_provenance`，
含 `source / artifact / sha256 / train_n / val_n`。

**为什么不改变数值：** 断言在数据集构造之后、训练之前执行，通过则不改变任何张量、
不改变采样顺序、不改变 RNG 状态。metadata 新增字段只影响记录，不影响计算。

**为什么必须改：** 三个 Organ 数据集使用 MedMNIST 官方划分，其来源不在任何冻结产物中，
改动前**完全不受保护**。换一个 MedMNIST 版本、npz 被重下、官方划分调整，
metadata 不会有任何变化，而 §4.1 的样本数是写死在预注册里的——312 个 run 会静默不可比。

### 2.5 EuroSAT 索引-图像绑定断言

**改动：** `_build_eurosat_loaders()` 在切 `Subset` 之前新增两条断言：
`ImageFolder.classes` 与冻结 JSON 的 `classes` 顺序敏感相等；
排序后 targets 的逐类计数与 `class_distributions["overall"]` 相等。
另将排序后 targets 的 `int64_c_sha256` 记入 `data_split_provenance["targets_sha256"]`，
**仅记录，不比对**。

**为什么不改变数值：** 同 §2.4，纯断言 + 纯记录。

**为什么必须改：** 改动前 `_load_eurosat_indices()` 只校验三个长度与"是 0..26999 的分区"，
索引指向哪张图完全由 `_sorted_imagefolder` 的路径排序决定，没有任何断言把该排序与冻结时的
排序绑定。CIFAR-10 那条路径有 `images_uint8_c_sha256` / `targets_int64_c_sha256` 做绑定，
EuroSAT 没有。目录中多一个文件即整体移位，而改动前的断言会全部通过。

EuroSAT 是 `MAIN_PREREG_01.md` §7 中 M3 事前预测序 `Organ > cifar10 > eurosat` 的低端锚点，
其静默错误产出的是偏低的效应量，方向恰好有利于假设，属 anticonservative。

**已核算的一致性（本件作者独立复算）：** 冻结 JSON 的 `class_distributions` 中，
overall 合计 27,000、train 22,000、validation 2,500、test 2,500，
逐类 `train + validation + test == overall` 十个类全部成立，
`classes` 为字典序（与 ImageFolder 的类发现顺序一致）。

**关于 `targets_sha256` 的效力边界：** 该值**不提供额外保护**。EuroSAT 的路径排序为
"类目录/文件名"，故 targets 必为各类连续块，其序列由 `classes` 顺序与逐类计数唯一确定——
而这两者已被上述两条断言覆盖。`targets_sha256` 仅作为记录保留，
**不得在论文中描述为对图像内容的绑定**。图像内容本身仍未绑定（某张 JPEG 被替换而计数不变则不可检出）；
该风险由发车前的一次性核对承担，见 §3.3。

### 2.6 `.gitattributes` 增加一条 `-text`

**改动：** 新增 `P0B_EUROSAT_SPLIT_FROZEN.json -text`。

**为什么不改变数值：** git 属性不进入运行期。

**为什么必须改：** `P0B_PREREG_ADDENDUM_01.md` §5.3 规定冻结产物必须标注 `-text`，
禁止任何行尾符转换。EuroSAT 划分是主实验新增的冻结产物，改动前遗漏了该条目。
若允许 `core.autocrlf` 规范化，跨平台 checkout 后 SHA 必然不匹配，
且失败原因（行尾符 vs 篡改）在 fail-closed 规则下不可区分。

---

## 3. 一处有意的 fail-open：`_compare_metadata` 的 legacy 豁免

### 3.1 事实

```python
if (
    "data_split_provenance" not in normalized
    and normalized.get("dataset") == "cifar10"
    and normalized.get("augmentation") == "p0b_legacy"
):
    normalized["data_split_provenance"] = expected.get("data_split_provenance")
```

该分支把 `expected` 的值抄进 `normalized`，使该字段的比较恒真。
另有一条同类归一化：`if "backbone" not in normalized: normalized["backbone"] = "mamba"`。

### 3.2 作用域

`build_metadata()` 在 `32edce6` 之后一定写入 `data_split_provenance`，
因此 `"data_split_provenance" not in normalized` 只对**改动之前写下的 metadata** 成立，
即 P0-B 的 104 个 legacy run（`git_commit` 前七位为 `34edddb`，已对照 ledger §8e.1 核实）。

主实验的 624 个 run 全部为 `augmentation = main_uniform`，不满足第三个条件，**永不进入该分支**。

### 3.3 为什么接受

这 104 个 run 已完成、已分析、其结论已定稿于 `docs/03_EVIDENCE_LEDGER.md` §8e。
它们的溯源由另外的机制承担：四源 SHA、`git_commit = 34edddb`、以及 ledger 中的逐条记录。
豁免只是为了让它们在新代码下仍返回 `COMPLETED_SKIP`，不触发重跑。

### 3.4 残留项（未修，如实记录）

该豁免目前仅按 `dataset` 与 `augmentation` 收口，**未按 `git_commit` 收口**。
更严的写法应追加 `and normalized.get("git_commit", "").startswith("34edddb")`，
把作用域收紧到字面意义上的那 104 个 run。本批次运行期间禁止 commit，故未实施。

**该项不影响主实验的 624 个 run**（它们不进入该分支），但应在批次结束后补上。

### 3.5 一个单点风险，记录在案

若在补跑单个 cifar10 run 时**漏传 `--augmentation main_uniform`**，该 run 会：
(a) 走 if 分支写进 P0-B legacy 目录，覆盖已完成的 104 个之一；
(b) 同时落入本节的 provenance 豁免分支。
一次输入疏漏会同时移除两层保护。发车脚本 `main624_launch.py` 的 preflight 含 legacy 命名空间
正则断言与 104 个 legacy 目录的完好性检查，但手动补跑不经过该脚本。

---

## 4. 两条事实记录

### 4.1 16 个增强敏感性 run 已迁出

`outputs_p0b_backup/` 原含 120 个目录：104 个 P0-B legacy run，
加 16 个增强敏感性检查 run（`p0b_cifar10_main_uniform_{GEO_DIV|GEO_SG1|RND_D1|RND_S1}_R_high_seed{0..3}`）。

后者与前者在 `(dataset, backbone, exp_id, reliance, seed)` 上碰撞（该键不含 augmentation），
导致分析脚本报 `duplicate metadata for design cell`。

已于 2026-07-29 迁至 `/root/autodl-tmp/outputs_aug16/`，两处目录数分别为 104 与 16。
分析脚本的去重键随后扩展为含 `augmentation`，并新增 `--augmentation` 过滤参数
（默认 `main_uniform`）与组内 augmentation 一致性断言。

### 4.2 批次 D 的复现基线 commit

批次 D 在云端运行时仓库处于 `02981d9`，**非 `8055759`**。其后四个 commit
（`f8b4785`、`1a22e26`、`fe2e8f0`、`2f3606d`）均为文档与 PSI 分析代码，不含训练代码。
此为 `P0B_PREREG_ADDENDUM_01.md` §6 第 3 条的待办，此前一直未落，现记于此。

---

### 4.3 `MAIN_PREREG_01.md` 内的失效交叉引用（erratum）

`MAIN_PREREG_01.md` §2.3（seed 数固定为 4 的理由）中写道：

> 增强敏感性检查显示禁用水平翻转后方差约翻倍（§9.2）

但 §9 为「事前预期与负值预案」，其第 2 条为禁止功效外推，与方差无关；
所引的 `P_G'` 对照表实际位于 **§5.2**（增强策略的理由）。

**该文件受 SHA 门控（`1ccd6245…` @ `68dff0b`），不得修改。**
本条作为 erratum 记录：§2.3 中的「§9.2」应读作「§5.2」。
该笔误不影响任何判据、任何数值、任何冻结产物，仅为章节指针错误。

---

## 5. 已完成的验证及其效力边界

| 验证 | 结果 | 它证明了什么 | 它**没有**证明什么 |
|---|---|---|---|
| legacy run 的 `COMPLETED_SKIP` 实测 | 通过 | 在 `32edce6` 下重新计算的 expected metadata（含 `architecture_signature`、`operator_signature`、`parameter_count`、`nominal_flops_equality_signature`、`channel_order_sha256`）与 `34edddb` 下写入的值逐字相等 | 前向/反向传播产出相同数值 |
| CIFAR 数据管线逐位回归 | 通过 | 数据张量未变 | 同上 |
| GRU CUDA 冒烟（debug, 2 epoch） | 通过（rc=0） | GRU 路径在 GPU 上可训练 | GRU 结果的正确性 |
| `verify_outdirs.py` | 624 唯一 | 目录单射 | — |
| 独立复算目录名（本件作者） | 624 唯一，与发车计划集合相等，与 legacy 命名空间零交集 | 同上，且交叉验证 | — |
| 发车 preflight | 14/14 | 冻结产物完整、HEAD 已 push、环境符合 | — |
| 分析脚本对 104 个 legacy run 的回归 | 48/48 断言在 ±0.01 pp 内 | 分析口径可复现 ledger §8e.2 | runner 的数值惰性 |

**明确的边界：** 上表最强的一条（`COMPLETED_SKIP`）证明的是**配置与架构派生逻辑未变**，
不是**训练动力学未变**。前者是后者的必要条件，不是充分条件。

### 5.1 尚待执行的直接检验

主实验的 cifar10-Mamba 组包含 16 个与增强敏感性检查完全同格的 cell：
`{GEO_DIV, GEO_SG1, RND_D1, RND_S1} × R_high × seed{0..3}`，`augmentation = main_uniform`。

**受限对比的定义（已核实，见 §6.2，出自受 SHA 门控的 `MAIN_PREREG_01.md` §5.2）：**

```text
P_G' = GEO_DIV − GEO_SG1
P_R' = RND_D1  − RND_S1
```

**记录值（翻转禁用，即 `main_uniform`）：** `P_G' = +3.50 [2.14, 4.86]`，`P_R' ≈ 0`。

#### 判定标准（须在执行复算之前冻结，本节即为该冻结）

| # | 标准 | 通过条件 |
|---|---|---|
| A | 主判据 | 复算的 `P_G'` **点估计**落在 `[2.14, 4.86]` 内 |
| B | 辅助判据 | 复算的 `P_R'` 的 95% CI 包含零 |
| C | 描述性 | 逐 cell 报告 16 个 tail-window validation accuracy 与旧值之差，**不设阈值** |

A 与 B 任一不通过 → §7 的数值惰性主张撤回，差异须调查并如实报告。

#### 本检验的功效边界（须与结果同时报告）

记录值的 CI 宽达 2.72 pp（增强敏感性检查本身方差翻倍所致），
因此**标准 A 对小幅系统性偏移的检出力很低**——一个 0.5 pp 的系统偏移不会使 A 失败。

标准 C 的逐 cell 绝对值比较在原理上更敏感（对比可能保持而绝对水平整体平移），
但本研究**没有对"纯 GPU 非确定性下重跑同一配置的差异幅度"的基线测量**，
因而无法为 C 事前设定阈值。C 只作描述性报告，
**不得在看到数值之后为其补设阈值并据以宣称通过或失败**。

#### 其他约束

1. 两次运行的 commit 不同、GPU 非确定性存在，故检验标准为量级与符号，不是逐位相等。
2. 该比较**不是**预注册判据的一部分，结果不得用于支持或否定 `MAIN_PREREG_01.md` §7 的任何命题。
   它只回答"代码改动是否惰性"这一个工程问题。
3. 该检验不消耗额外机时（16 个 cell 已在 624 个 run 之内）。

---

## 6. 引用核实状态

本件起草时 `MAIN_PREREG_01.md` 与 `docs/03_EVIDENCE_LEDGER.md` 不在起草者的可见上下文内，
部分引用凭先前阅读记忆写出。核实结果如下。

### 6.1 已核实（对照 ledger 原文）

| 引用 | 状态 |
|---|---|
| P0-B legacy 的 `git_commit` 前七位为 `34edddb` | **确认**。§8e.1 原文：「`git_commit` 前七位均为 `34edddb`」 |
| `§8e.2 五个主对比` 的八组值 | **确认**，与分析脚本回归所用目标值逐位相同 |
| `§8e.8` 成本实测 grid8 1690 s / grid32 6995 s / 5 进程 104 runs 约 27 h | **确认** |
| `§8e.1` 的 `git_dirty=true` 出现在 103/104 | **确认** |
| 四个冻结源 SHA 与 ledger SHA 全批一致 | **确认**（§8e.1） |

### 6.2 已核实：增强敏感性检查的受限对比定义

该定义**不在 ledger 中，而在 `MAIN_PREREG_01.md` §5.2**，原文：

```text
16-run 增强敏感性检查（CIFAR-10、grid32、GEO_SG1/GEO_DIV/RND_S1/RND_D1 × 4 seed，
formal 模式），与 P0-B 同名条件对照：

| P_G' = GEO_DIV − GEO_SG1 | +2.84 [2.23, 3.46] | +3.50 [2.14, 4.86] |
```
（左列为翻转开启，右列为翻转禁用。）

**这一位置优于 ledger：** `MAIN_PREREG_01.md` 受 SHA 门控
（`1ccd6245d2583e563086e09379b1e20944110a109c7374d7b8576b62a66cb7ff` @ `68dff0b`），
该定义因而是**冻结的、可用 `git show 68dff0b:MAIN_PREREG_01.md` 逐字复核的**，
不需要任何事后补记。

CI 宽度由 1.23 pp 增至 2.72 pp（2.2 倍），即 `MAIN_PREREG_01.md` §2.3 功效依据
所述的"方差约翻倍"。

**一致性交叉验证（起草者独立复算）：** 由 §8e.2 与 §8e.5，R_high 下
`mean(GEO_S*) = 74.29`、`P_G = +4.23`，故 `GEO_DIV ≈ 78.52`；
由 ③ `= +0.30`、④ `= +2.96`，并依设计取 `GEO_SG4 ≈ GEO_SG3`，解得 `GEO_SG1 ≈ 75.85`，
`GEO_DIV − GEO_SG1 ≈ +2.67`，与记录值 `+2.84` 相差 0.17 pp，
差异可由该近似解释。两条独立路径一致。

### 6.3 已核实（对照 `MAIN_PREREG_01.md` 原文）

| 引用 | 状态 |
|---|---|
| §3 冻结产物清单为六项 | **确认**，六项 SHA 与本件 §1、§2.3 所引逐位相同 |
| §4.1 五数据集样本数表 | **确认**，与本件 §2.4 的表逐字相同 |
| §7 M3 的预测序 `min(Organ) > cifar10 > eurosat` 与三档判定 | **确认** |
| §7.1 的投票规则「≥4 个数据集」 | **确认**。标题为「多重比较处理」，投票规则在其正文内；末句「M3 为单一检验，不参与投票」 |
| §7.2 探索性清单含 GRU 稳健性复现、天花板诊断、⑤ | **确认** |
| §10 记录项含 `dataset` / `augmentation` / `backbone` | **确认** |

**全部引用已核实完毕，无待决项。**

---

## 7. 结论与 claim boundary

**可主张：** `68dff0b..32edce6` 的全部代码改动由"新增维度"（backbone）、"新增校验"
（EuroSAT SHA 门、样本数断言、EuroSAT 类别绑定）、"新增记录"（`data_split_provenance`）
与"目录命名扩展"四类构成。四类均不进入训练计算路径。Mamba 默认路径的配置派生逻辑
经 `COMPLETED_SKIP` 实测确认与 `68dff0b` 等价。

**不可主张：** 上述证据不足以断言训练动力学逐位不变。§5.1 的 16 格比较执行之前，
主实验 `main_uniform` 路径的数值惰性属**推断**而非**实测**。

**论文中的表述规范：** 复现说明应写明预注册冻结于 `68dff0b`、执行于 `32edce6`，
并指向本件；不得写成"代码与预注册完全一致"。

---

## 8. 落盘后

1. 在 `.gitattributes` 中加入 `CODE_DELTA_68dff0b_32edce6.md -text`
2. `git add` 并 commit（**须在 624 个 run 全部完成之后**）
3. 计算并记录本件自身的 SHA-256
4. 在 `docs/03_EVIDENCE_LEDGER.md` 追加一条指向本件的记录
5. 执行 §5.1 的 16 格比较，结果追加到本件 §5.1 之下
6. 实施 §3.4 的豁免收口

---

## 5.2 §5.1 的执行结果（2026-08-02 追加）

**执行时点：** 624/624 全部完成之后，任何新 commit 之前。
**数据来源：** 复算侧 `/root/autodl-tmp/outputs_main`，归档侧 `/root/autodl-tmp/outputs_aug16`。

### 判定

| # | 标准 | 结果 | 判定 |
|---|---|---|---|
| A | 复算 `P_G'` 点估计落在 `[2.14, 4.86]` 内 | `+3.45 [+2.12, +4.78]` | **通过** |
| B | 复算 `P_R'` 的 95% CI 含零 | `+0.21 [-1.18, +1.59]` | **通过** |
| C | 逐 cell 报告 16 个尾窗 val acc 与旧值之差，不设阈值 | 见下表 | 描述性，无判定 |

A 与 B 均通过，故 §7 的数值惰性主张**不撤回**。

### 归档侧的自洽性校验

归档侧 16 格重算得 `P_G' = +3.50 [+2.14, +4.86]`，与 §5.1 记录值逐位相同，
确认所比较的正是判据注册时所指的那批 run。归档侧 `P_R' = +0.11 [-1.35, +1.58]`。

### 判据 C：逐格尾窗 validation accuracy（%）

| 条件 | 量 | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---:|---:|---:|---:|
| GEO_DIV | 复算 | 76.67 | 77.93 | 77.54 | 77.00 |
| GEO_DIV | 归档 | 76.70 | 77.92 | 77.61 | 77.01 |
| GEO_DIV | 差 | -0.03 | +0.01 | -0.07 | -0.01 |
| GEO_SG1 | 复算 | 74.41 | 74.17 | 73.36 | 73.41 |
| GEO_SG1 | 归档 | 74.41 | 74.12 | 73.32 | 73.39 |
| GEO_SG1 | 差 | +0.00 | +0.05 | +0.04 | +0.02 |
| RND_D1 | 复算 | 63.07 | 62.51 | 63.72 | 63.03 |
| RND_D1 | 归档 | 63.08 | 62.48 | 63.73 | 62.95 |
| RND_D1 | 差 | -0.01 | +0.03 | -0.01 | +0.08 |
| RND_S1 | 复算 | 62.78 | 62.52 | 62.40 | 63.82 |
| RND_S1 | 归档 | 62.81 | 62.75 | 62.42 | 63.80 |
| RND_S1 | 差 | -0.03 | -0.23 | -0.02 | +0.02 |

无一格逐位重现（符合 §5.1 其他约束第 1 条的预期）。16 个差跨 `-0.23` 到 `+0.08`，
15 个绝对值低于 0.10。最大者为 `RND_S1` seed 1 的 `-0.23`，该格解释了
`P_R'` 由归档侧 `+0.11` 移至复算侧 `+0.21` 的大部分。

**依 §5.1，不为 C 补设阈值。** 本研究没有"纯 GPU 非确定性下重跑同一配置的差异幅度"
的基线测量，故上述差值无法在任何方向上拆分为代码导致的部分与非确定性导致的部分。

### 功效边界（依 §5.1 要求，与结果同时报告）

记录值 CI 宽 2.72 pp，标准 A 对小幅系统性偏移检出力很低；通过 A 不等于排除小幅偏移。

### 论文对应位置

`sections/results.tex` §6.9（Code-inertness check）与 Table 11。
