# AGENTS.md — MambaScan 项目常驻规则

每个会话自动生效。以下规则优先于任何单轮指令中的相反倾向。
若某轮任务与本文件冲突,先指出冲突,不要自行取舍。

---

## 0. 前提纠正(每个新会话自动遵守,不主动复述)

旧结论"row = col = 100%,Mamba 无方向信息损失"**是错的,已作废**。

错因:mean-pooling readout 存在捷径——分类器可直接从全图读取 source patch,
无需沿扫描序列把信息传到 target token。

改成 **target-token readout** 后的真实结果(VerticalCarry / single_patch):

| grid | col (匹配) | row (不匹配) |
|---|---|---|
| 16 | 1.000 | 0.507 |
| 24 | 1.000 | 0.511 |
| 32 | 1.000 | 0.519 |

**不匹配扫描 = 随机猜测。方向信息传递的能力缺口真实存在,且是灾难性失败。**

与此同时,CIFAR-10 上 `real_4dir − same_row_4` 仅约 **+0.64pp**
(而多分支容量贡献 2.07pp)。

**本项目的核心命题**:能力缺口真实存在,但自然图像分类几乎没有使用它。
Phase 2 要解释这个鸿沟从何而来、边界在哪。

### 硬禁止
- 不得使用 mean-pooling 做任何 carry 实验
- 不得复述"Mamba 无方向信息损失"
- 不得在结果出来后修改任何预注册判定标准

注:mean-pooling 禁令**仅针对 synthetic carry 实验**(target-token 任务)。
真实图像分类任务使用 spatial mean-pool + linear head 是正确做法,不受此限。

---

## 1. 长任务:发车即下车(最重要的一条)

**禁止逐 epoch 轮询进度。** 曾有一轮做了 40+ 次 progress 检查,
浪费大量 token 且完全无用——盯着看不会让训练更快。

正确做法:

```bash
nohup python -m <module> ... > logs/<run_name>.log 2>&1 &
echo $!   # 报告 PID
```
Windows: `Start-Process -NoNewWindow -RedirectStandardOutput`

启动后**立即结束回合**,只报告 PID 和日志路径。

训练期间:
- 不检查进度
- 不"看一眼"
- 不汇报中间 epoch / 准确率

**只在两种情况下再开口:**
1. 全部完成 → 交最终报告
2. 崩溃 / OOM / 预注册闸门未通过 → 立即停止并汇报

---

## 2. 诚实性

- **绝不编造任何数据**(时间、显存、准确率、成本)。跑不了就说跑不了。
- 无 shell / 文件权限时,直接说"无权限"并停止。不要计划、不要重试、不要模拟。
- 负结果与正结果同等保存。不得删除或隐藏反例。
- 若新结果与第 0 节的前提冲突,**停下汇报**,不要忽略也不要自行调和。

---

## 3. 实现前先交计划

写任何实验代码前,先列出:
- 要改动哪些文件、具体改什么
- 新增哪些 config 字段和结果列
- 是否引入新的架构混杂

**等确认后再动手。** 不要顺手重构无关代码。不要一次实现多个 Experiment。

---

## 4. 实验规范

### 对照公平性
- 比较任何两个模型前,**参数量必须匹配**,或显式报告差异
- 配对实验:同 seed 下 data order 必须完全一致,保存 `data_order_hash`
- row vs shuffle_row:除扫描顺序外,所有可训练参数初值必须逐元素相同

### 指标
- 主指标 `last_acc`,`best_acc` 只作附加记录
- **收敛判据看 delta,不看绝对准确率**。
  主指标是配对差值,绝对准确率可以一直缓慢上升,
  只要两个模型同步上升,delta 早已稳定。
  判据:epoch 80–100 区间 delta 的波动 < 其 seed 间标准差。
- LR:cosine schedule + 5 epoch warmup,总长锁定,末端 LR → 0

### 必存字段
每个 run:完整 epoch 曲线(train/test acc、loss、lr)、逐样本 test logits (npz)、
`params`、`flops_proxy`、峰值显存、`seed`、`data_order_hash`、`git_commit`、`config_hash`

### 算力
- 新配置先做 timing pilot,不要直接开跑大矩阵
- 注意 **overhead-bound**:若 token 数翻 16 倍而 epoch 时间不变,
  说明 GPU 在空转(micro-batch 太小),不是模型慢。先修 batch 再估成本。
- OOM 就记录 OOM 并停止,**不要自动降级重试**

---

## 5. 报告格式

- 只交结果,不叙述过程。不要复述每个 run 的中间状态。
- 结论用表格,不用大段散文。
- 明确区分:**实测值** vs **外推估计值**。
- 不确定就说不确定。不要用"应该""大概"包装未验证的数字。

---

## 6. 已知环境约束

- 本地:Windows + WSL,RTX 3060 Laptop 6GB,`mair` conda 环境
- `mamba-ssm 1.1.3.post1` 与 `causal-conv1d 1.2.1` 的 fast path **ABI 不兼容**,
  当前走 unfused 慢路径。WSL 无 nvcc,**不要在本地折腾编译**。
  若在干净 CUDA 环境(云 GPU)运行,可花 20 分钟试一次 fast path。
- 云端:AutoDL

## 7. 版本控制
- 每次修改代码前先 git add -A && git commit,确保有干净回退点
- 改完跑 git diff --stat 汇报改动范围
- **训练进程运行期间严禁任何 git 写操作**(checkout/reset/stash/clean)。
  outputs/ 在训练时被持续写入,git 写操作会破坏它。
  训练期间只允许 git status / git diff 只读查看。

## 8. 对照实验的设计检查
提出任何"控制变量"对照前,必须先做两件事:
1. 显式论证被控制的变量真的被控制住了,并指定一个 manipulation check 指标
2. 检查该对照是否与已有条件数学等价(避免同义反复)
教训一:双线性上采样被误认为不改变信息覆盖,实际插值把邻居信息
        混进了每个 patch,导致对照失效。
教训二:NN 上采样 2x + 2x2 卷积,与原图 + 1x1 卷积是同一个函数
        (卷积核作用在同一像素的四份拷贝上 = 权重求和的 1x1 卷积)。
        此类对照跑了等于没跑。

## 9. outputs 不进 git
outputs/ 是数据不是代码,训练时被持续写入。
执行:git rm -r --cached mamba_scan_study/outputs/ 并加入 .gitignore

## 10. 汇总脚本复用
- 复用任何汇总/分析脚本前,先确认其分组维度与当前实验的自变量匹配。
  教训:`tail_80_100_summary.csv` 按 block×grid 汇总,套用到以 variant 为自变量的
  channel-split 上产生过错误数字。
