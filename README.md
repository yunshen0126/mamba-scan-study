# CIFAR-10 行遮掩重建辅助实验

测试核心命题：**给行扫描视觉模型加一个"遮行重建"辅助损失，能否提升性能；
且推理时丢掉辅助头，零额外开销。**

对照组：
- `baseline`  纯行扫描，无辅助（基线）
- `row_aux`   **你的方法**：遮整行 + 重建（测垂直）
- `col_aux`   对照：遮整列 + 重建（测水平，应帮助更小 → 证明垂直专属）

---

## 一、环境安装（Windows + 3060）

```bat
:: 建议先建一个干净的 conda 环境
conda create -n cifar python=3.10 -y
conda activate cifar

:: 装 PyTorch（CUDA 版，3060 用 cu121 即可；以官网为准）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: 装其余依赖
pip install numpy
```

验证 GPU 可用：
```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
应输出 `True NVIDIA GeForce RTX 3060`。

---

## 二、CIFAR-10 数据：在哪下、放哪里

### 方式 A（推荐，最省事）：让程序自动下载
**什么都不用做**。第一次运行 `run.py` 时会自动下载 CIFAR-10 到 `data/` 目录，
约 170MB，下完后存到 `data/cifar-10-batches-py/`。之后不再重复下载。

### 方式 B（手动下载，适合下载慢/断网的情况）
1. 打开官方页面下载 **CIFAR-10 python version**：
   https://www.cs.toronto.edu/~kriz/cifar.html
   下载文件名为：`cifar-10-python.tar.gz`

2. 在本项目根目录下新建 `data` 文件夹，把 `cifar-10-python.tar.gz` 放进去：
   ```
   cifar_rowmask/
   └── data/
       └── cifar-10-python.tar.gz
   ```

3. 解压它（Windows 可用 7-Zip / WinRAR，或命令行 `tar -xzf`）。
   解压后应得到这样的结构（**关键：必须是这个文件夹名**）：
   ```
   cifar_rowmask/
   └── data/
       └── cifar-10-batches-py/
           ├── data_batch_1
           ├── data_batch_2
           ├── data_batch_3
           ├── data_batch_4
           ├── data_batch_5
           ├── test_batch
           └── batches.meta
   ```
   程序检测到 `data/cifar-10-batches-py/` 就会直接用，不再下载。

> 如果你的数据放在别的盘/别的路径，运行时加 `--data-root D:\你的路径`。

---

## 三、怎么跑

**第 0 步：冒烟测试（30 秒，确认能跑通，不报错）**
```bat
python run.py --smoke
```
看到每个 variant 都打印了 train/test、最后出 SUMMARY，就说明环境和代码都 OK。

**第 1 步：先确认基线能学（只跑 baseline，看精度正常）**
```bat
python run.py --variants baseline --seeds 0 --epochs 50
```
CIFAR-10 上单向 GRU 行扫描基线大致能到 70%+（不会很高，因为是弱骨干，
但只要明显高于 10%、稳定上升，就说明管线没问题）。

**第 2 步：跑完整对照（你的核心实验）**
```bat
python run.py --epochs 100 --seeds 0 1 2
```
跑完看 SUMMARY 里两个关键对比：
- `row_aux` 相对 `baseline`：**正值 = 你的方法有效**
- `row_aux` 相对 `col_aux`：**正值 = 垂直专属性成立**（不是泛泛正则）

**第 3 步（可选）：扫 λ，找最佳辅助权重**
```bat
python run.py --variants baseline row_aux --aux-lambda 0.1 --seeds 0 --epochs 100
python run.py --variants baseline row_aux --aux-lambda 1.0 --seeds 0 --epochs 100
```

**用真实 Mamba（装好 mamba-ssm 后）：**
```bat
python run.py --block-type mamba --epochs 100 --seeds 0 1 2
```

---

## 四、判断标准（什么算成功）

| 现象 | 含义 |
|------|------|
| `row_aux` > `baseline`（稳定，超出 std） | 方法有效 ✅ |
| `row_aux` > `col_aux` | 垂直专属，不是泛泛正则 ✅ |
| 推理速度 `row_aux` ≈ `baseline` | 零开销卖点成立 ✅（代码会打印 img/s）|
| 3 个种子都同向 | 不是运气 ✅ |

即使提升只有 0.3~0.5% 但**稳定**且 row > col，就是一个能写的结果。

---

## 五、常见问题

- **显存不够（8GB）**：加 `--batch-size 64`，或 `--d-model 96`。
- **太慢**：先用 `--epochs 50` 看趋势；`--no-amp` 关掉混合精度只在调试时用（开着更快）。
- **Windows 多进程报错**：默认 `num_workers=0` 已避免；不要随意调大。
- **mamba-ssm 装不上**：正常，Windows 上很难装。直接用默认 `gru` 也能验证方法，
  GRU 是单向行扫描，同样存在垂直缺陷。

---

## 六、文件说明

```
cifar_rowmask/
├── README.md          本文件
├── requirements.txt   依赖
├── config.py          所有超参数（可直接改）
├── data.py            CIFAR-10 加载（自动下载 / 手动放置）
├── masking.py         行遮掩 / 列遮掩 / patch 切分
├── model.py           行扫描骨干 + MLP 重建头
├── train.py           训练循环（分类损失 + λ·辅助损失）
└── run.py             主入口（跑对照实验 + 汇总）
```

方法实现要点（与你之前验证过的设计一致）：
- 重建头用 **MLP**（不是卷积）→ 逼信息进主干，而非解码器自己看邻居
- 遮 **行**（不是列）→ 测垂直方向
- 辅助头 **推理时丢弃** → 零额外开销（代码里 evaluate/测速都只用 backbone）
- 两次前向：干净图算分类损失，遮掩图算辅助损失，加权相加
