"""
全局配置。所有超参数集中在这里，方便修改。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- 数据 ----
    data_root: str = "data"          # CIFAR-10 所在目录（见 README）
    img_size: int = 32
    patch_size: int = 4              # 32/4 = 8 -> 8x8 = 64 个 patch
    in_chans: int = 3
    n_classes: int = 10
    num_workers: int = 0             # Windows 上保持 0 最稳；想加速可设 2

    # ---- 模型 ----
    block_type: str = "gru"          # 'gru'（默认，到处能跑）或 'mamba'（需装 mamba-ssm）
    bidirectional: bool = False      # False = 单向行扫描（垂直缺陷最明显，推荐用于验证方法）
    d_model: int = 128
    n_layers: int = 4

    # ---- 辅助损失 ----
    aux_lambda: float = 0.5          # 辅助损失权重 λ（要扫的关键超参）
    mask_ratio: float = 0.25         # 遮掉多少比例的行/列
    lambda_warmup_epochs: int = 5    # λ 从 0 线性升到 aux_lambda 的轮数

    # ---- 训练 ----
    epochs: int = 100
    batch_size: int = 128            # 12GB 显存用 128，8GB 用 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    use_amp: bool = True             # 混合精度，3060 上更快更省显存
    grad_clip: float = 1.0

    # ---- 实验 ----
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    # 要跑的模型变体（对照组）
    variants: List[str] = field(default_factory=lambda: [
        "baseline",      # 纯行扫描，无辅助
        "row_aux",       # 你的方法：行遮掩重建辅助
        "col_aux",       # 对照：列遮掩（应该帮助更小 -> 证明垂直专属）
    ])

    outdir: str = "outputs"
