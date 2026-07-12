"""
CIFAR-10 数据加载。
支持两种方式：
  1) 自动下载（最省事）：download=True，会自动下到 data_root/cifar-10-batches-py/
  2) 手动放置：把官方的 cifar-10-python.tar.gz 解压后的
     cifar-10-batches-py 文件夹放到 data_root/ 下即可。
详见 README.md。
"""
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# CIFAR-10 的官方均值/标准差（用于归一化）
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_cifar10_loaders(data_root, batch_size, num_workers=0, download=True):
    """
    返回 (train_loader, test_loader)。
    train 带简单数据增强（随机裁剪+翻转），test 只做归一化。
    """
    os.makedirs(data_root, exist_ok=True)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # 检查数据是否已存在
    batches_dir = os.path.join(data_root, "cifar-10-batches-py")
    already_there = os.path.isdir(batches_dir)
    if already_there:
        print(f"[data] 检测到已有 CIFAR-10：{batches_dir}")
        download = False
    else:
        if download:
            print(f"[data] 未检测到数据，将自动下载到：{data_root}")
        else:
            raise FileNotFoundError(
                f"未找到 CIFAR-10 数据于 {batches_dir}，"
                f"且 download=False。请按 README 手动放置，或设 download=True。"
            )

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=train_tf, download=download
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=test_tf, download=download
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"[data] train={len(train_set)}  test={len(test_set)}")
    return train_loader, test_loader
