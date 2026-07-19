import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
TINY_MEAN = (0.4802, 0.4481, 0.3975)
TINY_STD = (0.2302, 0.2265, 0.2262)


def build_cifar10_loaders(
    data_root,
    batch_size,
    num_workers=0,
    download=True,
    generator=None,
    img_size=32,
):
    import torchvision
    import torchvision.transforms as transforms

    os.makedirs(data_root, exist_ok=True)
    resize = [] if img_size == 32 else [transforms.Resize((img_size, img_size), antialias=True)]
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *resize,
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_tf = transforms.Compose(
        [*resize, transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    batches_dir = os.path.join(data_root, "cifar-10-batches-py")
    if os.path.isdir(batches_dir):
        download = False
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=train_tf, download=download
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=test_tf, download=download
    )
    return _make_loaders(train_set, test_set, batch_size, num_workers, generator=generator)


def build_cifar100_loaders(
    data_root,
    batch_size,
    num_workers=0,
    download=True,
    generator=None,
    img_size=32,
):
    import torchvision
    import torchvision.transforms as transforms

    os.makedirs(data_root, exist_ok=True)
    resize = [] if img_size == 32 else [transforms.Resize((img_size, img_size), antialias=True)]
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *resize,
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_tf = transforms.Compose(
        [*resize, transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )
    batches_dir = os.path.join(data_root, "cifar-100-python")
    if os.path.isdir(batches_dir):
        download = False
    train_set = torchvision.datasets.CIFAR100(
        root=data_root, train=True, transform=train_tf, download=download
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_root, train=False, transform=test_tf, download=download
    )
    return _make_loaders(train_set, test_set, batch_size, num_workers, generator=generator)


class TinyImageNetValDataset(Dataset):
    """Official Tiny-ImageNet validation layout with annotation-file labels."""

    def __init__(self, val_dir, class_to_idx, transform=None):
        self.transform = transform
        annotation_path = os.path.join(val_dir, "val_annotations.txt")
        image_dir = os.path.join(val_dir, "images")
        if not os.path.isfile(annotation_path) or not os.path.isdir(image_dir):
            raise FileNotFoundError(
                "Expected Tiny-ImageNet val/images and val_annotations.txt under "
                f"{val_dir}"
            )
        self.samples = []
        with open(annotation_path, encoding="utf-8") as handle:
            for line in handle:
                image_name, wnid, *_ = line.rstrip("\n").split("\t")
                if wnid in class_to_idx:
                    self.samples.append((os.path.join(image_dir, image_name), class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_tiny_imagenet_loaders(data_root, batch_size, num_workers=0, img_size=64, generator=None):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")
    val_dir = os.path.join(data_root, "tiny-imagenet-200", "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            "Tiny-ImageNet not found. Expected train/val under "
            f"{os.path.join(data_root, 'tiny-imagenet-200')}"
        )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ]
    )
    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    class_folder_layout = any(
        entry.is_dir() for entry in os.scandir(val_dir) if entry.name != "images"
    )
    if class_folder_layout:
        val_set = datasets.ImageFolder(val_dir, transform=val_tf)
    else:
        val_set = TinyImageNetValDataset(val_dir, train_set.class_to_idx, transform=val_tf)
    return _make_loaders(train_set, val_set, batch_size, num_workers, generator=generator)


def _make_loaders(train_set, test_set, batch_size, num_workers, generator=None):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def build_real_loaders(
    dataset,
    data_root,
    batch_size,
    num_workers=0,
    img_size=32,
    download=True,
    generator=None,
):
    if dataset in ("cifar10", "cifar10_up64"):
        return build_cifar10_loaders(
            data_root,
            batch_size,
            num_workers,
            download=download,
            generator=generator,
            img_size=img_size,
        )
    if dataset == "cifar100":
        return build_cifar100_loaders(
            data_root,
            batch_size,
            num_workers,
            download=download,
            generator=generator,
            img_size=img_size,
        )
    if dataset == "tiny-imagenet":
        return build_tiny_imagenet_loaders(
            data_root,
            batch_size,
            num_workers,
            img_size=img_size,
            generator=generator,
        )
    raise ValueError(
        "dataset must be 'cifar10', 'cifar10_up64', 'cifar100', or 'tiny-imagenet'"
    )
