"""
训练与评估。
核心：两次前向
  1) 干净图像 -> 分类 logits -> 分类损失（与推理一致）
  2) 遮掩后图像 -> 主干特征 -> 重建被遮 patch -> 辅助损失
  总损失 = 分类损失 + λ · 辅助损失
推理时只走第 1 步，重建头丢弃 -> 零额外开销。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from masking import make_row_mask, make_col_mask, image_to_patches


def lr_lambda_fn(epoch, warmup_epochs, total_epochs):
    """warmup + cosine 学习率系数。"""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def current_lambda(epoch, cfg):
    """λ 线性 warmup。"""
    if cfg.lambda_warmup_epochs <= 0:
        return cfg.aux_lambda
    frac = min(1.0, (epoch + 1) / cfg.lambda_warmup_epochs)
    return cfg.aux_lambda * frac


def make_mask(variant, B, H, W, mask_ratio, device):
    if variant == "row_aux":
        return make_row_mask(B, H, W, mask_ratio, device, avoid_first=True)
    if variant == "col_aux":
        return make_col_mask(B, H, W, mask_ratio, device)
    return None  # baseline 无辅助


def train_one_epoch(backbone, recon_head, loader, optimizer, scaler,
                    cfg, device, variant, epoch):
    backbone.train()
    if recon_head is not None:
        recon_head.train()

    use_aux = variant in ("row_aux", "col_aux")
    lam = current_lambda(epoch, cfg) if use_aux else 0.0
    H, W, p = backbone.H, backbone.W, cfg.patch_size

    total, correct = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=cfg.use_amp):
            # --- 主任务：干净图像 ---
            logits, _ = backbone(images, token_mask=None)
            cls_loss = F.cross_entropy(logits, labels)
            loss = cls_loss

            # --- 辅助任务：遮掩后重建 ---
            if use_aux and lam > 0:
                mask2d = make_mask(variant, B, H, W, cfg.mask_ratio, device)  # (B,H,W)
                token_mask = mask2d.reshape(B, H * W)                         # (B,L)
                _, feat2d = backbone(images, token_mask=token_mask)          # (B,H,W,D)
                pred = recon_head(feat2d)                                     # (B,H,W,C*p*p)
                target = image_to_patches(images, p)                          # (B,H,W,C*p*p)
                aux_loss = F.mse_loss(pred[mask2d], target[mask2d])
                loss = cls_loss + lam * aux_loss

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            params = list(backbone.parameters())
            if recon_head is not None:
                params += list(recon_head.parameters())
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        correct += (logits.argmax(1) == labels).sum().item()
        total += B

    return correct / total, lam


@torch.no_grad()
def evaluate(backbone, loader, device):
    backbone.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = backbone(images, token_mask=None)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total


@torch.no_grad()
def measure_inference_speed(backbone, loader, device, n_batches=20):
    """测推理速度：images/sec。验证辅助头丢弃后与基线一样快。"""
    backbone.eval()
    import time
    # 预热
    it = iter(loader)
    for _ in range(2):
        images, _ = next(it)
        backbone(images.to(device), token_mask=None)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n_img = 0
    it = iter(loader)
    for _ in range(n_batches):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        backbone(images.to(device), token_mask=None)
        n_img += images.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return n_img / dt if dt > 0 else 0.0
