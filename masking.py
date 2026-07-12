"""
遮掩工具。
- make_row_mask：遮整行（你的方法，测垂直方向）
- make_col_mask：遮整列（对照组，测水平方向）
- image_to_patches：把图像切成 patch 像素，作为重建目标
"""
import torch


def make_row_mask(B, H, W, mask_ratio, device, avoid_first=True):
    """
    随机遮掉若干整行。返回 (B, H, W) 的 bool 张量，True=被遮。
    avoid_first=True 时不遮第 0 行（第 0 行没有上方历史可参考）。
    """
    mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    n_rows = max(1, int(round(H * mask_ratio)))
    start = 1 if avoid_first else 0
    candidate = torch.arange(start, H, device=device)
    for b in range(B):
        perm = candidate[torch.randperm(len(candidate), device=device)][:n_rows]
        mask[b, perm, :] = True
    return mask


def make_col_mask(B, H, W, mask_ratio, device):
    """随机遮掉若干整列（水平对照组）。返回 (B, H, W) bool。"""
    mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    n_cols = max(1, int(round(W * mask_ratio)))
    candidate = torch.arange(0, W, device=device)
    for b in range(B):
        perm = candidate[torch.randperm(len(candidate), device=device)][:n_cols]
        mask[b, :, perm] = True
    return mask


def image_to_patches(img, patch_size):
    """
    把图像切成不重叠 patch，每个 patch 展平成像素向量。
    img: (B, C, H_img, W_img)  ->  (B, H, W, C*p*p)
    其中 H = H_img/p, W = W_img/p。顺序与行扫描一致（row-major）。
    """
    B, C, Hi, Wi = img.shape
    p = patch_size
    # (B, C, H, p, W, p)
    x = img.reshape(B, C, Hi // p, p, Wi // p, p)
    # -> (B, H, W, C, p, p) -> (B, H, W, C*p*p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, Hi // p, Wi // p, C * p * p)
    return x
