import torch.nn as nn

from mamba_scan_study.models.backbone import (
    GRUBlock,
    HAS_MAMBA,
    MambaBlock,
    MultiDirBackbone,
    RowScanBackbone as _StudyRowScanBackbone,
    ScanBackbone,
    build_block,
)


class RowScanBackbone(_StudyRowScanBackbone):
    """Compatibility wrapper for the original root-level training scripts."""

    def forward_features(self, x, token_mask=None):
        feat2d = super().forward_features(x, token_mask=token_mask)
        return feat2d.reshape(x.shape[0], self.H * self.W, self.d_model)

    def forward(self, x, token_mask=None):
        feats = self.forward_features(x, token_mask)
        pooled = feats.mean(dim=1)
        logits = self.head(pooled)
        feat2d = feats.reshape(x.shape[0], self.H, self.W, self.d_model)
        return logits, feat2d


class ReconHead(nn.Module):
    """MLP 解码器：从每个位置自己的特征，重建该位置的原始 patch 像素。"""
    def __init__(self, d_model, patch_size, in_chans):
        super().__init__()
        self.out_dim = patch_size * patch_size * in_chans
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.out_dim),
        )

    def forward(self, feat2d):
        # feat2d: (B, H, W, D) -> (B, H, W, C*p*p)
        return self.net(feat2d)
