import random

import torch
import torch.nn as nn

from .scan_utils import SCAN_DIRS, flatten_scan, restore_scan, row_major_to_scan_mask


POS_MODES = ("none", "seq_learned", "xy_learned", "xy_sincos")


try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


class GRUBlock(nn.Module):
    """Single-direction GRU block with residual connection."""

    def __init__(self, d_model, bidirectional=False, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        hidden = d_model // 2 if bidirectional else d_model
        self.gru = nn.GRU(d_model, hidden, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y, _ = self.gru(self.norm(x))
        return x + self.dropout(y)


class MambaBlock(nn.Module):
    """Mamba block. Requires mamba-ssm."""

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        try:
            self.mamba = Mamba(d_model=d_model, use_fast_path=False)
        except TypeError:
            self.mamba = Mamba(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


def build_block(block_type, d_model, bidirectional=False, dropout=0.0):
    if block_type == "gru":
        return GRUBlock(d_model, bidirectional=bidirectional, dropout=dropout)
    if block_type == "mamba":
        if not HAS_MAMBA:
            raise RuntimeError(
                "mamba-ssm is not installed. Install `mamba-ssm causal-conv1d` "
                "or set block_type='gru'."
            )
        if bidirectional:
            raise ValueError("MambaBlock is causal and does not support bidirectional=True")
        return MambaBlock(d_model, dropout=dropout)
    raise ValueError(f"unknown block_type={block_type!r}")


def _sincos_1d(values, dim):
    if dim <= 0:
        return values.new_zeros(values.numel(), 0)
    half = dim // 2
    if half == 0:
        return values[:, None]
    freqs = torch.arange(half, dtype=values.dtype, device=values.device)
    freqs = 1.0 / (10000 ** (freqs / max(1, half - 1)))
    angles = values[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat([emb, values[:, None]], dim=1)
    return emb[:, :dim]


def build_xy_sincos(height, width, d_model):
    y_dim = d_model // 2
    x_dim = d_model - y_dim
    y = torch.linspace(0, 1, height) * 2.0 * torch.pi
    x = torch.linspace(0, 1, width) * 2.0 * torch.pi
    y_emb = _sincos_1d(y, y_dim)
    x_emb = _sincos_1d(x, x_dim)
    pos = torch.zeros(height, width, d_model)
    pos[:, :, :y_dim] = y_emb[:, None, :]
    pos[:, :, y_dim:] = x_emb[None, :, :]
    return pos


class ScanBackbone(nn.Module):
    """Image -> patch tokens -> one directional sequence model -> spatial token features."""

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        d_model=128,
        n_layers=4,
        block_type="gru",
        bidirectional=False,
        scan_dir="row",
        dropout=0.0,
        shuffle_order=False,
        shuffle_seed=None,
        pos_mode="seq_learned",
    ):
        super().__init__()
        if scan_dir not in SCAN_DIRS:
            raise ValueError(f"scan_dir must be one of {SCAN_DIRS}, got {scan_dir!r}")
        if pos_mode not in POS_MODES:
            raise ValueError(f"pos_mode must be one of {POS_MODES}, got {pos_mode!r}")
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.H = self.W = img_size // patch_size
        self.L = self.H * self.W
        self.d_model = d_model
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.scan_dir = scan_dir
        self.shuffle_order = shuffle_order
        self.shuffle_seed = shuffle_seed
        self.pos_mode = pos_mode

        self.patch_embed = nn.Conv2d(in_chans, d_model, patch_size, stride=patch_size)
        if pos_mode == "seq_learned":
            self.pos_emb = nn.Parameter(torch.randn(1, self.L, d_model) * 0.02)
        else:
            self.register_parameter("pos_emb", None)
        if pos_mode == "xy_learned":
            self.row_pos = nn.Parameter(torch.randn(self.H, d_model) * 0.02)
            self.col_pos = nn.Parameter(torch.randn(self.W, d_model) * 0.02)
        else:
            self.register_parameter("row_pos", None)
            self.register_parameter("col_pos", None)
        if pos_mode == "xy_sincos":
            self.register_buffer("xy_sincos", build_xy_sincos(self.H, self.W, d_model), persistent=False)
        else:
            self.register_buffer("xy_sincos", torch.zeros(self.H, self.W, d_model), persistent=False)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList(
            [
                build_block(block_type, d_model, bidirectional=bidirectional, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        if shuffle_order:
            if shuffle_seed is None:
                permutation = torch.randperm(self.L)
            else:
                generator = torch.Generator().manual_seed(int(shuffle_seed))
                permutation = torch.randperm(self.L, generator=generator)
            self.register_buffer("shuffle_perm", permutation, persistent=True)
            inv = torch.empty(self.L, dtype=torch.long)
            inv[self.shuffle_perm.cpu()] = torch.arange(self.L)
            self.register_buffer("shuffle_inv", inv, persistent=True)
        else:
            self.register_buffer("shuffle_perm", torch.arange(self.L), persistent=False)
            self.register_buffer("shuffle_inv", torch.arange(self.L), persistent=False)

    def positional_tokens(self):
        if self.pos_mode == "none":
            return None
        if self.pos_mode == "seq_learned":
            return self.pos_emb
        if self.pos_mode == "xy_learned":
            pos2d = self.row_pos[:, None, :] + self.col_pos[None, :, :]
        elif self.pos_mode == "xy_sincos":
            pos2d = self.xy_sincos
        else:
            raise ValueError(f"unknown pos_mode={self.pos_mode!r}")
        return flatten_scan(pos2d.unsqueeze(0), self.scan_dir)

    def forward_features(self, x, token_mask=None):
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # (B, H, W, D)
        x = flatten_scan(x, self.scan_dir)           # (B, L, D)
        mask = row_major_to_scan_mask(token_mask, self.H, self.W, self.scan_dir)
        pos = self.positional_tokens()

        if self.shuffle_order:
            x = x.index_select(1, self.shuffle_perm)
            if pos is not None:
                pos = pos.index_select(1, self.shuffle_perm)
            if mask is not None:
                mask = mask.index_select(1, self.shuffle_perm)

        if mask is not None:
            mt = self.mask_token.view(1, 1, -1)
            x = torch.where(mask.unsqueeze(-1), mt, x)

        if pos is not None:
            x = x + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.shuffle_order:
            x = x.index_select(1, self.shuffle_inv)
        return restore_scan(x, self.H, self.W, self.scan_dir)


def expand_branch_dirs(branch_dirs=None, n_branches=None):
    if branch_dirs is None:
        branch_dirs = ["row"]
    if isinstance(branch_dirs, str):
        if branch_dirs == "4dir":
            branch_dirs = ["row", "col", "diag", "anti_diag"]
        elif branch_dirs == "same_random":
            if n_branches is None:
                raise ValueError("n_branches is required for branch_dirs='same_random'")
            branch_dirs = [random.choice(SCAN_DIRS)] * n_branches
        elif branch_dirs == "random":
            if n_branches is None:
                raise ValueError("n_branches is required for branch_dirs='random'")
            branch_dirs = [random.choice(SCAN_DIRS) for _ in range(n_branches)]
        else:
            branch_dirs = [part.strip() for part in branch_dirs.split(",") if part.strip()]
    branch_dirs = list(branch_dirs)
    if n_branches is not None and len(branch_dirs) != n_branches:
        if len(branch_dirs) == 1:
            branch_dirs = branch_dirs * n_branches
        else:
            raise ValueError("len(branch_dirs) must equal n_branches unless one dir is given")
    for scan_dir in branch_dirs:
        if scan_dir not in SCAN_DIRS:
            raise ValueError(f"branch dir must be one of {SCAN_DIRS}, got {scan_dir!r}")
    return branch_dirs


class MultiDirBackbone(nn.Module):
    """Parallel directional branches fused into a classifier."""

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        d_model=128,
        n_layers=4,
        block_type="gru",
        bidirectional=False,
        n_classes=10,
        branch_dirs=None,
        n_branches=None,
        fusion="mean",
        dropout=0.0,
        shuffle_order=False,
        shuffle_seed=None,
        pos_mode="seq_learned",
    ):
        super().__init__()
        self.branch_dirs = expand_branch_dirs(branch_dirs, n_branches)
        self.H = self.W = img_size // patch_size
        self.L = self.H * self.W
        self.d_model = d_model
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.fusion = fusion
        self.branches = nn.ModuleList(
            [
                ScanBackbone(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    d_model=d_model,
                    n_layers=n_layers,
                    block_type=block_type,
                    bidirectional=bidirectional,
                    scan_dir=scan_dir,
                    dropout=dropout,
                    shuffle_order=shuffle_order,
                    shuffle_seed=None
                    if shuffle_seed is None
                    else int(shuffle_seed) + branch_index,
                    pos_mode=pos_mode,
                )
                for branch_index, scan_dir in enumerate(self.branch_dirs)
            ]
        )
        if fusion == "mean":
            head_dim = d_model
            self.proj = nn.Identity()
        elif fusion == "concat":
            head_dim = d_model
            self.proj = nn.Linear(d_model * len(self.branches), d_model)
        else:
            raise ValueError("fusion must be 'mean' or 'concat'")
        self.head = nn.Linear(head_dim, n_classes)

    def forward_features(self, x, token_mask=None):
        feats = [branch.forward_features(x, token_mask=token_mask) for branch in self.branches]
        if self.fusion == "mean":
            return torch.stack(feats, dim=0).mean(dim=0)
        x = torch.cat(feats, dim=-1)
        return self.proj(x)

    def forward(self, x, token_mask=None):
        feat2d = self.forward_features(x, token_mask=token_mask)
        pooled = feat2d.mean(dim=(1, 2))
        logits = self.head(pooled)
        return logits, feat2d

    def classify_from_target(self, feat2d, target_row, target_col):
        batch_idx = torch.arange(feat2d.shape[0], device=feat2d.device)
        target_feat = feat2d[
            batch_idx,
            target_row.to(feat2d.device),
            target_col.to(feat2d.device),
        ]
        return self.head(target_feat)


class ChannelSplitBackbone(nn.Module):
    """Four channel-partitioned scan paths with a shared patch embedding."""

    VARIANTS = {
        "channel_real_4dir": ("row", "col", "diag", "anti_diag"),
        "channel_same_row_4": ("row", "row", "row", "row"),
        "channel_same_perm_4": ("row", "row", "row", "row"),
        "channel_rand_perm_4": ("row", "row", "row", "row"),
    }

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        d_model=128,
        n_layers=4,
        block_type="gru",
        n_classes=10,
        variant="channel_real_4dir",
        pos_mode="seq_learned",
        dropout=0.0,
        shuffle_seed=0,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown channel-split variant={variant!r}")
        if d_model % 4:
            raise ValueError("channel-split d_model must be divisible by 4")
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        if pos_mode not in POS_MODES:
            raise ValueError(f"pos_mode must be one of {POS_MODES}, got {pos_mode!r}")

        self.variant = variant
        self.shuffle_seed = int(shuffle_seed)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = self.W = img_size // patch_size
        self.L = self.H * self.W
        self.d_model = d_model
        self.group_width = d_model // 4
        self.n_layers = n_layers
        self.pos_mode = pos_mode
        self.branch_dirs = self.VARIANTS[variant]
        self.patch_embed = nn.Conv2d(in_chans, d_model, patch_size, stride=patch_size)
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.group_width)) for _ in range(4)]
        )

        row_pos = []
        col_pos = []
        seq_pos = []
        xy_sincos = []
        for _ in range(4):
            if pos_mode == "seq_learned":
                seq_pos.append(nn.Parameter(torch.randn(1, self.L, self.group_width) * 0.02))
                row_pos.append(None)
                col_pos.append(None)
            elif pos_mode == "xy_learned":
                row_pos.append(nn.Parameter(torch.randn(self.H, self.group_width) * 0.02))
                col_pos.append(nn.Parameter(torch.randn(self.W, self.group_width) * 0.02))
                seq_pos.append(None)
            elif pos_mode == "xy_sincos":
                xy_sincos.append(build_xy_sincos(self.H, self.W, self.group_width))
                row_pos.append(None)
                col_pos.append(None)
                seq_pos.append(None)
            else:
                row_pos.append(None)
                col_pos.append(None)
                seq_pos.append(None)
        self.row_pos = nn.ParameterList([item for item in row_pos if item is not None])
        self.col_pos = nn.ParameterList([item for item in col_pos if item is not None])
        self.seq_pos = nn.ParameterList([item for item in seq_pos if item is not None])
        if pos_mode == "xy_sincos":
            self.register_buffer("xy_sincos", torch.stack(xy_sincos), persistent=False)
        else:
            self.register_buffer(
                "xy_sincos", torch.zeros(4, self.H, self.W, self.group_width), persistent=False
            )

        generator = torch.Generator().manual_seed(self.shuffle_seed)
        permutations = []
        inverses = []
        shared_permutation = None
        for group in range(4):
            if variant == "channel_same_perm_4":
                if shared_permutation is None:
                    shared_permutation = torch.randperm(self.L, generator=generator)
                permutation = shared_permutation.clone()
            elif variant == "channel_rand_perm_4":
                permutation = torch.randperm(self.L, generator=generator)
            else:
                permutation = torch.arange(self.L)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(self.L)
            permutations.append(permutation)
            inverses.append(inverse)
        self.register_buffer("channel_permutations", torch.stack(permutations), persistent=True)
        self.register_buffer("channel_inverse_permutations", torch.stack(inverses), persistent=True)

        self.group_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [build_block(block_type, self.group_width, dropout=dropout) for _ in range(n_layers)]
                )
                for _ in range(4)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def _position(self, group, batch_size, device):
        if self.pos_mode == "none":
            return None
        if self.pos_mode == "seq_learned":
            position = self.seq_pos[group]
        elif self.pos_mode == "xy_learned":
            pos2d = self.row_pos[group][:, None, :] + self.col_pos[group][None, :, :]
            position = flatten_scan(pos2d.unsqueeze(0), self.branch_dirs[group])
        else:
            position = flatten_scan(
                self.xy_sincos[group].unsqueeze(0), self.branch_dirs[group]
            )
        return position.to(device).expand(batch_size, -1, -1)

    def forward_features(self, x, token_mask=None):
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        groups = x.chunk(4, dim=-1)
        outputs = []
        for group, (group_x, blocks) in enumerate(zip(groups, self.group_blocks)):
            tokens = flatten_scan(group_x, self.branch_dirs[group])
            mask = row_major_to_scan_mask(token_mask, self.H, self.W, self.branch_dirs[group])
            position = self._position(group, tokens.shape[0], tokens.device)
            permutation = self.channel_permutations[group]
            inverse = self.channel_inverse_permutations[group]
            if not torch.equal(permutation, torch.arange(self.L, device=permutation.device)):
                tokens = tokens.index_select(1, permutation)
                if position is not None:
                    position = position.index_select(1, permutation)
                if mask is not None:
                    mask = mask.index_select(1, permutation)
            if mask is not None:
                mask_token = self.mask_tokens[group].view(1, 1, -1)
                tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)
            if position is not None:
                tokens = tokens + position
            for block in blocks:
                tokens = block(tokens)
            if not torch.equal(permutation, torch.arange(self.L, device=permutation.device)):
                tokens = tokens.index_select(1, inverse)
            outputs.append(restore_scan(tokens, self.H, self.W, self.branch_dirs[group]))
        return self.norm(torch.cat(outputs, dim=-1))

    def forward(self, x, token_mask=None):
        feat2d = self.forward_features(x, token_mask=token_mask)
        logits = self.head(feat2d.mean(dim=(1, 2)))
        return logits, feat2d


class RowScanBackbone(MultiDirBackbone):
    """Compatibility wrapper. Defaults to the old single row-scan baseline."""

    def __init__(self, *args, scan_dir="row", **kwargs):
        kwargs.pop("branch_dirs", None)
        kwargs.pop("n_branches", None)
        super().__init__(*args, branch_dirs=[scan_dir], n_branches=1, **kwargs)
