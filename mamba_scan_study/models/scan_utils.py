from functools import lru_cache

import torch


SCAN_DIRS = ("row", "col", "diag", "anti_diag")


def _validate_scan_dir(scan_dir):
    if scan_dir not in SCAN_DIRS:
        raise ValueError(f"scan_dir must be one of {SCAN_DIRS}, got {scan_dir!r}")


@lru_cache(maxsize=None)
def scan_indices(height, width, scan_dir):
    """Return row-major token indices in the requested scan order."""
    _validate_scan_dir(scan_dir)
    coords = []
    if scan_dir == "row":
        coords = [(r, c) for r in range(height) for c in range(width)]
    elif scan_dir == "col":
        coords = [(r, c) for c in range(width) for r in range(height)]
    elif scan_dir == "diag":
        for s in range(height + width - 1):
            for r in range(height):
                c = s - r
                if 0 <= c < width:
                    coords.append((r, c))
    elif scan_dir == "anti_diag":
        for s in range(height + width - 1):
            for r in range(height):
                c = width - 1 - (s - r)
                if 0 <= c < width:
                    coords.append((r, c))
    return tuple(r * width + c for r, c in coords)


def scan_permutation(height, width, scan_dir, device=None):
    return torch.tensor(scan_indices(height, width, scan_dir), dtype=torch.long, device=device)


def flatten_scan(x, scan_dir):
    """
    x: (B, H, W, D) or (B, H, W)
    returns tokens flattened in scan_dir order.
    """
    height, width = x.shape[1], x.shape[2]
    perm = scan_permutation(height, width, scan_dir, device=x.device)
    flat = x.reshape(x.shape[0], height * width, *x.shape[3:])
    return flat.index_select(1, perm)


def restore_scan(tokens, height, width, scan_dir):
    """
    Inverse of flatten_scan.
    tokens: (B, L, D) or (B, L)
    returns (B, H, W, D) or (B, H, W)
    """
    perm = scan_permutation(height, width, scan_dir, device=tokens.device)
    out = torch.empty_like(tokens)
    out.index_copy_(1, perm, tokens)
    return out.reshape(tokens.shape[0], height, width, *tokens.shape[2:])


def row_major_to_scan_mask(token_mask, height, width, scan_dir):
    if token_mask is None:
        return None
    mask2d = token_mask.reshape(token_mask.shape[0], height, width)
    return flatten_scan(mask2d, scan_dir)

