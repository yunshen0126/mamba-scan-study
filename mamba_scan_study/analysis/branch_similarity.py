import csv
import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch

from mamba_scan_study.data.real_datasets import build_real_loaders
from mamba_scan_study.models.backbone import MultiDirBackbone
import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref


DEVICE = torch.device("cpu")
torch.set_num_threads(1)
# The installed CUDA extensions cannot execute on CPU; use the package's exact
# reference equations for this analysis-only inference path.
mamba_simple.causal_conv1d_fn = None
mamba_simple.selective_scan_fn = selective_scan_ref
DATA_ROOT = "/home/tuling/datasets/mamba_scan_study"
CHECKPOINT_ROOT = "mamba_scan_study/outputs/stage1_seed0/checkpoints"
OUTDIR = "mamba_scan_study/outputs/branch_similarity"
BLOCKS = ("gru", "mamba")
GRIDS = (8, 16, 32)
VARIANTS = {
    "real_4dir": "row,col,diag,anti_diag",
    "same_row_4": "row,row,row,row",
}
BRANCH_DIRS = ("row", "col", "diag", "anti_diag")
N_SAMPLES = 2000
BATCH_SIZE = 512


def cka_from_gram(left, right):
    left = left - left.mean(dim=0, keepdim=True) - left.mean(dim=1, keepdim=True) + left.mean()
    right = right - right.mean(dim=0, keepdim=True) - right.mean(dim=1, keepdim=True) + right.mean()
    numerator = (left * right).sum()
    denominator = torch.sqrt((left * left).sum() * (right * right).sum())
    return float((numerator / denominator).item())


def gram(features):
    features = features.float()
    return features @ features.T


def checkpoint_path(block, grid, variant):
    return os.path.join(CHECKPOINT_ROOT, f"{block}_{variant}_grid{grid}_seed0.pt")


def load_model(block, grid, variant):
    checkpoint = torch.load(checkpoint_path(block, grid, variant), map_location=DEVICE)
    patch_size = 32 // grid
    model = MultiDirBackbone(
        img_size=32,
        patch_size=patch_size,
        in_chans=3,
        d_model=64,
        n_layers=2,
        block_type=block,
        n_classes=10,
        branch_dirs=VARIANTS[variant],
        pos_mode="xy_learned",
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def analyze_model(model, loader, branch_dirs):
    spatial_parts = [[] for _ in model.branches]
    pooled_parts = [[] for _ in model.branches]
    cosine_sums = {pair: 0.0 for pair in combinations(range(4), 2)}
    seen = 0
    with torch.no_grad():
        for images, _labels in loader:
            images = images[: N_SAMPLES - seen].to(DEVICE)
            features = [branch.forward_features(images) for branch in model.branches]
            for index, item in enumerate(features):
                spatial_parts[index].append(
                    torch.nn.functional.adaptive_avg_pool2d(
                        item.permute(0, 3, 1, 2), output_size=(4, 4)
                    ).flatten(1).cpu()
                )
                pooled_parts[index].append(item.mean(dim=(1, 2)).cpu())
            for i, j in cosine_sums:
                cosine_sums[(i, j)] += float(
                    torch.nn.functional.cosine_similarity(
                        features[i].flatten(1), features[j].flatten(1), dim=1
                    ).sum().item()
                )
            seen += images.shape[0]
            if seen >= N_SAMPLES:
                break

    spatial = [torch.cat(parts, dim=0) for parts in spatial_parts]
    pooled = [torch.cat(parts, dim=0) for parts in pooled_parts]
    spatial_grams = [gram(item) for item in spatial]
    pooled_grams = [gram(item) for item in pooled]
    rows = []
    for i, j in combinations(range(4), 2):
        rows.append(
            {
                "pair": f"{i}-{j}",
                "branch_dir_i": branch_dirs[i],
                "branch_dir_j": branch_dirs[j],
                "cka_spatial": cka_from_gram(spatial_grams[i], spatial_grams[j]),
                "cka_pooled": cka_from_gram(pooled_grams[i], pooled_grams[j]),
                "cosine": cosine_sums[(i, j)] / seen,
            }
        )
    return rows


def write_outputs(rows):
    os.makedirs(OUTDIR, exist_ok=True)
    detail_path = os.path.join(OUTDIR, "branch_similarity.csv")
    fields = [
        "block_type",
        "grid",
        "variant",
        "pair",
        "branch_dir_i",
        "branch_dir_j",
        "cka_spatial",
        "cka_pooled",
        "cosine",
    ]
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for block in BLOCKS:
        for grid in GRIDS:
            for variant in VARIANTS:
                values = [
                    row["cka_spatial"]
                    for row in rows
                    if row["block_type"] == block
                    and row["grid"] == grid
                    and row["variant"] == variant
                ]
                summary.append(
                    {
                        "block_type": block,
                        "grid": grid,
                        "variant": variant,
                        "mean_offdiag_cka": float(np.mean(values)),
                    }
                )
    summary_path = os.path.join(OUTDIR, "branch_similarity_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_type", "grid", "variant", "mean_offdiag_cka"])
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for axis, block in zip(axes, BLOCKS):
        for variant, marker in (("real_4dir", "o"), ("same_row_4", "s")):
            selected = [
                row for row in summary if row["block_type"] == block and row["variant"] == variant
            ]
            axis.plot(
                [row["grid"] for row in selected],
                [row["mean_offdiag_cka"] for row in selected],
                marker=marker,
                label=variant,
            )
        axis.set_title(block.upper())
        axis.set_xlabel("Grid")
        axis.set_xticks(GRIDS)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Mean off-diagonal spatial CKA")
    fig.savefig(os.path.join(OUTDIR, "mean_offdiag_cka_vs_grid.png"), dpi=180)
    plt.close(fig)


def main():
    loader_cache = {}
    os.makedirs(OUTDIR, exist_ok=True)
    partial_path = os.path.join(OUTDIR, "branch_similarity_partial.json")
    if os.path.isfile(partial_path):
        with open(partial_path, encoding="utf-8") as handle:
            rows = json.load(handle)
    else:
        rows = []
    completed = {(row["block_type"], row["grid"], row["variant"]) for row in rows}
    for block in BLOCKS:
        for grid in GRIDS:
            for variant in VARIANTS:
                key = (block, grid, variant)
                if key in completed:
                    continue
                batch_size = 64 if block == "mamba" and grid == 32 else BATCH_SIZE
                loader_key = (grid, batch_size)
                if loader_key not in loader_cache:
                    loader_cache[loader_key] = build_real_loaders(
                        "cifar10",
                        DATA_ROOT,
                        batch_size,
                        num_workers=0,
                        img_size=32,
                        download=False,
                        generator=torch.Generator().manual_seed(0),
                    )[1]
                print(f"START {block} grid{grid} {variant}", flush=True)
                model = load_model(block, grid, variant)
                result_rows = analyze_model(
                    model, loader_cache[loader_key], VARIANTS[variant].split(",")
                )
                for row in result_rows:
                    rows.append({"block_type": block, "grid": grid, "variant": variant, **row})
                del model
                with open(partial_path, "w", encoding="utf-8") as handle:
                    json.dump(rows, handle)
                completed.add(key)
                print(f"DONE {block} grid{grid} {variant}", flush=True)
    write_outputs(rows)


if __name__ == "__main__":
    main()
