import argparse
import csv
import json
import os
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Dataset

from mamba_scan_study.models.backbone import HAS_MAMBA, MultiDirBackbone


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--outdir", required=True)
    p.add_argument("--block-type", default="mamba", choices=["gru", "mamba"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", nargs="+", default=["row", "same_row_4", "real_4dir"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-per-category", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def build_test_loader(data_root, batch_size, num_workers):
    import torchvision
    import torchvision.transforms as transforms

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=tf, download=False
    )
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    model = MultiDirBackbone(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        in_chans=cfg["in_chans"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        block_type=cfg["block_type"],
        bidirectional=cfg["bidirectional"],
        n_classes=cfg["n_classes"],
        branch_dirs=cfg["branch_dirs"],
        dropout=cfg["dropout"],
        shuffle_order=cfg["shuffle_order"],
        pos_mode=cfg["pos_mode"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def checkpoint_path(args, variant):
    return os.path.join(args.checkpoint_dir, f"{args.block_type}_{variant}_seed{args.seed}.pt")


def ensure_models(args, device):
    if args.block_type == "mamba" and not HAS_MAMBA:
        raise RuntimeError("mamba_ssm is not available in this environment")
    models = {}
    ckpts = {}
    for variant in args.variants:
        path = checkpoint_path(args, variant)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        models[variant], ckpts[variant] = load_model(path, device)
    return models, ckpts


def unnormalize(x):
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(3, 1, 1)
    y = x * std + mean
    return y.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()


@torch.no_grad()
def collect_predictions(models, loader, device):
    rows = []
    for images, labels, indices in loader:
        images = images.to(device)
        labels = labels.to(device)
        row_batch = [
            {"index": int(idx), "label": int(label), "label_name": CIFAR_CLASSES[int(label)]}
            for idx, label in zip(indices, labels.cpu())
        ]
        for variant, model in models.items():
            logits, _ = model(images)
            probs = logits.softmax(dim=1)
            preds = probs.argmax(dim=1)
            for i, out in enumerate(row_batch):
                label = labels[i]
                pred = preds[i]
                out[f"{variant}_pred"] = int(pred)
                out[f"{variant}_pred_name"] = CIFAR_CLASSES[int(pred)]
                out[f"{variant}_correct"] = bool(pred == label)
                out[f"{variant}_true_conf"] = float(probs[i, label].detach().cpu())
                out[f"{variant}_pred_conf"] = float(probs[i, pred].detach().cpu())
        rows.extend(row_batch)
    return rows


def write_prediction_csv(path, rows):
    fieldnames = [
        "index",
        "label",
        "label_name",
        "row_pred",
        "row_pred_name",
        "row_correct",
        "row_true_conf",
        "row_pred_conf",
        "same_row_4_pred",
        "same_row_4_pred_name",
        "same_row_4_correct",
        "same_row_4_true_conf",
        "same_row_4_pred_conf",
        "real_4dir_pred",
        "real_4dir_pred_name",
        "real_4dir_correct",
        "real_4dir_true_conf",
        "real_4dir_pred_conf",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_cases(rows, max_per_category):
    selected = []
    used = set()

    def add_cases(category, candidates, key, reverse=True):
        count = 0
        for row in sorted(candidates, key=key, reverse=reverse):
            if row["index"] in used:
                continue
            row = dict(row)
            row["case_category"] = category
            selected.append(row)
            used.add(row["index"])
            count += 1
            if count >= max_per_category:
                break

    add_cases(
        "row_wrong_four_branch_correct",
        [
            r
            for r in rows
            if not r["row_correct"] and r["same_row_4_correct"] and r["real_4dir_correct"]
        ],
        lambda r: r["real_4dir_true_conf"] + r["same_row_4_true_conf"],
    )
    add_cases(
        "same_wrong_real_correct",
        [r for r in rows if not r["same_row_4_correct"] and r["real_4dir_correct"]],
        lambda r: r["real_4dir_true_conf"] - r["same_row_4_true_conf"],
    )
    add_cases(
        "both_correct_real_higher_confidence",
        [
            r
            for r in rows
            if r["same_row_4_correct"]
            and r["real_4dir_correct"]
            and r["real_4dir_true_conf"] > r["same_row_4_true_conf"]
        ],
        lambda r: r["real_4dir_true_conf"] - r["same_row_4_true_conf"],
    )
    negative = [r for r in rows if r["same_row_4_correct"] and not r["real_4dir_correct"]]
    if negative:
        add_cases(
            "real_4dir_negative_same_correct",
            negative,
            lambda r: r["same_row_4_true_conf"] - r["real_4dir_true_conf"],
        )
    else:
        add_cases(
            "real_4dir_no_advantage_lower_confidence",
            [
                r
                for r in rows
                if r["same_row_4_correct"]
                and r["real_4dir_correct"]
                and r["same_row_4_true_conf"] > r["real_4dir_true_conf"]
            ],
            lambda r: r["same_row_4_true_conf"] - r["real_4dir_true_conf"],
        )
    return selected


def gradcam_heatmap(model, image, target):
    model.zero_grad(set_to_none=True)
    logits, feat = model(image)
    feat.retain_grad()
    score = logits[0, target]
    score.backward()
    grad = feat.grad[0]
    fmap = feat.detach()[0]
    weights = grad.mean(dim=(0, 1))
    heat = torch.relu((fmap * weights.view(1, 1, -1)).sum(dim=-1))
    return heat.detach().float().cpu().numpy(), logits.detach()


def branch_gradcam(model, image, target):
    model.zero_grad(set_to_none=True)
    branch_feats = []
    for branch in model.branches:
        feat = branch.forward_features(image)
        feat.retain_grad()
        branch_feats.append(feat)
    fused = torch.stack(branch_feats, dim=0).mean(dim=0)
    fused.retain_grad()
    logits = model.head(fused.mean(dim=(1, 2)))
    logits[0, target].backward()

    out = []
    for feat in branch_feats:
        grad = feat.grad[0]
        fmap = feat.detach()[0]
        weights = grad.mean(dim=(0, 1))
        heat = torch.relu((fmap * weights.view(1, 1, -1)).sum(dim=-1))
        out.append(heat.detach().float().cpu().numpy())
    fused_grad = fused.grad[0]
    fused_map = fused.detach()[0]
    weights = fused_grad.mean(dim=(0, 1))
    fused_heat = torch.relu((fused_map * weights.view(1, 1, -1)).sum(dim=-1))
    out.append(fused_heat.detach().float().cpu().numpy())
    return out, logits.detach()


def spatial_distribution(heat):
    heat = np.maximum(np.asarray(heat, dtype=np.float64), 0.0)
    total = heat.sum()
    if total <= 1e-12:
        return np.full_like(heat, 1.0 / heat.size)
    return heat / total


def scale_maps_for_display(maps):
    values = np.concatenate([m.ravel() for m in maps])
    vmax = max(float(np.percentile(values, 98)), 1e-12)
    return [np.clip(m / vmax, 0, 1) for m in maps]


def scaled_difference(real_heat, same_heat):
    diff = real_heat - same_heat
    denom = max(abs(float(diff.min())), abs(float(diff.max())), 1e-12)
    return diff, diff / denom


def region_name(bbox):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    horiz = "left" if cx < 10.7 else "center" if cx < 21.3 else "right"
    vert = "top" if cy < 10.7 else "middle" if cy < 21.3 else "bottom"
    return f"{vert}-{horiz}"


def bbox_from_map(arr, positive=True):
    work = np.asarray(arr)
    score = work if positive else -work
    peak = np.unravel_index(np.argmax(score), score.shape)
    threshold = max(float(np.percentile(score, 75)), float(score[peak]) * 0.45)
    allowed = score >= threshold
    component = {peak}
    stack = [peak]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= ny < score.shape[0]
                and 0 <= nx < score.shape[1]
                and allowed[ny, nx]
                and (ny, nx) not in component
            ):
                component.add((ny, nx))
                stack.append((ny, nx))
    ys = np.array([p[0] for p in component])
    xs = np.array([p[1] for p in component])
    margin = 1
    x0 = max(0, (int(xs.min()) - margin) * 4)
    y0 = max(0, (int(ys.min()) - margin) * 4)
    x1 = min(32, (int(xs.max()) + 1 + margin) * 4)
    y1 = min(32, (int(ys.max()) + 1 + margin) * 4)
    return x0, y0, x1, y1


def add_box(ax, bbox, color="#00A6FF", label=None):
    x0, y0, x1, y1 = bbox
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.8)
    )
    if label:
        ax.annotate(
            label,
            xy=((x0 + x1) / 2, (y0 + y1) / 2),
            xytext=(1.5, 5.0),
            textcoords="data",
            color=color,
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2),
        )


def plot_heatmap_case(outdir, case, image_np, heats, logits_by_variant, dpi):
    distributions = {name: spatial_distribution(heat) for name, heat in heats.items()}
    display_values = scale_maps_for_display(
        [distributions[name] for name in ("row", "same_row_4", "real_4dir")]
    )
    display_maps = dict(zip(("row", "same_row_4", "real_4dir"), display_values))
    real_minus_same_raw, real_minus_same = scaled_difference(
        distributions["real_4dir"], distributions["same_row_4"]
    )
    bbox = bbox_from_map(real_minus_same_raw, positive=True)
    loc = region_name(bbox)
    label = int(case["label"])
    titles = [
        f"Input\ntrue: {CIFAR_CLASSES[label]}",
        f"row\npred {case['row_pred_name']} / true {case['row_true_conf']:.2f}",
        f"same_row_4\npred {case['same_row_4_pred_name']} / true {case['same_row_4_true_conf']:.2f}",
        f"real_4dir\npred {case['real_4dir_pred_name']} / true {case['real_4dir_true_conf']:.2f}",
        "real_4dir - same_row_4\nred: more real | blue: more same",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.1), constrained_layout=True)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].imshow(image_np)
    axes[0].set_title(titles[0], fontsize=9)
    add_box(axes[0], bbox, label=loc)
    for ax, variant, title in zip(axes[1:4], ["row", "same_row_4", "real_4dir"], titles[1:4]):
        ax.imshow(image_np, alpha=0.55)
        ax.imshow(
            display_maps[variant],
            cmap="magma",
            alpha=0.72,
            extent=(0, 32, 32, 0),
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        ax.set_title(title, fontsize=9)
        add_box(ax, bbox)
    axes[4].imshow(
        real_minus_same,
        cmap="RdBu_r",
        extent=(0, 32, 32, 0),
        vmin=-1,
        vmax=1,
        interpolation="bilinear",
    )
    axes[4].set_title(titles[4], fontsize=9)
    add_box(axes[4], bbox)
    fig.suptitle(
        f"{case['case_category']} | CIFAR index {case['index']} | marked region: {loc}",
        fontsize=10,
    )
    stem = f"case_{case['case_category']}_idx{case['index']}"
    save_figure(fig, outdir, stem, dpi)
    plt.close(fig)
    return {
        "stem": stem,
        "case": case,
        "region": loc,
        "bbox": bbox,
        "real_minus_same_peak_mass": float(real_minus_same_raw.max()),
        "caption": make_case_caption(case, loc),
        "render_data": {
            "image": image_np,
            "display_maps": display_maps,
            "difference": real_minus_same,
            "bbox": bbox,
        },
    }


def make_case_caption(case, loc):
    same_delta = case["real_4dir_true_conf"] - case["same_row_4_true_conf"]
    if case["case_category"] == "same_wrong_real_correct":
        verdict = (
            "This is the strongest visual test for a possible direction contribution; check whether "
            "the positive difference region marks object structure that same_row_4 misses."
        )
    elif case["case_category"] == "row_wrong_four_branch_correct":
        verdict = (
            "Both four-branch models fix the row baseline here, so the figure mainly tests whether "
            "the improvement is already explained by multi-branch capacity."
        )
    elif case["case_category"] == "both_correct_real_higher_confidence":
        verdict = (
            "Both controls are correct, so only a spatial pattern unique to real_4dir would support "
            "a cautious direction-specific interpretation."
        )
    else:
        verdict = (
            "This is a negative or ambiguous case; it should be used to limit any direction-specific claim."
        )
    return (
        f"Index {case['index']} ({case['label_name']}), category {case['case_category']}. "
        f"The marked positive real_4dir-minus-same_row_4 region is in the {loc} of the image. "
        f"True-class confidence changes from same_row_4={case['same_row_4_true_conf']:.3f} "
        f"to real_4dir={case['real_4dir_true_conf']:.3f} (delta {same_delta:+.3f}). "
        f"{verdict}"
    )


def plot_branch_case(outdir, case, image_np, branch_names, branch_heats, dpi):
    distributions = [spatial_distribution(heat) for heat in branch_heats]
    display_maps = scale_maps_for_display(distributions)
    bbox = bbox_from_map(distributions[-1], positive=True)
    loc = region_name(bbox)
    similarity = mean_pairwise_cosine(distributions[:-1])
    cols = ["Input"] + branch_names + ["fused"]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.1), constrained_layout=True)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].imshow(image_np)
    axes[0].set_title("Input", fontsize=9)
    add_box(axes[0], bbox, label=loc)
    for ax, name, heat in zip(axes[1:], cols[1:], display_maps):
        ax.imshow(image_np, alpha=0.55)
        ax.imshow(
            heat,
            cmap="magma",
            alpha=0.72,
            extent=(0, 32, 32, 0),
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        ax.set_title(name, fontsize=9)
        add_box(ax, bbox)
    fig.suptitle(
        f"real_4dir branch maps | CIFAR index {case['index']} | "
        f"mean branch cosine={similarity:.2f}",
        fontsize=10,
    )
    stem = f"branch_real_4dir_idx{case['index']}"
    save_figure(fig, outdir, stem, dpi)
    plt.close(fig)
    return {
        "stem": stem,
        "case": case,
        "region": loc,
        "mean_pairwise_cosine": similarity,
        "caption": (
            f"Branch-level Grad-CAM for real_4dir on index {case['index']}. "
            f"The marked fused high-response region is in the {loc}. "
            f"Mean pairwise cosine similarity between branch attribution maps is {similarity:.3f}. "
            "Use this panel to judge whether row/col/diag/anti_diag branches are visually complementary "
            "or mostly redundant."
        ),
        "render_data": {
            "image": image_np,
            "display_maps": display_maps,
            "bbox": bbox,
            "branch_names": branch_names,
        },
    }


def mean_pairwise_cosine(maps):
    sims = []
    for i in range(len(maps)):
        a = maps[i].ravel()
        for j in range(i + 1, len(maps)):
            b = maps[j].ravel()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            sims.append(float(np.dot(a, b) / max(denom, 1e-12)))
    return float(np.mean(sims)) if sims else 1.0


def plot_case_grid(outdir, rendered, dpi):
    if not rendered:
        return
    fig, axes = plt.subplots(
        len(rendered), 5, figsize=(13.5, 2.55 * len(rendered)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    column_titles = [
        "Input",
        "row",
        "same_row_4",
        "real_4dir",
        "real_4dir - same_row_4",
    ]
    for row_idx, item in enumerate(rendered):
        case = item["case"]
        data = item["render_data"]
        image_np = data["image"]
        bbox = data["bbox"]
        for col_idx, ax in enumerate(axes[row_idx]):
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=9)
        axes[row_idx, 0].imshow(image_np)
        add_box(axes[row_idx, 0], bbox)
        axes[row_idx, 0].set_ylabel(
            f"{case['case_category']}\nidx {case['index']} | {case['label_name']}", fontsize=7
        )
        for col_idx, variant in enumerate(("row", "same_row_4", "real_4dir"), start=1):
            axes[row_idx, col_idx].imshow(image_np, alpha=0.55)
            axes[row_idx, col_idx].imshow(
                data["display_maps"][variant],
                cmap="magma",
                alpha=0.72,
                extent=(0, 32, 32, 0),
                vmin=0,
                vmax=1,
                interpolation="bilinear",
            )
            add_box(axes[row_idx, col_idx], bbox)
        axes[row_idx, 4].imshow(
            data["difference"],
            cmap="RdBu_r",
            extent=(0, 32, 32, 0),
            vmin=-1,
            vmax=1,
            interpolation="bilinear",
        )
        add_box(axes[row_idx, 4], bbox)
    fig.suptitle(
        "Behavior-selected CIFAR-10 cases | red difference: more real_4dir attribution mass",
        fontsize=11,
    )
    save_figure(fig, outdir, "qualitative_grid", dpi)
    plt.close(fig)


def plot_branch_grid(outdir, rendered, dpi):
    if not rendered:
        return
    fig, axes = plt.subplots(
        len(rendered), 6, figsize=(16.0, 2.6 * len(rendered)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    titles = ["Input", "row", "col", "diag", "anti_diag", "fused"]
    for row_idx, item in enumerate(rendered):
        case = item["case"]
        data = item["render_data"]
        for col_idx, ax in enumerate(axes[row_idx]):
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=9)
        axes[row_idx, 0].imshow(data["image"])
        add_box(axes[row_idx, 0], data["bbox"])
        axes[row_idx, 0].set_ylabel(
            f"idx {case['index']} | {case['label_name']}\ncos={item['mean_pairwise_cosine']:.2f}",
            fontsize=8,
        )
        for col_idx, heat in enumerate(data["display_maps"], start=1):
            axes[row_idx, col_idx].imshow(data["image"], alpha=0.55)
            axes[row_idx, col_idx].imshow(
                heat,
                cmap="magma",
                alpha=0.72,
                extent=(0, 32, 32, 0),
                vmin=0,
                vmax=1,
                interpolation="bilinear",
            )
            add_box(axes[row_idx, col_idx], data["bbox"])
    fig.suptitle("real_4dir branch attribution maps", fontsize=11)
    save_figure(fig, outdir, "branch_grid", dpi)
    plt.close(fig)


def save_figure(fig, outdir, stem, dpi):
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        if ext == "png":
            fig.savefig(path, dpi=dpi, facecolor="white")
        else:
            fig.savefig(path, facecolor="white")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    figure_dir = os.path.join(args.outdir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, ckpts = ensure_models(args, device)
    loader = build_test_loader(args.data_root, args.batch_size, args.num_workers)

    rows = collect_predictions(models, loader, device)
    prediction_csv = os.path.join(args.outdir, "per_sample_predictions.csv")
    write_prediction_csv(prediction_csv, rows)
    selected = select_cases(rows, args.max_per_category)
    with open(os.path.join(args.outdir, "selected_cases.json"), "w") as f:
        json.dump(selected, f, indent=2)

    dataset = loader.dataset.dataset
    case_figures = []
    for case in selected:
        image_tensor, _label = dataset[case["index"]]
        image = image_tensor.unsqueeze(0).to(device)
        image_np = unnormalize(image_tensor.to(device))
        heats = {}
        logits_by_variant = {}
        for variant, model in models.items():
            heat, logits = gradcam_heatmap(model, image, int(case["label"]))
            heats[variant] = heat
            logits_by_variant[variant] = logits.softmax(dim=1).detach().cpu().numpy()[0].tolist()
        case_figures.append(
            plot_heatmap_case(figure_dir, case, image_np, heats, logits_by_variant, args.dpi)
        )
    plot_case_grid(figure_dir, case_figures, args.dpi)

    real_model = models.get("real_4dir")
    branch_figures = []
    if real_model is not None and selected:
        branch_names = ckpts["real_4dir"]["branch_dirs"].split(",")
        for case in selected[: min(3, len(selected))]:
            image_tensor, _label = dataset[case["index"]]
            image = image_tensor.unsqueeze(0).to(device)
            image_np = unnormalize(image_tensor.to(device))
            branch_heats, _logits = branch_gradcam(real_model, image, int(case["label"]))
            branch_figures.append(
                plot_branch_case(figure_dir, case, image_np, branch_names, branch_heats, args.dpi)
            )
    plot_branch_grid(figure_dir, branch_figures, args.dpi)

    figures_for_manifest = []
    for item in case_figures + branch_figures:
        figures_for_manifest.append(
            {key: value for key, value in item.items() if key not in ("render_data", "case")}
        )

    manifest = {
        "args": vars(args),
        "device": str(device),
        "checkpoint_meta": {
            variant: {
                "path": checkpoint_path(args, variant),
                "best_acc": ckpts[variant].get("best_acc"),
                "best_epoch": ckpts[variant].get("best_epoch"),
                "branch_dirs": ckpts[variant].get("branch_dirs"),
            }
            for variant in args.variants
        },
        "prediction_csv": prediction_csv,
        "selected_cases": selected,
        "figures": figures_for_manifest,
        "composite_figures": {
            "qualitative_grid": [
                os.path.join(figure_dir, f"qualitative_grid.{ext}")
                for ext in ("png", "svg", "pdf")
            ],
            "branch_grid": [
                os.path.join(figure_dir, f"branch_grid.{ext}")
                for ext in ("png", "svg", "pdf")
            ],
        },
        "note": (
            "Heatmaps are Grad-CAM style maps on final feat2d using the true class logit. "
            "Positive attribution is normalized to a spatial mass distribution per model before "
            "computing real_4dir - same_row_4. Marked boxes follow the connected component around "
            "the strongest positive difference cell."
        ),
    }
    with open(os.path.join(args.outdir, "visual_analysis_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    write_markdown_report(args.outdir, manifest)
    print(f"saved predictions: {prediction_csv}")
    print(f"saved figures: {figure_dir}")
    print(f"saved manifest: {os.path.join(args.outdir, 'visual_analysis_manifest.json')}")


def write_markdown_report(outdir, manifest):
    path = os.path.join(outdir, "VISUAL_ANALYSIS_REPORT.md")
    with open(path, "w") as f:
        f.write("# Stage1C CIFAR-10 Visual Analysis\n\n")
        f.write(
            "This analysis uses Grad-CAM style heatmaps on the final `feat2d` map. "
            "All heatmaps target the true CIFAR-10 class logit. Positive attribution is "
            "normalized to unit spatial mass per model before computing difference maps. "
            "The key comparison is "
            "`real_4dir - same_row_4`, because `same_row_4` controls for four full branches "
            "without adding new scan directions.\n\n"
        )
        f.write("## Checkpoints\n\n")
        for variant, meta in manifest["checkpoint_meta"].items():
            f.write(
                f"- `{variant}`: best_acc={meta['best_acc']}, best_epoch={meta['best_epoch']}, "
                f"branch_dirs=`{meta['branch_dirs']}`\n"
            )
        f.write("\n## Selected Cases\n\n")
        for fig in manifest["figures"]:
            f.write(f"### {fig['stem']}\n\n")
            f.write(f"{fig['caption']}\n\n")
        f.write("## Interpretation Rule\n\n")
        f.write(
            "If `same_row_4` already shifts attention from background to object regions, the gain "
            "should be attributed cautiously to multi-branch capacity or ensemble-like effects. "
            "Only cases where `real_4dir` repeatedly highlights spatial structures missed by "
            "`same_row_4` support a cautious direction-specific contribution.\n"
        )


if __name__ == "__main__":
    main()
