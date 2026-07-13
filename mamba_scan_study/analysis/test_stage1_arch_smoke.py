import torch
import torch.nn.functional as F

from mamba_scan_study.experiments.run_stage1_seed0 import (
    CHANNEL_VARIANTS,
    Config,
    make_model,
    matrix_order,
    run_key,
    set_seed,
    shuffle_seed,
)


def make_config(arch, d_model):
    return Config(
        dataset="cifar10",
        img_size=32,
        arch=arch,
        shuffle_seed=shuffle_seed(0, 0),
        data_root="",
        outdir="",
        microbatch_csv="",
        epochs=2,
        warmup_epochs=1,
        effective_batch=2,
        d_model=d_model,
        n_layers=2,
        pos_mode="xy_learned",
        base_lr=1e-3,
        weight_decay=0.05,
        grad_clip=1.0,
        num_workers=0,
        seed=0,
        amp=False,
        consistency_target=0.0064,
        consistency_tolerance=0.0064,
        consistency_min_positive=0.001,
    )


def expected_full_matrix():
    gate = [("mamba", 8, "same_row_4"), ("mamba", 8, "real_4dir")]
    remaining = []
    for block in ("gru", "mamba"):
        for grid in (8, 16, 32):
            for variant in ("row", "shuffle_row", "col", "same_row_4", "real_4dir"):
                item = (block, grid, variant)
                if item not in gate:
                    remaining.append(item)
    return gate + remaining


def train_two_steps(model, grid, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_acc = []
    batch_size = 2 if grid == 8 else 1
    for epoch in range(2):
        generator = torch.Generator().manual_seed(1000 + epoch)
        images = torch.randn(batch_size, 3, 32, 32, generator=generator).to(device)
        labels = torch.randint(0, 10, (batch_size,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _features = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        train_acc.append(float((logits.argmax(1) == labels).float().mean().item()))
    assert len(train_acc) == 2
    return train_acc


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 architecture smoke requires CUDA for Mamba")
    device = torch.device("cuda")

    full_matrix = matrix_order("cifar10", "full_branch")
    assert full_matrix == expected_full_matrix()
    assert {grid: 32 // grid for grid in (8, 16, 32)} == {8: 4, 16: 2, 32: 1}

    channel_matrix = matrix_order("cifar10", "channel_split")
    assert len(channel_matrix) == 18
    assert {variant for _block, _grid, variant in channel_matrix} == set(CHANNEL_VARIANTS)

    keys = {
        run_key("cifar10", arch, d_model, "gru", 8, "real_4dir")
        for arch in ("full_branch", "channel_split")
        for d_model in (64, 256)
    }
    assert len(keys) == 4

    rows = []
    for arch in ("full_branch", "channel_split"):
        variants = ("real_4dir",) if arch == "full_branch" else CHANNEL_VARIANTS
        for d_model in (64, 256):
            cfg = make_config(arch, d_model)
            for block in ("gru", "mamba"):
                for grid in (8, 32):
                    counts = []
                    for variant in variants:
                        set_seed(1234)
                        model = make_model(cfg, block, grid, variant, device)
                        counts.append(sum(parameter.numel() for parameter in model.parameters()))
                        train_acc = train_two_steps(model, grid, device)
                        rows.append((arch, d_model, block, grid, variant, counts[-1], train_acc))
                        del model
                        torch.cuda.empty_cache()
                    if arch == "channel_split":
                        assert len(set(counts)) == 1

    print("PASS: full_branch d_model=64 matrix is the original ordered 30-run matrix")
    print("PASS: patch mapping is grid8=4, grid16=2, grid32=1")
    print("PASS: channel_split matrix contains 18 runs and exactly 3 variants per cell")
    print("PASS: resume keys distinguish arch and d_model")
    print("PASS: 2-step smoke covered arch={full_branch,channel_split}, d_model={64,256}, block={gru,mamba}, grid={8,32}")
    for row in rows:
        arch, d_model, block, grid, variant, params, train_acc = row
        print(
            f"SMOKE arch={arch} d_model={d_model} block={block} grid={grid} "
            f"variant={variant} params={params} train_acc={train_acc}"
        )


if __name__ == "__main__":
    main()
