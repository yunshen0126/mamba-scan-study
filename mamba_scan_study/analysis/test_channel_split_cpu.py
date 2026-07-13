import torch

from mamba_scan_study.models.backbone import ChannelSplitBackbone, MultiDirBackbone
from mamba_scan_study.models.scan_utils import flatten_scan, restore_scan


def main():
    device = torch.device("cpu")
    torch.set_num_threads(1)
    kwargs = dict(
        img_size=32,
        patch_size=4,
        in_chans=3,
        d_model=64,
        n_layers=2,
        block_type="gru",
        n_classes=10,
        pos_mode="xy_learned",
    )
    variants = ("channel_real_4dir", "channel_same_row_4", "channel_rand_perm_4")
    models = []
    for name in variants:
        torch.manual_seed(1234)
        models.append(ChannelSplitBackbone(variant=name, shuffle_seed=77, **kwargs).to(device))

    param_specs = [
        [(name, tuple(param.shape), param.numel()) for name, param in model.named_parameters()]
        for model in models
    ]
    assert param_specs[0] == param_specs[1] == param_specs[2]

    torch.manual_seed(1234)
    full = MultiDirBackbone(branch_dirs="row,col,diag,anti_diag", **kwargs).to(device)
    channel_params = sum(param.numel() for param in models[0].parameters())
    full_params = sum(param.numel() for param in full.parameters())
    assert channel_params < full_params

    image = torch.arange(2 * 4 * 4 * 3, dtype=torch.float32).reshape(2, 4, 4, 3)
    for direction in ("row", "col", "diag", "anti_diag"):
        restored = restore_scan(flatten_scan(image, direction), 4, 4, direction)
        assert torch.equal(restored, image)

    for name, first in models[0].named_parameters():
        for other in models[1:]:
            assert torch.equal(first, dict(other.named_parameters())[name]), name

    permutations = models[2].channel_permutations
    assert any(
        not torch.equal(permutation, torch.arange(models[2].L))
        for permutation in permutations
    )
    torch.manual_seed(999)
    repeat = ChannelSplitBackbone(
        variant="channel_rand_perm_4", shuffle_seed=77, **kwargs
    ).channel_permutations
    assert torch.equal(permutations, repeat)

    x = torch.randn(2, 3, 32, 32, device=device)
    for model in models:
        logits, features = model(x)
        assert logits.shape == (2, 10)
        assert features.shape == (2, 8, 8, 64)

    checkpoint = torch.load(
        "mamba_scan_study/outputs/stage1_seed0/checkpoints/gru_real_4dir_grid8_seed0.pt",
        map_location=device,
    )
    old_a = MultiDirBackbone(
        img_size=32,
        patch_size=4,
        in_chans=3,
        d_model=64,
        n_layers=2,
        block_type="gru",
        n_classes=10,
        branch_dirs="row,col,diag,anti_diag",
        pos_mode="xy_learned",
    ).to(device)
    old_b = MultiDirBackbone(
        img_size=32,
        patch_size=4,
        in_chans=3,
        d_model=64,
        n_layers=2,
        block_type="gru",
        n_classes=10,
        branch_dirs="row,col,diag,anti_diag",
        pos_mode="xy_learned",
    ).to(device)
    old_a.load_state_dict(checkpoint["model_state"])
    old_b.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        output_a = old_a(x)[0]
        output_b = old_b(x)[0]
    assert torch.equal(output_a, output_b)

    print("PASS: parameter tensor shapes/counts identical across 3 variants")
    print(
        f"PASS: params channel_split={channel_params:,}; "
        f"full_branch_real4={full_params:,}; ratio={channel_params / full_params:.4f}"
    )
    print("PASS: flatten/restore reversible and spatially aligned for 4 scan directions")
    print("PASS: same-seed parameter initialization identical across variants")
    print("PASS: random permutations non-identity and reproducible from local seed")
    print("PASS: all 3 channel variants CPU forward shapes valid")
    print("PASS: existing MultiDirBackbone checkpoint loads and CPU outputs are deterministic")


if __name__ == "__main__":
    main()
