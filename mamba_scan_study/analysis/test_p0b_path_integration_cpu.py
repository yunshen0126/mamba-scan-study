"""CPU-only B4B resolver and explicit ChannelSplitBackbone integration checks."""

from __future__ import annotations

from dataclasses import replace
import copy
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from mamba_scan_study.experiments import p0b_path_bank
from mamba_scan_study.experiments.p0b_path_bank import (
    EXPECTED_SOURCE_SHA256,
    P0B_EXP_IDS,
    P0B_GRIDS,
    P0B_TRAINING_SEEDS,
    P0BSourcePaths,
    default_source_paths,
    iter_p0b_design_cells,
    resolve_p0b_paths,
    verify_source_hashes,
)
from mamba_scan_study.models.backbone import ChannelSplitBackbone


class Capture(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, tokens):
        self.seen = tokens.detach().clone()
        return tokens


def _expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _make_model(channel_orders, *, pos_mode="none", n_layers=0):
    model = ChannelSplitBackbone(
        img_size=8,
        patch_size=1,
        in_chans=4,
        d_model=4,
        n_layers=n_layers,
        block_type="gru",
        n_classes=3,
        variant="channel_same_row_4",
        pos_mode=pos_mode,
        dropout=0.0,
        shuffle_seed=17,
        channel_orders=channel_orders,
    )
    with torch.no_grad():
        model.patch_embed.weight.zero_()
        model.patch_embed.bias.zero_()
        for group in range(4):
            model.patch_embed.weight[group, group, 0, 0] = 1.0
    model.norm = nn.Identity()
    return model


def _cell_id_image():
    cells = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    return cells.repeat(1, 4, 1, 1)


def _reference_legacy_permutations(variant, length, seed):
    generator = torch.Generator().manual_seed(seed)
    permutations = []
    shared = None
    for _ in range(4):
        if variant == "channel_same_perm_4":
            if shared is None:
                shared = torch.randperm(length, generator=generator)
            permutation = shared.clone()
        elif variant == "channel_rand_perm_4":
            permutation = torch.randperm(length, generator=generator)
        else:
            permutation = torch.arange(length)
        permutations.append(permutation)
    return torch.stack(permutations)


def _capture_before_block(channel_orders):
    model = _make_model(channel_orders, n_layers=1)
    captures = [Capture() for _ in range(4)]
    model.group_blocks = nn.ModuleList(
        [nn.ModuleList([capture]) for capture in captures]
    )
    model.forward_features(_cell_id_image())
    return [capture.seen[0, :, 0].to(torch.long) for capture in captures]


def test_resolver_104_design_cells_and_guards():
    paths = default_source_paths()
    assert verify_source_hashes(paths) == EXPECTED_SOURCE_SHA256

    cells = list(iter_p0b_design_cells())
    assert len(cells) == 104
    assert len(set(cells)) == 104
    assert set(exp_id for exp_id, _, _ in cells) == set(P0B_EXP_IDS)
    assert set(grid for _, grid, _ in cells) == set(P0B_GRIDS)
    assert set(seed for _, _, seed in cells) == set(P0B_TRAINING_SEEDS)

    with mock.patch.object(torch, "randperm", side_effect=AssertionError("forbidden")):
        resolutions = [resolve_p0b_paths(*cell) for cell in cells]
    for resolution in resolutions:
        assert len(resolution.channel_orders) == 4
        assert len(resolution.channel_path_ids) == 4
        assert len(resolution.channel_order_sha256) == 4
        assert len(resolution.channel_inverse_order_sha256) == 4
        for order in resolution.channel_orders:
            assert order.device.type == "cpu"
            assert order.dtype == torch.long
            assert order.shape == (resolution.grid * resolution.grid,)

    assert resolve_p0b_paths("GEO_SG3", 8, 2).channel_path_ids == ("G3",) * 4
    assert resolve_p0b_paths("GEO_DIV", 8, 1).channel_path_ids == ("G2", "G3", "G4", "G1")
    assert resolve_p0b_paths("RND_S2", 8, 3).channel_path_ids == ("R2_4",) * 4
    assert resolve_p0b_paths("RND_D3", 8, 3).channel_path_ids == (
        "R3_4", "R3_1", "R3_2", "R3_3"
    )
    assert resolve_p0b_paths("LOC_S", 8, 2).channel_path_ids == ("L3",) * 4
    assert resolve_p0b_paths("LOC_D", 8, 2).channel_path_ids == ("L3", "L4", "L1", "L2")

    for grid in P0B_GRIDS:
        random_payload = __import__("json").loads(paths.random.read_text(encoding="utf-8"))
        random_paths = p0b_path_bank._random_paths(random_payload, grid)
        assert len(random_paths) == 12
        for set_number in (1, 2, 3):
            for path_number in (1, 2, 3, 4):
                path_id = f"R{set_number}_{path_number}"
                path = random_paths[path_id]
                assert path.path_id == path_id
                assert path.order.shape == (grid * grid,)
                assert path.order_sha256
                assert path.inverse_order_sha256

    with tempfile.TemporaryDirectory() as directory:
        for source_name in ("lmto", "random", "validation_split", "config"):
            source_path = getattr(paths, source_name)
            tampered_path = Path(directory) / source_path.name
            tampered_path.write_bytes(source_path.read_bytes() + b"\n")
            _expect_value_error(
                lambda source_name=source_name, tampered_path=tampered_path: resolve_p0b_paths(
                    "LOC_D", 8, 0, replace(paths, **{source_name: tampered_path})
                )
            )

    valid = np.arange(64, dtype=np.int64)
    digest = p0b_path_bank._int64_c_sha256(valid)
    inverse_digest = p0b_path_bank._int64_c_sha256(valid)
    _expect_value_error(
        lambda: p0b_path_bank._validate_order(
            path_id="bad-length", order_values=valid[:-1], grid=8,
            declared_order_sha256=digest, declared_inverse_sha256=inverse_digest,
        )
    )
    duplicate = valid.copy()
    duplicate[-1] = duplicate[-2]
    _expect_value_error(
        lambda: p0b_path_bank._validate_order(
            path_id="duplicate", order_values=duplicate, grid=8,
            declared_order_sha256=digest, declared_inverse_sha256=inverse_digest,
        )
    )
    out_of_range = valid.copy()
    out_of_range[-1] = 64
    _expect_value_error(
        lambda: p0b_path_bank._validate_order(
            path_id="out-of-range", order_values=out_of_range, grid=8,
            declared_order_sha256=digest, declared_inverse_sha256=inverse_digest,
        )
    )
    _expect_value_error(
        lambda: p0b_path_bank._validate_order(
            path_id="bad-hash", order_values=valid, grid=8,
            declared_order_sha256="0" * 64, declared_inverse_sha256=inverse_digest,
        )
    )
    _expect_value_error(
        lambda: p0b_path_bank._validate_order(
            path_id="bad-inverse-hash", order_values=valid, grid=8,
            declared_order_sha256=digest, declared_inverse_sha256="0" * 64,
        )
    )

    random_payload = __import__("json").loads(paths.random.read_text(encoding="utf-8"))
    record = random_payload["grids"][0]["sets"]["S1"]["paths"]["R1_1"]
    record["set_id"] = "S2"
    _expect_value_error(lambda: p0b_path_bank._random_paths(random_payload, 8))
    random_payload = __import__("json").loads(paths.random.read_text(encoding="utf-8"))
    random_payload["grids"][0]["sets"]["S1"]["paths"]["R1_1"]["seed"] += 1
    _expect_value_error(lambda: p0b_path_bank._random_paths(random_payload, 8))
    assert "torch.randperm(" not in Path(p0b_path_bank.__file__).read_text(encoding="utf-8")


def test_explicit_absolute_order_semantics():
    cases = {
        "G1": resolve_p0b_paths("GEO_SG1", 8, 0),
        "G2": resolve_p0b_paths("GEO_SG2", 8, 0),
        "G3": resolve_p0b_paths("GEO_SG3", 8, 0),
        "G4": resolve_p0b_paths("GEO_SG4", 8, 0),
        "R": resolve_p0b_paths("RND_S1", 8, 0),
        "LMTO": resolve_p0b_paths("LOC_S", 8, 0),
    }
    for label, resolution in cases.items():
        captured = _capture_before_block(resolution.channel_orders)
        for group in range(4):
            assert torch.equal(captured[group], resolution.channel_orders[group]), label

    g3 = cases["G3"].channel_orders[0]
    expected_column_order = torch.tensor(
        [row * 8 + column for column in range(8) for row in range(8)], dtype=torch.long
    )
    assert torch.equal(g3, expected_column_order)
    assert not torch.equal(g3, torch.arange(64))

    explicit = _make_model(cases["G1"].channel_orders)
    assert explicit.variant == "channel_same_row_4"
    assert explicit.branch_dirs == ("row", "row", "row", "row")
    assert explicit.explicit_channel_orders is True
    for variant in ("channel_real_4dir", "channel_same_perm_4", "channel_rand_perm_4"):
        _expect_value_error(
            lambda variant=variant: ChannelSplitBackbone(
                img_size=8, patch_size=1, in_chans=4, d_model=4, n_layers=0,
                block_type="gru", variant=variant, channel_orders=cases["G1"].channel_orders,
            )
        )


def test_position_mask_inverse_and_identity_operator_graph():
    g3 = resolve_p0b_paths("GEO_SG3", 8, 0).channel_orders
    model = _make_model(g3, pos_mode="seq_learned", n_layers=1)
    captures = [Capture() for _ in range(4)]
    model.group_blocks = nn.ModuleList([nn.ModuleList([capture]) for capture in captures])
    with torch.no_grad():
        for group in range(4):
            model.seq_pos[group].copy_(torch.arange(64, dtype=torch.float32).view(1, 64, 1) * 100)
            model.mask_tokens[group].fill_(-10000)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, 5] = True
    features = model.forward_features(_cell_id_image(), token_mask=mask)
    expected_row_major = torch.arange(64, dtype=torch.float32) * 101
    expected_row_major[5] = -9500
    for group, order in enumerate(g3):
        expected_sequence = expected_row_major.index_select(0, order)
        assert torch.equal(captures[group].seen[0, :, 0], expected_sequence)
        assert torch.equal(features[0, :, :, group].reshape(-1), expected_row_major)

    identity_orders = tuple(torch.arange(64, dtype=torch.long) for _ in range(4))
    explicit_identity = _make_model(identity_orders, pos_mode="seq_learned", n_layers=0)
    legacy_identity = ChannelSplitBackbone(
        img_size=8, patch_size=1, in_chans=4, d_model=4, n_layers=0,
        block_type="gru", variant="channel_same_row_4", pos_mode="seq_learned",
    )
    original_index_select = torch.Tensor.index_select
    explicit_calls = []
    legacy_calls = []

    def explicit_spy(tensor, dim, index):
        explicit_calls.append((tensor, dim, index))
        return original_index_select(tensor, dim, index)

    with mock.patch.object(torch.Tensor, "index_select", new=explicit_spy):
        explicit_identity.forward_features(_cell_id_image(), token_mask=mask)
    # Per group: token flatten, mask flatten, then token/position/mask permutation,
    # followed by inverse. Identity explicit paths therefore still perform 6 calls.
    assert len(explicit_calls) == 24
    permutation_ptrs = {explicit_identity.channel_permutations[group].data_ptr() for group in range(4)}
    inverse_ptrs = {explicit_identity.channel_inverse_permutations[group].data_ptr() for group in range(4)}
    selected_ptrs = [index.data_ptr() for _, _, index in explicit_calls]
    assert sum(pointer in permutation_ptrs for pointer in selected_ptrs) == 12
    assert sum(pointer in inverse_ptrs for pointer in selected_ptrs) == 4

    def legacy_spy(tensor, dim, index):
        legacy_calls.append((tensor, dim, index))
        return original_index_select(tensor, dim, index)

    with mock.patch.object(torch.Tensor, "index_select", new=legacy_spy):
        legacy_identity.forward_features(_cell_id_image(), token_mask=mask)
    # Legacy identity retains only row flattening for tokens and masks.
    assert len(legacy_calls) == 8


def test_legacy_permutation_regression_and_state_dict():
    kwargs = dict(
        img_size=8,
        patch_size=1,
        in_chans=3,
        d_model=8,
        n_layers=1,
        block_type="gru",
        n_classes=3,
        pos_mode="xy_learned",
        dropout=0.0,
        shuffle_seed=77,
    )
    for variant in ChannelSplitBackbone.VARIANTS:
        torch.manual_seed(1234)
        model = ChannelSplitBackbone(variant=variant, **kwargs).eval()
        expected = _reference_legacy_permutations(variant, model.L, 77)
        assert torch.equal(model.channel_permutations, expected)
        expected_inverse = torch.empty_like(expected)
        expected_inverse.scatter_(1, expected, torch.arange(model.L).expand(4, -1))
        assert torch.equal(model.channel_inverse_permutations, expected_inverse)
        assert model.channel_permutations.dtype == torch.long
        assert model.channel_inverse_permutations.dtype == torch.long
        assert model.channel_permutations.shape == (4, model.L)
        assert model.channel_inverse_permutations.shape == (4, model.L)
        state = model.state_dict()
        assert "channel_permutations" in state
        assert "channel_inverse_permutations" in state
        assert "explicit_channel_orders" not in state
        clone = ChannelSplitBackbone(variant=variant, **kwargs).eval()
        clone.load_state_dict(copy.deepcopy(state), strict=True)
        input_tensor = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            original_output = model(input_tensor)
            loaded_output = clone(input_tensor)
        assert torch.equal(original_output[0], loaded_output[0])
        assert torch.equal(original_output[1], loaded_output[1])
        if variant == "channel_same_row_4":
            explicit = ChannelSplitBackbone(
                variant=variant,
                channel_orders=tuple(torch.arange(model.L) for _ in range(4)),
                **kwargs,
            ).eval()
            explicit.load_state_dict(copy.deepcopy(state), strict=True)
            with torch.no_grad():
                explicit_output = explicit(input_tensor)
            assert torch.equal(original_output[0], explicit_output[0])
            assert torch.equal(original_output[1], explicit_output[1])


def test_six_explicit_structure_classes():
    representatives = (
        "GEO_SG1",
        "GEO_DIV",
        "RND_S1",
        "RND_D1",
        "LOC_S",
        "LOC_D",
    )
    models = []
    for exp_id in representatives:
        torch.manual_seed(20260720)
        resolution = resolve_p0b_paths(exp_id, 8, 1)
        model = ChannelSplitBackbone(
            img_size=8,
            patch_size=1,
            in_chans=3,
            d_model=8,
            n_layers=0,
            block_type="gru",
            n_classes=3,
            variant="channel_same_row_4",
            pos_mode="xy_learned",
            channel_orders=resolution.channel_orders,
        )
        models.append(model)
    parameter_specs = [
        [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
        for model in models
    ]
    buffer_specs = [
        [(name, tuple(buffer.shape)) for name, buffer in model.named_buffers()]
        for model in models
    ]
    assert all(spec == parameter_specs[0] for spec in parameter_specs[1:])
    assert all(spec == buffer_specs[0] for spec in buffer_specs[1:])
    assert len({sum(parameter.numel() for parameter in model.parameters()) for model in models}) == 1
    original_index_select = torch.Tensor.index_select
    for model in models:
        assert model.branch_dirs == ("row", "row", "row", "row")
        assert model.explicit_channel_orders is True
        assert model.channel_permutations.shape == model.channel_inverse_permutations.shape == (4, 64)
        logits, features = model(torch.randn(2, 3, 8, 8))
        assert logits.shape == (2, 3)
        assert features.shape == (2, 8, 8, 8)
        calls = []

        def spy(tensor, dim, index):
            calls.append(index.data_ptr())
            return original_index_select(tensor, dim, index)

        with mock.patch.object(torch.Tensor, "index_select", new=spy):
            model.forward_features(torch.randn(1, 3, 8, 8))
        # Four groups all use the same explicit operator graph: input/position
        # flattening, token/position permutation, then inverse (5 calls each).
        assert len(calls) == 20
        permutation_ptrs = {model.channel_permutations[group].data_ptr() for group in range(4)}
        inverse_ptrs = {model.channel_inverse_permutations[group].data_ptr() for group in range(4)}
        assert sum(pointer in permutation_ptrs for pointer in calls) == 8
        assert sum(pointer in inverse_ptrs for pointer in calls) == 4


def main():
    torch.set_num_threads(1)
    test_resolver_104_design_cells_and_guards()
    test_explicit_absolute_order_semantics()
    test_position_mask_inverse_and_identity_operator_graph()
    test_legacy_permutation_regression_and_state_dict()
    test_six_explicit_structure_classes()
    print("PASS: B4B resolver source gates, 104 design cells, and all frozen R records")
    print("PASS: explicit absolute-order semantics for G1/G2/G3/G4/R/LMTO")
    print("PASS: position/mask/inverse restoration and explicit identity operator graph")
    print("PASS: legacy permutations, persistent buffers, strict state_dict, and output regression")
    print("PASS: six explicit structure classes have matching trainable/buffer structure")


if __name__ == "__main__":
    main()
