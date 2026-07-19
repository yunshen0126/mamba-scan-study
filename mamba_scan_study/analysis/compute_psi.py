"""Compute task-agnostic permutation sensitivity indices for trained SSM models.

PSI compares a target SSM module's feature tensor under canonical token order and
under a temporary token permutation that is inverted immediately after the module.
It deliberately calls ``forward_features`` rather than ``forward``: no classifier
head, logits, loss, or labels participate in the calculation.
"""

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler

from mamba_scan_study.data.real_datasets import build_real_loaders
from mamba_scan_study.models.backbone import ChannelSplitBackbone, MultiDirBackbone


CSV_FIELDS = (
    "checkpoint",
    "dataset",
    "block",
    "grid",
    "seq_len",
    "variant",
    "seed",
    "layer_idx",
    "is_canonical",
    "psi_mean",
    "psi_std",
    "n_perms",
)


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint: str
    dataset: str
    arch: str
    block: str
    grid: int
    variant: str
    seed: int
    img_size: int
    d_model: int
    n_layers: int
    pos_mode: str
    n_classes: int
    branch_dirs: str
    shuffle_order: bool
    shuffle_seed: int


@dataclass(frozen=True)
class LayerTarget:
    layer_idx: str
    module: nn.Module
    seq_len: int


@dataclass(frozen=True)
class StackTarget:
    layer_idx: str
    entry_module: nn.Module
    exit_module: nn.Module
    seq_len: int


class PermutationBank:
    """Generate paired permutations shared by every checkpoint at one grid."""

    def __init__(self, base_seed: int, n_perms: int):
        self.base_seed = int(base_seed)
        self.n_perms = int(n_perms)
        self._cpu_cache: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    def for_grid(self, grid: int, seq_len: int, device: torch.device) -> List[torch.Tensor]:
        key = (int(grid), int(seq_len))
        if key not in self._cpu_cache:
            permutations = []
            for perm_index in range(self.n_perms):
                # The grid term keeps different sequence resolutions independent.
                generator = torch.Generator().manual_seed(
                    self.base_seed + key[0] * 1_000_003 + perm_index
                )
                permutations.append(torch.randperm(key[1], generator=generator))
            self._cpu_cache[key] = permutations
        return [permutation.to(device=device) for permutation in self._cpu_cache[key]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SSM permutation sensitivity indices.")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        help="One or more Stage 1 checkpoint paths used as trained-model metadata.",
    )
    parser.add_argument("--data-root", help="Existing dataset root. Downloads are disabled.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the dataset stored in the checkpoint metadata.",
    )
    parser.add_argument("--output-csv", default="psi_layer_level.csv")
    parser.add_argument("--config-output-csv", default="psi_config_level.csv")
    parser.add_argument("--stack-output-csv", default="psi_stack_level.csv")
    parser.add_argument("--stack-config-output-csv", default="psi_stack_config_level.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-permutations", type=int, default=8)
    parser.add_argument("--permutation-seed", type=int, default=17_071)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        default=True,
        help="Restrict PSI load-axis values to canonical variants (default).",
    )
    parser.add_argument(
        "--allow-treated",
        action="store_true",
        help="Explicitly allow non-canonical variants. Their CSV rows set is_canonical=False.",
    )
    parser.add_argument(
        "--at-init",
        action="store_true",
        help="Use checkpoint architecture metadata but skip model_state loading.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="Deterministic initialization seed used only with --at-init.",
    )
    parser.add_argument(
        "--self-test-hooks",
        action="store_true",
        help="Run only the hook output-replacement unit test, then exit.",
    )
    parser.set_defaults(module_level=True, stack_level=True)
    parser.add_argument(
        "--module-level",
        dest="module_level",
        action="store_true",
        help="Include per-block PSI rows (default).",
    )
    parser.add_argument(
        "--no-module-level",
        dest="module_level",
        action="store_false",
        help="Skip per-block PSI and emit only stack-level PSI.",
    )
    parser.add_argument(
        "--stack-level",
        dest="stack_level",
        action="store_true",
        help="Include whole-stack PSI rows, with pi before block 0 and pi^-1 after the last block (default).",
    )
    parser.add_argument(
        "--no-stack-level",
        dest="stack_level",
        action="store_false",
        help="Skip whole-stack PSI and emit only per-block PSI.",
    )
    return parser.parse_args()


def load_payload(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def checkpoint_metadata(path: str, payload: Dict) -> CheckpointMetadata:
    config = payload["config"]
    run = payload["run"]
    model_state = payload["model_state"]
    try:
        n_classes = int(model_state["head.weight"].shape[0])
    except KeyError as error:
        raise ValueError(f"checkpoint has no classifier head metadata: {path}") from error

    arch = run.get("arch", config.get("arch", "full_branch"))
    grid = int(run["grid"])
    return CheckpointMetadata(
        checkpoint=str(path),
        dataset=run.get("dataset", config["dataset"]),
        arch=arch,
        block=run["block_type"],
        grid=grid,
        variant=run["variant"],
        seed=int(run["seed"]),
        img_size=int(config["img_size"]),
        d_model=int(run.get("d_model", config["d_model"])),
        n_layers=int(config["n_layers"]),
        pos_mode=config["pos_mode"],
        n_classes=n_classes,
        branch_dirs=run.get("branch_dirs", "row"),
        shuffle_order=bool(run.get("shuffle_order", False)),
        shuffle_seed=run.get("shuffle_seed"),
    )


def is_canonical_variant(metadata: CheckpointMetadata) -> bool:
    if metadata.arch == "full_branch":
        return metadata.variant == "row"
    if metadata.arch == "channel_split":
        return metadata.variant == "channel_same_row_4"
    return False


def validate_variant_access(metadata: CheckpointMetadata, args: argparse.Namespace) -> bool:
    canonical = is_canonical_variant(metadata)
    if args.canonical_only and not canonical and not args.allow_treated:
        raise ValueError(
            f"{metadata.checkpoint} has treated variant={metadata.variant!r}. "
            "PSI load-axis values require full-branch row or channel_same_row_4; "
            "pass --allow-treated to override explicitly."
        )
    return canonical


def build_model(metadata: CheckpointMetadata, payload: Dict, at_init: bool, init_seed: int) -> nn.Module:
    """Reconstruct the checkpoint architecture without calling its classifier forward path."""

    def construct() -> nn.Module:
        patch_size = metadata.img_size // metadata.grid
        # Constructors require n_classes, but forward_features never reaches the head.
        # The checkpoint-derived class count therefore has no effect on PSI or at-init PSI.
        if metadata.arch == "channel_split":
            shuffle_seed = metadata.shuffle_seed
            if shuffle_seed is None:
                shuffle_seed = int(payload["config"]["shuffle_seed"]) + metadata.grid
            return ChannelSplitBackbone(
                img_size=metadata.img_size,
                patch_size=patch_size,
                in_chans=3,
                d_model=metadata.d_model,
                n_layers=metadata.n_layers,
                block_type=metadata.block,
                n_classes=metadata.n_classes,
                variant=metadata.variant,
                shuffle_seed=int(shuffle_seed),
                pos_mode=metadata.pos_mode,
            )
        if metadata.arch == "full_branch":
            return MultiDirBackbone(
                img_size=metadata.img_size,
                patch_size=patch_size,
                in_chans=3,
                d_model=metadata.d_model,
                n_layers=metadata.n_layers,
                block_type=metadata.block,
                n_classes=metadata.n_classes,
                branch_dirs=metadata.branch_dirs,
                shuffle_order=metadata.shuffle_order,
                shuffle_seed=metadata.shuffle_seed,
                pos_mode=metadata.pos_mode,
            )
        raise ValueError(f"unsupported checkpoint arch={metadata.arch!r}")

    if at_init:
        # A fixed seed makes the random-initialization control reproducible.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            return construct()

    model = construct()
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def enumerate_ssm_layers(model: nn.Module) -> List[LayerTarget]:
    if isinstance(model, MultiDirBackbone):
        targets = []
        for branch_index, branch in enumerate(model.branches):
            for depth, block in enumerate(branch.blocks):
                targets.append(
                    LayerTarget(f"branch{branch_index}.layer{depth}", block, model.L)
                )
        return targets
    if isinstance(model, ChannelSplitBackbone):
        targets = []
        for group_index, blocks in enumerate(model.group_blocks):
            for depth, block in enumerate(blocks):
                targets.append(
                    LayerTarget(f"group{group_index}.layer{depth}", block, model.L)
                )
        return targets
    raise TypeError(f"unsupported model type={type(model).__name__}")


def enumerate_ssm_stacks(model: nn.Module) -> List[StackTarget]:
    """Return one target per sequential SSM stack, not one target per SSM block."""

    if isinstance(model, MultiDirBackbone):
        return [
            StackTarget(
                f"branch{branch_index}.stack",
                branch.blocks[0],
                branch.blocks[-1],
                model.L,
            )
            for branch_index, branch in enumerate(model.branches)
        ]
    if isinstance(model, ChannelSplitBackbone):
        return [
            StackTarget(
                f"group{group_index}.stack",
                blocks[0],
                blocks[-1],
                model.L,
            )
            for group_index, blocks in enumerate(model.group_blocks)
        ]
    raise TypeError(f"unsupported model type={type(model).__name__}")


def load_image_batch(dataset: str, data_root: str, img_size: int, args: argparse.Namespace) -> torch.Tensor:
    """Load one deterministic real-image test batch; labels are intentionally discarded."""

    _, test_loader = build_real_loaders(
        dataset,
        data_root,
        args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        download=False,
        generator=None,
    )
    # _make_loaders constructs the test loader with shuffle=False. Sequential sampling
    # makes this first batch identical across all checkpoints in one PSI invocation.
    if not isinstance(test_loader.sampler, SequentialSampler):
        raise AssertionError("PSI requires a deterministic, non-shuffled test loader")
    images, _ = next(iter(test_loader))
    # The second item is never inspected, stored, or used in PSI.
    return images


def capture_natural_features(
    model: nn.Module, targets: Sequence[LayerTarget], images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    for target in targets:
        def capture_hook(_module, _inputs, output, layer_idx=target.layer_idx):
            captured[layer_idx] = output.detach()

        handles.append(target.module.register_forward_hook(capture_hook))
    try:
        # forward_features bypasses both spatial mean pooling and the classifier head.
        model.forward_features(images)
    finally:
        for handle in handles:
            handle.remove()
    missing = [target.layer_idx for target in targets if target.layer_idx not in captured]
    if missing:
        raise RuntimeError(f"natural forward did not reach SSM modules: {missing}")
    return captured


def inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


def single_permutation_psi(
    model: nn.Module,
    target: LayerTarget,
    natural_feature: torch.Tensor,
    images: torch.Tensor,
    permutation: torch.Tensor,
    epsilon: float,
) -> float:
    """Measure one target layer with pi before and pi^-1 after that SSM module."""

    if permutation.numel() != target.seq_len:
        raise ValueError("permutation length does not match target SSM sequence length")
    inverse = inverse_permutation(permutation)
    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        return (inputs[0].index_select(1, permutation),)

    def post_hook(_module, _inputs, output):
        # PyTorch forward hooks replace the module output when they return a Tensor.
        # Returning the restored tensor is therefore required for downstream layers to
        # remain in canonical token order, matching shuffle_row's pi / pi^-1 contract.
        restored = output.index_select(1, inverse)
        captured["feature"] = restored.detach()
        return restored

    pre_handle = target.module.register_forward_pre_hook(pre_hook)
    post_handle = target.module.register_forward_hook(post_hook)
    try:
        model.forward_features(images)
    finally:
        pre_handle.remove()
        post_handle.remove()

    permuted_feature = captured.get("feature")
    if permuted_feature is None:
        raise RuntimeError(f"permuted forward did not reach {target.layer_idx}")
    numerator = (natural_feature - permuted_feature).flatten(start_dim=1).norm(dim=1)
    denominator = natural_feature.flatten(start_dim=1).norm(dim=1).clamp_min(epsilon)
    return (numerator / denominator).mean().item()


def capture_natural_stack_features(
    model: nn.Module, targets: Sequence[StackTarget], images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    for target in targets:
        def capture_hook(_module, _inputs, output, layer_idx=target.layer_idx):
            captured[layer_idx] = output.detach()

        handles.append(target.exit_module.register_forward_hook(capture_hook))
    try:
        model.forward_features(images)
    finally:
        for handle in handles:
            handle.remove()
    missing = [target.layer_idx for target in targets if target.layer_idx not in captured]
    if missing:
        raise RuntimeError(f"natural forward did not reach SSM stacks: {missing}")
    return captured


def single_stack_permutation_psi(
    model: nn.Module,
    target: StackTarget,
    natural_feature: torch.Tensor,
    images: torch.Tensor,
    permutation: torch.Tensor,
    epsilon: float,
) -> float:
    """Measure one whole SSM stack with pi before block 0 and pi^-1 after its last block."""

    if permutation.numel() != target.seq_len:
        raise ValueError("permutation length does not match target SSM sequence length")
    inverse = inverse_permutation(permutation)
    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        # The entry tensor already includes its position encoding. Moving the complete
        # token vector matches shuffle_row's joint content/position permutation.
        return (inputs[0].index_select(1, permutation),)

    def post_hook(_module, _inputs, output):
        # For a non-None forward-hook return value, PyTorch replaces the module output.
        # Returning pi^-1 therefore restores canonical order only after the full stack.
        restored = output.index_select(1, inverse)
        captured["feature"] = restored.detach()
        return restored

    entry_handle = target.entry_module.register_forward_pre_hook(pre_hook)
    exit_handle = target.exit_module.register_forward_hook(post_hook)
    try:
        model.forward_features(images)
    finally:
        entry_handle.remove()
        exit_handle.remove()

    permuted_feature = captured.get("feature")
    if permuted_feature is None:
        raise RuntimeError(f"permuted forward did not reach {target.layer_idx}")
    numerator = (natural_feature - permuted_feature).flatten(start_dim=1).norm(dim=1)
    denominator = natural_feature.flatten(start_dim=1).norm(dim=1).clamp_min(epsilon)
    return (numerator / denominator).mean().item()


def mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty PSI sample")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def compute_checkpoint_rows(
    checkpoint: str,
    payload: Dict,
    metadata: CheckpointMetadata,
    canonical: bool,
    images: torch.Tensor,
    bank: PermutationBank,
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict]:
    model = build_model(metadata, payload, args.at_init, args.init_seed).to(images.device)
    model.eval()
    targets = enumerate_ssm_layers(model)
    if not targets:
        raise RuntimeError(f"no SSM modules found in {checkpoint}")

    with torch.no_grad():
        natural = capture_natural_features(model, targets, images)
        permutations = bank.for_grid(metadata.grid, targets[0].seq_len, images.device)
        module_samples = {
            target.layer_idx: [
                single_permutation_psi(
                    model,
                    target,
                    natural[target.layer_idx],
                    images,
                    permutation,
                    args.epsilon,
                )
                for permutation in permutations
            ]
            for target in targets
        }

    base = {
        "checkpoint": str(checkpoint),
        "dataset": metadata.dataset,
        "block": metadata.block,
        "grid": metadata.grid,
        "seq_len": targets[0].seq_len,
        "variant": metadata.variant,
        "seed": metadata.seed,
        "is_canonical": canonical,
        "n_perms": len(permutations),
    }
    layer_rows = []
    for target in targets:
        psi_mean, psi_std = mean_and_std(module_samples[target.layer_idx])
        layer_rows.append(
            {
                **base,
                "layer_idx": target.layer_idx,
                "seq_len": target.seq_len,
                "psi_mean": psi_mean,
                "psi_std": psi_std,
            }
        )

    # Pre-registered aggregation: for every pi, average all SSM module PSI values.
    # The final configuration PSI is the arithmetic mean of those per-pi averages.
    config_samples = [
        statistics.mean(module_samples[target.layer_idx][perm_index] for target in targets)
        for perm_index in range(len(permutations))
    ]
    config_mean, config_std = mean_and_std(config_samples)
    config_row = {
        **base,
        "layer_idx": "config_mean",
        "psi_mean": config_mean,
        "psi_std": config_std,
    }
    return layer_rows, config_row


def compute_stack_checkpoint_rows(
    checkpoint: str,
    payload: Dict,
    metadata: CheckpointMetadata,
    canonical: bool,
    images: torch.Tensor,
    bank: PermutationBank,
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict]:
    model = build_model(metadata, payload, args.at_init, args.init_seed).to(images.device)
    model.eval()
    targets = enumerate_ssm_stacks(model)
    if not targets:
        raise RuntimeError(f"no SSM stacks found in {checkpoint}")

    with torch.no_grad():
        natural = capture_natural_stack_features(model, targets, images)
        permutations = bank.for_grid(metadata.grid, targets[0].seq_len, images.device)
        stack_samples = {
            target.layer_idx: [
                single_stack_permutation_psi(
                    model,
                    target,
                    natural[target.layer_idx],
                    images,
                    permutation,
                    args.epsilon,
                )
                for permutation in permutations
            ]
            for target in targets
        }

    base = {
        "checkpoint": str(checkpoint),
        "dataset": metadata.dataset,
        "block": metadata.block,
        "grid": metadata.grid,
        "seq_len": targets[0].seq_len,
        "variant": metadata.variant,
        "seed": metadata.seed,
        "is_canonical": canonical,
        "n_perms": len(permutations),
    }
    stack_rows = []
    for target in targets:
        psi_mean, psi_std = mean_and_std(stack_samples[target.layer_idx])
        stack_rows.append(
            {
                **base,
                "layer_idx": target.layer_idx,
                "seq_len": target.seq_len,
                "psi_mean": psi_mean,
                "psi_std": psi_std,
            }
        )

    # Pre-registered stack aggregation: for each pi, average every branch/group stack.
    # The configuration PSI is the arithmetic mean of those per-pi stack averages.
    config_samples = [
        statistics.mean(stack_samples[target.layer_idx][perm_index] for target in targets)
        for perm_index in range(len(permutations))
    ]
    config_mean, config_std = mean_and_std(config_samples)
    config_row = {
        **base,
        "layer_idx": "config_mean",
        "psi_mean": config_mean,
        "psi_std": config_std,
    }
    return stack_rows, config_row


def write_rows(path: str, rows: Iterable[Dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_hook_replacement_self_test() -> None:
    """Verify that a non-None forward-hook result replaces downstream input."""

    first = nn.Identity()
    second = nn.Identity()
    model = nn.Sequential(first, second)
    observed: Dict[str, torch.Tensor] = {}

    def replace_with_zeros(_module, _inputs, output):
        return torch.zeros_like(output)

    def record_downstream_input(_module, inputs):
        observed["input"] = inputs[0].detach().clone()

    replace_handle = first.register_forward_hook(replace_with_zeros)
    record_handle = second.register_forward_pre_hook(record_downstream_input)
    try:
        result = model(torch.ones(2, 3))
    finally:
        replace_handle.remove()
        record_handle.remove()
    if not torch.equal(result, torch.zeros_like(result)):
        raise AssertionError("forward-hook return value did not replace module output")
    if not torch.equal(observed["input"], torch.zeros(2, 3)):
        raise AssertionError("downstream module did not receive hook replacement output")


def main() -> None:
    args = parse_args()
    if args.self_test_hooks:
        run_hook_replacement_self_test()
        print("PASS: forward-hook return value replaces downstream module input")
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --self-test-hooks is used")
    if not args.data_root:
        raise SystemExit("--data-root is required for PSI image input")
    if args.batch_size != 128:
        raise SystemExit("PSI is pre-registered for batch_size=128")
    if args.num_permutations < 1:
        raise SystemExit("--num-permutations must be positive")
    if not args.module_level and not args.stack_level:
        raise SystemExit("enable at least one of --module-level or --stack-level")

    device = torch.device(args.device)
    bank = PermutationBank(args.permutation_seed, args.num_permutations)
    image_batches: Dict[Tuple[str, int], torch.Tensor] = {}
    module_rows: List[Dict] = []
    module_config_rows: List[Dict] = []
    stack_rows: List[Dict] = []
    stack_config_rows: List[Dict] = []
    for checkpoint in args.checkpoint:
        payload = load_payload(checkpoint)
        metadata = checkpoint_metadata(checkpoint, payload)
        canonical = validate_variant_access(metadata, args)
        dataset = args.dataset or metadata.dataset
        cache_key = (dataset, metadata.img_size)
        if cache_key not in image_batches:
            image_batches[cache_key] = load_image_batch(
                dataset, args.data_root, metadata.img_size, args
            ).to(device, non_blocking=True)
        metadata = CheckpointMetadata(**{**metadata.__dict__, "dataset": dataset})
        if args.module_level:
            rows, config_row = compute_checkpoint_rows(
                checkpoint,
                payload,
                metadata,
                canonical,
                image_batches[cache_key],
                bank,
                args,
            )
            module_rows.extend(rows)
            module_config_rows.append(config_row)
        if args.stack_level:
            rows, config_row = compute_stack_checkpoint_rows(
                checkpoint,
                payload,
                metadata,
                canonical,
                image_batches[cache_key],
                bank,
                args,
            )
            stack_rows.extend(rows)
            stack_config_rows.append(config_row)

    if args.module_level:
        write_rows(args.output_csv, module_rows)
        write_rows(args.config_output_csv, module_config_rows)
    if args.stack_level:
        write_rows(args.stack_output_csv, stack_rows)
        write_rows(args.stack_config_output_csv, stack_config_rows)


if __name__ == "__main__":
    main()
