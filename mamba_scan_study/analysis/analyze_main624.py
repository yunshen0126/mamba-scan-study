"""Read-only analysis for the preregistered 624-run main experiment."""
# Display values use Decimal(repr(x)) with ROUND_HALF_UP to avoid binary and half-even rounding.

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable, Mapping, Sequence


BACKBONES = ("mamba", "gru")
AUGMENTATIONS = ("p0b_legacy", "main_uniform")
DATASETS = ("cifar10", "organamnist", "organcmnist", "organsmnist", "eurosat")
EXP_IDS = (
    "GEO_SG1",
    "GEO_SG2",
    "GEO_SG3",
    "GEO_SG4",
    "GEO_DIV",
    "RND_S1",
    "RND_S2",
    "RND_S3",
    "RND_D1",
    "RND_D2",
    "RND_D3",
    "LOC_S",
    "LOC_D",
)
RELIANCES = ("R_low", "R_high")
SEEDS = (0, 1, 2, 3)
GEO_SINGLE = ("GEO_SG1", "GEO_SG2", "GEO_SG3", "GEO_SG4")
RND_SINGLE = ("RND_S1", "RND_S2", "RND_S3")
RND_DIVERSE = ("RND_D1", "RND_D2", "RND_D3")
STRUCTURE_EXP_IDS = GEO_SINGLE + ("GEO_DIV",)
T_CRITICAL_N4 = 3.182
DISPLAY_QUANTUM = Decimal("0.01")


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    backbone: str
    augmentation: str
    exp_id: str
    reliance: str
    seed: int
    tail_train_pp: float
    tail_validation_pp: float
    gap_pp: float
    early_validation_pp: float
    late_validation_pp: float
    metadata_path: Path


@dataclass(frozen=True)
class Summary:
    mean_pp: float
    lower_pp: float
    upper_pp: float
    values_pp: tuple[float, float, float, float]


def _fail(message: str) -> None:
    raise ValueError(message)


def _accuracy_window(history: Sequence[Mapping[str, object]], field: str, start: int, end: int) -> float:
    values: list[float] = []
    for row in history[start - 1 : end]:
        value = row.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            _fail(f"{field} is not numeric at epoch {row.get('epoch')}")
        values.append(float(value))
    return mean(values) * 100.0


def _validated_history(metadata: Mapping[str, object], metadata_path: Path) -> Sequence[Mapping[str, object]]:
    history = metadata.get("validation_history")
    if not isinstance(history, list) or len(history) != 100:
        _fail(f"{metadata_path}: validation_history must contain exactly 100 rows")
    epochs = [row.get("epoch") if isinstance(row, Mapping) else None for row in history]
    if epochs != list(range(1, 101)):
        _fail(f"{metadata_path}: validation_history epochs must be continuous 1..100")
    return history


def _record_from_metadata(metadata_path: Path) -> RunRecord:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read metadata: {metadata_path}: {error}") from error
    if not isinstance(metadata, Mapping):
        _fail(f"{metadata_path}: metadata root must be an object")

    # P0-B legacy metadata predates the dataset field; it only ever denotes CIFAR-10.
    dataset = metadata.get("dataset", "cifar10")
    backbone = metadata.get("backbone", "mamba")
    augmentation = metadata.get("augmentation", "p0b_legacy")
    exp_id = metadata.get("exp_id")
    reliance = metadata.get("reliance")
    seed = metadata.get("training_seed")
    if dataset not in DATASETS:
        _fail(f"{metadata_path}: unsupported dataset {dataset!r}")
    if backbone not in BACKBONES:
        _fail(f"{metadata_path}: unsupported backbone {backbone!r}")
    if augmentation not in AUGMENTATIONS:
        _fail(f"{metadata_path}: unsupported augmentation {augmentation!r}")
    if exp_id not in EXP_IDS or reliance not in RELIANCES or seed not in SEEDS:
        _fail(f"{metadata_path}: invalid design-cell metadata")
    history = _validated_history(metadata, metadata_path)
    tail_train = _accuracy_window(history, "train_accuracy", 80, 100)
    tail_validation = _accuracy_window(history, "validation_accuracy", 80, 100)
    return RunRecord(
        dataset=str(dataset),
        backbone=str(backbone),
        augmentation=str(augmentation),
        exp_id=str(exp_id),
        reliance=str(reliance),
        seed=int(seed),
        tail_train_pp=tail_train,
        tail_validation_pp=tail_validation,
        gap_pp=tail_train - tail_validation,
        early_validation_pp=_accuracy_window(history, "validation_accuracy", 10, 20),
        late_validation_pp=_accuracy_window(history, "validation_accuracy", 90, 100),
        metadata_path=metadata_path,
    )


def load_records(runs_root: Path) -> list[RunRecord]:
    if not runs_root.is_dir():
        _fail(f"runs root does not exist: {runs_root}")
    paths = sorted(runs_root.rglob("metadata.json"))
    if not paths:
        _fail(f"no metadata.json files below: {runs_root}")
    records = [_record_from_metadata(path) for path in paths]
    keys: set[tuple[str, str, str, str, str, int]] = set()
    for record in records:
        key = (record.dataset, record.backbone, record.augmentation, record.exp_id, record.reliance, record.seed)
        if key in keys:
            _fail(f"duplicate metadata for design cell {key}")
        keys.add(key)
    return records


def _summary(values_pp: Iterable[float]) -> Summary:
    values = tuple(float(value) for value in values_pp)
    if len(values) != 4:
        _fail(f"a design-cell contrast requires exactly four seeds, got {len(values)}")
    uncertainty = T_CRITICAL_N4 * stdev(values) / math.sqrt(4.0)
    estimate = mean(values)
    return Summary(estimate, estimate - uncertainty, estimate + uncertainty, values)  # type: ignore[arg-type]


def _validated_group(
    records: Sequence[RunRecord], dataset: str, backbone: str, augmentation: str
) -> dict[tuple[str, str, int], RunRecord]:
    group = [
        record
        for record in records
        if record.dataset == dataset and record.backbone == backbone and record.augmentation == augmentation
    ]
    if {record.augmentation for record in group} != {augmentation}:
        _fail(f"{dataset}/{backbone}: augmentation group is not consistently {augmentation}")
    expected = {(exp_id, reliance, seed) for exp_id in EXP_IDS for reliance in RELIANCES for seed in SEEDS}
    observed = {(record.exp_id, record.reliance, record.seed) for record in group}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(f"{dataset}/{backbone}: incomplete design cells; missing={missing}, extra={extra}")
    return {(record.exp_id, record.reliance, record.seed): record for record in group}


def _seed_mean(cells: Mapping[tuple[str, str, int], RunRecord], exp_ids: Sequence[str], reliance: str, seed: int) -> float:
    return mean(cells[(exp_id, reliance, seed)].tail_validation_pp for exp_id in exp_ids)


def contrast_summaries(cells: Mapping[tuple[str, str, int], RunRecord]) -> dict[str, dict[str, Summary]]:
    summaries: dict[str, dict[str, Summary]] = {}
    for reliance in RELIANCES:
        values: dict[str, list[float]] = {name: [] for name in ("P_G", "P_R", "P_LMTO", "1", "2", "3", "4", "5")}
        for seed in SEEDS:
            geo_single = _seed_mean(cells, GEO_SINGLE, reliance, seed)
            rnd_single = _seed_mean(cells, RND_SINGLE, reliance, seed)
            rnd_diverse = _seed_mean(cells, RND_DIVERSE, reliance, seed)
            geo_diverse = cells[("GEO_DIV", reliance, seed)].tail_validation_pp
            loc_single = cells[("LOC_S", reliance, seed)].tail_validation_pp
            loc_diverse = cells[("LOC_D", reliance, seed)].tail_validation_pp
            p_g = geo_diverse - geo_single
            p_r = rnd_diverse - rnd_single
            p_lmto = loc_diverse - loc_single
            values["P_G"].append(p_g)
            values["P_R"].append(p_r)
            values["P_LMTO"].append(p_lmto)
            values["1"].append(geo_single - rnd_single)
            values["2"].append(p_g - p_r)
            values["3"].append(cells[("GEO_SG1", reliance, seed)].tail_validation_pp - cells[("GEO_SG2", reliance, seed)].tail_validation_pp)
            values["4"].append(cells[("GEO_SG1", reliance, seed)].tail_validation_pp - cells[("GEO_SG3", reliance, seed)].tail_validation_pp)
            values["5"].append(p_g - p_lmto)
        summaries[reliance] = {name: _summary(seed_values) for name, seed_values in values.items()}
    return summaries


def interaction_summaries(summaries: Mapping[str, Mapping[str, Summary]]) -> dict[str, Summary]:
    return {
        name: _summary(
            high - low
            for high, low in zip(summaries["R_high"][name].values_pp, summaries["R_low"][name].values_pp)
        )
        for name in summaries["R_low"]
    }


def ceiling_rows(cells: Mapping[tuple[str, str, int], RunRecord]) -> dict[str, tuple[float, bool]]:
    rows: dict[str, tuple[float, bool]] = {}
    for reliance in RELIANCES:
        values = [cells[(exp_id, reliance, seed)].tail_train_pp for exp_id in STRUCTURE_EXP_IDS for seed in SEEDS]
        value = median(values)
        rows[reliance] = (value, value > 95.0)
    return rows


def criterion_rows(summaries: Mapping[str, Mapping[str, Summary]]) -> tuple[bool, bool, bool]:
    m1 = summaries["R_high"]["2"].lower_pp > 0.0
    m2 = all(summary.lower_pp <= 0.0 <= summary.upper_pp for summary in (summaries["R_low"]["P_R"], summaries["R_high"]["P_R"]))
    return m1, m2, m1 and m2


def _format_summary(summary: Summary) -> str:
    return (
        f"{_format_display(summary.mean_pp, signed=True)} "
        f"[{_format_display(summary.lower_pp, signed=True)}, {_format_display(summary.upper_pp, signed=True)}]"
    )


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _format_display(value: float, signed: bool = False) -> str:
    rounded = Decimal(repr(value)).quantize(DISPLAY_QUANTUM, rounding=ROUND_HALF_UP)
    text = format(rounded, "f")
    return f"+{text}" if signed and not text.startswith("-") else text


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_signed(value: float) -> str:
    return _format_display(value, signed=True)


def _format_latex_summary(summary: Summary | None) -> str:
    if summary is None:
        return r"\R"
    return (
        f"${_latex_signed(summary.mean_pp)}$ "
        f"$[{_latex_signed(summary.lower_pp)}, {_latex_signed(summary.upper_pp)}]$"
    )


def _latex_row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def _complete_cells(
    records: Sequence[RunRecord], dataset: str, backbone: str, augmentation: str
) -> dict[tuple[str, str, int], RunRecord] | None:
    group = [
        record
        for record in records
        if record.dataset == dataset and record.backbone == backbone and record.augmentation == augmentation
    ]
    expected = {(exp_id, reliance, seed) for exp_id in EXP_IDS for reliance in RELIANCES for seed in SEEDS}
    observed = {(record.exp_id, record.reliance, record.seed) for record in group}
    if observed != expected:
        return None
    return {(record.exp_id, record.reliance, record.seed): record for record in group}


# CAP-01 臂的产物目录带宽度后缀 (run_p0b_feasibility.py:623)。
# 用模块级变量而非逐层传参: 本文件是单次执行的命令行工具, 且 d_model 只影响
# 目录名解析, 不进入任何统计计算。main() 在解析参数后立即设置它。
_D_MODEL = 256


def _expected_run_directory(
    runs_root: Path, dataset: str, backbone: str, augmentation: str, exp_id: str, reliance: str, seed: int
) -> Path:
    # 与 run_p0b_feasibility.py:623 的 width_suffix 逻辑逐字一致。
    # legacy 命名空间不带宽度后缀, 且 legacy 只在 d_model=256 下存在。
    width_suffix = "" if _D_MODEL == 256 else f"_d{_D_MODEL}"
    if dataset == "cifar10" and backbone == "mamba" and augmentation == "p0b_legacy":
        name = f"p0b_{exp_id}_{reliance}_seed{seed}"
    else:
        name = f"p0b_{dataset}_{augmentation}_{backbone}_{exp_id}_{reliance}_seed{seed}{width_suffix}"
    return runs_root / name


def _completion_status(
    directory: Path,
    records_by_metadata: Mapping[Path, RunRecord],
    dataset: str,
    backbone: str,
    augmentation: str,
    exp_id: str,
    reliance: str,
    seed: int,
) -> tuple[str, RunRecord | None]:
    if not directory.is_dir():
        return "absent", None
    metadata_path = directory / "metadata.json"
    record = records_by_metadata.get(metadata_path)
    marker_path = directory / "completed.json"
    checkpoint_path = directory / "final_checkpoint.pt"
    if record is None or not marker_path.is_file() or not checkpoint_path.is_file():
        return "failed", None
    if (record.dataset, record.backbone, record.augmentation, record.exp_id, record.reliance, record.seed) != (
        dataset,
        backbone,
        augmentation,
        exp_id,
        reliance,
        seed,
    ):
        return "failed", None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "failed", None
    if not isinstance(metadata, Mapping) or not isinstance(marker, Mapping):
        return "failed", None
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    if marker.get("status") != "completed" or marker.get("metadata_sha256") != expected_sha256:
        return "failed", None
    return "completed", record


def _completion_rows(
    records: Sequence[RunRecord], runs_root: Path, dataset: str, backbone: str, augmentation: str
) -> tuple[dict[str, int], list[RunRecord]]:
    records_by_metadata = {record.metadata_path: record for record in records}
    counts = {"completed": 0, "failed": 0, "absent": 0}
    completed_records: list[RunRecord] = []
    for exp_id in EXP_IDS:
        for reliance in RELIANCES:
            for seed in SEEDS:
                status, record = _completion_status(
                    _expected_run_directory(runs_root, dataset, backbone, augmentation, exp_id, reliance, seed),
                    records_by_metadata,
                    dataset,
                    backbone,
                    augmentation,
                    exp_id,
                    reliance,
                    seed,
                )
                counts[status] += 1
                if record is not None:
                    completed_records.append(record)
    return counts, completed_records


def _latex_commit(records: Sequence[RunRecord]) -> tuple[str, tuple[str, ...]]:
    commits: set[str] = set()
    for record in records:
        try:
            metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return r"\R", ()
        commit = metadata.get("git_commit") if isinstance(metadata, Mapping) else None
        if not isinstance(commit, str) or not commit:
            return r"\R", ()
        commits.add(commit)
    if len(commits) == 1:
        return _latex_escape(next(iter(commits))[:7]), ()
    return "mixed", tuple(sorted(commits))


def _complete_datasets(
    records: Sequence[RunRecord], backbones: Sequence[str], datasets: Sequence[str], augmentation: str
) -> tuple[str, ...]:
    return tuple(
        dataset
        for dataset in datasets
        if all(_complete_cells(records, dataset, backbone, augmentation) is not None for backbone in backbones)
    )


def render_latex(
    records: Sequence[RunRecord], augmentation: str, runs_root: Path, backbones: Sequence[str]
) -> str:
    """Render the fixed Results tables without changing analysis calculations."""
    if len(backbones) != 1:
        raise ValueError("--emit latex requires exactly one --backbone")
    selected_backbone = backbones[0]
    group_order = (
        ("cifar10", "mamba"),
        ("organamnist", "mamba"),
        ("organcmnist", "mamba"),
        ("organsmnist", "mamba"),
        ("eurosat", "mamba"),
        ("cifar10", "gru"),
    )
    planned_per_group = len(EXP_IDS) * len(RELIANCES) * len(SEEDS)
    completion_by_group = {
        group: _completion_rows(records, runs_root, group[0], group[1], augmentation) for group in group_order
    }
    cells_by_group = {
        group: _complete_cells(completed_records, group[0], group[1], augmentation)
        for group, (_, completed_records) in completion_by_group.items()
    }
    summaries_by_dataset: dict[str, dict[str, dict[str, Summary]]] = {}
    interactions_by_dataset: dict[str, dict[str, Summary]] = {}
    criteria_by_dataset: dict[str, tuple[bool, bool, bool]] = {}
    ceilings_by_dataset: dict[str, dict[str, tuple[float, bool]]] = {}
    table_datasets = tuple(
        dataset for dataset in DATASETS if cells_by_group.get((dataset, selected_backbone)) is not None
    )
    for dataset in table_datasets:
        cells = cells_by_group[(dataset, selected_backbone)]
        if cells is None:
            continue
        summaries = contrast_summaries(cells)
        summaries_by_dataset[dataset] = summaries
        interactions_by_dataset[dataset] = interaction_summaries(summaries)
        criteria_by_dataset[dataset] = criterion_rows(summaries)
        ceilings_by_dataset[dataset] = ceiling_rows(cells)

    lines = [
        "% ==== TABLE: completeness ====",
        "% source: _completion_rows",
        _latex_row("group", "planned", "completed", "failed", "absent", "commit"),
    ]
    all_group_records: list[RunRecord] = []
    total_counts = {"completed": 0, "failed": 0, "absent": 0}
    commit_comments: list[str] = []
    for dataset, backbone in group_order:
        counts, completed_records = completion_by_group[(dataset, backbone)]
        all_group_records.extend(completed_records)
        for status in total_counts:
            total_counts[status] += counts[status]
        commit, commits = _latex_commit(completed_records) if completed_records else (r"\R", ())
        if commits:
            commit_comments.append(
                f"% commits for {_latex_escape(f'{dataset}_{backbone}')}: {', '.join(_latex_escape(item[:7]) for item in commits)}"
            )
        lines.append(
            _latex_row(
                _latex_escape(f"{dataset}_{backbone}"),
                str(planned_per_group),
                str(counts["completed"]),
                str(counts["failed"]),
                str(counts["absent"]),
                commit,
            )
        )
    total_commit, total_commits = _latex_commit(all_group_records) if all_group_records else (r"\R", ())
    lines.append(
        _latex_row(
            "total",
            str(planned_per_group * len(group_order)),
            str(total_counts["completed"]),
            str(total_counts["failed"]),
            str(total_counts["absent"]),
            total_commit,
        )
    )
    if total_commits:
        commit_comments.append(
            "% commits for total: " + ", ".join(_latex_escape(item[:7]) for item in total_commits)
        )
    lines.extend(commit_comments)

    lines.extend(
        [
            "",
            "% ==== TABLE: contrasts ====",
            "% source: contrast_summaries, interaction_summaries",
            _latex_row("dataset", "quantity", "low load", "high load", "paired difference"),
        ]
    )
    contrast_rows = (
        (r"\ding{172} structure", "1"),
        (r"\ding{173} $P_G - P_R$", "2"),
        (r"\quad $P_G$", "P_G"),
        (r"\quad $P_R$", "P_R"),
        (r"\ding{174} polarity", "3"),
        (r"\ding{175} axis", "4"),
    )
    for dataset in table_datasets:
        summaries = summaries_by_dataset[dataset]
        interaction = interactions_by_dataset[dataset]
        for label, name in contrast_rows:
            lines.append(
                _latex_row(
                    _latex_escape(dataset),
                    label,
                    _format_latex_summary(summaries["R_low"][name]),
                    _format_latex_summary(summaries["R_high"][name]),
                    _format_latex_summary(interaction[name]),
                )
            )

    lines.extend(
        [
            "",
            "% ==== TABLE: criteria ====",
            "% source: criterion_rows",
            _latex_row("dataset", "M1", "M2", r"M1$\wedge$M2"),
        ]
    )
    for dataset in table_datasets:
        criteria = criteria_by_dataset[dataset]
        values = tuple(_format_bool(value) for value in criteria)
        lines.append(_latex_row(_latex_escape(dataset), *values))
    votes = [criteria[2] for criteria in criteria_by_dataset.values()]
    proposition = (
        f"{_format_bool(sum(votes) >= 4)} ({sum(votes)}/5)"
        if len(criteria_by_dataset) == len(DATASETS)
        else f"not evaluable ({len(criteria_by_dataset)}/5)"
    )
    lines.append(_latex_row("Proposition A", rf"\multicolumn{{3}}{{l}}{{{proposition}}}"))

    lines.extend(
        [
            "",
            "% ==== TABLE: ceiling ====",
            "% source: ceiling_rows",
            _latex_row("dataset", "low load", "flagged", "high load", "flagged"),
        ]
    )
    for dataset in table_datasets:
        rows = ceilings_by_dataset[dataset]
        low = rows["R_low"]
        high = rows["R_high"]
        lines.append(
            _latex_row(
                _latex_escape(dataset),
                _format_display(low[0]),
                _format_bool(low[1]),
                _format_display(high[0]),
                _format_bool(high[1]),
            )
        )

    lines.extend(
        [
            "",
            "% ==== TABLE: exploratory ====",
            "% source: contrast_summaries",
            _latex_row("dataset", "quantity", "low load", "high load"),
        ]
    )
    for dataset in table_datasets:
        summaries = summaries_by_dataset[dataset]
        for label, name in ((r"\ding{176} $P_G - P_L$", "5"), (r"\quad $P_L$", "P_LMTO")):
            lines.append(
                _latex_row(
                    _latex_escape(dataset),
                    label,
                    _format_latex_summary(summaries["R_low"][name]),
                    _format_latex_summary(summaries["R_high"][name]),
                )
            )
    return "\n".join(lines)


def _comparison_table(summaries: Mapping[str, Mapping[str, Summary]]) -> list[str]:
    rows = ["| comparison | R_low (pp, 95% CI) | R_high (pp, 95% CI) | paired R_high - R_low (pp, 95% CI) |", "|---|---:|---:|---:|"]
    interaction = interaction_summaries(summaries)
    for name in ("1", "2", "3", "4", "5", "P_G", "P_R", "P_LMTO"):
        rows.append(f"| {name} | {_format_summary(summaries['R_low'][name])} | {_format_summary(summaries['R_high'][name])} | {_format_summary(interaction[name])} |")
    return rows


def _diagnostic_table(cells: Mapping[tuple[str, str, int], RunRecord]) -> list[str]:
    rows = ["| exp_id | reliance | tail train (pp) | tail validation (pp) | train - validation (pp) | val epoch 10-20 (pp) | val epoch 90-100 (pp) |", "|---|---|---:|---:|---:|---:|---:|"]
    for exp_id in EXP_IDS:
        for reliance in RELIANCES:
            cell_records = [cells[(exp_id, reliance, seed)] for seed in SEEDS]
            rows.append(
                f"| {exp_id} | {reliance} | {_format_display(mean(row.tail_train_pp for row in cell_records))} | "
                f"{_format_display(mean(row.tail_validation_pp for row in cell_records))} | {_format_display(mean(row.gap_pp for row in cell_records), signed=True)} | "
                f"{_format_display(mean(row.early_validation_pp for row in cell_records))} | {_format_display(mean(row.late_validation_pp for row in cell_records))} |"
            )
    return rows


def analyze(
    records: Sequence[RunRecord], backbones: Sequence[str], datasets: Sequence[str], augmentation: str = "main_uniform"
) -> tuple[dict[tuple[str, str], dict[str, dict[str, Summary]]], str]:
    summaries_by_group: dict[tuple[str, str], dict[str, dict[str, Summary]]] = {}
    lines = ["# Main-624 Analysis", "", f"Augmentation: {augmentation}", "", "Primary endpoint: validation accuracy. No official test split is read, computed, or reported.", "", "Intervals are mean +/- 3.182 * s / sqrt(4), in pp; no family-wise error correction is applied to the five contrasts."]
    criterion_data: dict[tuple[str, str], tuple[bool, bool, bool]] = {}
    ceiling_data: dict[tuple[str, str], dict[str, tuple[float, bool]]] = {}
    diagnostic_data: dict[tuple[str, str], dict[tuple[str, str, int], RunRecord]] = {}
    for backbone in backbones:
        for dataset in datasets:
            cells = _validated_group(records, dataset, backbone, augmentation)
            summaries = contrast_summaries(cells)
            key = (dataset, backbone)
            summaries_by_group[key] = summaries
            criterion_data[key] = criterion_rows(summaries)
            ceiling_data[key] = ceiling_rows(cells)
            diagnostic_data[key] = cells
            lines.extend(["", f"## Contrasts: {dataset}, {backbone}", ""] + _comparison_table(summaries))
    for backbone in backbones:
        exploratory = " EXPLORATORY (§7.2)" if backbone == "gru" else ""
        lines.extend(["", f"## Criteria: {backbone}{exploratory}", "", "| dataset | M1 | M2 | M1 and M2 |", "|---|---|---|---|"])
        backbone_rows = [(dataset, row) for (dataset, candidate), row in criterion_data.items() if candidate == backbone]
        for dataset, (m1, m2, proposition_a) in backbone_rows:
            lines.append(f"| {dataset} | {_format_bool(m1)} | {_format_bool(m2)} | {_format_bool(proposition_a)} |")
        if backbone != "mamba":
            continue
        participating_datasets = {dataset for dataset, _ in backbone_rows}
        if participating_datasets != set(DATASETS):
            lines.append(f"\nProposition A: NOT EVALUABLE (需要全部五个数据集，当前 {len(participating_datasets)} 个)")
            continue
        votes = [row[2] for _, row in backbone_rows]
        lines.append(f"\nProposition A: {_format_bool(sum(votes) >= 4)} ({sum(votes)}/5 datasets satisfy M1 and M2).")
        r_high = {dataset: summaries_by_group[(dataset, backbone)]["R_high"]["2"] for dataset in DATASETS}
        weakest_organ_name = min(("organamnist", "organcmnist", "organsmnist"), key=lambda name: r_high[name].mean_pp)
        weakest_organ = r_high[weakest_organ_name]
        cifar = r_high["cifar10"]
        eurosat = r_high["eurosat"]
        if weakest_organ.mean_pp > cifar.mean_pp > eurosat.mean_pp:
            m3 = "order holds" if weakest_organ.lower_pp > cifar.upper_pp and cifar.lower_pp > eurosat.upper_pp else "direction consistent, discrimination insufficient"
        else:
            m3 = "order falsified"
        lines.append(f"M3: {m3}; weakest Organ={weakest_organ_name} {_format_summary(weakest_organ)}, cifar10 {_format_summary(cifar)}, eurosat {_format_summary(eurosat)}.")
    lines.extend(["", "## Ceiling Diagnostics", "", "| dataset | backbone | reliance | structure-group median tail train (pp) | strong saturation (>95%) |", "|---|---|---|---:|---|"])
    for (dataset, backbone), rows in ceiling_data.items():
        for reliance, (median_pp, saturated) in rows.items():
            lines.append(f"| {dataset} | {backbone} | {reliance} | {_format_display(median_pp)} | {_format_bool(saturated)} |")
    for (dataset, backbone), cells in diagnostic_data.items():
        lines.extend(["", f"## Diagnostics: {dataset}, {backbone}", ""] + _diagnostic_table(cells))
    return summaries_by_group, "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset", choices=DATASETS, action="append", dest="datasets")
    parser.add_argument("--backbone", choices=BACKBONES, action="append", dest="backbones")
    parser.add_argument("--augmentation", choices=AUGMENTATIONS, default="main_uniform")
    parser.add_argument("--emit", choices=("markdown", "latex"), default="markdown")
    parser.add_argument("--d-model", type=int, default=256,
                        help="产物目录名的宽度后缀; 256 时无后缀, 与既有 624 run 一致")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    global _D_MODEL
    _D_MODEL = args.d_model
    backbones = tuple(args.backbones) if args.backbones else ("mamba",)
    records = load_records(args.runs_root)
    if args.emit == "latex":
        print(render_latex(records, args.augmentation, args.runs_root, backbones))
        return
    datasets = tuple(args.datasets) if args.datasets else _complete_datasets(
        records, backbones, DATASETS, args.augmentation
    )
    if not datasets:
        raise ValueError("no complete datasets for the requested --backbone and --augmentation")
    _, report = analyze(records, backbones, datasets, args.augmentation)
    print(report)


if __name__ == "__main__":
    main()
