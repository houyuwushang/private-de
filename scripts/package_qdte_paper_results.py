#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from path_defaults import baseline_root, external_results, external_runs, repo_root
    from plot_qdte_paper_results import build_figures
except ModuleNotFoundError:
    from scripts.path_defaults import baseline_root, external_results, external_runs, repo_root
    from scripts.plot_qdte_paper_results import build_figures


ROOT = repo_root()
BASELINE_ROOT = baseline_root()
RUNS_ROOT = external_runs()
RESULTS_ROOT = external_results()
LEGACY_PACKAGE = RESULTS_ROOT / "paper_package_seed0to4_20260706"
DEFAULT_OUTPUT = RESULTS_ROOT / "qdte_paper_package_20260711"
CLAIM_MATRIX = ROOT / "docs" / "QDTE_PAPER_CLAIM_MATRIX_20260711.json"
CLAIM_MATRIX_MD = ROOT / "docs" / "QDTE_PAPER_CLAIM_MATRIX_20260711.md"
TRANSFER_SUMMARY = RESULTS_ROOT / "rtp_transfer_evidence_summary_20260708.json"
FIGURE_BUILDER = ROOT / "scripts" / "plot_qdte_paper_results.py"

DATASET_ORDER = [
    "adult_sage_strong",
    "acs_sage_strong",
    "br2000_sage_strong",
    "nltcs_sage_strong",
]
DATASET_LABELS = {
    "adult_sage_strong": "Adult",
    "acs_sage_strong": "ACS",
    "br2000_sage_strong": "BR2000",
    "nltcs_sage_strong": "NLTCS",
}
METRICS = {
    "mae": "full_true_mae",
    "rmse": "full_true_rmse",
    "avg_tvd": "full_true_avg_tvd",
    "max_tvd": "full_true_max_tvd",
    "max_error": "full_true_max_error",
}
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "avg_tvd": "AvgTVD",
    "max_tvd": "MaxTVD",
    "max_error": "MaxError",
}

PRIMARY_SPECS = [
    (
        "QDTE-Standard",
        "tables/sage_all4_seed0to4_summary_rho1_20260706.csv",
        {
            "mae": "MAE_mean",
            "rmse": "RMSE_mean",
            "avg_tvd": "AvgTVD_mean",
            "max_error": "MaxErr_mean",
            "max_tvd": "MaxTVD_mean",
        },
        "n",
    ),
    (
        "Private-GSD GPU 1M/full-N",
        "tables/private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
        {
            "mae": "MAE_mean",
            "rmse": "RMSE_mean",
            "avg_tvd": "AvgTVD_mean",
            "max_error": "MaxErr_mean",
            "max_tvd": "MaxTVD_mean",
        },
        "n",
    ),
    (
        "Private-PGM AIM",
        "tables/private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv",
        {
            "mae": "MAE_mean",
            "rmse": "RMSE_mean",
            "avg_tvd": "AvgTVD_mean",
            "max_error": "MaxErr_mean",
            "max_tvd": "MaxTVD_mean",
        },
        "n",
    ),
    (
        "Private-PGM MST",
        "tables/private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv",
        {
            "mae": "MAE_mean",
            "rmse": "RMSE_mean",
            "avg_tvd": "AvgTVD_mean",
            "max_error": "MaxErr_mean",
            "max_tvd": "MaxTVD_mean",
        },
        "n",
    ),
    (
        "RAP softmax",
        "tables/rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
        {
            "mae": "full_true_mae_mean",
            "rmse": "full_true_rmse_mean",
            "avg_tvd": "full_true_avg_tvd_mean",
            "max_error": "full_true_max_error_mean",
            "max_tvd": "full_true_max_tvd_mean",
        },
        "seed_count",
    ),
]


@dataclass(frozen=True)
class GeneratorSpec:
    dataset: str
    qdte_run: Path
    gsd_run: Path


GENERATOR_SPECS = [
    GeneratorSpec(
        "acs_sage_strong",
        RUNS_ROOT
        / "qdte_structured_optimized_gate1_v1/qdte/phase_b/acs_sage_strong/seed0",
        RUNS_ROOT / "qdte_gsd_converged_no_noise_v1/gsd/acs_sage_strong/seed0",
    ),
    GeneratorSpec(
        "br2000_sage_strong",
        RUNS_ROOT
        / "qdte_structured_optimized_gate1_v1/qdte/phase_b/br2000_sage_strong/seed0",
        RUNS_ROOT
        / "qdte_gsd_no_noise_breadth_v1/gsd/br2000_sage_strong/seed0",
    ),
    GeneratorSpec(
        "adult_sage_strong",
        RUNS_ROOT / "qdte_gsd_no_noise_breadth_v1/qdte/phase_b/adult_sage_strong/seed0",
        RUNS_ROOT / "qdte_gsd_no_noise_breadth_v1/gsd/adult_sage_strong/seed0",
    ),
]

STRUCTURED_VARIANTS = {
    "QDTE-Structured-v2": RUNS_ROOT
    / "qdte_structured_standard_v2_optimized_fixed_noise_seed0",
    "QDTE-Structured-SA": RUNS_ROOT
    / "qdte_structured_search_aware_v1_fixed_noise_seed0",
}
STRUCTURED_BASE_RUNS = {
    "acs_sage_strong": RUNS_ROOT
    / "qdte_interleaved_swap_v2_fixed_noise_seed0/base/acs_sage_strong/seed0",
    "adult_sage_strong": RUNS_ROOT
    / "qdte_interleaved_swap_v3_fixed_noise_seed0/base/adult_sage_strong/seed0",
    "br2000_sage_strong": RUNS_ROOT
    / "qdte_interleaved_swap_v3_fixed_noise_seed0/base/br2000_sage_strong/seed0",
    "nltcs_sage_strong": RUNS_ROOT
    / "qdte_interleaved_swap_v3_fixed_noise_seed0/base/nltcs_sage_strong/seed0",
}
FISSION_CONTROL_ROOT = RUNS_ROOT / "qdte_structured_sa_terminal_fixed_noise_seed3"
FISSION_ROOT = RUNS_ROOT / "qdte_structured_fission_refit_v2_fixed_noise_seed3"

TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceTracker:
    def __init__(self) -> None:
        self._entries: dict[Path, set[str]] = {}

    def add(self, path: Path, role: str) -> None:
        resolved = path.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(resolved)
        self._entries.setdefault(resolved, set()).add(str(role))

    def normalize(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return "$SAGE_BASELINE_ROOT/" + str(resolved.relative_to(BASELINE_ROOT.resolve()))
        except ValueError:
            pass
        try:
            return "$QDTE_REPO_ROOT/" + str(resolved.relative_to(ROOT.resolve()))
        except ValueError as exc:
            raise ValueError(f"source path is outside known roots: {resolved}") from exc

    def rows(self) -> list[dict[str, Any]]:
        output = []
        for path, roles in sorted(self._entries.items(), key=lambda item: str(item[0])):
            output.append(
                {
                    "path": self.normalize(path),
                    "roles": sorted(roles),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        return output


def _read_json(path: Path, tracker: SourceTracker, role: str) -> dict[str, Any]:
    tracker.add(path, role)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path, tracker: SourceTracker, role: str) -> list[dict[str, str]]:
    tracker.add(path, role)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _evaluation_path(run_dir: Path) -> Path:
    for name in ("external_evaluation.json", "evaluation_external.json", "evaluation.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing external evaluation in {run_dir}")


def _load_evaluation(run_dir: Path, tracker: SourceTracker, role: str) -> dict[str, Any]:
    return _read_json(_evaluation_path(run_dir), tracker, role)


def _load_metrics(run_dir: Path, tracker: SourceTracker, role: str) -> dict[str, Any]:
    return _read_json(run_dir / "metrics_final.json", tracker, role)


def _record_run_files(run_dir: Path, tracker: SourceTracker, role: str) -> None:
    for name in (
        "run_metadata.json",
        "run_status.json",
        "pipeline_metadata.json",
        "config_resolved.yaml",
        "metrics_final.json",
        "metrics_timeseries.csv",
        "runtime.json",
        "external_evaluation.json",
        "evaluation_external.json",
        "evaluation.json",
        "measurements.json",
        "queries.json",
        "schema.json",
        "synthetic_encoded.npy",
    ):
        path = run_dir / name
        if path.exists():
            tracker.add(path, role)


def _status(run_dir: Path, tracker: SourceTracker, role: str) -> str:
    for name in ("pipeline_metadata.json", "run_metadata.json", "run_status.json"):
        path = run_dir / name
        if not path.exists():
            continue
        data = _read_json(path, tracker, role)
        status = str(data.get("status", ""))
        if status in {"completed", "complete"}:
            return "completed"
        if status in {"failed", "interrupted", "running"}:
            return status
    required = [run_dir / "metrics_final.json", run_dir / "synthetic_encoded.npy"]
    if all(path.exists() for path in required) and _evaluation_path(run_dir).exists():
        return "completed_by_artifacts"
    return "unknown"


def _require_completed(run_dir: Path, tracker: SourceTracker, role: str) -> None:
    status = _status(run_dir, tracker, role)
    if status not in {"completed", "completed_by_artifacts"}:
        raise RuntimeError(f"run is not completed: {run_dir} ({status})")
    _record_run_files(run_dir, tracker, role)


def _runtime_seconds(run_dir: Path, tracker: SourceTracker, role: str) -> float:
    candidates = [
        ("run_metadata.json", "runtime_seconds"),
        ("metrics_final.json", "runtime_seconds"),
        ("runtime.json", "wall_clock_seconds"),
    ]
    for filename, key in candidates:
        path = run_dir / filename
        if not path.exists():
            continue
        data = _read_json(path, tracker, role)
        value = data.get(key)
        if value is not None and math.isfinite(float(value)) and float(value) >= 0.0:
            return float(value)
    raise ValueError(f"missing finite runtime for {run_dir}")


def _source_run(run_dir: Path, tracker: SourceTracker, role: str) -> Path | None:
    for filename in ("run_metadata.json", "metrics_final.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        data = _read_json(path, tracker, role)
        source = data.get("source_run")
        if source:
            return Path(str(source))
    return None


def _stage_chain(endpoint: Path, tracker: SourceTracker, role: str) -> list[Path]:
    chain: list[Path] = []
    seen: set[Path] = set()

    def visit(run_dir: Path) -> None:
        resolved = run_dir.resolve()
        if resolved in seen:
            raise ValueError(f"cycle in source_run chain: {resolved}")
        seen.add(resolved)
        source = _source_run(resolved, tracker, role)
        if source is not None:
            visit(source)
        chain.append(resolved)

    visit(endpoint)
    return chain


def _loss_from_metrics(metrics: dict[str, Any]) -> float:
    for key in ("final_measured_loss", "target_count_loss"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    raise ValueError("metrics do not contain a final target loss")


def _chain_curve(
    endpoint: Path, tracker: SourceTracker, role: str
) -> tuple[list[dict[str, float | str]], float, list[Path]]:
    chain = _stage_chain(endpoint, tracker, role)
    curve: list[dict[str, float | str]] = []
    offset = 0.0
    for stage in chain:
        _require_completed(stage, tracker, role)
        duration = _runtime_seconds(stage, tracker, role)
        timeseries_path = stage / "metrics_timeseries.csv"
        if timeseries_path.exists():
            rows = _read_csv(timeseries_path, tracker, role)
            for row in rows:
                wall_time = row.get("wall_time_seconds", row.get("wall_time", ""))
                if wall_time == "":
                    continue
                loss_text = row.get("measured_loss", "")
                if loss_text == "":
                    continue
                curve.append(
                    {
                        "time_seconds": offset + float(wall_time),
                        "loss": float(loss_text),
                        "stage": tracker.normalize(stage),
                    }
                )
        metrics = _load_metrics(stage, tracker, role)
        curve.append(
            {
                "time_seconds": offset + duration,
                "loss": _loss_from_metrics(metrics),
                "stage": tracker.normalize(stage),
            }
        )
        offset += duration
    curve.sort(key=lambda row: (float(row["time_seconds"]), float(row["loss"])))
    return curve, offset, chain


def _metric_values(evaluation: dict[str, Any]) -> dict[str, float]:
    return {metric: float(evaluation[field]) for metric, field in METRICS.items()}


def _relative_change_percent(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline must be positive")
    return 100.0 * (candidate / baseline - 1.0)


def _latex_escape(text: str) -> str:
    for source, replacement in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _fmt(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\times 10^{{{int(exponent)}}}$"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float) -> str:
    return f"{value:+.2f}\\%"


def _table(caption: str, label: str, columns: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def collect_primary(tracker: SourceTracker) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    for method, relpath, columns, n_column in PRIMARY_SPECS:
        path = LEGACY_PACKAGE / relpath
        for source in _read_csv(path, tracker, "primary_dp"):
            row: dict[str, Any] = {
                "dataset": source["dataset"],
                "dataset_label": DATASET_LABELS[source["dataset"]],
                "method": method,
                "seeds": int(source[n_column]),
            }
            for metric, column in columns.items():
                row[metric] = float(source[column])
                row[f"{metric}_sem"] = float(source[column.replace("_mean", "_sem")])
            if method == "RAP softmax":
                row["runtime_seconds"] = float(source["runtime_seconds_mean"])
                row["runtime_sem"] = float(source["runtime_seconds_sem"])
            else:
                row["runtime_seconds"] = float(source["runtime_s_mean"])
                row["runtime_sem"] = float(source["runtime_s_sem"])
            normalized.append(row)

    lookup = {(row["dataset"], row["method"]): row for row in normalized}
    standard = "QDTE-Standard"
    ratios: list[dict[str, Any]] = []
    for method in (
        "RAP softmax",
        "Private-PGM AIM",
        "Private-PGM MST",
        "Private-GSD GPU 1M/full-N",
    ):
        metric_ratios: dict[str, list[float]] = {metric: [] for metric in METRICS}
        wins = 0
        for dataset in DATASET_ORDER:
            base = lookup[(dataset, standard)]
            other = lookup[(dataset, method)]
            for metric in METRICS:
                ratio = float(other[metric]) / float(base[metric])
                metric_ratios[metric].append(ratio)
                wins += int(ratio > 1.0)
        ratios.append(
            {
                "baseline": method,
                "qdte_wins": wins,
                "cells": len(DATASET_ORDER) * len(METRICS),
                **{
                    f"{metric}_ratio_min": min(values)
                    for metric, values in metric_ratios.items()
                },
                **{
                    f"{metric}_ratio_max": max(values)
                    for metric, values in metric_ratios.items()
                },
            }
        )
    return normalized, ratios


def collect_generator(tracker: SourceTracker) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for spec in GENERATOR_SPECS:
        _require_completed(spec.qdte_run, tracker, "generator_qdte")
        _require_completed(spec.gsd_run, tracker, "generator_gsd")
        qdte_metrics = _load_metrics(spec.qdte_run, tracker, "generator_qdte")
        gsd_metrics = _load_metrics(spec.gsd_run, tracker, "generator_gsd")
        qdte_eval = _load_evaluation(spec.qdte_run, tracker, "generator_qdte")
        gsd_eval = _load_evaluation(spec.gsd_run, tracker, "generator_gsd")
        qdte_loss = _loss_from_metrics(qdte_metrics)
        gsd_loss = _loss_from_metrics(gsd_metrics)
        curve, qdte_total_runtime, chain = _chain_curve(
            spec.qdte_run, tracker, "generator_qdte_chain"
        )
        gsd_runtime = _runtime_seconds(spec.gsd_run, tracker, "generator_gsd")
        crossing = [point for point in curve if float(point["loss"]) < gsd_loss]
        if not crossing:
            raise RuntimeError(f"QDTE never crosses GSD endpoint for {spec.dataset}")
        first_pass = min(float(point["time_seconds"]) for point in crossing)
        available = [
            point for point in curve if float(point["time_seconds"]) <= gsd_runtime + 1.0e-9
        ]
        if not available:
            raise RuntimeError(f"no QDTE checkpoint before GSD completion for {spec.dataset}")
        loss_at_gsd_time = float(max(available, key=lambda point: float(point["time_seconds"]))["loss"])
        qdte_utility = _metric_values(qdte_eval)
        gsd_utility = _metric_values(gsd_eval)
        row: dict[str, Any] = {
            "dataset": spec.dataset,
            "dataset_label": DATASET_LABELS[spec.dataset],
            "seed": 0,
            "qdte_run": tracker.normalize(spec.qdte_run),
            "gsd_run": tracker.normalize(spec.gsd_run),
            "qdte_chain": [tracker.normalize(path) for path in chain],
            "qdte_target_loss": qdte_loss,
            "gsd_target_loss": gsd_loss,
            "qdte_target_loss_reduction_percent": 100.0 * (1.0 - qdte_loss / gsd_loss),
            "qdte_total_runtime_seconds": qdte_total_runtime,
            "gsd_runtime_seconds": gsd_runtime,
            "qdte_runtime_over_gsd": qdte_total_runtime / gsd_runtime,
            "qdte_time_to_first_pass_seconds": first_pass,
            "time_to_first_pass_over_gsd": first_pass / gsd_runtime,
            "qdte_loss_at_gsd_completion": loss_at_gsd_time,
            "qdte_better_at_gsd_completion": loss_at_gsd_time < gsd_loss,
        }
        for metric in METRICS:
            row[f"qdte_{metric}"] = qdte_utility[metric]
            row[f"gsd_{metric}"] = gsd_utility[metric]
            row[f"qdte_over_gsd_{metric}"] = qdte_utility[metric] / gsd_utility[metric]
            row[f"qdte_wins_{metric}"] = qdte_utility[metric] < gsd_utility[metric]
        rows.append(row)
        for point in curve:
            curves.append(
                {
                    "dataset": spec.dataset,
                    "dataset_label": DATASET_LABELS[spec.dataset],
                    **point,
                    "gsd_final_loss": gsd_loss,
                    "gsd_runtime_seconds": gsd_runtime,
                }
            )
    rows.sort(key=lambda row: DATASET_ORDER.index(str(row["dataset"])))
    return rows, curves


def collect_structured_gate(tracker: SourceTracker) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, root in STRUCTURED_VARIANTS.items():
        for dataset in DATASET_ORDER:
            base_run = STRUCTURED_BASE_RUNS[dataset]
            candidate_run = root / dataset / "seed0"
            _require_completed(base_run, tracker, "structured_gate_base")
            _require_completed(candidate_run, tracker, "structured_gate_candidate")
            base_metrics = _load_metrics(base_run, tracker, "structured_gate_base")
            candidate_metrics = _load_metrics(
                candidate_run, tracker, "structured_gate_candidate"
            )
            base_eval = _load_evaluation(base_run, tracker, "structured_gate_base")
            candidate_eval = _load_evaluation(
                candidate_run, tracker, "structured_gate_candidate"
            )
            row: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "variant": variant,
                "base_run": tracker.normalize(base_run),
                "candidate_run": tracker.normalize(candidate_run),
                "measurement_sha256_match": _sha256(base_run / "measurements.json")
                == _sha256(candidate_run / "measurements.json"),
                "measured_loss_change_percent": _relative_change_percent(
                    _loss_from_metrics(candidate_metrics), _loss_from_metrics(base_metrics)
                ),
                "runtime_ratio": _runtime_seconds(
                    candidate_run, tracker, "structured_gate_candidate"
                )
                / _runtime_seconds(base_run, tracker, "structured_gate_base"),
            }
            base_utility = _metric_values(base_eval)
            candidate_utility = _metric_values(candidate_eval)
            for metric in METRICS:
                row[f"{metric}_change_percent"] = _relative_change_percent(
                    candidate_utility[metric], base_utility[metric]
                )
            rows.append(row)

    summary: dict[str, Any] = {}
    for variant in STRUCTURED_VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "datasets": len(selected),
            "measured_loss_wins": sum(
                float(row["measured_loss_change_percent"]) < 0.0 for row in selected
            ),
            "runtime_within_1p2x": sum(float(row["runtime_ratio"]) <= 1.2 for row in selected),
            "mae_wins": sum(float(row["mae_change_percent"]) < 0.0 for row in selected),
            "rmse_wins": sum(float(row["rmse_change_percent"]) < 0.0 for row in selected),
            "avg_tvd_wins": sum(
                float(row["avg_tvd_change_percent"]) < 0.0 for row in selected
            ),
            "replacement_gate_passed": False,
        }
    return rows, summary


def collect_fission(tracker: SourceTracker) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        control = FISSION_CONTROL_ROOT / dataset / "seed3"
        pipeline = FISSION_ROOT / dataset / "seed3"
        refit = pipeline / "refit"
        _require_completed(control, tracker, "fission_control")
        _require_completed(pipeline, tracker, "fission_pipeline")
        _require_completed(refit, tracker, "fission_refit")
        pipeline_metadata = _read_json(
            pipeline / "pipeline_metadata.json", tracker, "fission_pipeline"
        )
        control_eval = _load_evaluation(control, tracker, "fission_control")
        refit_eval = _load_evaluation(refit, tracker, "fission_refit")
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "seed": 3,
            "control_run": tracker.normalize(control),
            "pipeline_run": tracker.normalize(pipeline),
            "selected_iteration": int(pipeline_metadata["selected_iteration"]),
            "terminal_iteration": int(pipeline_metadata["terminal_iteration"]),
            "selection_rule": str(pipeline_metadata["selection_rule"]),
            "pipeline_runtime_seconds": float(pipeline_metadata["runtime_seconds"]),
            "control_runtime_seconds": _runtime_seconds(
                control, tracker, "fission_control"
            ),
        }
        row["runtime_ratio"] = row["pipeline_runtime_seconds"] / row["control_runtime_seconds"]
        control_utility = _metric_values(control_eval)
        refit_utility = _metric_values(refit_eval)
        for metric in METRICS:
            row[f"{metric}_change_percent"] = _relative_change_percent(
                refit_utility[metric], control_utility[metric]
            )
        rows.append(row)

    mae_wins = sum(float(row["mae_change_percent"]) < 0.0 for row in rows)
    rmse_wins = sum(float(row["rmse_change_percent"]) < 0.0 for row in rows)
    l2_changes = [
        float(row[f"{metric}_change_percent"])
        for row in rows
        for metric in ("mae", "rmse")
    ]
    summary = {
        "datasets": len(rows),
        "mae_wins": mae_wins,
        "rmse_wins": rmse_wins,
        "combined_wins": mae_wins + rmse_wins,
        "combined_cells": 2 * len(rows),
        "largest_l2_regression_percent": max(l2_changes),
        "gate_passed": mae_wins == 4
        and rmse_wins >= 3
        and max(l2_changes) <= 2.0,
    }
    return rows, summary


def collect_transfer(tracker: SourceTracker) -> dict[str, Any]:
    summary = _read_json(TRANSFER_SUMMARY, tracker, "transfer_summary")
    sanitized: dict[str, dict[str, Any]] = {}
    for section in ("attrs0_14", "attrs3_4", "attrs3_4_transfer", "teacher_target"):
        sanitized[section] = dict(summary[section])
        raw_path = summary[section].get("path")
        if raw_path:
            source_path = Path(str(raw_path))
            tracker.add(source_path, f"transfer_{section}")
            sanitized[section]["path"] = tracker.normalize(source_path)
    return {
        "rtp_attrs0_14": sanitized["attrs0_14"],
        "rtp_attrs3_4": sanitized["attrs3_4"],
        "transfer_decomposition": sanitized["attrs3_4_transfer"],
        "teacher_target": sanitized["teacher_target"],
        "decision": summary["decision"],
    }


def _render_primary_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                _latex_escape(str(row["baseline"])),
                f"{int(row['qdte_wins'])}/{int(row['cells'])}",
                f"{float(row['mae_ratio_min']):.2f}--{float(row['mae_ratio_max']):.2f}x",
                f"{float(row['rmse_ratio_min']):.2f}--{float(row['rmse_ratio_max']):.2f}x",
                f"{float(row['avg_tvd_ratio_min']):.2f}--{float(row['avg_tvd_ratio_max']):.2f}x",
            ]
        )
    return _table(
        "Primary end-to-end DP utility at $\\rho=1$ over seeds 0--4. Ratios are baseline error divided by QDTE-Standard error; values above one favor QDTE-Standard.",
        "tab:qdte-primary-summary",
        "lcccc",
        ["Baseline", "QDTE wins", "MAE ratio", "RMSE ratio", "AvgTVD ratio"],
        body,
    )


def _render_primary_full(rows: list[dict[str, Any]]) -> str:
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    methods = [
        "QDTE-Standard",
        "RAP softmax",
        "Private-GSD GPU 1M/full-N",
        "Private-PGM AIM",
        "Private-PGM MST",
    ]
    body = []
    for dataset in DATASET_ORDER:
        for method in methods:
            row = lookup[(dataset, method)]
            body.append(
                [
                    str(row["dataset_label"]),
                    _latex_escape(method),
                    *[
                        rf"{_fmt(float(row[metric]))} $\pm$ {_fmt(float(row[f'{metric}_sem']))}"
                        for metric in ("mae", "rmse", "avg_tvd", "max_error", "max_tvd")
                    ],
                ]
            )
    return _table(
        "Full end-to-end row-level utility at $\\rho=1$ over seeds 0--4. Entries are mean $\\pm$ SEM; lower is better.",
        "tab:qdte-primary-full",
        "llccccc",
        ["Dataset", "Method", "MAE", "RMSE", "AvgTVD", "MaxError", "MaxTVD"],
        body,
    )


def _render_primary_runtime(rows: list[dict[str, Any]]) -> str:
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    methods = [
        "QDTE-Standard",
        "RAP softmax",
        "Private-GSD GPU 1M/full-N",
        "Private-PGM AIM",
        "Private-PGM MST",
    ]
    body = []
    for dataset in DATASET_ORDER:
        for method in methods:
            row = lookup[(dataset, method)]
            body.append(
                [
                    str(row["dataset_label"]),
                    _latex_escape(method),
                    rf"{float(row['runtime_seconds']):.1f} $\pm$ {float(row['runtime_sem']):.1f}",
                ]
            )
    return _table(
        "Wall-clock runtime for the primary row-level comparison. Entries are mean $\\pm$ SEM in seconds over seeds 0--4.",
        "tab:qdte-primary-runtime",
        "llc",
        ["Dataset", "Method", "Runtime (s)"],
        body,
    )


def _render_generator_endpoint(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                str(row["dataset_label"]),
                _fmt(float(row["qdte_target_loss"])),
                _fmt(float(row["gsd_target_loss"])),
                _fmt_pct(-float(row["qdte_target_loss_reduction_percent"])),
                f"{float(row['qdte_over_gsd_mae']):.3f}",
                f"{float(row['qdte_over_gsd_rmse']):.3f}",
                f"{float(row['qdte_over_gsd_avg_tvd']):.3f}",
                f"{float(row['qdte_over_gsd_max_tvd']):.3f}",
                f"{float(row['qdte_over_gsd_max_error']):.3f}",
            ]
        )
    return _table(
        "Controlled same-target no-noise generator endpoints. Utility columns are QDTE-Structured/GSD ratios; values below one favor QDTE. Target-loss delta is relative to GSD.",
        "tab:qdte-structured-gsd-endpoint",
        "lrrrrrrrr",
        [
            "Dataset",
            "QDTE loss",
            "GSD loss",
            "$\\Delta$ loss",
            "MAE",
            "RMSE",
            "AvgTVD",
            "MaxTVD",
            "MaxError",
        ],
        body,
    )


def _render_generator_time(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                str(row["dataset_label"]),
                f"{float(row['qdte_total_runtime_seconds']):.1f}",
                f"{float(row['gsd_runtime_seconds']):.1f}",
                f"{float(row['qdte_runtime_over_gsd']):.2f}x",
                f"{float(row['qdte_time_to_first_pass_seconds']):.1f}",
                f"{float(row['time_to_first_pass_over_gsd']):.2f}x",
                _fmt(float(row["qdte_loss_at_gsd_completion"])),
            ]
        )
    return _table(
        "Wall-time qualification for the controlled generator comparison. First pass is the earliest QDTE checkpoint below GSD's final target loss.",
        "tab:qdte-structured-gsd-time",
        "lrrrrrr",
        [
            "Dataset",
            "QDTE total (s)",
            "GSD total (s)",
            "QDTE/GSD",
            "First pass (s)",
            "Pass/GSD",
            "QDTE loss at GSD time",
        ],
        body,
    )


def _render_structured_gate(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                str(row["dataset_label"]),
                _latex_escape(str(row["variant"])),
                _fmt_pct(float(row["measured_loss_change_percent"])),
                _fmt_pct(float(row["mae_change_percent"])),
                _fmt_pct(float(row["rmse_change_percent"])),
                _fmt_pct(float(row["avg_tvd_change_percent"])),
                _fmt_pct(float(row["max_tvd_change_percent"])),
                _fmt_pct(float(row["max_error_change_percent"])),
                f"{float(row['runtime_ratio']):.2f}x",
            ]
        )
    return _table(
        "Fixed-noise seed0 Structured gate relative to its frozen matching QDTE-Standard control. Negative percentages are improvements.",
        "tab:qdte-structured-dp-gate",
        "llrrrrrrr",
        [
            "Dataset",
            "Variant",
            "Released loss",
            "MAE",
            "RMSE",
            "AvgTVD",
            "MaxTVD",
            "MaxError",
            "Runtime",
        ],
        body,
    )


def _render_fission(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                str(row["dataset_label"]),
                str(int(row["selected_iteration"])),
                _fmt_pct(float(row["mae_change_percent"])),
                _fmt_pct(float(row["rmse_change_percent"])),
                _fmt_pct(float(row["avg_tvd_change_percent"])),
                _fmt_pct(float(row["max_tvd_change_percent"])),
                _fmt_pct(float(row["max_error_change_percent"])),
                f"{float(row['runtime_ratio']):.2f}x",
            ]
        )
    return _table(
        "Predeclared seed3 QDTE-FissionRefit confirmation relative to the matching Structured-SA terminal control. Negative percentages are improvements.",
        "tab:qdte-fission-refit-l2",
        "lrrrrrrr",
        [
            "Dataset",
            "Selected iter.",
            "MAE",
            "RMSE",
            "AvgTVD",
            "MaxTVD",
            "MaxError",
            "Runtime",
        ],
        body,
    )


def _render_transfer(data: dict[str, Any]) -> str:
    rtp = data["rtp_attrs0_14"]["aggregates"]["QDTE-RTP-local"]
    decomposition = data["transfer_decomposition"]["relative_vs_baseline_pct"]
    teacher = data["teacher_target"]
    body = [
        [
            "RTP-local attrs0,14 (5 seeds)",
            _fmt_pct(float(rtp["measured_loss_rel_vs_QDTE-Base_pct"])),
            _fmt_pct(float(rtp["mae_rel_vs_QDTE-Base_pct"])),
            _fmt_pct(float(rtp["avg_tvd_rel_vs_QDTE-Base_pct"])),
            _fmt_pct(float(rtp["max_error_rel_vs_QDTE-Base_pct"])),
        ],
        [
            "RTP-local attrs3,4 transfer",
            _fmt_pct(float(decomposition["T_target_to_true_norm2"])),
            _fmt_pct(float(decomposition["F_synthetic_to_target_norm2"])),
            _fmt_pct(float(decomposition["E_synthetic_to_true_norm2"])),
            "--",
        ],
        [
            "Row-realizable teacher target",
            f"loss ratio {float(teacher['loss_ratio_vs_baseline']):.3f}",
            _fmt(float(teacher["teacher_rate_mae"])),
            _fmt(float(teacher["teacher_rate_rmse"])),
            "no true-answer generation",
        ],
    ]
    return _table(
        "Transfer-gap diagnostics. Columns have diagnostic-specific meanings; the table shows that target/fit gains need not transfer to final utility, while a row-realizable teacher is fitted much more tightly.",
        "tab:qdte-transfer-gap",
        "lrrrr",
        ["Diagnostic", "Target/loss signal", "Fit/MAE signal", "Final/RMSE signal", "Tail/boundary"],
        body,
    )


def _copy_claims(output_dir: Path, tracker: SourceTracker) -> None:
    claims = output_dir / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for source in (CLAIM_MATRIX, CLAIM_MATRIX_MD):
        tracker.add(source, "claim_matrix")
        shutil.copy2(source, claims / source.name)


def _write_file_manifests(output_dir: Path) -> None:
    excluded = {"FILELIST.txt", "SHA256SUMS.txt"}
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    relpaths = [str(path.relative_to(output_dir)) for path in files]
    (output_dir / "FILELIST.txt").write_text("\n".join(relpaths) + "\n", encoding="utf-8")
    hash_files = files + [output_dir / "FILELIST.txt"]
    lines = [
        f"{_sha256(path)}  {path.relative_to(output_dir)}"
        for path in sorted(hash_files, key=lambda path: str(path.relative_to(output_dir)))
    ]
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(output_dir: Path, *, force: bool, copy_legacy: bool = True) -> dict[str, Any]:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    tracker = SourceTracker()
    if copy_legacy:
        if not LEGACY_PACKAGE.exists():
            raise FileNotFoundError(LEGACY_PACKAGE)
        shutil.copytree(LEGACY_PACKAGE, output_dir / "legacy_seed0to4")
    _copy_claims(output_dir, tracker)

    primary_rows, primary_ratios = collect_primary(tracker)
    generator_rows, generator_curves = collect_generator(tracker)
    structured_rows, structured_summary = collect_structured_gate(tracker)
    fission_rows, fission_summary = collect_fission(tracker)
    transfer = collect_transfer(tracker)

    tables = output_dir / "tables"
    _write_csv(
        tables / "primary_dp_rows.csv",
        primary_rows,
        [
            "dataset",
            "dataset_label",
            "method",
            "seeds",
            *[
                field
                for metric in METRICS
                for field in (metric, f"{metric}_sem")
            ],
            "runtime_seconds",
            "runtime_sem",
        ],
    )
    ratio_fields = ["baseline", "qdte_wins", "cells"]
    for metric in METRICS:
        ratio_fields.extend([f"{metric}_ratio_min", f"{metric}_ratio_max"])
    _write_csv(tables / "primary_dp_summary.csv", primary_ratios, ratio_fields)
    generator_fields = [
        "dataset",
        "dataset_label",
        "seed",
        "qdte_run",
        "gsd_run",
        "qdte_target_loss",
        "gsd_target_loss",
        "qdte_target_loss_reduction_percent",
        "qdte_total_runtime_seconds",
        "gsd_runtime_seconds",
        "qdte_runtime_over_gsd",
        "qdte_time_to_first_pass_seconds",
        "time_to_first_pass_over_gsd",
        "qdte_loss_at_gsd_completion",
        "qdte_better_at_gsd_completion",
    ]
    for metric in METRICS:
        generator_fields.extend(
            [
                f"qdte_{metric}",
                f"gsd_{metric}",
                f"qdte_over_gsd_{metric}",
                f"qdte_wins_{metric}",
            ]
        )
    _write_csv(tables / "structured_vs_gsd_endpoint.csv", generator_rows, generator_fields)
    _write_json(tables / "structured_vs_gsd_endpoint.json", {"rows": generator_rows})
    _write_csv(
        tables / "structured_vs_gsd_time_curve.csv",
        generator_curves,
        [
            "dataset",
            "dataset_label",
            "time_seconds",
            "loss",
            "stage",
            "gsd_final_loss",
            "gsd_runtime_seconds",
        ],
    )
    structured_fields = [
        "dataset",
        "dataset_label",
        "variant",
        "base_run",
        "candidate_run",
        "measurement_sha256_match",
        "measured_loss_change_percent",
        "runtime_ratio",
        *[f"{metric}_change_percent" for metric in METRICS],
    ]
    _write_csv(tables / "structured_dp_gate.csv", structured_rows, structured_fields)
    _write_json(
        tables / "structured_dp_gate_summary.json", structured_summary
    )
    fission_fields = [
        "dataset",
        "dataset_label",
        "seed",
        "control_run",
        "pipeline_run",
        "selected_iteration",
        "terminal_iteration",
        "selection_rule",
        "pipeline_runtime_seconds",
        "control_runtime_seconds",
        "runtime_ratio",
        *[f"{metric}_change_percent" for metric in METRICS],
    ]
    _write_csv(tables / "fission_refit_l2.csv", fission_rows, fission_fields)
    _write_json(tables / "fission_refit_l2_summary.json", fission_summary)
    _write_json(tables / "transfer_gap_diagnostic.json", transfer)

    (tables / "table_primary_dp_summary.tex").write_text(
        _render_primary_table(primary_ratios), encoding="utf-8"
    )
    (tables / "table_primary_dp_full.tex").write_text(
        _render_primary_full(primary_rows), encoding="utf-8"
    )
    (tables / "table_primary_runtime.tex").write_text(
        _render_primary_runtime(primary_rows), encoding="utf-8"
    )
    (tables / "table_structured_vs_gsd_endpoint.tex").write_text(
        _render_generator_endpoint(generator_rows), encoding="utf-8"
    )
    (tables / "table_structured_vs_gsd_time_to_quality.tex").write_text(
        _render_generator_time(generator_rows), encoding="utf-8"
    )
    (tables / "table_structured_dp_gate.tex").write_text(
        _render_structured_gate(structured_rows), encoding="utf-8"
    )
    (tables / "table_fission_refit_l2.tex").write_text(
        _render_fission(fission_rows), encoding="utf-8"
    )
    (tables / "table_transfer_gap_diagnostic.tex").write_text(
        _render_transfer(transfer), encoding="utf-8"
    )
    tracker.add(FIGURE_BUILDER, "figure_builder")
    build_figures(output_dir)

    source_rows = tracker.rows()
    _write_json(output_dir / "source_manifest.json", {"sources": source_rows})
    facts = {
        "QDTE-C1": {
            row["baseline"]: {
                "qdte_wins": row["qdte_wins"],
                "cells": row["cells"],
            }
            for row in primary_ratios
        },
        "QDTE-C4": {
            "datasets": len(generator_rows),
            "target_loss_wins": sum(
                float(row["qdte_target_loss"]) < float(row["gsd_target_loss"])
                for row in generator_rows
            ),
            "offline_metric_wins": sum(
                bool(row[f"qdte_wins_{metric}"])
                for row in generator_rows
                for metric in METRICS
            ),
            "offline_metric_cells": len(generator_rows) * len(METRICS),
        },
        "QDTE-C5": structured_summary,
        "QDTE-C6": fission_summary,
        "QDTE-C7": {
            "target_change_percent": transfer["transfer_decomposition"][
                "relative_vs_baseline_pct"
            ]["T_target_to_true_norm2"],
            "fit_change_percent": transfer["transfer_decomposition"][
                "relative_vs_baseline_pct"
            ]["F_synthetic_to_target_norm2"],
            "final_change_percent": transfer["transfer_decomposition"][
                "relative_vs_baseline_pct"
            ]["E_synthetic_to_true_norm2"],
            "teacher_loss_ratio_vs_baseline": transfer["teacher_target"][
                "loss_ratio_vs_baseline"
            ],
        },
    }
    metadata = {
        "package": "QDTE paper evidence package",
        "version": "20260711-v2",
        "status": "complete",
        "legacy_primary_package": "$SAGE_BASELINE_ROOT/external_results/paper_package_seed0to4_20260706",
        "claim_matrix_sha256": _sha256(CLAIM_MATRIX),
        "num_source_artifacts": len(source_rows),
        "facts": facts,
        "dp_boundary": (
            "Exact true answers are read only from completed external-evaluation artifacts; "
            "they are not used to generate or select DP synthetic data."
        ),
    }
    _write_json(output_dir / "package_metadata.json", metadata)
    _write_file_manifests(output_dir)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the latest QDTE paper evidence package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-copy-legacy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_package(
        args.output_dir,
        force=bool(args.force),
        copy_legacy=not bool(args.no_copy_legacy),
    )
    print(f"QDTE paper package: {args.output_dir}")
    print(json.dumps(metadata["facts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
