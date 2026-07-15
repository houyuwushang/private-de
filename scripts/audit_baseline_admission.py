#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from path_defaults import baseline_root, external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import baseline_root, external_results, external_runs

BASELINE_ROOT = baseline_root()
RESULTS_ROOT = external_results()
RUNS_ROOT = external_runs()

DATASET_ORDER = ["adult_sage_strong", "acs_sage_strong", "br2000_sage_strong", "nltcs_sage_strong"]

METRIC_KEYS = {
    "mae": ("full_true_mae", "mae", "MAE"),
    "rmse": ("full_true_rmse", "rmse", "RMSE"),
    "avg_tvd": ("full_true_avg_tvd", "avg_tvd", "AvgTVD"),
    "max_error": ("full_true_max_error", "max_error", "MaxErr"),
    "max_tvd": ("full_true_max_tvd", "max_tvd", "MaxTVD"),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    expected_tier: str
    admission: str
    run_globs: tuple[str, ...]
    result_files: tuple[str, ...]
    result_method_keywords: tuple[str, ...]
    expected_datasets: int
    expected_seeds: int
    gpu_required: bool
    caveat: str
    next_action: str


CANDIDATES = [
    Candidate(
        name="RAP softmax",
        expected_tier="primary",
        admission="promote_primary_table",
        run_globs=("rap/*_sage_strong/rho1p0/seed*_T30K30M*I1000/run_metadata.json",),
        result_files=(
            "rap_sage_strong_stress_all4_seed0to4_rho1_20260706.csv",
            "rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
        ),
        result_method_keywords=("rap",),
        expected_datasets=4,
        expected_seeds=5,
        gpu_required=True,
        caveat="Complete four-dataset, five-seed GPU row-level baseline under the shared evaluator; not RAP++.",
        next_action="Include in the primary row-level utility table; no immediate rerun needed.",
    ),
    Candidate(
        name="GEM",
        expected_tier="appendix_diagnosis",
        admission="appendix_diagnosis_only",
        run_globs=("gem/*_sage_strong/rho1p0/*/run_metadata.json",),
        result_files=(
            "gem_adult_sage_strong_calibration_seed0_20260706.csv",
            "gem_row_realization_diagnostics_adult_sage_strong_20260706.csv",
            "gem_remap_all4_seed0_20260707.csv",
        ),
        result_method_keywords=(),
        expected_datasets=4,
        expected_seeds=1,
        gpu_required=True,
        caveat="GPU wrapper is runnable and the public transformer-coordinate remap removes the relaxed-to-row artifact, but seed0 all-dataset evidence remains much weaker than primary baselines.",
        next_action="Keep GEM in appendix/diagnostic evidence; run seed0--4 only if a reviewer asks for expanded GEM coverage.",
    ),
    Candidate(
        name="PrivMRF GPU",
        expected_tier="appendix_calibration",
        admission="appendix_calibration_only",
        run_globs=("privmrf_gpu/*_sage_strong/rho1p0/*/run_metadata.json",),
        result_files=(
            "privmrf_gpu_adult_calibration_seed0_20260706.csv",
            "privmrf_gpu_br2000_probe_seed0_20260706.csv",
        ),
        result_method_keywords=("privmrf",),
        expected_datasets=4,
        expected_seeds=3,
        gpu_required=True,
        caveat="Upstream GPU branch is dataset-name constrained and current rows remain much weaker.",
        next_action="Keep appendix/calibration; upgrade only if reviewer explicitly asks for PrivMRF breadth.",
    ),
    Candidate(
        name="PrivMRF official TVD",
        expected_tier="original_protocol",
        admission="original_protocol_reproduced",
        run_globs=("privmrf_official/*/run_metadata.json",),
        result_files=("privmrf_official_full_tvd_epsgrid_m300_20260706.csv",),
        result_method_keywords=("privmrf official",),
        expected_datasets=4,
        expected_seeds=1,
        gpu_required=False,
        caveat=(
            "Official PrivMRF TVD epsilon grid completed for nltcs, acs, adult, "
            "and br2000 with marginal_num=300; keep separate from SAGE's shared evaluator."
        ),
        next_action=(
            "Report as original-protocol reproduced evidence; use SAGE-matched "
            "PrivMRF rows only for ablation or appendix diagnostics."
        ),
    ),
    Candidate(
        name="DataSynthesizer PrivBayes",
        expected_tier="appendix_audit",
        admission="appendix_audit_only",
        run_globs=("datasynth_privbayes/*_sage_strong/rho1p0/*/run_metadata.json",),
        result_files=("datasynth_privbayes_adult_strong_seed0_20260706.csv",),
        result_method_keywords=("datasynthesizer",),
        expected_datasets=4,
        expected_seeds=3,
        gpu_required=False,
        caveat="CPU implementation; only Adult seed0 is currently calibrated under the strong evaluator.",
        next_action="Report as the cleaner PrivBayes audit row; avoid main table unless expanded and still meaningful.",
    ),
    Candidate(
        name="DPMM PrivBayes",
        expected_tier="appendix_failure_calibration",
        admission="appendix_failure_calibration_only",
        run_globs=("dpmm_privbayes/*/rho1p0/*/run_metadata.json",),
        result_files=("dpmm_privbayes_adult_calibration_seed0_20260706.csv",),
        result_method_keywords=("privbayes_degree", "dpmm privbayes"),
        expected_datasets=4,
        expected_seeds=3,
        gpu_required=False,
        caveat="CPU/NumPy path; degree/compression calibration remains far from competitive.",
        next_action="Do not spend more time unless replacing the implementation or changing the paper question.",
    ),
    Candidate(
        name="PrivSyn unofficial",
        expected_tier="appendix_audit",
        admission="appendix_unofficial_only",
        run_globs=("privsyn_unofficial/*_sage_strong/rho1p0/*/run_metadata.json",),
        result_files=("privsyn_unofficial_adult_strong_seed0_20260706.csv",),
        result_method_keywords=("privsyn",),
        expected_datasets=4,
        expected_seeds=3,
        gpu_required=False,
        caveat="Unofficial course-project implementation with wrapper-generated configuration patches.",
        next_action="Keep as transparency appendix; do not promote to main evidence.",
    ),
    Candidate(
        name="RAP++ official ACS grid",
        expected_tier="original_protocol",
        admission="original_protocol_reproduced",
        run_globs=("rappp_official_paper_grid/acs_*_*/eps1p0/seed*/run_metadata.json",),
        result_files=("rappp_official_paper_grid_seed0to4_20260707.csv",),
        result_method_keywords=(),
        expected_datasets=25,
        expected_seeds=5,
        gpu_required=False,
        caveat=(
            "Upstream ACS/Folktables protocol with RAP++ defaults and original-style "
            "marginal, prefix, and downstream metrics; separate from the strict SAGE evaluator."
        ),
        next_action=(
            "Report as original-protocol reproduced evidence; keep separate from the "
            "strict row-level evaluator unless a faithful same-protocol RAP++ bridge is built."
        ),
    ),
    Candidate(
        name="RAP++ marginal-only",
        expected_tier="path_check",
        admission="path_check_only",
        run_globs=("rappp_marginal/*_sage_strong/rho1p0/*/run_metadata.json",),
        result_files=("rappp_marginal_pathcheck_20260706.csv",),
        result_method_keywords=("rappp",),
        expected_datasets=4,
        expected_seeds=3,
        gpu_required=True,
        caveat="Uses RAP++ projection code with marginal statistics only; not the official full RAP++ halfspace/task-target setup.",
        next_action="Decide between a faithful RAP++ semantic conversion or leaving this as path-check only.",
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _metric_from_row(row: dict[str, Any], metric: str) -> float:
    for key in METRIC_KEYS[metric]:
        if key in row:
            return _safe_float(row[key])
    return math.nan


def _is_gpu_metadata(metadata: dict[str, Any]) -> bool:
    notes = metadata.get("notes") or {}
    torch_device = str(metadata.get("torch_device", ""))
    if metadata.get("torch_cuda_available") is True and torch_device.startswith("cuda"):
        return True
    if str(metadata.get("jax_default_backend", "")).lower() == "gpu":
        return True
    if str(notes.get("jax_default_backend", "")).lower() == "gpu":
        return True
    probe = metadata.get("jax_device_probe") or {}
    platforms = [str(value).lower() for value in probe.get("jax_device_platforms", [])]
    devices = [str(value).lower() for value in probe.get("jax_devices", [])]
    if "gpu" in platforms or any(device.startswith("cuda") for device in devices):
        return True
    if notes.get("cupy_device_count"):
        return int(notes.get("cupy_device_count") or 0) > 0
    return False


def _load_run_rows(candidate: Candidate, runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in candidate.run_globs:
        for metadata_path in sorted(runs_root.glob(pattern)):
            metadata = _read_json(metadata_path)
            evaluation_path = metadata_path.parent / "evaluation.json"
            evaluation = _read_json(evaluation_path) if evaluation_path.exists() else {}
            datasets = metadata.get("datasets") or [metadata.get("dataset") or metadata.get("dataset_name") or evaluation.get("dataset")]
            if not isinstance(datasets, list):
                datasets = [datasets]
            for dataset in datasets:
                rows.append(
                    {
                        "source": str(metadata_path),
                        "source_type": "run_metadata",
                        "dataset": dataset,
                        "seed": metadata.get("seed", "0" if metadata.get("method") == "privmrf_official" else None),
                        "status": metadata.get("status", "unknown"),
                        "runtime_seconds": metadata.get("runtime_seconds", evaluation.get("runtime_seconds")),
                        "gpu": _is_gpu_metadata(metadata),
                        "conda_env": metadata.get("conda_env"),
                        "n_real": metadata.get("n_real"),
                        "n_synthetic": metadata.get("n_synthetic"),
                        "mae": evaluation.get("full_true_mae"),
                        "rmse": evaluation.get("full_true_rmse"),
                        "avg_tvd": evaluation.get("full_true_avg_tvd"),
                        "max_error": evaluation.get("full_true_max_error"),
                        "max_tvd": evaluation.get("full_true_max_tvd"),
                    }
                )
    return rows


def _load_result_rows(candidate: Candidate, results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in candidate.result_files:
        path = results_root / name
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                method_text = str(row.get("method") or row.get("run") or row.get("source") or "").lower()
                if candidate.result_method_keywords and not any(
                    keyword.lower() in method_text for keyword in candidate.result_method_keywords
                ):
                    continue
                rows.append(
                    {
                        "source": str(path),
                        "source_type": "result_csv",
                        "dataset": row.get("dataset")
                        or row.get("dataset_name")
                        or ("adult_sage_strong" if "adult" in name else ""),
                        "seed": row.get("seed") or ("0" if "seed0" in name else ""),
                        "status": row.get("status", "completed"),
                        "runtime_seconds": row.get("runtime_seconds") or row.get("runtime_s") or row.get("runtime_seconds_mean"),
                        "gpu": row.get("torch_device", "").startswith("cuda")
                        or row.get("torch_devices", "").startswith("cuda")
                        or row.get("expected_gpu", "") == "True",
                        "conda_env": row.get("conda_env"),
                        "n_real": row.get("n_real"),
                        "n_synthetic": row.get("n_synthetic") or row.get("n_syn"),
                        "mae": _metric_from_row(row, "mae")
                        if not row.get("marginal_average_error")
                        else _safe_float(row.get("marginal_average_error")),
                        "rmse": _metric_from_row(row, "rmse"),
                        "avg_tvd": _safe_float(row.get("tvd_mean"))
                        if row.get("tvd_mean")
                        else (
                            _safe_float(row.get("prefix_average_error"))
                            if row.get("prefix_average_error")
                            else _metric_from_row(row, "avg_tvd")
                        ),
                        "max_error": _metric_from_row(row, "max_error")
                        if not row.get("marginal_max_error")
                        else _safe_float(row.get("marginal_max_error")),
                        "max_tvd": _metric_from_row(row, "max_tvd"),
                    }
                )
    return rows


def _unique_sorted(values: list[Any]) -> list[str]:
    clean = {str(value) for value in values if value is not None and str(value) != ""}
    return sorted(clean, key=lambda x: (DATASET_ORDER.index(x) if x in DATASET_ORDER else 999, x))


def _best_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    numeric = [row for row in rows if not math.isnan(_safe_float(row.get("mae")))]
    if not numeric:
        return None
    return min(numeric, key=lambda row: _safe_float(row.get("mae")))


def _admission_score(candidate: Candidate, run_rows: list[dict[str, Any]]) -> str:
    completed = [row for row in run_rows if row.get("status") == "completed"]
    datasets = _unique_sorted([row.get("dataset") for row in completed])
    seeds = _unique_sorted([row.get("seed") for row in completed])
    gpu_ok = (not candidate.gpu_required) or bool(completed and all(bool(row.get("gpu")) for row in completed))
    coverage_ok = len(datasets) >= candidate.expected_datasets and len(seeds) >= candidate.expected_seeds
    if coverage_ok and gpu_ok and candidate.expected_tier == "original_protocol":
        return "admitted_original_protocol"
    if coverage_ok and gpu_ok and candidate.expected_tier in {"primary", "main"}:
        return "admitted_" + candidate.expected_tier
    if coverage_ok and gpu_ok:
        return "complete_but_not_promoted"
    if completed:
        return "partial_evidence"
    return "missing_or_failed"


def required_admission_errors(rows: list[dict[str, Any]]) -> list[str]:
    expected_status = {
        "primary": "admitted_primary",
        "original_protocol": "admitted_original_protocol",
    }
    errors: list[str] = []
    for row in rows:
        expected = expected_status.get(str(row.get("expected_tier", "")))
        if expected is None:
            continue
        observed = str(row.get("machine_status", ""))
        if observed != expected:
            errors.append(
                f"{row.get('candidate')}: machine_status is {observed!r}, "
                f"expected {expected!r}"
            )
    return errors


def audit(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs_root = args.baseline_root / "external_runs"
    results_root = args.baseline_root / "external_results"
    out: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        run_rows = _load_run_rows(candidate, runs_root)
        result_rows = _load_result_rows(candidate, results_root)
        completed = [row for row in run_rows if row.get("status") == "completed"]
        datasets = _unique_sorted([row.get("dataset") for row in completed])
        seeds = _unique_sorted([row.get("seed") for row in completed])
        gpu_runs = sum(1 for row in completed if row.get("gpu"))
        best = _best_metric_row(run_rows + result_rows)
        out.append(
            {
                "candidate": candidate.name,
                "expected_tier": candidate.expected_tier,
                "admission": candidate.admission,
                "machine_status": _admission_score(candidate, run_rows),
                "completed_runs": len(completed),
                "datasets": ",".join(datasets),
                "dataset_count": len(datasets),
                "seeds": ",".join(seeds),
                "seed_count": len(seeds),
                "gpu_runs": gpu_runs,
                "gpu_required": candidate.gpu_required,
                "gpu_complete": bool(completed and gpu_runs == len(completed)) if candidate.gpu_required else "",
                "result_files_found": sum(1 for name in candidate.result_files if (results_root / name).exists()),
                "best_mae": _safe_float(best.get("mae")) if best else math.nan,
                "best_rmse": _safe_float(best.get("rmse")) if best else math.nan,
                "best_avg_tvd": _safe_float(best.get("avg_tvd")) if best else math.nan,
                "best_max_error": _safe_float(best.get("max_error")) if best else math.nan,
                "best_max_tvd": _safe_float(best.get("max_tvd")) if best else math.nan,
                "best_source_type": best.get("source_type") if best else "",
                "best_source": best.get("source") if best else "",
                "caveat": candidate.caveat,
                "next_action": candidate.next_action,
            }
        )
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "" if value is None else str(value)


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "candidate",
        "expected_tier",
        "machine_status",
        "completed_runs",
        "dataset_count",
        "seed_count",
        "gpu_runs",
        "best_mae",
        "admission",
        "next_action",
    ]
    lines = [
        "# Baseline Admission Audit",
        "",
        "Machine-readable admission pass for strict same-protocol, original-protocol, and appendix/audit baselines. Lower MAE is better for shared-evaluator rows. For original-protocol reproduced rows, `best_mae` records the closest available paper-protocol average-error proxy and should not be mixed with shared-evaluator MAE.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)).replace("|", "/") for col in cols) + " |")
    lines += [
        "",
        "## Caveats",
        "",
    ]
    for row in rows:
        lines.append(f"- **{row['candidate']}**: {row['caveat']}")
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paper-admission status for baseline evidence tiers.")
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_ROOT / "baseline_admission_audit_20260706.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_ROOT / "baseline_admission_audit_20260706.md",
    )
    parser.add_argument(
        "--allow-nonadmitted-required",
        action="store_true",
        help="Write the audit files but do not fail when primary/original-protocol rows lose admission.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = audit(args)
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    print(f"wrote markdown to {args.output_md}")
    if not args.allow_nonadmitted_required:
        errors = required_admission_errors(rows)
        if errors:
            print("baseline admission audit failed:")
            for error in errors:
                print(f"- {error}")
            return 1
    print("baseline admission audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
