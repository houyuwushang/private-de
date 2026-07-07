#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from path_defaults import paper_package_dir
except ModuleNotFoundError:
    from scripts.path_defaults import paper_package_dir

DEFAULT_PACKAGE_DIR = paper_package_dir()

DATASETS = {
    "adult_sage_strong",
    "acs_sage_strong",
    "br2000_sage_strong",
    "nltcs_sage_strong",
}
METRICS = ["MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"]
METRIC_MEAN_COLS = {
    "MAE": "MAE_mean",
    "RMSE": "RMSE_mean",
    "AvgTVD": "AvgTVD_mean",
    "MaxErr": "MaxErr_mean",
    "MaxTVD": "MaxTVD_mean",
}
RAP_METRIC_MEAN_COLS = {
    "MAE": "full_true_mae_mean",
    "RMSE": "full_true_rmse_mean",
    "AvgTVD": "full_true_avg_tvd_mean",
    "MaxErr": "full_true_max_error_mean",
    "MaxTVD": "full_true_max_tvd_mean",
}

PRIMARY_METHODS = [
    "SAGE",
    "RAP softmax",
    "Private-GSD GPU 1M/full-N",
    "Private-PGM AIM",
    "Private-PGM MST",
]

TRACEABILITY_IDS = {
    "primary-table-coverage",
    "sage-vs-aim",
    "sage-vs-mst",
    "sage-vs-rap",
    "sage-vs-gsd",
    "gpu-provenance",
    "baseline-tiering",
    "certified-selector-boundary",
    "sage-ablation-coverage",
}


@dataclass(frozen=True)
class SummarySpec:
    relpath: str
    method: str
    method_slug: str | None
    n_col: str
    metric_cols: dict[str, str] | None = None


@dataclass(frozen=True)
class PairwiseValueSpec:
    relpath: str
    summary_relpath: str
    other_label: str
    other_prefix: str
    other_n_col: str
    ratio_style: str
    require_winner_cols: bool
    other_summary_metric_cols: dict[str, str]


@dataclass
class ClaimCheck:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


SUMMARY_SPECS = [
    SummarySpec(
        "tables/sage_all4_seed0to4_summary_rho1_20260706.csv",
        "SAGE",
        "sage",
        "n",
        METRIC_MEAN_COLS,
    ),
    SummarySpec(
        "tables/private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
        "Private-GSD GPU 1M/full-N",
        "private_gsd_gpu_1m_fulln_audit",
        "n",
        METRIC_MEAN_COLS,
    ),
    SummarySpec(
        "tables/private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv",
        "Private-PGM AIM",
        "private_pgm_aim",
        "n",
        METRIC_MEAN_COLS,
    ),
    SummarySpec(
        "tables/private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv",
        "Private-PGM MST",
        "private_pgm_mst",
        "n",
        METRIC_MEAN_COLS,
    ),
    SummarySpec(
        "tables/rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
        "rap_softmax",
        None,
        "seed_count",
        RAP_METRIC_MEAN_COLS,
    ),
]

PAIRWISE_VALUE_SPECS = [
    PairwiseValueSpec(
        "tables/sage_vs_private_pgm_aim_seed0to4_20260706.csv",
        "tables/private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv",
        "Private-PGM AIM",
        "aim",
        "aim_n",
        "{metric}_ratio_{other}_over_sage",
        False,
        METRIC_MEAN_COLS,
    ),
    PairwiseValueSpec(
        "tables/sage_vs_private_pgm_mst_seed0to4_20260706.csv",
        "tables/private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv",
        "Private-PGM MST",
        "mst",
        "mst_n",
        "{metric}_ratio_{other}_over_sage",
        False,
        METRIC_MEAN_COLS,
    ),
    PairwiseValueSpec(
        "tables/sage_vs_rap_softmax_seed0to4_20260706.csv",
        "tables/rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
        "RAP softmax",
        "rap",
        "rap_n",
        "{other}_over_sage_{metric}",
        True,
        RAP_METRIC_MEAN_COLS,
    ),
    PairwiseValueSpec(
        "tables/sage_vs_private_gsd_1m_fulln_seed0to4_20260706.csv",
        "tables/private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
        "Private-GSD GPU 1M/full-N",
        "gsd",
        "gsd_n",
        "{other}_over_sage_{metric}",
        True,
        METRIC_MEAN_COLS,
    ),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _int(row: dict[str, str], key: str, context: str) -> int:
    try:
        return int(row[key])
    except Exception as exc:
        raise ValueError(f"{context}: invalid integer in {key!r}: {row.get(key)!r}") from exc


def _float(row: dict[str, str], key: str, context: str) -> float:
    try:
        return float(row[key])
    except Exception as exc:
        raise ValueError(f"{context}: invalid float in {key!r}: {row.get(key)!r}") from exc


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1e-8, abs_tol=1e-12)


def _require_file(package_dir: Path, relpath: str, errors: list[str]) -> Path | None:
    path = package_dir / relpath
    if not path.exists():
        errors.append(f"missing artifact: {relpath}")
        return None
    return path


def _check_summary_files(package_dir: Path, errors: list[str]) -> None:
    for spec in SUMMARY_SPECS:
        path = _require_file(package_dir, spec.relpath, errors)
        if path is None:
            continue
        try:
            rows = _read_csv(path)
        except Exception as exc:
            errors.append(f"failed to read {spec.relpath}: {exc}")
            continue
        datasets = {row.get("dataset", "") for row in rows}
        if datasets != DATASETS:
            errors.append(f"{spec.relpath}: datasets are {sorted(datasets)}, expected {sorted(DATASETS)}")
        for row in rows:
            context = f"{spec.relpath}:{row.get('dataset', '<missing dataset>')}"
            if row.get("method") != spec.method:
                errors.append(f"{context}: method is {row.get('method')!r}, expected {spec.method!r}")
            if spec.method_slug is not None and row.get("method_slug") != spec.method_slug:
                errors.append(
                    f"{context}: method_slug is {row.get('method_slug')!r}, expected {spec.method_slug!r}"
                )
            try:
                n_value = _int(row, spec.n_col, context)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if n_value != 5:
                errors.append(f"{context}: {spec.n_col} is {n_value}, expected 5")


def _check_main_table_tex(package_dir: Path, errors: list[str]) -> None:
    relpath = "tables/paper_tables_draft.tex"
    path = _require_file(package_dir, relpath, errors)
    if path is None:
        return
    text = path.read_text()
    label_pos = text.find(r"\label{tab:main-results}")
    if label_pos < 0:
        errors.append(f"{relpath}: missing tab:main-results label")
        return
    end_pos = text.find(r"\end{table*}", label_pos)
    if end_pos < 0:
        errors.append(f"{relpath}: malformed main-results table")
        return
    segment = text[label_pos:end_pos]
    for method in PRIMARY_METHODS:
        count = segment.count(f"& {method} &")
        if count != len(DATASETS):
            errors.append(
                f"{relpath}: method {method!r} appears {count} times in main table, "
                f"expected {len(DATASETS)}"
            )


def _check_admission_audit(package_dir: Path, errors: list[str]) -> None:
    relpath = "appendix/baseline_admission_audit_20260706.csv"
    path = _require_file(package_dir, relpath, errors)
    if path is None:
        return
    try:
        rows = {row.get("candidate", ""): row for row in _read_csv(path)}
    except Exception as exc:
        errors.append(f"failed to read {relpath}: {exc}")
        return

    expected = {
        "RAP softmax": {
            "expected_tier": "primary",
            "machine_status": "admitted_primary",
            "dataset_count": "4",
            "seed_count": "5",
            "gpu_complete": "True",
        },
        "RAP++ official ACS grid": {
            "expected_tier": "original_protocol",
            "machine_status": "admitted_original_protocol",
            "seed_count": "5",
        },
        "PrivMRF official TVD": {
            "expected_tier": "original_protocol",
            "machine_status": "admitted_original_protocol",
            "dataset_count": "4",
        },
    }
    for candidate, fields in expected.items():
        row = rows.get(candidate)
        if row is None:
            errors.append(f"{relpath}: missing candidate {candidate!r}")
            continue
        for key, value in fields.items():
            if row.get(key) != value:
                errors.append(
                    f"{relpath}:{candidate}: {key} is {row.get(key)!r}, expected {value!r}"
                )


def _check_original_protocol_audit(package_dir: Path, errors: list[str]) -> None:
    relpath = "appendix/original_protocol_baseline_audit_20260707.csv"
    path = _require_file(package_dir, relpath, errors)
    if path is None:
        return
    try:
        rows = {row.get("candidate", ""): row for row in _read_csv(path)}
    except Exception as exc:
        errors.append(f"failed to read {relpath}: {exc}")
        return

    expected = {
        "RAP++ official ACS grid": {
            "evidence_tier": "original_protocol",
            "audit_status": "passed",
            "row_count": "125",
            "dataset_count": "25",
            "seed_count": "5",
            "error_count": "0",
        },
        "PrivMRF official TVD": {
            "evidence_tier": "original_protocol",
            "audit_status": "passed",
            "row_count": "72",
            "dataset_count": "4",
            "seed_count": "1",
            "error_count": "0",
        },
    }
    for candidate, fields in expected.items():
        row = rows.get(candidate)
        if row is None:
            errors.append(f"{relpath}: missing candidate {candidate!r}")
            continue
        for key, value in fields.items():
            if row.get(key) != value:
                errors.append(
                    f"{relpath}:{candidate}: {key} is {row.get(key)!r}, expected {value!r}"
                )


def _sum_int(rows: list[dict[str, str]], col: str, relpath: str, errors: list[str]) -> int | None:
    total = 0
    for row in rows:
        try:
            total += _int(row, col, f"{relpath}:{row.get('dataset', '<missing dataset>')}")
        except ValueError as exc:
            errors.append(str(exc))
            return None
    return total


def _check_pairwise_win_counts(package_dir: Path, errors: list[str]) -> None:
    specs = [
        (
            "tables/sage_vs_private_pgm_aim_seed0to4_20260706.csv",
            "sage_wins",
            "aim_wins",
            20,
            0,
        ),
        (
            "tables/sage_vs_private_pgm_mst_seed0to4_20260706.csv",
            "sage_wins",
            "mst_wins",
            20,
            0,
        ),
        (
            "tables/sage_vs_rap_softmax_seed0to4_20260706.csv",
            "sage_metric_wins",
            "rap_metric_wins",
            20,
            0,
        ),
        (
            "tables/sage_vs_private_gsd_1m_fulln_seed0to4_20260706.csv",
            "sage_metric_wins",
            "gsd_metric_wins",
            16,
            4,
        ),
    ]
    for relpath, sage_col, other_col, expected_sage, expected_other in specs:
        path = _require_file(package_dir, relpath, errors)
        if path is None:
            continue
        try:
            rows = _read_csv(path)
        except Exception as exc:
            errors.append(f"failed to read {relpath}: {exc}")
            continue
        datasets = {row.get("dataset", "") for row in rows}
        if datasets != DATASETS:
            errors.append(f"{relpath}: datasets are {sorted(datasets)}, expected {sorted(DATASETS)}")
        sage_total = _sum_int(rows, sage_col, relpath, errors)
        other_total = _sum_int(rows, other_col, relpath, errors)
        if sage_total is not None and sage_total != expected_sage:
            errors.append(f"{relpath}: {sage_col} total is {sage_total}, expected {expected_sage}")
        if other_total is not None and other_total != expected_other:
            errors.append(f"{relpath}: {other_col} total is {other_total}, expected {expected_other}")


def _rows_by_dataset(package_dir: Path, relpath: str, errors: list[str]) -> dict[str, dict[str, str]] | None:
    path = _require_file(package_dir, relpath, errors)
    if path is None:
        return None
    try:
        rows = _read_csv(path)
    except Exception as exc:
        errors.append(f"failed to read {relpath}: {exc}")
        return None
    return {row.get("dataset", ""): row for row in rows}


def _ratio_col(spec: PairwiseValueSpec, metric: str) -> str:
    return spec.ratio_style.format(metric=metric, other=spec.other_prefix)


def _winner(sage_value: float, other_value: float, other_label: str) -> str:
    return "SAGE" if sage_value <= other_value else other_label


def _check_pairwise_values(package_dir: Path, errors: list[str]) -> None:
    sage_rows = _rows_by_dataset(package_dir, "tables/sage_all4_seed0to4_summary_rho1_20260706.csv", errors)
    if sage_rows is None:
        return
    for spec in PAIRWISE_VALUE_SPECS:
        pair_rows = _rows_by_dataset(package_dir, spec.relpath, errors)
        other_rows = _rows_by_dataset(package_dir, spec.summary_relpath, errors)
        if pair_rows is None or other_rows is None:
            continue
        if set(pair_rows) != DATASETS:
            errors.append(f"{spec.relpath}: datasets are {sorted(pair_rows)}, expected {sorted(DATASETS)}")
        for dataset in sorted(DATASETS):
            pair = pair_rows.get(dataset)
            sage = sage_rows.get(dataset)
            other = other_rows.get(dataset)
            if pair is None or sage is None or other is None:
                errors.append(f"{spec.relpath}:{dataset}: missing pairwise or summary row")
                continue
            context = f"{spec.relpath}:{dataset}"
            try:
                sage_n = _int(pair, "sage_n", context)
                other_n = _int(pair, spec.other_n_col, context)
                expected_sage_n = _int(sage, "n", f"tables/sage_all4_seed0to4_summary_rho1_20260706.csv:{dataset}")
                other_summary_n_col = "seed_count" if spec.other_prefix == "rap" else "n"
                expected_other_n = _int(other, other_summary_n_col, f"{spec.summary_relpath}:{dataset}")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if sage_n != expected_sage_n:
                errors.append(f"{context}: sage_n is {sage_n}, expected {expected_sage_n}")
            if other_n != expected_other_n:
                errors.append(f"{context}: {spec.other_n_col} is {other_n}, expected {expected_other_n}")
            for metric in METRICS:
                try:
                    expected_sage = _float(sage, METRIC_MEAN_COLS[metric], f"SAGE summary:{dataset}")
                    expected_other = _float(other, spec.other_summary_metric_cols[metric], f"{spec.summary_relpath}:{dataset}")
                    observed_sage = _float(pair, f"sage_{metric}", context)
                    observed_other = _float(pair, f"{spec.other_prefix}_{metric}", context)
                    observed_ratio = _float(pair, _ratio_col(spec, metric), context)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                expected_ratio = expected_other / expected_sage
                if not _close(observed_sage, expected_sage):
                    errors.append(
                        f"{context}: sage_{metric} is {observed_sage}, expected summary {expected_sage}"
                    )
                if not _close(observed_other, expected_other):
                    errors.append(
                        f"{context}: {spec.other_prefix}_{metric} is {observed_other}, "
                        f"expected summary {expected_other}"
                    )
                if not _close(observed_ratio, expected_ratio):
                    errors.append(
                        f"{context}: {_ratio_col(spec, metric)} is {observed_ratio}, "
                        f"expected {expected_ratio}"
                    )
                if spec.require_winner_cols:
                    expected_winner = _winner(expected_sage, expected_other, spec.other_label)
                    observed_winner = pair.get(f"winner_{metric}")
                    if observed_winner != expected_winner:
                        errors.append(
                            f"{context}: winner_{metric} is {observed_winner!r}, "
                            f"expected {expected_winner!r}"
                        )


def _check_claim_traceability(package_dir: Path, errors: list[str]) -> None:
    csv_relpath = "tables/paper_claim_traceability_20260707.csv"
    md_relpath = "tables/paper_claim_traceability_20260707.md"
    csv_path = _require_file(package_dir, csv_relpath, errors)
    _require_file(package_dir, md_relpath, errors)
    if csv_path is None:
        return
    try:
        rows = _read_csv(csv_path)
    except Exception as exc:
        errors.append(f"failed to read {csv_relpath}: {exc}")
        return
    observed_ids = {row.get("claim_id", "") for row in rows}
    if observed_ids != TRACEABILITY_IDS:
        errors.append(
            f"{csv_relpath}: claim ids are {sorted(observed_ids)}, "
            f"expected {sorted(TRACEABILITY_IDS)}"
        )
    for row in rows:
        if row.get("status") != "pass":
            errors.append(
                f"{csv_relpath}:{row.get('claim_id', '<missing claim_id>')}: "
                f"status is {row.get('status')!r}, expected 'pass'"
            )
        for key in ["evidence_artifacts", "verifier_or_gate", "machine_check"]:
            if not row.get(key):
                errors.append(
                    f"{csv_relpath}:{row.get('claim_id', '<missing claim_id>')}: "
                    f"missing {key}"
                )


def verify_claims(package_dir: Path) -> ClaimCheck:
    errors: list[str] = []
    if not package_dir.exists():
        return ClaimCheck(errors=[f"package directory is missing: {package_dir}"])
    if not package_dir.is_dir():
        return ClaimCheck(errors=[f"package path is not a directory: {package_dir}"])

    _check_summary_files(package_dir, errors)
    _check_main_table_tex(package_dir, errors)
    _check_admission_audit(package_dir, errors)
    _check_original_protocol_audit(package_dir, errors)
    _check_pairwise_win_counts(package_dir, errors)
    _check_pairwise_values(package_dir, errors)
    _check_claim_traceability(package_dir, errors)
    return ClaimCheck(errors=errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify paper-facing SAGE claim counts and baseline admission tiers."
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_claims(args.package_dir)
    print(f"paper claims package: {args.package_dir}")
    if result.ok:
        print("paper claim verification passed")
        return 0
    for error in result.errors:
        print(f"ERROR: {error}")
    print("paper claim verification failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
