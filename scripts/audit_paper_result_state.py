#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

try:
    from path_defaults import external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, external_runs

RESULTS_ROOT = external_results()
RUNS_ROOT = external_runs()

DATASETS = ["adult_sage_strong", "acs_sage_strong", "br2000_sage_strong", "nltcs_sage_strong"]
DATASET_LABELS = {
    "adult_sage_strong": "Adult",
    "acs_sage_strong": "ACS",
    "br2000_sage_strong": "BR2000",
    "nltcs_sage_strong": "NLTCS",
}

SUMMARY_FILES = {
    "sage": "sage_all4_seed0to4_summary_rho1_20260706.csv",
    "aim": "private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv",
    "mst": "private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv",
    "rap": "rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
    "gsd_1m": "private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
}

METRICS = [
    ("MAE", ("MAE_mean", "full_true_mae_mean")),
    ("RMSE", ("RMSE_mean", "full_true_rmse_mean")),
    ("AvgTVD", ("AvgTVD_mean", "full_true_avg_tvd_mean")),
    ("MaxErr", ("MaxErr_mean", "full_true_max_error_mean")),
    ("MaxTVD", ("MaxTVD_mean", "full_true_max_tvd_mean")),
]

EXPECTED_SAGE_WIN_COUNTS = {
    "aim": 20,
    "mst": 20,
    "rap": 20,
    "gsd_1m": 16,
}

EXPECTED_GSD_WINS = {
    ("acs_sage_strong", "AvgTVD"),
    ("acs_sage_strong", "MaxTVD"),
    ("br2000_sage_strong", "AvgTVD"),
    ("br2000_sage_strong", "MaxTVD"),
}

REQUIRED_SAGE_CONFIG = {
    "measurement_mode": "static_all",
    "objective_weighting": "variance",
    "score_backend": "dense_gpu",
    "transport_mode": "atom_flow",
    "max_iters": "5000",
    "project_partitions": "true",
    "clip_nonpartition": "true",
    "prefix_monotonicity": "true",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def rows_by_dataset(path: Path) -> dict[str, dict[str, str]]:
    return {row["dataset"]: row for row in read_csv(path)}


def metric(row: dict[str, str], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in row and row[key] != "":
            return float(row[key])
    raise KeyError(f"none of the metric columns exist: {keys}")


def format_ratio(value: float) -> str:
    return f"{value:.2f}"


def _load_summaries(results_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    return {name: rows_by_dataset(results_dir / file_name) for name, file_name in SUMMARY_FILES.items()}


def audit_result_state(results_dir: Path, runs_root: Path) -> list[str]:
    errors: list[str] = []
    try:
        summaries = _load_summaries(results_dir)
    except Exception as exc:
        return [f"failed to load summary files from {results_dir}: {exc}"]

    for name, rows in summaries.items():
        observed = set(rows)
        expected = set(DATASETS)
        if observed != expected:
            errors.append(f"{name}: datasets are {sorted(observed)}, expected {sorted(expected)}")

    for baseline in ["aim", "mst", "rap", "gsd_1m"]:
        wins = 0
        gsd_wins: set[tuple[str, str]] = set()
        for dataset in DATASETS:
            if dataset not in summaries["sage"] or dataset not in summaries[baseline]:
                continue
            sage = summaries["sage"][dataset]
            base = summaries[baseline][dataset]
            for label, key in METRICS:
                try:
                    ratio = metric(base, key) / metric(sage, key)
                except Exception as exc:
                    errors.append(f"{baseline}:{dataset}:{label}: failed to compute ratio: {exc}")
                    continue
                if ratio > 1.0:
                    wins += 1
                elif baseline == "gsd_1m":
                    gsd_wins.add((dataset, label))
                else:
                    errors.append(f"{baseline}:{dataset}:{label}: baseline/SAGE ratio {ratio:.6g} does not favor SAGE")
        expected_wins = EXPECTED_SAGE_WIN_COUNTS[baseline]
        if wins != expected_wins:
            errors.append(f"{baseline}: SAGE wins {wins}/20, expected {expected_wins}/20")
        if baseline == "gsd_1m" and gsd_wins != EXPECTED_GSD_WINS:
            errors.append(
                "gsd_1m: GSD-winning cells are "
                f"{sorted(gsd_wins)}, expected {sorted(EXPECTED_GSD_WINS)}"
            )

    gsd_1m_dir = runs_root / "private_gsd_gpu_1m_fulln_audit/adult_sage_strong/rho1p0/seed0"
    metadata_path = gsd_1m_dir / "run_metadata.json"
    if not metadata_path.exists():
        errors.append(f"missing GSD 1M metadata: {metadata_path}")
    else:
        try:
            metadata = read_json(metadata_path)
        except Exception as exc:
            errors.append(f"failed to read GSD 1M metadata {metadata_path}: {exc}")
        else:
            notes = metadata.get("notes") or {}
            required = {
                "method": ("top", "private_gsd_gpu_1m_fulln_audit"),
                "jax_backend": ("notes", "gpu"),
                "num_generations": ("notes", 1000000),
                "stop_early_min_generation": ("notes", 1000000),
                "tree_query_depth": ("notes", 2),
            }
            for key, (where, expected) in required.items():
                observed = metadata.get(key) if where == "top" else notes.get(key)
                if observed != expected:
                    errors.append(f"GSD 1M metadata {key} is {observed!r}, expected {expected!r}")
            devices = notes.get("jax_devices")
            if not devices or "gpu" not in str(devices).lower():
                errors.append(f"GSD 1M metadata jax_devices lacks GPU evidence: {devices!r}")

    config_path = runs_root / "sage/adult_sage_strong/rho1p0/seed0/config_resolved.yaml"
    if not config_path.exists():
        errors.append(f"missing SAGE resolved config: {config_path}")
    else:
        for key, expected in REQUIRED_SAGE_CONFIG.items():
            observed = grep_yaml_value(config_path, key)
            if observed != expected:
                errors.append(f"SAGE config {key} is {observed!r}, expected {expected!r}")
        consistency = grep_consistency_enabled(config_path)
        if consistency != "false":
            errors.append(f"SAGE config consistency.enabled is {consistency!r}, expected 'false'")

    return errors


def print_ratio_block(results_dir: Path) -> None:
    summaries = _load_summaries(results_dir)
    print("## Current seed0--4 baseline/SAGE ratios")
    print()
    print("Values are baseline error divided by SAGE error. Values above 1 mean SAGE is lower.")
    for baseline in ["aim", "mst", "rap", "gsd_1m"]:
        print()
        print(f"### {baseline}")
        wins = 0
        total = 0
        for dataset in DATASETS:
            sage = summaries["sage"][dataset]
            base = summaries[baseline][dataset]
            ratios = []
            for label, key in METRICS:
                ratio = metric(base, key) / metric(sage, key)
                wins += int(ratio > 1.0)
                total += 1
                ratios.append(f"{label}={format_ratio(ratio)}")
            print(f"- {DATASET_LABELS[dataset]}: " + ", ".join(ratios))
        print(f"- SAGE wins: {wins}/{total}")


def print_gsd_shift(results_dir: Path) -> None:
    early_path = results_dir / "private_gsd_gpu_multiseed_all4_20260706.csv"
    one_m = rows_by_dataset(results_dir / SUMMARY_FILES["gsd_1m"])
    early_rows = {
        row["dataset"]: row
        for row in read_csv(early_path)
        if row.get("method_slug") == "private_gsd_gpu"
    }
    print()
    print("## Private-GSD configuration shift")
    print()
    print("Values are 1M/full-N GSD error divided by earlier 200k-GPU GSD error. Values below 1 mean 1M/full-N is better.")
    for dataset in DATASETS:
        ratios = []
        for label, key in METRICS:
            ratio = metric(one_m[dataset], key) / metric(early_rows[dataset], key)
            ratios.append(f"{label}={format_ratio(ratio)}")
        print(f"- {DATASET_LABELS[dataset]}: " + ", ".join(ratios))


def first_matching_row(rows: Iterable[dict[str, str]], **predicates: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in predicates.items()):
            return row
    return None


def print_adult_sage_stability(results_dir: Path) -> None:
    old_rows = read_csv(results_dir / "adult_sage_strong_full_start_20260705.csv")
    current_rows = read_csv(results_dir / "adult_sage_strong_rho1_multiseed_20260706.csv")
    old = first_matching_row(old_rows, method="sage", seed="0")
    current = first_matching_row(current_rows, method="sage", seed="0")
    if old is None or current is None:
        return
    print()
    print("## Adult SAGE seed0 stability")
    print()
    for label, key in [
        ("MAE", "full_true_mae"),
        ("RMSE", "full_true_rmse"),
        ("AvgTVD", "full_true_avg_tvd"),
        ("MaxErr", "full_true_max_error"),
        ("MaxTVD", "full_true_max_tvd"),
        ("runtime", "runtime_seconds"),
    ]:
        ratio = float(current[key]) / float(old[key])
        print(f"- {label}: current/old = {ratio:.4f}")


def grep_yaml_value(path: Path, needle: str) -> str | None:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{needle}:"):
            return stripped.split(":", 1)[1].strip()
    return None


def grep_consistency_enabled(path: Path) -> str | None:
    in_consistency = False
    consistency_indent = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped == "consistency:":
            in_consistency = True
            consistency_indent = indent
            continue
        if in_consistency and indent <= consistency_indent:
            in_consistency = False
        if in_consistency and stripped.startswith("enabled:"):
            return stripped.split(":", 1)[1].strip()
    return None


def print_key_metadata(runs_root: Path) -> None:
    print()
    print("## Key metadata")
    sage_dir = runs_root / "sage/adult_sage_strong/rho1p0/seed0"
    gsd_1m_dir = runs_root / "private_gsd_gpu_1m_fulln_audit/adult_sage_strong/rho1p0/seed0"
    gsd_200k_dir = runs_root / "private_gsd_gpu/adult_sage_strong/rho1p0/seed0"

    for label, run_dir in [
        ("SAGE Adult seed0", sage_dir),
        ("GSD 1M/full-N Adult seed0", gsd_1m_dir),
        ("GSD 200k Adult seed0", gsd_200k_dir),
    ]:
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            print(f"- {label}: missing {metadata_path}")
            continue
        metadata = read_json(metadata_path)
        notes = metadata.get("notes") or {}
        print(f"- {label}:")
        print(f"  - method: {metadata.get('method')}")
        print(f"  - runtime_seconds: {metadata.get('runtime_seconds')}")
        print(f"  - queries_hash: {metadata.get('queries_hash')}")
        print(f"  - schema_hash: {metadata.get('schema_hash')}")
        for key in ["jax_backend", "jax_devices", "n_prime", "num_generations", "stop_early_min_generation", "tree_query_depth"]:
            if key in notes:
                print(f"  - {key}: {notes[key]}")

    config_path = sage_dir / "config_resolved.yaml"
    if config_path.exists():
        print("- SAGE config_resolved.yaml key fields:")
        for key in [
            "measurement_mode",
            "objective_weighting",
            "score_backend",
            "transport_mode",
            "max_iters",
            "project_partitions",
            "clip_nonpartition",
            "prefix_monotonicity",
        ]:
            value = grep_yaml_value(config_path, key)
            if value is not None:
                print(f"  - {key}: {value}")
        value = grep_consistency_enabled(config_path)
        if value is not None:
            print(f"  - consistency.enabled: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the current SAGE paper-result state without running experiments.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("# Paper Result State Audit")
    print()
    print(f"- results_dir: `{args.results_dir}`")
    print(f"- runs_root: `{args.runs_root}`")
    print_ratio_block(args.results_dir)
    print_gsd_shift(args.results_dir)
    print_adult_sage_stability(args.results_dir)
    print_key_metadata(args.runs_root)
    errors = audit_result_state(args.results_dir, args.runs_root)
    print()
    if not errors:
        print("paper result-state audit passed")
        return 0
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print("paper result-state audit failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
