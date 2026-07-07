#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from path_defaults import baseline_root
except ModuleNotFoundError:
    from scripts.path_defaults import baseline_root


BASELINE_ROOT = baseline_root()
RESULTS_ROOT = BASELINE_ROOT / "external_results"
RUNS_ROOT = BASELINE_ROOT / "external_runs"
DEFAULT_OUTPUT_CSV = RESULTS_ROOT / "original_protocol_baseline_audit_20260707.csv"
DEFAULT_OUTPUT_MD = RESULTS_ROOT / "original_protocol_baseline_audit_20260707.md"

RAPPP_CSV = "rappp_official_paper_grid_seed0to4_20260707.csv"
PRIVMRF_CSV = "privmrf_official_full_tvd_epsgrid_m300_20260706.csv"

RAPPP_STATES = ("CA", "FL", "NY", "PA", "TX")
RAPPP_TARGETS = ("coverage", "employment", "income", "mobility", "travel")
SEEDS = (0, 1, 2, 3, 4)
PRIVMRF_DATASETS = ("acs", "adult", "br2000", "nltcs")
PRIVMRF_EPSILONS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
PRIVMRF_WAYS = (3, 4, 5)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _float_eq(value: Any, expected: float, *, tol: float = 1e-9) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=tol, abs_tol=tol)
    except Exception:
        return False


def _int_eq(value: Any, expected: int) -> bool:
    try:
        return int(float(value)) == expected
    except Exception:
        return False


def _is_completed(value: Any) -> bool:
    return str(value).lower() == "completed"


def _exists_from_row(row: dict[str, str], key: str) -> bool:
    value = row.get(key) or ""
    return bool(value) and Path(value).exists()


def audit_rappp(results_root: Path, runs_root: Path) -> list[str]:
    errors: list[str] = []
    rows = _read_csv(results_root / RAPPP_CSV)
    expected_rows = len(RAPPP_STATES) * len(RAPPP_TARGETS) * len(SEEDS)
    if len(rows) != expected_rows:
        errors.append(f"RAP++ official grid row count is {len(rows)}, expected {expected_rows}")

    state_counts = Counter(row.get("state") for row in rows)
    target_counts = Counter(row.get("target") for row in rows)
    seed_counts = Counter(int(float(row.get("seed", "nan"))) for row in rows if row.get("seed"))
    dataset_names = {row.get("dataset_name") for row in rows}

    expected_per_state = len(RAPPP_TARGETS) * len(SEEDS)
    expected_per_target = len(RAPPP_STATES) * len(SEEDS)
    expected_per_seed = len(RAPPP_STATES) * len(RAPPP_TARGETS)
    for state in RAPPP_STATES:
        if state_counts[state] != expected_per_state:
            errors.append(f"RAP++ state {state} has {state_counts[state]} rows, expected {expected_per_state}")
    for target in RAPPP_TARGETS:
        if target_counts[target] != expected_per_target:
            errors.append(f"RAP++ target {target} has {target_counts[target]} rows, expected {expected_per_target}")
    for seed in SEEDS:
        if seed_counts[seed] != expected_per_seed:
            errors.append(f"RAP++ seed {seed} has {seed_counts[seed]} rows, expected {expected_per_seed}")

    expected_datasets = {f"acs_{state}_{target}" for state in RAPPP_STATES for target in RAPPP_TARGETS}
    if dataset_names != expected_datasets:
        missing = sorted(expected_datasets - dataset_names)
        extra = sorted(dataset_names - expected_datasets)
        errors.append(f"RAP++ dataset_name set mismatch; missing={missing}, extra={extra}")

    for idx, row in enumerate(rows, start=2):
        prefix = f"RAP++ csv line {idx}"
        if not _is_completed(row.get("run_status")):
            errors.append(f"{prefix}: run_status is not completed")
        for key, expected in (
            ("upstream_epsilon", 1.0),
            ("num_random_projections", 200000),
            ("top_q", 5),
            ("dp_select_epochs", 50),
        ):
            if not _float_eq(row.get(key), float(expected)):
                errors.append(f"{prefix}: {key}={row.get(key)!r}, expected {expected}")
        for path_key in ("metrics_path", "run_dir"):
            if not _exists_from_row(row, path_key):
                errors.append(f"{prefix}: {path_key} does not exist: {row.get(path_key)!r}")

        seed = int(float(row.get("seed", "-1")))
        if seed in {1, 2, 3, 4}:
            if row.get("jax_device_platforms") != "gpu":
                errors.append(f"{prefix}: seed {seed} missing GPU platform evidence")
            if not str(row.get("jax_devices", "")).startswith("cuda"):
                errors.append(f"{prefix}: seed {seed} missing CUDA device evidence")

    for state in RAPPP_STATES:
        for target in RAPPP_TARGETS:
            for seed in SEEDS:
                metadata_path = (
                    runs_root
                    / "rappp_official_paper_grid"
                    / f"acs_{state}_{target}"
                    / "eps1p0"
                    / f"seed{seed}"
                    / "run_metadata.json"
                )
                if not metadata_path.is_file():
                    errors.append(f"RAP++ metadata missing: {metadata_path}")
                    continue
                metadata = _read_json(metadata_path)
                expected = {
                    "method": "rappp_official_paper_grid",
                    "state": state,
                    "target": target,
                    "seed": seed,
                    "upstream_epsilon": 1.0,
                    "k": 2,
                    "num_random_projections": 200000,
                    "top_q": 5,
                    "dp_select_epochs": 50,
                    "status": "completed",
                }
                for key, value in expected.items():
                    observed = metadata.get(key)
                    if isinstance(value, float):
                        ok = _float_eq(observed, value)
                    else:
                        ok = observed == value
                    if not ok:
                        errors.append(f"RAP++ metadata {metadata_path}: {key}={observed!r}, expected {value!r}")
                if seed in {1, 2, 3, 4}:
                    probe = metadata.get("jax_device_probe") or {}
                    platforms = [str(item).lower() for item in probe.get("jax_device_platforms", [])]
                    devices = [str(item).lower() for item in probe.get("jax_devices", [])]
                    if "gpu" not in platforms:
                        errors.append(f"RAP++ metadata {metadata_path}: missing GPU platform probe")
                    if not any(device.startswith("cuda") for device in devices):
                        errors.append(f"RAP++ metadata {metadata_path}: missing CUDA device probe")
    return errors


def audit_privmrf(results_root: Path, runs_root: Path) -> list[str]:
    errors: list[str] = []
    rows = _read_csv(results_root / PRIVMRF_CSV)
    expected_rows = len(PRIVMRF_DATASETS) * len(PRIVMRF_EPSILONS) * len(PRIVMRF_WAYS)
    if len(rows) != expected_rows:
        errors.append(f"PrivMRF official TVD row count is {len(rows)}, expected {expected_rows}")

    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_tuple: set[tuple[str, float, int]] = set()
    for row in rows:
        dataset = str(row.get("dataset"))
        by_dataset[dataset].append(row)
        try:
            by_tuple.add((dataset, float(row.get("epsilon", "nan")), int(float(row.get("way", "nan")))))
        except Exception:
            errors.append(f"PrivMRF row has invalid dataset/epsilon/way: {row}")

    expected_tuples = {
        (dataset, epsilon, way)
        for dataset in PRIVMRF_DATASETS
        for epsilon in PRIVMRF_EPSILONS
        for way in PRIVMRF_WAYS
    }
    if by_tuple != expected_tuples:
        missing = sorted(expected_tuples - by_tuple)
        extra = sorted(by_tuple - expected_tuples)
        errors.append(f"PrivMRF dataset/epsilon/way grid mismatch; missing={missing}, extra={extra}")

    for dataset in PRIVMRF_DATASETS:
        if len(by_dataset[dataset]) != len(PRIVMRF_EPSILONS) * len(PRIVMRF_WAYS):
            errors.append(
                f"PrivMRF dataset {dataset} has {len(by_dataset[dataset])} rows, "
                f"expected {len(PRIVMRF_EPSILONS) * len(PRIVMRF_WAYS)}"
            )

    for idx, row in enumerate(rows, start=2):
        prefix = f"PrivMRF csv line {idx}"
        if row.get("method") != "PrivMRF official":
            errors.append(f"{prefix}: method={row.get('method')!r}")
        if row.get("task") != "TVD":
            errors.append(f"{prefix}: task={row.get('task')!r}")
        if not _int_eq(row.get("repeat"), 1):
            errors.append(f"{prefix}: repeat={row.get('repeat')!r}, expected 1")
        if not _int_eq(row.get("marginal_num"), 300):
            errors.append(f"{prefix}: marginal_num={row.get('marginal_num')!r}, expected 300")
        if not _exists_from_row(row, "run_dir"):
            errors.append(f"{prefix}: run_dir does not exist: {row.get('run_dir')!r}")
        if not _exists_from_row(row, "result_path"):
            errors.append(f"{prefix}: result_path does not exist: {row.get('result_path')!r}")
        if not _float_eq(row.get("tvd_mean"), float(row.get("tvd_values", "nan")), tol=1e-10):
            errors.append(f"{prefix}: tvd_mean and tvd_values disagree")

    for run_name, datasets in (
        ("official_nltcs_tvd_epsgrid_m300_20260706", ("nltcs",)),
        ("official_acs_adult_br2000_tvd_epsgrid_m300_20260706", ("acs", "adult", "br2000")),
    ):
        metadata_path = runs_root / "privmrf_official" / run_name / "run_metadata.json"
        if not metadata_path.is_file():
            errors.append(f"PrivMRF metadata missing: {metadata_path}")
            continue
        metadata = _read_json(metadata_path)
        expected = {
            "method": "privmrf_official",
            "task": "TVD",
            "status": "completed",
            "repeat": 1,
            "marginal_num": 300,
            "conda_env": "baseline_privmrf",
        }
        for key, value in expected.items():
            observed = metadata.get(key)
            if observed != value:
                errors.append(f"PrivMRF metadata {metadata_path}: {key}={observed!r}, expected {value!r}")
        if tuple(metadata.get("datasets") or []) != tuple(datasets):
            errors.append(
                f"PrivMRF metadata {metadata_path}: datasets={metadata.get('datasets')!r}, "
                f"expected {list(datasets)!r}"
            )
        if tuple(float(x) for x in metadata.get("epsilons") or []) != PRIVMRF_EPSILONS:
            errors.append(f"PrivMRF metadata {metadata_path}: epsilon grid mismatch")
        result_path = Path(str(metadata.get("copied_result_path") or ""))
        if not result_path.is_file():
            errors.append(f"PrivMRF metadata {metadata_path}: copied_result_path missing")
    return errors


def audit(results_root: Path, runs_root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(audit_rappp(results_root, runs_root))
    errors.extend(audit_privmrf(results_root, runs_root))
    return errors


def _error_count(errors: list[str], prefix: str) -> int:
    return sum(1 for error in errors if error.startswith(prefix))


def summary_rows(results_root: Path, runs_root: Path, errors: list[str]) -> list[dict[str, str]]:
    rappp_rows = _read_csv(results_root / RAPPP_CSV)
    privmrf_rows = _read_csv(results_root / PRIVMRF_CSV)
    rappp_seed_gpu = {
        int(float(row["seed"]))
        for row in rappp_rows
        if row.get("jax_device_platforms") == "gpu" and str(row.get("jax_devices", "")).startswith("cuda")
    }
    privmrf_datasets = sorted({row.get("dataset", "") for row in privmrf_rows})
    privmrf_epsilons = sorted({float(row.get("epsilon", "nan")) for row in privmrf_rows})
    privmrf_ways = sorted({int(float(row.get("way", "nan"))) for row in privmrf_rows})

    return [
        {
            "candidate": "RAP++ official ACS grid",
            "evidence_tier": "original_protocol",
            "audit_status": "passed" if _error_count(errors, "RAP++") == 0 else "failed",
            "row_count": str(len(rappp_rows)),
            "dataset_count": str(len({row.get("dataset_name", "") for row in rappp_rows})),
            "state_count": str(len({row.get("state", "") for row in rappp_rows})),
            "target_count": str(len({row.get("target", "") for row in rappp_rows})),
            "seed_count": str(len({row.get("seed", "") for row in rappp_rows})),
            "gpu_evidence": f"seed_gpu_probe={','.join(str(seed) for seed in sorted(rappp_seed_gpu))}",
            "protocol": "epsilon=1.0;k=2;num_random_projections=200000;top_q=5;dp_select_epochs=50",
            "source_csv": str(results_root / RAPPP_CSV),
            "source_runs": str(runs_root / "rappp_official_paper_grid"),
            "error_count": str(_error_count(errors, "RAP++")),
        },
        {
            "candidate": "PrivMRF official TVD",
            "evidence_tier": "original_protocol",
            "audit_status": "passed" if _error_count(errors, "PrivMRF") == 0 else "failed",
            "row_count": str(len(privmrf_rows)),
            "dataset_count": str(len(privmrf_datasets)),
            "state_count": "",
            "target_count": "",
            "seed_count": "1",
            "gpu_evidence": "not_claimed",
            "protocol": (
                "datasets="
                + ",".join(privmrf_datasets)
                + ";epsilons="
                + ",".join(f"{epsilon:g}" for epsilon in privmrf_epsilons)
                + ";ways="
                + ",".join(str(way) for way in privmrf_ways)
                + ";repeat=1;marginal_num=300"
            ),
            "source_csv": str(results_root / PRIVMRF_CSV),
            "source_runs": str(runs_root / "privmrf_official"),
            "error_count": str(_error_count(errors, "PrivMRF")),
        },
    ]


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "candidate",
        "evidence_tier",
        "audit_status",
        "row_count",
        "dataset_count",
        "seed_count",
        "gpu_evidence",
        "protocol",
        "error_count",
    ]
    lines = [
        "# Original-Protocol Baseline Audit",
        "",
        "This audit verifies reproduced upstream-protocol evidence separately from the strict SAGE shared-evaluator table.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in cols) + " |")
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit original-protocol reproduced baseline evidence for RAP++ and PrivMRF."
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        errors = audit(args.results_root, args.runs_root)
        rows = summary_rows(args.results_root, args.runs_root, errors)
        write_csv(rows, args.output_csv)
        write_markdown(rows, args.output_md)
    except Exception as exc:
        print(f"original-protocol baseline audit failed before checks: {exc}")
        return 1
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    print(f"wrote markdown to {args.output_md}")
    if errors:
        print("original-protocol baseline audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("original-protocol baseline audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
