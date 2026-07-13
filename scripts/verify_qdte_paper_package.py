#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from path_defaults import baseline_root, external_results, repo_root
except ModuleNotFoundError:
    from scripts.path_defaults import baseline_root, external_results, repo_root


ROOT = repo_root()
BASELINE_ROOT = baseline_root()
DEFAULT_PACKAGE = external_results() / "qdte_paper_package_20260711"
REQUIRED_FILES = {
    "package_metadata.json",
    "source_manifest.json",
    "FILELIST.txt",
    "SHA256SUMS.txt",
    "claims/QDTE_PAPER_CLAIM_MATRIX_20260711.json",
    "claims/QDTE_PAPER_CLAIM_MATRIX_20260711.md",
    "tables/primary_dp_rows.csv",
    "tables/primary_dp_summary.csv",
    "tables/structured_vs_gsd_endpoint.csv",
    "tables/structured_vs_gsd_endpoint.json",
    "tables/structured_vs_gsd_time_curve.csv",
    "tables/structured_dp_gate.csv",
    "tables/structured_dp_gate_summary.json",
    "tables/fission_refit_l2.csv",
    "tables/fission_refit_l2_summary.json",
    "tables/transfer_gap_diagnostic.json",
    "tables/table_primary_dp_summary.tex",
    "tables/table_primary_dp_full.tex",
    "tables/table_primary_runtime.tex",
    "tables/table_structured_vs_gsd_endpoint.tex",
    "tables/table_structured_vs_gsd_time_to_quality.tex",
    "tables/table_structured_dp_gate.tex",
    "tables/table_fission_refit_l2.tex",
    "tables/table_transfer_gap_diagnostic.tex",
    "figures/qdte_structured_time_to_quality.pdf",
    "figures/qdte_structured_time_to_quality.png",
    "figures/qdte_transfer_gap_decomposition.pdf",
    "figures/qdte_transfer_gap_decomposition.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_source(path: str) -> Path:
    if path.startswith("$SAGE_BASELINE_ROOT/"):
        return BASELINE_ROOT / path.removeprefix("$SAGE_BASELINE_ROOT/")
    if path.startswith("$QDTE_REPO_ROOT/"):
        return ROOT / path.removeprefix("$QDTE_REPO_ROOT/")
    raise ValueError(f"source path is not parameterized: {path}")


def audit_file_manifests(package_dir: Path) -> list[str]:
    errors: list[str] = []
    filelist_path = package_dir / "FILELIST.txt"
    hashes_path = package_dir / "SHA256SUMS.txt"
    if not filelist_path.exists() or not hashes_path.exists():
        return ["missing FILELIST.txt or SHA256SUMS.txt"]
    listed = {
        line.strip()
        for line in filelist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    actual = {
        str(path.relative_to(package_dir))
        for path in package_dir.rglob("*")
        if path.is_file() and path.name not in {"FILELIST.txt", "SHA256SUMS.txt"}
    }
    if listed != actual:
        errors.append(
            f"FILELIST mismatch: missing={sorted(actual - listed)}, extra={sorted(listed - actual)}"
        )
    expected_hashes: dict[str, str] = {}
    for line in hashes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relpath = line.split("  ", 1)
        expected_hashes[relpath] = digest
    hash_targets = actual | {"FILELIST.txt"}
    if set(expected_hashes) != hash_targets:
        errors.append("SHA256SUMS coverage does not match package files")
    for relpath, expected in expected_hashes.items():
        path = package_dir / relpath
        if not path.exists():
            errors.append(f"hashed file is missing: {relpath}")
        elif _sha256(path) != expected:
            errors.append(f"hash mismatch: {relpath}")
    return errors


def audit_facts(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if metadata.get("status") != "complete":
        errors.append("package status must be complete")
    facts = metadata.get("facts")
    if not isinstance(facts, dict):
        return errors + ["package facts must be an object"]

    c1 = facts.get("QDTE-C1", {})
    expected_c1 = {
        "RAP softmax": 20,
        "Private-PGM AIM": 20,
        "Private-PGM MST": 20,
        "Private-GSD GPU 1M/full-N": 16,
    }
    for method, wins in expected_c1.items():
        observed = c1.get(method, {})
        if observed.get("qdte_wins") != wins or observed.get("cells") != 20:
            errors.append(f"QDTE-C1 mismatch for {method}: {observed}")

    c4 = facts.get("QDTE-C4", {})
    expected_c4 = {
        "datasets": 3,
        "target_loss_wins": 3,
        "offline_metric_wins": 15,
        "offline_metric_cells": 15,
    }
    if c4 != expected_c4:
        errors.append(f"QDTE-C4 mismatch: {c4}")

    c5 = facts.get("QDTE-C5", {})
    standard = c5.get("QDTE-Structured-v2", {})
    search_aware = c5.get("QDTE-Structured-SA", {})
    if standard.get("measured_loss_wins") != 4:
        errors.append(f"Structured-v2 measured-loss wins are not 4: {standard}")
    if standard.get("replacement_gate_passed") is not False:
        errors.append("Structured-v2 replacement gate must remain failed")
    if search_aware.get("measured_loss_wins") != 4:
        errors.append(f"Structured-SA measured-loss wins are not 4: {search_aware}")
    if search_aware.get("replacement_gate_passed") is not False:
        errors.append("Structured-SA replacement gate must remain failed")

    c6 = facts.get("QDTE-C6", {})
    for key, expected in (
        ("datasets", 4),
        ("mae_wins", 4),
        ("rmse_wins", 3),
        ("combined_wins", 7),
        ("combined_cells", 8),
    ):
        if c6.get(key) != expected:
            errors.append(f"QDTE-C6 {key} is {c6.get(key)!r}, expected {expected}")
    regression = c6.get("largest_l2_regression_percent")
    if regression is None or not math.isclose(float(regression), 0.6172295822633345):
        errors.append(f"QDTE-C6 largest L2 regression changed: {regression}")
    if c6.get("gate_passed") is not True:
        errors.append("QDTE-C6 gate must pass")

    c7 = facts.get("QDTE-C7", {})
    if not float(c7.get("target_change_percent", 1.0)) < 0.0:
        errors.append("QDTE-C7 target diagnostic must improve")
    if not float(c7.get("fit_change_percent", 1.0)) < 0.0:
        errors.append("QDTE-C7 fit diagnostic must improve")
    if not float(c7.get("final_change_percent", -1.0)) > 0.0:
        errors.append("QDTE-C7 final diagnostic must regress")
    if not float(c7.get("teacher_loss_ratio_vs_baseline", 1.0)) < 0.1:
        errors.append("QDTE-C7 teacher loss ratio must remain below 0.1")
    return errors


def audit_sources(package_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest = _read_json(package_dir / "source_manifest.json")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        return ["source manifest must contain a non-empty sources list"]
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            errors.append("source entry must be an object")
            continue
        raw_path = str(source.get("path", ""))
        if raw_path in seen:
            errors.append(f"duplicate source path: {raw_path}")
        seen.add(raw_path)
        try:
            path = _resolve_source(raw_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.exists():
            errors.append(f"source artifact missing: {raw_path}")
            continue
        if path.stat().st_size != int(source.get("size_bytes", -1)):
            errors.append(f"source size mismatch: {raw_path}")
        if _sha256(path) != source.get("sha256"):
            errors.append(f"source hash mismatch: {raw_path}")
        roles = source.get("roles")
        if not isinstance(roles, list) or not roles:
            errors.append(f"source has no roles: {raw_path}")
    return errors


def audit_no_private_paths(package_dir: Path) -> list[str]:
    errors: list[str] = []
    forbidden = ("/home/" + "qianqiu", str(ROOT), str(BASELINE_ROOT))
    for path in package_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".csv",
            ".json",
            ".md",
            ".tex",
            ".txt",
            ".yaml",
            ".yml",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in forbidden:
            if token and token in text:
                errors.append(
                    f"private/local path token {token!r} in {path.relative_to(package_dir)}"
                )
                break
    return errors


def verify_package(package_dir: Path, *, verify_sources: bool = True) -> list[str]:
    errors: list[str] = []
    for relpath in sorted(REQUIRED_FILES):
        if not (package_dir / relpath).exists():
            errors.append(f"missing required package file: {relpath}")
    if errors:
        return errors
    metadata = _read_json(package_dir / "package_metadata.json")
    errors.extend(audit_facts(metadata))
    errors.extend(audit_file_manifests(package_dir))
    errors.extend(audit_no_private_paths(package_dir))
    claim_path = package_dir / "claims/QDTE_PAPER_CLAIM_MATRIX_20260711.json"
    if _sha256(claim_path) != metadata.get("claim_matrix_sha256"):
        errors.append("packaged claim matrix hash does not match package metadata")
    if verify_sources:
        errors.extend(audit_sources(package_dir))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the QDTE paper evidence package.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--skip-source-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify_package(
        args.package_dir, verify_sources=not bool(args.skip_source_hashes)
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"QDTE paper package: {args.package_dir}")
    print("QDTE paper package verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
