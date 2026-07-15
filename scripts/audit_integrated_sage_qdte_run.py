#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_ID = "SAGE-QDTE-FULL-20260713-v4-development"
LEGACY_PROTOCOL_IDS = {
    "SAGE-QDTE-FULL-20260712-v1",
    "SAGE-QDTE-FULL-20260712-v2",
    "SAGE-QDTE-FULL-20260713-v3",
}
PRIVATE_EM_PROTOCOL_IDS = {PROTOCOL_ID, "SAGE-QDTE-FULL-20260713-v3"}
REQUIRED_ARTIFACTS_V1 = {
    "protocol",
    "resolved_config",
    "setup",
    "schema",
    "query_catalogue",
    "initial_synthetic",
    "final_measurement",
    "adaptive_summary",
    "adaptive_timeseries",
    "final_synthetic",
    "final_generator_config",
    "final_runtime",
    "final_metrics",
    "final_logs",
}
REQUIRED_ARTIFACTS = REQUIRED_ARTIFACTS_V1 | {"coverage_measurement"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("integrated manifest root must be an object")
    return data


def audit_manifest(data: dict[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    protocol_id = data.get("protocol_id")
    if protocol_id not in {PROTOCOL_ID, *LEGACY_PROTOCOL_IDS}:
        errors.append(f"protocol_id must be {PROTOCOL_ID} or a registered legacy protocol")
    mode = data.get("protocol_mode")
    if mode not in {"final", "smoke"}:
        errors.append("protocol_mode must be final or smoke")

    privacy = data.get("privacy")
    if not isinstance(privacy, dict):
        errors.append("privacy must be an object")
        privacy = {}
    if privacy.get("mode") != "dp":
        errors.append("privacy.mode must be dp")
    if privacy.get("adjacency") != "add_remove_one":
        errors.append("privacy.adjacency must be add_remove_one")
    selection_rho_per_round = float(
        privacy.get("selection_rho_per_round", float("nan"))
    )
    if protocol_id in PRIVATE_EM_PROTOCOL_IDS:
        if not math.isfinite(selection_rho_per_round) or selection_rho_per_round <= 0.0:
            errors.append("private EM selection must spend positive selection rho")
    elif not math.isclose(
        selection_rho_per_round,
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        errors.append("legacy transcript-only selection must spend zero selection rho")
    try:
        rho_total = float(privacy["rho_total"])
        delta = float(privacy["delta"])
        expected_epsilon = rho_total + 2.0 * math.sqrt(rho_total * math.log(1.0 / delta))
        if not math.isclose(
            float(privacy.get("epsilon_recomputed", float("nan"))),
            expected_epsilon,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            errors.append("epsilon_recomputed does not match the zCDP conversion")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("privacy ledger is incomplete or invalid")

    method = data.get("method")
    if not isinstance(method, dict):
        errors.append("method must be an object")
        method = {}
    if protocol_id in PRIVATE_EM_PROTOCOL_IDS:
        expected_method_fields = {
            "label": "SAGE-QDTE",
            "scheme": "voi_sageordergain_harmonic_qproject_repeat",
            "selection_input": "private_true_answers",
            "selection_mechanism": "exponential",
            "selection_score_sensitivity": 1.0,
            "selection_ledger": "bounded_range_em_zcdp",
            "selection_rule": "sample",
            "public_bootstrap_rounds": 0,
            "generator_profile": "qdte_standard",
            "initialization": "dp_oneway_independent",
            "generator_seed_mode": "per_stage",
            "workload": "complete_orthogonal_oneway_twoway",
        }
    else:
        expected_method_fields = {
            "label": "SAGE-QDTE",
            "scheme": "voi_sageordergain_harmonic_qproject",
            "selection_input": "transcript",
            "selection_ledger": "measurement_only",
            "selection_rule": "argmax",
            "public_bootstrap_rounds": 3,
            "generator_profile": "qdte_standard",
            "initialization": "public_uniform_independent",
        }
    for key, expected in expected_method_fields.items():
        if method.get(key) != expected:
            errors.append(f"method.{key} is {method.get(key)!r}, expected {expected!r}")
    if protocol_id in PRIVATE_EM_PROTOCOL_IDS:
        expected_coverage = 0.1 if protocol_id == PROTOCOL_ID else 0.5
        if method.get("coverage_rho_fraction") != expected_coverage:
            errors.append(
                f"method.coverage_rho_fraction must be {expected_coverage} for {protocol_id}"
            )
        if method.get("adaptive_rho_fraction") != 1.0 - expected_coverage:
            errors.append(
                f"method.adaptive_rho_fraction must be {1.0 - expected_coverage} for {protocol_id}"
            )
        if method.get("adaptive_measurement_fraction") != 0.9:
            errors.append("private EM method.adaptive_measurement_fraction must be 0.9")
    if protocol_id == PROTOCOL_ID and method.get("coverage_mode") != "oneway_only":
        errors.append("v4 development method.coverage_mode must be oneway_only")

    checks = data.get("checks")
    if not isinstance(checks, dict) or not checks:
        errors.append("checks must be a non-empty object")
        checks = {}
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    if failed_checks:
        errors.append(f"integrated run checks failed: {failed_checks}")

    qualified = data.get("paper_evidence_qualified")
    if mode == "final":
        if (
            method.get("rounds"),
            method.get("inner_iters"),
            method.get("final_refit_iters"),
        ) != (50, 5000, 5000):
            errors.append("final mode must use rounds/inner/final-refit = 50/5000/5000")
        if protocol_id in PRIVATE_EM_PROTOCOL_IDS and method.get("initial_fit_iters") != 5000:
            errors.append("private EM final mode must use initial_fit_iters=5000")
        if method.get("projection_profile") != "P1_lightweight":
            errors.append("final protocol must use P1_lightweight until P3 is separately promoted")
        if protocol_id == PROTOCOL_ID and qualified is not False:
            errors.append("v4 development runs cannot be paper_evidence_qualified")
        elif protocol_id in LEGACY_PROTOCOL_IDS and qualified is True:
            errors.append("superseded legacy evidence cannot remain qualified")
    elif mode == "smoke" and qualified is not False:
        errors.append("smoke manifests cannot be paper_evidence_qualified")

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
        artifacts = {}
    expected_artifacts = (
        REQUIRED_ARTIFACTS
        if protocol_id in {PROTOCOL_ID, "SAGE-QDTE-FULL-20260712-v2", "SAGE-QDTE-FULL-20260713-v3"}
        else REQUIRED_ARTIFACTS_V1
    )
    if set(artifacts) != expected_artifacts:
        errors.append(f"artifact keys must be exactly {sorted(expected_artifacts)}")
    if verify_files:
        for name, raw in artifacts.items():
            if not isinstance(raw, dict):
                errors.append(f"artifact {name} must be an object")
                continue
            path = Path(str(raw.get("path", "")))
            if not path.is_file():
                errors.append(f"artifact {name} is missing: {path}")
                continue
            observed_size = int(path.stat().st_size)
            if raw.get("bytes") != observed_size:
                errors.append(f"artifact {name} byte size changed")
            observed_hash = sha256_file(path)
            if raw.get("sha256") != observed_hash:
                errors.append(f"artifact {name} SHA-256 changed")

    forbidden_manifest_keys = {
        "true_answers",
        "true_answer_vector",
        "promotion_true_utility",
        "selection_true_answers",
    }
    observed_forbidden = forbidden_manifest_keys & set(data)
    if observed_forbidden:
        errors.append(f"manifest contains forbidden active/private keys: {sorted(observed_forbidden)}")
    if not isinstance(data.get("source_tree_sha256"), str) or len(data["source_tree_sha256"]) != 64:
        errors.append("source_tree_sha256 must be present")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an integrated SAGE-QDTE run manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-file-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = audit_manifest(
        load_manifest(args.manifest),
        verify_files=not bool(args.skip_file_hashes),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"manifest: {args.manifest}")
    print("integrated SAGE-QDTE run audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
