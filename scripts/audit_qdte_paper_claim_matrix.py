#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "docs" / "QDTE_PAPER_CLAIM_MATRIX_20260711.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "QDTE_PAPER_CLAIM_MATRIX_20260711.md"

EXPECTED_CLAIMS = {f"QDTE-C{index}" for index in range(1, 10)}
EXPECTED_TERMINOLOGY = {
    "QDTE-Standard": "end_to_end_dp_default",
    "QDTE-Structured": "controlled_generator_profile",
    "QDTE-FissionRefit": "qualified_l2_variant",
    "QDTE-RTP-local": "transfer_diagnostic",
    "Query-LSQ": "projection_diagnostic",
}
EXPECTED_FACTS = {
    "QDTE-C1": {
        "aim_wins": 20,
        "mst_wins": 20,
        "rap_wins": 20,
        "gsd_wins": 16,
        "cells": 20,
    },
    "QDTE-C4": {
        "datasets": 3,
        "target_loss_wins": 3,
        "offline_metric_wins": 15,
        "offline_metric_cells": 15,
    },
    "QDTE-C5": {
        "released_loss_wins": 4,
        "datasets": 4,
        "replacement_gate_passed": False,
    },
    "QDTE-C6": {
        "mae_wins": 4,
        "rmse_wins": 3,
        "combined_wins": 7,
        "combined_cells": 8,
        "max_l2_regression_percent": 0.62,
    },
}


def load_matrix(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("claim matrix root must be an object")
    return data


def audit_matrix(data: dict[str, Any], markdown_text: str) -> list[str]:
    errors: list[str] = []
    if data.get("paper_family") != "QDTE":
        errors.append("paper_family must be QDTE")
    terminology = data.get("terminology")
    if not isinstance(terminology, dict):
        errors.append("terminology must be an object")
        terminology = {}
    for name, role in EXPECTED_TERMINOLOGY.items():
        entry = terminology.get(name)
        if not isinstance(entry, dict):
            errors.append(f"missing terminology entry: {name}")
            continue
        if entry.get("role") != role:
            errors.append(
                f"terminology {name} role is {entry.get('role')!r}, expected {role!r}"
            )
        aliases = entry.get("aliases")
        if not isinstance(aliases, list) or not aliases:
            errors.append(f"terminology {name} must have at least one alias")

    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list):
        errors.append("claims must be a list")
        raw_claims = []
    claims: dict[str, dict[str, Any]] = {}
    for raw in raw_claims:
        if not isinstance(raw, dict):
            errors.append("each claim must be an object")
            continue
        claim_id = str(raw.get("id", ""))
        if claim_id in claims:
            errors.append(f"duplicate claim id: {claim_id}")
        claims[claim_id] = raw
    observed_ids = set(claims)
    if observed_ids != EXPECTED_CLAIMS:
        errors.append(
            f"claim ids are {sorted(observed_ids)}, expected {sorted(EXPECTED_CLAIMS)}"
        )
    for claim_id, claim in claims.items():
        if not claim.get("role"):
            errors.append(f"{claim_id}: missing role")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{claim_id}: evidence must be a non-empty list")
        forbidden = claim.get("forbidden")
        if not isinstance(forbidden, list) or not forbidden:
            errors.append(f"{claim_id}: forbidden must be a non-empty list")
        if claim_id not in markdown_text:
            errors.append(f"{claim_id}: missing from markdown companion")

    for claim_id, expected in EXPECTED_FACTS.items():
        observed = claims.get(claim_id, {}).get("required_facts")
        if observed != expected:
            errors.append(
                f"{claim_id}: required_facts are {observed!r}, expected {expected!r}"
            )

    for key in (
        "main_contributions",
        "main_empirical_claims",
        "appendix_or_limitations",
    ):
        values = data.get(key)
        if not isinstance(values, list) or not values:
            errors.append(f"{key} must be a non-empty list")
            continue
        unknown = set(values) - EXPECTED_CLAIMS
        if unknown:
            errors.append(f"{key} contains unknown claims: {sorted(unknown)}")

    excluded = data.get("excluded_from_main_stack")
    if not isinstance(excluded, list) or "SAGE-Select" not in excluded:
        errors.append("excluded_from_main_stack must include SAGE-Select")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the frozen QDTE paper claim matrix.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_matrix(args.matrix)
    markdown_text = args.markdown.read_text(encoding="utf-8")
    errors = audit_matrix(data, markdown_text)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"claim matrix: {args.matrix}")
    print("QDTE paper claim-matrix audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
