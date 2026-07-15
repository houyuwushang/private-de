#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import ensure_dir, write_json
from qdte.measurement.factorization import (
    allocate_strategy_rho,
    compile_hierarchical_pair_strategy,
)
from qdte.measurement.gcea import (
    GCEAOptimizationResult,
    GCEAProfile,
    build_gcea_profile,
    optimize_gcea_allocation,
)
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema
from scripts.run_static_ice_measurement_pilot import public_pair_scopes, rho_for_epsilon


PROTOCOL_ID = "SAGE-QDTE-ICE-WP10B-GCEA-20260715-v1"
CANDIDATE_ID = "SAGE-QDTE-Static-ICE-GCEA-v1"
PUBLIC_GATE_ID = "SAGE-QDTE-ICE-WP10B-GCEA-PUBLIC-GATE-20260715-v1"
PROTOCOL_PATH = ROOT / "docs" / "SAGE_QDTE_ICE_WP10B_GCEA_PROTOCOL_20260715.md"
DATASETS = ("adult", "br2000")
EPSILONS = (0.1, 0.3)
DELTA_DP = 1.0e-9
MAX_PAIR_CELLS = 20_000
BETA_TAIL = 0.05
SHARE_FLOOR = 1.0e-14
SHARE_FLOOR_INACTIVE_FACTOR = 1.0e6
OPTIMAL_FACE_TOLERANCE = 1.0e-10
SOLVER_FTOL = 1.0e-13
SOLVER_MAX_ITERATIONS = 5_000
T_STAR_MAX = 0.97
TAIL_RATIO_MAX = 1.0000000001
SUM_RHO_RELATIVE_ERROR_MAX = 1.0e-12
KKT_GAP_MAX = 1.0e-8
RECONSTRUCTION_RESIDUAL_MAX = 1.0e-10
CROSS_EPSILON_SHARE_MAX_ABS = 1.0e-10


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _external_input_root() -> Path:
    return ROOT.parent / "baseline" / "private-de" / "external_inputs"


def build_plan(input_root: Path) -> dict[str, Any]:
    sources = {
        "protocol": _file_record(PROTOCOL_PATH),
        "runner": _file_record(Path(__file__)),
        "gcea": _file_record(ROOT / "qdte" / "measurement" / "gcea.py"),
        "factorization": _file_record(ROOT / "qdte" / "measurement" / "factorization.py"),
        "workload_factorization": _file_record(
            ROOT / "qdte" / "measurement" / "workload_factorization.py"
        ),
    }
    public_inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        directory = input_root / f"{dataset}_sage_strong"
        public_inputs[dataset] = {
            name: _file_record(directory / name)
            for name in (
                "schema.json",
                "metadata.json",
                "queries_full.json",
                "workload_groups.json",
            )
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "public_gate_id": PUBLIC_GATE_ID,
        "artifact_role": "pre_generation_public_only_mechanism_gate",
        "datasets": list(DATASETS),
        "epsilons": list(EPSILONS),
        "delta": DELTA_DP,
        "adjacency": "add_remove",
        "max_pair_cells": MAX_PAIR_CELLS,
        "beta_tail": BETA_TAIL,
        "solver": {
            "method": "SLSQP",
            "share_floor": SHARE_FLOOR,
            "share_floor_inactive_factor": SHARE_FLOOR_INACTIVE_FACTOR,
            "optimal_face_tolerance": OPTIMAL_FACE_TOLERANCE,
            "ftol": SOLVER_FTOL,
            "max_iterations": SOLVER_MAX_ITERATIONS,
        },
        "thresholds": {
            "t_star_max": T_STAR_MAX,
            "tail_ratio_max": TAIL_RATIO_MAX,
            "sum_rho_relative_error_max": SUM_RHO_RELATIVE_ERROR_MAX,
            "kkt_gap_max": KKT_GAP_MAX,
            "reconstruction_residual_max": RECONSTRUCTION_RESIDUAL_MAX,
            "cross_epsilon_share_max_abs": CROSS_EPSILON_SHARE_MAX_ABS,
        },
        "sources": sources,
        "public_inputs": public_inputs,
        "private_table_read": False,
        "released_transcript_read": False,
        "true_utility_evaluated": False,
    }


def public_gate_checks(
    profile: GCEAProfile,
    result: GCEAOptimizationResult,
) -> dict[str, bool]:
    shares = np.asarray(
        [
            result.rho_by_block[name] / profile.rho_movable
            for name in profile.movable_block_names
        ],
        dtype=np.float64,
    )
    return {
        "stage_one_solver_success": bool(result.stage_one.get("success", False)),
        "stage_two_solver_success": bool(result.stage_two.get("success", False)),
        "t_star_at_most_0p97": result.t_star <= T_STAR_MAX,
        "stage_two_on_optimal_face": result.t_final
        <= result.t_star + 1.1 * OPTIMAL_FACE_TOLERANCE,
        "max_error_envelope_not_worse": result.tail_ratios["max_error"]
        <= TAIL_RATIO_MAX,
        "max_tvd_envelope_not_worse": result.tail_ratios["max_tvd"]
        <= TAIL_RATIO_MAX,
        "sum_rho_relative_error": result.sum_rho_relative_error
        <= SUM_RHO_RELATIVE_ERROR_MAX,
        "allocation_kkt_gap": result.kkt_gap <= KKT_GAP_MAX,
        "reconstruction_residual": profile.reconstruction_max_abs
        <= RECONSTRUCTION_RESIDUAL_MAX,
        "shares_finite_positive": bool(np.all(np.isfinite(shares)) and np.all(shares > 0.0)),
        "numerical_share_floor_inactive": result.min_movable_share
        > SHARE_FLOOR_INACTIVE_FACTOR * SHARE_FLOOR,
    }


def _allocation_shares(profile: GCEAProfile, result: GCEAOptimizationResult) -> np.ndarray:
    return np.asarray(
        [
            result.rho_by_block[name] / profile.rho_movable
            for name in profile.movable_block_names
        ],
        dtype=np.float64,
    )


def run_public_cell(
    input_dir: Path,
    *,
    dataset: str,
    epsilon: float,
) -> tuple[dict[str, Any], np.ndarray]:
    schema = TableSchema.load_json(input_dir / "schema.json")
    metadata = _read_json(input_dir / "metadata.json")
    expected_dataset = f"{dataset}_sage_strong"
    if not isinstance(metadata, dict) or str(metadata.get("dataset")) != expected_dataset:
        raise ValueError(f"Public metadata mismatch for {expected_dataset}")
    public_total = int(metadata.get("n_rows", 0))
    if public_total <= 0:
        raise ValueError("GCEA requires an explicitly public positive n_rows")
    qcat_payload = _read_json(input_dir / "queries_full.json")
    groups = _read_json(input_dir / "workload_groups.json")
    if not isinstance(qcat_payload, dict) or not isinstance(groups, list):
        raise ValueError("GCEA public query/group payloads have invalid structure")
    qcat = QueryCatalogue.from_dict(qcat_payload)
    pairs = public_pair_scopes(schema.cardinalities, MAX_PAIR_CELLS)
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, pairs)
    rho_total = rho_for_epsilon(float(epsilon), DELTA_DP)
    control = allocate_strategy_rho(strategy, rho_total, "public_optimal")
    profile = build_gcea_profile(
        strategy,
        qcat,
        groups,
        public_total=public_total,
        control_rho_by_block=control,
        beta_tail=BETA_TAIL,
    )
    result = optimize_gcea_allocation(
        profile,
        share_floor=SHARE_FLOOR,
        optimal_face_tolerance=OPTIMAL_FACE_TOLERANCE,
        ftol=SOLVER_FTOL,
        max_iterations=SOLVER_MAX_ITERATIONS,
    )
    checks = public_gate_checks(profile, result)
    shares = _allocation_shares(profile, result)
    payload = {
        "dataset": dataset,
        "declared_dataset": expected_dataset,
        "epsilon": float(epsilon),
        "delta": DELTA_DP,
        "public_total": public_total,
        "num_evaluator_queries": int(qcat.m),
        "num_reconstructable_queries": profile.num_queries,
        "num_unsupported_queries": len(profile.unsupported_query_ids),
        "num_complete_reconstructable_partitions": profile.num_partitions,
        "num_strategy_blocks": len(strategy.blocks),
        "num_movable_blocks": int(profile.movable_block_indices.size),
        "num_fixed_blocks": int(profile.fixed_block_indices.size),
        "num_pair_scopes": len(strategy.pairs),
        "reconstruction_max_abs": profile.reconstruction_max_abs,
        "optimization": result.to_dict(),
        "checks": checks,
        "passed": all(checks.values()),
        "private_table_read": False,
        "released_transcript_read": False,
        "true_utility_evaluated": False,
    }
    return payload, shares


def _write_summary_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    fields = (
        "dataset",
        "epsilon",
        "passed",
        "t_star",
        "t_final",
        "mae_ratio",
        "rmse_ratio",
        "avg_tvd_ratio",
        "max_error_ratio",
        "max_tvd_ratio",
        "kkt_gap",
        "reconstruction_max_abs",
        "min_movable_share",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            optimization = cell["optimization"]
            writer.writerow(
                {
                    "dataset": cell["dataset"],
                    "epsilon": cell["epsilon"],
                    "passed": cell["passed"],
                    "t_star": optimization["t_star"],
                    "t_final": optimization["t_final"],
                    "mae_ratio": optimization["primary_ratios"]["mae"],
                    "rmse_ratio": optimization["primary_ratios"]["rmse"],
                    "avg_tvd_ratio": optimization["primary_ratios"]["avg_tvd"],
                    "max_error_ratio": optimization["tail_ratios"]["max_error"],
                    "max_tvd_ratio": optimization["tail_ratios"]["max_tvd"],
                    "kkt_gap": optimization["kkt_gap"],
                    "reconstruction_max_abs": cell["reconstruction_max_abs"],
                    "min_movable_share": optimization["min_movable_share"],
                }
            )


def run_gate(input_root: Path, output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"WP10b public-gate output must be new or empty: {output}")
    ensure_dir(output)
    plan = build_plan(input_root.resolve())
    write_json(plan, output / "plan.json")

    cells: list[dict[str, Any]] = []
    shares_by_dataset: dict[str, list[np.ndarray]] = {dataset: [] for dataset in DATASETS}
    for dataset in DATASETS:
        input_dir = input_root.resolve() / f"{dataset}_sage_strong"
        for epsilon in EPSILONS:
            cell, shares = run_public_cell(
                input_dir,
                dataset=dataset,
                epsilon=epsilon,
            )
            cells.append(cell)
            shares_by_dataset[dataset].append(shares)

    cross_epsilon: dict[str, dict[str, Any]] = {}
    for dataset, allocations in shares_by_dataset.items():
        max_abs = float(np.max(np.abs(allocations[0] - allocations[1])))
        cross_epsilon[dataset] = {
            "normalized_share_max_abs": max_abs,
            "passed": max_abs <= CROSS_EPSILON_SHARE_MAX_ABS,
        }
    mechanism_passed = all(cell["passed"] for cell in cells) and all(
        item["passed"] for item in cross_epsilon.values()
    )
    decision = "authorize_one_time_end_to_end_panel" if mechanism_passed else "q1_a_freeze_ice_binary_profile"
    summary = {
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "public_gate_id": PUBLIC_GATE_ID,
        "artifact_role": "sealed_pre_generation_public_only_mechanism_gate",
        "cells": cells,
        "cross_epsilon": cross_epsilon,
        "mechanism_gate_passed": mechanism_passed,
        "generation_authorized": mechanism_passed,
        "decision": decision,
        "private_table_read": False,
        "released_transcript_read": False,
        "true_utility_evaluated": False,
    }
    write_json(summary, output / "public_gate.json")
    _write_summary_csv(output / "public_gate.csv", cells)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "public_gate_id": PUBLIC_GATE_ID,
        "artifact_role": "sealed_pre_generation_public_only_mechanism_gate",
        "artifacts": [
            _file_record(output / name)
            for name in ("plan.json", "public_gate.json", "public_gate.csv")
        ],
        "generation_authorized": mechanism_passed,
        "decision": decision,
        "private_table_read": False,
        "released_transcript_read": False,
        "true_utility_evaluated": False,
    }
    write_json(manifest, output / "sealed_manifest.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen public-only WP10b GCEA mechanism gate."
    )
    parser.add_argument("--input-root", type=Path, default=_external_input_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "static_ice_wp10b_gcea_public_gate_20260715",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_gate(args.input_root, args.output_dir)
    print(json.dumps(
        {
            "protocol_id": summary["protocol_id"],
            "mechanism_gate_passed": summary["mechanism_gate_passed"],
            "generation_authorized": summary["generation_authorized"],
            "decision": summary["decision"],
            "output_dir": str(args.output_dir.resolve()),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
