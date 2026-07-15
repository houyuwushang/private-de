#!/usr/bin/env python3
from __future__ import annotations

import copy
import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import read_json, write_json
from qdte.measurement.consistency import (
    offline_query_space_projection_dominance_diagnostics,
    project_query_space_feasible_lsq,
    project_query_space_lsq,
    query_space_feasible_resource_diagnostics,
)
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import QueryCatalogue, filter_query_catalogue
from qdte.schema import TableSchema


PILOT_ID = "SAGE-QDTE-T4-FIXED-ARTIFACT-20260712-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def query_scope(qcat: QueryCatalogue, qid: int) -> tuple[int, ...]:
    attrs = {int(attr) for attr, *_ in qcat.query_terms(qid)}
    attrs.update(int(attr) for attr, _ in qcat.linear_terms(qid))
    return tuple(sorted(attrs))


def select_public_query_subset(
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    *,
    num_lowest_cardinality_attrs: int,
    max_scope_order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cards = np.asarray(cardinalities, dtype=np.int64)
    if num_lowest_cardinality_attrs <= 0 or num_lowest_cardinality_attrs > len(cards):
        raise ValueError("num_lowest_cardinality_attrs must be in [1, num_attributes]")
    if max_scope_order <= 0:
        raise ValueError("max_scope_order must be positive")
    ranked_attrs = sorted(range(len(cards)), key=lambda attr: (int(cards[attr]), int(attr)))
    selected_attrs = tuple(ranked_attrs[: int(num_lowest_cardinality_attrs)])
    selected_set = set(selected_attrs)
    keep = np.asarray(
        [
            qid
            for qid in range(qcat.m)
            if 0 < len(query_scope(qcat, qid)) <= int(max_scope_order)
            and set(query_scope(qcat, qid)).issubset(selected_set)
        ],
        dtype=np.int32,
    )
    if keep.size == 0:
        raise ValueError("public subset rule selected no queries")
    metadata = {
        "selection_rule": "lowest_cardinality_then_attribute_index",
        "selected_attrs": [int(attr) for attr in selected_attrs],
        "selected_cardinalities": [int(cards[attr]) for attr in selected_attrs],
        "max_scope_order": int(max_scope_order),
        "num_selected_queries": int(keep.size),
    }
    return keep, metadata


def run_pilot(
    *,
    measurement_path: Path,
    queries_path: Path,
    schema_path: Path,
    real_encoded_path: Path,
    output_dir: Path,
    total: int | None,
    num_lowest_cardinality_attrs: int,
    max_scope_order: int,
    max_constraints: int,
    max_dense_constraint_cells: int,
    solver_max_iterations: int,
    certificate_max_iterations: int,
    batch_size: int,
    require_qualified: bool,
    allow_resource_gate_failure: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Projection pilot output must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement = read_json(measurement_path)
    full_qcat = QueryCatalogue.from_dict(read_json(queries_path))
    schema = TableSchema.load_json(schema_path)
    keep, subset_metadata = select_public_query_subset(
        full_qcat,
        schema.cardinalities,
        num_lowest_cardinality_attrs=int(num_lowest_cardinality_attrs),
        max_scope_order=int(max_scope_order),
    )
    qcat = filter_query_catalogue(full_qcat, keep)
    noisy_full = np.asarray(measurement["target_noisy"], dtype=np.float64)
    variances_full = np.asarray(measurement["variances"], dtype=np.float64)
    if noisy_full.shape != (full_qcat.m,) or variances_full.shape != (full_qcat.m,):
        raise ValueError("measurement vector shapes do not match the query catalogue")
    noisy = noisy_full[keep]
    variances = variances_full[keep]
    if not np.all(np.isfinite(noisy)) or not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("selected noisy answers and variances must be finite with positive variance")

    public_total = int(measurement.get("num_rows") or total or 0)
    if public_total <= 0:
        raise ValueError("a positive public --total is required when the artifact has no num_rows")
    resource_diagnostics = query_space_feasible_resource_diagnostics(
        qcat,
        schema.cardinalities,
        public_total,
        max_constraints=int(max_constraints),
        max_dense_constraint_cells=int(max_dense_constraint_cells),
    )
    qcat.save_json(output_dir / "queries_subset.json")
    schema.save_json(output_dir / "schema.json")
    np.save(output_dir / "query_ids.npy", keep)
    np.save(output_dir / "target_noisy.npy", noisy)
    np.save(output_dir / "variances.npy", variances)
    write_json(resource_diagnostics, output_dir / "resource_diagnostics.json")
    if not bool(resource_diagnostics["resource_gate_passed"]):
        summary = {
            "pilot_id": PILOT_ID,
            "status": "resource_gate_failed",
            "dp_boundary": {
                "projection_inputs": "saved_noisy_measurement_public_metadata_variances",
                "true_answers_loaded": False,
                "true_metrics_used_for_selection_or_solver": False,
            },
            "source": {
                "measurement_path": str(measurement_path),
                "measurement_sha256": sha256_file(measurement_path),
                "queries_path": str(queries_path),
                "queries_sha256": sha256_file(queries_path),
                "schema_path": str(schema_path),
                "schema_sha256": sha256_file(schema_path),
            },
            "public_subset": subset_metadata,
            "total": public_total,
            "num_queries": int(qcat.m),
            "resource": resource_diagnostics,
        }
        write_json(summary, output_dir / "pilot_summary.json")
        if not allow_resource_gate_failure:
            raise RuntimeError(
                "P3 resource gate failed: "
                + ", ".join(resource_diagnostics["resource_gate_failures"])
            )
        return summary

    equality_start = time.perf_counter()
    equality = project_query_space_lsq(
        noisy,
        qcat,
        schema.cardinalities,
        total=public_total,
        variances=variances,
        max_constraints=int(max_constraints),
        solver_max_iterations=int(solver_max_iterations),
    )
    equality_seconds = time.perf_counter() - equality_start
    feasible_start = time.perf_counter()
    feasible = project_query_space_feasible_lsq(
        noisy,
        qcat,
        schema.cardinalities,
        total=public_total,
        variances=variances,
        max_constraints=int(max_constraints),
        solver_max_iterations=int(solver_max_iterations),
        max_dense_constraint_cells=int(max_dense_constraint_cells),
        certificate_max_iterations=int(certificate_max_iterations),
    )
    feasible_seconds = time.perf_counter() - feasible_start

    # The exact table is loaded only after both DP post-processing outputs are
    # fixed. It cannot affect subset or solver choice.
    real = np.load(real_encoded_path, mmap_mode="r")
    if real.ndim != 2 or real.shape[1] != schema.d:
        raise ValueError("real_encoded shape does not match the public schema")
    if public_total != int(real.shape[0]):
        raise ValueError("public row count and offline real table row count differ")
    truth = answer_queries(
        np.asarray(real, dtype=np.int32),
        qcat,
        batch_size=int(batch_size),
    ).astype(np.float64)
    dominance = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=truth,
        qcat=qcat,
        cardinalities=schema.cardinalities,
        total=public_total,
        variances=variances,
        equality_result=equality,
        feasible_result=feasible,
        max_constraints=int(max_constraints),
    )

    np.save(output_dir / "target_equality_float64.npy", equality.projected)
    np.save(output_dir / "target_nonnegative_float64.npy", feasible.projected)
    write_json(equality.diagnostics, output_dir / "equality_diagnostics.json")
    write_json(feasible.diagnostics, output_dir / "nonnegative_diagnostics.json")
    write_json(dominance, output_dir / "dominance_offline.json")
    projected_measurement: dict[str, Any] | None = None
    full_subset = bool(
        keep.size == full_qcat.m
        and np.array_equal(keep, np.arange(full_qcat.m, dtype=keep.dtype))
    )
    if full_subset:
        projected_dir = output_dir / "projected_measurement"
        projected_dir.mkdir(parents=True, exist_ok=False)
        qcat.save_json(projected_dir / "queries.json")
        schema.save_json(projected_dir / "schema.json")
        projected_payload = copy.deepcopy(measurement)
        projected_payload["target_projected"] = feasible.projected.tolist()
        projection_diagnostics = dict(projected_payload.get("projection_diagnostics", {}))
        projection_diagnostics["consistency"] = feasible.diagnostics
        projection_diagnostics["fixed_artifact_reprojection"] = {
            "pilot_id": PILOT_ID,
            "source_measurement_sha256": sha256_file(measurement_path),
            "uses_true_answers": False,
            "query_subset_is_full_catalogue": True,
        }
        projected_payload["projection_diagnostics"] = projection_diagnostics
        projected_path = projected_dir / "measurements.json"
        write_json(projected_payload, projected_path)
        projected_measurement = {
            "path": str(projected_path),
            "sha256": sha256_file(projected_path),
            "query_subset_is_full_catalogue": True,
        }
    summary = {
        "pilot_id": PILOT_ID,
        "status": "theorem_qualified" if dominance["theorem_qualified"] else "diagnostic_only",
        "dp_boundary": {
            "projection_inputs": "saved_noisy_measurement_public_metadata_variances",
            "true_answers": "offline_after_both_projections_fixed",
            "true_metrics_used_for_selection_or_solver": False,
        },
        "source": {
            "measurement_path": str(measurement_path),
            "measurement_sha256": sha256_file(measurement_path),
            "queries_path": str(queries_path),
            "queries_sha256": sha256_file(queries_path),
            "schema_path": str(schema_path),
            "schema_sha256": sha256_file(schema_path),
            "real_encoded_path": str(real_encoded_path),
            "real_encoded_role": "offline_truth_evaluation_only",
        },
        "public_subset": subset_metadata,
        "total": public_total,
        "num_queries": int(qcat.m),
        "runtime_seconds": {
            "equality": float(equality_seconds),
            "nonnegative": float(feasible_seconds),
        },
        "resource": resource_diagnostics,
        "fingerprints": {
            "query_vector": equality.diagnostics["query_vector_fingerprint"],
            "constraints": equality.diagnostics["constraint_fingerprint"],
            "precision": equality.diagnostics["precision_fingerprint"],
            "match": bool(dominance["fingerprints_match"]),
        },
        "solver": {
            "equality_certificate_passed": bool(
                equality.diagnostics["equality_certificate_passed"]
            ),
            "nonnegative_certificate_passed": bool(
                feasible.diagnostics["feasible_projection_certificate_passed"]
            ),
            "nonnegative_duality_gap": float(
                feasible.diagnostics["certificate_objective_suboptimality_upper_bound"]
            ),
            "nonnegative_final_max_constraint_violation": float(
                feasible.diagnostics["final_max_constraint_violation"]
            ),
        },
        "theorem": dominance,
        "projected_measurement": projected_measurement,
    }
    write_json(summary, output_dir / "pilot_summary.json")
    if require_qualified and not bool(dominance["theorem_qualified"]):
        raise RuntimeError(
            f"Projection pilot is not theorem-qualified: {dominance['failure_reasons']}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-artifact T4 projection pilot.")
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--real-encoded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int)
    parser.add_argument("--num-lowest-cardinality-attrs", type=int, default=4)
    parser.add_argument("--max-scope-order", type=int, default=2)
    parser.add_argument("--max-constraints", type=int, default=200000)
    parser.add_argument("--max-dense-constraint-cells", type=int, default=20000000)
    parser.add_argument("--solver-max-iterations", type=int, default=1000)
    parser.add_argument("--certificate-max-iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--allow-diagnostic-only", action="store_true")
    parser.add_argument("--allow-resource-gate-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_pilot(
        measurement_path=args.measurement,
        queries_path=args.queries,
        schema_path=args.schema,
        real_encoded_path=args.real_encoded,
        output_dir=args.output_dir,
        total=args.total,
        num_lowest_cardinality_attrs=int(args.num_lowest_cardinality_attrs),
        max_scope_order=int(args.max_scope_order),
        max_constraints=int(args.max_constraints),
        max_dense_constraint_cells=int(args.max_dense_constraint_cells),
        solver_max_iterations=int(args.solver_max_iterations),
        certificate_max_iterations=int(args.certificate_max_iterations),
        batch_size=int(args.batch_size),
        require_qualified=not bool(args.allow_diagnostic_only),
        allow_resource_gate_failure=bool(args.allow_resource_gate_failure),
    )
    print(f"pilot status: {summary['status']}")
    if summary["status"] == "resource_gate_failed":
        print(f"resource failures: {summary['resource']['resource_gate_failures']}")
        return 0
    theorem = summary["theorem"]
    print(
        "T4 errors: "
        f"equality={theorem['equality_to_truth_omega_squared']:.12g} "
        f"nonnegative={theorem['nonnegative_to_truth_omega_squared']:.12g} "
        f"margin={theorem['equality_to_nonnegative_omega_squared']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
