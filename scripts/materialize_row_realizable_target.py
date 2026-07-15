#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import read_json, write_json
from qdte.queries.eval_jax import eval_records_queries
from qdte.queries.types import QueryCatalogue, filter_query_catalogue
from qdte.schema import TableSchema


def _parse_attrs(raw: str) -> list[int]:
    attrs = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not attrs:
        raise ValueError("--attrs must include at least one attribute id")
    return sorted(set(attrs))


def _resolve_measurement_path(path: Path) -> Path:
    return path / "measurements.json" if path.is_dir() else path


def _first_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {label}; checked: " + ", ".join(str(path) for path in candidates))


def _load_qcat(args: argparse.Namespace, artifact_dir: Path) -> tuple[QueryCatalogue, Path]:
    path = _first_existing(
        [
            args.queries if args.queries is not None else Path("__missing__"),
            artifact_dir / "queries.json",
            args.input_dir / "queries_full.json" if args.input_dir is not None else Path("__missing__"),
        ],
        "query catalogue",
    )
    return QueryCatalogue.from_dict(read_json(path)), path


def _load_schema(args: argparse.Namespace, artifact_dir: Path) -> tuple[TableSchema, Path]:
    path = _first_existing(
        [
            args.schema if args.schema is not None else Path("__missing__"),
            artifact_dir / "schema.json",
            args.input_dir / "schema.json" if args.input_dir is not None else Path("__missing__"),
        ],
        "schema",
    )
    return TableSchema.load_json(path), path


def _infer_total(args: argparse.Namespace, measurement_data: dict[str, Any]) -> tuple[int, str]:
    if args.total is not None:
        return int(args.total), "--total"
    if args.input_dir is not None and (args.input_dir / "real_encoded.npy").exists():
        return int(np.load(args.input_dir / "real_encoded.npy", mmap_mode="r").shape[0]), "input-dir real row count"
    consistency = dict(measurement_data.get("projection_diagnostics", {}).get("consistency", {}))
    if consistency.get("known_total_count") is not None:
        return int(consistency["known_total_count"]), "projection_diagnostics.consistency.known_total_count"
    projected = np.asarray(measurement_data["target_projected"], dtype=np.float64)
    for group in measurement_data.get("groups", []):
        if bool(group.get("is_partition", False)) and group.get("query_indices"):
            idx = np.asarray(group["query_indices"], dtype=np.int32)
            return int(round(float(np.sum(projected[idx])))), f"partition sum {group.get('name', '')}"
    raise ValueError("Unable to infer total row count; pass --total or --input-dir with real_encoded.npy")


def _query_scope(qcat: QueryCatalogue, qid: int) -> tuple[int, ...]:
    attrs = {int(term[0]) for term in qcat.query_terms(qid)}
    attrs.update(int(attr) for attr, _ in qcat.linear_terms(qid))
    return tuple(sorted(attrs))


def _domain_records(schema: TableSchema, attrs: list[int], max_cells: int) -> tuple[np.ndarray, int]:
    cards = schema.cardinalities
    cells = 1
    for attr in attrs:
        cells *= int(cards[int(attr)])
    if cells > int(max_cells):
        raise ValueError(f"Reduced domain has {cells} cells, exceeding --max-domain-cells={max_cells}")
    records = np.zeros((int(cells), schema.d), dtype=np.int32)
    grids = itertools.product(*(range(int(cards[int(attr)])) for attr in attrs))
    for row_idx, values in enumerate(grids):
        for attr, value in zip(attrs, values, strict=True):
            records[row_idx, int(attr)] = int(value)
    return records, int(cells)


def _contained_query_ids(qcat: QueryCatalogue, attrs: list[int], query_limit: int) -> list[int]:
    attr_set = set(int(attr) for attr in attrs)
    qids = [qid for qid in range(qcat.m) if set(_query_scope(qcat, qid)).issubset(attr_set)]
    if int(query_limit) > 0:
        qids = qids[: int(query_limit)]
    if not qids:
        raise ValueError(f"No queries are contained in attrs={attrs}")
    return qids


def _solve_fractional_projection(
    *,
    qcat: QueryCatalogue,
    schema: TableSchema,
    attrs: list[int],
    qids: list[int],
    source: np.ndarray,
    inv_variance: np.ndarray,
    total: int,
    max_domain_cells: int,
    max_iterations: int,
) -> dict[str, Any]:
    records, domain_cells = _domain_records(schema, attrs, max_domain_cells)
    keep = np.asarray(qids, dtype=np.int32)
    qsub = filter_query_catalogue(qcat, keep)
    phi = np.asarray(eval_records_queries(records, qsub), dtype=np.float64).T
    y = np.asarray(source, dtype=np.float64)[keep]
    inv = np.asarray(inv_variance, dtype=np.float64)[keep]
    n_atoms = int(phi.shape[1])
    x0 = np.full(n_atoms, float(total) / max(n_atoms, 1), dtype=np.float64)

    def objective(p: np.ndarray) -> float:
        residual = phi @ p - y
        return 0.5 * float(np.sum(residual * residual * inv))

    def gradient(p: np.ndarray) -> np.ndarray:
        residual = phi @ p - y
        return np.asarray(phi.T @ (residual * inv), dtype=np.float64)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=Bounds(np.zeros(n_atoms, dtype=np.float64), np.full(n_atoms, float(total), dtype=np.float64)),
        constraints=[LinearConstraint(np.ones((1, n_atoms), dtype=np.float64), [float(total)], [float(total)])],
        method="SLSQP",
        options={"ftol": 1.0e-9, "maxiter": int(max_iterations), "disp": False},
    )
    fitted = np.asarray(result.x, dtype=np.float64)
    answers = np.asarray(phi @ fitted, dtype=np.float64)
    return {
        "answers": answers,
        "weights": fitted,
        "objective": objective(fitted),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "domain_cells": int(domain_cells),
        "num_active_atoms": int(np.sum(fitted > 1.0e-8)),
        "min_atom_weight": float(np.min(fitted)) if fitted.size else 0.0,
        "max_atom_weight": float(np.max(fitted)) if fitted.size else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    measurement_path = _resolve_measurement_path(args.measurement)
    artifact_dir = measurement_path.parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_data = read_json(measurement_path)
    qcat, queries_path = _load_qcat(args, artifact_dir)
    schema, schema_path = _load_schema(args, artifact_dir)
    total, total_source = _infer_total(args, measurement_data)
    attrs = _parse_attrs(args.attrs)
    qids = _contained_query_ids(qcat, attrs, int(args.query_limit))

    target_projected = np.asarray(measurement_data["target_projected"], dtype=np.float64)
    target_noisy = np.asarray(measurement_data["target_noisy"], dtype=np.float64)
    variances = np.asarray(measurement_data["variances"], dtype=np.float64)
    inv_variance = 1.0 / np.maximum(variances, 1.0e-12)
    source = target_projected if args.source == "target_projected" else target_noisy

    solution = _solve_fractional_projection(
        qcat=qcat,
        schema=schema,
        attrs=attrs,
        qids=qids,
        source=source,
        inv_variance=inv_variance,
        total=total,
        max_domain_cells=int(args.max_domain_cells),
        max_iterations=int(args.max_iterations),
    )
    if bool(args.require_success) and not bool(solution["solver_success"]):
        raise RuntimeError(f"Row-realizable projection failed: {solution['solver_message']}")

    updated = target_projected.copy()
    keep = np.asarray(qids, dtype=np.int32)
    before = updated[keep].copy()
    updated[keep] = np.asarray(solution["answers"], dtype=np.float64)
    change = updated[keep] - before
    source_residual = before - source[keep]
    oracle_residual = updated[keep] - source[keep]
    source_objective_before = 0.5 * float(np.sum(source_residual * source_residual * inv_variance[keep]))
    source_objective_after = 0.5 * float(np.sum(oracle_residual * oracle_residual * inv_variance[keep]))

    diagnostics = dict(measurement_data.get("projection_diagnostics", {}))
    diagnostics["row_realizable_local"] = {
        "enabled": True,
        "method": "fractional_scope_oracle",
        "source": args.source,
        "scope_attrs": attrs,
        "scope_attr_names": [schema.columns[int(attr)].name for attr in attrs],
        "query_ids": qids,
        "num_queries": int(len(qids)),
        "domain_cells": int(solution["domain_cells"]),
        "known_total_count": int(total),
        "total_source": total_source,
        "objective_before": source_objective_before,
        "objective_after": source_objective_after,
        "source_to_previous_target_objective": source_objective_before,
        "source_to_oracle_target_objective": source_objective_after,
        "realizability_gap_objective": source_objective_after,
        "source_objective_increase": float(source_objective_after - source_objective_before),
        "target_update_l2": float(np.linalg.norm(change)),
        "target_update_linf": float(np.max(np.abs(change))) if change.size else 0.0,
        "solver_success": bool(solution["solver_success"]),
        "solver_message": str(solution["solver_message"]),
        "solver_iterations": int(solution["solver_iterations"]),
        "num_active_atoms": int(solution["num_active_atoms"]),
        "min_atom_weight": float(solution["min_atom_weight"]),
        "max_atom_weight": float(solution["max_atom_weight"]),
    }
    output_measurement = dict(measurement_data)
    output_measurement["target_projected"] = updated.astype(np.float32).tolist()
    output_measurement["projection_diagnostics"] = diagnostics

    write_json(output_measurement, output_dir / "measurements.json")
    qcat.save_json(output_dir / "queries.json")
    schema.save_json(output_dir / "schema.json")
    if "groups" in measurement_data:
        write_json(measurement_data["groups"], output_dir / "measurement_groups.json")
    metadata = {
        "variant": args.variant_name or output_dir.name,
        "measurement_path": str(measurement_path),
        "queries_path": str(queries_path),
        "schema_path": str(schema_path),
        "input_dir": str(args.input_dir) if args.input_dir is not None else "",
        "attrs": attrs,
        "query_ids": qids,
        "source": args.source,
        "total": int(total),
        "total_source": total_source,
        "runtime_seconds": float(time.perf_counter() - start),
        "offline_true_answers_used": False,
        "offline_true_answers_note": "No exact true answers are loaded by materialization.",
        "solver_success": bool(solution["solver_success"]),
        "solver_message": str(solution["solver_message"]),
        "command": " ".join(sys.argv),
    }
    write_json(metadata, output_dir / "row_realizable_materialization_metadata.json")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a reduced row-realizable target into a saved measurement artifact."
    )
    parser.add_argument("--measurement", required=True, type=Path, help="Path to measurements.json or a run directory.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attrs", required=True, help="Comma-separated reduced-domain attribute ids.")
    parser.add_argument("--source", choices=["target_projected", "target_noisy"], default="target_projected")
    parser.add_argument("--input-dir", type=Path, help="Canonical input dir for schema/queries and row count.")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--total", type=int)
    parser.add_argument("--query-limit", type=int, default=1024)
    parser.add_argument("--max-domain-cells", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--require-success", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(
        f"{metadata['variant']}: attrs={metadata['attrs']} queries={len(metadata['query_ids'])} "
        f"solver_success={metadata['solver_success']} runtime={metadata['runtime_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
