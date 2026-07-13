#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import read_json, write_json
from qdte.queries.eval_jax import answer_queries, eval_records_queries
from qdte.queries.types import QueryCatalogue, filter_query_catalogue
from qdte.schema import TableSchema


SUMMARY_COLUMNS = [
    "dataset",
    "seed",
    "synthetic_variant",
    "target_variant",
    "weight_variant",
    "synthetic_run_dir",
    "measurement_dir",
    "num_queries",
    "n_true",
    "n_synthetic",
    "T_target_to_true_norm2",
    "F_synthetic_to_target_norm2",
    "C_cross_term",
    "E_synthetic_to_true_norm2",
    "decomposition_residual",
    "target_to_true_weighted_l2",
    "synthetic_to_target_weighted_l2",
    "synthetic_to_true_weighted_l2",
    "synthetic_to_target_measured_loss",
]

BREAKDOWN_COLUMNS = [
    "dataset",
    "seed",
    "synthetic_variant",
    "target_variant",
    "weight_variant",
    "grouping",
    "group_key",
    "num_queries",
    "T_target_to_true_norm2",
    "F_synthetic_to_target_norm2",
    "C_cross_term",
    "E_synthetic_to_true_norm2",
    "decomposition_residual",
    "T_per_query",
    "F_per_query",
    "C_per_query",
    "E_per_query",
]

ORACLE_COLUMNS = [
    "dataset",
    "seed",
    "target_variant",
    "oracle_source",
    "status",
    "solver_success",
    "solver_message",
    "solver_iterations",
    "num_oracle_attrs",
    "oracle_attrs",
    "oracle_domain_cells",
    "num_oracle_queries",
    "oracle_query_limit",
    "source_to_true_norm2",
    "oracle_to_true_norm2",
    "source_to_oracle_norm2",
    "source_to_oracle_objective",
    "contraction_slack_norm2",
    "min_atom_weight",
    "max_atom_weight",
    "num_active_atoms",
]


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must include at least one seed")
    return seeds


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return values if values else None


def _first_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {label}; checked: " + ", ".join(str(path) for path in candidates))


def _variant_run_dir(root: Path, seed: int, variant: str) -> Path:
    return _first_existing(
        [
            root / f"qdte5000_seed{seed}_{variant}",
            root / f"qdte5000_{variant}" if seed == 0 else root / f"qdte5000_seed{seed}_{variant}",
        ],
        f"variant run dir for seed={seed}, variant={variant}",
    )


def _variant_measurement_dir(root: Path, seed: int, variant: str) -> Path:
    return _first_existing(
        [
            root / f"seed{seed}_{variant}",
            root / variant if seed == 0 else root / f"seed{seed}_{variant}",
        ],
        f"variant measurement dir for seed={seed}, variant={variant}",
    )


def _load_qcat(run_dir: Path, input_dir: Path) -> QueryCatalogue:
    path = _first_existing([run_dir / "queries.json", input_dir / "queries_full.json"], "query catalogue")
    return QueryCatalogue.from_dict(read_json(path))


def _load_schema(input_dir: Path, run_dir: Path) -> TableSchema:
    path = _first_existing([run_dir / "schema.json", input_dir / "schema.json"], "schema")
    return TableSchema.load_json(path)


def _load_measurement(measurement_dir: Path) -> dict[str, np.ndarray]:
    data = read_json(measurement_dir / "measurements.json")
    target = np.asarray(data["target_projected"], dtype=np.float64)
    noisy = np.asarray(data.get("target_noisy", target), dtype=np.float64)
    variances = np.asarray(data["variances"], dtype=np.float64)
    inv = np.asarray(data.get("inv_variances", 1.0 / np.maximum(variances, 1.0e-12)), dtype=np.float64)
    if target.shape != noisy.shape or target.shape != variances.shape or target.shape != inv.shape:
        raise ValueError(f"Measurement arrays in {measurement_dir} have inconsistent shapes")
    return {
        "target_projected": target,
        "target_noisy": noisy,
        "variances": variances,
        "inv_variances": inv,
    }


def _synthetic_answers(run_dir: Path, qcat: QueryCatalogue, batch_size: int) -> tuple[np.ndarray, int]:
    path = run_dir / "synthetic_encoded.npy"
    synthetic = np.load(path).astype(np.int32, copy=False)
    if synthetic.ndim != 2:
        raise ValueError(f"Expected 2D synthetic table in {path}, got shape {synthetic.shape}")
    answers = answer_queries(synthetic, qcat, batch_size=batch_size).astype(np.float64)
    return answers, int(synthetic.shape[0])


def _true_answers(input_dir: Path, qcat: QueryCatalogue, batch_size: int) -> tuple[np.ndarray, int]:
    real = np.load(input_dir / "real_encoded.npy").astype(np.int32, copy=False)
    if real.ndim != 2:
        raise ValueError(f"Expected 2D real table, got shape {real.shape}")
    answers = answer_queries(real, qcat, batch_size=batch_size).astype(np.float64)
    return answers, int(real.shape[0])


def _query_scope(qcat: QueryCatalogue, qid: int) -> tuple[int, ...]:
    attrs = {int(term[0]) for term in qcat.query_terms(qid)}
    attrs.update(int(attr) for attr, _ in qcat.linear_terms(qid))
    return tuple(sorted(attrs))


def _query_order(qcat: QueryCatalogue, qid: int) -> int:
    return int(qcat.num_terms[qid]) + int(qcat.linear_num_terms[qid])


def _scope_domain_size(qcat: QueryCatalogue, schema: TableSchema, qid: int) -> int:
    scope = _query_scope(qcat, qid)
    if not scope:
        return 1
    cards = schema.cardinalities
    size = 1
    for attr in scope:
        size *= int(cards[int(attr)])
    return int(size)


def _weighted_components(
    *,
    target: np.ndarray,
    synthetic: np.ndarray,
    true_answers: np.ndarray,
    inv_variance: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    s = np.asarray(synthetic, dtype=np.float64)
    a = np.asarray(true_answers, dtype=np.float64)
    inv = np.asarray(inv_variance, dtype=np.float64)
    if y.shape != s.shape or y.shape != a.shape or y.shape != inv.shape:
        raise ValueError("target, synthetic, true_answers, and inv_variance must have matching shapes")

    target_error = y - a
    fit_error = s - y
    final_error = s - a
    t_vec = target_error * target_error * inv
    f_vec = fit_error * fit_error * inv
    c_vec = 2.0 * fit_error * target_error * inv
    e_vec = final_error * final_error * inv
    T = float(np.sum(t_vec))
    F = float(np.sum(f_vec))
    C = float(np.sum(c_vec))
    E = float(np.sum(e_vec))
    return {
        "vectors": {"T": t_vec, "F": f_vec, "C": c_vec, "E": e_vec},
        "T": T,
        "F": F,
        "C": C,
        "E": E,
        "closure": float(E - (T + F + C)),
    }


def _summary_row(
    *,
    dataset: str,
    seed: int,
    synthetic_variant: str,
    target_variant: str,
    synthetic_run_dir: Path,
    measurement_dir: Path,
    qcat: QueryCatalogue,
    n_true: int,
    n_synthetic: int,
    components: dict[str, Any],
) -> dict[str, Any]:
    T = float(components["T"])
    F = float(components["F"])
    E = float(components["E"])
    return {
        "dataset": dataset,
        "seed": int(seed),
        "synthetic_variant": synthetic_variant,
        "target_variant": target_variant,
        "weight_variant": target_variant,
        "synthetic_run_dir": str(synthetic_run_dir),
        "measurement_dir": str(measurement_dir),
        "num_queries": int(qcat.m),
        "n_true": int(n_true),
        "n_synthetic": int(n_synthetic),
        "T_target_to_true_norm2": T,
        "F_synthetic_to_target_norm2": F,
        "C_cross_term": float(components["C"]),
        "E_synthetic_to_true_norm2": E,
        "decomposition_residual": float(components["closure"]),
        "target_to_true_weighted_l2": float(math.sqrt(max(T, 0.0))),
        "synthetic_to_target_weighted_l2": float(math.sqrt(max(F, 0.0))),
        "synthetic_to_true_weighted_l2": float(math.sqrt(max(E, 0.0))),
        "synthetic_to_target_measured_loss": 0.5 * F,
    }


def _grouping_labels(qcat: QueryCatalogue, schema: TableSchema) -> dict[str, list[str]]:
    families = [str(family) for family in qcat.families]
    orders = [str(_query_order(qcat, qid)) for qid in range(qcat.m)]
    catalogue_groups = [str(group) for group in qcat.groups]
    scope_domain_sizes = [str(_scope_domain_size(qcat, schema, qid)) for qid in range(qcat.m)]
    return {
        "all": ["all"] * qcat.m,
        "family": families,
        "marginal_order": orders,
        "catalogue_group": catalogue_groups,
        "scope_domain_size": scope_domain_sizes,
        "family_order": [f"{family}:order={order}" for family, order in zip(families, orders, strict=True)],
        "family_scope_domain_size": [
            f"{family}:domain={domain}" for family, domain in zip(families, scope_domain_sizes, strict=True)
        ],
    }


def _breakdown_rows(
    *,
    base: dict[str, Any],
    qcat: QueryCatalogue,
    schema: TableSchema,
    components: dict[str, Any],
) -> list[dict[str, Any]]:
    vectors = components["vectors"]
    rows: list[dict[str, Any]] = []
    for grouping, labels in _grouping_labels(qcat, schema).items():
        by_key: dict[str, list[int]] = {}
        for qid, label in enumerate(labels):
            by_key.setdefault(str(label), []).append(qid)
        for key in sorted(by_key):
            idx = np.asarray(by_key[key], dtype=np.int32)
            T = float(np.sum(vectors["T"][idx]))
            F = float(np.sum(vectors["F"][idx]))
            C = float(np.sum(vectors["C"][idx]))
            E = float(np.sum(vectors["E"][idx]))
            n = int(len(idx))
            rows.append(
                {
                    "dataset": base["dataset"],
                    "seed": base["seed"],
                    "synthetic_variant": base["synthetic_variant"],
                    "target_variant": base["target_variant"],
                    "weight_variant": base["weight_variant"],
                    "grouping": grouping,
                    "group_key": key,
                    "num_queries": n,
                    "T_target_to_true_norm2": T,
                    "F_synthetic_to_target_norm2": F,
                    "C_cross_term": C,
                    "E_synthetic_to_true_norm2": E,
                    "decomposition_residual": float(E - (T + F + C)),
                    "T_per_query": T / max(n, 1),
                    "F_per_query": F / max(n, 1),
                    "C_per_query": C / max(n, 1),
                    "E_per_query": E / max(n, 1),
                }
            )
    return rows


def _domain_records(schema: TableSchema, attrs: list[int], max_cells: int) -> tuple[np.ndarray | None, int]:
    cards = schema.cardinalities
    total_cells = 1
    for attr in attrs:
        total_cells *= int(cards[int(attr)])
    if total_cells > int(max_cells):
        return None, int(total_cells)
    records = np.zeros((int(total_cells), schema.d), dtype=np.int32)
    if not attrs:
        return records, int(total_cells)
    grids = itertools.product(*(range(int(cards[int(attr)])) for attr in attrs))
    for row_idx, values in enumerate(grids):
        for attr, value in zip(attrs, values, strict=True):
            records[row_idx, int(attr)] = int(value)
    return records, int(total_cells)


def _run_fractional_oracle(
    *,
    dataset: str,
    seed: int,
    target_variant: str,
    qcat: QueryCatalogue,
    schema: TableSchema,
    measurement: dict[str, np.ndarray],
    true_answers: np.ndarray,
    n_total: int,
    source_name: str,
    attrs: list[int] | None,
    max_domain_cells: int,
    query_limit: int,
    max_iterations: int,
) -> dict[str, Any]:
    if attrs is None:
        used_attrs = sorted({attr for qid in range(qcat.m) for attr in _query_scope(qcat, qid)})
    else:
        used_attrs = sorted(set(int(attr) for attr in attrs))
    records, domain_cells = _domain_records(schema, used_attrs, max_domain_cells)
    base: dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "target_variant": target_variant,
        "oracle_source": source_name,
        "num_oracle_attrs": int(len(used_attrs)),
        "oracle_attrs": ",".join(str(attr) for attr in used_attrs),
        "oracle_domain_cells": int(domain_cells),
        "oracle_query_limit": int(query_limit),
    }
    if records is None:
        return {
            **base,
            "status": "skipped_domain_cap",
            "solver_success": False,
            "solver_message": f"domain cells {domain_cells} exceed cap {max_domain_cells}",
        }

    attr_set = set(used_attrs)
    qids = [qid for qid in range(qcat.m) if set(_query_scope(qcat, qid)).issubset(attr_set)]
    if query_limit > 0:
        qids = qids[: int(query_limit)]
    if not qids:
        return {
            **base,
            "status": "skipped_no_queries",
            "solver_success": False,
            "solver_message": "no queries are contained in oracle attrs",
            "num_oracle_queries": 0,
        }

    keep = np.asarray(qids, dtype=np.int32)
    qsub = filter_query_catalogue(qcat, keep)
    phi = np.asarray(eval_records_queries(records, qsub), dtype=np.float64).T
    source = np.asarray(measurement[source_name], dtype=np.float64)[keep]
    inv = np.asarray(measurement["inv_variances"], dtype=np.float64)[keep]
    truth = np.asarray(true_answers, dtype=np.float64)[keep]
    n_atoms = int(phi.shape[1])
    x0 = np.full(n_atoms, float(n_total) / max(n_atoms, 1), dtype=np.float64)

    def objective(p: np.ndarray) -> float:
        residual = phi @ p - source
        return 0.5 * float(np.sum(residual * residual * inv))

    def gradient(p: np.ndarray) -> np.ndarray:
        residual = phi @ p - source
        return np.asarray(phi.T @ (residual * inv), dtype=np.float64)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=Bounds(np.zeros(n_atoms, dtype=np.float64), np.full(n_atoms, float(n_total), dtype=np.float64)),
        constraints=[LinearConstraint(np.ones((1, n_atoms), dtype=np.float64), [float(n_total)], [float(n_total)])],
        method="SLSQP",
        options={"ftol": 1.0e-9, "maxiter": int(max_iterations), "disp": False},
    )
    fitted = np.asarray(result.x, dtype=np.float64)
    oracle_answers = np.asarray(phi @ fitted, dtype=np.float64)
    source_error = source - truth
    oracle_error = oracle_answers - truth
    source_to_oracle = source - oracle_answers
    source_to_true_norm2 = float(np.sum(source_error * source_error * inv))
    oracle_to_true_norm2 = float(np.sum(oracle_error * oracle_error * inv))
    source_to_oracle_norm2 = float(np.sum(source_to_oracle * source_to_oracle * inv))
    return {
        **base,
        "status": "ok" if bool(result.success) else "solver_failed",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "num_oracle_queries": int(len(qids)),
        "source_to_true_norm2": source_to_true_norm2,
        "oracle_to_true_norm2": oracle_to_true_norm2,
        "source_to_oracle_norm2": source_to_oracle_norm2,
        "source_to_oracle_objective": objective(fitted),
        "contraction_slack_norm2": source_to_true_norm2 - oracle_to_true_norm2,
        "min_atom_weight": float(np.min(fitted)) if fitted.size else 0.0,
        "max_atom_weight": float(np.max(fitted)) if fitted.size else 0.0,
        "num_active_atoms": int(np.sum(fitted > 1.0e-8)),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "seed",
        "synthetic_variant",
        "target_variant",
        "T_target_to_true_norm2",
        "F_synthetic_to_target_norm2",
        "C_cross_term",
        "E_synthetic_to_true_norm2",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")


def run(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    seeds = _parse_seeds(args.seeds)
    oracle_attrs = _parse_int_list(args.oracle_attrs)
    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for seed in seeds:
        baseline_run_dir = args.baseline_root / f"seed{seed}"
        variant_run_dir = _variant_run_dir(args.variant_root, seed, args.variant)
        variant_measurement_dir = _variant_measurement_dir(args.variant_root, seed, args.variant)
        qcat = _load_qcat(baseline_run_dir, args.input_dir)
        schema = _load_schema(args.input_dir, baseline_run_dir)
        true, n_true = _true_answers(args.input_dir, qcat, args.batch_size)
        baseline_measurement = _load_measurement(baseline_run_dir)
        variant_measurement = _load_measurement(variant_measurement_dir)

        synthetic_by_variant = {
            "baseline": (baseline_run_dir, *_synthetic_answers(baseline_run_dir, qcat, args.batch_size)),
            args.variant: (variant_run_dir, *_synthetic_answers(variant_run_dir, qcat, args.batch_size)),
        }
        measurements_by_variant = {
            "baseline": (baseline_run_dir, baseline_measurement),
            args.variant: (variant_measurement_dir, variant_measurement),
        }
        for synthetic_variant, (run_dir, synthetic, n_synthetic) in synthetic_by_variant.items():
            for target_variant, (measurement_dir, measurement) in measurements_by_variant.items():
                components = _weighted_components(
                    target=measurement["target_projected"],
                    synthetic=synthetic,
                    true_answers=true,
                    inv_variance=measurement["inv_variances"],
                )
                row = _summary_row(
                    dataset=args.dataset,
                    seed=seed,
                    synthetic_variant=synthetic_variant,
                    target_variant=target_variant,
                    synthetic_run_dir=run_dir,
                    measurement_dir=measurement_dir,
                    qcat=qcat,
                    n_true=n_true,
                    n_synthetic=n_synthetic,
                    components=components,
                )
                summary_rows.append(row)
                breakdown_rows.extend(_breakdown_rows(base=row, qcat=qcat, schema=schema, components=components))

        if args.compute_fractional_oracle:
            for target_variant, (_, measurement) in measurements_by_variant.items():
                oracle_rows.append(
                    _run_fractional_oracle(
                        dataset=args.dataset,
                        seed=seed,
                        target_variant=target_variant,
                        qcat=qcat,
                        schema=schema,
                        measurement=measurement,
                        true_answers=true,
                        n_total=n_true,
                        source_name=args.oracle_source,
                        attrs=oracle_attrs,
                        max_domain_cells=args.oracle_max_domain_cells,
                        query_limit=args.oracle_query_limit,
                        max_iterations=args.oracle_max_iterations,
                    )
                )

    output = {"summary_rows": summary_rows, "breakdown_rows": breakdown_rows, "oracle_rows": oracle_rows}
    if args.output_json is not None:
        write_json(output, args.output_json)
    if args.output_csv is not None:
        _write_csv(summary_rows, args.output_csv, SUMMARY_COLUMNS)
    if args.output_breakdown_csv is not None:
        _write_csv(breakdown_rows, args.output_breakdown_csv, BREAKDOWN_COLUMNS)
    if args.output_oracle_csv is not None:
        _write_csv(oracle_rows, args.output_oracle_csv, ORACLE_COLUMNS)
    if args.output_md is not None:
        _write_markdown(summary_rows, args.output_md)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute offline projection-to-generation transfer diagnostics. "
            "Exact true answers are used only for post-run diagnostics."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--variant-root", required=True, type=Path)
    parser.add_argument("--variant", default="current_bootdiag16")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-breakdown-csv", type=Path)
    parser.add_argument("--output-oracle-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--compute-fractional-oracle", action="store_true")
    parser.add_argument(
        "--oracle-source",
        choices=["target_projected", "target_noisy"],
        default="target_projected",
        help="Measurement vector to project onto the reduced fractional row-realizable set.",
    )
    parser.add_argument(
        "--oracle-attrs",
        help="Comma-separated reduced-domain attribute ids. Defaults to all query attributes, usually requiring a cap skip.",
    )
    parser.add_argument("--oracle-max-domain-cells", type=int, default=20_000)
    parser.add_argument("--oracle-query-limit", type=int, default=512)
    parser.add_argument("--oracle-max-iterations", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
