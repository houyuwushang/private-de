#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import read_json, write_json
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema


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


def _load_encoded_table(path: Path, schema: TableSchema, label: str) -> np.ndarray:
    table = np.load(path)
    if table.ndim != 2:
        raise ValueError(f"{label} must be a 2D encoded table, got shape {table.shape}")
    if table.shape[1] != schema.d:
        raise ValueError(f"{label} has {table.shape[1]} columns, expected {schema.d}")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError(f"{label} must have integer dtype, got {table.dtype}")
    table = table.astype(np.int32, copy=False)
    cardinalities = schema.cardinalities
    for col_idx, cardinality in enumerate(cardinalities.tolist()):
        values = table[:, col_idx]
        if values.size and (int(values.min()) < 0 or int(values.max()) >= int(cardinality)):
            raise ValueError(
                f"{label} column {col_idx} has values outside [0,{int(cardinality)}): "
                f"min={int(values.min())}, max={int(values.max())}"
            )
    return table


def _answer_queries(X: np.ndarray, qcat: QueryCatalogue, batch_size: int) -> np.ndarray:
    try:
        from qdte.queries.eval_jax import answer_queries

        return answer_queries(X, qcat, batch_size=batch_size).astype(np.float64)
    except Exception:
        answers = np.zeros(qcat.m, dtype=np.float64)
        for qid in range(qcat.m):
            answers[qid] = float(np.sum(qcat.eval_query_np(X, qid)))
        return answers


def _anchor_balanced_lattice_target(
    mixture: np.ndarray,
    anchor: np.ndarray,
    groups: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    mixture_arr = np.asarray(mixture, dtype=np.float64)
    anchor_arr = np.asarray(anchor, dtype=np.float64)
    if mixture_arr.ndim != 1 or anchor_arr.shape != mixture_arr.shape:
        raise ValueError("mixture and anchor must be matching one-dimensional vectors")

    nearest = np.rint(mixture_arr)
    canonical = np.where(np.abs(mixture_arr - nearest) <= 1.0e-8, nearest, mixture_arr)
    output = np.floor(canonical)
    seen = np.zeros(mixture_arr.shape, dtype=np.int32)
    partition_mass_violations = 0
    anchor_ties = 0
    for group_position, group in enumerate(groups):
        idx = np.asarray(group.get("query_indices", []), dtype=np.int32)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"measurement group {group_position} has no query indices")
        if np.any(idx < 0) or np.any(idx >= mixture_arr.size):
            raise ValueError(f"measurement group {group_position} has out-of-range query indices")
        seen[idx] += 1
        local = canonical[idx]
        local_floor = np.floor(local)
        fractional = local - local_floor
        if bool(group.get("is_partition", False)):
            target_mass = int(round(float(np.sum(local, dtype=np.float64))))
            round_up_count = target_mass - int(np.sum(local_floor, dtype=np.float64))
            fractional_positions = np.flatnonzero(fractional > 1.0e-8)
            if not 0 <= round_up_count <= fractional_positions.size:
                raise ValueError(
                    f"partition group {group.get('name', group_position)!r} cannot be mass-balanced"
                )
            preference = anchor_arr[idx[fractional_positions]] - local[fractional_positions]
            order = np.lexsort((idx[fractional_positions], -preference))
            chosen = fractional_positions[order[:round_up_count]]
            output[idx] = local_floor
            output[idx[chosen]] += 1.0
            if int(round(float(np.sum(output[idx])))) != target_mass:
                partition_mass_violations += 1
            anchor_ties += int(np.sum(np.isclose(fractional[chosen], 0.5, atol=1.0e-8)))
        else:
            choose_up = fractional > 0.5 + 1.0e-8
            ties = np.isclose(fractional, 0.5, atol=1.0e-8)
            choose_up |= ties & (anchor_arr[idx] >= local)
            output[idx] = local_floor + choose_up.astype(np.float64)
            anchor_ties += int(np.sum(ties))

    if np.any(seen != 1):
        raise ValueError(
            "measurement groups must cover every query exactly once for anchor-balanced rounding"
        )
    if partition_mass_violations:
        raise AssertionError("anchor-balanced rounding failed to preserve a partition mass")
    rounding_error = output - mixture_arr
    diagnostics = {
        "mode": "anchor_balanced",
        "num_fractional_queries_before": int(
            np.sum(np.abs(mixture_arr - np.rint(mixture_arr)) > 1.0e-8)
        ),
        "num_anchor_ties": int(anchor_ties),
        "rounding_l1_counts": float(np.sum(np.abs(rounding_error), dtype=np.float64)),
        "rounding_l2_counts": float(np.linalg.norm(rounding_error)),
        "rounding_max_abs_count": float(np.max(np.abs(rounding_error))) if rounding_error.size else 0.0,
        "partition_mass_violations": 0,
    }
    return output, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    measurement_path = _resolve_measurement_path(args.measurement)
    artifact_dir = measurement_path.parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_data = read_json(measurement_path)
    qcat, queries_path = _load_qcat(args, artifact_dir)
    schema, schema_path = _load_schema(args, artifact_dir)
    teacher = _load_encoded_table(args.teacher_synthetic, schema, "teacher_synthetic")
    teacher_answers = _answer_queries(teacher, qcat, batch_size=int(args.batch_size))
    anchor_answers = teacher_answers.copy()
    secondary_path = getattr(args, "secondary_synthetic", None)
    secondary_weight = float(getattr(args, "secondary_weight", 0.5))
    secondary: np.ndarray | None = None
    if secondary_path is not None:
        if not 0.0 <= secondary_weight <= 1.0:
            raise ValueError("secondary_weight must be in [0, 1]")
        secondary = _load_encoded_table(
            Path(secondary_path), schema, "secondary_synthetic"
        )
        if secondary.shape[0] != teacher.shape[0]:
            raise ValueError(
                "teacher_synthetic and secondary_synthetic must have the same row count"
            )
        secondary_answers = _answer_queries(
            secondary, qcat, batch_size=int(args.batch_size)
        )
        teacher_answers = (
            (1.0 - secondary_weight) * teacher_answers
            + secondary_weight * secondary_answers
        )
    lattice_rounding = str(getattr(args, "lattice_rounding", "none"))
    lattice_diagnostics: dict[str, Any] = {"mode": "none"}
    if lattice_rounding == "anchor_balanced":
        if secondary is None:
            raise ValueError("anchor_balanced lattice rounding requires secondary_synthetic")
        teacher_answers, lattice_diagnostics = _anchor_balanced_lattice_target(
            teacher_answers,
            anchor_answers,
            list(measurement_data.get("groups", [])),
        )
    elif lattice_rounding != "none":
        raise ValueError("lattice_rounding must be none or anchor_balanced")

    variances = np.asarray(measurement_data["variances"], dtype=np.float64)
    if variances.shape != teacher_answers.shape:
        raise ValueError(
            f"variance shape {variances.shape} does not match teacher answers {teacher_answers.shape}"
        )
    inv_variances = 1.0 / np.maximum(variances, 1.0e-12)
    output_measurement = dict(measurement_data)
    output_measurement["target_projected"] = teacher_answers.astype(np.float32).tolist()
    output_measurement["target_noisy"] = teacher_answers.astype(np.float32).tolist()
    output_measurement["projection_diagnostics"] = {
        **dict(measurement_data.get("projection_diagnostics", {})),
        "teacher_target": {
            "enabled": True,
            "method": (
                "synthetic_teacher_convex_mixture"
                if secondary is not None
                else "synthetic_teacher_answers"
            ),
            "teacher_synthetic_path": str(args.teacher_synthetic),
            "secondary_synthetic_path": (
                str(secondary_path) if secondary is not None else ""
            ),
            "secondary_weight": (
                secondary_weight if secondary is not None else 0.0
            ),
            "teacher_n_rows": int(teacher.shape[0]),
            "num_queries": int(qcat.m),
            "variance_source": "reused_measurement_variances",
            "teacher_target_measured_self_loss": 0.0,
            "teacher_target_unweighted_self_loss": 0.0,
            "lattice_rounding": lattice_diagnostics,
            "inv_variance_min": float(np.min(inv_variances)) if inv_variances.size else 0.0,
            "inv_variance_max": float(np.max(inv_variances)) if inv_variances.size else 0.0,
        },
    }

    write_json(output_measurement, output_dir / "measurements.json")
    qcat.save_json(output_dir / "queries.json")
    schema.save_json(output_dir / "schema.json")
    if "groups" in measurement_data:
        write_json(measurement_data["groups"], output_dir / "measurement_groups.json")
    metadata = {
        "variant": args.variant_name or output_dir.name,
        "measurement_path": str(measurement_path),
        "teacher_synthetic_path": str(args.teacher_synthetic),
        "secondary_synthetic_path": (
            str(secondary_path) if secondary is not None else ""
        ),
        "secondary_weight": secondary_weight if secondary is not None else 0.0,
        "lattice_rounding": lattice_diagnostics,
        "queries_path": str(queries_path),
        "schema_path": str(schema_path),
        "input_dir": str(args.input_dir) if args.input_dir is not None else "",
        "teacher_n_rows": int(teacher.shape[0]),
        "num_queries": int(qcat.m),
        "runtime_seconds": float(time.perf_counter() - start),
        "offline_true_answers_used": False,
        "offline_true_answers_note": "No exact true answers are loaded by teacher-target materialization.",
        "command": " ".join(sys.argv),
    }
    write_json(metadata, output_dir / "teacher_target_materialization_metadata.json")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a row-realizable teacher target from a synthetic table into a measurement artifact."
    )
    parser.add_argument("--measurement", required=True, type=Path, help="Path to measurements.json or a run directory.")
    parser.add_argument("--teacher-synthetic", required=True, type=Path)
    parser.add_argument("--secondary-synthetic", type=Path)
    parser.add_argument("--secondary-weight", type=float, default=0.5)
    parser.add_argument(
        "--lattice-rounding",
        choices=["none", "anchor_balanced"],
        default="none",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-dir", type=Path, help="Canonical input dir for schema/queries.")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--variant-name", default="")
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(
        f"{metadata['variant']}: teacher_rows={metadata['teacher_n_rows']} "
        f"queries={metadata['num_queries']} runtime={metadata['runtime_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
