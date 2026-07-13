#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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

from qdte.config import load_yaml, parse_scalar, save_yaml, set_nested
from qdte.dataio import read_json, write_json
from qdte.eval.metrics import measured_loss, query_error_metrics
from qdte.measurement.measure import (
    Measurements,
    _apply_configured_projection,
    _apply_projection_aware_uncertainty,
    measurement_group_from_dict,
)
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema


def _resolve_measurement_path(path: Path) -> Path:
    if path.is_dir():
        return path / "measurements.json"
    return path


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these paths exists: " + ", ".join(str(path) for path in candidates))


def _apply_override_pairs(config: dict[str, Any], override_pairs: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for item in override_pairs:
        if "=" not in item:
            raise ValueError(f"--override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        set_nested(resolved, key, parse_scalar(value))
    return resolved


def _load_config(config_path: Path | None, overrides: list[str]) -> dict[str, Any]:
    config = load_yaml(config_path) if config_path is not None else {}
    return _apply_override_pairs(config, overrides)


def _load_schema(args: argparse.Namespace, artifact_dir: Path) -> tuple[TableSchema, Path]:
    candidates: list[Path] = []
    if args.schema is not None:
        candidates.append(args.schema)
    candidates.append(artifact_dir / "schema.json")
    if args.input_dir is not None:
        candidates.append(args.input_dir / "schema.json")
    path = _first_existing(candidates)
    return TableSchema.load_json(path), path


def _load_queries(args: argparse.Namespace, artifact_dir: Path) -> tuple[QueryCatalogue, Path]:
    candidates: list[Path] = []
    if args.queries is not None:
        candidates.append(args.queries)
    candidates.extend([artifact_dir / "queries.json", artifact_dir / "queries_full.json"])
    if args.input_dir is not None:
        candidates.append(args.input_dir / "queries_full.json")
    path = _first_existing(candidates)
    return QueryCatalogue.from_dict(read_json(path)), path


def _raw_variances_from_groups(groups: list[Any], num_queries: int, fallback_variances: np.ndarray) -> np.ndarray:
    raw = np.full(num_queries, np.nan, dtype=np.float64)
    for group in groups:
        idx = np.asarray(group.query_indices, dtype=np.int32)
        noise_var = float(group.noise_std) * float(group.noise_std)
        if noise_var > 0.0:
            raw[idx] = noise_var
    fallback = np.asarray(fallback_variances, dtype=np.float64)
    missing = ~np.isfinite(raw)
    if np.any(missing):
        raw[missing] = fallback[missing]
    return np.maximum(raw, 1.0e-12)


def _infer_total(
    args: argparse.Namespace,
    measurement_data: dict[str, Any],
    groups: list[Any],
    input_dir: Path | None,
) -> tuple[int, str]:
    if args.total is not None:
        return int(args.total), "--total"
    if args.real_encoded is not None:
        return int(np.load(args.real_encoded, mmap_mode="r").shape[0]), "--real-encoded rows"
    if input_dir is not None and (input_dir / "real_encoded.npy").exists():
        return int(np.load(input_dir / "real_encoded.npy", mmap_mode="r").shape[0]), "--input-dir real_encoded.npy rows"

    consistency = dict(measurement_data.get("projection_diagnostics", {}).get("consistency", {}))
    known_total = consistency.get("known_total_count")
    if known_total is not None:
        return int(known_total), "existing projection_diagnostics.consistency.known_total_count"

    target_projected = np.asarray(measurement_data["target_projected"], dtype=np.float64)
    for group in groups:
        if bool(group.is_partition) and len(group.query_indices) > 0:
            total = int(round(float(np.sum(target_projected[np.asarray(group.query_indices, dtype=np.int32)]))))
            if total >= 0:
                return total, f"rounded sum of existing projected partition group {group.name}"

    raise ValueError(
        "Unable to infer the public row count. Pass --total, --real-encoded, "
        "or --input-dir with real_encoded.npy."
    )


def _target_metrics(
    *,
    input_dir: Path | None,
    real_encoded_path: Path,
    qcat: QueryCatalogue,
    target_noisy: np.ndarray,
    target_projected: np.ndarray,
    variances: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    real = np.load(real_encoded_path).astype(np.int32, copy=False)
    if real.ndim != 2:
        raise ValueError(f"real_encoded must be a 2D encoded table, got shape {real.shape}")
    true_answers = answer_queries(real, qcat, batch_size=batch_size).astype(np.float64)
    n_real = int(real.shape[0])
    inv = 1.0 / np.maximum(np.asarray(variances, dtype=np.float64), 1.0e-12)
    projected_error = np.asarray(target_projected, dtype=np.float64) - true_answers
    noisy_error = np.asarray(target_noisy, dtype=np.float64) - true_answers

    projected = query_error_metrics(true_answers, target_projected, n_real, n_real, prefix="target_to_true")
    noisy = query_error_metrics(true_answers, target_noisy, n_real, n_real, prefix="noisy_to_true")
    output: dict[str, Any] = {
        "offline_true_answers_used": True,
        "offline_only": True,
        "n_real": n_real,
        "num_queries": int(qcat.m),
        "target_to_true_weighted_l2": float(np.sqrt(np.sum(projected_error * projected_error * inv))),
        "target_to_true_measured_loss": measured_loss(projected_error.astype(np.float32), inv.astype(np.float32)),
        "noisy_to_true_weighted_l2": float(np.sqrt(np.sum(noisy_error * noisy_error * inv))),
        "noisy_to_true_measured_loss": measured_loss(noisy_error.astype(np.float32), inv.astype(np.float32)),
        **projected,
        **noisy,
    }

    if input_dir is not None and (input_dir / "workload_groups.json").exists():
        from qdte.eval.external import _vector_block_tvd

        projected_blocks, _ = _vector_block_tvd(
            input_dir,
            qcat,
            true_answers,
            np.asarray(target_projected, dtype=np.float64),
            n_real,
            n_real,
        )
        noisy_blocks, _ = _vector_block_tvd(
            input_dir,
            qcat,
            true_answers,
            np.asarray(target_noisy, dtype=np.float64),
            n_real,
            n_real,
        )
        projected_tvd = np.asarray([float(item["tvd"]) for item in projected_blocks], dtype=np.float64)
        noisy_tvd = np.asarray([float(item["tvd"]) for item in noisy_blocks], dtype=np.float64)
        output.update(
            {
                "target_to_true_avg_tvd": float(np.mean(projected_tvd)) if projected_tvd.size else 0.0,
                "target_to_true_max_tvd": float(np.max(projected_tvd)) if projected_tvd.size else 0.0,
                "noisy_to_true_avg_tvd": float(np.mean(noisy_tvd)) if noisy_tvd.size else 0.0,
                "noisy_to_true_max_tvd": float(np.max(noisy_tvd)) if noisy_tvd.size else 0.0,
                "num_vector_blocks": int(projected_tvd.size),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    measurement_path = _resolve_measurement_path(args.measurement)
    artifact_dir = measurement_path.parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_data = read_json(measurement_path)
    qcat, queries_path = _load_queries(args, artifact_dir)
    schema, schema_path = _load_schema(args, artifact_dir)
    groups = [measurement_group_from_dict(group) for group in measurement_data.get("groups", [])]
    total, total_source = _infer_total(args, measurement_data, groups, args.input_dir)
    config = _load_config(args.config, list(args.override or []))
    privacy_cfg = config.get("privacy", {}) if isinstance(config.get("privacy", {}), dict) else {}
    projection_cfg = config.get("projection", {}) if isinstance(config.get("projection", {}), dict) else {}
    min_variance = float(privacy_cfg.get("min_variance", 1.0e-6))

    target_noisy = np.asarray(measurement_data["target_noisy"], dtype=np.float32)
    existing_projected = np.asarray(measurement_data["target_projected"], dtype=np.float32)
    existing_variances = np.asarray(measurement_data["variances"], dtype=np.float32)
    if target_noisy.shape != (qcat.m,):
        raise ValueError(f"target_noisy has shape {target_noisy.shape}, expected ({qcat.m},)")
    existing_uncertainty = dict(
        measurement_data.get("projection_diagnostics", {}).get("uncertainty", {})
    )
    if bool(existing_uncertainty.get("enabled", False)):
        raw_variances = _raw_variances_from_groups(groups, qcat.m, existing_variances).astype(np.float32)
        raw_variance_source = "measurement_groups_before_existing_uncertainty"
    else:
        # Adaptive artifacts may precision-combine repeated measurements. Their
        # saved variances describe target_noisy; a group's last noise_std does not.
        raw_variances = np.maximum(existing_variances, min_variance).astype(np.float32)
        raw_variance_source = "artifact_variances"

    start = time.perf_counter()
    if args.projection_mode == "raw_noisy":
        target_projected = target_noisy.copy()
        variances = raw_variances.copy()
        projection_diagnostics: dict[str, Any] = {
            "mode": "raw_noisy",
            "consistency": {"enabled": False},
            "uncertainty": {"enabled": False},
        }
        uncertainty_bias = None
    elif args.projection_mode == "keep_projected":
        target_projected = existing_projected.copy()
        variances = existing_variances.copy()
        projection_diagnostics = dict(measurement_data.get("projection_diagnostics", {}))
        projection_diagnostics["mode"] = "keep_projected"
        uncertainty_bias = None
    elif args.projection_mode == "configured":
        target_projected, projection_diagnostics = _apply_configured_projection(
            target_noisy,
            qcat,
            groups,
            total,
            projection_cfg,
            raw_variances,
            schema.cardinalities,
        )
        rng_seed = int(args.seed if args.seed is not None else config.get("run", {}).get("seed", 0))
        target_projected, variances, uncertainty_diagnostics, uncertainty_bias = _apply_projection_aware_uncertainty(
            target_noisy,
            target_projected,
            qcat,
            groups,
            total,
            projection_cfg,
            raw_variances,
            schema.cardinalities,
            np.random.default_rng(rng_seed),
            min_variance=min_variance,
        )
        projection_diagnostics["uncertainty"] = uncertainty_diagnostics
    else:
        raise ValueError(f"Unknown projection mode: {args.projection_mode}")
    projection_seconds = time.perf_counter() - start

    variances = np.maximum(np.asarray(variances, dtype=np.float32), min_variance).astype(np.float32)
    inv_variances = (1.0 / variances).astype(np.float32)
    measurements = Measurements(
        target_noisy=target_noisy.astype(np.float32),
        target_projected=np.asarray(target_projected, dtype=np.float32),
        variances=variances,
        inv_variances=inv_variances,
        groups=groups,
        mode=str(measurement_data.get("mode", "dp")),
        rho_total=float(measurement_data.get("rho_total", 0.0)),
        rho_spent=float(measurement_data.get("rho_spent", 0.0)),
        epsilon_delta=float(measurement_data.get("epsilon_delta", 0.0)),
        delta=float(measurement_data.get("delta", 0.0)),
        projection_diagnostics=projection_diagnostics,
        projection_uncertainty_bias=uncertainty_bias,
        num_rows=int(total),
    )

    write_json(measurements.to_public_dict(), output_dir / "measurements.json")
    qcat.save_json(output_dir / "queries.json")
    schema.save_json(output_dir / "schema.json")
    write_json([group.to_dict() for group in groups], output_dir / "measurement_groups.json")
    write_json(projection_diagnostics, output_dir / "projection_diagnostics.json")
    save_yaml(config, output_dir / "config_reprojection.yaml")

    metadata = {
        "variant": str(args.variant_name or output_dir.name),
        "projection_mode": str(args.projection_mode),
        "measurement_path": str(measurement_path),
        "queries_path": str(queries_path),
        "schema_path": str(schema_path),
        "input_dir": str(args.input_dir) if args.input_dir is not None else "",
        "total": int(total),
        "total_source": total_source,
        "projection_runtime_seconds": float(projection_seconds),
        "num_queries": int(qcat.m),
        "num_measurement_groups": int(len(groups)),
        "raw_variance_mean": float(np.mean(raw_variances)),
        "raw_variance_source": raw_variance_source,
        "projected_variance_mean": float(np.mean(variances)),
        "min_projected_answer": float(np.min(target_projected)) if qcat.m else 0.0,
        "max_projected_answer": float(np.max(target_projected)) if qcat.m else 0.0,
        "negative_projected_count": int(np.sum(np.asarray(target_projected) < -1.0e-9)),
        "offline_true_metrics_computed": bool(args.compute_target_metrics),
        "offline_true_metrics_note": (
            "Exact true answers are used only in projection_target_metrics_offline.json."
            if args.compute_target_metrics
            else "No exact true answers were loaded by this reprojection run."
        ),
        "command": " ".join(sys.argv),
    }
    write_json(metadata, output_dir / "reproject_metadata.json")

    if args.compute_target_metrics:
        real_encoded_path = args.real_encoded
        if real_encoded_path is None:
            if args.input_dir is None:
                raise ValueError("--compute-target-metrics requires --real-encoded or --input-dir")
            real_encoded_path = args.input_dir / "real_encoded.npy"
        metrics = _target_metrics(
            input_dir=args.input_dir,
            real_encoded_path=real_encoded_path,
            qcat=qcat,
            target_noisy=target_noisy,
            target_projected=np.asarray(target_projected, dtype=np.float32),
            variances=variances,
            batch_size=int(args.batch_size),
        )
        write_json(metrics, output_dir / "projection_target_metrics_offline.json")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproject a saved noisy DP measurement artifact without remeasuring the private data."
    )
    parser.add_argument("--measurement", required=True, type=Path, help="Path to measurements.json or a run directory.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="YAML config supplying projection/privacy settings.")
    parser.add_argument("--override", action="append", default=[], help="Override config value as KEY=VALUE.")
    parser.add_argument("--queries", type=Path, help="Query catalogue JSON. Defaults to artifact queries.json.")
    parser.add_argument("--schema", type=Path, help="Schema JSON. Defaults to artifact schema.json.")
    parser.add_argument("--input-dir", type=Path, help="Canonical external input dir for schema/queries/optional metrics.")
    parser.add_argument("--real-encoded", type=Path, help="Encoded real table for row count or offline target metrics.")
    parser.add_argument("--total", type=int, help="Public row count used by projection constraints.")
    parser.add_argument(
        "--projection-mode",
        choices=["configured", "raw_noisy", "keep_projected"],
        default="configured",
        help="Projection source: apply config to target_noisy, keep target_noisy, or preserve existing target_projected.",
    )
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--seed", type=int, help="RNG seed for bootstrap projection uncertainty.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--compute-target-metrics",
        action="store_true",
        help="Offline-only true-answer target diagnostics. Never written to measurements.json.",
    )
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(
        f"{metadata['variant']}: projection_mode={metadata['projection_mode']} "
        f"queries={metadata['num_queries']} projection_runtime={metadata['projection_runtime_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
