from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from qdte.eval.metrics import query_error_metrics
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

TRUE_ANSWER_CACHE_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    return orjson.loads(path.read_bytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_encoded_table(X: np.ndarray, schema: TableSchema, name: str) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D encoded table, got shape {arr.shape}")
    if arr.shape[1] != schema.d:
        raise ValueError(f"{name} has {arr.shape[1]} columns, expected {schema.d}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype, got {arr.dtype}")
    out = arr.astype(np.int32, copy=False)
    cardinalities = schema.cardinalities
    for col_idx, cardinality in enumerate(cardinalities.tolist()):
        values = out[:, col_idx]
        if values.size == 0:
            continue
        lo = int(values.min())
        hi = int(values.max())
        if lo < 0 or hi >= int(cardinality):
            raise ValueError(
                f"{name} column {col_idx} has values outside [0, {int(cardinality)}): "
                f"min={lo}, max={hi}"
            )
    return out


def _answer_queries(X: np.ndarray, qcat: QueryCatalogue, batch_size: int) -> np.ndarray:
    try:
        from qdte.queries.eval_jax import answer_queries

        return answer_queries(X, qcat, batch_size=batch_size).astype(np.float64)
    except Exception:
        counts = np.zeros(qcat.m, dtype=np.float64)
        for qid in range(qcat.m):
            counts[qid] = float(np.sum(qcat.eval_query_np(X, qid)))
        return counts


def _true_answer_cache_metadata(input_dir: Path, schema: TableSchema, qcat: QueryCatalogue, n_real: int) -> dict[str, Any]:
    return {
        "version": TRUE_ANSWER_CACHE_VERSION,
        "schema_sha256": _sha256_file(input_dir / "schema.json"),
        "queries_sha256": _sha256_file(input_dir / "queries_full.json"),
        "real_encoded_sha256": _sha256_file(input_dir / "real_encoded.npy"),
        "num_columns": int(schema.d),
        "num_queries": int(qcat.m),
        "n_real": int(n_real),
    }


def _load_true_answer_cache(cache_path: Path, expected_metadata: dict[str, Any]) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as data:
            cached_metadata = orjson.loads(np.asarray(data["metadata"], dtype=np.uint8).tobytes())
            if cached_metadata != expected_metadata:
                return None
            true_answers = np.asarray(data["true_answers"], dtype=np.float64)
    except Exception:
        return None
    if true_answers.shape != (int(expected_metadata["num_queries"]),):
        return None
    return true_answers


def _write_true_answer_cache(cache_path: Path, true_answers: np.ndarray, metadata: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    metadata_bytes = orjson.dumps(metadata)
    with tmp_path.open("wb") as fh:
        np.savez_compressed(
            fh,
            true_answers=np.asarray(true_answers, dtype=np.float64),
            metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
        )
    tmp_path.replace(cache_path)


def _answer_true_queries(
    input_dir: Path,
    real: np.ndarray,
    schema: TableSchema,
    qcat: QueryCatalogue,
    batch_size: int,
    cache_path: str | Path | None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if cache_path is None:
        return _answer_queries(real, qcat, batch_size=batch_size), None

    resolved_cache_path = Path(cache_path)
    metadata = _true_answer_cache_metadata(input_dir, schema, qcat, n_real=int(real.shape[0]))
    cached_answers = _load_true_answer_cache(resolved_cache_path, metadata)
    if cached_answers is not None:
        return cached_answers, {"path": str(resolved_cache_path), "status": "hit"}

    true_answers = _answer_queries(real, qcat, batch_size=batch_size)
    try:
        _write_true_answer_cache(resolved_cache_path, true_answers, metadata)
        status = "miss_written"
    except OSError as exc:
        status = f"miss_write_failed:{exc.__class__.__name__}"
    return true_answers, {"path": str(resolved_cache_path), "status": status}


def _group_indices(labels: list[str]) -> dict[str, np.ndarray]:
    positions: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        positions[str(label)].append(int(idx))
    return {label: np.asarray(indices, dtype=np.int32) for label, indices in positions.items()}


def _load_partition_blocks(input_dir: Path, qcat: QueryCatalogue) -> list[dict[str, Any]]:
    path = input_dir / "workload_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"External evaluation requires workload_groups.json: {path}")
    raw_groups = orjson.loads(path.read_bytes())
    if not isinstance(raw_groups, list):
        raise ValueError("workload_groups.json must contain a list")
    blocks: list[dict[str, Any]] = []
    coverage = np.zeros(qcat.m, dtype=np.int32)
    for group_id, raw in enumerate(raw_groups):
        if not isinstance(raw, dict):
            raise ValueError(f"Workload group {group_id} must be a mapping")
        idx = np.asarray(raw.get("query_indices", []), dtype=np.int32)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"Workload group {group_id} must contain a non-empty 1D query index vector")
        if np.any(idx < 0) or np.any(idx >= qcat.m):
            raise ValueError(f"Workload group {group_id} has out-of-range query indices")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"Workload group {group_id} contains duplicate query indices")
        coverage[idx] += 1
        if idx.size <= 1 or not bool(raw.get("is_partition", False)):
            continue
        blocks.append(
            {
                "name": str(raw.get("name", raw.get("group", ""))),
                "family": str(raw.get("family", "unknown")),
                "query_indices": idx,
            }
        )
    if np.any(coverage != 1):
        raise ValueError(
            "workload_groups.json must cover every query exactly once; "
            f"missing={int(np.sum(coverage == 0))}, overlapping={int(np.sum(coverage > 1))}"
        )
    if not blocks:
        raise ValueError("External TVD evaluation requires at least one partition block")
    return blocks


def _per_family_metrics(
    qcat: QueryCatalogue,
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    n_real: int,
    n_syn: int,
) -> dict[str, dict[str, float | int]]:
    by_family = _group_indices(qcat.families)
    output: dict[str, dict[str, float | int]] = {}
    for family, idx in sorted(by_family.items()):
        metrics = query_error_metrics(true_answers[idx], syn_answers[idx], n_real, n_syn, prefix="family")
        output[family] = {
            "num_queries": int(idx.size),
            "mae": float(metrics["family_mae"]),
            "rmse": float(metrics["family_rmse"]),
            "max_error": float(metrics["family_max_error"]),
        }
    return output


def _vector_block_tvd(
    input_dir: Path,
    qcat: QueryCatalogue,
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    n_real: int,
    n_syn: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, float | int]]]:
    true_rate = true_answers.astype(np.float64) / float(n_real)
    syn_rate = syn_answers.astype(np.float64) / float(n_syn)
    blocks = _load_partition_blocks(input_dir, qcat)
    details: list[dict[str, float | int | str]] = []
    tvd_by_family: dict[str, list[float]] = defaultdict(list)
    for block in sorted(blocks, key=lambda item: str(item["name"])):
        idx = np.asarray(block["query_indices"], dtype=np.int32)
        family = str(block["family"])
        true_mass = float(np.sum(true_answers[idx]))
        syn_mass = float(np.sum(syn_answers[idx]))
        if not math.isclose(true_mass, float(n_real), rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError(
                f"Partition block {block['name']!r} is incomplete for real data: "
                f"sum={true_mass}, expected={n_real}"
            )
        if not math.isclose(syn_mass, float(n_syn), rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError(
                f"Partition block {block['name']!r} is incomplete for synthetic data: "
                f"sum={syn_mass}, expected={n_syn}"
            )
        tvd = float(0.5 * np.sum(np.abs(syn_rate[idx] - true_rate[idx])))
        details.append(
            {
                "group": str(block["name"]),
                "family": family,
                "num_queries": int(idx.size),
                "tvd": tvd,
            }
        )
        tvd_by_family[family].append(tvd)
    family_summary: dict[str, dict[str, float | int]] = {}
    for family, values in sorted(tvd_by_family.items()):
        arr = np.asarray(values, dtype=np.float64)
        family_summary[family] = {
            "num_blocks": int(arr.size),
            "avg_tvd": float(np.mean(arr)) if arr.size else 0.0,
            "max_tvd": float(np.max(arr)) if arr.size else 0.0,
        }
    return details, family_summary


def evaluate_external_synthetic(
    input_dir: str | Path,
    synthetic_path: str | Path,
    *,
    run_metadata_path: str | Path | None = None,
    batch_size: int = 8192,
    include_block_details: bool = True,
    true_answers_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    synthetic_path = Path(synthetic_path)
    schema = TableSchema.from_dict(_read_json(input_dir / "schema.json"))
    qcat = QueryCatalogue.from_dict(_read_json(input_dir / "queries_full.json"))
    schema.validate()
    qcat.validate(schema.cardinalities)
    real = _validate_encoded_table(np.load(input_dir / "real_encoded.npy"), schema, "real_encoded")
    synthetic = _validate_encoded_table(np.load(synthetic_path), schema, "synthetic_encoded")
    if real.shape[0] <= 0:
        raise ValueError("real_encoded must contain at least one row")
    if synthetic.shape[0] <= 0:
        raise ValueError("synthetic_encoded must contain at least one row")

    true_answers, cache_metadata = _answer_true_queries(
        input_dir,
        real,
        schema,
        qcat,
        batch_size=batch_size,
        cache_path=true_answers_cache_path,
    )
    syn_answers = _answer_queries(synthetic, qcat, batch_size=batch_size)
    n_real = int(real.shape[0])
    n_syn = int(synthetic.shape[0])
    full_metrics = query_error_metrics(true_answers, syn_answers, n_real, n_syn, prefix="full_true")
    block_details, tvd_by_family = _vector_block_tvd(input_dir, qcat, true_answers, syn_answers, n_real, n_syn)
    tvd_values = np.asarray([float(item["tvd"]) for item in block_details], dtype=np.float64)

    metadata_path = input_dir / "metadata.json"
    input_metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    run_metadata: dict[str, Any] = {}
    if run_metadata_path is not None and Path(run_metadata_path).exists():
        run_metadata = _read_json(Path(run_metadata_path))
    elif (synthetic_path.parent / "run_metadata.json").exists():
        run_metadata = _read_json(synthetic_path.parent / "run_metadata.json")

    output: dict[str, Any] = {
        "dataset": input_metadata.get("dataset", input_dir.name),
        "input_dir": str(input_dir),
        "synthetic_path": str(synthetic_path),
        "n_real": n_real,
        "n_synthetic": n_syn,
        "num_columns": int(schema.d),
        "num_queries": int(qcat.m),
        "num_vector_blocks": int(tvd_values.size),
        "full_true_mae": float(full_metrics["full_true_mae"]),
        "full_true_rmse": float(full_metrics["full_true_rmse"]),
        "full_true_max_error": float(full_metrics["full_true_max_error"]),
        "full_true_avg_tvd": float(np.mean(tvd_values)) if tvd_values.size else 0.0,
        "full_true_max_tvd": float(np.max(tvd_values)) if tvd_values.size else 0.0,
        "per_family": _per_family_metrics(qcat, true_answers, syn_answers, n_real, n_syn),
        "per_family_tvd": tvd_by_family,
        "runtime_seconds": run_metadata.get("runtime_seconds"),
        "run_metadata": run_metadata,
    }
    if cache_metadata is not None:
        output["true_answers_cache"] = cache_metadata
    if include_block_details:
        output["vector_block_tvd"] = block_details
    return output


def write_external_evaluation(
    input_dir: str | Path,
    synthetic_path: str | Path,
    output_path: str | Path,
    *,
    run_metadata_path: str | Path | None = None,
    batch_size: int = 8192,
    include_block_details: bool = True,
    true_answers_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = evaluate_external_synthetic(
        input_dir,
        synthetic_path,
        run_metadata_path=run_metadata_path,
        batch_size=batch_size,
        include_block_details=include_block_details,
        true_answers_cache_path=true_answers_cache_path,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(metrics, option=orjson.OPT_INDENT_2))
    return metrics
