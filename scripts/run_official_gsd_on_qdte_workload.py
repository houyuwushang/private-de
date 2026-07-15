#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from path_defaults import baseline_root
except ModuleNotFoundError:
    from scripts.path_defaults import baseline_root

BASELINE_ROOT = baseline_root()
PRIVATE_GSD_REPO = BASELINE_ROOT / "private-gsd"
if str(PRIVATE_GSD_REPO / "src") not in sys.path:
    sys.path.insert(0, str(PRIVATE_GSD_REPO / "src"))

import genetic_sd.generator.generator_genetic_sd as genetic_sd_module
from genetic_sd.generator.generator_genetic_sd import GeneticSD
from genetic_sd.utils import Dataset, Domain
from qdte.eval.gsd_progress import GSDProgressObserver
from scripts.gsd_released_target_utils import (
    normalized_precision_scale as _normalized_precision_scale,
    released_measurement_target as _released_measurement_target_from_file,
)

OP_EQ = 0
OP_LE = 1
OP_GE = 2
OP_RANGE = 3


class QueryData:
    def __init__(self, data: dict[str, Any]):
        self.m = int(data["m"])
        self.max_terms = int(data["max_terms"])
        self.attrs = np.asarray(data["attrs"], dtype=np.int32)
        self.ops = np.asarray(data["ops"], dtype=np.int32)
        self.values = np.asarray(data["values"], dtype=np.int32)
        self.lows = np.asarray(data["lows"], dtype=np.int32)
        self.highs = np.asarray(data["highs"], dtype=np.int32)
        self.num_terms = np.asarray(data["num_terms"], dtype=np.int32)
        self.linear_attrs = np.asarray(
            data.get("linear_attrs", [[-1] * self.max_terms] * self.m),
            dtype=np.int32,
        )
        self.linear_weights = np.asarray(
            data.get("linear_weights", [[0.0] * self.max_terms] * self.m),
            dtype=np.float32,
        )
        self.linear_thresholds = np.asarray(data.get("linear_thresholds", [0.0] * self.m), dtype=np.float32)
        self.linear_num_terms = np.asarray(data.get("linear_num_terms", [0] * self.m), dtype=np.int32)
        self.names = list(data["names"])
        self.groups = list(data["groups"])
        self.families = list(data["families"])

    def validate(self, cardinalities: np.ndarray) -> None:
        shape = (self.m, self.max_terms)
        for name in ("attrs", "ops", "values", "lows", "highs", "linear_attrs", "linear_weights"):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"queries.json {name} must have shape {shape}")
        if self.num_terms.shape != (self.m,) or self.linear_num_terms.shape != (self.m,):
            raise ValueError("queries.json term-count vectors have invalid shapes")
        if len(self.names) != self.m or len(self.groups) != self.m or len(self.families) != self.m:
            raise ValueError("queries.json metadata vectors must match the query count")
        if not np.all(np.isfinite(self.linear_weights)) or not np.all(np.isfinite(self.linear_thresholds)):
            raise ValueError("queries.json linear terms must be finite")
        valid_ops = {OP_EQ, OP_LE, OP_GE, OP_RANGE}
        for qid in range(self.m):
            num_terms = int(self.num_terms[qid])
            linear_num_terms = int(self.linear_num_terms[qid])
            if not 0 <= num_terms <= self.max_terms or not 0 <= linear_num_terms <= self.max_terms:
                raise ValueError(f"Query {qid} has invalid term counts")
            if num_terms + linear_num_terms == 0 or (num_terms > 0 and linear_num_terms > 0):
                raise ValueError(f"Query {qid} has an unsupported term layout")
            attrs = self.attrs[qid]
            if np.any(attrs[:num_terms] < 0) or np.any(attrs[num_terms:] != -1):
                raise ValueError(f"Query {qid} ordinary terms are not canonically packed")
            for term in range(num_terms):
                attr = int(attrs[term])
                op = int(self.ops[qid, term])
                if attr >= len(cardinalities) or op not in valid_ops:
                    raise ValueError(f"Query {qid} has an invalid attribute or operator")
                card = int(cardinalities[attr])
                if op == OP_RANGE:
                    lo = int(self.lows[qid, term])
                    hi = int(self.highs[qid, term])
                    if not 0 <= lo <= hi < card:
                        raise ValueError(f"Query {qid} has an out-of-domain range")
                elif not 0 <= int(self.values[qid, term]) < card:
                    raise ValueError(f"Query {qid} has an out-of-domain value")
            linear_attrs = self.linear_attrs[qid]
            if np.any(linear_attrs[:linear_num_terms] < 0) or np.any(linear_attrs[linear_num_terms:] != -1):
                raise ValueError(f"Query {qid} linear terms are not canonically packed")
            if np.any(linear_attrs[:linear_num_terms] >= len(cardinalities)):
                raise ValueError(f"Query {qid} has an out-of-domain linear attribute")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _parse_n(value: str, n_real: int) -> int:
    if value == "same_as_real":
        return int(n_real)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("row count must be positive or same_as_real")
    return parsed


def _schema_domain(schema: dict[str, Any]) -> Domain:
    config: dict[str, dict[str, Any]] = {}
    for column in schema["columns"]:
        config[str(column["name"])] = {"type": "string", "size": int(column["cardinality"])}
    return Domain(config)


def _schema_width(schema: dict[str, Any]) -> int:
    return int(len(schema["columns"]))


def _schema_cardinalities(schema: dict[str, Any]) -> np.ndarray:
    return np.asarray([int(column["cardinality"]) for column in schema["columns"]], dtype=np.int32)


def _validate_schema(schema: dict[str, Any]) -> None:
    columns = schema.get("columns", [])
    if not isinstance(columns, list) or not columns:
        raise ValueError("schema.json must contain a non-empty columns list")
    names = [str(column.get("name", "")) for column in columns]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("schema.json column names must be non-empty and unique")
    if np.any(_schema_cardinalities(schema) <= 0):
        raise ValueError("schema.json cardinalities must be positive")


def _schema_column_names(schema: dict[str, Any]) -> list[str]:
    return [str(column["name"]) for column in schema["columns"]]


def _validate_table(X: np.ndarray, schema: dict[str, Any], name: str) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2 or arr.shape[1] != _schema_width(schema):
        raise ValueError(f"{name} shape {arr.shape} is incompatible with schema width {_schema_width(schema)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be integer encoded, got {arr.dtype}")
    out = arr.astype(np.int32, copy=False)
    for attr, cardinality in enumerate(_schema_cardinalities(schema).tolist()):
        col = out[:, attr]
        if col.size and (int(col.min()) < 0 or int(col.max()) >= int(cardinality)):
            raise ValueError(f"{name} column {attr} has values outside [0,{int(cardinality)})")
    return out


def _family_counts(families: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for family in families:
        out[str(family)] = out.get(str(family), 0) + 1
    return out


def _make_query_rate_fn(qcat: QueryData):
    attrs = jnp.asarray(qcat.attrs, dtype=jnp.int32)
    ops = jnp.asarray(qcat.ops, dtype=jnp.int32)
    values = jnp.asarray(qcat.values, dtype=jnp.int32)
    lows = jnp.asarray(qcat.lows, dtype=jnp.int32)
    highs = jnp.asarray(qcat.highs, dtype=jnp.int32)
    linear_attrs = jnp.asarray(qcat.linear_attrs, dtype=jnp.int32)
    linear_weights = jnp.asarray(qcat.linear_weights, dtype=jnp.float32)
    linear_thresholds = jnp.asarray(qcat.linear_thresholds, dtype=jnp.float32)
    linear_num_terms = jnp.asarray(qcat.linear_num_terms, dtype=jnp.int32)
    max_terms = int(qcat.max_terms)
    num_queries = int(qcat.m)

    def row_answers(row: jax.Array) -> jax.Array:
        row_i = row.astype(jnp.int32)
        satisfied = jnp.ones((num_queries,), dtype=jnp.bool_)
        for term in range(max_terms):
            attr = attrs[:, term]
            valid = attr >= 0
            attr_clipped = jnp.maximum(attr, 0)
            x = row_i[attr_clipped]
            op = ops[:, term]
            cond_eq = x == values[:, term]
            cond_le = x <= values[:, term]
            cond_ge = x >= values[:, term]
            cond_range = (x >= lows[:, term]) & (x <= highs[:, term])
            cond = jnp.where(op == OP_EQ, cond_eq, cond_le)
            cond = jnp.where(op == OP_GE, cond_ge, cond)
            cond = jnp.where(op == OP_RANGE, cond_range, cond)
            satisfied = satisfied & jnp.where(valid, cond, True)

        linear_scores = jnp.zeros((num_queries,), dtype=jnp.float32)
        for term in range(max_terms):
            attr = linear_attrs[:, term]
            valid = attr >= 0
            attr_clipped = jnp.maximum(attr, 0)
            x = row_i[attr_clipped].astype(jnp.float32)
            linear_scores = linear_scores + jnp.where(valid, x * linear_weights[:, term], 0.0)
        linear_valid = linear_num_terms > 0
        linear_cond = linear_scores <= linear_thresholds
        satisfied = satisfied & jnp.where(linear_valid, linear_cond, True)
        return satisfied.astype(jnp.float32)

    def rate_fn(X: jax.Array) -> jax.Array:
        X_i = X.astype(jnp.int32)

        def scan_fn(carry: jax.Array, row: jax.Array):
            return carry + row_answers(row), None

        total = jax.lax.scan(scan_fn, jnp.zeros((num_queries,), dtype=jnp.float32), X_i)[0]
        return total / X_i.shape[0]

    return rate_fn


class QDTEQueryStatistics:
    def __init__(
        self,
        domain: Domain,
        qcat: QueryData,
        target_rates: np.ndarray,
        statistic_scale: np.ndarray | None = None,
    ):
        self.domain = domain
        self.qcat = qcat
        target = np.asarray(target_rates, dtype=np.float64)
        scale = np.ones_like(target) if statistic_scale is None else np.asarray(statistic_scale, dtype=np.float64)
        if target.shape != (qcat.m,) or scale.shape != (qcat.m,):
            raise ValueError("target_rates and statistic_scale must match the query count")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("target_rates and statistic_scale must be finite with positive scale")
        self.target_rates_unscaled = jnp.asarray(target, dtype=jnp.float32)
        self.statistic_scale = jnp.asarray(scale, dtype=jnp.float32)
        self.target_rates = self.target_rates_unscaled * self.statistic_scale
        self.unscaled_rate_fn = _make_query_rate_fn(qcat)

        def scaled_rate_fn(X: jax.Array) -> jax.Array:
            return self.unscaled_rate_fn(X) * self.statistic_scale

        self.rate_fn = scaled_rate_fn

    def get_selected_trimmed_statistics_fn(self, stat_modules_ids=None, max_queries: int = -1):
        del stat_modules_ids
        del max_queries
        return self.target_rates, self.target_rates, self.rate_fn

    def get_selected_noised_statistics(self):
        return self.target_rates

    def get_selected_statistics_without_noise(self):
        return self.target_rates

    def get_selected_statistics_fn(self):
        return self.rate_fn

    def get_dataset_statistics_fn(self):
        def data_fn(data: Dataset):
            return self.rate_fn(data.to_numpy())

        return data_fn

    def get_all_true_statistics(self):
        return self.target_rates


def _released_measurement_target(
    measurement_dir: Path,
    qcat: QueryData,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    return _released_measurement_target_from_file(measurement_dir, qcat.m)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    start = time.time()
    input_dir = Path(args.input_dir)
    measurement_dir = Path(args.measurement_dir).resolve() if args.measurement_dir is not None else None
    if args.query_dir is None and measurement_dir is None:
        raise ValueError("Pass --query-dir for an exact target or --measurement-dir for a released DP target")
    query_dir = Path(args.query_dir).resolve() if args.query_dir is not None else measurement_dir
    assert query_dir is not None
    schema = read_json(input_dir / "schema.json")
    _validate_schema(schema)
    qcat = QueryData(read_json(query_dir / "queries.json"))
    qcat.validate(_schema_cardinalities(schema))
    query_schema_path = query_dir / "schema.json"
    if query_schema_path.exists() and read_json(query_schema_path) != schema:
        raise ValueError("query_dir schema does not match input_dir schema")
    domain = _schema_domain(schema)
    column_names = _schema_column_names(schema)
    target_rate_fn = _make_query_rate_fn(qcat)
    if measurement_dir is not None:
        target_counts, variances, n_real, measurement = _released_measurement_target(measurement_dir, qcat)
        n_syn = _parse_n(str(args.n_syn), n_real)
        if n_syn != n_real:
            raise ValueError("released-target E5 requires n_syn to equal the public measurement row count")
        inv_variance, precision_normalizer, statistic_scale = _normalized_precision_scale(variances)
        target_rates = target_counts / float(n_real)
        privacy_mode = "dp_released_target"
        target_label = "released_target_projected_variance_weighted"
    else:
        X_real = _validate_table(np.load(input_dir / "real_encoded.npy"), schema, "real_encoded")
        if X_real.shape[0] <= 0:
            raise ValueError("real_encoded must contain at least one row")
        n_real = int(X_real.shape[0])
        n_syn = _parse_n(str(args.n_syn), n_real)
        true_rates_jax = target_rate_fn(jnp.asarray(X_real, dtype=jnp.int32)).block_until_ready()
        target_rates = np.asarray(true_rates_jax, dtype=np.float64)
        target_counts = target_rates * float(n_real)
        inv_variance = np.ones(qcat.m, dtype=np.float64)
        precision_normalizer = 1.0
        statistic_scale = np.ones(qcat.m, dtype=np.float64)
        measurement = None
        privacy_mode = "non_dp_diagnostic"
        target_label = "qdte_mixed_workload_exact_no_noise_rates"
    stats = QDTEQueryStatistics(
        domain,
        qcat,
        target_rates,
        statistic_scale=statistic_scale,
    )
    if int(args.seed) < 0:
        raise ValueError("seed must be non-negative")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if int(args.num_generations) <= 0:
        raise ValueError("num_generations must be positive")
    if int(args.stop_early_min_generation) <= 0:
        raise ValueError("stop_early_min_generation must be positive")
    if not np.isfinite(float(args.early_stop_threshold)) or float(args.early_stop_threshold) < 0.0:
        raise ValueError("early_stop_threshold must be finite and non-negative")
    if int(args.plateau_min_generation) < 0:
        raise ValueError("plateau_min_generation must be non-negative")
    if int(args.plateau_checkpoints) < 0:
        raise ValueError("plateau_checkpoints must be non-negative")
    if (
        not np.isfinite(float(args.plateau_relative_improvement))
        or float(args.plateau_relative_improvement) < 0.0
    ):
        raise ValueError("plateau_relative_improvement must be finite and non-negative")
    operators = tuple(item.strip() for item in str(args.genetic_operators).split(",") if item.strip())
    if not operators:
        raise ValueError("At least one genetic operator is required")
    unknown_operators = sorted(set(operators) - {"mutate", "continuous", "swap", "cross"})
    if unknown_operators:
        raise ValueError(f"Unknown official GSD genetic operators: {unknown_operators}")
    initial_dataset = None
    initial_synthetic_path = Path(args.initial_synthetic).resolve() if args.initial_synthetic else None
    if initial_synthetic_path is not None:
        initial_synthetic = _validate_table(
            np.load(initial_synthetic_path),
            schema,
            "initial_synthetic",
        )
        if int(initial_synthetic.shape[0]) != n_syn:
            raise ValueError(
                "initial_synthetic row count does not match requested synthetic size: "
                f"{initial_synthetic.shape[0]} != {n_syn}"
            )
        initial_dataset = Dataset(
            pd.DataFrame(initial_synthetic.astype(np.int64), columns=column_names),
            domain,
        )
        np.save(output_dir / "synthetic_initial_encoded.npy", initial_synthetic)
    key = jax.random.PRNGKey(int(args.seed))
    generator = GeneticSD(
        domain=domain,
        data_size=n_syn,
        num_generations=int(args.num_generations),
        genetic_operators=operators,
        print_progress=bool(args.verbose),
        stop_early=True,
        stop_early_min_generation=int(args.stop_early_min_generation),
        stop_eary_threshold=float(args.early_stop_threshold),
        sparse_statistics=True,
    )
    observer = GSDProgressObserver(
        n_synthetic=n_syn,
        started_at=start,
        min_generation=int(args.plateau_min_generation),
        relative_improvement_threshold=float(args.plateau_relative_improvement),
        plateau_checkpoints=int(args.plateau_checkpoints),
    )
    original_print_progress = genetic_sd_module.print_progress_fn
    original_check_early_stop = genetic_sd_module.check_early_stop

    def observed_print_progress(
        generation: int,
        best_fitness: Any,
        previous_fitness: Any,
        strategy_weights: Any,
        loop_time_seconds: float,
        print_progress: bool = False,
    ) -> None:
        observer.record(
            generation,
            best_fitness,
            previous_fitness,
            strategy_weights,
            loop_time_seconds,
        )
        original_print_progress(
            generation,
            best_fitness,
            previous_fitness,
            strategy_weights,
            loop_time_seconds,
            print_progress,
        )

    def observed_check_early_stop(
        generation: int,
        best_fitness: Any,
        previous_fitness: Any,
        stop_early_min_generation: int,
        early_stop_threshold: float,
        print_progress: bool = False,
    ) -> bool:
        del stop_early_min_generation, early_stop_threshold
        should_stop = observer.should_stop(generation, best_fitness, previous_fitness)
        if should_stop and print_progress:
            print(
                "\t\t ### Stop at predeclared plateau "
                f"after {observer.stop_generation} generations ###"
            )
        return should_stop

    genetic_sd_module.print_progress_fn = observed_print_progress
    if int(args.plateau_checkpoints) > 0:
        genetic_sd_module.check_early_stop = observed_check_early_stop
    try:
        sync_data = generator.fit(key, stats, sync_dataset=initial_dataset)
    finally:
        genetic_sd_module.print_progress_fn = original_print_progress
        genetic_sd_module.check_early_stop = original_check_early_stop
    synthetic = np.asarray(sync_data.to_numpy_np(), dtype=np.int32)
    synthetic = _validate_table(synthetic, schema, "synthetic")
    np.save(output_dir / "synthetic_encoded.npy", synthetic)
    shutil.copyfile(input_dir / "schema.json", output_dir / "schema.json")
    shutil.copyfile(query_dir / "queries.json", output_dir / "queries.json")
    if measurement_dir is not None:
        shutil.copyfile(measurement_dir / "measurements.json", output_dir / "measurements.json")

    syn_rates = np.asarray(target_rate_fn(jnp.asarray(synthetic, dtype=jnp.int32)).block_until_ready(), dtype=np.float64)
    syn_counts = syn_rates * float(synthetic.shape[0])
    rate_error = syn_rates - target_rates
    normalized_weighted_rate_l2_squared = float(np.sum(rate_error * rate_error * statistic_scale * statistic_scale))
    target_count_loss = float(0.5 * np.sum((syn_counts - target_counts) ** 2 * inv_variance))
    actual_generations = int(observer.stop_generation or args.num_generations)
    stop_reason = "plateau" if observer.stop_generation is not None else "generation_cap"
    observer.append_final(
        actual_generations=actual_generations,
        rate_l2_squared=normalized_weighted_rate_l2_squared,
        target_count_loss=target_count_loss,
    )
    observer.write_csv(output_dir / "metrics_timeseries.csv")
    metrics: dict[str, Any] = {
        "method": str(args.method_label),
        "generator": "official_private_gsd_GeneticSD",
        "target": target_label,
        "privacy_mode": privacy_mode,
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "input_dir": str(input_dir),
        "query_dir": str(query_dir),
        "measurement_dir": str(measurement_dir) if measurement_dir is not None else None,
        "measurement_reused": measurement_dir is not None,
        "num_queries": int(qcat.m),
        "queries_by_family": _family_counts(qcat.families),
        "n_real": int(n_real),
        "n_synthetic": int(synthetic.shape[0]),
        "num_generations": int(args.num_generations),
        "actual_generations": actual_generations,
        "stop_reason": stop_reason,
        "stop_early_min_generation": int(args.stop_early_min_generation),
        "early_stop_threshold": float(args.early_stop_threshold),
        "plateau_min_generation": int(args.plateau_min_generation),
        "plateau_relative_improvement": float(args.plateau_relative_improvement),
        "plateau_checkpoints": int(args.plateau_checkpoints),
        "progress_checkpoint_interval": int(max(100, n_syn)),
        "initial_synthetic": str(initial_synthetic_path) if initial_synthetic_path is not None else None,
        "genetic_operators": [item.strip() for item in str(args.genetic_operators).split(",") if item.strip()],
        "target_rate_mae": float(np.mean(np.abs(rate_error))),
        "target_rate_rmse": float(np.sqrt(np.mean(rate_error * rate_error))),
        "target_rate_max_error": float(np.max(np.abs(rate_error))),
        "normalized_weighted_rate_l2_squared": normalized_weighted_rate_l2_squared,
        "precision_normalizer": float(precision_normalizer),
        "objective_equivalence": (
            "official GSD fitness equals the QDTE variance-weighted quadratic objective "
            "up to one positive global scale"
            if measurement_dir is not None
            else "unweighted exact-rate L2"
        ),
        "target_count_loss": target_count_loss,
        "runtime_seconds": float(time.time() - start),
    }
    write_json(metrics, output_dir / "metrics_final.json")
    metadata = {
        "method": str(args.method_label),
        "source_repository": str(PRIVATE_GSD_REPO),
        "commit": _git_commit(PRIVATE_GSD_REPO),
        "command": " ".join(sys.argv),
        "metrics": metrics,
        "notes": {
            "official_search_code": "genetic_sd.generator.generator_genetic_sd.GeneticSD",
            "custom_statistics_adapter": (
                "QDTEQueryStatistics supplies the released projected target and sqrt-normalized precision"
                if measurement_dir is not None
                else "QDTEQueryStatistics supplies fixed exact/no-noise SAGE/QDTE query rates"
            ),
            "observer_only_patch": (
                "module-level progress and early-stop callbacks are observed/replaced; "
                "official initialization, proposals, scoring, selection, and updates are unchanged"
            ),
            "reproducibility_seeds": "JAX, Python random, and NumPy are all seeded from --seed",
            "non_dp_diagnostic": measurement_dir is None,
            "dp_boundary": (
                "generation reads only released measurements, public query/schema metadata, and the fixed synthetic initialization"
                if measurement_dir is not None
                else "exact true answers define the explicitly non-DP diagnostic target"
            ),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    write_json(metadata, output_dir / "run_metadata.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official Private-GSD GeneticSD on a QDTE mixed query workload.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--query-dir", type=Path)
    parser.add_argument(
        "--measurement-dir",
        type=Path,
        help="Released DP target/variance artifact. When set, no exact real answer is loaded for generation.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--external-input-dir", type=Path, help="Accepted for command compatibility; evaluated separately.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-syn", default="same_as_real")
    parser.add_argument("--num-generations", type=int, default=200000)
    parser.add_argument("--stop-early-min-generation", type=int, default=200000)
    parser.add_argument("--early-stop-threshold", type=float, default=0.01)
    parser.add_argument(
        "--plateau-min-generation",
        type=int,
        default=0,
        help="Do not apply the observer plateau rule before this generation.",
    )
    parser.add_argument(
        "--plateau-relative-improvement",
        type=float,
        default=0.0,
        help="Per-checkpoint relative improvement below which a checkpoint counts as plateaued.",
    )
    parser.add_argument(
        "--plateau-checkpoints",
        type=int,
        default=0,
        help="Consecutive plateau checkpoints required to stop; zero preserves upstream stopping.",
    )
    parser.add_argument(
        "--initial-synthetic",
        type=Path,
        help="Optional encoded table passed through GeneticSD's supported sync_dataset warm start.",
    )
    parser.add_argument("--genetic-operators", default="mutate,swap,cross")
    parser.add_argument("--answer-batch-size", type=int, default=8192)
    parser.add_argument("--method-label", default="official_gsd_qdte_mixed_no_noise")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
