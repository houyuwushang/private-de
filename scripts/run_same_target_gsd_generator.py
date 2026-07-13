#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import apply_overrides, load_yaml
from qdte.dataio import ensure_dir, read_json, write_json
from qdte.eval.external import write_external_evaluation
from qdte.eval.metrics import measured_loss, rms_standardized_residual, rms_unweighted_residual, unweighted_measured_loss
from qdte.evolution.initialization import initialize_independent_oneway, split_run_rng_streams
from qdte.measurement.measure import measurements_from_public_dict
from qdte.preprocess import load_and_preprocess_csv
from qdte.queries.eval_jax import answer_queries, eval_records_queries
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema


def _resolve_n_syn(value: Any, n_real: int) -> int:
    if value is None or str(value) == "same_as_real":
        return int(n_real)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("init.N_syn must be positive or same_as_real")
    return parsed


def _validate_table(X: np.ndarray, schema: TableSchema, name: str) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2 or arr.shape[1] != schema.d:
        raise ValueError(f"{name} shape {arr.shape} is incompatible with schema width {schema.d}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be integer encoded, got {arr.dtype}")
    out = arr.astype(np.int32, copy=False)
    for attr, card in enumerate(schema.cardinalities.tolist()):
        col = out[:, attr]
        if col.size and (int(col.min()) < 0 or int(col.max()) >= int(card)):
            raise ValueError(f"{name} column {attr} has values outside [0,{int(card)})")
    return out


def _load_context(config: dict[str, Any], measurement_dir: Path) -> tuple[TableSchema, QueryCatalogue, Any]:
    del config
    schema = TableSchema.load_json(measurement_dir / "schema.json")
    qcat = QueryCatalogue.from_dict(read_json(measurement_dir / "queries.json"))
    schema.validate()
    qcat.validate(schema.cardinalities)
    measurements = measurements_from_public_dict(read_json(measurement_dir / "measurements.json"))
    if measurements.target_projected.shape[0] != qcat.m:
        raise ValueError(
            "measurement target length mismatch: "
            f"{measurements.target_projected.shape[0]} target values for {qcat.m} queries"
        )
    if str(measurements.mode).lower() == "dp":
        # Generation below only uses these released values and variances.
        pass
    return schema, qcat, measurements


def _init_synthetic(
    *,
    config: dict[str, Any],
    schema: TableSchema,
    qcat: QueryCatalogue,
    target: np.ndarray,
    n_real: int,
    rng: np.random.Generator,
    init_mode: str,
) -> np.ndarray:
    n_syn = _resolve_n_syn(config.get("init", {}).get("N_syn", "same_as_real"), n_real)
    if init_mode == "independent_oneway":
        return initialize_independent_oneway(qcat, target, schema, n_syn, rng)
    if init_mode == "random":
        cols = [rng.integers(0, int(card), size=n_syn, dtype=np.int32) for card in schema.cardinalities.tolist()]
        return np.stack(cols, axis=1).astype(np.int32, copy=False)
    raise ValueError("init_mode must be one of: independent_oneway, random")


def _attr_weights(cardinalities: np.ndarray) -> np.ndarray:
    mutable = cardinalities.astype(np.float64) > 1
    weights = np.where(mutable, np.maximum(np.log(np.maximum(cardinalities.astype(np.float64), 2.0)), 1.0), 0.0)
    if float(weights.sum()) <= 0.0:
        return np.ones_like(weights, dtype=np.float64) / max(1, weights.size)
    return weights / float(weights.sum())


def _random_new_values(old_values: np.ndarray, cards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = old_values.copy()
    for card in np.unique(cards).tolist():
        card = int(card)
        mask = cards == card
        if card <= 1 or not np.any(mask):
            continue
        vals = rng.integers(0, card - 1, size=int(mask.sum()), dtype=np.int32)
        old = old_values[mask].astype(np.int32)
        out[mask] = vals + (vals >= old)
    return out.astype(np.int32, copy=False)


def _mutate_proposals(
    X: np.ndarray,
    cardinalities: np.ndarray,
    population_size: int,
    attr_prob: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    row_ids = rng.integers(0, n, size=(population_size, 1), dtype=np.int32)
    old_rows = X[row_ids[:, 0]].reshape(population_size, 1, d).copy()
    new_rows = old_rows.copy()
    attrs = rng.choice(d, size=population_size, p=attr_prob).astype(np.int32)
    cards = cardinalities[attrs]
    rows = np.arange(population_size, dtype=np.int32)
    old_vals = new_rows[rows, 0, attrs].copy()
    new_rows[rows, 0, attrs] = _random_new_values(old_vals, cards, rng)
    row_counts = np.ones(population_size, dtype=np.int32)
    edit_cost = np.ones(population_size, dtype=np.float32)
    return row_ids, old_rows, new_rows, row_counts, edit_cost


def _swap_proposals(
    X: np.ndarray,
    cardinalities: np.ndarray,
    population_size: int,
    attr_prob: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    row_ids = rng.integers(0, n, size=(population_size, 2), dtype=np.int32)
    if n > 1:
        same = row_ids[:, 0] == row_ids[:, 1]
        row_ids[same, 1] = (row_ids[same, 1] + 1) % n
    old_rows = X[row_ids].copy()
    new_rows = old_rows.copy()
    attrs = rng.choice(d, size=population_size, p=attr_prob).astype(np.int32)
    for idx, attr in enumerate(attrs.tolist()):
        if int(cardinalities[attr]) <= 1:
            continue
        left = new_rows[idx, 0, attr].copy()
        new_rows[idx, 0, attr] = new_rows[idx, 1, attr]
        new_rows[idx, 1, attr] = left
    row_counts = np.full(population_size, 2, dtype=np.int32)
    edit_cost = np.sum(np.any(old_rows != new_rows, axis=2), axis=1).astype(np.float32)
    return row_ids, old_rows, new_rows, row_counts, edit_cost


def _cross_proposals(
    X: np.ndarray,
    schema: TableSchema,
    population_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    categorical = np.asarray(schema.categorical_indices or list(range(d)), dtype=np.int32)
    if categorical.size == 0:
        categorical = np.arange(d, dtype=np.int32)
    row_ids = rng.integers(0, n, size=(population_size, 1), dtype=np.int32)
    donor_ids = rng.integers(0, n, size=population_size, dtype=np.int32)
    old_rows = X[row_ids[:, 0]].reshape(population_size, 1, d).copy()
    new_rows = old_rows.copy()
    donor_rows = X[donor_ids]
    max_updates = 2 if categorical.size > 1 else 1
    update_counts = rng.integers(1, max_updates + 1, size=population_size, dtype=np.int32)
    for idx, count in enumerate(update_counts.tolist()):
        attrs = rng.choice(categorical, size=int(count), replace=False)
        new_rows[idx, 0, attrs] = donor_rows[idx, attrs]
    row_counts = np.ones(population_size, dtype=np.int32)
    edit_cost = np.any(old_rows != new_rows, axis=2).astype(np.float32).reshape(population_size)
    return row_ids, old_rows, new_rows, row_counts, edit_cost


def _proposal_deltas(old_rows: np.ndarray, new_rows: np.ndarray, qcat: QueryCatalogue) -> np.ndarray:
    c, rmax, d = old_rows.shape
    old_flat = old_rows.reshape(c * rmax, d)
    new_flat = new_rows.reshape(c * rmax, d)
    old_phi = np.asarray(eval_records_queries(old_flat, qcat), dtype=np.float32).reshape(c, rmax, qcat.m)
    new_phi = np.asarray(eval_records_queries(new_flat, qcat), dtype=np.float32).reshape(c, rmax, qcat.m)
    return np.sum(new_phi - old_phi, axis=1, dtype=np.float32)


def _score_proposals(
    *,
    old_rows: np.ndarray,
    new_rows: np.ndarray,
    row_counts: np.ndarray,
    edit_cost: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    qcat: QueryCatalogue,
    lambda_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    del row_counts
    deltas = _proposal_deltas(old_rows, new_rows, qcat)
    weighted_residual = residual.astype(np.float32) * inv_variance.astype(np.float32)
    linear_gain = deltas @ weighted_residual
    quad_penalty = 0.5 * ((deltas * deltas) @ inv_variance.astype(np.float32))
    advantages = linear_gain - quad_penalty - float(lambda_cost) * edit_cost.astype(np.float32)
    return advantages.astype(np.float32, copy=False), deltas


def _apply_proposal(
    X: np.ndarray,
    answer_syn: np.ndarray,
    residual: np.ndarray,
    *,
    candidate_idx: int,
    row_ids: np.ndarray,
    new_rows: np.ndarray,
    row_counts: np.ndarray,
    delta: np.ndarray,
) -> None:
    count = int(row_counts[candidate_idx])
    for local_idx in range(count):
        X[int(row_ids[candidate_idx, local_idx])] = new_rows[candidate_idx, local_idx]
    answer_syn += delta
    residual -= delta


def _operators_from_text(text: str) -> tuple[str, ...]:
    operators = tuple(item.strip().lower() for item in text.split(",") if item.strip())
    allowed = {"mutate", "swap", "cross"}
    unknown = sorted(set(operators) - allowed)
    if unknown:
        raise ValueError(f"Unknown GSD-style operators: {unknown}")
    if not operators:
        raise ValueError("At least one operator is required")
    return operators


def _write_timeseries(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    measurement_dir = Path(args.measurement_dir)
    schema, qcat, measurements = _load_context(config, measurement_dir)
    preprocess_result = load_and_preprocess_csv(config)
    X_real = _validate_table(preprocess_result.X, schema, "X_real")
    if preprocess_result.schema.to_dict() != schema.to_dict():
        raise ValueError("Preprocessed raw data schema does not match measurement schema")

    qdte_cfg = config.get("qdte", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    seed = int(args.seed if args.seed is not None else config.get("run", {}).get("seed", 0))
    if seed < 0:
        raise ValueError("seed must be non-negative")
    _, rng = split_run_rng_streams(seed)
    n_real = int(X_real.shape[0])
    if measurements.num_rows is not None and int(measurements.num_rows) != n_real:
        raise ValueError(
            f"measurement row count {measurements.num_rows} does not match real data row count {n_real}"
        )
    target = measurements.target_projected.astype(np.float32)
    inv_variance = measurements.inv_variances.astype(np.float32)
    if str(qdte_cfg.get("objective_weighting", "variance")).lower() == "unweighted":
        inv_variance = np.ones_like(inv_variance, dtype=np.float32)
    elif str(qdte_cfg.get("objective_weighting", "variance")).lower() != "variance":
        raise ValueError("This diagnostic supports qdte.objective_weighting in {variance, unweighted}")

    X_syn = _init_synthetic(
        config=config,
        schema=schema,
        qcat=qcat,
        target=target,
        n_real=n_real,
        rng=rng,
        init_mode=str(args.init_mode),
    )
    initial_X_syn = X_syn.copy()
    answer_batch_size = int(runtime_cfg.get("answer_batch_size", 8192))
    answer_syn = answer_queries(X_syn, qcat, batch_size=answer_batch_size).astype(np.float32)
    residual = (target - answer_syn).astype(np.float32)
    initial_answers = answer_syn.copy()
    initial_loss = measured_loss(residual, inv_variance)
    initial_unweighted = unweighted_measured_loss(residual)

    operators = _operators_from_text(str(args.operators))
    attr_prob = _attr_weights(schema.cardinalities)
    max_generations = int(args.max_generations if args.max_generations is not None else qdte_cfg.get("max_iters", 5000))
    population_size = int(args.population_size)
    min_advantage = float(args.min_advantage if args.min_advantage is not None else qdte_cfg.get("min_advantage", 1.0e-6))
    lambda_cost = float(args.lambda_cost if args.lambda_cost is not None else qdte_cfg.get("lambda_cost", 0.01))
    log_every = int(args.log_every if args.log_every is not None else qdte_cfg.get("log_every", 100))
    if max_generations < 0:
        raise ValueError("max_generations must be non-negative")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not np.isfinite(min_advantage) or min_advantage < 0.0:
        raise ValueError("min_advantage must be finite and non-negative")
    if not np.isfinite(lambda_cost) or lambda_cost < 0.0:
        raise ValueError("lambda_cost must be finite and non-negative")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    start = time.time()
    timeseries: list[dict[str, Any]] = []
    accepted = 0
    candidates_scored = 0
    best_advantage = 0.0
    operator_counts = {op: 0 for op in operators}
    operator_accepts = {op: 0 for op in operators}

    for generation in range(1, max_generations + 1):
        op = str(rng.choice(operators))
        operator_counts[op] += 1
        if op == "mutate":
            row_ids, old_rows, new_rows, row_counts, edit_cost = _mutate_proposals(
                X_syn, schema.cardinalities, population_size, attr_prob, rng
            )
        elif op == "swap":
            row_ids, old_rows, new_rows, row_counts, edit_cost = _swap_proposals(
                X_syn, schema.cardinalities, population_size, attr_prob, rng
            )
        else:
            row_ids, old_rows, new_rows, row_counts, edit_cost = _cross_proposals(X_syn, schema, population_size, rng)

        advantages, deltas = _score_proposals(
            old_rows=old_rows,
            new_rows=new_rows,
            row_counts=row_counts,
            edit_cost=edit_cost,
            residual=residual,
            inv_variance=inv_variance,
            qcat=qcat,
            lambda_cost=lambda_cost,
        )
        candidates_scored += int(population_size)
        best_idx = int(np.argmax(advantages))
        best_advantage = float(advantages[best_idx])
        if best_advantage > min_advantage:
            _apply_proposal(
                X_syn,
                answer_syn,
                residual,
                candidate_idx=best_idx,
                row_ids=row_ids,
                new_rows=new_rows,
                row_counts=row_counts,
                delta=deltas[best_idx],
            )
            accepted += 1
            operator_accepts[op] += 1

        if generation == 1 or generation % log_every == 0 or generation == max_generations:
            cur_loss = measured_loss(residual, inv_variance)
            cur_unweighted = unweighted_measured_loss(residual)
            row = {
                "generation": int(generation),
                "wall_time": float(time.time() - start),
                "operator": op,
                "measured_loss": cur_loss,
                "unweighted_measured_loss": cur_unweighted,
                "rms_standardized_residual": rms_standardized_residual(cur_loss, qcat.m),
                "rms_unweighted_residual": rms_unweighted_residual(residual),
                "best_advantage": best_advantage,
                "accepted_total": int(accepted),
                "candidates_scored": int(candidates_scored),
            }
            timeseries.append(row)
            print(
                f"gen={generation} op={op} loss={cur_loss:.6g} "
                f"unweighted={cur_unweighted:.6g} best_adv={best_advantage:.6g} accepted={accepted}",
                flush=True,
            )

    final_answers = answer_queries(X_syn, qcat, batch_size=answer_batch_size).astype(np.float32)
    drift = float(np.max(np.abs(final_answers - answer_syn))) if qcat.m else 0.0
    drift_tolerance = float(config.get("debug", {}).get("residual_drift_tolerance", 1.0e-5))
    if drift > drift_tolerance:
        raise AssertionError(f"Incremental answer drift {drift} exceeds tolerance {drift_tolerance}")
    answer_syn = final_answers
    residual = (target - answer_syn).astype(np.float32)
    final_loss = measured_loss(residual, inv_variance)
    final_unweighted = unweighted_measured_loss(residual)
    np.save(output_dir / "synthetic_encoded.npy", X_syn.astype(np.int32, copy=False))
    np.save(output_dir / "synthetic_initial_encoded.npy", initial_X_syn.astype(np.int32, copy=False))
    schema.save_json(output_dir / "schema.json")
    qcat.save_json(output_dir / "queries.json")
    write_json(measurements.to_public_dict(), output_dir / "measurements.json")
    _write_timeseries(output_dir / "metrics_timeseries.csv", timeseries)

    metrics: dict[str, Any] = {
        "method": str(args.method_label),
        "generator": "same_target_gsd_style",
        "measurement_reused": True,
        "measurement_reuse_from": str(measurement_dir),
        "privacy_mode": str(measurements.mode),
        "seed": int(seed),
        "num_queries": int(qcat.m),
        "n_real": int(n_real),
        "n_synthetic": int(X_syn.shape[0]),
        "objective_weighting": str(qdte_cfg.get("objective_weighting", "variance")).lower(),
        "objective_loss": "quadratic",
        "initial_measured_loss": float(initial_loss),
        "final_measured_loss": float(final_loss),
        "loss_reduction": float(initial_loss - final_loss),
        "initial_unweighted_measured_loss": float(initial_unweighted),
        "final_unweighted_measured_loss": float(final_unweighted),
        "unweighted_measured_loss_reduction": float(initial_unweighted - final_unweighted),
        "rms_standardized_residual": rms_standardized_residual(final_loss, qcat.m),
        "rms_unweighted_residual": rms_unweighted_residual(residual),
        "num_generations": int(max_generations),
        "population_size": int(population_size),
        "num_candidates_scored": int(candidates_scored),
        "num_accepted_edits": int(accepted),
        "operators": list(operators),
        "operator_counts": operator_counts,
        "operator_accepts": operator_accepts,
        "lambda_cost": float(lambda_cost),
        "min_advantage": float(min_advantage),
        "runtime_seconds": float(time.time() - start),
        "max_incremental_answer_drift": drift,
        "init_mode": str(args.init_mode),
    }

    if bool(config.get("evaluation", {}).get("compute_true_query_error", True)):
        true_answers = answer_queries(X_real, qcat, batch_size=answer_batch_size).astype(np.float32)
        from qdte.eval.metrics import query_error_metrics

        metrics.update(query_error_metrics(true_answers, initial_answers, n_real, X_syn.shape[0], prefix="initial_true_query"))
        metrics.update(query_error_metrics(true_answers, final_answers, n_real, X_syn.shape[0], prefix="final_true_query"))
        metrics["true_query_mae_reduction"] = float(
            metrics["initial_true_query_mae"] - metrics["final_true_query_mae"]
        )
        metrics["true_query_rmse_reduction"] = float(
            metrics["initial_true_query_rmse"] - metrics["final_true_query_rmse"]
        )

    write_json(metrics, output_dir / "metrics_final.json")
    run_metadata = {
        "method": str(args.method_label),
        "config": config,
        "measurement_dir": str(measurement_dir),
        "output_dir": str(output_dir),
        "runtime_seconds": metrics["runtime_seconds"],
        "dp_boundary": "generation uses only released target_projected and inv_variances; true answers are offline evaluation only",
    }
    write_json(run_metadata, output_dir / "run_metadata.json")

    if args.external_input_dir is not None:
        external_metrics = write_external_evaluation(
            args.external_input_dir,
            output_dir / "synthetic_encoded.npy",
            output_dir / "evaluation_external.json",
            run_metadata_path=output_dir / "run_metadata.json",
            batch_size=answer_batch_size,
            include_block_details=False,
            true_answers_cache_path=Path(args.external_input_dir) / "true_answers_cache.npz",
        )
        metrics.update(
            {
                "final_avg_tvd": float(external_metrics["full_true_avg_tvd"]),
                "final_max_tvd": float(external_metrics["full_true_max_tvd"]),
                "external_full_true_mae": float(external_metrics["full_true_mae"]),
                "external_full_true_rmse": float(external_metrics["full_true_rmse"]),
                "external_full_true_max_error": float(external_metrics["full_true_max_error"]),
            }
        )
        write_json(metrics, output_dir / "metrics_final.json")

    print(f"Final measured loss: {final_loss:.6g}", flush=True)
    print(f"Final unweighted measured loss: {final_unweighted:.6g}", flush=True)
    print(f"Candidates scored: {candidates_scored}", flush=True)
    print(f"Accepted edits: {accepted}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a same-target GSD-style generator against QDTE measurements.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--measurement-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--external-input-dir", type=Path)
    parser.add_argument("--method-label", default="same_target_gsd_style50")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--operators", default="mutate,swap,cross")
    parser.add_argument("--init-mode", choices=["independent_oneway", "random"], default="independent_oneway")
    parser.add_argument("--lambda-cost", type=float, default=None)
    parser.add_argument("--min-advantage", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    args, overrides = parser.parse_known_args()
    config = apply_overrides(load_yaml(args.config), overrides)
    run(args, config)


if __name__ == "__main__":
    main()
