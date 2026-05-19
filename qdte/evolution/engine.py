from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd

from qdte.config import save_yaml
from qdte.dataio import ensure_dir, save_npy, write_json
from qdte.eval.metrics import measured_loss, query_error_metrics
from qdte.eval.runtime import RuntimeStats
from qdte.evolution.candidates import generate_candidates
from qdte.evolution.initialization import initialize_independent_oneway
from qdte.evolution.gpu_candidates import (
    apply_edits_to_replicated_table,
    generate_and_score_candidates_gpu,
    replicate_table_to_devices,
)
from qdte.evolution.scheduler import select_active_queries
from qdte.evolution.scoring import compute_deltas, score_candidates, score_candidates_target_only
from qdte.evolution.state import QDTEState
from qdte.evolution.transport import (
    apply_edits,
    choose_transport_batch,
    choose_transport_batch_jax,
    select_top_nonconflicting,
)
from qdte.measurement.measure import measure_real_dataset
from qdte.preprocess import decode_array, load_and_preprocess_csv
from qdte.queries.eval_jax import answer_queries
from qdte.queries.workload import build_workload


def _resolve_n_syn(value: Any, n_real: int) -> int:
    if value is None or str(value) == "same_as_real":
        return int(n_real)
    return int(value)


def _write_timeseries(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "iteration",
                "wall_time",
                "measured_loss",
                "residual_l2",
                "residual_l1",
                "active_queries",
                "num_candidates",
                "positive_advantage_rate",
                "accepted_edits",
                "accepted_rate",
                "mean_advantage",
                "batch_advantage",
                "requested_candidates",
                "directed_candidates",
                "random_candidates",
                "candidate_shortfall",
                "source_filter_attempts",
                "source_filter_failures",
                "incremental_answer_drift",
            ]
        ).to_csv(path, index=False)


def run_qdte(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run", {})
    qdte_cfg = config.get("qdte", {})
    runtime_cfg = config.get("runtime", {})
    evaluation_cfg = config.get("evaluation", {})
    debug_cfg = config.get("debug", {})
    seed = int(run_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    output_dir = ensure_dir(run_cfg.get("output_dir", "outputs/qdte_run"))
    logs: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        logs.append(message)

    stats = RuntimeStats()
    save_yaml(config, output_dir / "config_resolved.yaml")
    log(f"Output dir: {output_dir}")
    log(f"JAX devices: {jax.devices()}")

    preprocess_result = load_and_preprocess_csv(config)
    X_real = preprocess_result.X
    schema = preprocess_result.schema
    n_real = int(X_real.shape[0])
    schema.save_json(output_dir / "schema.json")
    log(f"Loaded real data: rows={X_real.shape[0]}, cols={X_real.shape[1]}")

    qcat, workload_groups = build_workload(schema, config)
    qcat.save_json(output_dir / "queries.json")
    log(f"Constructed workload: queries={qcat.m}, groups={len(workload_groups)}")

    t0 = time.perf_counter()
    measurements = measure_real_dataset(
        X_real,
        qcat,
        workload_groups,
        config,
        rng,
        batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
    )
    stats.time_measurement_seconds = time.perf_counter() - t0
    write_json(measurements.to_public_dict(), output_dir / "measurements.json")
    if measurements.mode == "dp":
        log(
            "Real DP measurement performed on X_real: "
            f"rho_total={measurements.rho_total:.6g}, "
            f"epsilon(delta={measurements.delta:.2g})={measurements.epsilon_delta:.6g}, "
            f"measured_queries={qcat.m}, groups={len(measurements.groups)}"
        )
    else:
        log(
            "WARNING: oracle mode uses exact real query answers and is not differentially private. "
            "Do not use for paper DP results."
        )
    if bool(runtime_cfg.get("log_measurement_groups", True)):
        for group in measurements.groups:
            log(
                f"Measurement group {group.name}: family={group.family}, queries={len(group.query_indices)}, "
                f"sensitivity_l2={group.sensitivity_l2:.6g}, rho={group.rho:.6g}, noise_std={group.noise_std:.6g}"
            )

    t0 = time.perf_counter()
    n_syn = _resolve_n_syn(config.get("init", {}).get("N_syn", "same_as_real"), n_real)
    X_syn = initialize_independent_oneway(qcat, measurements.target_projected, schema, n_syn, rng)
    answer_syn = answer_queries(X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
    target = measurements.target_projected.astype(np.float32)
    residual = target - answer_syn
    variance = measurements.variances.astype(np.float32)
    inv_variance = measurements.inv_variances.astype(np.float32)
    sigma = np.sqrt(variance).astype(np.float32)
    state = QDTEState(
        X_syn=X_syn,
        answer_syn=answer_syn.astype(np.float32),
        target=target,
        residual=residual.astype(np.float32),
        variance=variance,
        inv_variance=inv_variance,
        sigma=sigma,
        debt=np.zeros(qcat.m, dtype=np.float32),
        iteration=0,
    )
    stats.time_init_seconds = time.perf_counter() - t0
    initial_loss = measured_loss(state.residual, state.inv_variance)
    log(f"Initial measured loss: {initial_loss:.6g}")

    X_real_for_evaluation = X_real if bool(evaluation_cfg.get("compute_true_query_error", True)) else None
    X_real = None
    preprocess_result = None

    max_iters = int(qdte_cfg.get("max_iters", 5000))
    accepted_per_iter = int(qdte_cfg.get("accepted_per_iter", 64))
    transport_mode = str(qdte_cfg.get("transport_mode", "microbatch_greedy"))
    transport_delta_backend = str(qdte_cfg.get("transport_delta_backend", "cpu"))
    if transport_mode == "sequential_greedy":
        accepted_per_iter = 1
    num_active_targets = int(qdte_cfg.get("num_active_targets", 64))
    kappa_noise = float(qdte_cfg.get("kappa_noise", 1.0))
    lambda_cost = float(qdte_cfg.get("lambda_cost", 0.01))
    min_advantage = float(qdte_cfg.get("min_advantage", 1.0e-6))
    transport_prefix_strategy = str(qdte_cfg.get("transport_prefix_strategy", "largest_positive"))
    stop_patience = int(qdte_cfg.get("stop_patience", 50))
    full_recompute_every = int(qdte_cfg.get("full_recompute_every", 50))
    log_every = int(qdte_cfg.get("log_every", 10))
    chunk_size = int(runtime_cfg.get("scoring_chunk_size", 4096))
    use_pmap = bool(runtime_cfg.get("use_pmap", True))
    score_backend = str(qdte_cfg.get("score_backend", "dense_gpu"))
    candidate_backend = str(qdte_cfg.get("candidate_backend", "cpu_repair"))
    debug_recompute_after_batch = bool(debug_cfg.get("recompute_after_batch", False))
    debug_assert_loss_decrease = bool(debug_cfg.get("assert_batch_loss_decrease", False))
    residual_drift_tolerance = float(debug_cfg.get("residual_drift_tolerance", 1.0e-5))
    loss_tolerance = float(debug_cfg.get("loss_tolerance", 1.0e-4))
    generation_start = time.perf_counter()
    patience = 0
    timeseries: list[dict[str, Any]] = []
    use_gpu_candidate_backend = candidate_backend in {"jax_repair", "gpu_repair"}
    X_syn_gpu = replicate_table_to_devices(state.X_syn) if use_gpu_candidate_backend else None

    for iteration in range(1, max_iters + 1):
        iter_start = time.perf_counter()
        state.iteration = iteration
        active = select_active_queries(
            state.residual,
            state.sigma,
            state.debt,
            num_active_targets=num_active_targets,
            kappa_noise=kappa_noise,
            debt_alpha=float(qdte_cfg.get("debt_alpha", 0.0)),
        )
        t_candidate = time.perf_counter()
        fused_advantages: np.ndarray | None = None
        if use_gpu_candidate_backend:
            if X_syn_gpu is None:
                raise RuntimeError("Internal error: GPU candidate backend requested but GPU table is not initialized.")
            gpu_batch = generate_and_score_candidates_gpu(
                X_syn_gpu,
                qcat,
                schema,
                active,
                state.residual,
                state.inv_variance,
                config,
                rng,
            )
            candidates = gpu_batch.candidates
            fused_advantages = gpu_batch.advantages
            stats.time_scoring_seconds += time.perf_counter() - t_candidate
        else:
            candidates = generate_candidates(state.X_syn, qcat, schema, active, state.residual, config, rng)
            stats.time_candidate_generation_seconds += time.perf_counter() - t_candidate
        diag = candidates.diagnostics
        stats.num_candidates_requested += int(diag.get("requested_candidates", candidates.size))
        stats.num_candidate_shortfall += int(diag.get("candidate_shortfall", 0.0))
        stats.num_directed_candidates += int(diag.get("directed_candidates", 0.0))
        stats.num_random_candidates += int(diag.get("random_candidates", 0.0))
        stats.num_source_filter_attempts += int(diag.get("source_filter_attempts", 0.0))
        stats.num_source_filter_failures += int(diag.get("source_filter_failures", 0.0))

        if candidates.size == 0:
            patience += 1
            if patience >= stop_patience:
                log(f"Stopping at iter={iteration}: no candidates for {patience} iterations")
                break
            continue

        if fused_advantages is None:
            t_score = time.perf_counter()
            if score_backend == "target_only":
                advantages = score_candidates_target_only(
                    candidates,
                    state.residual,
                    state.inv_variance,
                    qcat,
                    lambda_cost=lambda_cost,
                )
            else:
                advantages = score_candidates(
                    candidates,
                    state.residual,
                    state.inv_variance,
                    qcat,
                    lambda_cost=lambda_cost,
                    chunk_size=chunk_size,
                    use_pmap=use_pmap,
                )
            stats.time_scoring_seconds += time.perf_counter() - t_score
        else:
            advantages = fused_advantages
        stats.num_candidates_scored += int(diag.get("scored_candidates", candidates.size))
        positive_rate = float(np.mean(advantages > min_advantage)) if len(advantages) else 0.0

        t_transport = time.perf_counter()
        before_loss = measured_loss(state.residual, state.inv_variance)
        selected = select_top_nonconflicting(candidates, advantages, accepted_per_iter, min_advantage)
        if transport_delta_backend in {"jax_prefix", "gpu_prefix"}:
            transport = choose_transport_batch_jax(
                candidates,
                advantages,
                selected,
                state.residual,
                state.inv_variance,
                lambda_cost,
                qcat,
                prefix_strategy=transport_prefix_strategy,
            )
        else:
            if len(selected) > 0:
                deltas = compute_deltas(candidates.old_rows[selected], candidates.new_rows[selected], qcat)
            else:
                deltas = np.empty((0, qcat.m), dtype=np.int8)
            transport = choose_transport_batch(
                candidates,
                advantages,
                deltas,
                selected,
                state.residual,
                state.inv_variance,
                lambda_cost,
                prefix_strategy=transport_prefix_strategy,
            )
        if len(transport.accepted_indices) > 0:
            apply_edits(state.X_syn, candidates, transport.accepted_indices)
            if use_gpu_candidate_backend:
                if X_syn_gpu is None:
                    raise RuntimeError("Internal error: GPU candidate backend requested but GPU table is not initialized.")
                X_syn_gpu = apply_edits_to_replicated_table(
                    X_syn_gpu,
                    candidates.row_ids[transport.accepted_indices],
                    candidates.new_rows[transport.accepted_indices],
                )
            state.answer_syn = (state.answer_syn + transport.delta_sum).astype(np.float32)
            state.residual = (state.target - state.answer_syn).astype(np.float32)
            stats.num_accepted_edits += len(transport.accepted_indices)
            patience = 0
        else:
            patience += 1
        stats.time_transport_seconds += time.perf_counter() - t_transport
        debug_drift = 0.0

        if debug_recompute_after_batch and len(transport.accepted_indices) > 0:
            t_recompute = time.perf_counter()
            recomputed = answer_queries(state.X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
            debug_drift = float(np.max(np.abs(recomputed - state.answer_syn)))
            recomputed_residual = (state.target - recomputed).astype(np.float32)
            recomputed_loss = measured_loss(recomputed_residual, state.inv_variance)
            state.answer_syn = recomputed.astype(np.float32)
            state.residual = recomputed_residual
            stats.time_full_recompute_seconds += time.perf_counter() - t_recompute
            if debug_drift > residual_drift_tolerance:
                raise AssertionError(
                    f"Incremental answer drift {debug_drift} exceeds tolerance {residual_drift_tolerance}"
                )
            if debug_assert_loss_decrease and recomputed_loss > before_loss + loss_tolerance:
                raise AssertionError(
                    f"Accepted batch increased recomputed measured loss: before={before_loss}, after={recomputed_loss}"
                )

        if full_recompute_every > 0 and iteration % full_recompute_every == 0:
            t_recompute = time.perf_counter()
            recomputed = answer_queries(state.X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
            drift = float(np.max(np.abs(recomputed - state.answer_syn)))
            state.answer_syn = recomputed.astype(np.float32)
            state.residual = (state.target - state.answer_syn).astype(np.float32)
            stats.time_full_recompute_seconds += time.perf_counter() - t_recompute
            log(f"Full recompute iter={iteration}: max_incremental_drift={drift:.6g}")

        cur_loss = measured_loss(state.residual, state.inv_variance)
        if iteration == 1 or iteration % log_every == 0 or len(transport.accepted_indices) == 0:
            row = {
                "iteration": iteration,
                "wall_time": time.perf_counter() - stats.start_time,
                "measured_loss": cur_loss,
                "residual_l2": float(np.linalg.norm(state.residual)),
                "residual_l1": float(np.sum(np.abs(state.residual))),
                "active_queries": int(len(active)),
                "num_candidates": int(candidates.size),
                "positive_advantage_rate": positive_rate,
                "accepted_edits": int(len(transport.accepted_indices)),
                "accepted_rate": float(len(transport.accepted_indices) / max(1, candidates.size)),
                "mean_advantage": transport.mean_advantage,
                "batch_advantage": transport.batch_advantage,
                "requested_candidates": int(diag.get("requested_candidates", candidates.size)),
                "directed_candidates": int(diag.get("directed_candidates", 0.0)),
                "random_candidates": int(diag.get("random_candidates", 0.0)),
                "candidate_shortfall": int(diag.get("candidate_shortfall", 0.0)),
                "source_filter_attempts": int(diag.get("source_filter_attempts", 0.0)),
                "source_filter_failures": int(diag.get("source_filter_failures", 0.0)),
                "incremental_answer_drift": debug_drift,
            }
            timeseries.append(row)
            log(
                f"iter={iteration} loss={cur_loss:.6g} candidates={candidates.size} "
                f"positive={positive_rate:.3f} accepted={len(transport.accepted_indices)} "
                f"batch_adv={transport.batch_advantage:.6g}"
            )

        stats.num_iterations = iteration
        if patience >= stop_patience:
            log(f"Stopping at iter={iteration}: patience={patience}")
            break
        stats.time_generation_seconds += time.perf_counter() - iter_start

    stats.time_generation_seconds = time.perf_counter() - generation_start
    final_answers = answer_queries(state.X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
    final_incremental_answer_drift = float(np.max(np.abs(final_answers - state.answer_syn)))
    state.answer_syn = final_answers.astype(np.float32)
    state.residual = (state.target - state.answer_syn).astype(np.float32)
    final_loss = measured_loss(state.residual, state.inv_variance)
    log(f"Final measured loss: {final_loss:.6g}")
    log(f"Final incremental answer drift before recompute: {final_incremental_answer_drift:.6g}")
    log(f"Candidates scored: {stats.num_candidates_scored}")
    log(f"Accepted edits: {stats.num_accepted_edits}")

    save_npy(state.X_syn, output_dir / "synthetic_encoded.npy")
    if bool(evaluation_cfg.get("save_synthetic_csv", True)):
        decode_array(state.X_syn, schema).to_csv(output_dir / "synthetic_decoded.csv", index=False)

    final_metrics: dict[str, Any] = {
        "dataset_name": run_cfg.get("dataset_name"),
        "privacy_mode": measurements.mode,
        "rho_total": measurements.rho_total,
        "delta": measurements.delta,
        "epsilon_delta": measurements.epsilon_delta,
        "num_rows_real": n_real,
        "num_rows_synthetic": int(state.X_syn.shape[0]),
        "num_columns": int(schema.d),
        "num_queries": int(qcat.m),
        "num_measurement_groups": int(len(measurements.groups)),
        "initial_measured_loss": initial_loss,
        "final_measured_loss": final_loss,
        "loss_reduction": float(initial_loss - final_loss),
        "final_incremental_answer_drift": final_incremental_answer_drift,
        "num_candidates_scored": int(stats.num_candidates_scored),
        "num_candidates_requested": int(stats.num_candidates_requested),
        "num_candidate_shortfall": int(stats.num_candidate_shortfall),
        "num_accepted_edits": int(stats.num_accepted_edits),
        "gpu_device_count": int(jax.local_device_count()),
        "score_backend": score_backend,
        "candidate_backend": candidate_backend,
        "transport_delta_backend": transport_delta_backend,
        "transport_prefix_strategy": transport_prefix_strategy,
        "use_pmap": bool(use_pmap),
    }
    if bool(evaluation_cfg.get("compute_true_query_error", True)):
        if X_real_for_evaluation is None:
            raise RuntimeError("Internal error: true-query evaluation requested but real data reference was cleared.")
        log("Computing exact true query answers for post-run evaluation only.")
        true_answers = answer_queries(
            X_real_for_evaluation,
            qcat,
            batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
        )
        final_metrics.update(query_error_metrics(true_answers, final_answers, n_real, state.X_syn.shape[0]))
        log(
            "True query error for evaluation only: "
            f"MAE={final_metrics['true_query_mae']:.6g}, "
            f"RMSE={final_metrics['true_query_rmse']:.6g}"
        )

    runtime_dict = stats.as_dict()
    runtime_dict["gpu_devices"] = [str(d) for d in jax.devices()]
    runtime_dict["score_backend"] = score_backend
    runtime_dict["candidate_backend"] = candidate_backend
    runtime_dict["transport_delta_backend"] = transport_delta_backend
    runtime_dict["transport_prefix_strategy"] = transport_prefix_strategy
    runtime_dict["use_pmap"] = bool(use_pmap)
    write_json(final_metrics, output_dir / "metrics_final.json")
    _write_timeseries(timeseries, output_dir / "metrics_timeseries.csv")
    write_json(runtime_dict, output_dir / "runtime.json")
    (output_dir / "logs.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    return final_metrics
