#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, set_nested
from qdte.dataio import ensure_dir, write_json
from qdte.evolution.engine import run_qdte
from qdte.measurement.factorization import (
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
    interaction_transcript_to_local_polytope_bootdiag_measurements,
    interaction_transcript_to_local_polytope_measurements,
    interaction_transcript_to_shrunk_measurements,
)
from qdte.measurement.measure import MeasurementGroup, Measurements
from qdte.measurement.projection import project_simplex
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup
from qdte.schema import TableSchema
from scripts.run_orthogonal_low_budget_pilot import build_complete_low_order_workload
from scripts.run_static_ice_measurement_pilot import (
    _full_partition_blocks,
    _full_partition_rho,
    exact_low_order_marginals,
    public_pair_scopes,
    rho_for_epsilon,
)


PROTOCOL_ID = "SAGE-QDTE-STATIC-ICE-TRANSFER-20260714-v7"


def _o1_public_optimal_measurements(
    rows: np.ndarray,
    qcat: QueryCatalogue,
    groups: list[WorkloadGroup],
    cardinalities: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    *,
    public_total: int,
    rho_total: float,
    delta: float,
    rng: np.random.Generator,
) -> Measurements:
    exact_oneway, exact_pairs = exact_low_order_marginals(rows, cardinalities, pairs)
    blocks = _full_partition_blocks(cardinalities, pairs)
    rho_by_block = _full_partition_rho(blocks, rho_total, "public_optimal")
    target_noisy = np.zeros(qcat.m, dtype=np.float64)
    target_projected = np.zeros(qcat.m, dtype=np.float64)
    variances = np.zeros(qcat.m, dtype=np.float64)
    measured_groups: list[MeasurementGroup] = []
    coverage = np.zeros(qcat.m, dtype=np.int32)

    for group in groups:
        parts = group.name.split(":")
        if group.family == "oneway" and len(parts) == 2:
            attr = int(parts[1])
            exact = exact_oneway[attr]
            block_name = f"oneway:{attr}"
        elif group.family == "twoway" and len(parts) == 3:
            pair = (int(parts[1]), int(parts[2]))
            exact = exact_pairs[pair]
            block_name = f"pair:{pair[0]}:{pair[1]}"
        else:
            raise ValueError(f"Unexpected complete-partition group {group.name!r}")
        rho = rho_by_block[block_name]
        noise_std = math.sqrt(1.0 / (2.0 * rho))
        noisy = exact.ravel() + rng.normal(0.0, noise_std, size=exact.size)
        projected = project_simplex(noisy, float(public_total))
        idx = group.query_indices.astype(np.int32, copy=False)
        if len(idx) != exact.size:
            raise ValueError(f"Query count mismatch for O1 group {group.name!r}")
        target_noisy[idx] = noisy
        target_projected[idx] = projected
        variances[idx] = noise_std * noise_std
        coverage[idx] += 1
        measured_groups.append(
            MeasurementGroup(
                query_indices=idx.copy(),
                sensitivity_l2=1.0,
                rho=float(rho),
                sigma=float(noise_std),
                noise_std=float(noise_std),
                name=group.name,
                family=group.family,
                is_partition=True,
            )
        )
    if not np.all(coverage == 1):
        raise ValueError("O1 groups must cover every query exactly once")
    rho_spent = float(sum(rho_by_block.values()))
    return Measurements(
        target_noisy=target_noisy.astype(np.float32),
        target_projected=target_projected.astype(np.float32),
        variances=variances.astype(np.float32),
        inv_variances=(1.0 / variances).astype(np.float32),
        groups=measured_groups,
        mode="dp",
        rho_total=float(rho_total),
        rho_spent=rho_spent,
        epsilon_delta=zcdp_epsilon(rho_spent, float(delta)),
        delta=float(delta),
        projection_diagnostics={
            "measurement_strategy": {
                "name": "complete_partitions_o1",
                "allocation_mode": "public_optimal",
                "adjacency": "add_remove",
            },
            "consistency": {"enabled": False, "method": "per_scope_simplex_p1"},
            "uncertainty": {"enabled": False},
        },
        num_rows=int(public_total),
    )


def _write_measurement_artifact(
    artifact_dir: Path,
    *,
    qcat: QueryCatalogue,
    schema: TableSchema,
    measurements: Measurements,
    extra_public: dict[str, Any] | None = None,
) -> None:
    ensure_dir(artifact_dir)
    qcat.save_json(artifact_dir / "queries.json")
    schema.save_json(artifact_dir / "schema.json")
    payload = measurements.to_public_dict()
    if extra_public:
        payload["strategy_transcript"] = extra_public
    write_json(payload, artifact_dir / "measurements.json")


def _generation_config(
    base: dict[str, Any],
    *,
    input_csv: Path,
    public_schema: Path,
    artifact_dir: Path,
    output_dir: Path,
    dataset_name: str,
    epsilon: float,
    rho_total: float,
    delta: float,
    seed: int,
    max_iters: int,
    precision_operator: str = "diagonal",
    exact_scoring_chunk_size: int = 512,
    confidence_stop: bool = False,
    confidence_stop_alpha: float = 0.05,
    entropy: bool = False,
    interaction_cycles: bool = False,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    set_nested(config, "run.dataset_name", dataset_name)
    set_nested(config, "run.input_csv", str(input_csv))
    set_nested(config, "run.output_dir", str(output_dir))
    set_nested(config, "run.seed", int(seed))
    set_nested(config, "preprocess.public_schema_json", str(public_schema))
    set_nested(config, "privacy.mode", "dp")
    set_nested(config, "privacy.adjacency", "add_remove")
    set_nested(config, "privacy.rho_total", float(rho_total))
    set_nested(config, "privacy.delta", float(delta))
    set_nested(config, "privacy.measurement_mode", "static_all")
    set_nested(config, "measurement.reuse_from", str(artifact_dir))
    set_nested(config, "measurement.fission.enabled", False)
    set_nested(config, "workload.reuse_from_measurement", True)
    set_nested(config, "projection.uncertainty.enabled", False)
    set_nested(config, "qdte.max_iters", int(max_iters))
    set_nested(config, "qdte.confidence_stop.enabled", False)
    set_nested(config, "qdte.entropy.enabled", False)
    set_nested(config, "qdte.structured_swap_enabled", False)
    set_nested(config, "qdte.structured_swap_compiler", "global_random_v1")
    if precision_operator in {
        "orthogonal_interaction",
        "orthogonal_interaction_shrink_raw",
        "orthogonal_interaction_shrink_analytic",
        "orthogonal_interaction_p3_raw",
        "orthogonal_interaction_p3_bootdiag",
        "orthogonal_interaction_p3_active_set",
    }:
        set_nested(config, "qdte.precision_operator", precision_operator)
        set_nested(config, "qdte.score_backend", "precision_operator")
        set_nested(config, "qdte.candidate_backend", "cpu_repair")
        set_nested(config, "qdte.objective_loss", "quadratic")
        set_nested(config, "qdte.objective_weighting", "variance")
        set_nested(config, "qdte.objective_weight_profile", "none")
        set_nested(config, "qdte.transport_mode", "atom_flow")
        set_nested(config, "qdte.atom_flow_update_mode", "batch")
        set_nested(config, "qdte.structured_swap_enabled", False)
        set_nested(config, "qdte.candidate_diagnostics", False)
        set_nested(config, "runtime.scoring_chunk_size", int(exact_scoring_chunk_size))
        if confidence_stop:
            if precision_operator != "orthogonal_interaction":
                raise ValueError("confidence_stop is calibrated only for raw orthogonal precision")
            set_nested(config, "qdte.confidence_stop.enabled", True)
            set_nested(config, "qdte.confidence_stop.method", "chi_square")
            set_nested(config, "qdte.confidence_stop.alpha", float(confidence_stop_alpha))
        if entropy:
            if precision_operator != "orthogonal_interaction":
                raise ValueError("entropy is calibrated only for unshrunk raw orthogonal precision")
            if confidence_stop:
                raise ValueError("entropy and confidence_stop are mutually exclusive")
            set_nested(config, "qdte.allow_below_noise_fallback", True)
            set_nested(config, "qdte.entropy.enabled", True)
            set_nested(
                config,
                "qdte.entropy.method",
                "confidence_constrained_product_kl_primal_dual_v1",
            )
            set_nested(config, "qdte.entropy.alpha", 0.05)
            set_nested(config, "qdte.entropy.product_prior_smoothing", 1.0)
            set_nested(config, "qdte.entropy.dual_initial", 1.0)
        if interaction_cycles:
            if precision_operator != "orthogonal_interaction":
                raise ValueError(
                    "interaction_cycles is calibrated only for unshrunk raw "
                    "orthogonal_interaction precision"
                )
            if confidence_stop or entropy:
                raise ValueError(
                    "interaction_cycles is isolated from confidence stopping and entropy"
                )
            set_nested(config, "qdte.structured_swap_enabled", True)
            set_nested(
                config,
                "qdte.structured_swap_compiler",
                "interaction_rectangle_v1",
            )
            set_nested(config, "qdte.structured_swap_start_iter", 1)
            set_nested(config, "qdte.structured_swap_interval", 4)
            set_nested(config, "qdte.structured_swap_candidate_units", 2048)
            set_nested(config, "qdte.structured_swap_transport_pool", 512)
            set_nested(config, "qdte.structured_swap_accept_start", 64)
            set_nested(config, "qdte.structured_swap_accept_end", 4)
            set_nested(config, "qdte.structured_swap_accept_schedule", "cosine")
            set_nested(config, "qdte.structured_swap_noise_guard_kappa", 2.0)
            set_nested(config, "qdte.structured_swap_noise_guard_mode", "fixed")
            set_nested(config, "qdte.structured_swap_trigger_rms", 1.0)
            set_nested(config, "qdte.structured_swap_delta_backend", "sparse_cpu")
    elif precision_operator != "diagonal":
        raise ValueError(f"Unknown precision_operator {precision_operator!r}")
    elif confidence_stop:
        raise ValueError("confidence_stop requires orthogonal_interaction precision")
    elif entropy:
        raise ValueError("entropy requires orthogonal_interaction precision")
    elif interaction_cycles:
        raise ValueError("interaction_cycles requires orthogonal_interaction precision")
    set_nested(config, "runtime.xla_preallocate", False)
    set_nested(config, "runtime.log_measurement_groups", False)
    set_nested(config, "evaluation.compute_true_query_error", True)
    set_nested(config, "evaluation.compute_heldout_query_error", False)
    set_nested(config, "evaluation.save_synthetic_csv", False)
    config.setdefault("method", {})
    config["method"].update(
        {
            "protocol_id": PROTOCOL_ID,
            "epsilon_requested": float(epsilon),
            "artifact_role": "development_transfer_gate_not_for_release",
        }
    )
    return config


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "final_measured_loss",
        "final_optimization_objective",
        "final_true_query_mae",
        "final_true_query_rmse",
        "true_query_avg_tvd",
        "true_query_max_tvd",
        "true_query_max_error",
        "offline_avg_tvd",
        "offline_max_tvd",
        "num_accepted_edits",
        "num_candidates_scored",
        "confidence_stop_enabled",
        "confidence_stop_triggered",
        "confidence_stop_iteration",
        "confidence_stop_objective_at_trigger",
        "entropy_enabled",
        "initial_entropy_regularizer",
        "final_entropy_regularizer",
        "initial_entropy_kl_per_row",
        "final_entropy_kl_per_row",
        "final_regularized_objective",
        "structured_swap_compiler",
        "structured_swap_total_candidates",
        "structured_swap_total_accepted",
        "structured_swap_total_noise_guard_rejections",
        "structured_swap_total_objective_advantage",
        "structured_swap_compiler_calls",
        "structured_swap_total_positive_rectangles",
        "structured_swap_max_selected_scopes",
        "structured_swap_target_linear_gain_mean",
        "structured_swap_time_seconds",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _offline_partition_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
    qcat: QueryCatalogue,
    groups: list[WorkloadGroup],
) -> dict[str, float]:
    true_answers = answer_queries(real, qcat).astype(np.float64) / float(len(real))
    synthetic_answers = answer_queries(synthetic, qcat).astype(np.float64) / float(len(synthetic))
    tvds = [
        0.5
        * float(
            np.sum(
                np.abs(
                    synthetic_answers[np.asarray(group.query_indices, dtype=np.int32)]
                    - true_answers[np.asarray(group.query_indices, dtype=np.int32)]
                )
            )
        )
        for group in groups
        if group.is_partition
    ]
    if not tvds:
        raise ValueError("Static-ICE offline TVD evaluation requires partition groups")
    return {
        "offline_avg_tvd": float(np.mean(tvds)),
        "offline_max_tvd": float(np.max(tvds)),
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_dir.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_root}")
    ensure_dir(output_root)
    input_dir = args.input_dir.resolve()
    schema = TableSchema.load_json(input_dir / "schema.json")
    rows = np.load(input_dir / "real_encoded.npy", allow_pickle=False).astype(np.int32)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    public_total = int(metadata["n_rows"])
    if rows.shape != (public_total, schema.d):
        raise ValueError("Encoded input does not match public metadata/schema")
    cards = schema.cardinalities
    pairs = public_pair_scopes(cards, int(args.max_pair_cells))
    qcat, groups = build_complete_low_order_workload(
        schema,
        max_pair_cells=int(args.max_pair_cells),
    )
    rho_total = rho_for_epsilon(float(args.epsilon), float(args.delta))
    base_config = load_yaml(args.config)

    artifacts_dir = output_root / "measurement_artifacts"
    o1_artifact = artifacts_dir / "o1_public_optimal"
    o1_measurements = _o1_public_optimal_measurements(
        rows,
        qcat,
        groups,
        cards,
        pairs,
        public_total=public_total,
        rho_total=rho_total,
        delta=float(args.delta),
        rng=np.random.default_rng(np.random.SeedSequence([int(args.seed), 0x01])),
    )
    _write_measurement_artifact(
        o1_artifact,
        qcat=qcat,
        schema=schema,
        measurements=o1_measurements,
    )

    strategy = compile_hierarchical_pair_strategy(cards, pairs)
    interaction_transcript = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=public_total,
        rho_total=rho_total,
        rng=np.random.default_rng(np.random.SeedSequence([int(args.seed), 0x1CE])),
        allocation_mode="public_optimal",
    )
    ice_measurements = interaction_transcript_to_diagonal_measurements(
        interaction_transcript,
        qcat,
        groups,
        delta=float(args.delta),
    )
    ice_artifact = artifacts_dir / "ice_public_optimal_diag"
    _write_measurement_artifact(
        ice_artifact,
        qcat=qcat,
        schema=schema,
        measurements=ice_measurements,
        extra_public=interaction_transcript.to_public_dict(),
    )

    exact_artifact: Path | None = None
    if (
        bool(args.include_exact)
        or bool(args.include_exact_stop)
        or bool(getattr(args, "include_shrink_raw", False))
        or bool(getattr(args, "include_shrink_analytic", False))
        or bool(getattr(args, "include_entropy", False))
        or bool(getattr(args, "include_cycle", False))
    ):
        ice_exact_measurements = interaction_transcript_to_diagonal_measurements(
            interaction_transcript,
            qcat,
            groups,
            delta=float(args.delta),
            target_projection="raw_reconstruction",
        )
        exact_artifact = artifacts_dir / "ice_public_optimal_exact"
        _write_measurement_artifact(
            exact_artifact,
            qcat=qcat,
            schema=schema,
            measurements=ice_exact_measurements,
        )

    shrink_artifact: Path | None = None
    if bool(getattr(args, "include_shrink_raw", False)) or bool(
        getattr(args, "include_shrink_analytic", False)
    ):
        shrink_measurements, _ = interaction_transcript_to_shrunk_measurements(
            interaction_transcript,
            qcat,
            groups,
            delta=float(args.delta),
        )
        shrink_artifact = artifacts_dir / "ice_public_optimal_shrink"
        _write_measurement_artifact(
            shrink_artifact,
            qcat=qcat,
            schema=schema,
            measurements=shrink_measurements,
        )

    p3_artifact: Path | None = None
    if (
        bool(getattr(args, "include_p3_raw", False))
        or bool(getattr(args, "include_p3_bootdiag", False))
        or bool(getattr(args, "include_p3_active_set", False))
    ):
        p3_measurements, _ = interaction_transcript_to_local_polytope_measurements(
            interaction_transcript,
            qcat,
            groups,
            delta=float(args.delta),
        )
        p3_artifact = artifacts_dir / "ice_public_optimal_p3"
        _write_measurement_artifact(
            p3_artifact,
            qcat=qcat,
            schema=schema,
            measurements=p3_measurements,
        )

    p3_bootdiag_artifact: Path | None = None
    if bool(getattr(args, "include_p3_bootdiag", False)):
        p3_bootdiag_measurements, _, _ = (
            interaction_transcript_to_local_polytope_bootdiag_measurements(
                interaction_transcript,
                qcat,
                groups,
                delta=float(args.delta),
                rng=np.random.default_rng(
                    np.random.SeedSequence([int(args.seed), 0xB007D1A6])
                ),
                num_samples=16,
                min_variance=1.0e-6,
                min_raw_variance_fraction=0.02,
            )
        )
        p3_bootdiag_artifact = artifacts_dir / "ice_public_optimal_p3_bootdiag16"
        _write_measurement_artifact(
            p3_bootdiag_artifact,
            qcat=qcat,
            schema=schema,
            measurements=p3_bootdiag_measurements,
        )

    run_metrics: dict[str, dict[str, Any]] = {}
    only_p3_arms = bool(getattr(args, "only_p3_arms", False))
    only_shrink_arms = bool(getattr(args, "only_shrink_arms", False))
    only_entropy_arm = bool(getattr(args, "only_entropy_arm", False))
    only_cycle_arm = bool(getattr(args, "only_cycle_arm", False))
    if sum((only_p3_arms, only_shrink_arms, only_entropy_arm, only_cycle_arm)) > 1:
        raise ValueError(
            "--only-p3-arms, --only-shrink-arms, --only-entropy-arm, and "
            "--only-cycle-arm are mutually exclusive"
        )
    if only_p3_arms and p3_artifact is None:
        raise ValueError("--only-p3-arms requires at least one requested P3 arm")
    if only_shrink_arms and shrink_artifact is None:
        raise ValueError("--only-shrink-arms requires at least one requested shrinkage arm")
    if only_entropy_arm and not bool(getattr(args, "include_entropy", False)):
        raise ValueError("--only-entropy-arm requires --include-entropy")
    if only_cycle_arm and not bool(getattr(args, "include_cycle", False)):
        raise ValueError("--only-cycle-arm requires --include-cycle")
    if bool(getattr(args, "include_entropy", False)) and exact_artifact is None:
        raise RuntimeError("Entropy arm is missing the exact raw ICE artifact")
    if bool(getattr(args, "include_cycle", False)) and exact_artifact is None:
        raise RuntimeError("Interaction-cycle arm is missing the exact raw ICE artifact")
    arms: list[tuple[str, Path, str, bool, bool, bool]] = []
    if not only_p3_arms and not only_shrink_arms and not only_entropy_arm and not only_cycle_arm:
        arms.extend(
            [
                ("o1_public_optimal", o1_artifact, "diagonal", False, False, False),
                ("ice_public_optimal_diag", ice_artifact, "diagonal", False, False, False),
            ]
        )
    if exact_artifact is not None and (bool(args.include_exact) or only_shrink_arms):
        arms.append(
            (
                "ice_public_optimal_exact",
                exact_artifact,
                "orthogonal_interaction",
                False,
                False,
                False,
            )
        )
    if shrink_artifact is not None and bool(getattr(args, "include_shrink_raw", False)):
        arms.append(
            (
                "ice_public_optimal_shrink_raw",
                shrink_artifact,
                "orthogonal_interaction_shrink_raw",
                False,
                False,
                False,
            )
        )
    if shrink_artifact is not None and bool(
        getattr(args, "include_shrink_analytic", False)
    ):
        arms.append(
            (
                "ice_public_optimal_shrink_analytic",
                shrink_artifact,
                "orthogonal_interaction_shrink_analytic",
                False,
                False,
                False,
            )
        )
    if exact_artifact is not None and bool(args.include_exact_stop):
        arms.append(
            (
                "ice_public_optimal_exact_confstop",
                exact_artifact,
                "orthogonal_interaction",
                True,
                False,
                False,
            )
        )
    if p3_artifact is not None and bool(getattr(args, "include_p3_raw", False)):
        arms.append(
            (
                "ice_public_optimal_p3_raw",
                p3_artifact,
                "orthogonal_interaction_p3_raw",
                False,
                False,
                False,
            )
        )
    if p3_bootdiag_artifact is not None:
        arms.append(
            (
                "ice_public_optimal_p3_bootdiag16",
                p3_bootdiag_artifact,
                "orthogonal_interaction_p3_bootdiag",
                False,
                False,
                False,
            )
        )
    if p3_artifact is not None and bool(getattr(args, "include_p3_active_set", False)):
        arms.append(
            (
                "ice_public_optimal_p3_active_set",
                p3_artifact,
                "orthogonal_interaction_p3_active_set",
                False,
                False,
                False,
            )
        )
    if bool(getattr(args, "include_entropy", False)):
        if exact_artifact is None:
            raise RuntimeError("Entropy arm is missing the exact raw ICE artifact")
        arms.append(
            (
                "ice_public_optimal_exact_entropy",
                exact_artifact,
                "orthogonal_interaction",
                False,
                True,
                False,
            )
        )
    if bool(getattr(args, "include_cycle", False)):
        if exact_artifact is None:
            raise RuntimeError("Interaction-cycle arm is missing the exact raw ICE artifact")
        arms.append(
            (
                "ice_public_optimal_exact_cycle",
                exact_artifact,
                "orthogonal_interaction",
                False,
                False,
                True,
            )
        )
    for arm, artifact, precision_operator, confidence_stop, entropy, interaction_cycles in arms:
        run_dir = output_root / "runs" / arm
        config = _generation_config(
            base_config,
            input_csv=input_dir / "raw.csv",
            public_schema=input_dir / "schema.json",
            artifact_dir=artifact,
            output_dir=run_dir,
            dataset_name=str(metadata.get("dataset", input_dir.name)),
            epsilon=float(args.epsilon),
            rho_total=rho_total,
            delta=float(args.delta),
            seed=int(args.seed),
            max_iters=int(args.max_iters),
            precision_operator=precision_operator,
            exact_scoring_chunk_size=int(args.exact_scoring_chunk_size),
            confidence_stop=confidence_stop,
            confidence_stop_alpha=float(args.confidence_stop_alpha),
            entropy=entropy,
            interaction_cycles=interaction_cycles,
        )
        run_metrics[arm] = run_qdte(config)
        synthetic = np.load(run_dir / "synthetic_encoded.npy", allow_pickle=False).astype(np.int32)
        run_metrics[arm].update(_offline_partition_metrics(rows, synthetic, qcat, groups))

    summary = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "development_transfer_gate_not_for_release",
        "dataset": str(metadata.get("dataset", input_dir.name)),
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "rho_total": rho_total,
        "seed": int(args.seed),
        "adjacency": "add_remove",
        "public_total": public_total,
        "num_queries": qcat.m,
        "num_pairs": len(pairs),
        "max_iters": int(args.max_iters),
        "confidence_stop_alpha": float(args.confidence_stop_alpha),
        "only_p3_arms": only_p3_arms,
        "only_shrink_arms": only_shrink_arms,
        "only_entropy_arm": only_entropy_arm,
        "only_cycle_arm": only_cycle_arm,
        "covariance_note": (
            "ICE-diag drops cross-cell and cross-scope covariance. Exact, shrinkage, and P3 "
            "arms isolate target estimation from uncertainty propagation."
        ),
        "arms": {arm: _headline(metrics) for arm, metrics in run_metrics.items()},
    }
    write_json(summary, output_root / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Static-ICE target-to-QDTE transfer gate against strong O1."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pair-cells", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--include-exact", action="store_true")
    parser.add_argument("--include-exact-stop", action="store_true")
    parser.add_argument("--include-shrink-raw", action="store_true")
    parser.add_argument("--include-shrink-analytic", action="store_true")
    parser.add_argument("--include-p3-raw", action="store_true")
    parser.add_argument("--include-p3-bootdiag", action="store_true")
    parser.add_argument("--include-p3-active-set", action="store_true")
    parser.add_argument("--include-entropy", action="store_true")
    parser.add_argument("--include-cycle", action="store_true")
    parser.add_argument("--only-p3-arms", action="store_true")
    parser.add_argument("--only-shrink-arms", action="store_true")
    parser.add_argument("--only-entropy-arm", action="store_true")
    parser.add_argument("--only-cycle-arm", action="store_true")
    parser.add_argument("--confidence-stop-alpha", type=float, default=0.05)
    parser.add_argument("--exact-scoring-chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    summary = run_pilot(parse_args())
    print(json.dumps(summary["arms"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
