from __future__ import annotations

import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import jax
import numpy as np
import pandas as pd

from qdte.config import save_yaml
from qdte.config_validation import validate_config
from qdte.dataio import ensure_dir, read_json, save_npy, write_json
from qdte.eval.metrics import (
    measured_loss,
    query_error_metrics,
    rms_standardized_residual,
    rms_unweighted_residual,
    unweighted_measured_loss,
)
from qdte.eval.runtime import RuntimeStats
from qdte.evolution.candidates import generate_candidates
from qdte.evolution.confidence import ChiSquareDiscrepancyStop
from qdte.evolution.entropy import (
    AtomEntropyState,
    EntropyDiscrepancyController,
    ReleasedProductPrior,
)
from qdte.evolution.initialization import initialize_independent_oneway, split_run_rng_streams
from qdte.evolution.gpu_candidates import (
    apply_edits_to_replicated_table,
    generate_and_score_candidates_gpu,
    prepare_gpu_candidate_context,
    replicate_table_to_devices,
)
from qdte.evolution.hybrid_candidates import (
    AdaptiveSwapAttributePolicy,
    candidate_unit_deltas,
    candidate_unit_feature_deltas,
    generate_global_directed_swap_units,
    prepare_hybrid_score_context,
    score_candidate_units as score_hybrid_candidate_units,
    score_candidate_units_l1 as score_hybrid_candidate_units_l1,
    score_candidate_units_precision,
)
from qdte.evolution.hybrid_transport import (
    apply_candidate_units as apply_hybrid_candidate_units,
    select_nonconflicting_candidate_units,
)
from qdte.evolution.interaction_cycles import (
    INTERACTION_CYCLE_METHOD,
    compile_interaction_cycle_candidates,
)
from qdte.evolution.precision import (
    ActiveSetOrthogonalInteractionPrecision,
    BootstrapDiagonalOrthogonalInteractionPrecision,
    DiagonalPrecision,
    OrthogonalInteractionPrecision,
    PrecisionOperator,
    ShrinkageAnalyticOrthogonalInteractionPrecision,
)
from qdte.measurement.covariance import (
    ActiveSetCoefficientCovariance,
    CoefficientBootstrapDiagonal,
)
from qdte.evolution.scheduler import debt_diagnostics, select_active_queries, update_query_debt_from_loss_vectors
from qdte.evolution.scoring import (
    compute_deltas,
    compute_deltas_sparse,
    prepare_orthogonal_precision_score_context,
    prepare_score_context,
    score_candidates,
    score_candidates_active_set_orthogonal_precision_with_quadratic,
    score_candidates_bootstrap_orthogonal_precision_with_quadratic,
    score_candidates_l1,
    score_candidates_orthogonal_precision_with_quadratic,
    score_candidates_precision,
    score_candidates_sparse,
    score_candidates_target_only,
)
from qdte.evolution.state import QDTEState
from qdte.evolution.transport import (
    apply_edits,
    batch_advantage,
    batch_advantage_l1,
    batch_advantage_precision,
    choose_atom_flow_batch_transport,
    choose_atom_flow_transport,
    choose_blind_transport,
    choose_constructive_pair_transport,
    choose_directed_group_transport,
    choose_directed_group_transport_jax,
    choose_random_group_transport,
    choose_transport_batch,
    choose_transport_batch_jax,
    select_nonconflicting_in_order,
    select_top_nonconflicting,
)
from qdte.measurement.measure import _apply_configured_projection, measure_real_dataset, measurements_from_public_dict
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_shrunk_measurements,
    local_polytope_projection_to_query_target,
)
from qdte.measurement.local_polytope import project_hierarchical_local_polytope
from qdte.measurement.public_artifact import verify_public_transcript
from qdte.measurement.shrinkage import (
    positive_part_block_james_stein,
    validate_frozen_shrinkage_artifact,
)
from qdte.measurement.fission import (
    FissionCheckpointSelection,
    GaussianFissionSplit,
    gaussian_fission_split,
    select_fission_checkpoint,
)
from qdte.preprocess import decode_array, load_and_preprocess_csv
from qdte.queries.eval_jax import answer_queries
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.types import QueryCatalogue, filter_query_catalogue, query_key
from qdte.queries.workload import WorkloadGroup, build_workload, filter_workload_groups
from qdte.schema import TableSchema


HELDOUT_WORKLOAD_DEFAULTS: dict[str, Any] = {
    "include_oneway": False,
    "include_2way_cat": True,
    "include_prefix": True,
    "include_range": True,
    "include_mixed": True,
    "include_kway": False,
    "include_kway_prefix": False,
    "include_kway_range": False,
    "include_kway_mixed": False,
    "include_orthogonal_kway_mixed": False,
    "include_halfspace": False,
    "max_queries": 10000,
    "max_terms": 4,
    "max_2way_cells": 10000,
    "range_intervals_per_num_attr": 128,
    "mixed_queries_per_pair": 128,
    "kway_orders": [3],
    "kway_prefix_orders": [3],
    "kway_range_orders": [3],
    "kway_mixed_orders": [3],
    "orthogonal_kway_mixed_orders": [2],
    "kway_queries_per_order": 128,
    "kway_prefix_queries_per_order": 128,
    "kway_range_queries_per_order": 128,
    "kway_mixed_queries_per_order": 128,
    "orthogonal_kway_mixed_scopes_per_order": 16,
    "orthogonal_kway_mixed_range_bins": 4,
    "orthogonal_kway_mixed_max_cells_per_group": 4096,
    "halfspace_queries": 128,
    "exact_group_sensitivity_max_cells": 200_000,
    "random_seed": 10000,
}


REPAIR_TYPE_NAMES: dict[int, str] = {
    0: "random",
    1: "single_enter",
    2: "single_exit",
    3: "paired",
    4: "masked_paired",
    5: "masked_exit",
    6: "directed_exit_only",
    7: "random_source_directed_exit",
    8: "masked_exit_only",
    9: "masked_single",
    10: "residual_weighted",
    11: "enumerated_local",
    12: "soft_single",
    13: "residual_value",
    14: "relaxed_masked_single",
    15: "constructive_partner",
    16: "constructive_attached_partner",
    17: "protected_same_row",
    18: "bounded_best_partner",
}


def _mean_or_zero(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    return float(np.mean(finite))


def _rate(mask: np.ndarray, denom: int) -> float:
    return float(np.sum(mask) / max(1, int(denom)))


def _candidate_diagnostic_summary(
    *,
    iteration: int,
    candidates: Any,
    advantages: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    qcat: QueryCatalogue,
    lambda_cost: float,
    min_advantage: float,
    selected_indices: np.ndarray,
    accepted_indices: np.ndarray,
    delta_index: QueryDeltaIndex | None,
) -> dict[str, float | int]:
    if candidates.size == 0:
        return {"iteration": int(iteration), "candidate_diag_enabled": 1, "candidate_diag_count": 0}
    if delta_index is not None:
        deltas = compute_deltas_sparse(candidates.old_rows, candidates.new_rows, delta_index)
    else:
        deltas = compute_deltas(candidates.old_rows, candidates.new_rows, qcat)
    d = deltas.astype(np.float32, copy=False)
    weights = residual.astype(np.float32, copy=False) * inv_variance.astype(np.float32, copy=False)
    full_linear = d @ weights
    full_quad = (d * d) @ inv_variance.astype(np.float32, copy=False)
    full_component = full_linear - 0.5 * full_quad
    full_advantage = full_component - float(lambda_cost) * candidates.edit_cost.astype(np.float32, copy=False)

    target_component = np.full(candidates.size, np.nan, dtype=np.float32)
    valid_target = (candidates.target_query_ids >= 0) & (candidates.target_query_ids < qcat.m)
    valid_idx = np.flatnonzero(valid_target)
    if len(valid_idx) > 0:
        qids = candidates.target_query_ids[valid_idx].astype(np.int32, copy=False)
        target_delta = d[valid_idx, qids]
        target_component[valid_idx] = target_delta * weights[qids] - 0.5 * (target_delta * target_delta) * inv_variance[qids]
    collateral_component = full_component - target_component

    contribution = d * weights.reshape(1, -1)
    beneficial_count = np.sum(contribution > 0.0, axis=1).astype(np.float32)
    harmful_count = np.sum(contribution < 0.0, axis=1).astype(np.float32)
    affected_count = np.sum(d != 0.0, axis=1).astype(np.float32)
    conflict = (beneficial_count > 0.0) & (harmful_count > 0.0)
    positive = np.asarray(advantages > float(min_advantage), dtype=bool)
    selected_mask = np.zeros(candidates.size, dtype=bool)
    selected_indices = np.asarray(selected_indices, dtype=np.int32)
    selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < candidates.size)]
    selected_mask[selected_indices] = True
    accepted_mask = np.zeros(candidates.size, dtype=bool)
    accepted_indices = np.asarray(accepted_indices, dtype=np.int32)
    accepted_indices = accepted_indices[(accepted_indices >= 0) & (accepted_indices < candidates.size)]
    accepted_mask[accepted_indices] = True
    valid_target_positive = valid_target & (target_component > 0.0)
    target_positive_full_negative = valid_target_positive & (~positive)
    target_positive_collateral_negative = valid_target_positive & (collateral_component < 0.0)

    row: dict[str, float | int] = {
        "iteration": int(iteration),
        "candidate_diag_enabled": 1,
        "candidate_diag_count": int(candidates.size),
        "diag_positive_full_rate": _rate(positive, candidates.size),
        "diag_selected_rate": _rate(selected_mask, candidates.size),
        "diag_accepted_rate": _rate(accepted_mask, candidates.size),
        "diag_mean_full_advantage": _mean_or_zero(np.asarray(advantages, dtype=np.float32)),
        "diag_mean_full_component": _mean_or_zero(full_component),
        "diag_mean_target_component": _mean_or_zero(target_component[valid_target]),
        "diag_mean_collateral_component": _mean_or_zero(collateral_component[valid_target]),
        "diag_target_positive_rate": _rate(valid_target_positive, int(np.sum(valid_target))),
        "diag_target_positive_full_negative_rate": _rate(target_positive_full_negative, int(np.sum(valid_target))),
        "diag_target_positive_collateral_negative_rate": _rate(
            target_positive_collateral_negative,
            int(np.sum(valid_target_positive)),
        ),
        "diag_mean_affected_queries": _mean_or_zero(affected_count),
        "diag_mean_beneficial_queries": _mean_or_zero(beneficial_count),
        "diag_mean_harmful_queries": _mean_or_zero(harmful_count),
        "diag_residual_conflict_rate": _rate(conflict, candidates.size),
    }
    for diag_key in (
        "requested_candidates",
        "directed_candidate_budget",
        "random_candidate_budget",
        "directed_candidates",
        "random_candidates",
        "planned_random_candidates",
        "fallback_random_candidates",
        "mixture_random_candidates",
        "directed_candidate_shortfall",
        "candidate_shortfall",
        "source_filter_attempts",
        "source_filter_failures",
        "source_filter_kept",
        "paired_source_filter_attempts",
        "paired_source_filter_failures",
        "paired_source_filter_kept",
        "random_source_exit_attempts",
        "qdte_mixture_candidates",
        "constructive_partner_candidates",
        "constructive_attached_partner_candidates",
        "constructive_attached_pair_units",
        "constructive_partner_seed_candidates",
        "constructive_partner_source_attempts",
        "constructive_partner_source_failures",
        "best_partner_candidates",
        "best_partner_pair_units",
        "best_partner_seed_candidates",
        "best_partner_source_attempts",
        "best_partner_source_failures",
        "best_partner_pairs_evaluated",
        "best_partner_positive_pairs",
        "protected_same_row_candidates",
        "protected_repair_seed_candidates",
        "protected_repair_attempts",
        "protected_repair_target_failures",
        "protected_repair_protection_successes",
    ):
        row[f"diag_{diag_key}"] = float(candidates.diagnostics.get(diag_key, 0.0))

    present_types = sorted(set(int(x) for x in candidates.repair_type.tolist()))
    for repair_type in present_types:
        name = REPAIR_TYPE_NAMES.get(repair_type, f"type_{repair_type}")
        prefix = f"rtype_{name}"
        mask = candidates.repair_type == repair_type
        count = int(np.sum(mask))
        valid_mask = mask & valid_target
        valid_count = int(np.sum(valid_mask))
        target_pos = valid_mask & (target_component > 0.0)
        row[f"{prefix}_generated"] = count
        row[f"{prefix}_positive"] = int(np.sum(mask & positive))
        row[f"{prefix}_selected"] = int(np.sum(mask & selected_mask))
        row[f"{prefix}_accepted"] = int(np.sum(mask & accepted_mask))
        row[f"{prefix}_positive_rate"] = _rate(mask & positive, count)
        row[f"{prefix}_selected_rate"] = _rate(mask & selected_mask, count)
        row[f"{prefix}_accepted_rate"] = _rate(mask & accepted_mask, count)
        row[f"{prefix}_mean_full_advantage"] = _mean_or_zero(np.asarray(advantages, dtype=np.float32)[mask])
        row[f"{prefix}_mean_target_component"] = _mean_or_zero(target_component[valid_mask])
        row[f"{prefix}_mean_collateral_component"] = _mean_or_zero(collateral_component[valid_mask])
        row[f"{prefix}_target_positive_rate"] = _rate(target_pos, valid_count)
        row[f"{prefix}_target_positive_full_negative_rate"] = _rate(target_pos & (~positive), valid_count)
        row[f"{prefix}_target_positive_collateral_negative_rate"] = _rate(
            target_pos & (collateral_component < 0.0),
            int(np.sum(target_pos)),
        )
        row[f"{prefix}_mean_affected_queries"] = _mean_or_zero(affected_count[mask])
        row[f"{prefix}_mean_beneficial_queries"] = _mean_or_zero(beneficial_count[mask])
        row[f"{prefix}_mean_harmful_queries"] = _mean_or_zero(harmful_count[mask])
        row[f"{prefix}_residual_conflict_rate"] = _rate(conflict & mask, count)
    return row


def _resolve_n_syn(value: Any, n_real: int) -> int:
    if value is None or str(value) == "same_as_real":
        return int(n_real)
    return int(value)


def _run_rng_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    return split_run_rng_streams(seed)


def _mark_run_started(output_dir: Path, seed: int) -> None:
    for name in (
        "metrics_final.json",
        "runtime.json",
        "synthetic_encoded.npy",
        "synthetic_decoded.csv",
        "run_metadata.json",
        "evaluation_external.json",
        "measurements_fission.json",
        "fission_checkpoints.json",
        "synthetic_train_terminal_encoded.npy",
    ):
        (output_dir / name).unlink(missing_ok=True)
    write_json({"status": "running", "seed": int(seed)}, output_dir / "run_status.json")


def _scheduled_accept_limit(
    *,
    iteration: int,
    max_iters: int,
    base_accept: int,
    qdte_cfg: dict[str, Any],
) -> int:
    schedule = str(qdte_cfg.get("accepted_per_iter_schedule", "fixed"))
    if schedule in {"fixed", "none"}:
        return max(1, int(base_accept))

    start = int(qdte_cfg.get("accepted_per_iter_start", base_accept))
    end = int(qdte_cfg.get("accepted_per_iter_end", 1))
    warmup_iters = max(0, int(qdte_cfg.get("accepted_per_iter_warmup_iters", 0)))
    anneal_iters = int(qdte_cfg.get("accepted_per_iter_anneal_iters", max_iters - warmup_iters))
    anneal_iters = max(1, anneal_iters)

    if iteration <= warmup_iters:
        value = float(start)
    else:
        progress = (iteration - warmup_iters - 1) / max(1, anneal_iters - 1)
        progress = min(1.0, max(0.0, float(progress)))
        if schedule == "linear":
            value = start + (end - start) * progress
        elif schedule == "cosine":
            value = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
        elif schedule == "exponential":
            if start <= 0 or end <= 0:
                value = start + (end - start) * progress
            else:
                value = start * ((end / start) ** progress)
        else:
            raise ValueError(f"Unknown qdte.accepted_per_iter_schedule={schedule!r}")
    return max(1, int(round(value)))


def _search_adjusted_noise_guard_kappa(
    *,
    mode: str,
    fixed_kappa: float,
    family_size: int,
    alpha: float,
) -> float:
    base = float(fixed_kappa)
    if mode == "fixed":
        return base
    if mode != "bonferroni":
        raise ValueError(f"Unknown structured swap noise-guard mode {mode!r}")
    if family_size <= 0:
        return base
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("structured swap noise-guard alpha must be in (0, 1)")
    one_sided_tail = float(alpha) / float(family_size)
    corrected = NormalDist().inv_cdf(1.0 - one_sided_tail)
    return max(base, float(corrected))


def _count_by_family(families: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    return counts


def _query_indices_by_family(families: list[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for idx, family in enumerate(families):
        grouped.setdefault(family, []).append(idx)
    return {family: np.asarray(indices, dtype=np.int32) for family, indices in grouped.items()}


def _vector_error_summary(prefix: str, values: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {
            f"{prefix}_l2": 0.0,
            f"{prefix}_linf": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_mae": 0.0,
            f"{prefix}_rmse": 0.0,
            f"{prefix}_negative_count": 0,
            f"{prefix}_positive_count": 0,
        }
    return {
        f"{prefix}_l2": float(np.linalg.norm(v)),
        f"{prefix}_linf": float(np.max(np.abs(v))),
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_mae": float(np.mean(np.abs(v))),
        f"{prefix}_rmse": float(math.sqrt(float(np.mean(v * v)))),
        f"{prefix}_negative_count": int(np.sum(v < -1.0e-9)),
        f"{prefix}_positive_count": int(np.sum(v > 1.0e-9)),
    }


def _raw_variances_from_measurement_groups(
    groups: list[Any],
    num_queries: int,
    fallback_variances: np.ndarray,
) -> np.ndarray:
    fallback = np.asarray(fallback_variances, dtype=np.float64)
    raw = np.full(num_queries, np.nan, dtype=np.float64)
    for group in groups:
        idx = np.asarray(group.query_indices, dtype=np.int32)
        noise_var = float(group.noise_std) * float(group.noise_std)
        if noise_var > 0.0:
            raw[idx] = noise_var
    missing = ~np.isfinite(raw)
    if np.any(missing):
        raw[missing] = fallback[missing]
    return np.maximum(raw, 1.0e-12)


def _objective_variance_arrays(
    measurement_variance: np.ndarray,
    measurement_inv_variance: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = str(mode).lower()
    if normalized == "variance":
        variance = measurement_variance.astype(np.float32, copy=True)
        inv_variance = measurement_inv_variance.astype(np.float32, copy=True)
    elif normalized == "unweighted":
        variance = np.ones_like(measurement_variance, dtype=np.float32)
        inv_variance = np.ones_like(measurement_inv_variance, dtype=np.float32)
    else:
        raise ValueError("qdte.objective_weighting must be one of: unweighted, variance")
    sigma = np.sqrt(variance).astype(np.float32)
    return variance, inv_variance, sigma


def _objective_loss_weights(
    mode: str,
    workload_groups: list[WorkloadGroup],
    num_queries: int,
) -> np.ndarray:
    normalized = str(mode).lower()
    if normalized in {"quadratic", "l2", "measured_l2"}:
        return np.ones(int(num_queries), dtype=np.float32)
    if normalized != "tvd_l1":
        raise ValueError("qdte.objective_loss must be one of: quadratic, tvd_l1")

    weights = np.zeros(int(num_queries), dtype=np.float32)
    partition_groups = [
        group
        for group in workload_groups
        if bool(getattr(group, "is_partition", False)) and len(getattr(group, "query_indices", [])) > 1
    ]
    if not partition_groups:
        weights.fill(1.0)
        return weights
    for group in partition_groups:
        idx = np.asarray(group.query_indices, dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < int(num_queries))]
        weights[idx] += 0.5
    return weights.astype(np.float32, copy=False)


def _objective_query_weight_multipliers(
    profile: str,
    workload_groups: list[WorkloadGroup],
    num_queries: int,
) -> np.ndarray:
    normalized = str(profile).lower()
    multipliers = np.ones(int(num_queries), dtype=np.float32)
    if normalized == "none":
        return multipliers
    if normalized != "joint_utility_envelope":
        raise ValueError(
            "qdte.objective_weight_profile must be one of: "
            "none, joint_utility_envelope"
        )

    partition_groups = [
        group
        for group in workload_groups
        if bool(getattr(group, "is_partition", False))
        and len(getattr(group, "query_indices", [])) > 1
    ]
    if not partition_groups:
        return multipliers
    max_block_size = max(len(group.query_indices) for group in partition_groups)
    for group in partition_groups:
        idx = np.asarray(group.query_indices, dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < int(num_queries))]
        multipliers[idx] += float(len(group.query_indices)) / float(max_block_size)
    return multipliers


def _objective_selection_sigma(objective_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(objective_weights, dtype=np.float32)
    sigma = np.full(weights.shape, 1.0e12, dtype=np.float32)
    positive = weights > 0.0
    sigma[positive] = 1.0 / np.maximum(weights[positive], 1.0e-6)
    return sigma


def _optimization_objective_value(
    residual: np.ndarray,
    objective_loss: str,
    inv_variance: np.ndarray,
    objective_loss_weights: np.ndarray,
    precision_operator: PrecisionOperator | None = None,
) -> float:
    if str(objective_loss).lower() == "tvd_l1":
        return float(
            np.abs(residual.astype(np.float64))
            @ objective_loss_weights.astype(np.float64)
        )
    if precision_operator is not None:
        return precision_operator.loss(residual)
    return measured_loss(residual, inv_variance)


def _optimization_loss_vector(
    residual: np.ndarray,
    objective_loss: str,
    inv_variance: np.ndarray,
    objective_loss_weights: np.ndarray,
) -> np.ndarray:
    r = residual.astype(np.float64)
    if str(objective_loss).lower() == "tvd_l1":
        return np.abs(r) * objective_loss_weights.astype(np.float64)
    return 0.5 * r * r * inv_variance.astype(np.float64)


def _measurement_reuse_path(config: dict[str, Any]) -> Path | None:
    measurement_cfg = config.get("measurement", {})
    if measurement_cfg is None:
        return None
    if not isinstance(measurement_cfg, dict):
        raise ValueError("measurement must be a mapping")
    raw = measurement_cfg.get("reuse_from", measurement_cfg.get("artifact_dir"))
    if raw in {None, ""}:
        return None
    return Path(str(raw))


def _measurement_json_path(path: Path) -> Path:
    if path.is_dir():
        return path / "measurements.json"
    return path


def _validate_reused_queries(path: Path, qcat: QueryCatalogue) -> None:
    query_path = path / "queries.json" if path.is_dir() else path.parent / "queries.json"
    if not query_path.exists():
        raise FileNotFoundError(f"measurement.reuse_from does not contain queries.json near {path}")
    reused_qcat = QueryCatalogue.from_dict(read_json(query_path))
    if reused_qcat.m != qcat.m:
        raise ValueError(
            f"measurement.reuse_from query count mismatch: artifact has {reused_qcat.m}, current workload has {qcat.m}"
        )
    for qid in range(qcat.m):
        if query_key(reused_qcat, qid) != query_key(qcat, qid):
            raise ValueError(f"measurement.reuse_from query mismatch at qid={qid}")


def _validate_reused_schema(path: Path, schema: TableSchema) -> None:
    schema_path = path / "schema.json" if path.is_dir() else path.parent / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"measurement.reuse_from does not contain schema.json near {path}")
    reused_schema = TableSchema.load_json(schema_path)
    if reused_schema.to_dict() != schema.to_dict():
        raise ValueError("measurement.reuse_from schema does not match the current encoded data schema")


def _reuse_workload_from_measurement(config: dict[str, Any]) -> bool:
    workload_cfg = config.get("workload", {})
    if not isinstance(workload_cfg, dict):
        return False
    return bool(workload_cfg.get("reuse_from_measurement", False))


def _load_reused_workload_groups(path: Path, qcat: QueryCatalogue) -> list[WorkloadGroup]:
    measurement_json_path = _measurement_json_path(path)
    if not measurement_json_path.exists():
        raise FileNotFoundError(f"measurement.reuse_from does not contain measurements.json: {path}")
    data = read_json(measurement_json_path)
    groups: list[WorkloadGroup] = []
    for raw in data.get("groups", []):
        idx = np.asarray(raw.get("query_indices", []), dtype=np.int32)
        if np.any(idx < 0) or np.any(idx >= int(qcat.m)):
            raise ValueError(f"measurement.reuse_from group {raw.get('name', '')!r} has out-of-range query indices")
        groups.append(
            WorkloadGroup(
                name=str(raw.get("name", "")),
                family=str(raw.get("family", "unknown")),
                query_indices=idx,
                sensitivity_l2=float(raw.get("sensitivity_l2", 1.0)),
                is_partition=bool(raw.get("is_partition", False)),
            )
        )
    if not groups and qcat.m:
        groups.append(
            WorkloadGroup(
                name="reused_measurement",
                family="reused",
                query_indices=np.arange(qcat.m, dtype=np.int32),
                sensitivity_l2=1.0,
                is_partition=False,
            )
        )
    return groups


def _oracle_projection_bias_diagnostics(
    *,
    true_answers: np.ndarray,
    measurements: Any,
    qcat: QueryCatalogue,
    schema: Any,
    n_real: int,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    log: Any,
) -> dict[str, Any] | None:
    evaluation_cfg = config.get("evaluation", {})
    oracle_cfg = evaluation_cfg.get("oracle_projection_bias", {})
    if oracle_cfg is None or not bool(oracle_cfg.get("enabled", False)):
        return None
    if not isinstance(oracle_cfg, dict):
        raise ValueError("evaluation.oracle_projection_bias must be a mapping")
    num_samples = int(oracle_cfg.get("num_samples", 32))
    if num_samples <= 1:
        raise ValueError("evaluation.oracle_projection_bias.num_samples must be greater than 1")
    diag_seed = int(oracle_cfg.get("seed", seed + 91_729))
    rng = np.random.default_rng(diag_seed)
    projection_cfg = config.get("projection", {})
    raw_var = _raw_variances_from_measurement_groups(
        measurements.groups,
        qcat.m,
        measurements.variances,
    )
    raw_std = np.sqrt(raw_var)
    true = np.asarray(true_answers, dtype=np.float64)
    observed_projected, _ = _apply_configured_projection(
        measurements.target_noisy.astype(np.float32),
        qcat,
        measurements.groups,
        int(n_real),
        projection_cfg,
        raw_var.astype(np.float32),
        schema.cardinalities,
    )
    mean = np.zeros(qcat.m, dtype=np.float64)
    m2 = np.zeros(qcat.m, dtype=np.float64)
    for sample_idx in range(1, num_samples + 1):
        noisy = true + rng.normal(loc=0.0, scale=raw_std, size=qcat.m)
        projected, _ = _apply_configured_projection(
            noisy.astype(np.float32),
            qcat,
            measurements.groups,
            int(n_real),
            projection_cfg,
            raw_var.astype(np.float32),
            schema.cardinalities,
        )
        x = projected.astype(np.float64)
        delta = x - mean
        mean += delta / float(sample_idx)
        m2 += delta * (x - mean)
    projected_var = m2 / float(num_samples - 1)
    oracle_bias = mean - true
    final_target = np.asarray(measurements.target_projected, dtype=np.float64)
    observed_projected = np.asarray(observed_projected, dtype=np.float64)
    oracle_debiased = observed_projected - oracle_bias
    oracle_debiased_reprojected, _ = _apply_configured_projection(
        oracle_debiased.astype(np.float32),
        qcat,
        measurements.groups,
        int(n_real),
        projection_cfg,
        raw_var.astype(np.float32),
        schema.cardinalities,
    )
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "num_samples": int(num_samples),
        "seed": int(diag_seed),
        "num_queries": int(qcat.m),
        "raw_variance_mean": float(np.mean(raw_var)),
        "raw_variance_min": float(np.min(raw_var)),
        "raw_variance_max": float(np.max(raw_var)),
        "projected_mc_variance_mean": float(np.mean(projected_var)),
        "projected_mc_variance_min": float(np.min(projected_var)),
        "projected_mc_variance_max": float(np.max(projected_var)),
    }
    diagnostics.update(_vector_error_summary("oracle_bias", oracle_bias))
    diagnostics.update(_vector_error_summary("observed_noisy_error", np.asarray(measurements.target_noisy) - true))
    diagnostics.update(_vector_error_summary("observed_projected_error", observed_projected - true))
    diagnostics.update(_vector_error_summary("final_target_error", final_target - true))
    diagnostics.update(_vector_error_summary("oracle_debiased_target_error", oracle_debiased - true))
    diagnostics.update(
        _vector_error_summary("oracle_debiased_reprojected_error", oracle_debiased_reprojected.astype(np.float64) - true)
    )
    diagnostics["oracle_debiased_target_min"] = float(np.min(oracle_debiased)) if oracle_debiased.size else 0.0
    diagnostics["oracle_debiased_target_negative_count"] = int(np.sum(oracle_debiased < -1.0e-9))
    diagnostics["oracle_debiased_reprojected_min"] = (
        float(np.min(oracle_debiased_reprojected)) if oracle_debiased_reprojected.size else 0.0
    )
    diagnostics["oracle_debiased_reprojected_negative_count"] = int(
        np.sum(np.asarray(oracle_debiased_reprojected) < -1.0e-9)
    )
    plugin_bias = getattr(measurements, "projection_uncertainty_bias", None)
    if plugin_bias is not None:
        plugin = np.asarray(plugin_bias, dtype=np.float64)
        diff = plugin - oracle_bias
        diagnostics.update(_vector_error_summary("plugin_bias", plugin))
        diagnostics.update(_vector_error_summary("plugin_minus_oracle_bias", diff))
        denom = float(np.linalg.norm(plugin) * np.linalg.norm(oracle_bias))
        diagnostics["plugin_oracle_bias_cosine"] = float(np.dot(plugin, oracle_bias) / denom) if denom > 0.0 else 0.0
    write_json(diagnostics, output_dir / "oracle_projection_bias.json")
    log(
        "Oracle projection-bias diagnostic for offline evaluation only: "
        f"samples={num_samples}, oracle_bias_l2={diagnostics['oracle_bias_l2']:.6g}, "
        f"observed_projected_error_l2={diagnostics['observed_projected_error_l2']:.6g}, "
        f"final_target_error_l2={diagnostics['final_target_error_l2']:.6g}"
    )
    return diagnostics


def _heldout_workload_config(evaluation_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = evaluation_cfg.get("heldout_workload", {})
    return {key: cfg.get(key, default) for key, default in HELDOUT_WORKLOAD_DEFAULTS.items()}


def _workload_summary(qcat: Any, workload_groups: list[Any], schema: Any, config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("workload", {})
    defaults = {
        "max_queries": 10000,
        "max_2way_cells": 5000,
        "range_intervals_per_num_attr": 64,
        "mixed_queries_per_pair": 64,
        "halfspace_queries": 64,
        "include_oneway": True,
        "include_2way_cat": True,
        "include_prefix": True,
        "include_range": True,
        "include_mixed": True,
        "include_kway": False,
        "include_kway_prefix": False,
        "include_kway_range": False,
        "include_kway_mixed": False,
        "include_orthogonal_kway_mixed": False,
        "include_halfspace": False,
        "kway_orders": [3],
        "kway_prefix_orders": [3],
        "kway_range_orders": [3],
        "kway_mixed_orders": [3],
        "orthogonal_kway_mixed_orders": [2],
        "kway_queries_per_order": 64,
        "kway_prefix_queries_per_order": 64,
        "kway_range_queries_per_order": 64,
        "kway_mixed_queries_per_order": 64,
        "orthogonal_kway_mixed_scopes_per_order": 4,
        "orthogonal_kway_mixed_range_bins": 4,
        "orthogonal_kway_mixed_max_cells_per_group": 4096,
        "halfspace_queries": 64,
        "exact_group_sensitivity_max_cells": 200_000,
    }
    config_values = {key: cfg.get(key, default) for key, default in defaults.items()}
    query_counts = _count_by_family(qcat.families)
    group_counts = _count_by_family([group.family for group in workload_groups])
    max_queries = int(config_values["max_queries"])
    max_queries_hit = int(qcat.m) >= max_queries

    pair_cells = [
        int(schema.columns[a].cardinality) * int(schema.columns[b].cardinality)
        for a in range(schema.d)
        for b in range(a + 1, schema.d)
    ]
    total_2way_cells_possible = int(sum(pair_cells))
    twoway_queries_constructed = int(query_counts.get("twoway", 0))
    max_2way_cells_appears_limiting = (
        bool(config_values["include_2way_cat"])
        and not max_queries_hit
        and total_2way_cells_possible > int(config_values["max_2way_cells"])
        and twoway_queries_constructed < total_2way_cells_possible
    )
    return {
        "total_num_queries": int(qcat.m),
        "total_queries": int(qcat.m),
        "num_queries_by_family": query_counts,
        "queries_by_family": query_counts,
        "num_groups_by_family": group_counts,
        "groups_by_family": group_counts,
        "workload_config": config_values,
        "max_queries_hit": bool(max_queries_hit),
        "max_2way_cells_appears_limiting": bool(max_2way_cells_appears_limiting),
        "num_2way_pairs_possible": int(len(pair_cells)),
        "num_2way_groups_constructed": int(group_counts.get("twoway", 0)),
        "num_2way_cells_possible": total_2way_cells_possible,
        "num_2way_queries_constructed": twoway_queries_constructed,
    }


def _heldout_workload_summary(
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    workload_config: dict[str, Any],
    num_removed_as_measured_duplicates: int,
    heldout_exclude_measured_queries: bool,
) -> dict[str, Any]:
    query_counts = _count_by_family(qcat.families)
    group_counts = _count_by_family([group.family for group in workload_groups])
    return {
        "total_num_queries": int(qcat.m),
        "total_queries": int(qcat.m),
        "num_queries_by_family": query_counts,
        "queries_by_family": query_counts,
        "num_groups_by_family": group_counts,
        "groups_by_family": group_counts,
        "num_removed_as_measured_duplicates": int(num_removed_as_measured_duplicates),
        "heldout_exclude_measured_queries": bool(heldout_exclude_measured_queries),
        "workload_config": workload_config,
    }


def _query_error_metrics_or_zero(
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    n_real: int,
    n_syn: int,
    prefix: str,
) -> dict[str, float]:
    if len(true_answers) == 0:
        return {
            f"{prefix}_mae": 0.0,
            f"{prefix}_rmse": 0.0,
            f"{prefix}_max_error": 0.0,
        }
    return query_error_metrics(true_answers, syn_answers, n_real, n_syn, prefix=prefix)


def _true_query_evaluation_metrics(
    qcat: QueryCatalogue,
    true_answers: np.ndarray,
    initial_answers: np.ndarray,
    final_answers: np.ndarray,
    n_real: int,
    n_syn: int,
) -> dict[str, Any]:
    initial_metrics = _query_error_metrics_or_zero(
        true_answers,
        initial_answers,
        n_real,
        n_syn,
        prefix="initial_true_query",
    )
    final_metrics = _query_error_metrics_or_zero(
        true_answers,
        final_answers,
        n_real,
        n_syn,
        prefix="final_true_query",
    )
    return {
        "num_queries": int(qcat.m),
        "queries_by_family": _count_by_family(qcat.families),
        **initial_metrics,
        **final_metrics,
        "true_query_mae_reduction": float(
            initial_metrics["initial_true_query_mae"] - final_metrics["final_true_query_mae"]
        ),
        "true_query_rmse_reduction": float(
            initial_metrics["initial_true_query_rmse"] - final_metrics["final_true_query_rmse"]
        ),
    }


def _true_query_metrics_by_family(
    qcat: QueryCatalogue,
    true_answers: np.ndarray,
    initial_answers: np.ndarray,
    final_answers: np.ndarray,
    n_real: int,
    n_syn: int,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for family, idx in _query_indices_by_family(qcat.families).items():
        family_metrics = _true_query_evaluation_metrics(
            filter_query_catalogue(qcat, idx),
            true_answers[idx],
            initial_answers[idx],
            final_answers[idx],
            n_real,
            n_syn,
        )
        family_metrics.pop("queries_by_family")
        result[family] = family_metrics
    return result


def _metrics_by_family(
    qcat: Any,
    initial_residual: np.ndarray,
    final_residual: np.ndarray,
    inv_variance: np.ndarray,
    true_answers: np.ndarray | None,
    initial_answers: np.ndarray,
    final_answers: np.ndarray,
    n_real: int,
    n_syn: int,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for family, idx in _query_indices_by_family(qcat.families).items():
        initial_family_loss = measured_loss(initial_residual[idx], inv_variance[idx])
        final_family_loss = measured_loss(final_residual[idx], inv_variance[idx])
        initial_family_unweighted_loss = unweighted_measured_loss(initial_residual[idx])
        final_family_unweighted_loss = unweighted_measured_loss(final_residual[idx])
        family_metrics: dict[str, float | int] = {
            "num_queries": int(len(idx)),
            "initial_measured_loss": initial_family_loss,
            "final_measured_loss": final_family_loss,
            "measured_loss_reduction": float(initial_family_loss - final_family_loss),
            "initial_unweighted_measured_loss": initial_family_unweighted_loss,
            "final_unweighted_measured_loss": final_family_unweighted_loss,
            "unweighted_measured_loss_reduction": float(
                initial_family_unweighted_loss - final_family_unweighted_loss
            ),
            "initial_rms_unweighted_residual": rms_unweighted_residual(initial_residual[idx]),
            "final_rms_unweighted_residual": rms_unweighted_residual(final_residual[idx]),
        }
        if true_answers is not None:
            family_metrics.update(
                query_error_metrics(
                    true_answers[idx],
                    initial_answers[idx],
                    n_real,
                    n_syn,
                    prefix="initial_true_query",
                )
            )
            family_metrics.update(
                query_error_metrics(
                    true_answers[idx],
                    final_answers[idx],
                    n_real,
                    n_syn,
                    prefix="final_true_query",
                )
            )
            family_metrics["true_query_mae_reduction"] = float(
                family_metrics["initial_true_query_mae"] - family_metrics["final_true_query_mae"]
            )
            family_metrics["true_query_rmse_reduction"] = float(
                family_metrics["initial_true_query_rmse"] - family_metrics["final_true_query_rmse"]
            )
        result[family] = family_metrics
    return result


def _write_timeseries(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "iteration",
                "wall_time",
                "measured_loss",
                "optimization_objective",
                "unweighted_measured_loss",
                "rms_standardized_residual",
                "rms_unweighted_residual",
                "residual_l2",
                "residual_l1",
                "active_queries",
                "num_candidates",
                "raw_returned_candidates_before_edge_dedup",
                "duplicate_edge_candidates",
                "duplicate_edge_rate",
                "edge_candidates_trimmed_to_return_limit",
                "candidates_scored_this_iter",
                "positive_advantage_rate",
                "positive_returned_rate",
                "selected_nonconflicting",
                "accept_limit",
                "accepted_edits",
                "accepted_rate",
                "mean_advantage",
                "batch_advantage",
                "requested_candidates",
                "directed_candidate_budget",
                "random_candidate_budget",
                "directed_candidates",
                "random_candidates",
                "planned_random_candidates",
                "fallback_random_candidates",
                "mixture_random_candidates",
                "paired_candidates",
                "masked_paired_candidates",
                "masked_exit_candidates",
                "masked_single_query_candidates",
                "relaxed_masked_single_query_candidates",
                "directed_exit_only_candidates",
                "masked_exit_only_candidates",
                "random_source_directed_exit_candidates",
                "residual_weighted_mutation_candidates",
                "enumerated_local_candidates",
                "soft_single_query_candidates",
                "residual_value_mutation_candidates",
                "constructive_partner_candidates",
                "constructive_attached_partner_candidates",
                "constructive_attached_pair_units",
                "constructive_partner_seed_candidates",
                "constructive_partner_source_attempts",
                "constructive_partner_source_failures",
                "best_partner_candidates",
                "best_partner_pair_units",
                "best_partner_seed_candidates",
                "best_partner_source_attempts",
                "best_partner_source_failures",
                "best_partner_pairs_evaluated",
                "best_partner_positive_pairs",
                "protected_same_row_candidates",
                "protected_repair_seed_candidates",
                "protected_repair_attempts",
                "protected_repair_target_failures",
                "protected_repair_protection_successes",
                "proposal_mixture_candidates",
                "qdte_mixture_candidates",
                "single_directed_candidates",
                "directed_candidate_shortfall",
                "candidate_shortfall",
                "source_filter_attempts",
                "source_filter_failures",
                "paired_source_filter_attempts",
                "paired_source_filter_failures",
                "random_source_exit_attempts",
                "atom_flow_pool_candidates",
                "atom_flow_edges",
                "atom_flow_source_atoms",
                "atom_flow_target_atoms",
                "atom_flow_augments",
                "atom_flow_batch_mode",
                "atom_flow_exact_mode",
                "atom_flow_selected_candidates",
                "atom_flow_prefix_candidates",
                "constructive_pair_pool_candidates",
                "constructive_pair_seed_candidates",
                "constructive_pair_pairs_evaluated",
                "constructive_pair_positive_pairs",
                "constructive_pair_explicit_pairs_evaluated",
                "constructive_pair_explicit_positive_pairs",
                "constructive_pair_units",
                "constructive_pair_single_units",
                "constructive_pair_pair_units",
                "constructive_pair_explicit_pair_units",
                "constructive_pair_selected_units",
                "constructive_pair_prefix_units",
                "constructive_pair_selected_candidates",
                "constructive_pair_accepted_candidates",
                "random_group_pool_candidates",
                "random_group_groups_evaluated",
                "random_group_positive_groups",
                "random_group_best_group_size",
                "random_group_accepted_candidates",
                "random_group_groups_with_negative_member",
                "directed_group_pool_candidates",
                "directed_group_seed_candidates",
                "directed_group_groups_evaluated",
                "directed_group_positive_groups",
                "directed_group_expansion_steps",
                "directed_group_best_group_size",
                "directed_group_accepted_candidates",
                "directed_group_groups_with_negative_member",
                "incremental_answer_drift",
                "mean_debt",
                "max_debt",
                "num_positive_debt_queries",
                "mean_collateral_damage",
                "max_collateral_damage",
            ]
        ).to_csv(path, index=False)


def run_qdte(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    run_cfg = config.get("run", {})
    privacy_cfg = config.get("privacy", {})
    qdte_cfg = config.get("qdte", {})
    measurement_cfg = config.get("measurement", {})
    runtime_cfg = config.get("runtime", {})
    evaluation_cfg = config.get("evaluation", {})
    debug_cfg = config.get("debug", {})
    seed = int(run_cfg.get("seed", 0))
    measurement_rng, generation_rng = _run_rng_streams(seed)
    output_dir = ensure_dir(run_cfg.get("output_dir", "outputs/qdte_run"))
    _mark_run_started(output_dir, seed)
    logs: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        logs.append(message)

    stats = RuntimeStats()
    save_yaml(config, output_dir / "config_resolved.yaml")
    log(f"Output dir: {output_dir}")
    log(f"JAX devices: {jax.devices()}")

    dp_release_mode = bool(privacy_cfg.get("dp_release_mode", False))
    reuse_path = _measurement_reuse_path(config)
    transcript_only_generation = bool(run_cfg.get("transcript_only_generation", False))
    preprocess_result = None
    X_real: np.ndarray | None = None
    preloaded_measurements = None
    transcript_verify_seconds = 0.0
    if transcript_only_generation:
        if reuse_path is None or not reuse_path.is_dir():
            raise ValueError(
                "Transcript-only generation requires measurement.reuse_from to be a "
                "public transcript directory"
            )
        t0 = time.perf_counter()
        verified_transcript = verify_public_transcript(reuse_path)
        transcript_verify_seconds = time.perf_counter() - t0
        schema = verified_transcript.schema
        qcat = verified_transcript.queries
        preloaded_measurements = verified_transcript.measurements
        workload_groups = _load_reused_workload_groups(reuse_path, qcat)
        public_schema_path = verified_transcript.root / "schema.json"
        public_n_rows = int(privacy_cfg["public_n_rows"])
        if public_n_rows != int(preloaded_measurements.num_rows):
            raise ValueError(
                "Configured public row count does not match the sealed transcript: "
                f"config={public_n_rows}, transcript={preloaded_measurements.num_rows}"
            )
        configured_schema = TableSchema.load_json(
            Path(str(config.get("preprocess", {})["public_schema_json"]))
        )
        if configured_schema.to_dict() != schema.to_dict():
            raise ValueError(
                "Configured public schema does not match the sealed transcript schema"
            )
        if not np.isclose(
            float(privacy_cfg.get("rho_total", float("nan"))),
            float(preloaded_measurements.rho_total),
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Configured privacy.rho_total does not match the sealed transcript"
            )
        if not np.isclose(
            float(privacy_cfg.get("delta", float("nan"))),
            float(preloaded_measurements.delta),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Configured privacy.delta does not match the sealed transcript")
        n_real = public_n_rows
        mechanism_num_rows = public_n_rows
        log(
            "Loaded sealed public transcript without private rows: "
            f"rows={public_n_rows}, cols={schema.d}, queries={qcat.m}, "
            f"groups={len(workload_groups)}"
        )
    else:
        preprocess_result = load_and_preprocess_csv(config)
        X_real = preprocess_result.X
        schema = preprocess_result.schema
        public_schema_path = preprocess_result.public_schema_path
        n_real = int(X_real.shape[0])
        if n_real <= 0 or schema.d <= 0:
            raise ValueError("Input data must contain at least one row and one column")
        if dp_release_mode:
            public_n_rows = int(privacy_cfg["public_n_rows"])
            if n_real != public_n_rows:
                raise ValueError(
                    "Private input row count does not match declared privacy.public_n_rows: "
                    f"input={n_real}, public={public_n_rows}"
                )
            mechanism_num_rows = public_n_rows
        else:
            public_n_rows = None
            mechanism_num_rows = n_real
        log(f"Loaded real data: rows={X_real.shape[0]}, cols={X_real.shape[1]}")
        if public_schema_path is not None:
            log(f"Encoded input using public schema: {public_schema_path}")

        if _reuse_workload_from_measurement(config):
            if reuse_path is None:
                raise ValueError("workload.reuse_from_measurement requires measurement.reuse_from")
            query_path = reuse_path / "queries.json" if reuse_path.is_dir() else reuse_path.parent / "queries.json"
            if not query_path.exists():
                raise FileNotFoundError(f"workload.reuse_from_measurement could not find queries.json near {reuse_path}")
            qcat = QueryCatalogue.from_dict(read_json(query_path))
            workload_groups = _load_reused_workload_groups(reuse_path, qcat)
            log(f"Reused workload from measurement artifact: queries={qcat.m}, groups={len(workload_groups)}")
        else:
            qcat, workload_groups = build_workload(schema, config)
            log(f"Constructed workload: queries={qcat.m}, groups={len(workload_groups)}")
    if qcat.m <= 0:
        raise ValueError("Configured workload produced no queries")
    schema.validate()
    qcat.validate(schema.cardinalities)
    schema.save_json(output_dir / "schema.json")
    qcat.save_json(output_dir / "queries.json")
    workload_summary = _workload_summary(qcat, workload_groups, schema, config)
    compute_heldout_eval = bool(evaluation_cfg.get("compute_heldout_query_error", False))
    heldout_qcat: QueryCatalogue | None = None
    heldout_workload_groups: list[WorkloadGroup] = []
    if compute_heldout_eval:
        heldout_config = _heldout_workload_config(evaluation_cfg)
        heldout_qcat, heldout_workload_groups = build_workload(schema, {"workload": heldout_config})
        heldout_exclude_measured_queries = bool(evaluation_cfg.get("heldout_exclude_measured_queries", True))
        num_removed_as_measured_duplicates = 0
        if heldout_exclude_measured_queries:
            measured_keys = {query_key(qcat, qid) for qid in range(qcat.m)}
            keep_indices = np.asarray(
                [qid for qid in range(heldout_qcat.m) if query_key(heldout_qcat, qid) not in measured_keys],
                dtype=np.int32,
            )
            num_removed_as_measured_duplicates = int(heldout_qcat.m - len(keep_indices))
            heldout_qcat = filter_query_catalogue(heldout_qcat, keep_indices)
            heldout_workload_groups = filter_workload_groups(heldout_workload_groups, keep_indices)
        heldout_qcat.validate(schema.cardinalities)
        heldout_qcat.save_json(output_dir / "queries_holdout.json")
        heldout_summary = _heldout_workload_summary(
            heldout_qcat,
            heldout_workload_groups,
            heldout_config,
            num_removed_as_measured_duplicates,
            heldout_exclude_measured_queries,
        )
        write_json(heldout_summary, output_dir / "workload_summary_holdout.json")
        log(
            "Constructed held-out workload for offline evaluation only: "
            f"queries={heldout_qcat.m}, groups={len(heldout_workload_groups)}, "
            f"removed_measured_duplicates={num_removed_as_measured_duplicates}"
        )

    if transcript_only_generation:
        if preloaded_measurements is None:
            raise RuntimeError("Internal error: verified public transcript was not retained")
        measurements = preloaded_measurements
        stats.time_measurement_seconds = transcript_verify_seconds
        measurement_json_path = _measurement_json_path(reuse_path)
        log(f"Verified and reused sealed public measurement artifact from {measurement_json_path}")
    elif reuse_path is not None:
        t0 = time.perf_counter()
        _validate_reused_queries(reuse_path, qcat)
        _validate_reused_schema(reuse_path, schema)
        measurement_json_path = _measurement_json_path(reuse_path)
        if not measurement_json_path.exists():
            raise FileNotFoundError(f"measurement.reuse_from does not contain measurements.json: {reuse_path}")
        measurements = measurements_from_public_dict(read_json(measurement_json_path))
        if measurements.target_projected.shape[0] != qcat.m:
            raise ValueError(
                "measurement.reuse_from target length mismatch: "
                f"artifact has {measurements.target_projected.shape[0]}, current workload has {qcat.m}"
            )
        if measurements.num_rows is not None and int(measurements.num_rows) != mechanism_num_rows:
            raise ValueError(
                "measurement.reuse_from row-count mismatch: "
                f"artifact has {measurements.num_rows}, expected public total is {mechanism_num_rows}"
            )
        stats.time_measurement_seconds = time.perf_counter() - t0
        log(f"Reused measurement artifact from {measurement_json_path}")
    else:
        if X_real is None:
            raise RuntimeError("Internal error: private measurement requires an encoded input table")
        t0 = time.perf_counter()
        measurements = measure_real_dataset(
            X_real,
            qcat,
            workload_groups,
            config,
            measurement_rng,
            batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
            cardinalities=schema.cardinalities,
        )
        stats.time_measurement_seconds = time.perf_counter() - t0
    configured_privacy_mode = str(config.get("privacy", {}).get("mode", "dp")).lower()
    if configured_privacy_mode == "dp" and measurements.mode != "dp":
        raise ValueError(
            "privacy.mode=dp cannot reuse or consume a non-DP measurement artifact; "
            f"artifact mode is {measurements.mode!r}"
        )
    write_json(measurements.to_public_dict(), output_dir / "measurements.json")
    if measurements.mode == "dp" and reuse_path is not None:
        log(
            "Reused DP measurement artifact: "
            f"rho_total={measurements.rho_total:.6g}, "
            f"rho_spent={measurements.rho_spent:.6g}, "
            f"epsilon(delta={measurements.delta:.2g})={measurements.epsilon_delta:.6g}, "
            f"measured_queries={qcat.m}, groups={len(measurements.groups)}"
        )
    elif measurements.mode == "dp":
        log(
            "Real DP measurement performed on X_real: "
            f"rho_total={measurements.rho_total:.6g}, "
            f"rho_spent={measurements.rho_spent:.6g}, "
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

    precision_operator_name = str(qdte_cfg.get("precision_operator", "diagonal")).lower()
    optimization_precision: PrecisionOperator | None = None
    precision_extra_diagnostics: dict[str, Any] = {}
    if precision_operator_name in {
        "orthogonal_interaction",
        "orthogonal_interaction_shrink_raw",
        "orthogonal_interaction_shrink_analytic",
        "orthogonal_interaction_p3_raw",
        "orthogonal_interaction_p3_bootdiag",
        "orthogonal_interaction_p3_active_set",
    }:
        if measurements.strategy_transcript is None:
            raise ValueError(
                "orthogonal interaction precision requires a released strategy_transcript "
                "in the measurement artifact"
            )
        interaction_transcript = HierarchicalInteractionTranscript.from_public_dict(
            measurements.strategy_transcript
        )
        if interaction_transcript.strategy.cardinalities != tuple(
            int(value) for value in schema.cardinalities.tolist()
        ):
            raise ValueError("Interaction transcript cardinalities do not match the public schema")
        if interaction_transcript.public_total != mechanism_num_rows:
            raise ValueError("Interaction transcript public_total does not match the declared row count")
        if not np.isclose(
            interaction_transcript.rho_spent,
            measurements.rho_spent,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("Interaction transcript rho ledger does not match the measurement artifact")
        if precision_operator_name == "orthogonal_interaction":
            target_difference = float(
                np.max(
                    np.abs(
                        measurements.target_projected.astype(np.float64)
                        - measurements.target_noisy.astype(np.float64)
                    )
                )
            )
            if target_difference > 1.0e-7:
                raise ValueError(
                    "orthogonal_interaction precision requires target_projected to be the raw "
                    "hierarchical reconstruction"
                )
            optimization_precision = OrthogonalInteractionPrecision(
                qcat,
                workload_groups,
                interaction_transcript,
            )
            log(
                "Loaded exact orthogonal interaction precision: "
                f"rank={optimization_precision.diagnostics()['effective_rank']}, "
                f"dimension={optimization_precision.dimension}"
            )
        elif precision_operator_name in {
            "orthogonal_interaction_shrink_raw",
            "orthogonal_interaction_shrink_analytic",
        }:
            shrinkage_payload = (measurements.projection_diagnostics or {}).get(
                "interaction_shrinkage"
            )
            shrinkage = positive_part_block_james_stein(interaction_transcript)
            validate_frozen_shrinkage_artifact(shrinkage, shrinkage_payload)
            expected_measurements, _ = interaction_transcript_to_shrunk_measurements(
                interaction_transcript,
                qcat,
                workload_groups,
                delta=float(measurements.delta),
            )
            if not np.array_equal(
                measurements.target_projected,
                expected_measurements.target_projected,
            ):
                target_difference = float(
                    np.max(
                        np.abs(
                            measurements.target_projected.astype(np.float64)
                            - expected_measurements.target_projected.astype(np.float64)
                        )
                    )
                )
                raise ValueError(
                    "Interaction shrinkage target does not match the deterministic released "
                    f"post-processing result; max_abs_difference={target_difference:.6g}"
                )
            if not np.array_equal(measurements.target_noisy, expected_measurements.target_noisy):
                raise ValueError("Interaction shrinkage artifact changed the raw released target")
            if precision_operator_name == "orthogonal_interaction_shrink_raw":
                optimization_precision = OrthogonalInteractionPrecision(
                    qcat,
                    workload_groups,
                    interaction_transcript,
                )
                uncertainty_status = "raw_covariance_control_not_propagated"
            else:
                if not shrinkage.analytic_stable:
                    raise ValueError(
                        "Interaction shrinkage is too close to a positive-part kink for the "
                        "frozen analytic covariance profile"
                    )
                optimization_precision = ShrinkageAnalyticOrthogonalInteractionPrecision(
                    qcat,
                    workload_groups,
                    interaction_transcript,
                    shrinkage,
                )
                uncertainty_status = "local_shrinkage_jacobian_covariance"
            precision_extra_diagnostics = {
                "interaction_shrinkage": shrinkage.diagnostics(),
                "uncertainty_status": uncertainty_status,
            }
            log(
                "Loaded deterministic interaction shrinkage target: "
                f"precision={precision_operator_name}, "
                f"affected_blocks={shrinkage.diagnostics()['num_affected_pair_blocks']}, "
                f"factor_min={shrinkage.diagnostics()['factor_min']:.6g}, "
                f"rank={optimization_precision.diagnostics()['effective_rank']}"
            )
        else:
            p3_projection = project_hierarchical_local_polytope(interaction_transcript)
            expected_target = local_polytope_projection_to_query_target(
                p3_projection,
                qcat,
                workload_groups,
            ).astype(np.float32)
            if not np.array_equal(measurements.target_projected, expected_target):
                target_difference = float(
                    np.max(
                        np.abs(
                            measurements.target_projected.astype(np.float64)
                            - expected_target.astype(np.float64)
                        )
                    )
                )
                raise ValueError(
                    "P3 active-set precision target does not match the certified released "
                    f"local-polytope projection; max_abs_difference={target_difference:.6g}"
                )
            if precision_operator_name == "orthogonal_interaction_p3_raw":
                optimization_precision = OrthogonalInteractionPrecision(
                    qcat,
                    workload_groups,
                    interaction_transcript,
                )
                precision_extra_diagnostics = {
                    "local_polytope_p3": dict(p3_projection.diagnostics),
                    "uncertainty_status": "raw_covariance_control_not_propagated",
                }
                log(
                    "Loaded certified P3 target with raw orthogonal covariance control: "
                    f"rank={optimization_precision.diagnostics()['effective_rank']}, "
                    f"dimension={optimization_precision.dimension}"
                )
                active_covariance = None
            elif precision_operator_name == "orthogonal_interaction_p3_bootdiag":
                uncertainty = measurements.projection_diagnostics or {}
                bootstrap_payload = uncertainty.get("coefficient_bootstrap_diagonal")
                if not isinstance(bootstrap_payload, dict):
                    raise ValueError(
                        "P3 coefficient BootDiag precision requires its released bootstrap artifact"
                    )
                coefficient_bootstrap = CoefficientBootstrapDiagonal.from_public_dict(
                    interaction_transcript,
                    bootstrap_payload,
                )
                bootstrap_diagnostics = coefficient_bootstrap.diagnostics
                frozen_bootstrap = bool(
                    bootstrap_diagnostics.get("method")
                    == "coefficient_space_local_polytope_bootstrap_diagonal"
                    and int(bootstrap_diagnostics.get("num_samples", 0)) == 16
                    and bootstrap_diagnostics.get("center") == "projected"
                    and bootstrap_diagnostics.get("debias_target") is False
                    and np.isclose(
                        float(bootstrap_diagnostics.get("min_variance", float("nan"))),
                        1.0e-6,
                    )
                    and np.isclose(
                        float(
                            bootstrap_diagnostics.get(
                                "min_raw_variance_fraction",
                                float("nan"),
                            )
                        ),
                        0.02,
                    )
                )
                if not frozen_bootstrap:
                    raise ValueError("P3 coefficient BootDiag artifact violates the frozen B=16 profile")
                optimization_precision = BootstrapDiagonalOrthogonalInteractionPrecision(
                    qcat,
                    workload_groups,
                    interaction_transcript,
                    coefficient_bootstrap,
                )
                precision_extra_diagnostics = {
                    "local_polytope_p3": dict(p3_projection.diagnostics),
                    "coefficient_bootstrap_diagonal": dict(bootstrap_diagnostics),
                }
                log(
                    "Loaded certified P3 coefficient BootDiag16 precision: "
                    f"rank={optimization_precision.diagnostics()['effective_rank']}, "
                    f"dimension={optimization_precision.dimension}"
                )
                active_covariance = None
            else:
                active_covariance = ActiveSetCoefficientCovariance(
                    interaction_transcript,
                    p3_projection,
                )
            if active_covariance is None:
                covariance_diagnostics = None
            else:
                covariance_diagnostics = active_covariance.diagnostics()
            if active_covariance is None:
                pass
            elif bool(covariance_diagnostics["bootdiag_fallback_recommended"]):
                raise ValueError(
                    "P3 active set is numerically unstable; use coefficient-space BootDiag "
                    "instead of analytic active-set precision"
                )
            else:
                covariance_certificate = active_covariance.certificate(
                    num_probes=(
                        3
                        if covariance_diagnostics["application_mode"] == "dense_svd_reference"
                        else 1
                    ),
                    tolerance=2.0e-7,
                )
                if not covariance_certificate.passed:
                    raise RuntimeError("P3 active-set covariance failed its operator certificate")
                optimization_precision = ActiveSetOrthogonalInteractionPrecision(
                    qcat,
                    workload_groups,
                    interaction_transcript,
                    active_covariance,
                )
                precision_extra_diagnostics = {
                    "local_polytope_p3": dict(p3_projection.diagnostics),
                    "active_set_covariance_certificate": covariance_certificate.to_dict(),
                }
                log(
                    "Loaded certified P3 active-set orthogonal precision: "
                    f"active={covariance_diagnostics['active_constraint_count']}, "
                    f"rank={covariance_diagnostics['effective_rank']}, "
                    f"dimension={optimization_precision.dimension}"
                )

    fission_cfg = measurement_cfg.get("fission", {}) or {}
    fission_enabled = bool(fission_cfg.get("enabled", False))
    fission_split: GaussianFissionSplit | None = None
    fission_projection_diagnostics: dict[str, Any] | None = None
    fission_reconstruction_max_abs = 0.0
    fission_optimization_branch = str(
        fission_cfg.get("optimization_branch", "train")
    ).lower()
    fission_holdout_branch = "validation"
    fission_holdout_noisy: np.ndarray | None = None
    fission_holdout_variance: np.ndarray | None = None
    generation_target_projected = measurements.target_projected.astype(np.float32)
    generation_measurement_variance = measurements.variances.astype(np.float32)
    if fission_enabled:
        raw_variance = _raw_variances_from_measurement_groups(
            measurements.groups,
            qcat.m,
            measurements.variances,
        ).astype(np.float32)
        fission_seed_offset = int(fission_cfg.get("seed_offset", 51_771))
        fission_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0xF15510, fission_seed_offset])
        )
        fission_split = gaussian_fission_split(
            measurements.target_noisy,
            raw_variance,
            train_fraction=float(fission_cfg.get("train_fraction", 0.8)),
            rng=fission_rng,
        )
        if fission_optimization_branch == "train":
            optimization_noisy = fission_split.train_noisy
            optimization_variance = fission_split.train_variances
            fission_holdout_branch = "validation"
            fission_holdout_noisy = fission_split.validation_noisy
            fission_holdout_variance = fission_split.validation_variances
        elif fission_optimization_branch == "validation":
            optimization_noisy = fission_split.validation_noisy
            optimization_variance = fission_split.validation_variances
            fission_holdout_branch = "train"
            fission_holdout_noisy = fission_split.train_noisy
            fission_holdout_variance = fission_split.train_variances
        else:
            raise ValueError(
                "measurement.fission.optimization_branch must be 'train' or 'validation'"
            )
        generation_target_projected, fission_projection_diagnostics = _apply_configured_projection(
            optimization_noisy,
            qcat,
            measurements.groups,
            mechanism_num_rows,
            config.get("projection", {}),
            optimization_variance,
            schema.cardinalities,
        )
        generation_measurement_variance = optimization_variance.astype(np.float32)
        reconstructed = (
            fission_split.train_fraction * fission_split.train_noisy.astype(np.float64)
            + (1.0 - fission_split.train_fraction)
            * fission_split.validation_noisy.astype(np.float64)
        )
        fission_reconstruction_max_abs = float(
            np.max(np.abs(reconstructed - measurements.target_noisy.astype(np.float64)))
        )
        write_json(
            {
                "enabled": True,
                "train_fraction": float(fission_split.train_fraction),
                "gamma": float(fission_split.gamma),
                "seed_offset": int(fission_seed_offset),
                "checkpoint_interval": int(fission_cfg.get("checkpoint_interval", 100)),
                "selection_rule": str(
                    fission_cfg.get("selection_rule", "earliest_within_one_se")
                ),
                "one_se_multiplier": float(fission_cfg.get("one_se_multiplier", 1.0)),
                "optimization_branch": fission_optimization_branch,
                "holdout_branch": fission_holdout_branch,
                "optimization_noisy": optimization_noisy.tolist(),
                "optimization_projected": generation_target_projected.tolist(),
                "optimization_variances": optimization_variance.tolist(),
                "holdout_noisy": fission_holdout_noisy.tolist(),
                "holdout_variances": fission_holdout_variance.tolist(),
                "train_noisy": fission_split.train_noisy.tolist(),
                "train_variances": fission_split.train_variances.tolist(),
                "validation_noisy": fission_split.validation_noisy.tolist(),
                "validation_variances": fission_split.validation_variances.tolist(),
                "original_reconstruction_max_abs": fission_reconstruction_max_abs,
                "train_projection_diagnostics": fission_projection_diagnostics or {},
            },
            output_dir / "measurements_fission.json",
        )
        log(
            "Applied DP post-processing Gaussian fission: "
            f"train_fraction={fission_split.train_fraction:.6g}, "
            f"optimization_branch={fission_optimization_branch}, "
            f"gamma={fission_split.gamma:.6g}, "
            f"reconstruction_max_abs={fission_reconstruction_max_abs:.6g}"
        )

    t0 = time.perf_counter()
    init_cfg = config.get("init", {})
    init_encoded_npy = init_cfg.get("encoded_npy")
    if init_encoded_npy not in {None, ""}:
        X_syn = np.load(Path(str(init_encoded_npy))).astype(np.int32)
        if X_syn.ndim != 2 or X_syn.shape[1] != schema.d:
            raise ValueError(
                "init.encoded_npy shape mismatch: "
                f"expected (*, {schema.d}), got {tuple(X_syn.shape)}"
            )
        if X_syn.shape[0] <= 0:
            raise ValueError("init.encoded_npy must contain at least one synthetic row")
        for attr, cardinality in enumerate(schema.cardinalities):
            values = X_syn[:, attr]
            if np.any(values < 0) or np.any(values >= int(cardinality)):
                raise ValueError(f"init.encoded_npy has out-of-range encoded values for attribute {attr}")
        n_syn = int(X_syn.shape[0])
        log(f"Initialized synthetic table from {init_encoded_npy}: rows={n_syn}, cols={schema.d}")
    else:
        n_syn = _resolve_n_syn(
            init_cfg.get("N_syn", "same_as_real"),
            mechanism_num_rows,
        )
        X_syn = initialize_independent_oneway(qcat, generation_target_projected, schema, n_syn, generation_rng)
    save_npy(X_syn, output_dir / "synthetic_initial_encoded.npy")
    answer_syn = answer_queries(X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
    target = generation_target_projected.astype(np.float32)
    residual = target - answer_syn
    measurement_variance = generation_measurement_variance.astype(np.float32)
    measurement_inv_variance = (1.0 / np.maximum(measurement_variance, 1.0e-12)).astype(np.float32)
    objective_weighting = str(qdte_cfg.get("objective_weighting", "variance")).lower()
    objective_weight_profile = str(
        qdte_cfg.get("objective_weight_profile", "none")
    ).lower()
    objective_loss = str(qdte_cfg.get("objective_loss", "quadratic")).lower()
    objective_loss_weights = _objective_loss_weights(objective_loss, workload_groups, qcat.m)
    objective_query_weight_multipliers = _objective_query_weight_multipliers(
        objective_weight_profile,
        workload_groups,
        qcat.m,
    )
    variance, inv_variance, sigma = _objective_variance_arrays(
        measurement_variance,
        measurement_inv_variance,
        objective_weighting,
    )
    inv_variance = (
        inv_variance * objective_query_weight_multipliers
    ).astype(np.float32)
    if (
        optimization_precision is None
        and str(qdte_cfg.get("score_backend", "dense_gpu")) == "precision_operator"
    ):
        optimization_precision = DiagonalPrecision(inv_variance)
    precision_diagnostics = (
        optimization_precision.diagnostics()
        if optimization_precision is not None
        else DiagonalPrecision(inv_variance).diagnostics()
    )
    precision_diagnostics.update(precision_extra_diagnostics)
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
    log(
        "QDTE objective weighting: "
        f"{objective_weighting}; "
        f"weight_profile={objective_weight_profile}; "
        f"objective_loss={objective_loss}; "
        f"precision_operator={precision_operator_name}; "
        f"objective_variance_mean={float(np.mean(state.variance)):.6g}; "
        f"objective_loss_weight_mean={float(np.mean(objective_loss_weights)):.6g}; "
        f"measurement_variance_mean={float(np.mean(measurement_variance)):.6g}"
    )
    initial_answers = state.answer_syn.copy()
    initial_residual = state.residual.copy()
    initial_loss = measured_loss(state.residual, state.inv_variance)
    initial_optimization_objective = _optimization_objective_value(
        state.residual,
        objective_loss,
        state.inv_variance,
        objective_loss_weights,
        optimization_precision,
    )
    confidence_stop_cfg = qdte_cfg.get("confidence_stop", {}) or {}
    confidence_stop: ChiSquareDiscrepancyStop | None = None
    if bool(confidence_stop_cfg.get("enabled", False)):
        if not isinstance(optimization_precision, OrthogonalInteractionPrecision):
            raise ValueError(
                "chi-square confidence stopping requires exact orthogonal interaction precision"
            )
        confidence_stop = ChiSquareDiscrepancyStop.create(
            alpha=float(confidence_stop_cfg.get("alpha", 0.05)),
            effective_rank=int(precision_diagnostics["effective_rank"]),
        )
        log(
            "Enabled released-only chi-square confidence stop: "
            f"alpha={confidence_stop.alpha:.6g}, "
            f"rank={confidence_stop.effective_rank}, "
            f"objective_threshold={confidence_stop.objective_threshold:.6g}"
        )
    confidence_stop_initial_reached = bool(
        confidence_stop is not None
        and confidence_stop.reached(initial_optimization_objective)
    )
    confidence_stop_triggered = confidence_stop_initial_reached
    confidence_stop_iteration: int | None = 0 if confidence_stop_initial_reached else None
    confidence_stop_objective_at_trigger: float | None = (
        float(initial_optimization_objective) if confidence_stop_initial_reached else None
    )
    if confidence_stop_initial_reached:
        log(
            "Initial synthetic table already satisfies the released confidence set: "
            f"objective={initial_optimization_objective:.6g}, "
            f"threshold={confidence_stop.objective_threshold:.6g}"
        )
    entropy_cfg = qdte_cfg.get("entropy", {}) or {}
    entropy_enabled = bool(entropy_cfg.get("enabled", False))
    entropy_prior: ReleasedProductPrior | None = None
    entropy_state: AtomEntropyState | None = None
    entropy_controller: EntropyDiscrepancyController | None = None
    initial_entropy_regularizer = 0.0
    initial_regularized_objective = initial_optimization_objective
    if entropy_enabled:
        if precision_operator_name != "orthogonal_interaction" or not isinstance(
            optimization_precision,
            OrthogonalInteractionPrecision,
        ):
            raise ValueError(
                "The first entropy profile requires unshrunk exact orthogonal interaction precision"
            )
        if objective_loss != "quadratic" or str(qdte_cfg.get("score_backend")) != "precision_operator":
            raise ValueError("Entropy requires quadratic precision-operator scoring")
        if str(qdte_cfg.get("transport_mode")) != "atom_flow" or str(
            qdte_cfg.get("atom_flow_update_mode", "batch")
        ) != "batch":
            raise ValueError("Entropy requires atom_flow transport with batch updates")
        if bool(qdte_cfg.get("structured_swap_enabled", False)):
            raise ValueError("The first entropy profile does not support structured swaps")
        if str(qdte_cfg.get("candidate_backend", "cpu_repair")) != "cpu_repair":
            raise ValueError("The first entropy profile requires the cpu_repair candidate backend")
        if fission_enabled:
            raise ValueError("The first entropy profile does not compose Gaussian fission")
        if confidence_stop is not None:
            raise ValueError("Entropy uses its own confidence controller; disable confidence_stop")
        if not bool(qdte_cfg.get("allow_below_noise_fallback", False)):
            raise ValueError(
                "Entropy requires allow_below_noise_fallback=true to search within the confidence set"
            )
        smoothing = float(entropy_cfg.get("product_prior_smoothing", 1.0))
        if not np.isclose(smoothing, 1.0, rtol=0.0, atol=0.0):
            raise ValueError("The frozen first entropy profile requires product_prior_smoothing=1.0")
        alpha = float(entropy_cfg.get("alpha", 0.05))
        if not np.isclose(alpha, 0.05, rtol=0.0, atol=0.0):
            raise ValueError("The frozen first entropy profile requires alpha=0.05")
        dual_initial = float(entropy_cfg.get("dual_initial", 1.0))
        if not np.isclose(dual_initial, 1.0, rtol=0.0, atol=0.0):
            raise ValueError("The frozen first entropy profile requires dual_initial=1.0")
        entropy_prior = ReleasedProductPrior.from_released_oneway(
            qcat,
            state.target,
            schema.cardinalities,
            public_total=mechanism_num_rows,
            smoothing=smoothing,
        )
        entropy_state = AtomEntropyState.from_rows(state.X_syn, entropy_prior)
        entropy_controller = EntropyDiscrepancyController.create(
            alpha=alpha,
            effective_rank=int(precision_diagnostics["effective_rank"]),
            max_iterations=int(qdte_cfg.get("max_iters", 5000)),
            dual_initial=dual_initial,
        )
        initial_entropy_regularizer = float(entropy_state.regularizer)
        initial_regularized_objective = entropy_controller.regularized_objective(
            initial_optimization_objective,
            initial_entropy_regularizer,
        )
        log(
            "Enabled confidence-constrained product entropy: "
            f"regularizer={initial_entropy_regularizer:.6g}, "
            f"kl_per_row={initial_entropy_regularizer / n_syn:.6g}, "
            f"data_threshold={entropy_controller.objective_threshold:.6g}, "
            f"dual={entropy_controller.dual_weight:.6g}"
        )
    initial_unweighted_loss = unweighted_measured_loss(state.residual)
    initial_rms = rms_standardized_residual(initial_loss, qcat.m)
    initial_unweighted_rms = rms_unweighted_residual(state.residual)
    heldout_initial_answers: np.ndarray | None = None
    if heldout_qcat is not None:
        heldout_initial_answers = answer_queries(
            state.X_syn,
            heldout_qcat,
            batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
        )
    log(f"Initial measured loss: {initial_loss:.6g}")

    fission_checkpoint_interval = int(fission_cfg.get("checkpoint_interval", 100))
    fission_checkpoint_iterations: list[int] = []
    fission_checkpoint_tables: list[np.ndarray] = []
    fission_checkpoint_answers: list[np.ndarray] = []
    if fission_enabled:
        fission_checkpoint_iterations.append(0)
        fission_checkpoint_tables.append(state.X_syn.copy())
        fission_checkpoint_answers.append(state.answer_syn.copy())

    compute_true_eval = bool(evaluation_cfg.get("compute_true_query_error", True))
    X_real_for_evaluation = X_real if compute_true_eval or compute_heldout_eval else None
    X_real = None
    preprocess_result = None

    max_iters = int(qdte_cfg.get("max_iters", 5000))
    accepted_per_iter = int(qdte_cfg.get("accepted_per_iter", 64))
    transport_mode = str(qdte_cfg.get("transport_mode", "microbatch_greedy"))
    transport_delta_backend = str(qdte_cfg.get("transport_delta_backend", "cpu"))
    if transport_mode == "sequential_greedy":
        accepted_per_iter = 1
    accepted_per_iter_schedule = str(qdte_cfg.get("accepted_per_iter_schedule", "fixed"))
    accepted_per_iter_start = int(qdte_cfg.get("accepted_per_iter_start", accepted_per_iter))
    accepted_per_iter_end = int(qdte_cfg.get("accepted_per_iter_end", 1))
    accepted_per_iter_warmup_iters = int(qdte_cfg.get("accepted_per_iter_warmup_iters", 0))
    accepted_per_iter_anneal_iters = int(
        qdte_cfg.get("accepted_per_iter_anneal_iters", max_iters - accepted_per_iter_warmup_iters)
    )
    atom_flow_pool_multiplier = int(qdte_cfg.get("atom_flow_pool_multiplier", 16))
    atom_flow_max_pool = int(qdte_cfg.get("atom_flow_max_pool", 0))
    atom_flow_update_mode = str(qdte_cfg.get("atom_flow_update_mode", "batch"))
    constructive_pair_pool_multiplier = int(qdte_cfg.get("constructive_pair_pool_multiplier", atom_flow_pool_multiplier))
    constructive_pair_max_pool = int(qdte_cfg.get("constructive_pair_max_pool", atom_flow_max_pool))
    constructive_pair_partner_limit = int(qdte_cfg.get("constructive_pair_partner_limit", 16))
    constructive_pair_harm_query_limit = int(qdte_cfg.get("constructive_pair_harm_query_limit", 16))
    constructive_pair_max_units = int(qdte_cfg.get("constructive_pair_max_units", 0))
    constructive_pair_min_target_component = float(qdte_cfg.get("constructive_pair_min_target_component", 0.0))
    constructive_pair_group_augment = bool(qdte_cfg.get("constructive_pair_group_augment", False))
    constructive_pair_group_augment_trigger = str(
        qdte_cfg.get("constructive_pair_group_augment_trigger", "adaptive")
    ).lower()
    constructive_pair_group_augment_threshold = int(
        qdte_cfg.get("constructive_pair_group_augment_threshold", accepted_per_iter)
    )
    constructive_pair_group_augment_max_loss = float(
        qdte_cfg.get("constructive_pair_group_augment_max_loss", 0.0)
    )
    constructive_pair_group_augment_noise_floor_ratio = float(
        qdte_cfg.get("constructive_pair_group_augment_noise_floor_ratio", 0.0)
    )
    constructive_pair_group_augment_plateau_window = int(
        qdte_cfg.get("constructive_pair_group_augment_plateau_window", 0)
    )
    constructive_pair_group_augment_plateau_relative_drop = float(
        qdte_cfg.get("constructive_pair_group_augment_plateau_relative_drop", 0.0)
    )
    random_group_count = int(qdte_cfg.get("random_group_count", 64))
    random_group_min_size = int(qdte_cfg.get("random_group_min_size", 2))
    random_group_max_size = int(qdte_cfg.get("random_group_max_size", 0))
    random_group_pool_multiplier = int(qdte_cfg.get("random_group_pool_multiplier", 0))
    random_group_max_pool = int(qdte_cfg.get("random_group_max_pool", 0))
    directed_group_seed_count = int(qdte_cfg.get("directed_group_seed_count", 32))
    directed_group_min_size = int(qdte_cfg.get("directed_group_min_size", 1))
    directed_group_max_size = int(qdte_cfg.get("directed_group_max_size", 0))
    directed_group_pool_multiplier = int(qdte_cfg.get("directed_group_pool_multiplier", 0))
    directed_group_max_pool = int(qdte_cfg.get("directed_group_max_pool", 0))
    directed_group_allow_negative_steps = bool(qdte_cfg.get("directed_group_allow_negative_steps", False))
    directed_group_backend = str(qdte_cfg.get("directed_group_backend", "cpu")).lower()
    directed_group_positive_fill = bool(qdte_cfg.get("directed_group_positive_fill", False))
    directed_group_positive_fill_augment = bool(qdte_cfg.get("directed_group_positive_fill_augment", False))
    directed_group_positive_fill_augment_trigger = str(
        qdte_cfg.get("directed_group_positive_fill_augment_trigger", "loss_gate")
    ).lower()
    directed_group_positive_fill_augment_threshold = int(
        qdte_cfg.get("directed_group_positive_fill_augment_threshold", accepted_per_iter)
    )
    directed_group_positive_fill_augment_max_loss = float(
        qdte_cfg.get("directed_group_positive_fill_augment_max_loss", 0.0)
    )
    directed_group_positive_fill_augment_noise_floor_ratio = float(
        qdte_cfg.get("directed_group_positive_fill_augment_noise_floor_ratio", 0.0)
    )
    directed_group_positive_fill_augment_plateau_window = int(
        qdte_cfg.get("directed_group_positive_fill_augment_plateau_window", 0)
    )
    directed_group_positive_fill_augment_plateau_relative_drop = float(
        qdte_cfg.get("directed_group_positive_fill_augment_plateau_relative_drop", 0.0)
    )
    directed_group_augment_seed_count = int(qdte_cfg.get("directed_group_augment_seed_count", directed_group_seed_count))
    directed_group_augment_min_size = int(qdte_cfg.get("directed_group_augment_min_size", directed_group_min_size))
    directed_group_augment_max_size = int(qdte_cfg.get("directed_group_augment_max_size", directed_group_max_size))
    directed_group_augment_pool_multiplier = int(
        qdte_cfg.get("directed_group_augment_pool_multiplier", directed_group_pool_multiplier)
    )
    directed_group_augment_max_pool = int(qdte_cfg.get("directed_group_augment_max_pool", directed_group_max_pool))
    num_active_targets = int(qdte_cfg.get("num_active_targets", 64))
    kappa_noise = float(qdte_cfg.get("kappa_noise", 1.0))
    allow_below_noise_fallback = bool(qdte_cfg.get("allow_below_noise_fallback", False))
    lambda_cost = float(qdte_cfg.get("lambda_cost", 0.01))
    debt_alpha = float(qdte_cfg.get("debt_alpha", 0.0))
    debt_decay = float(qdte_cfg.get("debt_decay", 0.95))
    debt_repay = float(qdte_cfg.get("debt_repay", 1.0))
    debt_cap = float(qdte_cfg.get("debt_cap", 1.0e6))
    min_advantage = float(qdte_cfg.get("min_advantage", 1.0e-6))
    structured_swap_enabled = bool(qdte_cfg.get("structured_swap_enabled", False))
    structured_swap_compiler = str(
        qdte_cfg.get("structured_swap_compiler", "global_random_v1")
    )
    structured_swap_start_iter = max(1, int(qdte_cfg.get("structured_swap_start_iter", 1)))
    structured_swap_interval = max(1, int(qdte_cfg.get("structured_swap_interval", 10)))
    structured_swap_candidate_units = max(1, int(qdte_cfg.get("structured_swap_candidate_units", 2048)))
    structured_swap_transport_pool = max(1, int(qdte_cfg.get("structured_swap_transport_pool", 256)))
    structured_swap_accept_start = max(1, int(qdte_cfg.get("structured_swap_accept_start", 16)))
    structured_swap_accept_end = max(1, int(qdte_cfg.get("structured_swap_accept_end", 1)))
    structured_swap_accept_schedule = str(qdte_cfg.get("structured_swap_accept_schedule", "cosine"))
    structured_swap_noise_guard_kappa = float(qdte_cfg.get("structured_swap_noise_guard_kappa", kappa_noise))
    structured_swap_noise_guard_mode = str(qdte_cfg.get("structured_swap_noise_guard_mode", "fixed"))
    structured_swap_noise_guard_alpha = float(qdte_cfg.get("structured_swap_noise_guard_alpha", 0.05))
    structured_swap_trigger_rms = float(qdte_cfg.get("structured_swap_trigger_rms", 0.0))
    structured_swap_delta_backend = str(qdte_cfg.get("structured_swap_delta_backend", "sparse_cpu"))
    structured_swap_exploration_floor = float(qdte_cfg.get("structured_swap_exploration_floor", 0.20))
    structured_swap_policy_decay = float(qdte_cfg.get("structured_swap_policy_decay", 0.95))
    structured_swap_policy_prior_strength = float(
        qdte_cfg.get("structured_swap_policy_prior_strength", 32.0)
    )
    if structured_swap_accept_start < structured_swap_accept_end:
        raise ValueError("qdte.structured_swap_accept_start must be at least structured_swap_accept_end")
    if structured_swap_noise_guard_kappa < 0.0 or not np.isfinite(structured_swap_noise_guard_kappa):
        raise ValueError("qdte.structured_swap_noise_guard_kappa must be finite and non-negative")
    if structured_swap_noise_guard_mode not in {"fixed", "bonferroni"}:
        raise ValueError(
            "qdte.structured_swap_noise_guard_mode must be one of {'fixed', 'bonferroni'}"
        )
    if not 0.0 < structured_swap_noise_guard_alpha < 1.0:
        raise ValueError("qdte.structured_swap_noise_guard_alpha must be in (0, 1)")
    if structured_swap_trigger_rms < 0.0 or not np.isfinite(structured_swap_trigger_rms):
        raise ValueError("qdte.structured_swap_trigger_rms must be finite and non-negative")
    if structured_swap_delta_backend not in {"sparse_cpu", "dense_gpu"}:
        raise ValueError(
            "qdte.structured_swap_delta_backend must be one of {'sparse_cpu', 'dense_gpu'}"
        )
    if structured_swap_compiler not in {"global_random_v1", INTERACTION_CYCLE_METHOD}:
        raise ValueError(
            "qdte.structured_swap_compiler must be one of "
            "{'global_random_v1', 'interaction_rectangle_v1'}"
        )
    if structured_swap_enabled and structured_swap_compiler == INTERACTION_CYCLE_METHOD:
        if not isinstance(optimization_precision, OrthogonalInteractionPrecision):
            raise ValueError(
                "interaction_rectangle_v1 requires unshrunk exact orthogonal interaction precision"
            )
        if (
            str(qdte_cfg.get("score_backend", "dense_gpu")) != "precision_operator"
            or objective_loss != "quadratic"
        ):
            raise ValueError(
                "interaction_rectangle_v1 requires quadratic precision-operator scoring"
            )
        if structured_swap_delta_backend != "sparse_cpu":
            raise ValueError("interaction_rectangle_v1 requires sparse_cpu query deltas")
    transport_prefix_strategy = str(qdte_cfg.get("transport_prefix_strategy", "largest_positive"))
    stop_patience = int(qdte_cfg.get("stop_patience", 50))
    full_recompute_every = int(qdte_cfg.get("full_recompute_every", 50))
    log_every = int(qdte_cfg.get("log_every", 10))
    candidate_diagnostics_enabled = bool(qdte_cfg.get("candidate_diagnostics", False))
    chunk_size = int(runtime_cfg.get("scoring_chunk_size", 4096))
    use_pmap = bool(runtime_cfg.get("use_pmap", True))
    score_backend = str(qdte_cfg.get("score_backend", "dense_gpu"))
    candidate_backend = str(qdte_cfg.get("candidate_backend", "cpu_repair"))
    use_sparse_delta_backend = transport_delta_backend == "sparse_cpu" or score_backend == "sparse_delta"
    query_delta_index = QueryDeltaIndex.build(qcat, num_attrs=schema.d) if use_sparse_delta_backend else None
    debug_recompute_after_batch = bool(debug_cfg.get("recompute_after_batch", False))
    debug_assert_loss_decrease = bool(debug_cfg.get("assert_batch_loss_decrease", False))
    residual_drift_tolerance = float(debug_cfg.get("residual_drift_tolerance", 1.0e-5))
    loss_tolerance = float(debug_cfg.get("loss_tolerance", 1.0e-4))
    generation_start = time.perf_counter()
    patience = 0
    timeseries: list[dict[str, Any]] = []
    pre_transport_loss_history: list[float] = []
    candidate_diagnostic_rows: list[dict[str, float | int]] = []
    use_gpu_candidate_backend = candidate_backend in {"jax_repair", "gpu_repair"}
    if use_gpu_candidate_backend and objective_loss == "tvd_l1":
        raise ValueError("qdte.objective_loss=tvd_l1 is not supported with GPU fused candidate scoring yet")
    X_syn_gpu = replicate_table_to_devices(state.X_syn) if use_gpu_candidate_backend else None
    gpu_candidate_context = prepare_gpu_candidate_context(qcat, schema, config) if use_gpu_candidate_backend else None
    dense_score_context = (
        prepare_score_context(qcat)
        if (
            not use_gpu_candidate_backend
            and score_backend not in {"target_only", "sparse_delta", "precision_operator"}
        )
        else None
    )
    orthogonal_precision_score_context = (
        prepare_orthogonal_precision_score_context(optimization_precision)
        if isinstance(
            optimization_precision,
            (
                OrthogonalInteractionPrecision,
                ShrinkageAnalyticOrthogonalInteractionPrecision,
            ),
        )
        else None
    )
    structured_swap_score_context = prepare_hybrid_score_context(qcat) if structured_swap_enabled else None
    structured_swap_delta_index = (
        QueryDeltaIndex.build(qcat, num_attrs=schema.d) if structured_swap_enabled else None
    )
    structured_swap_policy = (
        AdaptiveSwapAttributePolicy.create(
            schema,
            exploration_floor=structured_swap_exploration_floor,
            decay=structured_swap_policy_decay,
            prior_strength=structured_swap_policy_prior_strength,
        )
        if structured_swap_enabled and structured_swap_compiler == "global_random_v1"
        else None
    )
    structured_swap_rng = (
        np.random.default_rng(np.random.SeedSequence([seed, 0x51574450]))
        if structured_swap_enabled
        else None
    )
    structured_swap_schedule_cfg = {
        "accepted_per_iter_schedule": structured_swap_accept_schedule,
        "accepted_per_iter_start": structured_swap_accept_start,
        "accepted_per_iter_end": structured_swap_accept_end,
        "accepted_per_iter_warmup_iters": 0,
        "accepted_per_iter_anneal_iters": max_iters,
    }
    structured_swap_total_candidates = 0
    structured_swap_total_accepted = 0
    structured_swap_total_noise_guard_rejections = 0
    structured_swap_total_objective_advantage = 0.0
    structured_swap_compiler_calls = 0
    structured_swap_total_positive_rectangles = 0
    structured_swap_max_selected_scopes = 0
    structured_swap_target_linear_gain_sum = 0.0
    structured_swap_target_linear_gain_count = 0
    structured_swap_last_compiler_diagnostics: dict[str, Any] = {}
    structured_swap_max_effective_noise_guard_kappa = structured_swap_noise_guard_kappa
    structured_swap_time_seconds = 0.0
    structured_swap_generation_time_seconds = 0.0
    structured_swap_scoring_time_seconds = 0.0
    structured_swap_policy_pool_time_seconds = 0.0
    structured_swap_delta_time_seconds = 0.0
    structured_swap_transport_time_seconds = 0.0
    structured_swap_apply_time_seconds = 0.0
    configured_total_candidates = int(
        qdte_cfg.get("total_candidates_per_iter", max(1, num_active_targets * int(qdte_cfg.get("candidates_per_target", 64))))
    )
    gpu_return_top_k = int(qdte_cfg.get("gpu_return_top_k", 0))
    per_device_total_candidates = int(math.ceil(configured_total_candidates / max(1, jax.local_device_count())))
    gpu_topk_return_mode = bool(
        use_gpu_candidate_backend and 0 < gpu_return_top_k < per_device_total_candidates
    )
    last_debt_diagnostics = debt_diagnostics(state.debt)
    last_transport_diagnostics: dict[str, float | int] = {}
    if optimization_precision is not None:
        objective_noise_floor_loss = 0.5 * float(precision_diagnostics["effective_rank"])
    else:
        objective_noise_floor_loss = float(
            0.5
            * np.sum(
                measurement_variance.astype(np.float64, copy=False)
                * state.inv_variance.astype(np.float64, copy=False)
            )
        )
    objective_advantage_noise_variance = (
        measurement_variance.astype(np.float64)
        * state.inv_variance.astype(np.float64) ** 2
    ).astype(np.float32)

    effective_max_iters = 0 if confidence_stop_initial_reached else max_iters
    for iteration in range(1, effective_max_iters + 1):
        iter_start = time.perf_counter()
        state.iteration = iteration
        entropy_iteration_data_objective: float | None = None
        if entropy_controller is not None:
            entropy_iteration_data_objective = _optimization_objective_value(
                state.residual,
                objective_loss,
                state.inv_variance,
                objective_loss_weights,
                optimization_precision,
            )
            entropy_controller.update_dual(entropy_iteration_data_objective)
        accept_limit = _scheduled_accept_limit(
            iteration=iteration,
            max_iters=max_iters,
            base_accept=accepted_per_iter,
            qdte_cfg=qdte_cfg,
        )
        debt_info = debt_diagnostics(state.debt)
        active = select_active_queries(
            state.residual,
            _objective_selection_sigma(objective_loss_weights) if objective_loss == "tvd_l1" else state.sigma,
            state.debt,
            num_active_targets=num_active_targets,
            kappa_noise=kappa_noise,
            debt_alpha=debt_alpha,
            allow_below_noise_fallback=allow_below_noise_fallback,
        )
        if len(active) == 0:
            patience += 1
            cur_loss = measured_loss(state.residual, state.inv_variance)
            cur_optimization_objective = _optimization_objective_value(
                state.residual,
                objective_loss,
                state.inv_variance,
                objective_loss_weights,
                optimization_precision,
            )
            cur_unweighted_loss = unweighted_measured_loss(state.residual)
            if iteration == 1 or iteration % log_every == 0:
                timeseries.append(
                    {
                        "iteration": iteration,
                        "wall_time": time.perf_counter() - stats.start_time,
                        "measured_loss": cur_loss,
                        "optimization_objective": cur_optimization_objective,
                        "unweighted_measured_loss": cur_unweighted_loss,
                        "rms_standardized_residual": rms_standardized_residual(cur_loss, qcat.m),
                        "rms_unweighted_residual": rms_unweighted_residual(state.residual),
                        "residual_l2": float(np.linalg.norm(state.residual)),
                        "residual_l1": float(np.sum(np.abs(state.residual))),
                        "active_queries": 0,
                        "num_candidates": 0,
                        "raw_returned_candidates_before_edge_dedup": 0,
                        "duplicate_edge_candidates": 0,
                        "duplicate_edge_rate": 0.0,
                        "edge_candidates_trimmed_to_return_limit": 0,
                        "candidates_scored_this_iter": 0,
                        "positive_advantage_rate": 0.0,
                        "positive_returned_rate": 0.0,
                        "selected_nonconflicting": 0,
                        "accept_limit": int(accept_limit),
                        "accepted_edits": 0,
                        "accepted_rate": 0.0,
                        "mean_advantage": 0.0,
                        "batch_advantage": 0.0,
                        "requested_candidates": 0,
                        "directed_candidate_budget": 0,
                        "random_candidate_budget": 0,
                        "directed_candidates": 0,
                        "random_candidates": 0,
                        "planned_random_candidates": 0,
                        "fallback_random_candidates": 0,
                        "mixture_random_candidates": 0,
                        "qdte_mixture_candidates": 0,
                        "directed_candidate_shortfall": 0,
                        "candidate_shortfall": 0,
                        "source_filter_attempts": 0,
                        "source_filter_failures": 0,
                        "incremental_answer_drift": 0.0,
                        **debt_info,
                    }
                )
            log(
                "No active queries above noise threshold at "
                f"iter={iteration}: patience={patience}, kappa_noise={kappa_noise:.6g}"
            )
            stats.num_iterations = iteration
            stats.time_generation_seconds += time.perf_counter() - iter_start
            if patience >= stop_patience:
                log(f"Stopping at iter={iteration}: patience={patience}")
                break
            continue
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
                generation_rng,
                context=gpu_candidate_context,
            )
            candidates = gpu_batch.candidates
            fused_advantages = gpu_batch.advantages
            stats.time_scoring_seconds += time.perf_counter() - t_candidate
        else:
            candidates = generate_candidates(
                state.X_syn,
                qcat,
                schema,
                active,
                state.residual,
                config,
                generation_rng,
                inv_variance=objective_loss_weights if objective_loss == "tvd_l1" else state.inv_variance,
            )
            stats.time_candidate_generation_seconds += time.perf_counter() - t_candidate
        diag = candidates.diagnostics
        stats.num_candidates_requested += int(diag.get("requested_candidates", candidates.size))
        stats.num_candidate_shortfall += int(diag.get("candidate_shortfall", 0.0))
        stats.num_directed_candidates += int(diag.get("directed_candidates", 0.0))
        stats.num_random_candidates += int(diag.get("random_candidates", 0.0))
        stats.num_planned_random_candidates += int(diag.get("planned_random_candidates", 0.0))
        stats.num_fallback_random_candidates += int(diag.get("fallback_random_candidates", 0.0))
        stats.num_mixture_random_candidates += int(diag.get("mixture_random_candidates", 0.0))
        stats.num_directed_candidate_shortfall += int(diag.get("directed_candidate_shortfall", 0.0))
        stats.num_source_filter_attempts += int(diag.get("source_filter_attempts", 0.0))
        stats.num_source_filter_failures += int(diag.get("source_filter_failures", 0.0))
        stats.num_candidates_returned_to_cpu += int(candidates.size)
        stats.num_raw_candidates_returned_before_edge_dedup += int(
            diag.get("raw_returned_candidates_before_edge_dedup", candidates.size)
        )
        stats.num_duplicate_edge_candidates_filtered += int(diag.get("duplicate_edge_candidates", 0.0))
        stats.num_edge_candidates_trimmed_to_return_limit += int(
            diag.get("edge_candidates_trimmed_to_return_limit", 0.0)
        )
        candidates_scored_this_iter = int(diag.get("scored_candidates", candidates.size))
        stats.num_candidates_scored += candidates_scored_this_iter

        if candidates.size == 0:
            patience += 1
            stats.num_iterations = iteration
            if patience >= stop_patience:
                log(f"Stopping at iter={iteration}: no candidates for {patience} iterations")
                break
            continue

        precision_candidate_quadratic: np.ndarray | None = None
        scoring_lambda_cost = 0.0 if entropy_state is not None else lambda_cost
        if fused_advantages is None:
            t_score = time.perf_counter()
            if objective_loss == "tvd_l1":
                advantages = score_candidates_l1(
                    candidates,
                    state.residual,
                    objective_loss_weights,
                    qcat,
                    lambda_cost=scoring_lambda_cost,
                    chunk_size=chunk_size,
                    use_pmap=use_pmap,
                    context=dense_score_context,
                )
            elif score_backend == "target_only":
                advantages = score_candidates_target_only(
                    candidates,
                    state.residual,
                    state.inv_variance,
                    qcat,
                    lambda_cost=scoring_lambda_cost,
                )
            elif score_backend == "sparse_delta":
                if query_delta_index is None:
                    raise RuntimeError("Internal error: sparse_delta score backend requires QueryDeltaIndex.")
                advantages = score_candidates_sparse(
                    candidates,
                    state.residual,
                    state.inv_variance,
                    query_delta_index,
                    lambda_cost=scoring_lambda_cost,
                )
            elif score_backend == "precision_operator":
                if optimization_precision is None:
                    raise RuntimeError("Precision-operator scoring was not initialized")
                if isinstance(
                    optimization_precision,
                    (
                        OrthogonalInteractionPrecision,
                        ShrinkageAnalyticOrthogonalInteractionPrecision,
                    ),
                ):
                    advantages, precision_candidate_quadratic = (
                        score_candidates_orthogonal_precision_with_quadratic(
                            candidates,
                            state.residual,
                            optimization_precision,
                            lambda_cost=scoring_lambda_cost,
                            chunk_size=chunk_size,
                            context=orthogonal_precision_score_context,
                        )
                    )
                elif isinstance(
                    optimization_precision,
                    ActiveSetOrthogonalInteractionPrecision,
                ):
                    advantages, precision_candidate_quadratic = (
                        score_candidates_active_set_orthogonal_precision_with_quadratic(
                            candidates,
                            state.residual,
                            optimization_precision,
                            lambda_cost=scoring_lambda_cost,
                            chunk_size=chunk_size,
                        )
                    )
                elif isinstance(
                    optimization_precision,
                    BootstrapDiagonalOrthogonalInteractionPrecision,
                ):
                    advantages, precision_candidate_quadratic = (
                        score_candidates_bootstrap_orthogonal_precision_with_quadratic(
                            candidates,
                            state.residual,
                            optimization_precision,
                            lambda_cost=scoring_lambda_cost,
                            chunk_size=chunk_size,
                        )
                    )
                else:
                    advantages = score_candidates_precision(
                        candidates,
                        state.residual,
                        optimization_precision,
                        qcat,
                        lambda_cost=scoring_lambda_cost,
                        chunk_size=chunk_size,
                        delta_index=(
                            query_delta_index
                            if transport_delta_backend == "sparse_cpu"
                            else None
                        ),
                    )
            else:
                advantages = score_candidates(
                    candidates,
                    state.residual,
                    state.inv_variance,
                    qcat,
                    lambda_cost=scoring_lambda_cost,
                    chunk_size=chunk_size,
                    use_pmap=use_pmap,
                    context=dense_score_context,
                )
            stats.time_scoring_seconds += time.perf_counter() - t_score
        else:
            advantages = fused_advantages
        entropy_data_candidate_gains: np.ndarray | None = None
        entropy_candidate_gains: np.ndarray | None = None
        if entropy_state is not None:
            if entropy_controller is None or optimization_precision is None:
                raise RuntimeError("Entropy state was initialized without its controller or precision")
            if fused_advantages is not None:
                raise RuntimeError("Entropy does not support fused GPU candidate scoring")
            if entropy_iteration_data_objective is None:
                raise RuntimeError("Entropy iteration data objective was not initialized")
            t_entropy_score = time.perf_counter()
            entropy_data_candidate_gains = np.asarray(advantages, dtype=np.float64)
            entropy_candidate_gains = entropy_state.candidate_gains(
                candidates.old_rows,
                candidates.new_rows,
            )
            entropy_controller.calibrate_feasibility_floor(
                data_objective=entropy_iteration_data_objective,
                data_gains=entropy_data_candidate_gains,
                entropy_gains=entropy_candidate_gains,
                edit_costs=candidates.edit_cost,
                lambda_cost=lambda_cost,
                min_advantage=min_advantage,
            )
            advantages = entropy_controller.candidate_advantages(
                data_objective=entropy_iteration_data_objective,
                data_gains=entropy_data_candidate_gains,
                entropy_gains=entropy_candidate_gains,
                edit_costs=candidates.edit_cost,
                lambda_cost=lambda_cost,
            ).astype(np.float32)
            if precision_candidate_quadratic is not None:
                precision_candidate_quadratic = (
                    np.asarray(precision_candidate_quadratic, dtype=np.float64)
                    * entropy_controller.dual_weight
                ).astype(np.float32)
            stats.time_scoring_seconds += time.perf_counter() - t_entropy_score
        positive_returned_count = int(np.sum(advantages > min_advantage)) if len(advantages) else 0
        stats.num_positive_returned_candidates += positive_returned_count
        positive_rate = float(positive_returned_count / max(1, len(advantages)))

        t_transport = time.perf_counter()
        before_loss = measured_loss(state.residual, state.inv_variance)
        before_optimization_objective = _optimization_objective_value(
            state.residual,
            objective_loss,
            state.inv_variance,
            objective_loss_weights,
            optimization_precision,
        )
        before_entropy_regularizer = (
            float(entropy_state.regularizer) if entropy_state is not None else 0.0
        )
        before_regularized_objective = (
            entropy_controller.regularized_objective(
                before_optimization_objective,
                before_entropy_regularizer,
            )
            if entropy_controller is not None
            else before_optimization_objective
        )
        pre_transport_loss_history.append(float(before_loss))
        residual_before_debt = state.residual.copy()
        loss_vec_before_debt = _optimization_loss_vector(
            residual_before_debt,
            objective_loss,
            state.inv_variance,
            objective_loss_weights,
        )
        entropy_prefix_evaluator = None
        if entropy_state is not None:
            if entropy_controller is None or optimization_precision is None:
                raise RuntimeError("Entropy prefix evaluation is missing its controller or precision")

            def entropy_prefix_evaluator(
                selected_indices: np.ndarray,
                delta_prefix: np.ndarray,
                cost_prefix: np.ndarray,
            ) -> np.ndarray:
                prefix_indices = np.asarray(selected_indices, dtype=np.int32)
                data_gains = optimization_precision.advantages(
                    state.residual,
                    np.asarray(delta_prefix, dtype=np.float64),
                    np.zeros(len(prefix_indices), dtype=np.float64),
                    0.0,
                )
                regularizer_gains = entropy_state.prefix_gains(
                    candidates.old_rows[prefix_indices],
                    candidates.new_rows[prefix_indices],
                )
                return entropy_controller.prefix_advantages(
                    data_objective=before_optimization_objective,
                    data_gains=data_gains,
                    entropy_gains=regularizer_gains,
                    edit_costs=np.asarray(cost_prefix, dtype=np.float64),
                    lambda_cost=lambda_cost,
                )

        if transport_mode == "blind_accept":
            selected = select_nonconflicting_in_order(
                candidates,
                np.arange(candidates.size, dtype=np.int32),
                accept_limit,
            )
            stats.num_selected_nonconflicting_candidates += int(len(selected))
            if len(selected) > 0 and transport_delta_backend == "sparse_cpu":
                if query_delta_index is None:
                    raise RuntimeError("Internal error: sparse_cpu transport backend requires QueryDeltaIndex.")
                deltas = compute_deltas_sparse(
                    candidates.old_rows[selected],
                    candidates.new_rows[selected],
                    query_delta_index,
                )
            elif len(selected) > 0:
                deltas = compute_deltas(candidates.old_rows[selected], candidates.new_rows[selected], qcat)
            else:
                deltas = np.empty((0, qcat.m), dtype=np.int8)
            transport = choose_blind_transport(
                candidates,
                deltas,
                selected,
                state.residual,
                state.inv_variance,
                lambda_cost,
            )
        elif transport_mode == "atom_flow":
            selected = np.empty(0, dtype=np.int32)
            if atom_flow_update_mode == "exact":
                if objective_loss == "tvd_l1":
                    raise ValueError("qdte.objective_loss=tvd_l1 supports atom_flow_update_mode=batch only")
                transport = choose_atom_flow_transport(
                    candidates,
                    advantages,
                    state.residual,
                    state.inv_variance,
                    lambda_cost,
                    qcat,
                    max_accept=accept_limit,
                    min_advantage=min_advantage,
                    pool_multiplier=atom_flow_pool_multiplier,
                    max_pool=atom_flow_max_pool,
                    delta_index=query_delta_index if transport_delta_backend == "sparse_cpu" else None,
                )
            elif atom_flow_update_mode == "batch":
                transport = choose_atom_flow_batch_transport(
                    candidates,
                    advantages,
                    state.residual,
                    state.inv_variance,
                    lambda_cost,
                    qcat,
                    max_accept=accept_limit,
                    min_advantage=min_advantage,
                    pool_multiplier=atom_flow_pool_multiplier,
                    max_pool=atom_flow_max_pool,
                    prefix_strategy=transport_prefix_strategy,
                    delta_index=query_delta_index if transport_delta_backend == "sparse_cpu" else None,
                    objective="l1" if objective_loss == "tvd_l1" else "quadratic",
                    objective_weights=objective_loss_weights if objective_loss == "tvd_l1" else None,
                    precision_operator=optimization_precision,
                    candidate_quadratic=precision_candidate_quadratic,
                    prefix_advantage_evaluator=entropy_prefix_evaluator,
                )
            else:
                raise ValueError(f"Unknown qdte.atom_flow_update_mode={atom_flow_update_mode!r}")
            stats.num_selected_nonconflicting_candidates += int(
                transport.diagnostics.get("atom_flow_pool_candidates", 0)
            )
        elif transport_mode == "constructive_pair":
            selected = np.empty(0, dtype=np.int32)
            transport = choose_constructive_pair_transport(
                candidates,
                advantages,
                state.residual,
                state.inv_variance,
                lambda_cost,
                qcat,
                max_accept=accept_limit,
                min_advantage=min_advantage,
                pool_multiplier=constructive_pair_pool_multiplier,
                max_pool=constructive_pair_max_pool,
                partner_limit=constructive_pair_partner_limit,
                harm_query_limit=constructive_pair_harm_query_limit,
                max_units=constructive_pair_max_units,
                min_target_component=constructive_pair_min_target_component,
                prefix_strategy=transport_prefix_strategy,
                delta_index=query_delta_index if transport_delta_backend == "sparse_cpu" else None,
            )
            if constructive_pair_group_augment:
                remaining_accept = int(accept_limit) - int(len(transport.accepted_indices))
                current_loss_for_augment = measured_loss(state.residual, state.inv_variance)
                augment_manual_loss_gate_passed = (
                    constructive_pair_group_augment_max_loss <= 0.0
                    or current_loss_for_augment <= constructive_pair_group_augment_max_loss
                )
                noise_floor_gate_loss = objective_noise_floor_loss * constructive_pair_group_augment_noise_floor_ratio
                augment_noise_floor_gate_passed = (
                    constructive_pair_group_augment_noise_floor_ratio > 0.0
                    and current_loss_for_augment <= noise_floor_gate_loss
                )
                plateau_reference_loss = 0.0
                augment_plateau_gate_passed = False
                if (
                    constructive_pair_group_augment_plateau_window > 1
                    and constructive_pair_group_augment_plateau_relative_drop >= 0.0
                    and len(pre_transport_loss_history) >= constructive_pair_group_augment_plateau_window
                ):
                    plateau_reference_loss = float(
                        pre_transport_loss_history[-constructive_pair_group_augment_plateau_window]
                    )
                    recent_relative_drop = (
                        (plateau_reference_loss - current_loss_for_augment)
                        / max(abs(plateau_reference_loss), 1.0e-9)
                    )
                    augment_plateau_gate_passed = (
                        recent_relative_drop <= constructive_pair_group_augment_plateau_relative_drop
                    )
                else:
                    recent_relative_drop = float("inf")

                if constructive_pair_group_augment_trigger == "always":
                    augment_trigger_passed = True
                elif constructive_pair_group_augment_trigger == "loss_gate":
                    augment_trigger_passed = augment_manual_loss_gate_passed
                elif constructive_pair_group_augment_trigger == "adaptive":
                    augment_trigger_passed = (
                        augment_noise_floor_gate_passed
                        or augment_plateau_gate_passed
                        or (
                            constructive_pair_group_augment_max_loss > 0.0
                            and augment_manual_loss_gate_passed
                        )
                    )
                else:
                    raise ValueError(
                        "qdte.constructive_pair_group_augment_trigger must be one of: "
                        "always, loss_gate, adaptive"
                    )

                should_augment = (
                    remaining_accept > 0
                    and int(len(transport.accepted_indices)) <= int(constructive_pair_group_augment_threshold)
                    and augment_trigger_passed
                )
                transport.diagnostics.update(
                    {
                        "constructive_pair_group_augment": 1,
                        "constructive_pair_group_augment_attempted": 0,
                        "constructive_pair_group_augment_accepted": 0,
                        "constructive_pair_group_augment_threshold": int(
                            constructive_pair_group_augment_threshold
                        ),
                        "constructive_pair_group_augment_trigger_adaptive": int(
                            constructive_pair_group_augment_trigger == "adaptive"
                        ),
                        "constructive_pair_group_augment_current_loss": float(current_loss_for_augment),
                        "constructive_pair_group_augment_max_loss": float(
                            constructive_pair_group_augment_max_loss
                        ),
                        "constructive_pair_group_augment_loss_gate_passed": int(augment_trigger_passed),
                        "constructive_pair_group_augment_manual_loss_gate_passed": int(
                            augment_manual_loss_gate_passed
                        ),
                        "constructive_pair_group_augment_noise_floor_loss": float(objective_noise_floor_loss),
                        "constructive_pair_group_augment_noise_floor_ratio": float(
                            constructive_pair_group_augment_noise_floor_ratio
                        ),
                        "constructive_pair_group_augment_noise_floor_gate_passed": int(
                            augment_noise_floor_gate_passed
                        ),
                        "constructive_pair_group_augment_plateau_window": int(
                            constructive_pair_group_augment_plateau_window
                        ),
                        "constructive_pair_group_augment_plateau_reference_loss": float(plateau_reference_loss),
                        "constructive_pair_group_augment_plateau_relative_drop": float(recent_relative_drop),
                        "constructive_pair_group_augment_plateau_gate_passed": int(
                            augment_plateau_gate_passed
                        ),
                    }
                )
                if should_augment:
                    blocked_rows = candidates.row_ids[transport.accepted_indices]
                    augment_advantages = advantages.astype(np.float32, copy=True)
                    if len(blocked_rows) > 0:
                        augment_advantages[np.isin(candidates.row_ids, blocked_rows)] = -np.inf
                    virtual_residual = (state.residual - transport.delta_sum).astype(np.float32)
                    if directed_group_backend in {"jax", "gpu"}:
                        augment_transport = choose_directed_group_transport_jax(
                            candidates,
                            augment_advantages,
                            virtual_residual,
                            state.inv_variance,
                            lambda_cost,
                            qcat,
                            max_accept=remaining_accept,
                            min_advantage=min_advantage,
                            seed_count=directed_group_augment_seed_count,
                            min_group_size=directed_group_augment_min_size,
                            max_group_size=directed_group_augment_max_size,
                            pool_multiplier=directed_group_augment_pool_multiplier,
                            max_pool=directed_group_augment_max_pool,
                            allow_negative_steps=directed_group_allow_negative_steps,
                        )
                    else:
                        augment_transport = choose_directed_group_transport(
                            candidates,
                            augment_advantages,
                            virtual_residual,
                            state.inv_variance,
                            lambda_cost,
                            qcat,
                            max_accept=remaining_accept,
                            min_advantage=min_advantage,
                            seed_count=directed_group_augment_seed_count,
                            min_group_size=directed_group_augment_min_size,
                            max_group_size=directed_group_augment_max_size,
                            pool_multiplier=directed_group_augment_pool_multiplier,
                            max_pool=directed_group_augment_max_pool,
                            allow_negative_steps=directed_group_allow_negative_steps,
                            delta_index=query_delta_index if transport_delta_backend == "sparse_cpu" else None,
                        )
                    transport.diagnostics.update(
                        {
                            "constructive_pair_group_augment_attempted": 1,
                            "constructive_pair_group_augment_accepted": int(
                                len(augment_transport.accepted_indices)
                            ),
                            "constructive_pair_group_augment_pool_candidates": int(
                                augment_transport.diagnostics.get("directed_group_pool_candidates", 0)
                            ),
                            "constructive_pair_group_augment_groups_evaluated": int(
                                augment_transport.diagnostics.get("directed_group_groups_evaluated", 0)
                            ),
                        }
                    )
                    if len(augment_transport.accepted_indices) > 0:
                        combined_indices = np.concatenate(
                            [transport.accepted_indices, augment_transport.accepted_indices]
                        ).astype(np.int32, copy=False)
                        combined_delta = (
                            transport.delta_sum.astype(np.float32)
                            + augment_transport.delta_sum.astype(np.float32)
                        )
                        combined_cost = float(candidates.edit_cost[combined_indices].sum())
                        combined_advantage = batch_advantage(
                            state.residual,
                            state.inv_variance,
                            combined_delta,
                            combined_cost,
                            lambda_cost,
                        )
                        combined_diagnostics = dict(transport.diagnostics)
                        combined_diagnostics.update(augment_transport.diagnostics)
                        combined_diagnostics.update(
                            {
                                "constructive_pair_group_augment": 1,
                                "constructive_pair_group_augment_attempted": 1,
                                "constructive_pair_group_augment_accepted": int(
                                    len(augment_transport.accepted_indices)
                                ),
                                "constructive_pair_group_augment_total_accepted": int(len(combined_indices)),
                                "constructive_pair_group_augment_combined_batch_advantage": float(
                                    combined_advantage
                                ),
                            }
                        )
                        transport = transport.__class__(
                            accepted_indices=combined_indices,
                            delta_sum=combined_delta.astype(np.float32, copy=False),
                            batch_advantage=float(combined_advantage),
                            mean_advantage=float(advantages[combined_indices].mean()),
                            diagnostics=combined_diagnostics,
                        )
            selected = transport.accepted_indices
            stats.num_selected_nonconflicting_candidates += int(
                transport.diagnostics.get("constructive_pair_selected_candidates", len(transport.accepted_indices))
            )
        elif transport_mode == "random_group":
            selected = np.empty(0, dtype=np.int32)
            transport = choose_random_group_transport(
                candidates,
                advantages,
                state.residual,
                state.inv_variance,
                lambda_cost,
                qcat,
                max_accept=accept_limit,
                min_advantage=min_advantage,
                rng=generation_rng,
                group_count=random_group_count,
                min_group_size=random_group_min_size,
                max_group_size=random_group_max_size,
                pool_multiplier=random_group_pool_multiplier,
                max_pool=random_group_max_pool,
                delta_index=query_delta_index if transport_delta_backend == "sparse_cpu" else None,
            )
            selected = transport.accepted_indices
            stats.num_selected_nonconflicting_candidates += int(
                transport.diagnostics.get("random_group_groups_evaluated", 0)
            )
        elif transport_mode == "directed_group":
            selected = np.empty(0, dtype=np.int32)
            transport = None
            positive_fill_diagnostics: dict[str, float | int] = {}
            if directed_group_positive_fill:
                fill_selected = select_top_nonconflicting(candidates, advantages, accept_limit, min_advantage)
                if len(fill_selected) > 0:
                    if directed_group_backend in {"jax", "gpu"} or transport_delta_backend in {"jax_prefix", "gpu_prefix"}:
                        fill_transport = choose_transport_batch_jax(
                            candidates,
                            advantages,
                            fill_selected,
                            state.residual,
                            state.inv_variance,
                            lambda_cost,
                            qcat,
                            prefix_strategy=transport_prefix_strategy,
                        )
                    else:
                        if transport_delta_backend == "sparse_cpu":
                            if query_delta_index is None:
                                raise RuntimeError(
                                    "Internal error: sparse_cpu transport backend requires QueryDeltaIndex."
                                )
                            fill_deltas = compute_deltas_sparse(
                                candidates.old_rows[fill_selected],
                                candidates.new_rows[fill_selected],
                                query_delta_index,
                            )
                        else:
                            fill_deltas = compute_deltas(
                                candidates.old_rows[fill_selected],
                                candidates.new_rows[fill_selected],
                                qcat,
                            )
                        fill_transport = choose_transport_batch(
                            candidates,
                            advantages,
                            fill_deltas,
                            fill_selected,
                            state.residual,
                            state.inv_variance,
                            lambda_cost,
                            prefix_strategy=transport_prefix_strategy,
                        )
                    fill_transport.diagnostics.update(
                        {
                            "directed_group_positive_fill": 1,
                            "directed_group_positive_fill_selected": int(len(fill_selected)),
                            "directed_group_positive_fill_accepted": int(len(fill_transport.accepted_indices)),
                            "directed_group_positive_fill_batch_advantage": float(fill_transport.batch_advantage),
                            "directed_group_positive_fill_augment_enabled": int(
                                directed_group_positive_fill_augment
                            ),
                            "directed_group_positive_fill_augment_attempted": 0,
                            "directed_group_positive_fill_augment_accepted": 0,
                            "directed_group_positive_fill_augment_threshold": int(
                                directed_group_positive_fill_augment_threshold
                            ),
                            "directed_group_positive_fill_total_accepted": int(
                                len(fill_transport.accepted_indices)
                            ),
                        }
                    )
                    positive_fill_diagnostics = dict(fill_transport.diagnostics)
                    if len(fill_transport.accepted_indices) > 0:
                        transport = fill_transport
                        remaining_accept = int(accept_limit) - int(len(fill_transport.accepted_indices))
                        current_loss_for_augment = measured_loss(state.residual, state.inv_variance)
                        augment_manual_loss_gate_passed = (
                            directed_group_positive_fill_augment_max_loss <= 0.0
                            or current_loss_for_augment <= directed_group_positive_fill_augment_max_loss
                        )
                        noise_floor_gate_loss = (
                            objective_noise_floor_loss
                            * directed_group_positive_fill_augment_noise_floor_ratio
                        )
                        augment_noise_floor_gate_passed = (
                            directed_group_positive_fill_augment_noise_floor_ratio > 0.0
                            and current_loss_for_augment <= noise_floor_gate_loss
                        )
                        plateau_reference_loss = 0.0
                        augment_plateau_gate_passed = False
                        if (
                            directed_group_positive_fill_augment_plateau_window > 1
                            and directed_group_positive_fill_augment_plateau_relative_drop >= 0.0
                            and len(pre_transport_loss_history)
                            >= directed_group_positive_fill_augment_plateau_window
                        ):
                            plateau_reference_loss = float(
                                pre_transport_loss_history[-directed_group_positive_fill_augment_plateau_window]
                            )
                            recent_relative_drop = (
                                (plateau_reference_loss - current_loss_for_augment)
                                / max(abs(plateau_reference_loss), 1.0e-9)
                            )
                            augment_plateau_gate_passed = (
                                recent_relative_drop
                                <= directed_group_positive_fill_augment_plateau_relative_drop
                            )
                        else:
                            recent_relative_drop = float("inf")

                        if directed_group_positive_fill_augment_trigger == "always":
                            augment_trigger_passed = True
                        elif directed_group_positive_fill_augment_trigger == "loss_gate":
                            augment_trigger_passed = augment_manual_loss_gate_passed
                        elif directed_group_positive_fill_augment_trigger == "adaptive":
                            augment_trigger_passed = (
                                augment_noise_floor_gate_passed
                                or augment_plateau_gate_passed
                                or (
                                    directed_group_positive_fill_augment_max_loss > 0.0
                                    and augment_manual_loss_gate_passed
                                )
                            )
                        else:
                            raise ValueError(
                                "qdte.directed_group_positive_fill_augment_trigger must be one of: "
                                "always, loss_gate, adaptive"
                            )
                        should_augment = (
                            directed_group_positive_fill_augment
                            and remaining_accept > 0
                            and int(len(fill_transport.accepted_indices))
                            <= int(directed_group_positive_fill_augment_threshold)
                            and augment_trigger_passed
                        )
                        fill_transport.diagnostics.update(
                            {
                                "directed_group_positive_fill_augment_trigger_adaptive": int(
                                    directed_group_positive_fill_augment_trigger == "adaptive"
                                ),
                                "directed_group_positive_fill_augment_current_loss": float(
                                    current_loss_for_augment
                                ),
                                "directed_group_positive_fill_augment_max_loss": float(
                                    directed_group_positive_fill_augment_max_loss
                                ),
                                "directed_group_positive_fill_augment_loss_gate_passed": int(
                                    augment_trigger_passed
                                ),
                                "directed_group_positive_fill_augment_manual_loss_gate_passed": int(
                                    augment_manual_loss_gate_passed
                                ),
                                "directed_group_positive_fill_augment_noise_floor_loss": float(
                                    objective_noise_floor_loss
                                ),
                                "directed_group_positive_fill_augment_noise_floor_ratio": float(
                                    directed_group_positive_fill_augment_noise_floor_ratio
                                ),
                                "directed_group_positive_fill_augment_noise_floor_gate_passed": int(
                                    augment_noise_floor_gate_passed
                                ),
                                "directed_group_positive_fill_augment_plateau_window": int(
                                    directed_group_positive_fill_augment_plateau_window
                                ),
                                "directed_group_positive_fill_augment_plateau_reference_loss": float(
                                    plateau_reference_loss
                                ),
                                "directed_group_positive_fill_augment_plateau_relative_drop": float(
                                    recent_relative_drop
                                ),
                                "directed_group_positive_fill_augment_plateau_gate_passed": int(
                                    augment_plateau_gate_passed
                                ),
                            }
                        )
                        if should_augment:
                            blocked_rows = candidates.row_ids[fill_transport.accepted_indices]
                            augment_advantages = advantages.astype(np.float32, copy=True)
                            if len(blocked_rows) > 0:
                                augment_advantages[np.isin(candidates.row_ids, blocked_rows)] = -np.inf
                            virtual_residual = (state.residual - fill_transport.delta_sum).astype(np.float32)
                            if directed_group_backend in {"jax", "gpu"}:
                                augment_transport = choose_directed_group_transport_jax(
                                    candidates,
                                    augment_advantages,
                                    virtual_residual,
                                    state.inv_variance,
                                    lambda_cost,
                                    qcat,
                                    max_accept=remaining_accept,
                                    min_advantage=min_advantage,
                                    seed_count=directed_group_augment_seed_count,
                                    min_group_size=directed_group_augment_min_size,
                                    max_group_size=directed_group_augment_max_size,
                                    pool_multiplier=directed_group_augment_pool_multiplier,
                                    max_pool=directed_group_augment_max_pool,
                                    allow_negative_steps=directed_group_allow_negative_steps,
                                )
                            else:
                                augment_transport = choose_directed_group_transport(
                                    candidates,
                                    augment_advantages,
                                    virtual_residual,
                                    state.inv_variance,
                                    lambda_cost,
                                    qcat,
                                    max_accept=remaining_accept,
                                    min_advantage=min_advantage,
                                    seed_count=directed_group_augment_seed_count,
                                    min_group_size=directed_group_augment_min_size,
                                    max_group_size=directed_group_augment_max_size,
                                    pool_multiplier=directed_group_augment_pool_multiplier,
                                    max_pool=directed_group_augment_max_pool,
                                    allow_negative_steps=directed_group_allow_negative_steps,
                                    delta_index=query_delta_index if transport_delta_backend == "sparse_cpu" else None,
                                )
                            fill_transport.diagnostics.update(
                                {
                                    "directed_group_positive_fill_augment_attempted": 1,
                                    "directed_group_positive_fill_augment_pool_candidates": int(
                                        augment_transport.diagnostics.get("directed_group_pool_candidates", 0)
                                    ),
                                    "directed_group_positive_fill_augment_seed_count": int(
                                        directed_group_augment_seed_count
                                    ),
                                    "directed_group_positive_fill_augment_max_pool": int(
                                        directed_group_augment_max_pool
                                    ),
                                    "directed_group_positive_fill_augment_groups_evaluated": int(
                                        augment_transport.diagnostics.get("directed_group_groups_evaluated", 0)
                                    ),
                                    "directed_group_positive_fill_augment_accepted": int(
                                        len(augment_transport.accepted_indices)
                                    ),
                                }
                            )
                            if len(augment_transport.accepted_indices) > 0:
                                combined_indices = np.concatenate(
                                    [fill_transport.accepted_indices, augment_transport.accepted_indices]
                                ).astype(np.int32, copy=False)
                                combined_delta = (
                                    fill_transport.delta_sum.astype(np.float32)
                                    + augment_transport.delta_sum.astype(np.float32)
                                )
                                combined_cost = float(candidates.edit_cost[combined_indices].sum())
                                combined_advantage = batch_advantage(
                                    state.residual,
                                    state.inv_variance,
                                    combined_delta,
                                    combined_cost,
                                    lambda_cost,
                                )
                                combined_diagnostics = dict(fill_transport.diagnostics)
                                combined_diagnostics.update(augment_transport.diagnostics)
                                combined_diagnostics.update(
                                    {
                                        "directed_group_positive_fill": 1,
                                        "directed_group_positive_fill_augment_enabled": 1,
                                        "directed_group_positive_fill_augment_attempted": 1,
                                        "directed_group_positive_fill_augment_accepted": int(
                                            len(augment_transport.accepted_indices)
                                        ),
                                        "directed_group_positive_fill_augment_threshold": int(
                                            directed_group_positive_fill_augment_threshold
                                        ),
                                        "directed_group_positive_fill_augment_max_loss": float(
                                            directed_group_positive_fill_augment_max_loss
                                        ),
                                        "directed_group_positive_fill_augment_loss_gate_passed": int(
                                            augment_trigger_passed
                                        ),
                                        "directed_group_positive_fill_augment_seed_count": int(
                                            directed_group_augment_seed_count
                                        ),
                                        "directed_group_positive_fill_augment_max_pool": int(
                                            directed_group_augment_max_pool
                                        ),
                                        "directed_group_positive_fill_total_accepted": int(len(combined_indices)),
                                        "directed_group_positive_fill_combined_batch_advantage": float(
                                            combined_advantage
                                        ),
                                    }
                                )
                                transport = fill_transport.__class__(
                                    accepted_indices=combined_indices,
                                    delta_sum=combined_delta.astype(np.float32, copy=False),
                                    batch_advantage=float(combined_advantage),
                                    mean_advantage=float(advantages[combined_indices].mean()),
                                    diagnostics=combined_diagnostics,
                                )
            if transport is None and directed_group_backend in {"jax", "gpu"}:
                transport = choose_directed_group_transport_jax(
                    candidates,
                    advantages,
                    state.residual,
                    state.inv_variance,
                    lambda_cost,
                    qcat,
                    max_accept=accept_limit,
                    min_advantage=min_advantage,
                    seed_count=directed_group_seed_count,
                    min_group_size=directed_group_min_size,
                    max_group_size=directed_group_max_size,
                    pool_multiplier=directed_group_pool_multiplier,
                    max_pool=directed_group_max_pool,
                    allow_negative_steps=directed_group_allow_negative_steps,
                )
                if positive_fill_diagnostics:
                    merged = dict(positive_fill_diagnostics)
                    merged.update(transport.diagnostics)
                    transport.diagnostics = merged
            elif transport is None:
                transport = choose_directed_group_transport(
                    candidates,
                    advantages,
                    state.residual,
                    state.inv_variance,
                    lambda_cost,
                    qcat,
                    max_accept=accept_limit,
                    min_advantage=min_advantage,
                    seed_count=directed_group_seed_count,
                    min_group_size=directed_group_min_size,
                    max_group_size=directed_group_max_size,
                    pool_multiplier=directed_group_pool_multiplier,
                    max_pool=directed_group_max_pool,
                    allow_negative_steps=directed_group_allow_negative_steps,
                    delta_index=query_delta_index if transport_delta_backend == "sparse_cpu" else None,
                )
                if positive_fill_diagnostics:
                    merged = dict(positive_fill_diagnostics)
                    merged.update(transport.diagnostics)
                    transport.diagnostics = merged
            selected = transport.accepted_indices
            stats.num_selected_nonconflicting_candidates += int(
                transport.diagnostics.get(
                    "directed_group_positive_fill_selected",
                    transport.diagnostics.get("directed_group_groups_evaluated", 0),
                )
            )
        else:
            selected = select_top_nonconflicting(candidates, advantages, accept_limit, min_advantage)
            stats.num_selected_nonconflicting_candidates += int(len(selected))
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
                if len(selected) > 0 and transport_delta_backend == "sparse_cpu":
                    if query_delta_index is None:
                        raise RuntimeError("Internal error: sparse_cpu transport backend requires QueryDeltaIndex.")
                    deltas = compute_deltas_sparse(
                        candidates.old_rows[selected],
                        candidates.new_rows[selected],
                        query_delta_index,
                    )
                elif len(selected) > 0:
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
        accepted_entropy_gain: float | None = None
        if len(transport.accepted_indices) > 0:
            accepted_row_ids = candidates.row_ids[transport.accepted_indices]
            if len(np.unique(accepted_row_ids)) != len(accepted_row_ids):
                raise AssertionError("Transport returned conflicting edits for the same synthetic row")
            accepted_cost = float(
                candidates.edit_cost[transport.accepted_indices].astype(np.float64, copy=False).sum()
            )
            if entropy_state is not None:
                if entropy_controller is None or optimization_precision is None:
                    raise RuntimeError("Entropy batch verification is missing its controller or precision")
                data_batch_advantage = optimization_precision.advantage(
                    state.residual,
                    transport.delta_sum,
                    edit_cost=0.0,
                    lambda_cost=0.0,
                )
                accepted_entropy_gain = entropy_state.batch_gain(
                    candidates.old_rows[transport.accepted_indices],
                    candidates.new_rows[transport.accepted_indices],
                )
                exact_batch_advantage = float(
                    entropy_controller.prefix_advantages(
                        data_objective=before_optimization_objective,
                        data_gains=np.asarray([data_batch_advantage], dtype=np.float64),
                        entropy_gains=np.asarray([accepted_entropy_gain], dtype=np.float64),
                        edit_costs=np.asarray([accepted_cost], dtype=np.float64),
                        lambda_cost=lambda_cost,
                    )[0]
                )
                transport.diagnostics.update(
                    {
                        "entropy_data_batch_advantage": float(data_batch_advantage),
                        "entropy_regularizer_batch_gain": float(accepted_entropy_gain),
                        "entropy_dual_weight": float(entropy_controller.dual_weight),
                    }
                )
            elif objective_loss == "tvd_l1":
                exact_batch_advantage = batch_advantage_l1(
                    state.residual,
                    objective_loss_weights,
                    transport.delta_sum,
                    accepted_cost,
                    lambda_cost,
                )
            elif optimization_precision is not None:
                exact_batch_advantage = batch_advantage_precision(
                    state.residual,
                    optimization_precision,
                    transport.delta_sum,
                    accepted_cost,
                    lambda_cost,
                )
            else:
                exact_batch_advantage = batch_advantage(
                    state.residual,
                    state.inv_variance,
                    transport.delta_sum,
                    accepted_cost,
                    lambda_cost,
                )
            transport.diagnostics["exact_batch_advantage"] = float(exact_batch_advantage)
            transport.diagnostics["approx_batch_advantage"] = float(transport.batch_advantage)
            if transport_mode != "blind_accept" and exact_batch_advantage <= 0.0:
                transport.diagnostics["rejected_by_exact_batch_check"] = 1
                transport.accepted_indices = np.empty(0, dtype=np.int32)
                transport.delta_sum = np.zeros_like(state.residual, dtype=np.float32)
                transport.batch_advantage = float(exact_batch_advantage)
                transport.mean_advantage = 0.0
            else:
                transport.diagnostics["rejected_by_exact_batch_check"] = 0
                transport.batch_advantage = float(exact_batch_advantage)

        if candidate_diagnostics_enabled:
            diagnostic_delta_index = query_delta_index if transport_delta_backend == "sparse_cpu" else None
            candidate_diagnostic_rows.append(
                _candidate_diagnostic_summary(
                    iteration=iteration,
                    candidates=candidates,
                    advantages=advantages,
                    residual=residual_before_debt,
                    inv_variance=state.inv_variance,
                    qcat=qcat,
                    lambda_cost=lambda_cost,
                    min_advantage=min_advantage,
                    selected_indices=selected,
                    accepted_indices=transport.accepted_indices,
                    delta_index=diagnostic_delta_index,
                )
            )
        if len(transport.accepted_indices) > 0:
            if entropy_state is not None:
                applied_entropy_gain = entropy_state.apply_batch(
                    candidates.old_rows[transport.accepted_indices],
                    candidates.new_rows[transport.accepted_indices],
                )
                if accepted_entropy_gain is None or not np.isclose(
                    applied_entropy_gain,
                    accepted_entropy_gain,
                    rtol=1.0e-10,
                    atol=1.0e-10,
                ):
                    raise AssertionError("Applied entropy batch gain differs from its exact certificate")
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
            loss_vec_after_debt = _optimization_loss_vector(
                state.residual,
                objective_loss,
                state.inv_variance,
                objective_loss_weights,
            )
            state.debt, debt_info = update_query_debt_from_loss_vectors(
                state.debt,
                loss_vec_before_debt,
                loss_vec_after_debt,
                debt_decay=debt_decay,
                debt_repay=debt_repay,
                debt_cap=debt_cap,
            )
            last_debt_diagnostics = debt_info
            stats.num_accepted_edits += len(transport.accepted_indices)
            patience = 0
        else:
            patience += 1

        last_transport_diagnostics = transport.diagnostics
        stats.time_transport_seconds += time.perf_counter() - t_transport

        structured_swap_candidates_this_iter = 0
        structured_swap_accepted_this_iter = 0
        structured_swap_noise_rejections_this_iter = 0
        structured_swap_advantage_this_iter = 0.0
        structured_swap_compiler_diagnostics_this_iter: dict[str, Any] = {}
        structured_swap_effective_noise_guard_kappa = structured_swap_noise_guard_kappa
        if structured_swap_compiler == INTERACTION_CYCLE_METHOD:
            effective_rank = max(1, int(precision_diagnostics["effective_rank"]))
            structured_swap_current_rms = float(
                np.sqrt(
                    max(
                        0.0,
                        2.0
                        * _optimization_objective_value(
                            state.residual,
                            objective_loss,
                            state.inv_variance,
                            objective_loss_weights,
                            optimization_precision,
                        )
                        / effective_rank,
                    )
                )
            )
        else:
            structured_swap_current_rms = rms_standardized_residual(
                measured_loss(state.residual, state.inv_variance),
                qcat.m,
            )
        structured_swap_due = bool(
            structured_swap_enabled
            and iteration >= structured_swap_start_iter
            and (iteration - structured_swap_start_iter) % structured_swap_interval == 0
            and structured_swap_current_rms > structured_swap_trigger_rms
        )
        if structured_swap_due:
            if (
                structured_swap_score_context is None
                or structured_swap_delta_index is None
                or structured_swap_rng is None
            ):
                raise RuntimeError("Structured swap contexts were not initialized")
            if (
                structured_swap_compiler == "global_random_v1"
                and structured_swap_policy is None
            ):
                raise RuntimeError("Global random structured-swap policy was not initialized")
            structured_start = time.perf_counter()
            structured_stage_start = structured_start
            if structured_swap_compiler == INTERACTION_CYCLE_METHOD:
                if not isinstance(optimization_precision, OrthogonalInteractionPrecision):
                    raise RuntimeError(
                        "Interaction-cycle compiler lost its exact orthogonal precision"
                    )
                compilation = compile_interaction_cycle_candidates(
                    state.X_syn,
                    schema,
                    state.residual,
                    optimization_precision,
                    structured_swap_candidate_units,
                    structured_swap_rng,
                    numerical_gamma=float(qdte_cfg.get("numerical_distance_gamma", 0.1)),
                )
                structured_candidates = compilation.candidates
                structured_swap_compiler_diagnostics_this_iter = dict(
                    compilation.diagnostics
                )
                structured_swap_last_compiler_diagnostics = dict(compilation.diagnostics)
                structured_swap_compiler_calls += 1
                structured_swap_total_positive_rectangles += int(
                    compilation.diagnostics.get("positive_rectangles", 0)
                )
                structured_swap_max_selected_scopes = max(
                    structured_swap_max_selected_scopes,
                    int(compilation.diagnostics.get("selected_scopes", 0)),
                )
                structured_swap_target_linear_gain_sum += float(
                    np.sum(compilation.target_linear_gains, dtype=np.float64)
                )
                structured_swap_target_linear_gain_count += int(
                    len(compilation.target_linear_gains)
                )
            else:
                if structured_swap_policy is None:
                    raise RuntimeError("Global random structured-swap policy is missing")
                structured_candidates = generate_global_directed_swap_units(
                    state.X_syn,
                    qcat,
                    schema,
                    state.residual,
                    objective_loss_weights if objective_loss == "tvd_l1" else state.inv_variance,
                    structured_swap_candidate_units,
                    structured_swap_rng,
                    exploration_floor=structured_swap_exploration_floor,
                    attr_probabilities=structured_swap_policy.probabilities(),
                    numerical_gamma=float(qdte_cfg.get("numerical_distance_gamma", 0.1)),
                )
            structured_swap_generation_time_seconds += time.perf_counter() - structured_stage_start
            structured_stage_start = time.perf_counter()
            if structured_swap_compiler == INTERACTION_CYCLE_METHOD:
                if not isinstance(optimization_precision, OrthogonalInteractionPrecision):
                    raise RuntimeError(
                        "Interaction-cycle scoring lost its exact orthogonal precision"
                    )
                structured_scores = score_candidate_units_precision(
                    structured_candidates,
                    state.residual,
                    optimization_precision,
                    lambda_cost=lambda_cost,
                    chunk_size=min(chunk_size, 256),
                )
            elif objective_loss == "tvd_l1":
                structured_scores = score_hybrid_candidate_units_l1(
                    structured_candidates,
                    state.residual,
                    objective_loss_weights,
                    structured_swap_score_context,
                    lambda_cost=lambda_cost,
                    chunk_size=min(chunk_size, 256),
                )
            else:
                structured_scores = score_hybrid_candidate_units(
                    structured_candidates,
                    state.residual,
                    state.inv_variance,
                    structured_swap_score_context,
                    lambda_cost=lambda_cost,
                    chunk_size=chunk_size,
                    use_pmap=use_pmap,
                )
            structured_swap_scoring_time_seconds += time.perf_counter() - structured_stage_start
            structured_stage_start = time.perf_counter()
            if structured_swap_policy is not None:
                structured_swap_policy.update(structured_candidates, structured_scores)
            structured_swap_effective_noise_guard_kappa = _search_adjusted_noise_guard_kappa(
                mode=structured_swap_noise_guard_mode,
                fixed_kappa=structured_swap_noise_guard_kappa,
                family_size=structured_candidates.count,
                alpha=structured_swap_noise_guard_alpha,
            )
            structured_swap_max_effective_noise_guard_kappa = max(
                structured_swap_max_effective_noise_guard_kappa,
                structured_swap_effective_noise_guard_kappa,
            )
            structured_pool_size = min(structured_candidates.count, structured_swap_transport_pool)
            structured_pool_indices = np.argsort(-structured_scores, kind="stable")[:structured_pool_size]
            structured_transport_candidates = structured_candidates.take(structured_pool_indices)
            structured_transport_scores = structured_scores[structured_pool_indices]
            structured_accept_limit = _scheduled_accept_limit(
                iteration=iteration,
                max_iters=max_iters,
                base_accept=structured_swap_accept_start,
                qdte_cfg=structured_swap_schedule_cfg,
            )
            structured_swap_policy_pool_time_seconds += time.perf_counter() - structured_stage_start
            structured_precomputed_deltas = None
            structured_precomputed_feature_deltas = None
            if structured_swap_delta_backend == "dense_gpu":
                structured_stage_start = time.perf_counter()
                structured_precomputed_deltas = candidate_unit_deltas(
                    structured_transport_candidates,
                    structured_swap_score_context,
                )
                structured_swap_delta_time_seconds += time.perf_counter() - structured_stage_start
            if structured_swap_compiler == INTERACTION_CYCLE_METHOD:
                if not isinstance(optimization_precision, OrthogonalInteractionPrecision):
                    raise RuntimeError(
                        "Interaction-cycle transport lost its exact orthogonal precision"
                    )
                structured_stage_start = time.perf_counter()
                structured_precomputed_feature_deltas = candidate_unit_feature_deltas(
                    structured_transport_candidates,
                    optimization_precision,
                )
                structured_swap_delta_time_seconds += time.perf_counter() - structured_stage_start
            structured_stage_start = time.perf_counter()
            structured_transport = select_nonconflicting_candidate_units(
                structured_transport_candidates,
                structured_transport_scores,
                state.residual,
                state.inv_variance,
                qcat,
                max_accept=structured_accept_limit,
                min_advantage=min_advantage,
                lambda_cost=lambda_cost,
                noise_guard_kappa=structured_swap_effective_noise_guard_kappa,
                advantage_noise_variance=objective_advantage_noise_variance,
                delta_index=structured_swap_delta_index,
                precomputed_deltas=structured_precomputed_deltas,
                objective="l1" if objective_loss == "tvd_l1" else "quadratic",
                objective_weights=(
                    objective_loss_weights if objective_loss == "tvd_l1" else None
                ),
                precision_operator=(
                    optimization_precision
                    if structured_swap_compiler == INTERACTION_CYCLE_METHOD
                    else None
                ),
                precomputed_feature_deltas=structured_precomputed_feature_deltas,
            )
            structured_swap_transport_time_seconds += time.perf_counter() - structured_stage_start
            structured_swap_candidates_this_iter = structured_candidates.count
            structured_swap_accepted_this_iter = structured_transport.count
            structured_swap_noise_rejections_this_iter = structured_transport.num_noise_guard_rejections
            structured_swap_advantage_this_iter = float(
                np.sum(structured_transport.objective_advantages, dtype=np.float64)
            )
            structured_swap_total_candidates += structured_candidates.count
            structured_swap_total_accepted += structured_transport.count
            structured_swap_total_noise_guard_rejections += structured_transport.num_noise_guard_rejections
            structured_swap_total_objective_advantage += structured_swap_advantage_this_iter
            stats.num_candidates_requested += structured_swap_candidate_units
            stats.num_candidates_scored += structured_candidates.count
            stats.num_candidates_returned_to_cpu += structured_candidates.count

            if structured_transport.count > 0:
                structured_stage_start = time.perf_counter()
                loss_before_structured = _optimization_objective_value(
                    state.residual,
                    objective_loss,
                    state.inv_variance,
                    objective_loss_weights,
                    optimization_precision,
                )
                loss_vec_before_structured = _optimization_loss_vector(
                    state.residual,
                    objective_loss,
                    state.inv_variance,
                    objective_loss_weights,
                )
                apply_hybrid_candidate_units(
                    state.X_syn,
                    structured_transport_candidates,
                    structured_transport.accepted_indices,
                )
                accepted_units = structured_transport_candidates.take(structured_transport.accepted_indices)
                if use_gpu_candidate_backend:
                    if X_syn_gpu is None:
                        raise RuntimeError("GPU table is not initialized for structured swap updates")
                    accepted_mask = accepted_units.row_mask
                    X_syn_gpu = apply_edits_to_replicated_table(
                        X_syn_gpu,
                        accepted_units.row_ids[accepted_mask],
                        accepted_units.new_rows[accepted_mask],
                    )
                state.answer_syn = (state.answer_syn + structured_transport.delta_sum).astype(np.float32)
                state.residual = (state.target - state.answer_syn).astype(np.float32)
                loss_after_structured = _optimization_objective_value(
                    state.residual,
                    objective_loss,
                    state.inv_variance,
                    objective_loss_weights,
                    optimization_precision,
                )
                accepted_cost = float(
                    np.sum(
                        structured_transport_candidates.edit_cost[structured_transport.accepted_indices],
                        dtype=np.float64,
                    )
                )
                expected_reduction = structured_swap_advantage_this_iter + lambda_cost * accepted_cost
                actual_reduction = float(loss_before_structured - loss_after_structured)
                tolerance = 1.0e-5 * max(1.0, abs(expected_reduction))
                if abs(actual_reduction - expected_reduction) > tolerance:
                    raise AssertionError(
                        "Structured swap measured-loss reduction disagrees with exact advantage: "
                        f"actual={actual_reduction}, expected={expected_reduction}"
                    )
                loss_vec_after_structured = _optimization_loss_vector(
                    state.residual,
                    objective_loss,
                    state.inv_variance,
                    objective_loss_weights,
                )
                state.debt, debt_info = update_query_debt_from_loss_vectors(
                    state.debt,
                    loss_vec_before_structured,
                    loss_vec_after_structured,
                    debt_decay=debt_decay,
                    debt_repay=debt_repay,
                    debt_cap=debt_cap,
                )
                last_debt_diagnostics = debt_info
                stats.num_accepted_edits += structured_transport.count
                patience = 0
                structured_swap_apply_time_seconds += time.perf_counter() - structured_stage_start
            elapsed_structured = time.perf_counter() - structured_start
            structured_swap_time_seconds += elapsed_structured
        debug_drift = 0.0

        if debug_recompute_after_batch and (
            len(transport.accepted_indices) > 0 or structured_swap_accepted_this_iter > 0
        ):
            t_recompute = time.perf_counter()
            recomputed = answer_queries(state.X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
            debug_drift = float(np.max(np.abs(recomputed - state.answer_syn)))
            recomputed_residual = (state.target - recomputed).astype(np.float32)
            recomputed_loss = measured_loss(recomputed_residual, state.inv_variance)
            recomputed_optimization_objective = _optimization_objective_value(
                recomputed_residual,
                objective_loss,
                state.inv_variance,
                objective_loss_weights,
                optimization_precision,
            )
            if entropy_state is not None:
                if entropy_controller is None:
                    raise RuntimeError("Entropy recomputation is missing its controller")
                entropy_state.assert_matches(state.X_syn)
                recomputed_regularized_objective = entropy_controller.regularized_objective(
                    recomputed_optimization_objective,
                    entropy_state.regularizer,
                )
            else:
                recomputed_regularized_objective = recomputed_optimization_objective
            state.answer_syn = recomputed.astype(np.float32)
            state.residual = recomputed_residual
            stats.time_full_recompute_seconds += time.perf_counter() - t_recompute
            if debug_drift > residual_drift_tolerance:
                raise AssertionError(
                    f"Incremental answer drift {debug_drift} exceeds tolerance {residual_drift_tolerance}"
                )
            if (
                debug_assert_loss_decrease
                and recomputed_regularized_objective > before_regularized_objective + loss_tolerance
            ):
                raise AssertionError(
                    "Accepted batch increased recomputed optimization objective: "
                    f"before={before_regularized_objective}, after={recomputed_regularized_objective}; "
                    f"measured_loss_before={before_loss}, measured_loss_after={recomputed_loss}"
                )

        if full_recompute_every > 0 and iteration % full_recompute_every == 0:
            t_recompute = time.perf_counter()
            recomputed = answer_queries(state.X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
            drift = float(np.max(np.abs(recomputed - state.answer_syn)))
            state.answer_syn = recomputed.astype(np.float32)
            state.residual = (state.target - state.answer_syn).astype(np.float32)
            if entropy_state is not None:
                entropy_state.assert_matches(state.X_syn)
            stats.time_full_recompute_seconds += time.perf_counter() - t_recompute
            log(f"Full recompute iter={iteration}: max_incremental_drift={drift:.6g}")
            if drift > residual_drift_tolerance:
                raise AssertionError(
                    f"Incremental answer drift {drift} exceeds tolerance {residual_drift_tolerance}"
                )

        if fission_enabled and iteration % fission_checkpoint_interval == 0:
            fission_checkpoint_iterations.append(int(iteration))
            fission_checkpoint_tables.append(state.X_syn.copy())
            fission_checkpoint_answers.append(state.answer_syn.copy())

        cur_loss = measured_loss(state.residual, state.inv_variance)
        cur_optimization_objective = _optimization_objective_value(
            state.residual,
            objective_loss,
            state.inv_variance,
            objective_loss_weights,
            optimization_precision,
        )
        confidence_stop_reached_this_iter = bool(
            confidence_stop is not None
            and confidence_stop.reached(cur_optimization_objective)
        )
        if confidence_stop_reached_this_iter:
            confidence_stop_triggered = True
            confidence_stop_iteration = int(iteration)
            confidence_stop_objective_at_trigger = float(cur_optimization_objective)
        cur_entropy_regularizer = (
            float(entropy_state.regularizer) if entropy_state is not None else 0.0
        )
        cur_regularized_objective = (
            entropy_controller.regularized_objective(
                cur_optimization_objective,
                cur_entropy_regularizer,
            )
            if entropy_controller is not None
            else cur_optimization_objective
        )
        cur_unweighted_loss = unweighted_measured_loss(state.residual)
        if (
            iteration == 1
            or iteration % log_every == 0
            or len(transport.accepted_indices) == 0
            or confidence_stop_reached_this_iter
        ):
            row = {
                "iteration": iteration,
                "wall_time": time.perf_counter() - stats.start_time,
                "measured_loss": cur_loss,
                "optimization_objective": cur_optimization_objective,
                "entropy_regularizer": cur_entropy_regularizer,
                "entropy_kl_per_row": (
                    cur_entropy_regularizer / n_syn if entropy_state is not None else 0.0
                ),
                "entropy_regularized_objective": cur_regularized_objective,
                "entropy_dual_weight": (
                    float(entropy_controller.dual_weight)
                    if entropy_controller is not None
                    else 0.0
                ),
                "entropy_inside_confidence_set": int(
                    entropy_controller.inside(cur_optimization_objective)
                    if entropy_controller is not None
                    else False
                ),
                "confidence_stop_reached": int(confidence_stop_reached_this_iter),
                "confidence_stop_objective_threshold": (
                    float(confidence_stop.objective_threshold)
                    if confidence_stop is not None
                    else 0.0
                ),
                "unweighted_measured_loss": cur_unweighted_loss,
                "rms_standardized_residual": rms_standardized_residual(cur_loss, qcat.m),
                "rms_unweighted_residual": rms_unweighted_residual(state.residual),
                "residual_l2": float(np.linalg.norm(state.residual)),
                "residual_l1": float(np.sum(np.abs(state.residual))),
                "active_queries": int(len(active)),
                "num_candidates": int(candidates.size),
                "raw_returned_candidates_before_edge_dedup": int(
                    diag.get("raw_returned_candidates_before_edge_dedup", candidates.size)
                ),
                "duplicate_edge_candidates": int(diag.get("duplicate_edge_candidates", 0.0)),
                "duplicate_edge_rate": float(diag.get("duplicate_edge_rate", 0.0)),
                "edge_candidates_trimmed_to_return_limit": int(
                    diag.get("edge_candidates_trimmed_to_return_limit", 0.0)
                ),
                "candidates_scored_this_iter": candidates_scored_this_iter,
                "positive_advantage_rate": positive_rate,
                "positive_returned_rate": positive_rate,
                "selected_nonconflicting": int(
                    transport.diagnostics.get("atom_flow_pool_candidates", len(selected))
                ),
                "accept_limit": int(accept_limit),
                "accepted_edits": int(len(transport.accepted_indices)),
                "accepted_rate": float(len(transport.accepted_indices) / max(1, candidates.size)),
                "mean_advantage": transport.mean_advantage,
                "batch_advantage": transport.batch_advantage,
                "requested_candidates": int(diag.get("requested_candidates", candidates.size)),
                "directed_candidate_budget": int(diag.get("directed_candidate_budget", 0.0)),
                "random_candidate_budget": int(diag.get("random_candidate_budget", 0.0)),
                "directed_candidates": int(diag.get("directed_candidates", 0.0)),
                "random_candidates": int(diag.get("random_candidates", 0.0)),
                "planned_random_candidates": int(diag.get("planned_random_candidates", 0.0)),
                "fallback_random_candidates": int(diag.get("fallback_random_candidates", 0.0)),
                "mixture_random_candidates": int(diag.get("mixture_random_candidates", 0.0)),
                "paired_candidates": int(diag.get("paired_candidates", 0.0)),
                "masked_paired_candidates": int(diag.get("masked_paired_candidates", 0.0)),
                "masked_exit_candidates": int(diag.get("masked_exit_candidates", 0.0)),
                "masked_single_query_candidates": int(diag.get("masked_single_query_candidates", 0.0)),
                "relaxed_masked_single_query_candidates": int(
                    diag.get("relaxed_masked_single_query_candidates", 0.0)
                ),
                "directed_exit_only_candidates": int(diag.get("directed_exit_only_candidates", 0.0)),
                "masked_exit_only_candidates": int(diag.get("masked_exit_only_candidates", 0.0)),
                "random_source_directed_exit_candidates": int(
                    diag.get("random_source_directed_exit_candidates", 0.0)
                ),
                "residual_weighted_mutation_candidates": int(
                    diag.get("residual_weighted_mutation_candidates", 0.0)
                ),
                "enumerated_local_candidates": int(diag.get("enumerated_local_candidates", 0.0)),
                "soft_single_query_candidates": int(diag.get("soft_single_query_candidates", 0.0)),
                "residual_value_mutation_candidates": int(diag.get("residual_value_mutation_candidates", 0.0)),
                "constructive_partner_candidates": int(diag.get("constructive_partner_candidates", 0.0)),
                "constructive_attached_partner_candidates": int(
                    diag.get("constructive_attached_partner_candidates", 0.0)
                ),
                "constructive_attached_pair_units": int(diag.get("constructive_attached_pair_units", 0.0)),
                "constructive_partner_seed_candidates": int(
                    diag.get("constructive_partner_seed_candidates", 0.0)
                ),
                "constructive_partner_source_attempts": int(
                    diag.get("constructive_partner_source_attempts", 0.0)
                ),
                "constructive_partner_source_failures": int(
                    diag.get("constructive_partner_source_failures", 0.0)
                ),
                "best_partner_candidates": int(diag.get("best_partner_candidates", 0.0)),
                "best_partner_pair_units": int(diag.get("best_partner_pair_units", 0.0)),
                "best_partner_seed_candidates": int(diag.get("best_partner_seed_candidates", 0.0)),
                "best_partner_source_attempts": int(diag.get("best_partner_source_attempts", 0.0)),
                "best_partner_source_failures": int(diag.get("best_partner_source_failures", 0.0)),
                "best_partner_pairs_evaluated": int(diag.get("best_partner_pairs_evaluated", 0.0)),
                "best_partner_positive_pairs": int(diag.get("best_partner_positive_pairs", 0.0)),
                "protected_same_row_candidates": int(diag.get("protected_same_row_candidates", 0.0)),
                "protected_repair_seed_candidates": int(diag.get("protected_repair_seed_candidates", 0.0)),
                "protected_repair_attempts": int(diag.get("protected_repair_attempts", 0.0)),
                "protected_repair_target_failures": int(diag.get("protected_repair_target_failures", 0.0)),
                "protected_repair_protection_successes": int(
                    diag.get("protected_repair_protection_successes", 0.0)
                ),
                "proposal_mixture_candidates": int(diag.get("proposal_mixture_candidates", 0.0)),
                "qdte_mixture_candidates": int(diag.get("qdte_mixture_candidates", 0.0)),
                "single_directed_candidates": int(diag.get("single_directed_candidates", 0.0)),
                "directed_candidate_shortfall": int(diag.get("directed_candidate_shortfall", 0.0)),
                "candidate_shortfall": int(diag.get("candidate_shortfall", 0.0)),
                "source_filter_attempts": int(diag.get("source_filter_attempts", 0.0)),
                "source_filter_failures": int(diag.get("source_filter_failures", 0.0)),
                "paired_source_filter_attempts": int(diag.get("paired_source_filter_attempts", 0.0)),
                "paired_source_filter_failures": int(diag.get("paired_source_filter_failures", 0.0)),
                "random_source_exit_attempts": int(diag.get("random_source_exit_attempts", 0.0)),
                "atom_flow_pool_candidates": int(transport.diagnostics.get("atom_flow_pool_candidates", 0)),
                "atom_flow_edges": int(transport.diagnostics.get("atom_flow_edges", 0)),
                "atom_flow_source_atoms": int(transport.diagnostics.get("atom_flow_source_atoms", 0)),
                "atom_flow_target_atoms": int(transport.diagnostics.get("atom_flow_target_atoms", 0)),
                "atom_flow_augments": int(transport.diagnostics.get("atom_flow_augments", 0)),
                "atom_flow_batch_mode": int(transport.diagnostics.get("atom_flow_batch_mode", 0)),
                "atom_flow_exact_mode": int(transport.diagnostics.get("atom_flow_exact_mode", 0)),
                "atom_flow_custom_prefix": int(
                    transport.diagnostics.get("atom_flow_custom_prefix", 0)
                ),
                "atom_flow_selected_candidates": int(
                    transport.diagnostics.get("atom_flow_selected_candidates", 0)
                ),
                "atom_flow_prefix_candidates": int(transport.diagnostics.get("atom_flow_prefix_candidates", 0)),
                "exact_batch_advantage": float(
                    transport.diagnostics.get("exact_batch_advantage", 0.0)
                ),
                "rejected_by_exact_batch_check": int(
                    transport.diagnostics.get("rejected_by_exact_batch_check", 0)
                ),
                "entropy_data_batch_advantage": float(
                    transport.diagnostics.get("entropy_data_batch_advantage", 0.0)
                ),
                "entropy_regularizer_batch_gain": float(
                    transport.diagnostics.get("entropy_regularizer_batch_gain", 0.0)
                ),
                "constructive_pair_pool_candidates": int(
                    transport.diagnostics.get("constructive_pair_pool_candidates", 0)
                ),
                "constructive_pair_seed_candidates": int(
                    transport.diagnostics.get("constructive_pair_seed_candidates", 0)
                ),
                "constructive_pair_pairs_evaluated": int(
                    transport.diagnostics.get("constructive_pair_pairs_evaluated", 0)
                ),
                "constructive_pair_positive_pairs": int(
                    transport.diagnostics.get("constructive_pair_positive_pairs", 0)
                ),
                "constructive_pair_explicit_pairs_evaluated": int(
                    transport.diagnostics.get("constructive_pair_explicit_pairs_evaluated", 0)
                ),
                "constructive_pair_explicit_positive_pairs": int(
                    transport.diagnostics.get("constructive_pair_explicit_positive_pairs", 0)
                ),
                "constructive_pair_units": int(transport.diagnostics.get("constructive_pair_units", 0)),
                "constructive_pair_single_units": int(
                    transport.diagnostics.get("constructive_pair_single_units", 0)
                ),
                "constructive_pair_pair_units": int(transport.diagnostics.get("constructive_pair_pair_units", 0)),
                "constructive_pair_explicit_pair_units": int(
                    transport.diagnostics.get("constructive_pair_explicit_pair_units", 0)
                ),
                "constructive_pair_selected_units": int(
                    transport.diagnostics.get("constructive_pair_selected_units", 0)
                ),
                "constructive_pair_prefix_units": int(
                    transport.diagnostics.get("constructive_pair_prefix_units", 0)
                ),
                "constructive_pair_selected_candidates": int(
                    transport.diagnostics.get("constructive_pair_selected_candidates", 0)
                ),
                "constructive_pair_accepted_candidates": int(
                    transport.diagnostics.get("constructive_pair_accepted_candidates", 0)
                ),
                "constructive_pair_group_augment": int(
                    transport.diagnostics.get("constructive_pair_group_augment", 0)
                ),
                "constructive_pair_group_augment_attempted": int(
                    transport.diagnostics.get("constructive_pair_group_augment_attempted", 0)
                ),
                "constructive_pair_group_augment_accepted": int(
                    transport.diagnostics.get("constructive_pair_group_augment_accepted", 0)
                ),
                "constructive_pair_group_augment_loss_gate_passed": int(
                    transport.diagnostics.get("constructive_pair_group_augment_loss_gate_passed", 0)
                ),
                "random_group_pool_candidates": int(transport.diagnostics.get("random_group_pool_candidates", 0)),
                "random_group_groups_evaluated": int(
                    transport.diagnostics.get("random_group_groups_evaluated", 0)
                ),
                "random_group_positive_groups": int(transport.diagnostics.get("random_group_positive_groups", 0)),
                "random_group_best_group_size": int(transport.diagnostics.get("random_group_best_group_size", 0)),
                "random_group_accepted_candidates": int(
                    transport.diagnostics.get("random_group_accepted_candidates", 0)
                ),
                "random_group_groups_with_negative_member": int(
                    transport.diagnostics.get("random_group_groups_with_negative_member", 0)
                ),
                "directed_group_pool_candidates": int(transport.diagnostics.get("directed_group_pool_candidates", 0)),
                "directed_group_seed_candidates": int(transport.diagnostics.get("directed_group_seed_candidates", 0)),
                "directed_group_groups_evaluated": int(
                    transport.diagnostics.get("directed_group_groups_evaluated", 0)
                ),
                "directed_group_positive_groups": int(
                    transport.diagnostics.get("directed_group_positive_groups", 0)
                ),
                "directed_group_expansion_steps": int(
                    transport.diagnostics.get("directed_group_expansion_steps", 0)
                ),
                "directed_group_best_group_size": int(
                    transport.diagnostics.get("directed_group_best_group_size", 0)
                ),
                "directed_group_accepted_candidates": int(
                    transport.diagnostics.get("directed_group_accepted_candidates", 0)
                ),
                "directed_group_groups_with_negative_member": int(
                    transport.diagnostics.get("directed_group_groups_with_negative_member", 0)
                ),
                "directed_group_positive_fill": int(
                    transport.diagnostics.get("directed_group_positive_fill", 0)
                ),
                "directed_group_positive_fill_selected": int(
                    transport.diagnostics.get("directed_group_positive_fill_selected", 0)
                ),
                "directed_group_positive_fill_accepted": int(
                    transport.diagnostics.get("directed_group_positive_fill_accepted", 0)
                ),
                "directed_group_positive_fill_augment_attempted": int(
                    transport.diagnostics.get("directed_group_positive_fill_augment_attempted", 0)
                ),
                "directed_group_positive_fill_augment_accepted": int(
                    transport.diagnostics.get("directed_group_positive_fill_augment_accepted", 0)
                ),
                "directed_group_positive_fill_augment_threshold": int(
                    transport.diagnostics.get(
                        "directed_group_positive_fill_augment_threshold",
                        directed_group_positive_fill_augment_threshold,
                    )
                ),
                "directed_group_positive_fill_augment_loss_gate_passed": int(
                    transport.diagnostics.get("directed_group_positive_fill_augment_loss_gate_passed", 0)
                ),
                "directed_group_positive_fill_augment_noise_floor_gate_passed": int(
                    transport.diagnostics.get(
                        "directed_group_positive_fill_augment_noise_floor_gate_passed",
                        0,
                    )
                ),
                "directed_group_positive_fill_augment_plateau_gate_passed": int(
                    transport.diagnostics.get("directed_group_positive_fill_augment_plateau_gate_passed", 0)
                ),
                "directed_group_positive_fill_augment_plateau_relative_drop": float(
                    transport.diagnostics.get(
                        "directed_group_positive_fill_augment_plateau_relative_drop",
                        0.0,
                    )
                ),
                "directed_group_positive_fill_total_accepted": int(
                    transport.diagnostics.get("directed_group_positive_fill_total_accepted", 0)
                ),
                "structured_swap_due": int(structured_swap_due),
                "structured_swap_compiler": structured_swap_compiler,
                "structured_swap_trigger_rms": float(structured_swap_trigger_rms),
                "structured_swap_current_rms": float(structured_swap_current_rms),
                "structured_swap_candidates": int(structured_swap_candidates_this_iter),
                "structured_swap_accepted": int(structured_swap_accepted_this_iter),
                "structured_swap_noise_guard_rejections": int(
                    structured_swap_noise_rejections_this_iter
                ),
                "structured_swap_effective_noise_guard_kappa": float(
                    structured_swap_effective_noise_guard_kappa
                ),
                "structured_swap_objective_advantage": float(structured_swap_advantage_this_iter),
                "structured_swap_positive_rectangles": int(
                    structured_swap_compiler_diagnostics_this_iter.get(
                        "positive_rectangles", 0
                    )
                ),
                "structured_swap_selected_scopes": int(
                    structured_swap_compiler_diagnostics_this_iter.get(
                        "selected_scopes", 0
                    )
                ),
                "structured_swap_target_linear_gain_mean": float(
                    structured_swap_compiler_diagnostics_this_iter.get(
                        "target_linear_gain_mean", 0.0
                    )
                ),
                "incremental_answer_drift": debug_drift,
                **debt_info,
            }
            timeseries.append(row)
            log(
                f"iter={iteration} loss={cur_loss:.6g} unweighted_loss={cur_unweighted_loss:.6g} "
                f"candidates={candidates.size} "
                f"positive={positive_rate:.3f} accept_limit={accept_limit} accepted={len(transport.accepted_indices)} "
                f"batch_adv={transport.batch_advantage:.6g} "
                f"entropy_kl={cur_entropy_regularizer / n_syn:.6g} "
                f"entropy_dual={entropy_controller.dual_weight if entropy_controller is not None else 0.0:.6g} "
                f"structured_accepted={structured_swap_accepted_this_iter}"
            )

        stats.num_iterations = iteration
        if confidence_stop_reached_this_iter:
            log(
                "Stopping at released chi-square confidence boundary: "
                f"iter={iteration}, objective={cur_optimization_objective:.6g}, "
                f"threshold={confidence_stop.objective_threshold:.6g}"
            )
            stats.time_generation_seconds += time.perf_counter() - iter_start
            break
        if patience >= stop_patience:
            log(f"Stopping at iter={iteration}: patience={patience}")
            break
        stats.time_generation_seconds += time.perf_counter() - iter_start

    stats.time_generation_seconds = time.perf_counter() - generation_start
    fission_selection: FissionCheckpointSelection | None = None
    fission_selected_iteration = int(stats.num_iterations)
    fission_best_iteration = int(stats.num_iterations)
    fission_terminal_iteration = int(stats.num_iterations)
    fission_terminal_train_loss = measured_loss(state.residual, state.inv_variance)
    fission_selected_validation_loss = 0.0
    fission_best_validation_loss = 0.0
    fission_terminal_validation_loss = 0.0
    fission_selected_original_released_loss = 0.0
    fission_terminal_original_released_loss = 0.0
    if fission_enabled:
        if fission_split is None:
            raise RuntimeError("Fission was enabled but split measurements are unavailable")
        if fission_holdout_noisy is None or fission_holdout_variance is None:
            raise RuntimeError("Fission holdout measurements are unavailable")
        terminal_answers = answer_queries(
            state.X_syn,
            qcat,
            batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
        ).astype(np.float32)
        terminal_residual = (state.target - terminal_answers).astype(np.float32)
        fission_terminal_train_loss = measured_loss(terminal_residual, state.inv_variance)
        if (
            not fission_checkpoint_iterations
            or fission_checkpoint_iterations[-1] != fission_terminal_iteration
        ):
            fission_checkpoint_iterations.append(fission_terminal_iteration)
            fission_checkpoint_tables.append(state.X_syn.copy())
            fission_checkpoint_answers.append(terminal_answers.copy())
        else:
            fission_checkpoint_tables[-1] = state.X_syn.copy()
            fission_checkpoint_answers[-1] = terminal_answers.copy()
        save_npy(state.X_syn, output_dir / "synthetic_train_terminal_encoded.npy")

        checkpoint_answer_matrix = np.stack(fission_checkpoint_answers, axis=0)
        fission_partition_blocks = [
            np.asarray(group.query_indices, dtype=np.int32)
            for group in measurements.groups
            if bool(group.is_partition) and len(group.query_indices) > 1
        ]
        fission_selection = select_fission_checkpoint(
            checkpoint_answer_matrix,
            fission_holdout_noisy,
            fission_holdout_variance,
            selection_rule=str(
                fission_cfg.get("selection_rule", "earliest_within_one_se")
            ),
            one_se_multiplier=float(fission_cfg.get("one_se_multiplier", 1.0)),
            partition_blocks=fission_partition_blocks,
            query_weight_multipliers=objective_query_weight_multipliers,
        )
        selected_index = int(fission_selection.selected_index)
        best_index = int(fission_selection.best_index)
        fission_selected_iteration = int(fission_checkpoint_iterations[selected_index])
        fission_best_iteration = int(fission_checkpoint_iterations[best_index])
        fission_selected_validation_loss = float(
            fission_selection.validation_losses[selected_index]
        )
        fission_best_validation_loss = float(fission_selection.validation_losses[best_index])
        fission_terminal_validation_loss = float(fission_selection.validation_losses[-1])

        original_variance, original_inv_variance, _ = _objective_variance_arrays(
            measurements.variances.astype(np.float32),
            measurements.inv_variances.astype(np.float32),
            objective_weighting,
        )
        del original_variance
        original_inv_variance = (
            original_inv_variance * objective_query_weight_multipliers
        ).astype(np.float32)
        train_losses: list[float] = []
        original_released_losses: list[float] = []
        for checkpoint_answers in fission_checkpoint_answers:
            train_losses.append(
                measured_loss(
                    (state.target - checkpoint_answers).astype(np.float32),
                    state.inv_variance,
                )
            )
            original_released_losses.append(
                measured_loss(
                    (
                        measurements.target_projected.astype(np.float32)
                        - checkpoint_answers
                    ).astype(np.float32),
                    original_inv_variance,
                )
            )
        fission_selected_original_released_loss = float(
            original_released_losses[selected_index]
        )
        fission_terminal_original_released_loss = float(original_released_losses[-1])
        checkpoint_rows = []
        for index, iteration_value in enumerate(fission_checkpoint_iterations):
            checkpoint_rows.append(
                {
                    "index": int(index),
                    "iteration": int(iteration_value),
                    "train_measured_loss": float(train_losses[index]),
                    "original_released_measured_loss": float(
                        original_released_losses[index]
                    ),
                    "validation_loss": float(fission_selection.validation_losses[index]),
                    "validation_avg_partition_l1": float(
                        fission_selection.validation_avg_partition_l1[index]
                    ),
                    "validation_avg_partition_l2_upper": float(
                        fission_selection.validation_avg_partition_l2_upper[index]
                    ),
                    "standard_error_to_best": float(
                        fission_selection.standard_errors_to_best[index]
                    ),
                    "eligible_within_one_se": bool(fission_selection.eligible[index]),
                    "is_validation_best": bool(index == best_index),
                    "is_selected": bool(index == selected_index),
                }
            )
        write_json(
            {
                "enabled": True,
                "optimization_branch": fission_optimization_branch,
                "holdout_branch": fission_holdout_branch,
                "selection_rule": str(
                    fission_cfg.get("selection_rule", "earliest_within_one_se")
                ),
                "one_se_multiplier": float(fission_cfg.get("one_se_multiplier", 1.0)),
                "selected_index": selected_index,
                "selected_iteration": fission_selected_iteration,
                "best_index": best_index,
                "best_iteration": fission_best_iteration,
                "terminal_iteration": fission_terminal_iteration,
                "checkpoints": checkpoint_rows,
            },
            output_dir / "fission_checkpoints.json",
        )
        state.X_syn = fission_checkpoint_tables[selected_index].copy()
        state.answer_syn = fission_checkpoint_answers[selected_index].copy()
        state.residual = (state.target - state.answer_syn).astype(np.float32)
        log(
            "Fission checkpoint selection: "
            f"selected_iter={fission_selected_iteration}, "
            f"best_iter={fission_best_iteration}, "
            f"terminal_iter={fission_terminal_iteration}, "
            f"selected_validation_loss={fission_selected_validation_loss:.6g}, "
            f"best_validation_loss={fission_best_validation_loss:.6g}"
        )
        fission_checkpoint_tables.clear()

    final_answers = answer_queries(state.X_syn, qcat, batch_size=int(runtime_cfg.get("answer_batch_size", 8192)))
    heldout_final_answers: np.ndarray | None = None
    if heldout_qcat is not None:
        heldout_final_answers = answer_queries(
            state.X_syn,
            heldout_qcat,
            batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
        )
    final_incremental_answer_drift = float(np.max(np.abs(final_answers - state.answer_syn)))
    if final_incremental_answer_drift > residual_drift_tolerance:
        raise AssertionError(
            "Final incremental answer drift "
            f"{final_incremental_answer_drift} exceeds tolerance {residual_drift_tolerance}"
        )
    state.answer_syn = final_answers.astype(np.float32)
    state.residual = (state.target - state.answer_syn).astype(np.float32)
    final_loss = measured_loss(state.residual, state.inv_variance)
    final_optimization_objective = _optimization_objective_value(
        state.residual,
        objective_loss,
        state.inv_variance,
        objective_loss_weights,
        optimization_precision,
    )
    if entropy_state is not None:
        if entropy_controller is None or entropy_prior is None:
            raise RuntimeError("Final entropy state is missing its controller or prior")
        entropy_state.assert_matches(state.X_syn)
        final_entropy_regularizer = float(entropy_state.regularizer)
        final_regularized_objective = entropy_controller.regularized_objective(
            final_optimization_objective,
            final_entropy_regularizer,
        )
        entropy_diagnostics = {
            "enabled": True,
            "prior": entropy_prior.diagnostics(),
            "state": entropy_state.diagnostics(),
            "controller": entropy_controller.diagnostics(final_optimization_objective),
        }
    else:
        final_entropy_regularizer = 0.0
        final_regularized_objective = final_optimization_objective
        entropy_diagnostics = {
            "enabled": False,
            "prior": None,
            "state": None,
            "controller": None,
        }
    confidence_stop_final_reached = bool(
        confidence_stop is not None
        and confidence_stop.reached(final_optimization_objective)
    )
    confidence_stop_diagnostics = (
        confidence_stop.diagnostics()
        if confidence_stop is not None
        else {
            "method": "none",
            "alpha": None,
            "confidence_level": None,
            "effective_rank": None,
            "squared_discrepancy_threshold": None,
            "objective_threshold": None,
        }
    )
    final_unweighted_loss = unweighted_measured_loss(state.residual)
    final_rms = rms_standardized_residual(final_loss, qcat.m)
    final_unweighted_rms = rms_unweighted_residual(state.residual)
    loss_reduction = float(initial_loss - final_loss)
    optimization_objective_reduction = float(initial_optimization_objective - final_optimization_objective)
    unweighted_loss_reduction = float(initial_unweighted_loss - final_unweighted_loss)
    log(f"Final measured loss: {final_loss:.6g}")
    log(f"Final optimization objective: {final_optimization_objective:.6g}")
    if entropy_state is not None and entropy_controller is not None:
        log(
            "Final entropy state: "
            f"regularizer={final_entropy_regularizer:.6g}, "
            f"kl_per_row={final_entropy_regularizer / n_syn:.6g}, "
            f"regularized_objective={final_regularized_objective:.6g}, "
            f"dual={entropy_controller.dual_weight:.6g}, "
            f"inside_confidence={entropy_controller.inside(final_optimization_objective)}"
        )
    log(f"Final unweighted measured loss: {final_unweighted_loss:.6g}")
    log(f"Final incremental answer drift before recompute: {final_incremental_answer_drift:.6g}")
    log(f"Candidates scored: {stats.num_candidates_scored}")
    log(f"Accepted edits: {stats.num_accepted_edits}")

    save_npy(state.X_syn, output_dir / "synthetic_encoded.npy")
    if bool(evaluation_cfg.get("save_synthetic_csv", True)):
        decode_array(state.X_syn, schema).to_csv(output_dir / "synthetic_decoded.csv", index=False)

    final_metrics: dict[str, Any] = {
        "dataset_name": run_cfg.get("dataset_name"),
        "privacy_mode": measurements.mode,
        "dp_release_mode": bool(config.get("privacy", {}).get("dp_release_mode", False)),
        "privacy_adjacency": str(config.get("privacy", {}).get("adjacency", "unspecified")),
        "public_row_count_declared": bool(config.get("privacy", {}).get("public_row_count", False)),
        "public_n_rows_declared": public_n_rows,
        "public_schema_path": str(public_schema_path) if public_schema_path is not None else "",
        "transcript_only_generation": transcript_only_generation,
        "private_input_loaded_by_process": not transcript_only_generation,
        "measurement_reused": bool(reuse_path is not None),
        "measurement_reuse_from": str(reuse_path) if reuse_path is not None else "",
        "rho_total": measurements.rho_total,
        "rho_spent": measurements.rho_spent,
        "delta": measurements.delta,
        "epsilon_delta": measurements.epsilon_delta,
        "num_rows_real": n_real,
        "num_rows_synthetic": int(state.X_syn.shape[0]),
        "num_columns": int(schema.d),
        "num_queries": int(qcat.m),
        "num_measurement_groups": int(len(measurements.groups)),
        "initial_measured_loss": initial_loss,
        "final_measured_loss": final_loss,
        "loss_reduction": loss_reduction,
        "initial_optimization_objective": initial_optimization_objective,
        "final_optimization_objective": final_optimization_objective,
        "optimization_objective_reduction": optimization_objective_reduction,
        "entropy_enabled": bool(entropy_state is not None),
        "entropy_diagnostics": entropy_diagnostics,
        "initial_entropy_regularizer": float(initial_entropy_regularizer),
        "final_entropy_regularizer": float(final_entropy_regularizer),
        "entropy_regularizer_reduction": float(
            initial_entropy_regularizer - final_entropy_regularizer
        ),
        "initial_entropy_kl_per_row": float(initial_entropy_regularizer / n_syn),
        "final_entropy_kl_per_row": float(final_entropy_regularizer / n_syn),
        "initial_regularized_objective": float(initial_regularized_objective),
        "final_regularized_objective": float(final_regularized_objective),
        "initial_unweighted_measured_loss": initial_unweighted_loss,
        "final_unweighted_measured_loss": final_unweighted_loss,
        "unweighted_measured_loss_reduction": unweighted_loss_reduction,
        "initial_rms_standardized_residual": initial_rms,
        "final_rms_standardized_residual": final_rms,
        "initial_rms_unweighted_residual": initial_unweighted_rms,
        "final_rms_unweighted_residual": final_unweighted_rms,
        "objective_weighting": objective_weighting,
        "objective_loss": objective_loss,
        "precision_operator": precision_operator_name,
        "precision_operator_active": bool(optimization_precision is not None),
        "precision_operator_diagnostics": precision_diagnostics,
        "confidence_stop_enabled": bool(confidence_stop is not None),
        "confidence_stop_diagnostics": confidence_stop_diagnostics,
        "confidence_stop_initial_reached": confidence_stop_initial_reached,
        "confidence_stop_triggered": confidence_stop_triggered,
        "confidence_stop_iteration": confidence_stop_iteration,
        "confidence_stop_objective_at_trigger": confidence_stop_objective_at_trigger,
        "confidence_stop_final_reached": confidence_stop_final_reached,
        "proposal_precision": (
            "marginal_diagonal_surrogate"
            if precision_operator_name.startswith("orthogonal_interaction")
            else "objective_diagonal"
        ),
        "objective_loss_weight_mean": float(np.mean(objective_loss_weights)),
        "objective_weight_profile": objective_weight_profile,
        "objective_query_weight_multiplier_mean": float(
            np.mean(objective_query_weight_multipliers)
        ),
        "objective_query_weight_multiplier_min": float(
            np.min(objective_query_weight_multipliers)
        ),
        "objective_query_weight_multiplier_max": float(
            np.max(objective_query_weight_multipliers)
        ),
        "objective_loss_weight_min": float(np.min(objective_loss_weights)),
        "objective_loss_weight_max": float(np.max(objective_loss_weights)),
        "objective_variance_mean": float(np.mean(state.variance)),
        "objective_variance_min": float(np.min(state.variance)),
        "objective_variance_max": float(np.max(state.variance)),
        "objective_inv_variance_mean": float(np.mean(state.inv_variance)),
        "objective_inv_variance_min": float(np.min(state.inv_variance)),
        "objective_inv_variance_max": float(np.max(state.inv_variance)),
        "measurement_variance_mean": float(np.mean(measurement_variance)),
        "measurement_variance_min": float(np.min(measurement_variance)),
        "measurement_variance_max": float(np.max(measurement_variance)),
        "measurement_inv_variance_mean": float(np.mean(measurement_inv_variance)),
        "measurement_inv_variance_min": float(np.min(measurement_inv_variance)),
        "measurement_inv_variance_max": float(np.max(measurement_inv_variance)),
        "measurement_fission_enabled": bool(fission_enabled),
        "measurement_fission_optimization_branch": fission_optimization_branch,
        "measurement_fission_holdout_branch": fission_holdout_branch,
        "measurement_fission_train_fraction": float(
            fission_split.train_fraction if fission_split is not None else 1.0
        ),
        "measurement_fission_gamma": float(
            fission_split.gamma if fission_split is not None else 0.0
        ),
        "measurement_fission_reconstruction_max_abs": float(
            fission_reconstruction_max_abs
        ),
        "measurement_fission_checkpoint_interval": int(fission_checkpoint_interval),
        "measurement_fission_selected_iteration": int(fission_selected_iteration),
        "measurement_fission_best_iteration": int(fission_best_iteration),
        "measurement_fission_terminal_iteration": int(fission_terminal_iteration),
        "measurement_fission_selected_validation_loss": float(
            fission_selected_validation_loss
        ),
        "measurement_fission_best_validation_loss": float(fission_best_validation_loss),
        "measurement_fission_terminal_validation_loss": float(
            fission_terminal_validation_loss
        ),
        "measurement_fission_terminal_train_loss": float(fission_terminal_train_loss),
        "measurement_fission_selected_original_released_loss": float(
            fission_selected_original_released_loss
        ),
        "measurement_fission_terminal_original_released_loss": float(
            fission_terminal_original_released_loss
        ),
        "final_incremental_answer_drift": final_incremental_answer_drift,
        "num_candidates_scored": int(stats.num_candidates_scored),
        "num_candidates_requested": int(stats.num_candidates_requested),
        "num_candidate_shortfall": int(stats.num_candidate_shortfall),
        "num_accepted_edits": int(stats.num_accepted_edits),
        "structured_swap_enabled": bool(structured_swap_enabled),
        "structured_swap_compiler": structured_swap_compiler,
        "structured_swap_start_iter": int(structured_swap_start_iter),
        "structured_swap_interval": int(structured_swap_interval),
        "structured_swap_candidate_units": int(structured_swap_candidate_units),
        "structured_swap_transport_pool": int(structured_swap_transport_pool),
        "structured_swap_accept_start": int(structured_swap_accept_start),
        "structured_swap_accept_end": int(structured_swap_accept_end),
        "structured_swap_accept_schedule": structured_swap_accept_schedule,
        "structured_swap_noise_guard_kappa": float(structured_swap_noise_guard_kappa),
        "structured_swap_noise_guard_mode": structured_swap_noise_guard_mode,
        "structured_swap_noise_guard_alpha": float(structured_swap_noise_guard_alpha),
        "structured_swap_max_effective_noise_guard_kappa": float(
            structured_swap_max_effective_noise_guard_kappa
        ),
        "structured_swap_trigger_rms": float(structured_swap_trigger_rms),
        "structured_swap_delta_backend": structured_swap_delta_backend,
        "structured_swap_exploration_floor": float(structured_swap_exploration_floor),
        "structured_swap_policy_decay": float(structured_swap_policy_decay),
        "structured_swap_policy_prior_strength": float(structured_swap_policy_prior_strength),
        "structured_swap_total_candidates": int(structured_swap_total_candidates),
        "structured_swap_total_accepted": int(structured_swap_total_accepted),
        "structured_swap_total_noise_guard_rejections": int(
            structured_swap_total_noise_guard_rejections
        ),
        "structured_swap_total_objective_advantage": float(
            structured_swap_total_objective_advantage
        ),
        "structured_swap_compiler_calls": int(structured_swap_compiler_calls),
        "structured_swap_total_positive_rectangles": int(
            structured_swap_total_positive_rectangles
        ),
        "structured_swap_max_selected_scopes": int(structured_swap_max_selected_scopes),
        "structured_swap_target_linear_gain_mean": float(
            structured_swap_target_linear_gain_sum
            / max(1, structured_swap_target_linear_gain_count)
        ),
        "structured_swap_last_compiler_diagnostics": (
            structured_swap_last_compiler_diagnostics
            if structured_swap_last_compiler_diagnostics
            else None
        ),
        "structured_swap_time_seconds": float(structured_swap_time_seconds),
        "structured_swap_generation_time_seconds": float(
            structured_swap_generation_time_seconds
        ),
        "structured_swap_scoring_time_seconds": float(structured_swap_scoring_time_seconds),
        "structured_swap_policy_pool_time_seconds": float(
            structured_swap_policy_pool_time_seconds
        ),
        "structured_swap_delta_time_seconds": float(structured_swap_delta_time_seconds),
        "structured_swap_transport_time_seconds": float(
            structured_swap_transport_time_seconds
        ),
        "structured_swap_apply_time_seconds": float(structured_swap_apply_time_seconds),
        "structured_swap_policy_final": (
            structured_swap_policy.diagnostics() if structured_swap_policy is not None else None
        ),
        "accepted_per_iter": int(accepted_per_iter),
        "accepted_per_iter_schedule": accepted_per_iter_schedule,
        "accepted_per_iter_start": int(accepted_per_iter_start),
        "accepted_per_iter_end": int(accepted_per_iter_end),
        "accepted_per_iter_warmup_iters": int(accepted_per_iter_warmup_iters),
        "accepted_per_iter_anneal_iters": int(accepted_per_iter_anneal_iters),
        "gpu_device_count": int(jax.local_device_count()),
        "score_backend": score_backend,
        "candidate_backend": candidate_backend,
        "transport_mode": transport_mode,
        "transport_delta_backend": transport_delta_backend,
        "transport_prefix_strategy": transport_prefix_strategy,
        "atom_flow_pool_multiplier": atom_flow_pool_multiplier,
        "atom_flow_max_pool": atom_flow_max_pool,
        "atom_flow_update_mode": atom_flow_update_mode,
        "constructive_pair_pool_multiplier": constructive_pair_pool_multiplier,
        "constructive_pair_max_pool": constructive_pair_max_pool,
        "constructive_pair_partner_limit": constructive_pair_partner_limit,
        "constructive_pair_harm_query_limit": constructive_pair_harm_query_limit,
        "constructive_pair_max_units": constructive_pair_max_units,
        "constructive_pair_min_target_component": constructive_pair_min_target_component,
        "constructive_pair_group_augment": constructive_pair_group_augment,
        "constructive_pair_group_augment_trigger": constructive_pair_group_augment_trigger,
        "constructive_pair_group_augment_threshold": constructive_pair_group_augment_threshold,
        "constructive_pair_group_augment_max_loss": constructive_pair_group_augment_max_loss,
        "constructive_pair_group_augment_noise_floor_ratio": constructive_pair_group_augment_noise_floor_ratio,
        "constructive_pair_group_augment_plateau_window": constructive_pair_group_augment_plateau_window,
        "constructive_pair_group_augment_plateau_relative_drop": (
            constructive_pair_group_augment_plateau_relative_drop
        ),
        "random_group_count": random_group_count,
        "random_group_min_size": random_group_min_size,
        "random_group_max_size": random_group_max_size,
        "random_group_pool_multiplier": random_group_pool_multiplier,
        "random_group_max_pool": random_group_max_pool,
        "directed_group_seed_count": directed_group_seed_count,
        "directed_group_min_size": directed_group_min_size,
        "directed_group_max_size": directed_group_max_size,
        "directed_group_pool_multiplier": directed_group_pool_multiplier,
        "directed_group_max_pool": directed_group_max_pool,
        "directed_group_allow_negative_steps": directed_group_allow_negative_steps,
        "directed_group_backend": directed_group_backend,
        "directed_group_positive_fill": directed_group_positive_fill,
        "directed_group_positive_fill_augment": directed_group_positive_fill_augment,
        "directed_group_positive_fill_augment_trigger": directed_group_positive_fill_augment_trigger,
        "directed_group_positive_fill_augment_threshold": directed_group_positive_fill_augment_threshold,
        "directed_group_positive_fill_augment_max_loss": directed_group_positive_fill_augment_max_loss,
        "directed_group_positive_fill_augment_noise_floor_loss": objective_noise_floor_loss,
        "directed_group_positive_fill_augment_noise_floor_ratio": (
            directed_group_positive_fill_augment_noise_floor_ratio
        ),
        "directed_group_positive_fill_augment_plateau_window": (
            directed_group_positive_fill_augment_plateau_window
        ),
        "directed_group_positive_fill_augment_plateau_relative_drop": (
            directed_group_positive_fill_augment_plateau_relative_drop
        ),
        "directed_group_augment_seed_count": directed_group_augment_seed_count,
        "directed_group_augment_min_size": directed_group_augment_min_size,
        "directed_group_augment_max_size": directed_group_augment_max_size,
        "directed_group_augment_pool_multiplier": directed_group_augment_pool_multiplier,
        "directed_group_augment_max_pool": directed_group_augment_max_pool,
        "candidate_diagnostics_enabled": bool(candidate_diagnostics_enabled),
        "use_pmap": bool(use_pmap),
    }
    true_answers: np.ndarray | None = None
    metrics_holdout: dict[str, Any] | None = None
    metrics_by_family_holdout: dict[str, dict[str, float | int]] | None = None
    if compute_true_eval:
        if X_real_for_evaluation is None:
            raise RuntimeError("Internal error: true-query evaluation requested but real data reference was cleared.")
        log("Computing exact true query answers for offline evaluation metrics only.")
        true_answers = answer_queries(
            X_real_for_evaluation,
            qcat,
            batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
        )
        initial_true_metrics = query_error_metrics(
            true_answers,
            initial_answers,
            n_real,
            state.X_syn.shape[0],
            prefix="initial_true_query",
        )
        final_true_metrics = query_error_metrics(
            true_answers,
            final_answers,
            n_real,
            state.X_syn.shape[0],
            prefix="final_true_query",
        )
        final_alias_metrics = query_error_metrics(
            true_answers,
            final_answers,
            n_real,
            state.X_syn.shape[0],
            prefix="true_query",
        )
        final_metrics.update(initial_true_metrics)
        final_metrics.update(final_true_metrics)
        final_metrics.update(final_alias_metrics)
        final_metrics["true_query_mae_reduction"] = float(
            initial_true_metrics["initial_true_query_mae"] - final_true_metrics["final_true_query_mae"]
        )
        final_metrics["true_query_rmse_reduction"] = float(
            initial_true_metrics["initial_true_query_rmse"] - final_true_metrics["final_true_query_rmse"]
        )
        log(
            "True query error for evaluation only: "
            f"initial_MAE={final_metrics['initial_true_query_mae']:.6g}, "
            f"final_MAE={final_metrics['final_true_query_mae']:.6g}, "
            f"final_RMSE={final_metrics['final_true_query_rmse']:.6g}"
        )
        oracle_bias_diagnostics = _oracle_projection_bias_diagnostics(
            true_answers=true_answers,
            measurements=measurements,
            qcat=qcat,
            schema=schema,
            n_real=n_real,
            config=config,
            seed=seed,
            output_dir=output_dir,
            log=log,
        )
        if oracle_bias_diagnostics is not None:
            for key, value in oracle_bias_diagnostics.items():
                if isinstance(value, (bool, int, float, str)):
                    final_metrics[f"oracle_projection_bias_{key}"] = value
    if compute_heldout_eval:
        if X_real_for_evaluation is None:
            raise RuntimeError("Internal error: held-out evaluation requested but real data reference was cleared.")
        if heldout_qcat is None or heldout_initial_answers is None or heldout_final_answers is None:
            raise RuntimeError("Internal error: held-out evaluation requested but held-out answers are unavailable.")
        log("Computing held-out exact true query answers for offline evaluation metrics only.")
        heldout_true_answers = answer_queries(
            X_real_for_evaluation,
            heldout_qcat,
            batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
        )
        metrics_holdout = _true_query_evaluation_metrics(
            heldout_qcat,
            heldout_true_answers,
            heldout_initial_answers,
            heldout_final_answers,
            n_real,
            state.X_syn.shape[0],
        )
        metrics_by_family_holdout = _true_query_metrics_by_family(
            heldout_qcat,
            heldout_true_answers,
            heldout_initial_answers,
            heldout_final_answers,
            n_real,
            state.X_syn.shape[0],
        )
        final_metrics["heldout_num_queries"] = int(metrics_holdout["num_queries"])
        final_metrics["heldout_initial_true_query_mae"] = float(metrics_holdout["initial_true_query_mae"])
        final_metrics["heldout_final_true_query_mae"] = float(metrics_holdout["final_true_query_mae"])
        final_metrics["heldout_true_query_mae_reduction"] = float(metrics_holdout["true_query_mae_reduction"])
        final_metrics["heldout_initial_true_query_rmse"] = float(metrics_holdout["initial_true_query_rmse"])
        final_metrics["heldout_final_true_query_rmse"] = float(metrics_holdout["final_true_query_rmse"])
        final_metrics["heldout_true_query_rmse_reduction"] = float(metrics_holdout["true_query_rmse_reduction"])
        log(
            "Held-out true query error for evaluation only: "
            f"queries={metrics_holdout['num_queries']}, "
            f"initial_MAE={metrics_holdout['initial_true_query_mae']:.6g}, "
            f"final_MAE={metrics_holdout['final_true_query_mae']:.6g}, "
            f"final_RMSE={metrics_holdout['final_true_query_rmse']:.6g}"
        )

    runtime_dict = stats.as_dict()
    efficiency_metrics = {
        "loss_reduction_per_second": float(loss_reduction / max(1.0e-9, stats.time_generation_seconds)),
        "loss_reduction_per_million_candidates": float(
            loss_reduction / max(1.0e-9, stats.num_candidates_scored / 1.0e6)
        ),
        "unweighted_loss_reduction_per_second": float(
            unweighted_loss_reduction / max(1.0e-9, stats.time_generation_seconds)
        ),
        "unweighted_loss_reduction_per_million_candidates": float(
            unweighted_loss_reduction / max(1.0e-9, stats.num_candidates_scored / 1.0e6)
        ),
        "candidates_scored_per_second": float(
            stats.num_candidates_scored / max(1.0e-9, stats.time_generation_seconds)
        ),
        "accepted_edits_per_second": float(stats.num_accepted_edits / max(1.0e-9, stats.time_generation_seconds)),
        "accepted_per_scored_candidate": float(stats.num_accepted_edits / max(1, stats.num_candidates_scored)),
        "scoring_time_fraction": float(stats.time_scoring_seconds / max(1.0e-9, stats.time_generation_seconds)),
        "transport_time_fraction": float(stats.time_transport_seconds / max(1.0e-9, stats.time_generation_seconds)),
    }
    final_metrics.update(efficiency_metrics)
    runtime_dict.update(efficiency_metrics)
    runtime_dict.update(last_debt_diagnostics)
    runtime_dict.update({f"last_{key}": value for key, value in last_transport_diagnostics.items()})
    runtime_dict["positive_advantage_all_rate_available"] = bool(not gpu_topk_return_mode)
    runtime_dict["positive_returned_rate_is_topk_biased"] = bool(gpu_topk_return_mode)
    final_metrics["positive_advantage_all_rate_available"] = bool(not gpu_topk_return_mode)
    final_metrics["positive_returned_rate_is_topk_biased"] = bool(gpu_topk_return_mode)
    runtime_dict["measurement_reused"] = bool(reuse_path is not None)
    runtime_dict["measurement_reuse_from"] = str(reuse_path) if reuse_path is not None else ""
    runtime_dict["objective_weighting"] = objective_weighting
    runtime_dict["objective_weight_profile"] = objective_weight_profile
    runtime_dict["objective_query_weight_multiplier_mean"] = float(
        np.mean(objective_query_weight_multipliers)
    )
    runtime_dict["objective_loss"] = objective_loss
    runtime_dict["entropy_enabled"] = bool(entropy_state is not None)
    runtime_dict["initial_entropy_regularizer"] = float(initial_entropy_regularizer)
    runtime_dict["final_entropy_regularizer"] = float(final_entropy_regularizer)
    runtime_dict["initial_entropy_kl_per_row"] = float(initial_entropy_regularizer / n_syn)
    runtime_dict["final_entropy_kl_per_row"] = float(final_entropy_regularizer / n_syn)
    runtime_dict["initial_regularized_objective"] = float(initial_regularized_objective)
    runtime_dict["final_regularized_objective"] = float(final_regularized_objective)
    runtime_dict["entropy_diagnostics"] = entropy_diagnostics
    runtime_dict["objective_loss_weight_mean"] = float(np.mean(objective_loss_weights))
    runtime_dict["objective_variance_mean"] = float(np.mean(state.variance))
    runtime_dict["objective_inv_variance_mean"] = float(np.mean(state.inv_variance))
    runtime_dict["measurement_variance_mean"] = float(np.mean(measurement_variance))
    runtime_dict["measurement_inv_variance_mean"] = float(np.mean(measurement_inv_variance))
    runtime_dict["measurement_fission_enabled"] = bool(fission_enabled)
    runtime_dict["measurement_fission_optimization_branch"] = fission_optimization_branch
    runtime_dict["measurement_fission_holdout_branch"] = fission_holdout_branch
    runtime_dict["measurement_fission_selected_iteration"] = int(fission_selected_iteration)
    runtime_dict["measurement_fission_best_iteration"] = int(fission_best_iteration)
    runtime_dict["measurement_fission_terminal_iteration"] = int(fission_terminal_iteration)
    runtime_dict["gpu_devices"] = [str(d) for d in jax.devices()]
    runtime_dict["score_backend"] = score_backend
    runtime_dict["candidate_backend"] = candidate_backend
    runtime_dict.update(
        {
            key: value
            for key, value in final_metrics.items()
            if key.startswith("structured_swap_")
        }
    )
    runtime_dict["accepted_per_iter"] = int(accepted_per_iter)
    runtime_dict["accepted_per_iter_schedule"] = accepted_per_iter_schedule
    runtime_dict["accepted_per_iter_start"] = int(accepted_per_iter_start)
    runtime_dict["accepted_per_iter_end"] = int(accepted_per_iter_end)
    runtime_dict["accepted_per_iter_warmup_iters"] = int(accepted_per_iter_warmup_iters)
    runtime_dict["accepted_per_iter_anneal_iters"] = int(accepted_per_iter_anneal_iters)
    runtime_dict["transport_delta_backend"] = transport_delta_backend
    runtime_dict["transport_prefix_strategy"] = transport_prefix_strategy
    runtime_dict["transport_mode"] = transport_mode
    runtime_dict["atom_flow_pool_multiplier"] = atom_flow_pool_multiplier
    runtime_dict["atom_flow_max_pool"] = atom_flow_max_pool
    runtime_dict["atom_flow_update_mode"] = atom_flow_update_mode
    runtime_dict["constructive_pair_pool_multiplier"] = constructive_pair_pool_multiplier
    runtime_dict["constructive_pair_max_pool"] = constructive_pair_max_pool
    runtime_dict["constructive_pair_partner_limit"] = constructive_pair_partner_limit
    runtime_dict["constructive_pair_harm_query_limit"] = constructive_pair_harm_query_limit
    runtime_dict["constructive_pair_max_units"] = constructive_pair_max_units
    runtime_dict["constructive_pair_min_target_component"] = constructive_pair_min_target_component
    runtime_dict["constructive_pair_group_augment"] = bool(constructive_pair_group_augment)
    runtime_dict["constructive_pair_group_augment_trigger"] = constructive_pair_group_augment_trigger
    runtime_dict["constructive_pair_group_augment_threshold"] = int(constructive_pair_group_augment_threshold)
    runtime_dict["constructive_pair_group_augment_max_loss"] = float(constructive_pair_group_augment_max_loss)
    runtime_dict["constructive_pair_group_augment_noise_floor_ratio"] = float(
        constructive_pair_group_augment_noise_floor_ratio
    )
    runtime_dict["constructive_pair_group_augment_plateau_window"] = int(
        constructive_pair_group_augment_plateau_window
    )
    runtime_dict["constructive_pair_group_augment_plateau_relative_drop"] = float(
        constructive_pair_group_augment_plateau_relative_drop
    )
    runtime_dict["random_group_count"] = random_group_count
    runtime_dict["random_group_min_size"] = random_group_min_size
    runtime_dict["random_group_max_size"] = random_group_max_size
    runtime_dict["random_group_pool_multiplier"] = random_group_pool_multiplier
    runtime_dict["random_group_max_pool"] = random_group_max_pool
    runtime_dict["directed_group_seed_count"] = directed_group_seed_count
    runtime_dict["directed_group_min_size"] = directed_group_min_size
    runtime_dict["directed_group_max_size"] = directed_group_max_size
    runtime_dict["directed_group_pool_multiplier"] = directed_group_pool_multiplier
    runtime_dict["directed_group_max_pool"] = directed_group_max_pool
    runtime_dict["directed_group_allow_negative_steps"] = bool(directed_group_allow_negative_steps)
    runtime_dict["directed_group_backend"] = directed_group_backend
    runtime_dict["directed_group_positive_fill"] = bool(directed_group_positive_fill)
    runtime_dict["directed_group_positive_fill_augment"] = bool(directed_group_positive_fill_augment)
    runtime_dict["directed_group_positive_fill_augment_trigger"] = directed_group_positive_fill_augment_trigger
    runtime_dict["directed_group_positive_fill_augment_threshold"] = int(
        directed_group_positive_fill_augment_threshold
    )
    runtime_dict["directed_group_positive_fill_augment_max_loss"] = float(
        directed_group_positive_fill_augment_max_loss
    )
    runtime_dict["directed_group_positive_fill_augment_noise_floor_loss"] = float(objective_noise_floor_loss)
    runtime_dict["directed_group_positive_fill_augment_noise_floor_ratio"] = float(
        directed_group_positive_fill_augment_noise_floor_ratio
    )
    runtime_dict["directed_group_positive_fill_augment_plateau_window"] = int(
        directed_group_positive_fill_augment_plateau_window
    )
    runtime_dict["directed_group_positive_fill_augment_plateau_relative_drop"] = float(
        directed_group_positive_fill_augment_plateau_relative_drop
    )
    runtime_dict["directed_group_augment_seed_count"] = int(directed_group_augment_seed_count)
    runtime_dict["directed_group_augment_min_size"] = int(directed_group_augment_min_size)
    runtime_dict["directed_group_augment_max_size"] = int(directed_group_augment_max_size)
    runtime_dict["directed_group_augment_pool_multiplier"] = int(directed_group_augment_pool_multiplier)
    runtime_dict["directed_group_augment_max_pool"] = int(directed_group_augment_max_pool)
    runtime_dict["candidate_diagnostics_enabled"] = bool(candidate_diagnostics_enabled)
    runtime_dict["precision_operator"] = precision_operator_name
    runtime_dict["precision_operator_active"] = bool(optimization_precision is not None)
    runtime_dict["precision_operator_diagnostics"] = precision_diagnostics
    runtime_dict["confidence_stop_enabled"] = bool(confidence_stop is not None)
    runtime_dict["confidence_stop_diagnostics"] = confidence_stop_diagnostics
    runtime_dict["confidence_stop_initial_reached"] = confidence_stop_initial_reached
    runtime_dict["confidence_stop_triggered"] = confidence_stop_triggered
    runtime_dict["confidence_stop_iteration"] = confidence_stop_iteration
    runtime_dict["confidence_stop_objective_at_trigger"] = confidence_stop_objective_at_trigger
    runtime_dict["confidence_stop_final_reached"] = confidence_stop_final_reached
    runtime_dict["use_pmap"] = bool(use_pmap)
    runtime_dict["gpu_batches_per_iter"] = int(qdte_cfg.get("gpu_batches_per_iter", 1))
    runtime_dict["gpu_score_query_block_size"] = int(
        gpu_candidate_context.score_query_block_size if gpu_candidate_context is not None else 0
    )
    runtime_dict["gpu_score_query_block_count"] = int(
        gpu_candidate_context.score_query_block_count if gpu_candidate_context is not None else 0
    )
    runtime_dict["gpu_sparse_score_mode"] = int(
        gpu_candidate_context.sparse_score_mode if gpu_candidate_context is not None else 0
    )
    runtime_dict["gpu_sparse_query_block_size"] = int(
        gpu_candidate_context.sparse_query_block_size if gpu_candidate_context is not None else 0
    )
    runtime_dict["gpu_sparse_query_block_count"] = int(
        gpu_candidate_context.sparse_query_block_count if gpu_candidate_context is not None else 0
    )
    runtime_dict["gpu_sparse_changed_attr_capacity"] = int(
        gpu_candidate_context.sparse_changed_attr_capacity if gpu_candidate_context is not None else 0
    )
    runtime_dict["gpu_return_top_k"] = int(qdte_cfg.get("gpu_return_top_k", 0))
    runtime_dict["total_candidates_per_iter"] = configured_total_candidates
    runtime_dict["transcript_only_generation"] = transcript_only_generation
    runtime_dict["private_input_loaded_by_process"] = not transcript_only_generation
    metrics_by_family = _metrics_by_family(
        qcat,
        initial_residual,
        state.residual,
        state.inv_variance,
        true_answers,
        initial_answers,
        final_answers,
        n_real,
        state.X_syn.shape[0],
    )
    write_json(final_metrics, output_dir / "metrics_final.json")
    write_json(metrics_by_family, output_dir / "metrics_by_family.json")
    if metrics_holdout is not None and metrics_by_family_holdout is not None:
        write_json(metrics_holdout, output_dir / "metrics_holdout.json")
        write_json(metrics_by_family_holdout, output_dir / "metrics_by_family_holdout.json")
    write_json(workload_summary, output_dir / "workload_summary.json")
    _write_timeseries(timeseries, output_dir / "metrics_timeseries.csv")
    if candidate_diagnostic_rows:
        pd.DataFrame(candidate_diagnostic_rows).to_csv(
            output_dir / "candidate_diagnostics_timeseries.csv",
            index=False,
        )
    write_json(runtime_dict, output_dir / "runtime.json")
    (output_dir / "logs.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    write_json(
        {
            "status": "completed",
            "seed": int(seed),
            "num_iterations": int(stats.num_iterations),
            "final_measured_loss": float(final_loss),
            "final_optimization_objective": float(final_optimization_objective),
            "confidence_stop_triggered": confidence_stop_triggered,
            "confidence_stop_iteration": confidence_stop_iteration,
        },
        output_dir / "run_status.json",
    )
    return final_metrics
