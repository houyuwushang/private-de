#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import apply_overrides, load_yaml, save_yaml, set_nested
from qdte.dataio import ensure_dir, read_json, write_json
from qdte.eval.metrics import query_error_metrics
from qdte.evolution.engine import run_qdte
from qdte.measurement.measure import Measurements, MeasurementGroup, _apply_configured_projection
from qdte.preprocess import load_and_preprocess_csv
from qdte.privacy.accountant import zcdp_epsilon
from qdte.privacy.exponential import sample_exponential_mechanism
from qdte.queries.eval_jax import answer_queries, eval_records_queries
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue, filter_query_catalogue
from qdte.queries.workload import WorkloadGroup, build_workload
from qdte.schema import TableSchema


_NORMAL = NormalDist()
_HALFNORMAL_TOP_CUMSUM_CACHE: dict[int, np.ndarray] = {}


@dataclass(frozen=True)
class AdaptiveBlock:
    name: str
    family: str
    query_indices: np.ndarray
    delta_l2: float
    is_vector: bool
    scope: tuple[int, ...]
    coverage_weight: float


@dataclass(frozen=True)
class MeasurementRecord:
    block_id: int
    noisy: np.ndarray
    measurement_sigma: float | None = None
    selection_epsilon: float | None = None
    selection_rho: float | None = None


@dataclass(frozen=True)
class BudgetConfig:
    mode: str
    epsilon: float
    measurement_sigma: float
    rho_total: float
    budget_split_mu: float | None


@dataclass(frozen=True)
class MeasurementRoundPlan:
    sigma: float
    rho: float
    exhausts_budget: bool


@dataclass(frozen=True)
class PrivateRoundPlan:
    measurement_sigma: float
    measurement_rho: float
    selection_epsilon: float
    selection_rho: float
    exhausts_budget: bool


@dataclass(frozen=True)
class QueryRepairPlan:
    target_qids: np.ndarray
    cell_indices: tuple[np.ndarray, ...]
    noise_counts: np.ndarray


def _plan_annealed_measurement_round(
    *,
    current_sigma: float,
    remaining_rho: float,
    remaining_rounds: int,
) -> MeasurementRoundPlan:
    """Match AIM's final-round budget handling without reading private answers."""
    if not np.isfinite(current_sigma) or float(current_sigma) <= 0.0:
        raise ValueError("current_sigma must be positive and finite")
    if not np.isfinite(remaining_rho) or float(remaining_rho) <= 0.0:
        raise ValueError("remaining_rho must be positive and finite")
    if int(remaining_rounds) <= 0:
        raise ValueError("remaining_rounds must be positive")

    nominal_rho = 1.0 / (2.0 * float(current_sigma) ** 2)
    tolerance = 1.0e-12 * max(1.0, float(remaining_rho), nominal_rho)
    exhausts_budget = (
        int(remaining_rounds) == 1
        or float(remaining_rho) < 2.0 * nominal_rho - tolerance
    )
    round_rho = float(remaining_rho) if exhausts_budget else nominal_rho
    round_sigma = math.sqrt(1.0 / (2.0 * round_rho))
    return MeasurementRoundPlan(
        sigma=float(round_sigma),
        rho=float(round_rho),
        exhausts_budget=bool(exhausts_budget),
    )


def _plan_annealed_private_round(
    *,
    current_measurement_sigma: float,
    current_selection_epsilon: float,
    remaining_rho: float,
    remaining_rounds: int,
    charge_selection: bool,
) -> PrivateRoundPlan:
    """Plan one AIM-style round while accounting measurement and EM together."""
    if not np.isfinite(current_measurement_sigma) or float(current_measurement_sigma) <= 0.0:
        raise ValueError("current_measurement_sigma must be positive and finite")
    if not np.isfinite(current_selection_epsilon) or float(current_selection_epsilon) < 0.0:
        raise ValueError("current_selection_epsilon must be finite and non-negative")
    if bool(charge_selection) and float(current_selection_epsilon) <= 0.0:
        raise ValueError("A charged private-selection round needs positive epsilon")
    if not np.isfinite(remaining_rho) or float(remaining_rho) <= 0.0:
        raise ValueError("remaining_rho must be positive and finite")
    if int(remaining_rounds) <= 0:
        raise ValueError("remaining_rounds must be positive")

    nominal_measurement_rho = 1.0 / (2.0 * float(current_measurement_sigma) ** 2)
    nominal_selection_rho = (
        float(current_selection_epsilon) ** 2 / 8.0 if bool(charge_selection) else 0.0
    )
    nominal_total_rho = nominal_measurement_rho + nominal_selection_rho
    tolerance = 1.0e-12 * max(1.0, float(remaining_rho), nominal_total_rho)
    exhausts_budget = (
        int(remaining_rounds) == 1
        or float(remaining_rho) < 2.0 * nominal_total_rho - tolerance
    )
    if exhausts_budget:
        measurement_share = nominal_measurement_rho / nominal_total_rho
        measurement_rho = float(remaining_rho) * measurement_share
        selection_rho = float(remaining_rho) - measurement_rho
    else:
        measurement_rho = nominal_measurement_rho
        selection_rho = nominal_selection_rho
    measurement_sigma = math.sqrt(1.0 / (2.0 * measurement_rho))
    selection_epsilon = math.sqrt(8.0 * selection_rho) if charge_selection else 0.0
    return PrivateRoundPlan(
        measurement_sigma=float(measurement_sigma),
        measurement_rho=float(measurement_rho),
        selection_epsilon=float(selection_epsilon),
        selection_rho=float(selection_rho),
        exhausts_budget=bool(exhausts_budget),
    )


def _released_model_change_anneal_diagnostic(
    *,
    before_answers: np.ndarray,
    after_answers: np.ndarray,
    query_indices: np.ndarray,
    noise_std: float,
) -> tuple[float, float, bool]:
    """Return AIM's L1 annealing signal using released-state predictions only."""
    idx = np.asarray(query_indices, dtype=np.int32)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("query_indices must be a non-empty vector")
    if not np.isfinite(noise_std) or float(noise_std) < 0.0:
        raise ValueError("noise_std must be finite and non-negative")
    before = np.asarray(before_answers, dtype=np.float64)
    after = np.asarray(after_answers, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("before_answers and after_answers must be same-shape vectors")
    if int(np.max(idx)) >= int(before.size) or int(np.min(idx)) < 0:
        raise ValueError("query_indices are outside the answer vectors")

    model_change_l1 = float(np.sum(np.abs(after[idx] - before[idx])))
    expected_noise_l1 = float(noise_std) * math.sqrt(2.0 / math.pi) * float(idx.size)
    return model_change_l1, expected_noise_l1, bool(model_change_l1 <= expected_noise_l1)


@dataclass
class ConditionalPropagationContext:
    assignments: dict[int, np.ndarray]
    counts: dict[int, np.ndarray]
    vector_block_ids: list[int]


@dataclass(frozen=True)
class CapacityRiskContext:
    families: tuple[str, ...]
    qids_by_family: tuple[np.ndarray, ...]
    vector_positions_by_family: tuple[np.ndarray, ...]
    atom_caps: np.ndarray


@dataclass(frozen=True)
class EdgeFlowRiskContext:
    qids: np.ndarray
    block_positions: np.ndarray
    edge_caps: np.ndarray
    block_caps: np.ndarray
    query_caps: np.ndarray
    block_qids_by_position: tuple[np.ndarray, ...]


def _resolve_budget(args: argparse.Namespace, config: dict[str, Any]) -> BudgetConfig:
    if args.budget_mode == "per_round":
        epsilon = float(args.epsilon)
        measurement_sigma = float(args.measurement_sigma)
        rho_select = (
            0.0
            if args.selection_input == "transcript" and args.selection_ledger == "measurement_only"
            else epsilon * epsilon / 8.0
        )
        rho_total = int(args.rounds) * (rho_select + 1.0 / (2.0 * measurement_sigma * measurement_sigma))
        return BudgetConfig(
            mode="per_round",
            epsilon=epsilon,
            measurement_sigma=measurement_sigma,
            rho_total=float(rho_total),
            budget_split_mu=None,
        )
    rho_total = float(args.rho_total)
    if rho_total <= 0.0:
        rho_total = float(config.get("privacy", {}).get("rho_total", 1.0))
    rounds = max(1, int(args.rounds))
    mu = float(args.budget_split_mu)
    if not 0.0 < mu < 1.0:
        raise ValueError("--budget-split-mu must be in (0, 1)")
    if args.selection_input == "transcript" and args.selection_ledger == "measurement_only":
        epsilon = float(args.selection_temperature)
        if epsilon <= 0.0:
            raise ValueError("--selection-temperature must be positive for measurement-only transcript selection")
        measurement_sigma = math.sqrt(rounds / (2.0 * rho_total))
        return BudgetConfig(
            mode="total_zcdp_measurement_only_selection",
            epsilon=float(epsilon),
            measurement_sigma=float(measurement_sigma),
            rho_total=float(rho_total),
            budget_split_mu=None,
        )
    epsilon = math.sqrt(8.0 * (1.0 - mu) * rho_total / rounds)
    measurement_sigma = math.sqrt(rounds / (2.0 * mu * rho_total))
    return BudgetConfig(
        mode="total_zcdp",
        epsilon=float(epsilon),
        measurement_sigma=float(measurement_sigma),
        rho_total=float(rho_total),
        budget_split_mu=float(mu),
    )


def _chi_mean(k: int) -> float:
    if k <= 0:
        return 0.0
    return float(math.sqrt(2.0) * math.exp(math.lgamma((k + 1.0) / 2.0) - math.lgamma(k / 2.0)))


def _top_sum(values: np.ndarray, m: int) -> float:
    if values.size == 0 or m <= 0:
        return 0.0
    keep = min(int(m), int(values.size))
    if keep >= values.size:
        return float(np.sum(values))
    partitioned = np.partition(values, values.size - keep)
    return float(np.sum(partitioned[-keep:]))


def _top_l2(values: np.ndarray, m: int) -> float:
    if values.size == 0 or m <= 0:
        return 0.0
    keep = min(int(m), int(values.size))
    if keep >= values.size:
        selected = values
    else:
        partitioned = np.partition(values, values.size - keep)
        selected = partitioned[-keep:]
    return float(np.linalg.norm(selected, ord=2))


def _noise_adaptive_keep_linear(k: int, sigma: float) -> int:
    if k <= 0:
        return 0
    return max(1, int(math.ceil(float(k) / (1.0 + float(sigma)))))


def _public_propagation_weight(block: AdaptiveBlock) -> float:
    return math.sqrt(min(1.0, max(0.0, float(block.coverage_weight))))


def _l1_l2_mix_top_score(values: np.ndarray, keep: int, alpha: float) -> float:
    top_l1 = _top_sum(values, keep)
    top_l2 = _top_l2(values, keep)
    alpha = min(1.0, max(0.0, float(alpha)))
    return float((1.0 - alpha) * top_l1 + alpha * top_l2)


def _l1_l2_calibrated_top_score(values: np.ndarray, keep: int, alpha: float) -> float:
    top_l1 = _top_sum(values, keep)
    top_l2 = _top_l2(values, keep)
    alpha = min(1.0, max(0.0, float(alpha)))
    scaled_l2_weight = alpha * math.sqrt(max(1, keep))
    l1_weight = 1.0 - alpha
    normalizer = max(l1_weight + scaled_l2_weight, 1.0e-12)
    return float((l1_weight * top_l1 + scaled_l2_weight * top_l2) / normalizer)


def _entropic_excess_risk(values: np.ndarray, tau: float) -> float:
    positive = values[values > 0.0].astype(np.float64, copy=False)
    if positive.size == 0:
        return 0.0
    tau = max(float(tau), 1.0e-12)
    scaled = positive / tau
    max_scaled = float(np.max(scaled))
    if max_scaled < 50.0:
        return float(tau * math.log1p(float(np.sum(np.expm1(scaled)))))
    shifted_sum = float(np.sum(np.exp(scaled - max_scaled)))
    correction = float(max(0, positive.size - 1)) * math.exp(-max_scaled)
    inside = max(shifted_sum - correction, 1.0e-300)
    return float(tau * (max_scaled + math.log(inside)))


def _smooth_positive(value: float, temperature: float) -> float:
    temperature = max(float(temperature), 1.0e-12)
    scaled = float(value) / temperature
    if scaled > 50.0:
        return float(value)
    if scaled < -50.0:
        return float(temperature * math.exp(scaled))
    return float(temperature * math.log1p(math.exp(scaled)))


def _vector_tvd_noise_floor(k: int, sigma: float, delta_l2: float) -> float:
    return 0.5 * math.sqrt(2.0 / math.pi) * float(k) * float(sigma) * max(float(delta_l2), 1.0e-12)


def _vector_distribution_anchor_value(block: AdaptiveBlock, residual: np.ndarray, measurement_sigma: float) -> float:
    if not block.is_vector or int(block.query_indices.size) <= 1:
        return 0.0
    idx = block.query_indices
    tvd_counts = 0.5 * float(np.sum(np.abs(residual[idx].astype(np.float64, copy=False))))
    floor = _vector_tvd_noise_floor(int(idx.size), measurement_sigma, float(block.delta_l2))
    temperature = max(float(measurement_sigma) * max(float(block.delta_l2), 1.0e-12), 1.0e-12)
    return _smooth_positive(tvd_counts - floor, temperature)


def _halfnormal_top_cumsum(k: int) -> np.ndarray:
    cached = _HALFNORMAL_TOP_CUMSUM_CACHE.get(int(k))
    if cached is not None:
        return cached
    if k <= 0:
        result = np.asarray([], dtype=np.float64)
        _HALFNORMAL_TOP_CUMSUM_CACHE[int(k)] = result
        return result
    quantiles = []
    for rank in range(1, k + 1):
        p = (rank - 0.375) / (k + 0.25)
        p = min(max(p, 1.0e-12), 1.0 - 1.0e-12)
        quantiles.append(_NORMAL.inv_cdf((1.0 + p) / 2.0))
    descending = np.sort(np.asarray(quantiles, dtype=np.float64))[::-1]
    expected_total = math.sqrt(2.0 / math.pi) * float(k)
    observed_total = float(np.sum(descending))
    if observed_total > 0.0:
        descending = descending * (expected_total / observed_total)
    result = np.cumsum(descending)
    _HALFNORMAL_TOP_CUMSUM_CACHE[int(k)] = result
    return result


def _halfnormal_top_expected(k: int) -> np.ndarray:
    cumsum = _halfnormal_top_cumsum(k)
    if cumsum.size == 0:
        return cumsum
    return np.diff(np.concatenate([np.asarray([0.0], dtype=np.float64), cumsum]))


def _adaptive_order_excess(abs_err: np.ndarray, sigma: float, normalize: str = "none") -> float:
    k = int(abs_err.size)
    if k <= 0:
        return 0.0
    sorted_abs = np.sort(abs_err.astype(np.float64, copy=False))[::-1]
    signal_cumsum = np.cumsum(sorted_abs)
    noise_floor = float(sigma) * _halfnormal_top_cumsum(k)
    excess = signal_cumsum - noise_floor
    if normalize == "sqrtm":
        excess = excess / np.sqrt(np.arange(1, k + 1, dtype=np.float64))
    elif normalize == "mdl":
        m = np.arange(1, k + 1, dtype=np.float64)
        excess = excess - float(sigma) * np.sqrt(2.0 * m * np.log(np.e * float(k) / m))
    elif normalize != "none":
        raise ValueError(f"Unknown adaptive order normalization {normalize!r}")
    return float(max(0.0, float(np.max(excess))))


def _adaptive_rank_excess(abs_err: np.ndarray, sigma: float, norm: str = "l1") -> float:
    k = int(abs_err.size)
    if k <= 0:
        return 0.0
    sorted_abs = np.sort(abs_err.astype(np.float64, copy=False))[::-1]
    rank_floor = float(sigma) * _halfnormal_top_expected(k)
    excess = np.maximum(sorted_abs - rank_floor, 0.0)
    if norm == "l1":
        return float(np.sum(excess))
    if norm == "l2":
        return float(np.linalg.norm(excess, ord=2))
    raise ValueError(f"Unknown adaptive rank norm {norm!r}")


def _adaptive_spectral_excess(abs_err: np.ndarray, sigma: float, weight: str) -> float:
    k = int(abs_err.size)
    if k <= 0:
        return 0.0
    sorted_abs = np.sort(abs_err.astype(np.float64, copy=False))[::-1]
    rank_floor = float(sigma) * _halfnormal_top_expected(k)
    excess = np.maximum(sorted_abs - rank_floor, 0.0)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    if weight == "sqrt":
        weights = 1.0 / np.sqrt(ranks)
    elif weight == "harmonic":
        weights = 1.0 / ranks
    elif weight == "log":
        weights = 1.0 / np.log2(ranks + 1.0)
    else:
        raise ValueError(f"Unknown adaptive spectral weight {weight!r}")
    return float(np.sum(weights * excess))


def _scope_overlap_matrix(blocks: list[AdaptiveBlock]) -> tuple[np.ndarray, float]:
    scope_sets = [set(block.scope) for block in blocks]
    overlap = np.zeros((len(blocks), len(blocks)), dtype=np.float32)
    for source_idx, source in enumerate(scope_sets):
        if not source:
            continue
        for target_idx, target in enumerate(scope_sets):
            if not target:
                continue
            overlap[source_idx, target_idx] = float(len(source & target)) / float(len(target))
    max_mass = float(np.max(np.sum(overlap, axis=1))) if overlap.size else 1.0
    return overlap, max(max_mass, 1.0e-12)


def _scope_projection_matrix(blocks: list[AdaptiveBlock]) -> tuple[np.ndarray, float]:
    scope_sets = [set(block.scope) for block in blocks]
    projection = np.zeros((len(blocks), len(blocks)), dtype=np.float32)
    for source_idx, source in enumerate(scope_sets):
        if not source:
            continue
        for target_idx, target in enumerate(scope_sets):
            if source_idx == target_idx:
                projection[source_idx, target_idx] = 1.0
            elif blocks[source_idx].is_vector and target and target.issubset(source):
                projection[source_idx, target_idx] = 1.0
    max_mass = float(np.max(np.sum(projection, axis=1))) if projection.size else 1.0
    return projection, max(max_mass, 1.0e-12)


def _term_interval(op: int, value: int, lo: int, hi: int) -> tuple[float, float] | None:
    if op == OP_EQ:
        point = float(value)
        return point, point
    if op == OP_LE:
        return -math.inf, float(value)
    if op == OP_GE:
        return float(value), math.inf
    if op == OP_RANGE:
        return float(lo), float(hi)
    return None


def _query_interval_cells(qcat: QueryCatalogue, qid: int) -> dict[int, tuple[float, float]] | None:
    if int(qcat.linear_num_terms[qid]) > 0:
        return None
    intervals: dict[int, tuple[float, float]] = {}
    for attr, op, value, lo, hi in qcat.query_terms(qid):
        bounds = _term_interval(op, value, lo, hi)
        if bounds is None:
            return None
        prev = intervals.get(attr)
        if prev is None:
            intervals[attr] = bounds
        else:
            merged = max(prev[0], bounds[0]), min(prev[1], bounds[1])
            if merged[0] > merged[1]:
                return None
            intervals[attr] = merged
    return intervals


def _source_partition_cells(
    qcat: QueryCatalogue, block: AdaptiveBlock
) -> tuple[tuple[int, ...], list[dict[int, tuple[float, float]]]] | None:
    if not block.is_vector:
        return None
    cells: list[dict[int, tuple[float, float]]] = []
    source_scope: tuple[int, ...] | None = None
    for qid_raw in block.query_indices.tolist():
        cell = _query_interval_cells(qcat, int(qid_raw))
        if not cell:
            return None
        cell_scope = tuple(sorted(cell))
        if source_scope is None:
            source_scope = cell_scope
        elif cell_scope != source_scope:
            return None
        cells.append(cell)
    if source_scope is None or not cells:
        return None
    return source_scope, cells


def _cell_query_relation(
    qcat: QueryCatalogue,
    qid: int,
    cell: dict[int, tuple[float, float]],
) -> int:
    # 1: cell is inside query, -1: cell is outside query, 0: partial/unknown.
    if int(qcat.linear_num_terms[qid]) > 0:
        return 0
    all_inside = True
    for attr, op, value, lo, hi in qcat.query_terms(qid):
        cell_bounds = cell.get(attr)
        target_bounds = _term_interval(op, value, lo, hi)
        if cell_bounds is None or target_bounds is None:
            return 0
        cell_lo, cell_hi = cell_bounds
        target_lo, target_hi = target_bounds
        if cell_hi < target_lo or cell_lo > target_hi:
            return -1
        if cell_lo < target_lo or cell_hi > target_hi:
            all_inside = False
    return 1 if all_inside else 0


def _query_derivable_from_partition(
    qcat: QueryCatalogue,
    qid: int,
    source_scope: tuple[int, ...],
    cells: list[dict[int, tuple[float, float]]],
) -> bool:
    query_scope = set(_block_scope(qcat, np.asarray([qid], dtype=np.int32)))
    if not query_scope or not query_scope.issubset(set(source_scope)):
        return False
    for cell in cells:
        if _cell_query_relation(qcat, qid, cell) == 0:
            return False
    return True


def _query_projection_matrix(qcat: QueryCatalogue, blocks: list[AdaptiveBlock]) -> tuple[np.ndarray, float]:
    projection = np.zeros((len(blocks), len(blocks)), dtype=np.float32)
    block_query_indices = [block.query_indices.astype(np.int32, copy=False) for block in blocks]
    query_scopes = [
        set(_block_scope(qcat, np.asarray([qid], dtype=np.int32)))
        for qid in range(int(qcat.m))
    ]
    for source_idx, source_block in enumerate(blocks):
        projection[source_idx, source_idx] = 1.0
        parsed = _source_partition_cells(qcat, source_block)
        if parsed is None:
            continue
        source_scope, cells = parsed
        source_scope_set = set(source_scope)
        derivable = np.zeros(int(qcat.m), dtype=bool)
        for qid in range(int(qcat.m)):
            if not query_scopes[qid] or not query_scopes[qid].issubset(source_scope_set):
                continue
            derivable[qid] = _query_derivable_from_partition(qcat, qid, source_scope, cells)
        for target_idx, target_idx_array in enumerate(block_query_indices):
            if source_idx == target_idx or target_idx_array.size == 0:
                continue
            weight = float(np.mean(derivable[target_idx_array]))
            if weight > 0.0:
                projection[source_idx, target_idx] = weight
    max_mass = float(np.max(np.sum(projection, axis=1))) if projection.size else 1.0
    return projection, max(max_mass, 1.0e-12)


def _query_repair_plans(qcat: QueryCatalogue, blocks: list[AdaptiveBlock]) -> list[QueryRepairPlan | None]:
    query_scopes = [
        set(_block_scope(qcat, np.asarray([qid], dtype=np.int32)))
        for qid in range(int(qcat.m))
    ]
    plans: list[QueryRepairPlan | None] = []
    for block in blocks:
        if not block.is_vector:
            plans.append(
                QueryRepairPlan(
                    target_qids=block.query_indices.astype(np.int32, copy=True),
                    cell_indices=(np.asarray([0], dtype=np.int32),),
                    noise_counts=np.asarray([1.0], dtype=np.float64),
                )
            )
            continue
        parsed = _source_partition_cells(qcat, block)
        if parsed is None:
            plans.append(None)
            continue
        source_scope, cells = parsed
        source_scope_set = set(source_scope)
        target_qids: list[int] = []
        cell_indices: list[np.ndarray] = []
        noise_counts: list[float] = []
        for qid in range(int(qcat.m)):
            if not query_scopes[qid] or not query_scopes[qid].issubset(source_scope_set):
                continue
            inside_cells: list[int] = []
            exact = True
            for cell_idx, cell in enumerate(cells):
                relation = _cell_query_relation(qcat, qid, cell)
                if relation == 0:
                    exact = False
                    break
                if relation == 1:
                    inside_cells.append(cell_idx)
            if exact and inside_cells:
                target_qids.append(qid)
                selected = np.asarray(inside_cells, dtype=np.int32)
                cell_indices.append(selected)
                noise_counts.append(float(selected.size))
        if not target_qids:
            plans.append(None)
            continue
        plans.append(
            QueryRepairPlan(
                target_qids=np.asarray(target_qids, dtype=np.int32),
                cell_indices=tuple(cell_indices),
                noise_counts=np.asarray(noise_counts, dtype=np.float64),
            )
        )
    return plans


def _is_value_of_information_scheme(scheme: str) -> bool:
    return _scheme_base(scheme).startswith("voi_")


def _is_operator_value_of_information_scheme(scheme: str) -> bool:
    return _scheme_base(scheme) in {
        "voi_absop_qproject",
        "voi_sqop_qproject",
        "voi_absdiff_qproject",
        "voi_sqdiff_qproject",
        "voi_rankop_sqrt_qproject",
        "voi_rankop_harmonic_qproject",
        "voi_rankop_log_qproject",
        "voi_atomop_harmonic_qproject",
        "voi_atomlev_harmonic_qproject",
        "voi_atomloclev_harmonic_qproject",
        "voi_genactive_atomlev_harmonic_qproject",
        "voi_genexposure_atomlev_harmonic_qproject",
        "voi_graphctx_atomlev_harmonic_qproject",
        "voi_condprop_atomlev_harmonic_qproject",
        "voi_condctx_atomlev_harmonic_qproject",
        "voi_condbridge_atomlev_harmonic_qproject",
        "voi_condnorm_atomlev_harmonic_qproject",
        "voi_condbridgenorm_atomlev_harmonic_qproject",
        "voi_precinnov_atomlev_harmonic_qproject",
        "voi_distanchor_atomlev_harmonic_qproject",
        "voi_atomtaillev_harmonic_qproject",
        "voi_atomenvlev_harmonic_qproject",
        "voi_dualop_harmonic_qproject",
        "voi_nashop_harmonic_qproject",
        "voi_l4query_qproject",
        "voi_l4atom_qproject",
        "voi_l4dual_qproject",
        "voi_relmaxop_harmonic_qproject",
        "voi_relmaxscaled_harmonic_qproject",
        "voi_relmaxposscaled_harmonic_qproject",
        "voi_relmaxposeff_harmonic_qproject",
        "voi_relmaxposnorm_harmonic_qproject",
        "voi_dpstateposnorm_harmonic_qproject",
        "voi_dpstatecovsplit_harmonic_qproject",
        "voi_dpstatesqrtcovsplit_harmonic_qproject",
        "voi_dpstatesplittail_harmonic_qproject",
        "voi_dpstatecovtail_harmonic_qproject",
        "voi_dpstatebreadthtail_harmonic_qproject",
        "voi_dpstateriskmaxmrr_harmonic_qproject",
        "voi_dpstateriskenv025mrr_harmonic_qproject",
        "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
        "voi_sageordergain_harmonic_qproject",
        "voi_sagemrr_harmonic_qproject",
        "voi_sageroutecmrr_harmonic_qproject",
        "voi_sagecovmrr_harmonic_qproject",
        "voi_sagesqcovmrr_harmonic_qproject",
        "voi_sagesoftcovmrr_harmonic_qproject",
        "voi_sagebreadthmrr_harmonic_qproject",
        "voi_sageblockmrr_harmonic_qproject",
        "voi_sagelocalblockmrr_harmonic_qproject",
        "voi_sagelocalvalidtailmrr_harmonic_qproject",
        "voi_sagelocaltailcap65mrr_harmonic_qproject",
        "voi_sagelocaltailcap75mrr_harmonic_qproject",
        "voi_sagelocaltailcap90mrr_harmonic_qproject",
        "voi_sagelocaltailcap95mrr_harmonic_qproject",
        "voi_sagelocaltailblend02mrr_harmonic_qproject",
        "voi_sagelocaltailblend03mrr_harmonic_qproject",
        "voi_sagelocaltailblend04mrr_harmonic_qproject",
        "voi_sagelocaltailblend05mrr_harmonic_qproject",
        "voi_sagelocaltailblend10mrr_harmonic_qproject",
        "voi_sagelocalagree2of3mrr_harmonic_qproject",
        "voi_sagelocaltailflow010mrr_harmonic_qproject",
        "voi_sagelocaltailflow020mrr_harmonic_qproject",
        "voi_sagelocaltailflow040mrr_harmonic_qproject",
        "voi_sagelocaltailflowmax025mrr_harmonic_qproject",
        "voi_sagelocaltailflowmax050mrr_harmonic_qproject",
        "voi_sagelocaltailflowmax100mrr_harmonic_qproject",
        "voi_sagelocaltailflowenv025mrr_harmonic_qproject",
        "voi_sagelocaltailflowenv050mrr_harmonic_qproject",
        "voi_sagelocaltailflowenv075mrr_harmonic_qproject",
        "voi_sagelocaltailblendfloor98mrr_harmonic_qproject",
        "voi_sagelocaltailblendfloor99mrr_harmonic_qproject",
        "voi_sagelocaltailblendfloor995mrr_harmonic_qproject",
        "voi_sagelocaltailbonus04mrr_harmonic_qproject",
        "voi_sagelocaltailbonus05mrr_harmonic_qproject",
        "voi_sagelocaltailbonus10mrr_harmonic_qproject",
        "voi_sagelocaltailbonus20mrr_harmonic_qproject",
        "voi_sagelocaltailbonus50mrr_harmonic_qproject",
        "voi_sagelocaltailhalf05mrr_harmonic_qproject",
        "voi_sagelocaltaillastq05mrr_harmonic_qproject",
        "voi_sagelocaltailramp05mrr_harmonic_qproject",
        "voi_sagedpblend10mrr_harmonic_qproject",
        "voi_sagedpblend25mrr_harmonic_qproject",
        "voi_sagedpblend50mrr_harmonic_qproject",
        "voi_sagedpblenddecay5010mrr_harmonic_qproject",
        "voi_sagedpblendramp1050mrr_harmonic_qproject",
        "voi_sagedpblendmax1050mrr_harmonic_qproject",
        "voi_sagedpblendstategatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstateqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstatevgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstategate4rtmrr_harmonic_qproject",
        "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpconfirmfloor25mrr_harmonic_qproject",
        "voi_sagedpconfirmfloor50mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor05mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor10mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor25mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor50mrr_harmonic_qproject",
        "voi_sagedpstageholdfloor05mrr_harmonic_qproject",
        "voi_sagedpstageholdfloor10mrr_harmonic_qproject",
        "voi_sagenrlocalmrr_harmonic_qproject",
        "voi_sagecellblockmrr_harmonic_qproject",
        "voi_sagegraphcvarmrr_harmonic_qproject",
        "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
        "voi_sageenv025mrr_harmonic_qproject",
        "voi_sageenv05mrr_harmonic_qproject",
        "voi_sageenv1mrr_harmonic_qproject",
        "voi_sagecvar4mrr_harmonic_qproject",
        "voi_sagecvarsqrtmrr_harmonic_qproject",
        "voi_sagevcvarsqrtmrr_harmonic_qproject",
        "voi_sagevtopsqrtmrr_harmonic_qproject",
        "voi_sagestudcvarmrr_harmonic_qproject",
        "voi_sagespectralmrr_harmonic_qproject",
        "voi_sageholder15mrr_harmonic_qproject",
        "voi_sageholder2mrr_harmonic_qproject",
        "voi_sagenestedcvarmrr_harmonic_qproject",
        "voi_sagehiercvarmrr_harmonic_qproject",
        "voi_sagerepaircap10mrr_harmonic_qproject",
        "voi_sagerepaircap25mrr_harmonic_qproject",
        "voi_sagerepaircap50mrr_harmonic_qproject",
        "voi_sagebalcvarmrr_harmonic_qproject",
        "voi_sagebalvcvarmrr_harmonic_qproject",
        "voi_sagebalvtopsqrtmrr_harmonic_qproject",
        "voi_sageflowmrr_harmonic_qproject",
        "voi_sagegraphcapmrr_harmonic_qproject",
        "voi_sagegraphcapmassmrr_harmonic_qproject",
        "voi_sagegraphcapfullmrr_harmonic_qproject",
        "voi_sageqvcapmrr_harmonic_qproject",
        "voi_sageqvcapenvmrr_harmonic_qproject",
        "voi_sageqvcapcurrmrr_harmonic_qproject",
        "voi_sageqvcapresmrr_harmonic_qproject",
        "voi_sageqvcapbandmrr_harmonic_qproject",
        "voi_sageqvcalmrr_harmonic_qproject",
        "voi_sageqvcalcurrmrr_harmonic_qproject",
        "voi_sageqvcalbandmrr_harmonic_qproject",
        "voi_sageqvcaldromrr_harmonic_qproject",
        "voi_sageqvcaltopdromrr_harmonic_qproject",
        "voi_sageqvcalboxdromrr_harmonic_qproject",
        "voi_sageqvcaltopboxdromrr_harmonic_qproject",
        "voi_sageqvcalstaildromrr_harmonic_qproject",
        "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
        "voi_sageqvcaltailcapdromrr_harmonic_qproject",
        "voi_sageqvcalatomcapdromrr_harmonic_qproject",
        "voi_sageqvcalpairdromrr_harmonic_qproject",
        "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
        "voi_sageqvcalmaxdromrr_harmonic_qproject",
        "voi_sageqvcalfrontdromrr_harmonic_qproject",
        "voi_sageqvcaljointdromrr_harmonic_qproject",
        "voi_sageqvcalflowguarddromrr_harmonic_qproject",
        "voi_sagevalidfullmaxmrr_harmonic_qproject",
        "voi_sagevalidfullcapmrr_harmonic_qproject",
        "voi_sagerrcvalidfullmrr_harmonic_qproject",
        "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
        "voi_sagerrcvalidtopkmrr_harmonic_qproject",
        "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
        "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
        "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
        "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
        "voi_sagerrcnestedvalidmrr_harmonic_qproject",
        "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
        "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
        "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
        "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
        "voi_sagecapdromrr_harmonic_qproject",
        "voi_sagemetricavgmrr_harmonic_qproject",
        "voi_sagemetricenvmrr_harmonic_qproject",
        "voi_sagefloorcapmrr_harmonic_qproject",
        "voi_sageparetomrr_harmonic_qproject",
        "voi_sageharmmrr_harmonic_qproject",
        "voi_relmaxsqrtlevscaled_harmonic_qproject",
        "voi_relmaxlevscaled_harmonic_qproject",
        "voi_mixlp2_qproject",
        "voi_mixlp4_qproject",
        "voi_mixlp8_qproject",
        "voi_worstop_qproject",
    }


def _expected_abs_normal(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(std.astype(np.float64, copy=False), 1.0e-12)
    abs_mean = np.abs(mean.astype(np.float64, copy=False))
    z = abs_mean / std
    return std * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * z * z) + abs_mean * erf(z / math.sqrt(2.0))


def _rank_spectral_weights(residual: np.ndarray, kind: str) -> np.ndarray:
    abs_residual = np.abs(residual.astype(np.float64, copy=False))
    order = np.argsort(abs_residual)[::-1]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    if kind == "sqrt":
        return 1.0 / np.sqrt(ranks)
    if kind == "harmonic":
        return 1.0 / ranks
    if kind == "log":
        return 1.0 / np.log2(ranks + 1.0)
    raise ValueError(f"Unknown rank spectral kind {kind!r}")


def _public_harmonic_weights(k: int) -> np.ndarray:
    ranks = np.arange(1, int(k) + 1, dtype=np.float64)
    return 1.0 / np.maximum(ranks, 1.0)


def _atom_spectral_weights(
    residual: np.ndarray, blocks: list[AdaptiveBlock], kind: str
) -> tuple[np.ndarray, list[int], np.ndarray]:
    query_values = np.abs(residual.astype(np.float64, copy=False))
    vector_block_ids = [
        block_id
        for block_id, block in enumerate(blocks)
        if block.is_vector and int(block.query_indices.size) > 1
    ]
    vector_values = np.asarray(
        [0.5 * float(np.sum(query_values[blocks[block_id].query_indices])) for block_id in vector_block_ids],
        dtype=np.float64,
    )
    atom_values = np.concatenate([query_values, vector_values])
    order = np.argsort(atom_values)[::-1]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    if kind == "harmonic":
        weights = 1.0 / ranks
    elif kind == "sqrt":
        weights = 1.0 / np.sqrt(ranks)
    elif kind == "log":
        weights = 1.0 / np.log2(ranks + 1.0)
    else:
        raise ValueError(f"Unknown atom spectral kind {kind!r}")
    return weights[: residual.size], vector_block_ids, weights[residual.size :]


def _atom_tail_spectral_weights(
    residual: np.ndarray, blocks: list[AdaptiveBlock], kind: str
) -> tuple[np.ndarray, list[int], np.ndarray]:
    query_values = np.abs(residual.astype(np.float64, copy=False))
    vector_block_ids = [
        block_id
        for block_id, block in enumerate(blocks)
        if block.is_vector and int(block.query_indices.size) > 1
    ]
    vector_values = []
    for block_id in vector_block_ids:
        block_abs = np.sort(query_values[blocks[block_id].query_indices])[::-1]
        ranks = np.arange(1, block_abs.size + 1, dtype=np.float64)
        vector_values.append(float(np.sum(block_abs / ranks)))
    atom_values = np.concatenate([query_values, np.asarray(vector_values, dtype=np.float64)])
    order = np.argsort(atom_values)[::-1]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    if kind == "harmonic":
        weights = 1.0 / ranks
    elif kind == "sqrt":
        weights = 1.0 / np.sqrt(ranks)
    elif kind == "log":
        weights = 1.0 / np.log2(ranks + 1.0)
    else:
        raise ValueError(f"Unknown atom spectral kind {kind!r}")
    return weights[: residual.size], vector_block_ids, weights[residual.size :]


def _vector_local_spectral_weights(
    residual: np.ndarray, blocks: list[AdaptiveBlock], vector_block_ids: list[int], kind: str
) -> dict[int, np.ndarray]:
    abs_residual = np.abs(residual.astype(np.float64, copy=False))
    result: dict[int, np.ndarray] = {}
    for block_id in vector_block_ids:
        idx = blocks[block_id].query_indices
        order = np.argsort(abs_residual[idx])[::-1]
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
        if kind == "harmonic":
            result[block_id] = 1.0 / ranks
        elif kind == "sqrt":
            result[block_id] = 1.0 / np.sqrt(ranks)
        elif kind == "log":
            result[block_id] = 1.0 / np.log2(ranks + 1.0)
        else:
            raise ValueError(f"Unknown vector spectral kind {kind!r}")
    return result


def _atom_envelope_spectral_weights(
    residual: np.ndarray, blocks: list[AdaptiveBlock], kind: str
) -> tuple[np.ndarray, list[int], np.ndarray, dict[int, np.ndarray]]:
    query_values = np.abs(residual.astype(np.float64, copy=False))
    vector_block_ids = [
        block_id
        for block_id, block in enumerate(blocks)
        if block.is_vector and int(block.query_indices.size) > 1
    ]
    vector_values: list[float] = []
    vector_local_weights: dict[int, np.ndarray] = {}
    for block_id in vector_block_ids:
        idx = blocks[block_id].query_indices
        block_abs = query_values[idx]
        order = np.argsort(block_abs)[::-1]
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
        if kind == "harmonic":
            tail_weights = 1.0 / ranks
        elif kind == "sqrt":
            tail_weights = 1.0 / np.sqrt(ranks)
        elif kind == "log":
            tail_weights = 1.0 / np.log2(ranks + 1.0)
        else:
            raise ValueError(f"Unknown vector spectral kind {kind!r}")
        tvd_value = 0.5 * float(np.sum(block_abs))
        tail_value = float(np.sum(tail_weights * block_abs))
        if tail_value > tvd_value:
            vector_values.append(tail_value)
            vector_local_weights[block_id] = tail_weights
        else:
            vector_values.append(tvd_value)
            vector_local_weights[block_id] = np.full(idx.size, 0.5, dtype=np.float64)
    atom_values = np.concatenate([query_values, np.asarray(vector_values, dtype=np.float64)])
    order = np.argsort(atom_values)[::-1]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    if kind == "harmonic":
        weights = 1.0 / ranks
    elif kind == "sqrt":
        weights = 1.0 / np.sqrt(ranks)
    elif kind == "log":
        weights = 1.0 / np.log2(ranks + 1.0)
    else:
        raise ValueError(f"Unknown atom spectral kind {kind!r}")
    return weights[: residual.size], vector_block_ids, weights[residual.size :], vector_local_weights


def _vector_spectral_weights(
    residual: np.ndarray, blocks: list[AdaptiveBlock], kind: str
) -> tuple[list[int], np.ndarray]:
    abs_residual = np.abs(residual.astype(np.float64, copy=False))
    vector_block_ids = [
        block_id
        for block_id, block in enumerate(blocks)
        if block.is_vector and int(block.query_indices.size) > 1
    ]
    vector_values = np.asarray(
        [0.5 * float(np.sum(abs_residual[blocks[block_id].query_indices])) for block_id in vector_block_ids],
        dtype=np.float64,
    )
    order = np.argsort(vector_values)[::-1]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    if kind == "harmonic":
        weights = 1.0 / ranks
    elif kind == "sqrt":
        weights = 1.0 / np.sqrt(ranks)
    elif kind == "log":
        weights = 1.0 / np.log2(ranks + 1.0)
    else:
        raise ValueError(f"Unknown vector spectral kind {kind!r}")
    return vector_block_ids, weights


def _vector_public_harmonic_weights(blocks: list[AdaptiveBlock]) -> tuple[list[int], np.ndarray]:
    vector_block_ids = [
        block_id
        for block_id, block in enumerate(blocks)
        if block.is_vector and int(block.query_indices.size) > 1
    ]
    return vector_block_ids, _public_harmonic_weights(len(vector_block_ids))


def _vector_local_harmonic_weights(
    residual: np.ndarray | None,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> list[np.ndarray]:
    local_weights: list[np.ndarray] = []
    for block_id in vector_block_ids:
        idx = blocks[block_id].query_indices
        if residual is None:
            local_weights.append(_public_harmonic_weights(int(idx.size)))
            continue
        values = np.abs(residual[idx].astype(np.float64, copy=False))
        order = np.argsort(values)[::-1]
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
        local_weights.append(1.0 / np.maximum(ranks, 1.0))
    return local_weights


def _query_vector_membership_split(
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    num_queries: int,
) -> np.ndarray:
    counts = np.zeros(int(num_queries), dtype=np.float64)
    for block_id in vector_block_ids:
        np.add.at(counts, blocks[block_id].query_indices, 1.0)
    return 1.0 / np.maximum(counts, 1.0)


def _valid_tail_propagation_weight(scheme: str, repaired: float, block_size: float) -> float:
    block_size = max(float(block_size), 1.0)
    coverage = max(0.0, min(1.0, float(repaired) / block_size))
    base = _scheme_base(scheme)
    if base == "voi_dpstatecovtail_harmonic_qproject":
        return coverage
    if base == "voi_dpstatebreadthtail_harmonic_qproject":
        if block_size <= 1.0:
            return coverage
        return max(0.0, min(1.0, (float(repaired) - 1.0) / (block_size - 1.0)))
    return 1.0


def _valid_local_tail_propagation_weight(repaired: float, block_size: float) -> float:
    block_size = max(float(block_size), 1.0)
    repaired = max(0.0, float(repaired))
    if block_size <= 1.0:
        return max(0.0, min(1.0, repaired / block_size))
    tail_width = max(1.0, math.sqrt(block_size))
    if tail_width <= 1.0:
        return max(0.0, min(1.0, repaired / block_size))
    return max(0.0, min(1.0, (repaired - 1.0) / (tail_width - 1.0)))


def _local_tail_capacity_cap(scheme: str) -> float:
    base = _scheme_base(scheme)
    if base == "voi_sagelocaltailcap65mrr_harmonic_qproject":
        return 0.65
    if base == "voi_sagelocaltailcap75mrr_harmonic_qproject":
        return 0.75
    if base == "voi_sagelocaltailcap90mrr_harmonic_qproject":
        return 0.90
    if base == "voi_sagelocaltailcap95mrr_harmonic_qproject":
        return 0.95
    raise ValueError(f"Unknown local-tail capacity scheme {scheme!r}")


def _mixed_lp_order(scheme: str) -> float:
    base = _scheme_base(scheme)
    if base == "voi_mixlp2_qproject":
        return 2.0
    if base == "voi_mixlp4_qproject":
        return 4.0
    if base == "voi_mixlp8_qproject":
        return 8.0
    raise ValueError(f"Unknown mixed Lp scheme {scheme!r}")


def _mixed_lp_risk_weights(
    residual: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    before_vector_tvd: np.ndarray,
    p_order: float,
) -> tuple[np.ndarray, np.ndarray]:
    query_values = np.abs(residual.astype(np.float64, copy=False))
    vector_values = before_vector_tvd.astype(np.float64, copy=False)
    p = max(float(p_order), 1.0 + 1.0e-12)
    query_count = max(1, int(query_values.size))
    vector_count = max(1, int(vector_values.size))
    risk_power = float(np.sum(query_values**p) / query_count)
    risk_power += float(np.sum(vector_values**p) / vector_count)
    if risk_power <= 0.0:
        return np.zeros_like(query_values, dtype=np.float64), np.zeros_like(vector_values, dtype=np.float64)
    normalizer = risk_power ** ((p - 1.0) / p)
    normalizer = max(float(normalizer), 1.0e-12)
    query_weights = (query_values ** (p - 1.0)) / (float(query_count) * normalizer)
    vector_weights = (vector_values ** (p - 1.0)) / (float(vector_count) * normalizer)
    return query_weights.astype(np.float64, copy=False), vector_weights.astype(np.float64, copy=False)


def _query_repair_leverage(repair_plans: list[QueryRepairPlan | None], num_queries: int) -> np.ndarray:
    degrees = np.zeros(int(num_queries), dtype=np.float64)
    for plan in repair_plans:
        if plan is None or plan.target_qids.size == 0:
            continue
        np.add.at(degrees, plan.target_qids, 1.0)
    degrees = np.maximum(degrees, 1.0)
    return 1.0 / degrees


def _harmonic_number(k: int) -> float:
    if k <= 0:
        return 0.0
    ranks = np.arange(1, int(k) + 1, dtype=np.float64)
    return float(np.sum(1.0 / ranks))


def _harmonic_ordered_sum(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    ordered = np.sort(values.astype(np.float64, copy=False))[::-1]
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(np.sum(ordered / ranks))


def _weighted_harmonic_ordered_sum(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.size != weights.size:
        raise ValueError("values and weights must have the same size")
    effective = np.maximum(values.astype(np.float64, copy=False), 0.0) * np.maximum(
        weights.astype(np.float64, copy=False),
        0.0,
    )
    return _harmonic_ordered_sum(effective)


def _source_cell_propagation_weight(
    plan: QueryRepairPlan,
    target_subset_qids: np.ndarray,
    source_size: int,
    target_block_size: int,
) -> float:
    if target_subset_qids.size == 0 or source_size <= 0 or target_block_size <= 0:
        return 0.0
    positions = np.flatnonzero(np.isin(plan.target_qids, target_subset_qids, assume_unique=False))
    if positions.size == 0:
        return 0.0
    used_cells = np.unique(np.concatenate([plan.cell_indices[int(pos)] for pos in positions]))
    source_coverage = float(used_cells.size) / max(float(source_size), 1.0)
    target_coverage = float(target_subset_qids.size) / max(float(target_block_size), 1.0)
    return math.sqrt(max(0.0, min(1.0, source_coverage)) * max(0.0, min(1.0, target_coverage)))


def _risk_envelope_tau(scheme: str) -> float:
    base = _scheme_base(scheme)
    if base == "voi_sageenv025mrr_harmonic_qproject":
        return 0.25
    if base == "voi_sageenv05mrr_harmonic_qproject":
        return 0.5
    if base == "voi_sageenv1mrr_harmonic_qproject":
        return 1.0
    if base in {
        "voi_sagecvar4mrr_harmonic_qproject",
        "voi_sagecvarsqrtmrr_harmonic_qproject",
        "voi_sagevcvarsqrtmrr_harmonic_qproject",
        "voi_sagevtopsqrtmrr_harmonic_qproject",
        "voi_sagestudcvarmrr_harmonic_qproject",
    }:
        return 0.25
    raise ValueError(f"Unknown risk envelope scheme {scheme!r}")


def _cvar_tail_k(scheme: str, size: int) -> int:
    size = int(size)
    if size <= 0:
        return 0
    base = _scheme_base(scheme)
    if base == "voi_sagecvar4mrr_harmonic_qproject":
        return min(4, size)
    if base == "voi_sagecvarsqrtmrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base == "voi_sagevcvarsqrtmrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base in {
        "voi_sagegraphcvarmrr_harmonic_qproject",
        "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
    }:
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base == "voi_sagevtopsqrtmrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base == "voi_sagerrcvalidtopkmrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base == "voi_sagerrcvalidtopkgainmrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base == "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base == "voi_sagecapdromrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    if base == "voi_sagestudcvarmrr_harmonic_qproject":
        return min(size, max(1, int(math.ceil(math.sqrt(size)))))
    raise ValueError(f"Unknown CVaR risk scheme {scheme!r}")


def _topk_mean(values: np.ndarray, k: int) -> float:
    values = values.astype(np.float64, copy=False)
    if values.size == 0 or k <= 0:
        return 0.0
    k = min(int(k), int(values.size))
    if k == values.size:
        return float(np.mean(values))
    topk = np.partition(values, values.size - k)[values.size - k :]
    return float(np.mean(topk))


def _head_topk_mean(values: np.ndarray, k: int) -> float:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    return 0.5 * (float(np.max(values)) + _topk_mean(values, k))


def _harmonic_ordered_mean(values: np.ndarray) -> float:
    values = np.maximum(values.astype(np.float64, copy=False), 0.0)
    if values.size == 0:
        return 0.0
    return _harmonic_ordered_sum(values) / max(_harmonic_number(values.size), 1.0e-12)


def _lorenz_head_broad_mean(values: np.ndarray) -> float:
    values = np.maximum(values.astype(np.float64, copy=False), 0.0)
    if values.size == 0:
        return 0.0
    ordered = np.sort(values)[::-1]
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    cumulative = np.cumsum(ordered)
    # Max over Ky-Fan top-k sums with sqrt(k) scaling: k=1 preserves the head,
    # while larger k wins only when the repair gain is coherently broad.
    return float(np.max(cumulative / np.sqrt(ranks)) / math.sqrt(float(values.size)))


def _harmonic_pair_mean(left: float, right: float) -> float:
    left = max(float(left), 0.0)
    right = max(float(right), 0.0)
    total = left + right
    if total <= 1.0e-12:
        return 0.0
    return 2.0 * left * right / total


def _softmax_envelope(values: np.ndarray, tau: float) -> float:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    tau = max(float(tau), 1.0e-12)
    top = float(np.max(values))
    return float(top + tau * np.log(np.sum(np.exp((values - top) / tau))))


def _normalized_risk_channels(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    vector_sensitivity: float,
) -> np.ndarray:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_ordered = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    query_tail = float(np.max(abs_values)) if abs_values.size else 0.0
    vector_tvd_values = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_ordered = _harmonic_ordered_sum(vector_tvd_values) / max(float(vector_sensitivity), 1.0e-12)
    vector_tail = float(np.max(vector_tvd_values)) / 0.5 if vector_tvd_values.size else 0.0
    vector_block_ordered_values = np.asarray(
        [
            0.5 * _harmonic_ordered_sum(abs_values[blocks[block_id].query_indices])
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_block_tail = (
        float(np.max(vector_block_ordered_values)) / 0.5 if vector_block_ordered_values.size else 0.0
    )
    return np.asarray(
        [
            query_ordered,
            query_tail,
            vector_ordered,
            vector_tail,
            vector_block_tail,
        ],
        dtype=np.float64,
    )


def _normalized_cvar_risk_channels(
    scheme: str,
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    vector_sensitivity: float,
) -> np.ndarray:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_ordered = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    query_cvar = _topk_mean(abs_values, _cvar_tail_k(scheme, abs_values.size))
    vector_tvd_values = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_ordered = _harmonic_ordered_sum(vector_tvd_values) / max(float(vector_sensitivity), 1.0e-12)
    vector_cvar = (
        _topk_mean(vector_tvd_values, _cvar_tail_k(scheme, vector_tvd_values.size)) / 0.5
        if vector_tvd_values.size
        else 0.0
    )
    vector_block_ordered_values = np.asarray(
        [
            0.5 * _harmonic_ordered_sum(abs_values[blocks[block_id].query_indices])
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_block_cvar = (
        _topk_mean(vector_block_ordered_values, _cvar_tail_k(scheme, vector_block_ordered_values.size)) / 0.5
        if vector_block_ordered_values.size
        else 0.0
    )
    return np.asarray(
        [
            query_ordered,
            query_cvar,
            vector_ordered,
            vector_cvar,
            vector_block_cvar,
        ],
        dtype=np.float64,
    )


def _normalized_vector_cvar_risk_channels(
    scheme: str,
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    vector_sensitivity: float,
) -> np.ndarray:
    abs_values = abs_values.astype(np.float64, copy=False)
    vector_tvd_values = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_ordered = _harmonic_ordered_sum(vector_tvd_values) / max(float(vector_sensitivity), 1.0e-12)
    vector_cvar = (
        _topk_mean(vector_tvd_values, _cvar_tail_k(scheme, vector_tvd_values.size)) / 0.5
        if vector_tvd_values.size
        else 0.0
    )
    vector_block_ordered_values = np.asarray(
        [
            0.5 * _harmonic_ordered_sum(abs_values[blocks[block_id].query_indices])
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_block_cvar = (
        _topk_mean(vector_block_ordered_values, _cvar_tail_k(scheme, vector_block_ordered_values.size)) / 0.5
        if vector_block_ordered_values.size
        else 0.0
    )
    return np.asarray(
        [
            vector_ordered,
            vector_cvar,
            vector_block_cvar,
        ],
        dtype=np.float64,
    )


def _normalized_studentized_cvar_risk_channels(
    scheme: str,
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    measurement_sigma: float,
) -> np.ndarray:
    abs_values = abs_values.astype(np.float64, copy=False)
    sigma = max(float(measurement_sigma), 1.0e-12)
    scale = max(sigma, 1.0)
    query_floor = math.sqrt(2.0 / math.pi) * sigma
    query_z = np.maximum(abs_values - query_floor, 0.0) / scale
    query_ordered = _harmonic_ordered_sum(query_z) / max(_harmonic_number(query_z.size), 1.0e-12)
    query_cvar = _topk_mean(query_z, _cvar_tail_k(scheme, query_z.size))

    vector_z_values: list[float] = []
    vector_block_z_values: list[float] = []
    for block_id in vector_block_ids:
        block = blocks[block_id]
        idx = block.query_indices
        k = max(int(idx.size), 1)
        tvd_value = 0.5 * float(np.sum(abs_values[idx]))
        tvd_floor = _vector_tvd_noise_floor(k, sigma, float(block.delta_l2))
        tvd_scale = max(0.5 * scale * max(float(block.delta_l2), 1.0e-12) * math.sqrt(float(k)), 1.0e-12)
        vector_z_values.append(max(0.0, tvd_value - tvd_floor) / tvd_scale)

        local_weights = 1.0 / np.arange(1, k + 1, dtype=np.float64)
        block_value = 0.5 * _harmonic_ordered_sum(abs_values[idx])
        block_floor = 0.5 * sigma * max(float(block.delta_l2), 1.0e-12) * float(np.sum(local_weights)) * math.sqrt(
            2.0 / math.pi
        )
        block_scale = max(
            0.5 * scale * max(float(block.delta_l2), 1.0e-12) * float(np.linalg.norm(local_weights, ord=2)),
            1.0e-12,
        )
        vector_block_z_values.append(max(0.0, block_value - block_floor) / block_scale)

    vector_z = np.asarray(vector_z_values, dtype=np.float64)
    vector_block_z = np.asarray(vector_block_z_values, dtype=np.float64)
    vector_ordered = _harmonic_ordered_sum(vector_z) / max(_harmonic_number(vector_z.size), 1.0e-12)
    vector_cvar = _topk_mean(vector_z, _cvar_tail_k(scheme, vector_z.size)) if vector_z.size else 0.0
    vector_block_cvar = _topk_mean(vector_block_z, _cvar_tail_k(scheme, vector_block_z.size)) if vector_block_z.size else 0.0
    return np.asarray(
        [
            query_ordered,
            query_cvar,
            vector_ordered,
            vector_cvar,
            vector_block_cvar,
        ],
        dtype=np.float64,
    )


def _normalized_metric_risk_channels(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> np.ndarray:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_mean = float(np.mean(abs_values)) if abs_values.size else 0.0
    query_rmse = (
        float(np.linalg.norm(abs_values, ord=2) / math.sqrt(float(abs_values.size)))
        if abs_values.size
        else 0.0
    )
    query_tail = float(np.max(abs_values)) if abs_values.size else 0.0
    vector_tvd_values = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_mean = float(np.mean(vector_tvd_values)) if vector_tvd_values.size else 0.0
    vector_tail = float(np.max(vector_tvd_values)) if vector_tvd_values.size else 0.0
    return np.asarray(
        [
            query_mean,
            query_rmse,
            query_tail,
            vector_mean,
            vector_tail,
        ],
        dtype=np.float64,
    )


def _normalized_spectral_atom_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    vector_sensitivity: float,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_atoms = abs_values / max(float(query_sensitivity), 1.0e-12)
    vector_atoms = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices]))
            / max(float(vector_sensitivity), 1.0e-12)
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    atoms = np.concatenate([query_atoms, vector_atoms]) if vector_atoms.size else query_atoms
    # Scalar and vector atom subspaces each have normalized ordered-risk
    # sensitivity at most one; the half factor makes the unified atom risk
    # a conservative unit-sensitivity risk functional.
    return 0.5 * _harmonic_ordered_sum(atoms)


def _holder_risk_power(scheme: str) -> float:
    base = _scheme_base(scheme)
    if base == "voi_sageholder15mrr_harmonic_qproject":
        return 1.5
    if base == "voi_sageholder2mrr_harmonic_qproject":
        return 2.0
    raise ValueError(f"Unknown Holder risk scheme {scheme!r}")


def _normalized_holder_atom_risk(
    scheme: str,
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    vector_atoms = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices])) / 0.5
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    atoms = np.concatenate([abs_values, vector_atoms]) if vector_atoms.size else abs_values
    if atoms.size == 0:
        return 0.0
    ordered = np.sort(atoms)[::-1]
    weights = 1.0 / np.arange(1, ordered.size + 1, dtype=np.float64)
    p = _holder_risk_power(scheme)
    raw = float(np.sum(weights * np.power(ordered, p)) ** (1.0 / p))
    sensitivity = float(np.sum(weights) ** (1.0 / p))
    return raw / max(sensitivity, 1.0e-12)


def _nested_block_cvar_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    def block_tail_mass(values: np.ndarray, k: int) -> float:
        values = values.astype(np.float64, copy=False)
        if values.size == 0 or k <= 0:
            return 0.0
        k = min(int(k), int(values.size))
        if k == values.size:
            return float(np.sum(values))
        return float(np.sum(np.partition(values, values.size - k)[values.size - k :]))

    block_risks = np.asarray(
        [
            0.5
            * block_tail_mass(
                abs_values[blocks[block_id].query_indices],
                max(1, int(math.ceil(math.sqrt(float(max(1, blocks[block_id].query_indices.size)))))),
            )
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    if block_risks.size == 0:
        return 0.0
    outer_k = max(1, int(math.ceil(math.sqrt(float(block_risks.size)))))
    return _topk_mean(block_risks, outer_k)


def _hierarchical_block_tail_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)

    def tail_mass(values: np.ndarray, fraction: float) -> float:
        values = values.astype(np.float64, copy=False)
        if values.size == 0:
            return 0.0
        keep = min(values.size, max(1, int(math.ceil(float(fraction) * float(values.size)))))
        if keep == values.size:
            return float(np.sum(values))
        return float(np.sum(np.partition(values, values.size - keep)[values.size - keep :]))

    block_risks = np.asarray(
        [
            0.5 * tail_mass(abs_values[blocks[block_id].query_indices], 0.25)
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    if block_risks.size == 0:
        return 0.0
    outer_keep = min(block_risks.size, max(1, int(math.ceil(0.25 * float(block_risks.size)))))
    if outer_keep == block_risks.size:
        return float(np.sum(block_risks))
    return float(np.sum(np.partition(block_risks, block_risks.size - outer_keep)[block_risks.size - outer_keep :]))


def _repair_capacity_fraction(scheme: str) -> float:
    base = _scheme_base(scheme)
    if base == "voi_sagerepaircap10mrr_harmonic_qproject":
        return 0.10
    if base == "voi_sagerepaircap25mrr_harmonic_qproject":
        return 0.25
    if base == "voi_sagerepaircap50mrr_harmonic_qproject":
        return 0.50
    raise ValueError(f"Unknown repair capacity scheme {scheme!r}")


def _repair_graph_block_capacity_risk(
    scheme: str,
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    fraction = _repair_capacity_fraction(scheme)
    block_values: list[float] = []
    for block_id in vector_block_ids:
        idx = blocks[block_id].query_indices
        if idx.size == 0:
            continue
        keep = min(idx.size, max(1, int(math.ceil(fraction * float(idx.size)))))
        block_values.append(0.5 * _top_sum(abs_values[idx], keep))
    values = np.asarray(block_values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    m = float(values.size)
    # A public capacity envelope: all blocks receive a small floor mass, while
    # no block can monopolize the outer risk. This keeps the risk broad after
    # the first few tail blocks have been repaired.
    return _floored_capacity_envelope(values, floor=0.25 / m, cap=4.0 / m)


def _graph_capacity_mass_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    block_values: list[float] = []
    for block_id in vector_block_ids:
        idx = blocks[block_id].query_indices
        if idx.size == 0:
            continue
        keep = min(idx.size, max(1, int(math.ceil(math.sqrt(float(idx.size))))))
        block_values.append(0.5 * _top_sum(abs_values[idx], keep))
    values = np.asarray(block_values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    m = float(values.size)
    return _floored_capacity_envelope(values, floor=0.05 / m, cap=0.95)


def _graph_capacity_full_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    values = _vector_full_tvd_values(abs_values, blocks, vector_block_ids)
    return _graph_capacity_full_values(values)


def _vector_full_tvd_values(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> np.ndarray:
    abs_values = abs_values.astype(np.float64, copy=False)
    return np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )


def _graph_capacity_full_values(values: np.ndarray) -> float:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    m = float(values.size)
    return _floored_capacity_envelope(values, floor=0.05 / m, cap=0.95)


def _query_vector_capacity_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_risk = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    vector_risk = _graph_capacity_full_risk(abs_values, blocks, vector_block_ids) / 0.5
    return 0.5 * (query_risk + vector_risk)


def _query_vector_curriculum_alpha(round_id: int | None, rounds: int | None) -> float:
    if round_id is None or rounds is None:
        return 0.5
    vector_phase = min(max(1, int(rounds)), max(1, int(math.ceil(math.sqrt(float(max(1, rounds)))))))
    return 0.25 if int(round_id) <= vector_phase else 0.5


def _query_vector_curriculum_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    round_id: int | None,
    rounds: int | None,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    alpha = _query_vector_curriculum_alpha(round_id, rounds)
    query_risk = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    vector_risk = _graph_capacity_full_risk(abs_values, blocks, vector_block_ids) / 0.5
    return alpha * query_risk + (1.0 - alpha) * vector_risk


def _query_vector_reserve_capacity_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    round_id: int | None,
    rounds: int | None,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    alpha = _query_vector_curriculum_alpha(round_id, rounds)
    query_risk = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    vector_risk = _graph_capacity_full_risk(abs_values, blocks, vector_block_ids) / 0.5
    if query_risk <= vector_risk:
        return vector_risk
    return alpha * query_risk + (1.0 - alpha) * vector_risk


def _query_vector_capacity_band_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    round_id: int | None,
    rounds: int | None,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    lower = 0.25
    upper = max(lower, _query_vector_curriculum_alpha(round_id, rounds))
    query_risk = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    vector_risk = _graph_capacity_full_risk(abs_values, blocks, vector_block_ids) / 0.5
    if query_risk <= vector_risk:
        return lower * query_risk + (1.0 - lower) * vector_risk
    return upper * query_risk + (1.0 - upper) * vector_risk


def _harmonic_ordered_noise_floor(size: int, sigma: float, sensitivity: float) -> float:
    size = int(size)
    if size <= 0:
        return 0.0
    expected = _halfnormal_top_expected(size)
    ranks = np.arange(1, size + 1, dtype=np.float64)
    floor = float(sigma) * float(np.sum(expected / ranks))
    return floor / max(float(sensitivity), 1.0e-12)


def _query_vector_calibrated_channels(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
) -> tuple[float, float]:
    abs_values = abs_values.astype(np.float64, copy=False)
    n_public = max(float(row_count or 1), 1.0)
    sigma = max(float(measurement_sigma), 0.0)

    query_raw = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    query_floor = _harmonic_ordered_noise_floor(abs_values.size, sigma, query_sensitivity)
    query_denom = max(n_public - query_floor, n_public, 1.0e-12)
    query_calibrated = n_public * max(0.0, query_raw - query_floor) / query_denom

    vector_raw_values = _vector_full_tvd_values(abs_values, blocks, vector_block_ids)
    vector_raw = _graph_capacity_full_values(vector_raw_values) / 0.5
    vector_floor_values = np.asarray(
        [
            _vector_tvd_noise_floor(
                int(blocks[block_id].query_indices.size),
                sigma,
                float(blocks[block_id].delta_l2),
            )
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_floor = _graph_capacity_full_values(vector_floor_values) / 0.5
    vector_denom = max(2.0 * n_public - vector_floor, n_public, 1.0e-12)
    vector_calibrated = n_public * max(0.0, vector_raw - vector_floor) / vector_denom
    return float(query_calibrated), float(vector_calibrated)


def _query_vector_calibrated_tail_channel(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    measurement_sigma: float,
    row_count: int | None,
    mode: str = "max",
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    n_public = max(float(row_count or 1), 1.0)
    sigma = max(float(measurement_sigma), 0.0)
    vector_raw_values = _vector_full_tvd_values(abs_values, blocks, vector_block_ids)
    if vector_raw_values.size == 0:
        return 0.0
    vector_floor_values = np.asarray(
        [
            _vector_tvd_noise_floor(
                int(blocks[block_id].query_indices.size),
                sigma,
                float(blocks[block_id].delta_l2),
            )
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    if mode == "max":
        vector_tail_raw = float(np.max(vector_raw_values)) / 0.5
        vector_tail_floor = float(np.max(vector_floor_values)) / 0.5 if vector_floor_values.size else 0.0
    elif mode == "top_sqrt":
        tail_k = max(1, int(math.ceil(math.sqrt(float(vector_raw_values.size)))))
        vector_tail_raw = _topk_mean(vector_raw_values, tail_k) / 0.5
        vector_tail_floor = _topk_mean(vector_floor_values, tail_k) / 0.5
    else:
        raise ValueError(f"Unknown calibrated tail mode {mode!r}")
    vector_tail_denom = max(2.0 * n_public - vector_tail_floor, n_public, 1.0e-12)
    return float(n_public * max(0.0, vector_tail_raw - vector_tail_floor) / vector_tail_denom)


def _query_vector_calibrated_scalar_tail_channel(
    abs_values: np.ndarray,
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    mode: str = "max",
    indices: np.ndarray | None = None,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    if indices is not None:
        abs_values = abs_values[indices.astype(np.int64, copy=False)]
    n_public = max(float(row_count or 1), 1.0)
    sigma = max(float(measurement_sigma), 0.0)
    sensitivity = 1.0
    if abs_values.size == 0:
        return 0.0
    expected = _halfnormal_top_expected(abs_values.size)
    if mode == "max":
        query_tail_raw = float(np.max(abs_values)) / sensitivity
        query_tail_floor = sigma * float(expected[0]) / sensitivity
    elif mode == "top_sqrt":
        tail_k = max(1, int(math.ceil(math.sqrt(float(abs_values.size)))))
        query_tail_raw = _topk_mean(abs_values, tail_k) / sensitivity
        query_tail_floor = sigma * float(np.mean(expected[:tail_k])) / sensitivity
    else:
        raise ValueError(f"Unknown calibrated scalar tail mode {mode!r}")
    query_tail_denom = max(n_public - query_tail_floor, n_public, 1.0e-12)
    return float(n_public * max(0.0, query_tail_raw - query_tail_floor) / query_tail_denom)


def _query_vector_calibrated_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    mode: str,
    round_id: int | None,
    rounds: int | None,
) -> float:
    query_risk, vector_risk = _query_vector_calibrated_channels(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
    )
    if mode == "avg":
        return 0.5 * (query_risk + vector_risk)
    alpha = _query_vector_curriculum_alpha(round_id, rounds)
    if mode == "curr":
        return alpha * query_risk + (1.0 - alpha) * vector_risk
    if mode == "band":
        lower = 0.25
        upper = max(lower, alpha)
        if query_risk <= vector_risk:
            return lower * query_risk + (1.0 - lower) * vector_risk
        return upper * query_risk + (1.0 - upper) * vector_risk
    raise ValueError(f"Unknown calibrated query/vector risk mode {mode!r}")


def _query_vector_calibrated_dro_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    tail_mode: str = "max",
    constraint_mode: str = "reserved_tail",
    scalar_tail_mode: str | None = None,
    scalar_tail_indices: np.ndarray | None = None,
) -> float:
    query_risk, vector_risk = _query_vector_calibrated_channels(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
    )
    vector_tail_risk = _query_vector_calibrated_tail_channel(
        abs_values,
        blocks,
        vector_block_ids,
        measurement_sigma,
        row_count,
        mode=tail_mode,
    )
    if scalar_tail_mode is None:
        channels = np.asarray([query_risk, vector_risk, vector_tail_risk], dtype=np.float64)
    else:
        query_tail_risk = _query_vector_calibrated_scalar_tail_channel(
            abs_values,
            query_sensitivity,
            measurement_sigma,
            row_count,
            mode=scalar_tail_mode,
            indices=scalar_tail_indices,
        )
        channels = np.asarray([query_risk, query_tail_risk, vector_risk, vector_tail_risk], dtype=np.float64)
    if constraint_mode == "reserved_tail":
        if scalar_tail_mode is None:
            floors = np.asarray([0.25, 0.25, 0.0], dtype=np.float64)
            caps = np.asarray([1.0, 1.0, 0.5], dtype=np.float64)
        else:
            floors = np.asarray([0.25, 0.0, 0.25, 0.0], dtype=np.float64)
            caps = np.asarray([1.0, 0.5, 1.0, 0.5], dtype=np.float64)
    elif constraint_mode == "uniform_box":
        m = float(channels.size)
        center = 1.0 / max(m, 1.0)
        radius = 0.5 / max(m, 1.0)
        floors = np.full(channels.size, max(0.0, center - radius), dtype=np.float64)
        caps = np.full(channels.size, min(1.0, center + radius), dtype=np.float64)
    else:
        raise ValueError(f"Unknown calibrated DRO constraint mode {constraint_mode!r}")
    return _bounded_capacity_envelope(channels, floors, caps)


def _query_vector_calibrated_tail_capacity_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
) -> float:
    query_risk, vector_risk = _query_vector_calibrated_channels(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
    )
    scalar_tail = _query_vector_calibrated_scalar_tail_channel(
        abs_values,
        query_sensitivity,
        measurement_sigma,
        row_count,
        mode="max",
    )
    vector_tail = _query_vector_calibrated_tail_channel(
        abs_values,
        blocks,
        vector_block_ids,
        measurement_sigma,
        row_count,
        mode="max",
    )
    channels = np.asarray([query_risk, vector_risk, scalar_tail, vector_tail], dtype=np.float64)
    floors = np.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)
    caps = np.asarray([0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    return _bounded_capacity_envelope(channels, floors, caps)


def _query_vector_calibrated_atom_capacity_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_risk, vector_risk = _query_vector_calibrated_channels(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
    )
    n_public = max(float(row_count or 1), 1.0)
    sigma = max(float(measurement_sigma), 0.0)
    scalar_floor = sigma * math.sqrt(2.0 / math.pi)
    scalar_atoms = n_public * np.maximum(abs_values - scalar_floor, 0.0) / max(
        n_public - scalar_floor,
        n_public,
        1.0e-12,
    )
    scalar_cap = 1.0 / max(1.0, math.sqrt(float(max(1, scalar_atoms.size))))
    scalar_atom_risk = _capacity_envelope(scalar_atoms, scalar_cap) if scalar_atoms.size else 0.0

    vector_raw_values = _vector_full_tvd_values(abs_values, blocks, vector_block_ids)
    vector_floor_values = np.asarray(
        [
            _vector_tvd_noise_floor(
                int(blocks[block_id].query_indices.size),
                sigma,
                float(blocks[block_id].delta_l2),
            )
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_atoms = n_public * np.maximum(
        vector_raw_values / 0.5 - vector_floor_values / 0.5,
        0.0,
    ) / np.maximum(2.0 * n_public - vector_floor_values / 0.5, n_public)
    vector_cap = 1.0 / max(1.0, math.sqrt(float(max(1, vector_atoms.size))))
    vector_atom_risk = _capacity_envelope(vector_atoms, vector_cap) if vector_atoms.size else 0.0

    channels = np.asarray([query_risk, vector_risk, scalar_atom_risk, vector_atom_risk], dtype=np.float64)
    floors = np.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)
    caps = np.asarray([0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    return _bounded_capacity_envelope(channels, floors, caps)


def _query_vector_calibrated_max_frontier_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
) -> float:
    qv_risk = _query_vector_calibrated_dro_risk(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
        tail_mode="max",
        constraint_mode="reserved_tail",
        scalar_tail_mode=None,
    )
    scalar_max_risk = _query_vector_calibrated_scalar_tail_channel(
        abs_values,
        query_sensitivity,
        measurement_sigma,
        row_count,
        mode="max",
    )
    return max(float(qv_risk), float(scalar_max_risk))


def _query_vector_calibrated_target_frontier_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    target_qids: np.ndarray,
) -> float:
    qv_risk = _query_vector_calibrated_dro_risk(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
        tail_mode="max",
        constraint_mode="reserved_tail",
        scalar_tail_mode=None,
    )
    target_tail_risk = _query_vector_calibrated_scalar_tail_channel(
        abs_values,
        query_sensitivity,
        measurement_sigma,
        row_count,
        mode="max",
        indices=target_qids,
    )
    return max(float(qv_risk), float(target_tail_risk))


def _query_vector_calibrated_joint_tail_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    target_qids: np.ndarray,
) -> float:
    qv_risk = _query_vector_calibrated_dro_risk(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
        tail_mode="max",
        constraint_mode="reserved_tail",
        scalar_tail_mode=None,
    )
    target_scalar_tail = _query_vector_calibrated_scalar_tail_channel(
        abs_values,
        query_sensitivity,
        measurement_sigma,
        row_count,
        mode="max",
        indices=target_qids,
    )
    vector_tail = _query_vector_calibrated_tail_channel(
        abs_values,
        blocks,
        vector_block_ids,
        measurement_sigma,
        row_count,
        mode="max",
    )
    return max(float(qv_risk), float(target_scalar_tail), float(vector_tail))


def _query_vector_calibrated_paired_frontier_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    target_qids: np.ndarray,
) -> float:
    qv_risk = _query_vector_calibrated_dro_risk(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
        tail_mode="max",
        constraint_mode="reserved_tail",
        scalar_tail_mode=None,
    )
    scalar_frontier = _query_vector_calibrated_scalar_tail_channel(
        abs_values,
        query_sensitivity,
        measurement_sigma,
        row_count,
        mode="max",
        indices=target_qids,
    )
    target_set = set(int(qid) for qid in target_qids.astype(np.int64, copy=False))
    paired_vector_ids = [
        block_id
        for block_id in vector_block_ids
        if any(int(qid) in target_set for qid in blocks[block_id].query_indices)
    ]
    vector_frontier = _query_vector_calibrated_tail_channel(
        abs_values,
        blocks,
        paired_vector_ids,
        measurement_sigma,
        row_count,
        mode="max",
    )
    paired_tail = max(
        (2.0 / 3.0) * scalar_frontier + (1.0 / 3.0) * vector_frontier,
        (1.0 / 3.0) * scalar_frontier + (2.0 / 3.0) * vector_frontier,
    )
    return max(float(qv_risk), float(paired_tail))


def _query_vector_calibrated_balanced_frontier_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    target_qids: np.ndarray,
) -> float:
    qv_risk = _query_vector_calibrated_dro_risk(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
        tail_mode="max",
        constraint_mode="reserved_tail",
        scalar_tail_mode=None,
    )
    scalar_frontier = _query_vector_calibrated_scalar_tail_channel(
        abs_values,
        query_sensitivity,
        measurement_sigma,
        row_count,
        mode="max",
        indices=target_qids,
    )
    target_set = set(int(qid) for qid in target_qids.astype(np.int64, copy=False))
    paired_vector_ids = [
        block_id
        for block_id in vector_block_ids
        if any(int(qid) in target_set for qid in blocks[block_id].query_indices)
    ]
    vector_frontier = _query_vector_calibrated_tail_channel(
        abs_values,
        blocks,
        paired_vector_ids,
        measurement_sigma,
        row_count,
        mode="max",
    )
    return max(float(qv_risk), min(float(scalar_frontier), float(vector_frontier)))


def _query_vector_calibrated_flow_guard_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    measurement_sigma: float,
    row_count: int | None,
    edge_flow_context: EdgeFlowRiskContext,
) -> float:
    qv_risk = _query_vector_calibrated_dro_risk(
        abs_values,
        blocks,
        vector_block_ids,
        query_sensitivity,
        measurement_sigma,
        row_count,
        tail_mode="max",
        constraint_mode="reserved_tail",
        scalar_tail_mode=None,
    )
    flow_guard = _edge_flow_capacity_risk(abs_values, edge_flow_context)
    return max((1.0 / 3.0) * float(qv_risk), float(flow_guard))


def _query_vector_capacity_envelope_risk(
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    channels = np.asarray(
        [
            _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12),
            _graph_capacity_full_risk(abs_values, blocks, vector_block_ids) / 0.5,
        ],
        dtype=np.float64,
    )
    m = float(channels.size)
    return _floored_capacity_envelope(channels, floor=0.5 / m, cap=2.0 / m)


def _valid_full_tvd_gain_values(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> np.ndarray:
    target_qids = plan.target_qids
    source_size = int(source_block.query_indices.size)
    gains: list[float] = []
    for target_block_id in vector_block_ids:
        target_idx = blocks[target_block_id].query_indices
        repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
        repaired = float(np.count_nonzero(repair_mask))
        if repaired <= 0.0:
            gains.append(0.0)
            continue
        local_idx = target_idx[repair_mask]
        breadth = _valid_local_tail_propagation_weight(repaired, float(target_idx.size))
        source_coverage = _source_cell_propagation_weight(
            plan,
            local_idx,
            source_size,
            int(target_idx.size),
        )
        block_gain = max(
            0.0,
            0.5 * float(np.sum(before_abs[target_idx]) - np.sum(after_abs[target_idx])),
        )
        gains.append(breadth * source_coverage * block_gain)
    return np.asarray(gains, dtype=np.float64)


def _valid_full_tvd_score(
    scheme: str,
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    if values.size == 0:
        return 0.0
    base = _scheme_base(scheme)
    if base == "voi_sagevalidfullmaxmrr_harmonic_qproject":
        return float(np.max(values))
    if base == "voi_sagevalidfullcapmrr_harmonic_qproject":
        m = float(values.size)
        return _floored_capacity_envelope(values, floor=0.05 / m, cap=0.95)
    raise ValueError(f"Unknown valid full TVD scheme {scheme!r}")


def _valid_max_tvd_drop_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    target_qids = plan.target_qids
    source_size = int(source_block.query_indices.size)
    before_values: list[float] = []
    after_values: list[float] = []
    for target_block_id in vector_block_ids:
        target_idx = blocks[target_block_id].query_indices
        if target_idx.size == 0:
            continue
        before_block = 0.5 * float(np.sum(before_abs[target_idx]))
        raw_after_block = 0.5 * float(np.sum(after_abs[target_idx]))
        repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
        repaired = float(np.count_nonzero(repair_mask))
        validity = 0.0
        if repaired > 0.0:
            local_idx = target_idx[repair_mask]
            breadth = _valid_local_tail_propagation_weight(repaired, float(target_idx.size))
            source_coverage = _source_cell_propagation_weight(
                plan,
                local_idx,
                source_size,
                int(target_idx.size),
            )
            validity = breadth * source_coverage
        valid_drop = validity * max(0.0, before_block - raw_after_block)
        before_values.append(before_block)
        after_values.append(max(0.0, before_block - valid_drop))
    if not before_values:
        return 0.0
    return max(0.0, max(before_values) - max(after_values))


def _valid_topk_tvd_drop_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    target_qids = plan.target_qids
    source_size = int(source_block.query_indices.size)
    before_values: list[float] = []
    after_values: list[float] = []
    for target_block_id in vector_block_ids:
        target_idx = blocks[target_block_id].query_indices
        if target_idx.size == 0:
            continue
        before_block = 0.5 * float(np.sum(before_abs[target_idx]))
        raw_after_block = 0.5 * float(np.sum(after_abs[target_idx]))
        repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
        repaired = float(np.count_nonzero(repair_mask))
        validity = 0.0
        if repaired > 0.0:
            local_idx = target_idx[repair_mask]
            breadth = _valid_local_tail_propagation_weight(repaired, float(target_idx.size))
            source_coverage = _source_cell_propagation_weight(
                plan,
                local_idx,
                source_size,
                int(target_idx.size),
            )
            validity = breadth * source_coverage
        valid_drop = validity * max(0.0, before_block - raw_after_block)
        before_values.append(before_block)
        after_values.append(max(0.0, before_block - valid_drop))
    if not before_values:
        return 0.0
    before_arr = np.asarray(before_values, dtype=np.float64)
    after_arr = np.asarray(after_values, dtype=np.float64)
    k = _cvar_tail_k("voi_sagerrcvalidtopkmrr_harmonic_qproject", before_arr.size)
    return max(0.0, _topk_mean(before_arr, k) - _topk_mean(after_arr, k))


def _valid_topk_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    if values.size == 0:
        return 0.0
    k = _cvar_tail_k("voi_sagerrcvalidtopkgainmrr_harmonic_qproject", values.size)
    return _topk_mean(values, k)


def _valid_head_topk_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    if values.size == 0:
        return 0.0
    k = _cvar_tail_k("voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject", values.size)
    return _head_topk_mean(values, k)


def _valid_harmonic_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    return _harmonic_ordered_mean(values)


def _valid_lorenz_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    return _lorenz_head_broad_mean(values)


def _valid_nested_head_broad_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    before_values = np.asarray(
        [
            0.5 * float(np.sum(before_abs[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    if before_values.size == 0:
        return 0.0
    gain_values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    head_risk_norm = max(float(np.max(before_values)) / 0.5, 1.0e-12)
    broad_risk_norm = max(_harmonic_ordered_mean(before_values) / 0.5, 1.0e-12)
    head_gain_norm = float(np.max(gain_values)) if gain_values.size else 0.0
    broad_gain_norm = _harmonic_ordered_mean(gain_values)
    parity_scale = min(head_risk_norm, broad_risk_norm)
    return max(
        head_gain_norm / head_risk_norm,
        broad_gain_norm / broad_risk_norm,
    ) * parity_scale


def _valid_nested_lorenz_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    before_values = np.asarray(
        [
            0.5 * float(np.sum(before_abs[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    if before_values.size == 0:
        return 0.0
    gain_values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    head_risk_norm = max(float(np.max(before_values)) / 0.5, 1.0e-12)
    broad_risk_norm = max(_lorenz_head_broad_mean(before_values) / 0.5, 1.0e-12)
    head_gain_norm = float(np.max(gain_values)) if gain_values.size else 0.0
    broad_gain_norm = _lorenz_head_broad_mean(gain_values)
    parity_scale = min(head_risk_norm, broad_risk_norm)
    return max(
        head_gain_norm / head_risk_norm,
        broad_gain_norm / broad_risk_norm,
    ) * parity_scale


def _valid_nested_topk_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    before_values = np.asarray(
        [
            0.5 * float(np.sum(before_abs[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    if before_values.size == 0:
        return 0.0
    gain_values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    k = _cvar_tail_k("voi_sagerrcnestedtopkvalidmrr_harmonic_qproject", before_values.size)
    head_risk_norm = max(float(np.max(before_values)) / 0.5, 1.0e-12)
    broad_risk_norm = max(_topk_mean(before_values, k) / 0.5, 1.0e-12)
    head_gain_norm = float(np.max(gain_values)) if gain_values.size else 0.0
    broad_gain_norm = _topk_mean(gain_values, k)
    parity_scale = min(head_risk_norm, broad_risk_norm)
    return max(
        head_gain_norm / head_risk_norm,
        broad_gain_norm / broad_risk_norm,
    ) * parity_scale


def _valid_nested_pareto_tvd_gain_score(
    plan: QueryRepairPlan,
    source_block: AdaptiveBlock,
    before_abs: np.ndarray,
    after_abs: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
) -> float:
    before_values = np.asarray(
        [
            0.5 * float(np.sum(before_abs[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    if before_values.size == 0:
        return 0.0
    gain_values = _valid_full_tvd_gain_values(
        plan,
        source_block,
        before_abs,
        after_abs,
        blocks,
        vector_block_ids,
    )
    head_risk_norm = max(float(np.max(before_values)) / 0.5, 1.0e-12)
    broad_risk_norm = max(_lorenz_head_broad_mean(before_values) / 0.5, 1.0e-12)
    head_gain_norm = float(np.max(gain_values)) if gain_values.size else 0.0
    broad_gain_norm = _lorenz_head_broad_mean(gain_values)
    head_rel = head_gain_norm / head_risk_norm
    broad_rel = broad_gain_norm / broad_risk_norm
    parity_scale = min(head_risk_norm, broad_risk_norm)
    return _harmonic_pair_mean(head_rel, broad_rel) * parity_scale


def _floored_capacity_envelope(values: np.ndarray, floor: float, cap: float) -> float:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    m = int(values.size)
    floors = np.full(m, max(0.0, float(floor)), dtype=np.float64)
    caps = np.full(m, max(0.0, float(cap)), dtype=np.float64)
    floors = np.minimum(floors, caps)
    floor_mass = min(float(np.sum(floors)), 1.0)
    floor_part = float(np.sum(floors * values))
    remaining_mass = max(0.0, 1.0 - floor_mass)
    if remaining_mass <= 1.0e-12:
        return floor_part / max(floor_mass, 1.0e-12)
    residual_caps = np.maximum(caps - floors, 0.0)
    residual_total = float(np.sum(residual_caps))
    if residual_total <= 1.0e-12:
        return floor_part + remaining_mass * float(np.mean(values))
    adjusted_caps = residual_caps / remaining_mass
    return floor_part + remaining_mass * _capacity_envelope(values, adjusted_caps)


def _bounded_capacity_envelope(values: np.ndarray, floors: np.ndarray, caps: np.ndarray) -> float:
    values = values.astype(np.float64, copy=False)
    floors = np.maximum(floors.astype(np.float64, copy=False), 0.0)
    caps = np.maximum(caps.astype(np.float64, copy=False), 0.0)
    if values.size == 0:
        return 0.0
    if values.size != floors.size or values.size != caps.size:
        raise ValueError("values, floors, and caps must have the same size")
    floors = np.minimum(floors, caps)
    floor_mass = float(np.sum(floors))
    if floor_mass >= 1.0:
        weights = floors / max(floor_mass, 1.0e-12)
        return float(np.sum(weights * values))
    floor_part = float(np.sum(floors * values))
    remaining_mass = 1.0 - floor_mass
    residual_caps = np.maximum(caps - floors, 0.0)
    residual_total = float(np.sum(residual_caps))
    if residual_total <= 1.0e-12:
        return floor_part + remaining_mass * float(np.mean(values))
    adjusted_caps = residual_caps / remaining_mass
    return floor_part + remaining_mass * _capacity_envelope(values, adjusted_caps)


def _normalized_metric_risk_value(scheme: str, channels: np.ndarray) -> float:
    channels = channels.astype(np.float64, copy=False)
    if channels.size == 0:
        return 0.0
    base = _scheme_base(scheme)
    if base == "voi_sagemetricavgmrr_harmonic_qproject":
        return float(np.mean(channels))
    if base == "voi_sagemetricenvmrr_harmonic_qproject":
        return _softmax_envelope(channels, 0.5)
    if base == "voi_sagefloorcapmrr_harmonic_qproject":
        m = float(channels.size)
        return _floored_capacity_envelope(channels, floor=0.5 / m, cap=2.0 / m)
    raise ValueError(f"Unknown metric risk scheme {scheme!r}")


def _dpstate_risk_envelope_value(
    scheme: str,
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_weights: np.ndarray,
    vector_weights: np.ndarray,
    vector_local_weights: list[np.ndarray] | None,
    query_membership_split: np.ndarray | None,
    query_sensitivity: float,
    vector_sensitivity: float,
) -> float:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_weights = query_weights.astype(np.float64, copy=False)
    query_ordered = float(np.sum(query_weights * abs_values)) / max(float(query_sensitivity), 1.0e-12)

    if vector_local_weights is not None and query_membership_split is not None:
        vector_total = 0.0
        for target_block_id, block_weight, local_weights in zip(
            vector_block_ids,
            vector_weights,
            vector_local_weights,
        ):
            target_idx = blocks[target_block_id].query_indices
            vector_total += (
                float(block_weight)
                * 0.5
                * float(
                    np.sum(
                        local_weights
                        * query_membership_split[target_idx]
                        * abs_values[target_idx]
                    )
                )
            )
        vector_ordered = vector_total / max(float(vector_sensitivity), 1.0e-12)
    else:
        vector_ordered = (
            float(
                sum(
                    float(weight) * 0.5 * float(np.sum(abs_values[blocks[target_block_id].query_indices]))
                    for target_block_id, weight in zip(vector_block_ids, vector_weights)
                )
            )
            / max(float(vector_sensitivity), 1.0e-12)
        )

    query_tail = float(np.max(abs_values)) if abs_values.size else 0.0
    vector_tails = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[target_block_id].query_indices]))
            for target_block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_tail = float(np.max(vector_tails)) / 0.5 if vector_tails.size else 0.0
    base = _scheme_base(scheme)
    if base == "voi_dpstatevtailriskenv025mrr_harmonic_qproject":
        channels = np.asarray(
            [
                query_ordered,
                vector_ordered,
                vector_tail,
            ],
            dtype=np.float64,
        )
        return _softmax_envelope(channels, 0.25)
    channels = np.asarray(
        [
            query_ordered,
            vector_ordered,
            query_tail,
            vector_tail,
        ],
        dtype=np.float64,
    )
    if base == "voi_dpstateriskmaxmrr_harmonic_qproject":
        return float(np.max(channels))
    if base == "voi_dpstateriskenv025mrr_harmonic_qproject":
        return _softmax_envelope(channels, 0.25)
    raise ValueError(f"Unknown DP-state risk-envelope scheme {scheme!r}")


def _build_capacity_risk_context(
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    num_queries: int,
) -> CapacityRiskContext:
    families = tuple(sorted({block.family for block in blocks}))
    vector_position_by_block = {int(block_id): pos for pos, block_id in enumerate(vector_block_ids)}
    qids_by_family: list[np.ndarray] = []
    vector_positions_by_family: list[np.ndarray] = []
    family_counts = np.asarray(
        [max(1, sum(1 for block in blocks if block.family == family)) for family in families],
        dtype=np.float64,
    )
    for family in families:
        qid_mask = np.zeros(int(num_queries), dtype=bool)
        vector_positions: list[int] = []
        for block_id, block in enumerate(blocks):
            if block.family != family:
                continue
            qid_mask[block.query_indices] = True
            if block_id in vector_position_by_block:
                vector_positions.append(vector_position_by_block[block_id])
        qids_by_family.append(np.flatnonzero(qid_mask).astype(np.int32, copy=False))
        vector_positions_by_family.append(np.asarray(vector_positions, dtype=np.int32))

    # Public MDL-style capacity: large candidate families receive less risk mass,
    # but the square-root softens the prior so real tail signal can still compete.
    family_prior = 1.0 / np.sqrt(family_counts)
    family_prior = family_prior / max(float(np.sum(family_prior)), 1.0e-12)
    caps: list[float] = [
        0.30,  # global query ordered risk
        0.20,  # global query CVaR risk
        0.30,  # global vector ordered risk
        0.20,  # global vector CVaR risk
        0.20,  # global within-block vector CVaR risk
    ]
    for prior in family_prior.tolist():
        caps.extend(
            [
                0.30 * float(prior),  # family query tail
                0.30 * float(prior),  # family vector TVD tail
                0.20 * float(prior),  # family within-block tail
            ]
        )
    return CapacityRiskContext(
        families=families,
        qids_by_family=tuple(qids_by_family),
        vector_positions_by_family=tuple(vector_positions_by_family),
        atom_caps=np.asarray(caps, dtype=np.float64),
    )


def _capacity_envelope(values: np.ndarray, caps: np.ndarray) -> float:
    values = values.astype(np.float64, copy=False)
    caps = caps.astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    if values.size != caps.size:
        raise ValueError("capacity envelope values/caps size mismatch")
    caps = np.maximum(caps, 0.0)
    total_cap = float(np.sum(caps))
    if total_cap <= 0.0:
        return float(np.max(values))
    order = np.argsort(values)[::-1]
    remaining = 1.0
    total = 0.0
    for idx_raw in order.tolist():
        mass = min(float(caps[int(idx_raw)]), remaining)
        if mass <= 0.0:
            continue
        total += mass * float(values[int(idx_raw)])
        remaining -= mass
        if remaining <= 1.0e-12:
            break
    if remaining > 1.0e-12:
        # The configured caps should sum to more than one, but this fallback
        # preserves a valid convex risk if a future config changes that.
        total += remaining * float(values[int(order[0])])
    return float(total)


def _build_edge_flow_risk_context(
    blocks: list[AdaptiveBlock],
    repair_plans: list[QueryRepairPlan | None],
    vector_block_ids: list[int],
    num_queries: int,
) -> EdgeFlowRiskContext:
    vector_count = max(1, int(len(vector_block_ids)))
    block_tail_k = max(1, int(math.ceil(math.sqrt(float(vector_count)))))
    block_caps = np.full(vector_count, 1.0 / float(block_tail_k), dtype=np.float64)
    query_tail_k = max(1, int(math.ceil(math.sqrt(float(max(1, num_queries))))))
    query_caps = np.full(int(num_queries), 1.0 / float(query_tail_k), dtype=np.float64)
    qid_parts: list[np.ndarray] = []
    block_pos_parts: list[np.ndarray] = []
    edge_cap_parts: list[np.ndarray] = []
    block_qids_by_position = tuple(
        blocks[int(block_id)].query_indices.astype(np.int32, copy=False)
        for block_id in vector_block_ids
    )
    for block_pos, block_id in enumerate(vector_block_ids):
        qid_parts.append(np.asarray([-1], dtype=np.int32))
        block_pos_parts.append(np.asarray([int(block_pos)], dtype=np.int32))
        edge_cap_parts.append(np.asarray([float(block_caps[block_pos])], dtype=np.float64))
        plan = repair_plans[block_id]
        if plan is None or plan.target_qids.size == 0:
            target_qids = blocks[block_id].query_indices.astype(np.int32, copy=False)
        else:
            target_qids = np.unique(plan.target_qids.astype(np.int32, copy=False))
        if target_qids.size == 0:
            continue
        local_tail_k = max(1, int(math.ceil(math.sqrt(float(max(1, blocks[block_id].query_indices.size))))))
        qid_parts.append(target_qids)
        block_pos_parts.append(np.full(target_qids.size, int(block_pos), dtype=np.int32))
        edge_cap_parts.append(
            np.full(
                target_qids.size,
                float(block_caps[block_pos]) / float(local_tail_k),
                dtype=np.float64,
            )
        )
    if not qid_parts:
        return EdgeFlowRiskContext(
            qids=np.asarray([], dtype=np.int32),
            block_positions=np.asarray([], dtype=np.int32),
            edge_caps=np.asarray([], dtype=np.float64),
            block_caps=block_caps,
            query_caps=query_caps,
            block_qids_by_position=block_qids_by_position,
        )
    return EdgeFlowRiskContext(
        qids=np.concatenate(qid_parts).astype(np.int32, copy=False),
        block_positions=np.concatenate(block_pos_parts).astype(np.int32, copy=False),
        edge_caps=np.concatenate(edge_cap_parts).astype(np.float64, copy=False),
        block_caps=block_caps,
        query_caps=query_caps,
        block_qids_by_position=block_qids_by_position,
    )


def _build_graph_capacity_risk_context(
    blocks: list[AdaptiveBlock],
    repair_plans: list[QueryRepairPlan | None],
    vector_block_ids: list[int],
    num_queries: int,
) -> EdgeFlowRiskContext:
    vector_count = max(1, int(len(vector_block_ids)))
    block_tail_k = max(1, int(math.ceil(math.sqrt(float(vector_count)))))
    block_caps = np.full(vector_count, 1.0 / float(block_tail_k), dtype=np.float64)
    query_tail_k = max(1, int(math.ceil(math.sqrt(float(max(1, num_queries))))))
    query_caps = np.full(int(num_queries), 1.0 / float(query_tail_k), dtype=np.float64)
    qid_parts: list[np.ndarray] = []
    block_pos_parts: list[np.ndarray] = []
    edge_cap_parts: list[np.ndarray] = []
    block_qids_by_position = tuple(
        blocks[int(block_id)].query_indices.astype(np.int32, copy=False)
        for block_id in vector_block_ids
    )
    for block_pos, block_id in enumerate(vector_block_ids):
        plan = repair_plans[block_id]
        if plan is None or plan.target_qids.size == 0:
            target_qids = blocks[block_id].query_indices.astype(np.int32, copy=False)
            noise_counts = np.ones(int(target_qids.size), dtype=np.float64)
        else:
            target_qids = plan.target_qids.astype(np.int32, copy=False)
            noise_counts = plan.noise_counts.astype(np.float64, copy=False)
        if target_qids.size == 0:
            continue
        block_size = max(float(blocks[block_id].query_indices.size), 1.0)
        local_tail_k = max(1, int(math.ceil(math.sqrt(block_size))))
        source_reliability = 1.0 / np.sqrt(np.maximum(noise_counts, 1.0))
        edge_caps = (
            float(block_caps[block_pos])
            * np.minimum(source_reliability, 1.0)
            / float(local_tail_k)
        )
        qid_parts.append(target_qids)
        block_pos_parts.append(np.full(target_qids.size, int(block_pos), dtype=np.int32))
        edge_cap_parts.append(edge_caps.astype(np.float64, copy=False))
    if not qid_parts:
        return EdgeFlowRiskContext(
            qids=np.asarray([], dtype=np.int32),
            block_positions=np.asarray([], dtype=np.int32),
            edge_caps=np.asarray([], dtype=np.float64),
            block_caps=block_caps,
            query_caps=query_caps,
            block_qids_by_position=block_qids_by_position,
        )
    return EdgeFlowRiskContext(
        qids=np.concatenate(qid_parts).astype(np.int32, copy=False),
        block_positions=np.concatenate(block_pos_parts).astype(np.int32, copy=False),
        edge_caps=np.concatenate(edge_cap_parts).astype(np.float64, copy=False),
        block_caps=block_caps,
        query_caps=query_caps,
        block_qids_by_position=block_qids_by_position,
    )


def _edge_flow_capacity_risk(abs_values: np.ndarray, context: EdgeFlowRiskContext) -> float:
    if context.qids.size == 0:
        return 0.0
    block_values = np.asarray(
        [0.5 * float(np.sum(abs_values[qids])) for qids in context.block_qids_by_position],
        dtype=np.float64,
    )
    values = np.empty(context.qids.size, dtype=np.float64)
    scalar_mask = context.qids >= 0
    values[scalar_mask] = abs_values[context.qids[scalar_mask]].astype(np.float64, copy=False)
    values[~scalar_mask] = block_values[context.block_positions[~scalar_mask]]
    order = np.argsort(values)[::-1]
    block_remaining = context.block_caps.copy()
    query_remaining = context.query_caps.copy()
    remaining = 1.0
    total = 0.0
    for edge_idx_raw in order.tolist():
        edge_idx = int(edge_idx_raw)
        qid = int(context.qids[edge_idx])
        block_pos = int(context.block_positions[edge_idx])
        query_cap = float(query_remaining[qid]) if qid >= 0 else remaining
        mass = min(
            float(context.edge_caps[edge_idx]),
            float(block_remaining[block_pos]),
            query_cap,
            remaining,
        )
        if mass <= 0.0:
            continue
        total += mass * float(values[edge_idx])
        block_remaining[block_pos] -= mass
        if qid >= 0:
            query_remaining[qid] -= mass
        remaining -= mass
        if remaining <= 1.0e-12:
            break
    return float(total)


def _normalized_capdro_risk_atoms(
    scheme: str,
    abs_values: np.ndarray,
    blocks: list[AdaptiveBlock],
    vector_block_ids: list[int],
    query_sensitivity: float,
    vector_sensitivity: float,
    context: CapacityRiskContext,
) -> np.ndarray:
    abs_values = abs_values.astype(np.float64, copy=False)
    query_ordered = _harmonic_ordered_sum(abs_values) / max(float(query_sensitivity), 1.0e-12)
    query_cvar = _topk_mean(abs_values, _cvar_tail_k(scheme, abs_values.size))
    vector_tvd_values = np.asarray(
        [
            0.5 * float(np.sum(abs_values[blocks[block_id].query_indices]))
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_ordered = _harmonic_ordered_sum(vector_tvd_values) / max(float(vector_sensitivity), 1.0e-12)
    vector_cvar = (
        _topk_mean(vector_tvd_values, _cvar_tail_k(scheme, vector_tvd_values.size)) / 0.5
        if vector_tvd_values.size
        else 0.0
    )
    vector_block_ordered_values = np.asarray(
        [
            0.5 * _harmonic_ordered_sum(abs_values[blocks[block_id].query_indices])
            for block_id in vector_block_ids
        ],
        dtype=np.float64,
    )
    vector_block_cvar = (
        _topk_mean(vector_block_ordered_values, _cvar_tail_k(scheme, vector_block_ordered_values.size)) / 0.5
        if vector_block_ordered_values.size
        else 0.0
    )
    values: list[float] = [
        query_ordered,
        query_cvar,
        vector_ordered,
        vector_cvar,
        vector_block_cvar,
    ]
    for qids, vector_positions in zip(context.qids_by_family, context.vector_positions_by_family):
        family_query = _topk_mean(abs_values[qids], _cvar_tail_k(scheme, qids.size)) if qids.size else 0.0
        if vector_positions.size:
            family_vector_values = vector_tvd_values[vector_positions]
            family_block_values = vector_block_ordered_values[vector_positions]
            family_vector = _topk_mean(family_vector_values, _cvar_tail_k(scheme, family_vector_values.size)) / 0.5
            family_block = _topk_mean(family_block_values, _cvar_tail_k(scheme, family_block_values.size)) / 0.5
        else:
            family_vector = 0.0
            family_block = 0.0
        values.extend([family_query, family_vector, family_block])
    return np.asarray(values, dtype=np.float64)


def _rrc_effective_sensitivities(
    blocks: list[AdaptiveBlock],
    repair_plans: list[QueryRepairPlan | None],
    vector_block_ids: list[int],
    num_queries: int,
) -> np.ndarray:
    """Public sensitivity bound for the positive qproject RRC-VOI numerator."""
    global_query = _harmonic_number(int(num_queries))
    # Current vector workloads are partition blocks, so one record contributes
    # at most one cell per vector block and changes each block TVD by at most 1/2.
    global_vector = 0.5 * _harmonic_number(len(vector_block_ids))
    qid_to_vector_positions: list[list[int]] = [[] for _ in range(int(num_queries))]
    for vector_pos, block_id in enumerate(vector_block_ids):
        for qid in blocks[block_id].query_indices.tolist():
            qid_to_vector_positions[int(qid)].append(vector_pos)

    result = np.full(len(blocks), max(global_query + global_vector, 1.0e-12), dtype=np.float64)
    num_vectors = len(vector_block_ids)
    for block_id, block in enumerate(blocks):
        plan = repair_plans[block_id]
        if plan is None or plan.target_qids.size == 0:
            continue
        source_size = int(block.query_indices.size)
        if source_size <= 0:
            continue
        query_counts = np.zeros(source_size, dtype=np.float64)
        vector_counts = np.zeros((source_size, num_vectors), dtype=np.float64)
        for qid_raw, cells in zip(plan.target_qids.tolist(), plan.cell_indices):
            if cells.size == 0:
                continue
            query_counts[cells] += 1.0
            for vector_pos in qid_to_vector_positions[int(qid_raw)]:
                vector_counts[cells, vector_pos] += 0.5
        gain_query = _harmonic_number(int(np.max(query_counts))) if query_counts.size else 0.0
        gain_vector = 0.0
        if vector_counts.size:
            gain_vector = max(_harmonic_ordered_sum(row) for row in vector_counts)
        result[block_id] = max(global_query + global_vector + gain_query + gain_vector, 1.0e-12)
    return result


def _generator_reachable_source_signal(
    source_residual: np.ndarray,
    *,
    measurement_sigma: float,
    delta_l2: float,
    active_query_budget: int,
    kappa_noise: float,
    mode: str,
) -> np.ndarray:
    residual = source_residual.astype(np.float64, copy=False)
    if residual.size == 0 or active_query_budget <= 0:
        return np.zeros_like(residual, dtype=np.float64)
    sigma = max(float(measurement_sigma) * max(float(delta_l2), 1.0e-12), 1.0e-12)
    priority = (np.abs(residual) - float(kappa_noise) * sigma) / sigma
    reachable = np.flatnonzero(priority > 0.0)
    if reachable.size == 0:
        return np.zeros_like(residual, dtype=np.float64)
    order = reachable[np.argsort(-priority[reachable])]
    weights = np.zeros_like(residual, dtype=np.float64)
    if mode == "hard":
        weights[order[: min(int(active_query_budget), order.size)]] = 1.0
    elif mode == "exposure":
        ranks = np.arange(1, order.size + 1, dtype=np.float64)
        weights[order] = np.minimum(1.0, float(active_query_budget) / ranks)
    else:
        raise ValueError(f"Unknown generator source signal mode {mode!r}")
    return residual * weights


def _measured_context_degrees(blocks: list[AdaptiveBlock], selected_block_ids: list[int] | None) -> np.ndarray | None:
    if not selected_block_ids:
        return None
    max_attr = -1
    for block in blocks:
        if block.scope:
            max_attr = max(max_attr, max(block.scope))
    if max_attr < 0:
        return None
    degrees = np.zeros(max_attr + 1, dtype=np.float64)
    for block_id in selected_block_ids:
        scope = blocks[int(block_id)].scope
        if len(scope) < 2:
            continue
        for attr in scope:
            degrees[int(attr)] += 1.0
    return degrees


def _graph_context_multiplier(block: AdaptiveBlock, degrees: np.ndarray | None) -> float:
    if degrees is None or len(block.scope) < 2:
        return 1.0
    vals = [float(degrees[int(attr)]) if int(attr) < degrees.size else 0.0 for attr in block.scope]
    pair_products = [left * right for pos, left in enumerate(vals) for right in vals[pos + 1 :]]
    if not pair_products:
        return 1.0
    return 1.0 + float(np.mean(pair_products))


def _build_conditional_propagation_context(
    qcat: QueryCatalogue,
    blocks: list[AdaptiveBlock],
    X_syn: np.ndarray,
    batch_size: int,
) -> ConditionalPropagationContext:
    assignments: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    vector_block_ids = [
        block_id
        for block_id, block in enumerate(blocks)
        if block.is_vector and int(block.query_indices.size) > 1
    ]
    for block_id in vector_block_ids:
        block = blocks[block_id]
        subcat = filter_query_catalogue(qcat, block.query_indices.astype(np.int32, copy=False))
        assignment_parts: list[np.ndarray] = []
        for start in range(0, X_syn.shape[0], int(batch_size)):
            batch = X_syn[start : start + int(batch_size)]
            sat = np.asarray(eval_records_queries(batch, subcat), dtype=np.bool_)
            if sat.ndim != 2 or sat.shape[1] == 0:
                assignment_parts.append(np.full(batch.shape[0], -1, dtype=np.int32))
                continue
            any_sat = np.any(sat, axis=1)
            local = np.argmax(sat, axis=1).astype(np.int32, copy=False)
            local[~any_sat] = -1
            assignment_parts.append(local)
        assignment = np.concatenate(assignment_parts).astype(np.int32, copy=False)
        assignments[block_id] = assignment
        valid = assignment[assignment >= 0]
        counts[block_id] = np.bincount(valid, minlength=int(block.query_indices.size)).astype(np.float64)
    return ConditionalPropagationContext(
        assignments=assignments,
        counts=counts,
        vector_block_ids=vector_block_ids,
    )


def _conditional_vector_propagation_score(
    *,
    block_id: int,
    block: AdaptiveBlock,
    source_repair_signal: np.ndarray,
    residual: np.ndarray,
    measurement_sigma: float,
    exact_target_qids: np.ndarray,
    blocks: list[AdaptiveBlock],
    context: ConditionalPropagationContext | None,
    atom_vector_block_ids: list[int],
    atom_vector_weights: np.ndarray,
    target_context_degrees: np.ndarray | None = None,
    target_context_excess_only: bool = False,
    source_context_factor: float = 1.0,
    bridge_context_excess_only: bool = False,
    normalize_effective_targets: bool = False,
    existing_precision: np.ndarray | None = None,
    precision_innovation: bool = False,
) -> float:
    if context is None or block_id not in context.assignments:
        return 0.0
    source_scope = set(block.scope)
    if not source_scope:
        return 0.0
    source_assignment = context.assignments[block_id]
    source_counts = context.counts.get(block_id)
    if source_counts is None or source_counts.size != source_repair_signal.size:
        return 0.0
    valid_source = source_assignment >= 0
    if not np.any(valid_source):
        return 0.0
    exact_qids = exact_target_qids.astype(np.int32, copy=False)
    score = 0.0
    noise_scale = float(measurement_sigma) * max(float(block.delta_l2), 1.0e-12)
    for vector_weight, target_block_id in zip(atom_vector_weights, atom_vector_block_ids):
        if target_block_id == block_id or target_block_id not in context.assignments:
            continue
        target_block = blocks[target_block_id]
        if not source_scope.intersection(target_block.scope):
            continue
        target_idx = target_block.query_indices.astype(np.int32, copy=False)
        if target_idx.size == 0:
            continue
        new_mask = ~np.isin(target_idx, exact_qids)
        if not np.any(new_mask):
            continue
        target_assignment = context.assignments[target_block_id]
        valid = valid_source & (target_assignment >= 0)
        if not np.any(valid):
            continue
        k_target = int(target_idx.size)
        pair_ids = source_assignment[valid].astype(np.int64) * int(k_target) + target_assignment[valid].astype(np.int64)
        joint = np.bincount(
            pair_ids,
            minlength=int(source_repair_signal.size) * int(k_target),
        ).reshape(int(source_repair_signal.size), int(k_target))
        denom = np.maximum(source_counts, 1.0)[:, None]
        probabilities = joint.astype(np.float64) / denom
        probabilities[source_counts <= 0.0, :] = 0.0
        projected_repair = np.asarray(probabilities.T @ source_repair_signal, dtype=np.float64)
        noise_var = noise_scale * noise_scale * np.sum(probabilities * probabilities, axis=0)
        target_residual = residual[target_idx]
        before = np.abs(target_residual[new_mask])
        after = _expected_abs_normal(
            target_residual[new_mask] - projected_repair[new_mask],
            np.sqrt(noise_var[new_mask]),
        )
        gain = before - after
        if precision_innovation and existing_precision is not None:
            candidate_precision = 1.0 / np.maximum(noise_var[new_mask], 1.0e-12)
            prior_precision = existing_precision[target_idx[new_mask]]
            gain = gain * (
                candidate_precision / np.maximum(candidate_precision + prior_precision, 1.0e-12)
            )
        context_factor = _graph_context_multiplier(target_block, target_context_degrees)
        if bridge_context_excess_only:
            context_factor = max(math.sqrt(max(float(source_context_factor), 0.0) * context_factor) - 1.0, 0.0)
        if target_context_excess_only:
            context_factor = max(context_factor - 1.0, 0.0)
        edge_gain = float(np.sum(gain))
        if normalize_effective_targets:
            col_norm = np.sqrt(np.sum(probabilities[:, new_mask] * probabilities[:, new_mask], axis=0))
            norm_sum = float(np.sum(col_norm))
            norm_sq = float(np.sum(col_norm * col_norm))
            if norm_sum > 0.0 and norm_sq > 0.0:
                effective_targets = (norm_sum * norm_sum) / norm_sq
                edge_gain /= math.sqrt(max(effective_targets, 1.0))
        score += float(context_factor) * float(vector_weight) * 0.5 * edge_gain
    return score


def _existing_qproject_precision(
    *,
    blocks: list[AdaptiveBlock],
    repair_plans: list[QueryRepairPlan | None],
    selected_block_ids: list[int] | None,
    measurement_sigma: float,
    num_queries: int,
) -> np.ndarray:
    precision = np.zeros(int(num_queries), dtype=np.float64)
    if not selected_block_ids:
        return precision
    sigma = float(measurement_sigma)
    for block_id_raw in selected_block_ids:
        block_id = int(block_id_raw)
        plan = repair_plans[block_id]
        if plan is None or plan.target_qids.size == 0:
            continue
        block = blocks[block_id]
        delta = max(float(block.delta_l2), 1.0e-12)
        noise_var = (sigma * delta) ** 2 * plan.noise_counts
        precision[plan.target_qids] += 1.0 / np.maximum(noise_var, 1.0e-12)
    return precision


def _score_blocks_operator_value_of_information(
    scheme: str,
    blocks: list[AdaptiveBlock],
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    measurement_sigma: float,
    repair_plans: list[QueryRepairPlan | None],
    active_query_budget: int = 64,
    kappa_noise: float = 1.0,
    selected_block_ids: list[int] | None = None,
    conditional_context: ConditionalPropagationContext | None = None,
    state_weight_answers: np.ndarray | None = None,
    round_id: int | None = None,
    rounds: int | None = None,
    row_count: int | None = None,
) -> np.ndarray:
    base_scheme = _scheme_base(scheme)
    residual = true_answers.astype(np.float64, copy=False) - syn_answers.astype(np.float64, copy=False)
    before_abs_all = np.abs(residual)
    if base_scheme in {
        "voi_sagebalcvarmrr_harmonic_qproject",
        "voi_sagebalvcvarmrr_harmonic_qproject",
        "voi_sagebalvtopsqrtmrr_harmonic_qproject",
    }:
        tail_scheme = (
            "voi_sagevcvarsqrtmrr_harmonic_qproject"
            if base_scheme == "voi_sagebalvcvarmrr_harmonic_qproject"
            else "voi_sagecvarsqrtmrr_harmonic_qproject"
        )
        if base_scheme == "voi_sagebalvtopsqrtmrr_harmonic_qproject":
            tail_scheme = "voi_sagevtopsqrtmrr_harmonic_qproject"
        local_scores = _score_blocks_operator_value_of_information(
            "voi_sagelocalblockmrr_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        )
        tail_scores = _score_blocks_operator_value_of_information(
            tail_scheme,
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        )
        local_scores = np.maximum(local_scores.astype(np.float64, copy=False), 0.0)
        tail_scores = np.maximum(tail_scores.astype(np.float64, copy=False), 0.0)
        denom = local_scores + tail_scores
        balanced = np.zeros_like(local_scores, dtype=np.float64)
        np.divide(2.0 * local_scores * tail_scores, denom, out=balanced, where=denom > 0.0)
        return 0.25 * balanced
    if base_scheme == "voi_sagelocalagree2of3mrr_harmonic_qproject":
        local_scores = _score_blocks_operator_value_of_information(
            "voi_sagelocalvalidtailmrr_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        ).astype(np.float64, copy=False)
        avg_scores = _score_blocks_operator_value_of_information(
            "voi_sagelocaltailcap75mrr_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        ).astype(np.float64, copy=False)
        flow_scores = _score_blocks_operator_value_of_information(
            "voi_sageflowmrr_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        ).astype(np.float64, copy=False)
        return np.median(np.stack([local_scores, avg_scores, flow_scores], axis=0), axis=0)
    if base_scheme in {
        "voi_sagedpblend10mrr_harmonic_qproject",
        "voi_sagedpblend25mrr_harmonic_qproject",
        "voi_sagedpblend50mrr_harmonic_qproject",
        "voi_sagedpblenddecay5010mrr_harmonic_qproject",
        "voi_sagedpblendramp1050mrr_harmonic_qproject",
        "voi_sagedpblendmax1050mrr_harmonic_qproject",
        "voi_sagedpblendstategatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstateqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstatevgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstategate4rtmrr_harmonic_qproject",
        "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpconfirmfloor25mrr_harmonic_qproject",
        "voi_sagedpconfirmfloor50mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor05mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor10mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor25mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor50mrr_harmonic_qproject",
        "voi_sagedpstageholdfloor05mrr_harmonic_qproject",
        "voi_sagedpstageholdfloor10mrr_harmonic_qproject",
    }:
        if _is_sagedp_stage_floor_scheme(base_scheme):
            if _is_sagedp_stage_hold_floor_scheme(base_scheme):
                stage_diag = _sagedp_stage_hold_diagnostics(
                    blocks,
                    selected_block_ids,
                    state_weight_answers,
                    syn_answers,
                )
            else:
                stage_diag = _sagedp_stage_switch_diagnostics(
                    blocks,
                    selected_block_ids,
                    state_weight_answers,
                    syn_answers,
                )
            if float(stage_diag["sagedp_stage_switch_active"]) <= 0.0:
                return _score_blocks_operator_value_of_information(
                    "voi_sagedpblendstategatesqrtmrr_harmonic_qproject",
                    blocks,
                    true_answers,
                    syn_answers,
                    measurement_sigma,
                    repair_plans,
                    active_query_budget=active_query_budget,
                    kappa_noise=kappa_noise,
                    selected_block_ids=selected_block_ids,
                    conditional_context=conditional_context,
                    state_weight_answers=state_weight_answers,
                    round_id=round_id,
                    rounds=rounds,
                    row_count=row_count,
                )
            floor_alpha = _sagedp_confirmation_floor_alpha(base_scheme)
            local_scores = _score_blocks_operator_value_of_information(
                "voi_sagelocaltailblend04mrr_harmonic_qproject",
                blocks,
                true_answers,
                syn_answers,
                measurement_sigma,
                repair_plans,
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=state_weight_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=row_count,
            ).astype(np.float64, copy=False)
            dp_scores = _score_blocks_operator_value_of_information(
                "voi_dpstatecovsplit_harmonic_qproject",
                blocks,
                true_answers,
                syn_answers,
                measurement_sigma,
                repair_plans,
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=state_weight_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=row_count,
            ).astype(np.float64, copy=False)
            return np.maximum(np.minimum(local_scores, dp_scores), float(floor_alpha) * local_scores)
        if base_scheme == "voi_sagedpblendmax1050mrr_harmonic_qproject":
            tail_scores = _score_blocks_operator_value_of_information(
                "voi_sagedpblend10mrr_harmonic_qproject",
                blocks,
                true_answers,
                syn_answers,
                measurement_sigma,
                repair_plans,
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=state_weight_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=row_count,
            ).astype(np.float64, copy=False)
            avg_scores = _score_blocks_operator_value_of_information(
                "voi_sagedpblend50mrr_harmonic_qproject",
                blocks,
                true_answers,
                syn_answers,
                measurement_sigma,
                repair_plans,
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=state_weight_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=row_count,
            ).astype(np.float64, copy=False)
            return np.maximum(tail_scores, avg_scores)
        if _is_sagedp_confirmation_floor_scheme(base_scheme):
            floor_alpha = _sagedp_confirmation_floor_alpha(base_scheme)
            local_scores = _score_blocks_operator_value_of_information(
                "voi_sagelocaltailblend04mrr_harmonic_qproject",
                blocks,
                true_answers,
                syn_answers,
                measurement_sigma,
                repair_plans,
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=state_weight_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=row_count,
            ).astype(np.float64, copy=False)
            dp_scores = _score_blocks_operator_value_of_information(
                "voi_dpstatecovsplit_harmonic_qproject",
                blocks,
                true_answers,
                syn_answers,
                measurement_sigma,
                repair_plans,
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=state_weight_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=row_count,
            ).astype(np.float64, copy=False)
            return np.maximum(np.minimum(local_scores, dp_scores), float(floor_alpha) * local_scores)
        if base_scheme == "voi_sagedpblend10mrr_harmonic_qproject":
            dp_weight = 0.10
        elif base_scheme == "voi_sagedpblend25mrr_harmonic_qproject":
            dp_weight = 0.25
        elif base_scheme == "voi_sagedpblend50mrr_harmonic_qproject":
            dp_weight = 0.50
        elif base_scheme == "voi_sagedpblenddecay5010mrr_harmonic_qproject":
            total_rounds = max(1, int(rounds or 1))
            current_round = min(max(1, int(round_id or 1)), total_rounds)
            progress = 0.0 if total_rounds <= 1 else float(current_round - 1) / float(total_rounds - 1)
            dp_weight = 0.50 - 0.40 * progress
        elif base_scheme == "voi_sagedpblendramp1050mrr_harmonic_qproject":
            total_rounds = max(1, int(rounds or 1))
            current_round = min(max(1, int(round_id or 1)), total_rounds)
            progress = 0.0 if total_rounds <= 1 else float(current_round - 1) / float(total_rounds - 1)
            dp_weight = 0.10 + 0.40 * progress
        elif base_scheme == "voi_sagedpblendstategate4rtmrr_harmonic_qproject":
            dp_weight = _dp_state_tail_pressure_weight(blocks, state_weight_answers, syn_answers, root_power=0.25)
        elif base_scheme == "voi_sagedpblendstateqvgatesqrtmrr_harmonic_qproject":
            dp_weight = _dp_state_tail_pressure_weight(
                blocks,
                state_weight_answers,
                syn_answers,
                root_power=0.5,
                pressure_mode="geomean",
            )
        elif base_scheme == "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject":
            dp_weight = _dp_state_tail_pressure_weight(
                blocks,
                state_weight_answers,
                syn_answers,
                root_power=0.5,
                pressure_mode="geomean",
            )
        elif base_scheme == "voi_sagedpblendstatevgatesqrtmrr_harmonic_qproject":
            dp_weight = _dp_state_tail_pressure_weight(
                blocks,
                state_weight_answers,
                syn_answers,
                root_power=0.5,
                pressure_mode="vector",
            )
        else:
            dp_weight = _dp_state_tail_pressure_weight(blocks, state_weight_answers, syn_answers)
        local_scores = _score_blocks_operator_value_of_information(
            "voi_sagelocaltailblend04mrr_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        ).astype(np.float64, copy=False)
        dp_scores = _score_blocks_operator_value_of_information(
            "voi_dpstatecovsplit_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        ).astype(np.float64, copy=False)
        if base_scheme == "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject":
            dp_scores = np.minimum(dp_scores, local_scores)
        return (local_scores + dp_weight * dp_scores) / (1.0 + dp_weight)
    if base_scheme in {
        "voi_sagelocaltailblend02mrr_harmonic_qproject",
        "voi_sagelocaltailblend03mrr_harmonic_qproject",
        "voi_sagelocaltailblend04mrr_harmonic_qproject",
        "voi_sagelocaltailblend05mrr_harmonic_qproject",
        "voi_sagelocaltailblend10mrr_harmonic_qproject",
        "voi_sagelocaltailflow010mrr_harmonic_qproject",
        "voi_sagelocaltailflow020mrr_harmonic_qproject",
        "voi_sagelocaltailflow040mrr_harmonic_qproject",
        "voi_sagelocaltailflowmax025mrr_harmonic_qproject",
        "voi_sagelocaltailflowmax050mrr_harmonic_qproject",
        "voi_sagelocaltailflowmax100mrr_harmonic_qproject",
        "voi_sagelocaltailflowenv025mrr_harmonic_qproject",
        "voi_sagelocaltailflowenv050mrr_harmonic_qproject",
        "voi_sagelocaltailflowenv075mrr_harmonic_qproject",
        "voi_sagelocaltailblendfloor98mrr_harmonic_qproject",
        "voi_sagelocaltailblendfloor99mrr_harmonic_qproject",
        "voi_sagelocaltailblendfloor995mrr_harmonic_qproject",
        "voi_sagelocaltailbonus04mrr_harmonic_qproject",
        "voi_sagelocaltailbonus05mrr_harmonic_qproject",
        "voi_sagelocaltailbonus10mrr_harmonic_qproject",
        "voi_sagelocaltailbonus20mrr_harmonic_qproject",
        "voi_sagelocaltailbonus50mrr_harmonic_qproject",
        "voi_sagelocaltailhalf05mrr_harmonic_qproject",
        "voi_sagelocaltaillastq05mrr_harmonic_qproject",
        "voi_sagelocaltailramp05mrr_harmonic_qproject",
    }:
        total_rounds = max(1, int(rounds or 1))
        current_round = min(max(1, int(round_id or 1)), total_rounds)
        if base_scheme == "voi_sagelocaltailblend02mrr_harmonic_qproject":
            avg_weight = 0.02
        elif base_scheme == "voi_sagelocaltailblend03mrr_harmonic_qproject":
            avg_weight = 0.03
        elif base_scheme == "voi_sagelocaltailblend04mrr_harmonic_qproject":
            avg_weight = 0.04
        elif base_scheme == "voi_sagelocaltailblend05mrr_harmonic_qproject":
            avg_weight = 0.05
        elif base_scheme == "voi_sagelocaltailblend10mrr_harmonic_qproject":
            avg_weight = 0.10
        elif base_scheme in {
            "voi_sagelocaltailflow010mrr_harmonic_qproject",
            "voi_sagelocaltailflow020mrr_harmonic_qproject",
            "voi_sagelocaltailflow040mrr_harmonic_qproject",
            "voi_sagelocaltailflowmax025mrr_harmonic_qproject",
            "voi_sagelocaltailflowmax050mrr_harmonic_qproject",
            "voi_sagelocaltailflowmax100mrr_harmonic_qproject",
            "voi_sagelocaltailflowenv025mrr_harmonic_qproject",
            "voi_sagelocaltailflowenv050mrr_harmonic_qproject",
            "voi_sagelocaltailflowenv075mrr_harmonic_qproject",
        }:
            avg_weight = 0.04
        elif base_scheme in {
            "voi_sagelocaltailblendfloor98mrr_harmonic_qproject",
            "voi_sagelocaltailblendfloor99mrr_harmonic_qproject",
            "voi_sagelocaltailblendfloor995mrr_harmonic_qproject",
        }:
            avg_weight = 0.04
        elif base_scheme == "voi_sagelocaltailbonus04mrr_harmonic_qproject":
            avg_weight = 0.04
        elif base_scheme == "voi_sagelocaltailbonus05mrr_harmonic_qproject":
            avg_weight = 0.05
        elif base_scheme == "voi_sagelocaltailbonus10mrr_harmonic_qproject":
            avg_weight = 0.10
        elif base_scheme == "voi_sagelocaltailbonus20mrr_harmonic_qproject":
            avg_weight = 0.20
        elif base_scheme == "voi_sagelocaltailbonus50mrr_harmonic_qproject":
            avg_weight = 0.50
        elif base_scheme == "voi_sagelocaltailhalf05mrr_harmonic_qproject":
            avg_weight = 0.05 if current_round > int(math.ceil(total_rounds / 2.0)) else 0.0
        elif base_scheme == "voi_sagelocaltaillastq05mrr_harmonic_qproject":
            tail_rounds = max(1, int(math.ceil(total_rounds / 4.0)))
            avg_weight = 0.05 if current_round > total_rounds - tail_rounds else 0.0
        else:
            avg_weight = 0.05 if total_rounds <= 1 else 0.05 * float(current_round - 1) / float(total_rounds - 1)
        local_scores = _score_blocks_operator_value_of_information(
            "voi_sagelocalvalidtailmrr_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        )
        avg_scores = _score_blocks_operator_value_of_information(
            "voi_sagelocaltailcap75mrr_harmonic_qproject",
            blocks,
            true_answers,
            syn_answers,
            measurement_sigma,
            repair_plans,
            active_query_budget=active_query_budget,
            kappa_noise=kappa_noise,
            selected_block_ids=selected_block_ids,
            conditional_context=conditional_context,
            state_weight_answers=state_weight_answers,
            round_id=round_id,
            rounds=rounds,
            row_count=row_count,
        )
        local_scores = local_scores.astype(np.float64, copy=False)
        avg_scores = avg_scores.astype(np.float64, copy=False)
        if base_scheme in {
            "voi_sagelocaltailflow010mrr_harmonic_qproject",
            "voi_sagelocaltailflow020mrr_harmonic_qproject",
            "voi_sagelocaltailflow040mrr_harmonic_qproject",
            "voi_sagelocaltailflowmax025mrr_harmonic_qproject",
            "voi_sagelocaltailflowmax050mrr_harmonic_qproject",
            "voi_sagelocaltailflowmax100mrr_harmonic_qproject",
            "voi_sagelocaltailflowenv025mrr_harmonic_qproject",
            "voi_sagelocaltailflowenv050mrr_harmonic_qproject",
            "voi_sagelocaltailflowenv075mrr_harmonic_qproject",
        }:
            if base_scheme == "voi_sagelocaltailflow010mrr_harmonic_qproject":
                flow_weight = 0.01
            elif base_scheme == "voi_sagelocaltailflow020mrr_harmonic_qproject":
                flow_weight = 0.02
            elif base_scheme == "voi_sagelocaltailflow040mrr_harmonic_qproject":
                flow_weight = 0.04
            elif base_scheme == "voi_sagelocaltailflowmax025mrr_harmonic_qproject":
                flow_weight = 0.25
            elif base_scheme == "voi_sagelocaltailflowmax050mrr_harmonic_qproject":
                flow_weight = 0.50
            elif base_scheme == "voi_sagelocaltailflowmax100mrr_harmonic_qproject":
                flow_weight = 1.00
            elif base_scheme == "voi_sagelocaltailflowenv025mrr_harmonic_qproject":
                flow_weight = 0.25
            elif base_scheme == "voi_sagelocaltailflowenv050mrr_harmonic_qproject":
                flow_weight = 0.50
            else:
                flow_weight = 0.75
            flow_scores = _score_blocks_operator_value_of_information(
                "voi_sageflowmrr_harmonic_qproject",
                blocks,
                true_answers,
                syn_answers,
                measurement_sigma,
                repair_plans,
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=state_weight_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=row_count,
            )
            flow_scores = flow_scores.astype(np.float64, copy=False)
            if "localtailflowmax" in base_scheme:
                blended_scores = (local_scores + avg_weight * avg_scores) / (1.0 + avg_weight)
                return np.maximum(blended_scores, flow_weight * flow_scores)
            if "localtailflowenv" in base_scheme:
                blended_scores = (local_scores + avg_weight * avg_scores) / (1.0 + avg_weight)
                return np.maximum(flow_weight * blended_scores, flow_scores)
            return (local_scores + avg_weight * avg_scores + flow_weight * flow_scores) / (
                1.0 + avg_weight + flow_weight
            )
        if "localtailbonus" in base_scheme:
            return (1.0 - avg_weight) * local_scores + avg_weight * np.maximum(local_scores, avg_scores)
        if base_scheme == "voi_sagelocaltailblendfloor98mrr_harmonic_qproject":
            local_floor = 0.98
        elif base_scheme == "voi_sagelocaltailblendfloor99mrr_harmonic_qproject":
            local_floor = 0.99
        elif base_scheme == "voi_sagelocaltailblendfloor995mrr_harmonic_qproject":
            local_floor = 0.995
        else:
            local_floor = None
        if local_floor is not None:
            blended_scores = (local_scores + avg_weight * avg_scores) / (1.0 + avg_weight)
            return np.maximum(blended_scores, local_floor * local_scores)
        return (local_scores + avg_weight * avg_scores) / (1.0 + avg_weight)
    vector_block_ids = [
        idx
        for idx, candidate in enumerate(blocks)
        if candidate.is_vector and int(candidate.query_indices.size) > 1
    ]
    before_vector_tvd = np.asarray(
        [0.5 * float(np.sum(before_abs_all[blocks[idx].query_indices])) for idx in vector_block_ids],
        dtype=np.float64,
    )
    scores = np.zeros(len(blocks), dtype=np.float64)
    rank_weights: np.ndarray | None = None
    atom_query_weights: np.ndarray | None = None
    atom_vector_block_ids: list[int] = []
    atom_vector_weights: np.ndarray | None = None
    atom_query_leverage: np.ndarray | None = None
    atom_vector_local_weights: dict[int, np.ndarray] | None = None
    dual_query_weights: np.ndarray | None = None
    dual_vector_block_ids: list[int] = []
    dual_vector_weights: np.ndarray | None = None
    dual_vector_local_weights: list[np.ndarray] | None = None
    dual_query_membership_split: np.ndarray | None = None
    dual_query_leverage: np.ndarray | None = None
    dual_query_risk = 1.0
    dual_vector_risk = 1.0
    dual_query_sensitivity = 1.0
    dual_vector_sensitivity = 1.0
    dual_query_tail_risk = 1.0
    dual_vector_tail_risk = 1.0
    rrc_delta_eff: np.ndarray | None = None
    mixed_query_weights: np.ndarray | None = None
    mixed_vector_weights: np.ndarray | None = None
    capacity_risk_context: CapacityRiskContext | None = None
    edge_flow_risk_context: EdgeFlowRiskContext | None = None
    edge_flow_before_risk = 0.0
    if base_scheme == "voi_rankop_sqrt_qproject":
        rank_weights = _rank_spectral_weights(residual, "sqrt")
    elif base_scheme == "voi_rankop_harmonic_qproject":
        rank_weights = _rank_spectral_weights(residual, "harmonic")
    elif base_scheme == "voi_rankop_log_qproject":
        rank_weights = _rank_spectral_weights(residual, "log")
    elif base_scheme in {
        "voi_atomop_harmonic_qproject",
        "voi_atomlev_harmonic_qproject",
        "voi_atomloclev_harmonic_qproject",
        "voi_genactive_atomlev_harmonic_qproject",
        "voi_genexposure_atomlev_harmonic_qproject",
        "voi_graphctx_atomlev_harmonic_qproject",
        "voi_condprop_atomlev_harmonic_qproject",
        "voi_condctx_atomlev_harmonic_qproject",
        "voi_condbridge_atomlev_harmonic_qproject",
        "voi_condnorm_atomlev_harmonic_qproject",
        "voi_condbridgenorm_atomlev_harmonic_qproject",
        "voi_precinnov_atomlev_harmonic_qproject",
        "voi_distanchor_atomlev_harmonic_qproject",
        "voi_atomtaillev_harmonic_qproject",
        "voi_atomenvlev_harmonic_qproject",
    }:
        if base_scheme == "voi_atomtaillev_harmonic_qproject":
            atom_query_weights, atom_vector_block_ids, atom_vector_weights = _atom_tail_spectral_weights(
                residual, blocks, "harmonic"
            )
            atom_vector_local_weights = _vector_local_spectral_weights(
                residual, blocks, atom_vector_block_ids, "harmonic"
            )
        elif base_scheme == "voi_atomenvlev_harmonic_qproject":
            (
                atom_query_weights,
                atom_vector_block_ids,
                atom_vector_weights,
                atom_vector_local_weights,
            ) = _atom_envelope_spectral_weights(residual, blocks, "harmonic")
        else:
            atom_query_weights, atom_vector_block_ids, atom_vector_weights = _atom_spectral_weights(
                residual, blocks, "harmonic"
            )
        if base_scheme in {
            "voi_atomlev_harmonic_qproject",
            "voi_atomloclev_harmonic_qproject",
            "voi_genactive_atomlev_harmonic_qproject",
            "voi_genexposure_atomlev_harmonic_qproject",
            "voi_graphctx_atomlev_harmonic_qproject",
            "voi_condprop_atomlev_harmonic_qproject",
            "voi_condctx_atomlev_harmonic_qproject",
            "voi_condbridge_atomlev_harmonic_qproject",
            "voi_condnorm_atomlev_harmonic_qproject",
            "voi_condbridgenorm_atomlev_harmonic_qproject",
            "voi_precinnov_atomlev_harmonic_qproject",
            "voi_distanchor_atomlev_harmonic_qproject",
            "voi_atomtaillev_harmonic_qproject",
            "voi_atomenvlev_harmonic_qproject",
        }:
            atom_query_leverage = _query_repair_leverage(repair_plans, residual.size)
    elif base_scheme in {
        "voi_dualop_harmonic_qproject",
        "voi_nashop_harmonic_qproject",
        "voi_relmaxop_harmonic_qproject",
        "voi_relmaxscaled_harmonic_qproject",
        "voi_relmaxposscaled_harmonic_qproject",
        "voi_relmaxposeff_harmonic_qproject",
        "voi_relmaxposnorm_harmonic_qproject",
        "voi_dpstateposnorm_harmonic_qproject",
        "voi_dpstatecovsplit_harmonic_qproject",
        "voi_dpstatesqrtcovsplit_harmonic_qproject",
        "voi_dpstatesplittail_harmonic_qproject",
        "voi_dpstatecovtail_harmonic_qproject",
        "voi_dpstatebreadthtail_harmonic_qproject",
        "voi_dpstateriskmaxmrr_harmonic_qproject",
        "voi_dpstateriskenv025mrr_harmonic_qproject",
        "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
        "voi_sageordergain_harmonic_qproject",
        "voi_sagemrr_harmonic_qproject",
        "voi_sageroutecmrr_harmonic_qproject",
        "voi_sagecovmrr_harmonic_qproject",
        "voi_sagesqcovmrr_harmonic_qproject",
        "voi_sagesoftcovmrr_harmonic_qproject",
        "voi_sagebreadthmrr_harmonic_qproject",
        "voi_sageblockmrr_harmonic_qproject",
        "voi_sagelocalblockmrr_harmonic_qproject",
        "voi_sagelocalvalidtailmrr_harmonic_qproject",
        "voi_sagelocaltailcap65mrr_harmonic_qproject",
        "voi_sagelocaltailcap75mrr_harmonic_qproject",
        "voi_sagelocaltailcap90mrr_harmonic_qproject",
        "voi_sagelocaltailcap95mrr_harmonic_qproject",
        "voi_sagenrlocalmrr_harmonic_qproject",
        "voi_sagecellblockmrr_harmonic_qproject",
        "voi_sagegraphcvarmrr_harmonic_qproject",
        "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
        "voi_sageenv025mrr_harmonic_qproject",
        "voi_sageenv05mrr_harmonic_qproject",
        "voi_sageenv1mrr_harmonic_qproject",
        "voi_sagecvar4mrr_harmonic_qproject",
        "voi_sagecvarsqrtmrr_harmonic_qproject",
        "voi_sagevcvarsqrtmrr_harmonic_qproject",
        "voi_sagevtopsqrtmrr_harmonic_qproject",
        "voi_sagestudcvarmrr_harmonic_qproject",
        "voi_sagespectralmrr_harmonic_qproject",
        "voi_sageholder15mrr_harmonic_qproject",
        "voi_sageholder2mrr_harmonic_qproject",
        "voi_sagenestedcvarmrr_harmonic_qproject",
        "voi_sagehiercvarmrr_harmonic_qproject",
        "voi_sagerepaircap10mrr_harmonic_qproject",
        "voi_sagerepaircap25mrr_harmonic_qproject",
        "voi_sagerepaircap50mrr_harmonic_qproject",
        "voi_sageflowmrr_harmonic_qproject",
        "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagerrcvalidfullmrr_harmonic_qproject",
                "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
                "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
        "voi_sagemetricavgmrr_harmonic_qproject",
        "voi_sagemetricenvmrr_harmonic_qproject",
        "voi_sagefloorcapmrr_harmonic_qproject",
        "voi_sageparetomrr_harmonic_qproject",
        "voi_sageharmmrr_harmonic_qproject",
        "voi_relmaxsqrtlevscaled_harmonic_qproject",
        "voi_relmaxlevscaled_harmonic_qproject",
    }:
        if base_scheme in {
            "voi_dpstateposnorm_harmonic_qproject",
            "voi_dpstatecovsplit_harmonic_qproject",
            "voi_dpstatesqrtcovsplit_harmonic_qproject",
            "voi_dpstatesplittail_harmonic_qproject",
            "voi_dpstatecovtail_harmonic_qproject",
            "voi_dpstatebreadthtail_harmonic_qproject",
            "voi_dpstateriskmaxmrr_harmonic_qproject",
            "voi_dpstateriskenv025mrr_harmonic_qproject",
            "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
        }:
            if state_weight_answers is None:
                state_residual_for_weights = None
                dual_query_weights = _public_harmonic_weights(residual.size)
                dual_vector_block_ids, dual_vector_weights = _vector_public_harmonic_weights(blocks)
            else:
                state_residual = state_weight_answers.astype(np.float64, copy=False) - syn_answers.astype(
                    np.float64, copy=False
                )
                state_residual_for_weights = state_residual
                dual_query_weights = _rank_spectral_weights(state_residual, "harmonic")
                dual_vector_block_ids, dual_vector_weights = _vector_spectral_weights(state_residual, blocks, "harmonic")
            if base_scheme in {
                "voi_dpstatecovsplit_harmonic_qproject",
                "voi_dpstatesqrtcovsplit_harmonic_qproject",
                "voi_dpstatesplittail_harmonic_qproject",
                "voi_dpstatecovtail_harmonic_qproject",
                "voi_dpstatebreadthtail_harmonic_qproject",
                "voi_dpstateriskmaxmrr_harmonic_qproject",
                "voi_dpstateriskenv025mrr_harmonic_qproject",
                "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
            }:
                dual_vector_local_weights = _vector_local_harmonic_weights(
                    state_residual_for_weights,
                    blocks,
                    dual_vector_block_ids,
                )
                dual_query_membership_split = _query_vector_membership_split(
                    blocks,
                    dual_vector_block_ids,
                    residual.size,
                )
        else:
            dual_query_weights = _rank_spectral_weights(residual, "harmonic")
            dual_vector_block_ids, dual_vector_weights = _vector_spectral_weights(residual, blocks, "harmonic")
        if base_scheme == "voi_relmaxsqrtlevscaled_harmonic_qproject":
            dual_query_leverage = np.sqrt(_query_repair_leverage(repair_plans, residual.size))
            dual_query_weights = dual_query_weights * dual_query_leverage
        elif base_scheme == "voi_relmaxlevscaled_harmonic_qproject":
            dual_query_leverage = _query_repair_leverage(repair_plans, residual.size)
            dual_query_weights = dual_query_weights * dual_query_leverage
        if base_scheme == "voi_relmaxposeff_harmonic_qproject":
            rrc_delta_eff = _rrc_effective_sensitivities(
                blocks,
                repair_plans,
                dual_vector_block_ids,
                residual.size,
            )
        elif base_scheme in {
            "voi_relmaxposnorm_harmonic_qproject",
            "voi_dpstateposnorm_harmonic_qproject",
            "voi_dpstatecovsplit_harmonic_qproject",
            "voi_dpstatesqrtcovsplit_harmonic_qproject",
            "voi_dpstatesplittail_harmonic_qproject",
            "voi_dpstatecovtail_harmonic_qproject",
            "voi_dpstatebreadthtail_harmonic_qproject",
            "voi_dpstateriskmaxmrr_harmonic_qproject",
            "voi_dpstateriskenv025mrr_harmonic_qproject",
            "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
            "voi_sageordergain_harmonic_qproject",
            "voi_sagemrr_harmonic_qproject",
            "voi_sageroutecmrr_harmonic_qproject",
            "voi_sagecovmrr_harmonic_qproject",
            "voi_sagesqcovmrr_harmonic_qproject",
            "voi_sagesoftcovmrr_harmonic_qproject",
            "voi_sagebreadthmrr_harmonic_qproject",
            "voi_sageblockmrr_harmonic_qproject",
            "voi_sagelocalblockmrr_harmonic_qproject",
            "voi_sagelocalvalidtailmrr_harmonic_qproject",
            "voi_sagelocaltailcap65mrr_harmonic_qproject",
            "voi_sagelocaltailcap75mrr_harmonic_qproject",
            "voi_sagelocaltailcap90mrr_harmonic_qproject",
            "voi_sagelocaltailcap95mrr_harmonic_qproject",
            "voi_sagenrlocalmrr_harmonic_qproject",
            "voi_sagecellblockmrr_harmonic_qproject",
            "voi_sagegraphcvarmrr_harmonic_qproject",
            "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
            "voi_sageenv025mrr_harmonic_qproject",
            "voi_sageenv05mrr_harmonic_qproject",
            "voi_sageenv1mrr_harmonic_qproject",
            "voi_sagecvar4mrr_harmonic_qproject",
            "voi_sagecvarsqrtmrr_harmonic_qproject",
            "voi_sagevcvarsqrtmrr_harmonic_qproject",
            "voi_sagevtopsqrtmrr_harmonic_qproject",
            "voi_sagestudcvarmrr_harmonic_qproject",
            "voi_sagespectralmrr_harmonic_qproject",
            "voi_sageholder15mrr_harmonic_qproject",
            "voi_sageholder2mrr_harmonic_qproject",
            "voi_sagenestedcvarmrr_harmonic_qproject",
            "voi_sagehiercvarmrr_harmonic_qproject",
            "voi_sagerepaircap10mrr_harmonic_qproject",
            "voi_sagerepaircap25mrr_harmonic_qproject",
            "voi_sagerepaircap50mrr_harmonic_qproject",
            "voi_sageflowmrr_harmonic_qproject",
            "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagerrcvalidfullmrr_harmonic_qproject",
                "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
                "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
            "voi_sagemetricavgmrr_harmonic_qproject",
            "voi_sagemetricenvmrr_harmonic_qproject",
            "voi_sagefloorcapmrr_harmonic_qproject",
            "voi_sageparetomrr_harmonic_qproject",
            "voi_sageharmmrr_harmonic_qproject",
        }:
            dual_query_sensitivity = max(_harmonic_number(residual.size), 1.0e-12)
            # Vector workloads are partition blocks in this ablation, so a row
            # changes each block TVD by at most 1/2.
            dual_vector_sensitivity = max(0.5 * _harmonic_number(len(dual_vector_block_ids)), 1.0e-12)
            if base_scheme == "voi_sagecapdromrr_harmonic_qproject":
                capacity_risk_context = _build_capacity_risk_context(
                    blocks,
                    dual_vector_block_ids,
                    residual.size,
                )
            elif base_scheme == "voi_sageflowmrr_harmonic_qproject":
                edge_flow_risk_context = _build_edge_flow_risk_context(
                    blocks,
                    repair_plans,
                    dual_vector_block_ids,
                    residual.size,
                )
                edge_flow_before_risk = _edge_flow_capacity_risk(before_abs_all, edge_flow_risk_context)
            elif base_scheme == "voi_sageqvcalflowguarddromrr_harmonic_qproject":
                edge_flow_risk_context = _build_edge_flow_risk_context(
                    blocks,
                    repair_plans,
                    dual_vector_block_ids,
                    residual.size,
                )
            elif base_scheme == "voi_sagegraphcapmrr_harmonic_qproject":
                edge_flow_risk_context = _build_graph_capacity_risk_context(
                    blocks,
                    repair_plans,
                    dual_vector_block_ids,
                    residual.size,
                )
                edge_flow_before_risk = _edge_flow_capacity_risk(before_abs_all, edge_flow_risk_context)
        dual_query_risk = max(float(np.sum(dual_query_weights * before_abs_all)), 1.0e-12)
        dual_vector_risk = max(
            float(
                sum(
                    float(weight) * 0.5 * float(np.sum(before_abs_all[blocks[target_block_id].query_indices]))
                    for target_block_id, weight in zip(dual_vector_block_ids, dual_vector_weights)
                )
            ),
            1.0e-12,
        )
        if base_scheme in {
        "voi_sageblockmrr_harmonic_qproject",
        "voi_sagelocalblockmrr_harmonic_qproject",
        "voi_sagelocalvalidtailmrr_harmonic_qproject",
        "voi_sagelocaltailcap65mrr_harmonic_qproject",
        "voi_sagelocaltailcap75mrr_harmonic_qproject",
        "voi_sagelocaltailcap90mrr_harmonic_qproject",
        "voi_sagelocaltailcap95mrr_harmonic_qproject",
        "voi_sagenrlocalmrr_harmonic_qproject",
            "voi_sagecellblockmrr_harmonic_qproject",
        }:
            dual_vector_risk = max(
                _harmonic_ordered_sum(
                    np.asarray(
                        [
                            0.5
                            * _harmonic_ordered_sum(
                                before_abs_all[blocks[target_block_id].query_indices],
                            )
                            for target_block_id in dual_vector_block_ids
                        ],
                        dtype=np.float64,
                    )
                ),
                1.0e-12,
            )
        if base_scheme in {
            "voi_sagegraphcvarmrr_harmonic_qproject",
            "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
        }:
            graph_risks = np.asarray(
                [
                    _topk_mean(
                        before_abs_all[blocks[target_block_id].query_indices],
                        _cvar_tail_k(base_scheme, int(blocks[target_block_id].query_indices.size)),
                    )
                    for target_block_id in dual_vector_block_ids
                ],
                dtype=np.float64,
            )
            dual_vector_risk = max(float(np.max(graph_risks)) if graph_risks.size else 0.0, 1.0e-12)
            dual_vector_sensitivity = 1.0
        if base_scheme in {
            "voi_sagerrcvalidfullmrr_harmonic_qproject",
            "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
            "voi_sagerrcvalidtopkmrr_harmonic_qproject",
            "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
            "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
            "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
            "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
            "voi_sagerrcnestedvalidmrr_harmonic_qproject",
            "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
            "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
            "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
            "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
        }:
            vector_tvd_values = np.asarray(
                [
                    0.5 * float(np.sum(before_abs_all[blocks[target_block_id].query_indices]))
                    for target_block_id in dual_vector_block_ids
                ],
                dtype=np.float64,
            )
            if base_scheme in {
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
                "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
            }:
                if base_scheme == "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject":
                    k = _cvar_tail_k(base_scheme, vector_tvd_values.size)
                    risk_value = _head_topk_mean(vector_tvd_values, k)
                elif base_scheme == "voi_sagerrcvalidharmgainmrr_harmonic_qproject":
                    risk_value = _harmonic_ordered_mean(vector_tvd_values)
                elif base_scheme == "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject":
                    risk_value = _lorenz_head_broad_mean(vector_tvd_values)
                elif base_scheme == "voi_sagerrcnestedvalidmrr_harmonic_qproject":
                    risk_value = float(np.max(vector_tvd_values)) if vector_tvd_values.size else 0.0
                elif base_scheme == "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject":
                    risk_value = float(np.max(vector_tvd_values)) if vector_tvd_values.size else 0.0
                elif base_scheme == "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject":
                    risk_value = float(np.max(vector_tvd_values)) if vector_tvd_values.size else 0.0
                elif base_scheme == "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject":
                    risk_value = _lorenz_head_broad_mean(vector_tvd_values)
                elif base_scheme == "voi_sagerrcnestedparetovalidmrr_harmonic_qproject":
                    risk_value = float(np.max(vector_tvd_values)) if vector_tvd_values.size else 0.0
                else:
                    k = _cvar_tail_k(base_scheme, vector_tvd_values.size)
                    risk_value = _topk_mean(vector_tvd_values, k)
            else:
                risk_value = float(np.max(vector_tvd_values)) if vector_tvd_values.size else 0.0
            dual_vector_risk = max(risk_value, 1.0e-12)
            dual_vector_sensitivity = 0.5
        if base_scheme in {
            "voi_dpstatecovsplit_harmonic_qproject",
            "voi_dpstatesqrtcovsplit_harmonic_qproject",
            "voi_dpstatesplittail_harmonic_qproject",
            "voi_dpstatecovtail_harmonic_qproject",
            "voi_dpstatebreadthtail_harmonic_qproject",
        }:
            if dual_vector_local_weights is None or dual_query_membership_split is None:
                raise AssertionError("split vector weights must be initialized")
            dual_vector_risk = max(
                float(
                    sum(
                        float(block_weight)
                        * 0.5
                        * float(
                            np.sum(
                                local_weights
                                * dual_query_membership_split[blocks[target_block_id].query_indices]
                                * before_abs_all[blocks[target_block_id].query_indices]
                            )
                        )
                        for target_block_id, block_weight, local_weights in zip(
                            dual_vector_block_ids,
                            dual_vector_weights,
                            dual_vector_local_weights,
                        )
                    )
                ),
                1.0e-12,
            )
        if base_scheme in {
            "voi_dpstatesplittail_harmonic_qproject",
            "voi_dpstatecovtail_harmonic_qproject",
            "voi_dpstatebreadthtail_harmonic_qproject",
        }:
            dual_query_tail_risk = max(float(np.max(before_abs_all)), 1.0e-12)
            dual_vector_tail_risk = max(
                float(
                    max(
                        0.5 * float(np.sum(before_abs_all[blocks[target_block_id].query_indices]))
                        for target_block_id in dual_vector_block_ids
                    )
                )
                if dual_vector_block_ids
                else 0.0,
                1.0e-12,
            )
    elif base_scheme in {
        "voi_mixlp2_qproject",
        "voi_mixlp4_qproject",
        "voi_mixlp8_qproject",
    }:
        mixed_query_weights, mixed_vector_weights = _mixed_lp_risk_weights(
            residual,
            blocks,
            vector_block_ids,
            before_vector_tvd,
            _mixed_lp_order(base_scheme),
        )
    existing_precision = (
        _existing_qproject_precision(
            blocks=blocks,
            repair_plans=repair_plans,
            selected_block_ids=selected_block_ids,
            measurement_sigma=measurement_sigma,
            num_queries=residual.size,
        )
        if base_scheme == "voi_precinnov_atomlev_harmonic_qproject"
        else None
    )
    context_degrees = (
        _measured_context_degrees(blocks, selected_block_ids)
        if base_scheme
        in {
            "voi_graphctx_atomlev_harmonic_qproject",
            "voi_condctx_atomlev_harmonic_qproject",
            "voi_condbridge_atomlev_harmonic_qproject",
            "voi_condbridgenorm_atomlev_harmonic_qproject",
        }
        else None
    )
    worst_query_qid = int(np.argmax(before_abs_all)) if before_abs_all.size else 0
    worst_query_risk = max(float(before_abs_all[worst_query_qid]), 1.0e-12) if before_abs_all.size else 1.0
    worst_vector_pos = int(np.argmax(before_vector_tvd)) if before_vector_tvd.size else 0
    worst_vector_block_id = vector_block_ids[worst_vector_pos] if vector_block_ids else 0
    worst_vector_risk = max(float(before_vector_tvd[worst_vector_pos]), 1.0e-12) if before_vector_tvd.size else 1.0
    for block_id, block in enumerate(blocks):
        plan = repair_plans[block_id]
        if plan is None or plan.target_qids.size == 0:
            continue
        source_residual = residual[block.query_indices]
        if base_scheme == "voi_genactive_atomlev_harmonic_qproject":
            source_repair_signal = _generator_reachable_source_signal(
                source_residual,
                measurement_sigma=measurement_sigma,
                delta_l2=float(block.delta_l2),
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                mode="hard",
            )
        elif base_scheme == "voi_genexposure_atomlev_harmonic_qproject":
            source_repair_signal = _generator_reachable_source_signal(
                source_residual,
                measurement_sigma=measurement_sigma,
                delta_l2=float(block.delta_l2),
                active_query_budget=active_query_budget,
                kappa_noise=kappa_noise,
                mode="exposure",
            )
        else:
            source_repair_signal = source_residual
        projected_repair = np.asarray([float(np.sum(source_repair_signal[cells])) for cells in plan.cell_indices])
        target_residual = residual[plan.target_qids]
        delta = max(float(block.delta_l2), 1.0e-12)
        noise_var = (float(measurement_sigma) * delta) ** 2 * plan.noise_counts
        if base_scheme == "voi_absop_qproject":
            before = np.abs(target_residual)
            after = _expected_abs_normal(target_residual - projected_repair, np.sqrt(noise_var))
            gain = np.maximum(before - after, 0.0)
            scores[block_id] = float(np.sum(gain) / delta)
        elif base_scheme == "voi_absdiff_qproject":
            before = np.abs(target_residual)
            after = _expected_abs_normal(target_residual - projected_repair, np.sqrt(noise_var))
            scores[block_id] = float(np.sum(before - after) / delta)
        elif base_scheme == "voi_sqop_qproject":
            gain = target_residual * target_residual - ((target_residual - projected_repair) ** 2 + noise_var)
            scores[block_id] = float(np.sum(np.maximum(gain, 0.0)) / (delta * delta))
        elif base_scheme == "voi_sqdiff_qproject":
            gain = target_residual * target_residual - ((target_residual - projected_repair) ** 2 + noise_var)
            scores[block_id] = float(np.sum(gain) / (delta * delta))
        elif base_scheme in {
            "voi_rankop_sqrt_qproject",
            "voi_rankop_harmonic_qproject",
            "voi_rankop_log_qproject",
        }:
            if rank_weights is None:
                raise AssertionError("rank weights must be initialized for rank operator VOI")
            before = np.abs(target_residual)
            after = _expected_abs_normal(target_residual - projected_repair, np.sqrt(noise_var))
            scores[block_id] = float(np.sum(rank_weights[plan.target_qids] * (before - after)) / delta)
        elif base_scheme in {
            "voi_atomop_harmonic_qproject",
            "voi_atomlev_harmonic_qproject",
            "voi_atomloclev_harmonic_qproject",
            "voi_genactive_atomlev_harmonic_qproject",
            "voi_genexposure_atomlev_harmonic_qproject",
            "voi_graphctx_atomlev_harmonic_qproject",
            "voi_condprop_atomlev_harmonic_qproject",
            "voi_condctx_atomlev_harmonic_qproject",
            "voi_condbridge_atomlev_harmonic_qproject",
            "voi_condnorm_atomlev_harmonic_qproject",
            "voi_condbridgenorm_atomlev_harmonic_qproject",
            "voi_precinnov_atomlev_harmonic_qproject",
            "voi_distanchor_atomlev_harmonic_qproject",
            "voi_atomtaillev_harmonic_qproject",
            "voi_atomenvlev_harmonic_qproject",
        }:
            if atom_query_weights is None or atom_vector_weights is None:
                raise AssertionError("atom weights must be initialized for atom operator VOI")
            before = np.abs(target_residual)
            after = _expected_abs_normal(target_residual - projected_repair, np.sqrt(noise_var))
            gain = before - after
            if base_scheme == "voi_precinnov_atomlev_harmonic_qproject":
                if existing_precision is None:
                    raise AssertionError("existing precision must be initialized for precision-innovation VOI")
                candidate_precision = 1.0 / np.maximum(noise_var, 1.0e-12)
                prior_precision = existing_precision[plan.target_qids]
                gain = gain * (
                    candidate_precision / np.maximum(candidate_precision + prior_precision, 1.0e-12)
                )
            query_weights = atom_query_weights[plan.target_qids]
            if base_scheme in {
                "voi_atomlev_harmonic_qproject",
                "voi_atomloclev_harmonic_qproject",
                "voi_genactive_atomlev_harmonic_qproject",
                "voi_genexposure_atomlev_harmonic_qproject",
                "voi_graphctx_atomlev_harmonic_qproject",
                "voi_condprop_atomlev_harmonic_qproject",
                "voi_condctx_atomlev_harmonic_qproject",
                "voi_condbridge_atomlev_harmonic_qproject",
                "voi_condnorm_atomlev_harmonic_qproject",
                "voi_condbridgenorm_atomlev_harmonic_qproject",
                "voi_precinnov_atomlev_harmonic_qproject",
                "voi_distanchor_atomlev_harmonic_qproject",
                "voi_atomtaillev_harmonic_qproject",
                "voi_atomenvlev_harmonic_qproject",
            }:
                if atom_query_leverage is None:
                    raise AssertionError("query leverage must be initialized for leverage-calibrated VOI")
                query_weights = query_weights * atom_query_leverage[plan.target_qids]
            if base_scheme == "voi_atomloclev_harmonic_qproject":
                direct_mask = np.isin(plan.target_qids, block.query_indices)
                score = float(np.sum(query_weights[direct_mask] * gain[direct_mask]))
                score += float(block.coverage_weight) * float(np.sum(query_weights[~direct_mask] * gain[~direct_mask]))
            else:
                score = float(np.sum(query_weights * gain))
            gain_by_qid = np.zeros_like(residual, dtype=np.float64)
            gain_by_qid[plan.target_qids] = gain
            for vector_weight, target_block_id in zip(atom_vector_weights, atom_vector_block_ids):
                target_idx = blocks[target_block_id].query_indices
                if base_scheme in {
                    "voi_atomtaillev_harmonic_qproject",
                    "voi_atomenvlev_harmonic_qproject",
                }:
                    if atom_vector_local_weights is None:
                        raise AssertionError("vector local weights must be initialized for tail VOI")
                    score += float(vector_weight) * float(
                        np.sum(atom_vector_local_weights[target_block_id] * gain_by_qid[target_idx])
                    )
                else:
                    vector_gain = float(vector_weight) * 0.5 * float(np.sum(gain_by_qid[target_idx]))
                    if base_scheme == "voi_atomloclev_harmonic_qproject" and target_block_id != block_id:
                        vector_gain *= float(block.coverage_weight)
                    if base_scheme == "voi_condctx_atomlev_harmonic_qproject":
                        vector_gain *= _graph_context_multiplier(blocks[target_block_id], context_degrees)
                    elif base_scheme == "voi_condbridge_atomlev_harmonic_qproject":
                        vector_gain *= _graph_context_multiplier(block, context_degrees)
                    score += vector_gain
            if base_scheme in {
                "voi_condprop_atomlev_harmonic_qproject",
                "voi_condctx_atomlev_harmonic_qproject",
                "voi_condbridge_atomlev_harmonic_qproject",
                "voi_condnorm_atomlev_harmonic_qproject",
                "voi_condbridgenorm_atomlev_harmonic_qproject",
                "voi_precinnov_atomlev_harmonic_qproject",
            }:
                source_context = _graph_context_multiplier(block, context_degrees)
                score += _conditional_vector_propagation_score(
                    block_id=block_id,
                    block=block,
                    source_repair_signal=source_repair_signal,
                    residual=residual,
                    measurement_sigma=measurement_sigma,
                    exact_target_qids=plan.target_qids,
                    blocks=blocks,
                    context=conditional_context,
                    atom_vector_block_ids=atom_vector_block_ids,
                    atom_vector_weights=atom_vector_weights,
                    target_context_degrees=context_degrees,
                    target_context_excess_only=base_scheme == "voi_condctx_atomlev_harmonic_qproject",
                    source_context_factor=source_context,
                    bridge_context_excess_only=base_scheme
                    in {
                        "voi_condbridge_atomlev_harmonic_qproject",
                        "voi_condbridgenorm_atomlev_harmonic_qproject",
                    },
                    normalize_effective_targets=base_scheme
                    in {
                        "voi_condnorm_atomlev_harmonic_qproject",
                        "voi_condbridgenorm_atomlev_harmonic_qproject",
                        "voi_precinnov_atomlev_harmonic_qproject",
                    },
                    existing_precision=existing_precision,
                    precision_innovation=base_scheme == "voi_precinnov_atomlev_harmonic_qproject",
                )
            candidate_context = (
                _graph_context_multiplier(block, context_degrees)
                if base_scheme == "voi_graphctx_atomlev_harmonic_qproject"
                else 1.0
            )
            if base_scheme == "voi_distanchor_atomlev_harmonic_qproject":
                anchor_value = _vector_distribution_anchor_value(block, residual, measurement_sigma)
                if anchor_value > 0.0 and block_id in atom_vector_block_ids:
                    vector_pos = atom_vector_block_ids.index(block_id)
                    score += float(atom_vector_weights[vector_pos]) * anchor_value
            scores[block_id] = float(score / delta) * candidate_context
        elif base_scheme in {
            "voi_dualop_harmonic_qproject",
            "voi_nashop_harmonic_qproject",
            "voi_relmaxop_harmonic_qproject",
            "voi_relmaxscaled_harmonic_qproject",
            "voi_relmaxposscaled_harmonic_qproject",
            "voi_relmaxposeff_harmonic_qproject",
            "voi_relmaxposnorm_harmonic_qproject",
            "voi_dpstateposnorm_harmonic_qproject",
            "voi_dpstatecovsplit_harmonic_qproject",
            "voi_dpstatesqrtcovsplit_harmonic_qproject",
            "voi_dpstatesplittail_harmonic_qproject",
            "voi_dpstatecovtail_harmonic_qproject",
            "voi_dpstatebreadthtail_harmonic_qproject",
            "voi_dpstateriskmaxmrr_harmonic_qproject",
            "voi_dpstateriskenv025mrr_harmonic_qproject",
            "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
            "voi_sageordergain_harmonic_qproject",
            "voi_sagemrr_harmonic_qproject",
            "voi_sageroutecmrr_harmonic_qproject",
            "voi_sagecovmrr_harmonic_qproject",
            "voi_sagesqcovmrr_harmonic_qproject",
            "voi_sagesoftcovmrr_harmonic_qproject",
            "voi_sagebreadthmrr_harmonic_qproject",
            "voi_sageblockmrr_harmonic_qproject",
            "voi_sagelocalblockmrr_harmonic_qproject",
            "voi_sagelocalvalidtailmrr_harmonic_qproject",
            "voi_sagelocaltailcap65mrr_harmonic_qproject",
            "voi_sagelocaltailcap75mrr_harmonic_qproject",
            "voi_sagelocaltailcap90mrr_harmonic_qproject",
            "voi_sagelocaltailcap95mrr_harmonic_qproject",
            "voi_sagenrlocalmrr_harmonic_qproject",
            "voi_sagecellblockmrr_harmonic_qproject",
            "voi_sagegraphcvarmrr_harmonic_qproject",
            "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
            "voi_sageenv025mrr_harmonic_qproject",
            "voi_sageenv05mrr_harmonic_qproject",
            "voi_sageenv1mrr_harmonic_qproject",
            "voi_sagecvar4mrr_harmonic_qproject",
            "voi_sagecvarsqrtmrr_harmonic_qproject",
            "voi_sagevcvarsqrtmrr_harmonic_qproject",
            "voi_sagevtopsqrtmrr_harmonic_qproject",
            "voi_sagestudcvarmrr_harmonic_qproject",
            "voi_sagespectralmrr_harmonic_qproject",
            "voi_sageholder15mrr_harmonic_qproject",
            "voi_sageholder2mrr_harmonic_qproject",
            "voi_sagenestedcvarmrr_harmonic_qproject",
            "voi_sagehiercvarmrr_harmonic_qproject",
            "voi_sagerepaircap10mrr_harmonic_qproject",
            "voi_sagerepaircap25mrr_harmonic_qproject",
            "voi_sagerepaircap50mrr_harmonic_qproject",
            "voi_sageflowmrr_harmonic_qproject",
            "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagerrcvalidfullmrr_harmonic_qproject",
                "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
                "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
            "voi_sagemetricavgmrr_harmonic_qproject",
            "voi_sagemetricenvmrr_harmonic_qproject",
            "voi_sagefloorcapmrr_harmonic_qproject",
            "voi_sageparetomrr_harmonic_qproject",
            "voi_sageharmmrr_harmonic_qproject",
            "voi_relmaxsqrtlevscaled_harmonic_qproject",
            "voi_relmaxlevscaled_harmonic_qproject",
        }:
            if dual_query_weights is None or dual_vector_weights is None:
                raise AssertionError("dual weights must be initialized for dual operator VOI")
            before = np.abs(target_residual)
            after = _expected_abs_normal(target_residual - projected_repair, np.sqrt(noise_var))
            gain = before - after
            if base_scheme in {
                "voi_relmaxposscaled_harmonic_qproject",
                "voi_relmaxposeff_harmonic_qproject",
                "voi_relmaxposnorm_harmonic_qproject",
                "voi_dpstateposnorm_harmonic_qproject",
                "voi_dpstatecovsplit_harmonic_qproject",
                "voi_dpstatesqrtcovsplit_harmonic_qproject",
                "voi_dpstatesplittail_harmonic_qproject",
                "voi_dpstatecovtail_harmonic_qproject",
                "voi_dpstatebreadthtail_harmonic_qproject",
                "voi_dpstateriskmaxmrr_harmonic_qproject",
                "voi_dpstateriskenv025mrr_harmonic_qproject",
                "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
                "voi_sageordergain_harmonic_qproject",
                "voi_sagemrr_harmonic_qproject",
                "voi_sageroutecmrr_harmonic_qproject",
                "voi_sagecovmrr_harmonic_qproject",
                "voi_sagesqcovmrr_harmonic_qproject",
                "voi_sagesoftcovmrr_harmonic_qproject",
                "voi_sagebreadthmrr_harmonic_qproject",
                "voi_sageblockmrr_harmonic_qproject",
                "voi_sagelocalblockmrr_harmonic_qproject",
                "voi_sagelocalvalidtailmrr_harmonic_qproject",
                "voi_sagelocaltailcap65mrr_harmonic_qproject",
                "voi_sagelocaltailcap75mrr_harmonic_qproject",
                "voi_sagelocaltailcap90mrr_harmonic_qproject",
                "voi_sagelocaltailcap95mrr_harmonic_qproject",
                "voi_sagenrlocalmrr_harmonic_qproject",
                "voi_sagecellblockmrr_harmonic_qproject",
                "voi_sagegraphcvarmrr_harmonic_qproject",
                "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
                "voi_sageenv025mrr_harmonic_qproject",
                "voi_sageenv05mrr_harmonic_qproject",
                "voi_sageenv1mrr_harmonic_qproject",
                "voi_sagecvar4mrr_harmonic_qproject",
                "voi_sagecvarsqrtmrr_harmonic_qproject",
                "voi_sagevcvarsqrtmrr_harmonic_qproject",
                "voi_sagevtopsqrtmrr_harmonic_qproject",
                "voi_sagestudcvarmrr_harmonic_qproject",
                "voi_sagespectralmrr_harmonic_qproject",
                "voi_sageholder15mrr_harmonic_qproject",
                "voi_sageholder2mrr_harmonic_qproject",
                "voi_sagenestedcvarmrr_harmonic_qproject",
                "voi_sagehiercvarmrr_harmonic_qproject",
                "voi_sagerepaircap10mrr_harmonic_qproject",
                "voi_sagerepaircap25mrr_harmonic_qproject",
                "voi_sagerepaircap50mrr_harmonic_qproject",
                "voi_sageflowmrr_harmonic_qproject",
                "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagerrcvalidfullmrr_harmonic_qproject",
                "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
                "voi_dpstateriskmaxmrr_harmonic_qproject",
                "voi_dpstateriskenv025mrr_harmonic_qproject",
                "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
                "voi_sagemetricavgmrr_harmonic_qproject",
                "voi_sagemetricenvmrr_harmonic_qproject",
                "voi_sagefloorcapmrr_harmonic_qproject",
                "voi_sageparetomrr_harmonic_qproject",
                "voi_sageharmmrr_harmonic_qproject",
            }:
                gain = np.maximum(gain, 0.0)
            gain_by_qid = np.zeros_like(residual, dtype=np.float64)
            gain_by_qid[plan.target_qids] = gain
            valid_tail_gain = 0.0
            if base_scheme == "voi_sageordergain_harmonic_qproject":
                query_score = _harmonic_ordered_sum(gain_by_qid)
                vector_gain_values = np.asarray(
                    [
                        0.5 * float(np.sum(gain_by_qid[blocks[target_block_id].query_indices]))
                        for target_block_id in dual_vector_block_ids
                    ],
                    dtype=np.float64,
                )
                vector_score = _harmonic_ordered_sum(vector_gain_values)
            elif base_scheme in {
                "voi_sagemrr_harmonic_qproject",
                "voi_sageroutecmrr_harmonic_qproject",
                "voi_sagecovmrr_harmonic_qproject",
                "voi_sagesqcovmrr_harmonic_qproject",
                "voi_sagesoftcovmrr_harmonic_qproject",
                "voi_sagebreadthmrr_harmonic_qproject",
                "voi_sageblockmrr_harmonic_qproject",
                "voi_sagelocalblockmrr_harmonic_qproject",
                "voi_sagelocalvalidtailmrr_harmonic_qproject",
                "voi_sagelocaltailcap65mrr_harmonic_qproject",
                "voi_sagelocaltailcap75mrr_harmonic_qproject",
                "voi_sagelocaltailcap90mrr_harmonic_qproject",
                "voi_sagelocaltailcap95mrr_harmonic_qproject",
                "voi_sagenrlocalmrr_harmonic_qproject",
                "voi_sagecellblockmrr_harmonic_qproject",
                "voi_sagegraphcvarmrr_harmonic_qproject",
                "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
                "voi_sageenv025mrr_harmonic_qproject",
                "voi_sageenv05mrr_harmonic_qproject",
                "voi_sageenv1mrr_harmonic_qproject",
                "voi_sagecvar4mrr_harmonic_qproject",
                "voi_sagecvarsqrtmrr_harmonic_qproject",
                "voi_sagevcvarsqrtmrr_harmonic_qproject",
                "voi_sagevtopsqrtmrr_harmonic_qproject",
                "voi_sagestudcvarmrr_harmonic_qproject",
                "voi_sagespectralmrr_harmonic_qproject",
                "voi_sageholder15mrr_harmonic_qproject",
                "voi_sageholder2mrr_harmonic_qproject",
                "voi_sagenestedcvarmrr_harmonic_qproject",
                "voi_sagehiercvarmrr_harmonic_qproject",
                "voi_sagerepaircap10mrr_harmonic_qproject",
                "voi_sagerepaircap25mrr_harmonic_qproject",
                "voi_sagerepaircap50mrr_harmonic_qproject",
                "voi_sageflowmrr_harmonic_qproject",
                "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagerrcvalidfullmrr_harmonic_qproject",
                "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
                "voi_dpstateriskmaxmrr_harmonic_qproject",
                "voi_dpstateriskenv025mrr_harmonic_qproject",
                "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
                "voi_sagemetricavgmrr_harmonic_qproject",
                "voi_sagemetricenvmrr_harmonic_qproject",
                "voi_sagefloorcapmrr_harmonic_qproject",
                "voi_sageparetomrr_harmonic_qproject",
                "voi_sageharmmrr_harmonic_qproject",
            }:
                clipped_abs = before_abs_all.copy()
                clipped_abs[plan.target_qids] = np.minimum(
                    clipped_abs[plan.target_qids],
                    after,
                )
                if base_scheme in {
                    "voi_sageenv025mrr_harmonic_qproject",
                    "voi_sageenv05mrr_harmonic_qproject",
                    "voi_sageenv1mrr_harmonic_qproject",
                    "voi_sagecvar4mrr_harmonic_qproject",
                    "voi_sagecvarsqrtmrr_harmonic_qproject",
                    "voi_sagevcvarsqrtmrr_harmonic_qproject",
                    "voi_sagevtopsqrtmrr_harmonic_qproject",
                    "voi_sagestudcvarmrr_harmonic_qproject",
                }:
                    envelope_tau = _risk_envelope_tau(base_scheme)
                    if base_scheme in {
                        "voi_sagecvar4mrr_harmonic_qproject",
                        "voi_sagecvarsqrtmrr_harmonic_qproject",
                    }:
                        risk_channel_fn = lambda abs_values: _normalized_cvar_risk_channels(
                            base_scheme,
                            abs_values,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            dual_vector_sensitivity,
                        )
                    elif base_scheme == "voi_sagevcvarsqrtmrr_harmonic_qproject":
                        risk_channel_fn = lambda abs_values: _normalized_vector_cvar_risk_channels(
                            base_scheme,
                            abs_values,
                            blocks,
                            dual_vector_block_ids,
                            dual_vector_sensitivity,
                        )
                    elif base_scheme == "voi_sagevtopsqrtmrr_harmonic_qproject":
                        def risk_channel_fn(abs_values: np.ndarray) -> np.ndarray:
                            vector_tvd_values = np.asarray(
                                [
                                    0.5 * float(np.sum(abs_values[blocks[target_block_id].query_indices]))
                                    for target_block_id in dual_vector_block_ids
                                ],
                                dtype=np.float64,
                            )
                            tail_k = _cvar_tail_k(base_scheme, int(vector_tvd_values.size))
                            return np.asarray([_topk_mean(vector_tvd_values, tail_k)], dtype=np.float64)
                    elif base_scheme == "voi_sagestudcvarmrr_harmonic_qproject":
                        risk_channel_fn = lambda abs_values: _normalized_studentized_cvar_risk_channels(
                            base_scheme,
                            abs_values,
                            blocks,
                            dual_vector_block_ids,
                            measurement_sigma,
                        )
                    else:
                        risk_channel_fn = lambda abs_values: _normalized_risk_channels(
                            abs_values,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            dual_vector_sensitivity,
                        )
                    before_envelope = _softmax_envelope(
                        risk_channel_fn(before_abs_all),
                        envelope_tau,
                    )
                    after_envelope = _softmax_envelope(
                        risk_channel_fn(clipped_abs),
                        envelope_tau,
                    )
                    query_score = max(0.0, before_envelope - after_envelope)
                    vector_score = 0.0
                elif base_scheme == "voi_sagespectralmrr_harmonic_qproject":
                    before_spectral_risk = _normalized_spectral_atom_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        dual_vector_sensitivity,
                    )
                    after_spectral_risk = _normalized_spectral_atom_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        dual_vector_sensitivity,
                    )
                    query_score = max(0.0, before_spectral_risk - after_spectral_risk)
                    vector_score = 0.0
                elif base_scheme in {
                    "voi_sageholder15mrr_harmonic_qproject",
                    "voi_sageholder2mrr_harmonic_qproject",
                }:
                    before_holder_risk = _normalized_holder_atom_risk(
                        base_scheme,
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                    )
                    after_holder_risk = _normalized_holder_atom_risk(
                        base_scheme,
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                    )
                    query_score = max(0.0, before_holder_risk - after_holder_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sagenestedcvarmrr_harmonic_qproject":
                    before_nested_risk = _nested_block_cvar_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                    )
                    after_nested_risk = _nested_block_cvar_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                    )
                    query_score = max(0.0, before_nested_risk - after_nested_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sagehiercvarmrr_harmonic_qproject":
                    before_hier_risk = _hierarchical_block_tail_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                    )
                    after_hier_risk = _hierarchical_block_tail_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                    )
                    query_score = max(0.0, before_hier_risk - after_hier_risk)
                    vector_score = 0.0
                elif base_scheme in {
                    "voi_sagerepaircap10mrr_harmonic_qproject",
                    "voi_sagerepaircap25mrr_harmonic_qproject",
                    "voi_sagerepaircap50mrr_harmonic_qproject",
                }:
                    before_repair_cap_risk = _repair_graph_block_capacity_risk(
                        base_scheme,
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                    )
                    after_repair_cap_risk = _repair_graph_block_capacity_risk(
                        base_scheme,
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                    )
                    query_score = max(0.0, before_repair_cap_risk - after_repair_cap_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sagecapdromrr_harmonic_qproject":
                    if capacity_risk_context is None:
                        raise AssertionError("capacity risk context must be initialized")
                    before_atoms = _normalized_capdro_risk_atoms(
                        base_scheme,
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        dual_vector_sensitivity,
                        capacity_risk_context,
                    )
                    after_atoms = _normalized_capdro_risk_atoms(
                        base_scheme,
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        dual_vector_sensitivity,
                        capacity_risk_context,
                    )
                    before_envelope = _capacity_envelope(
                        before_atoms,
                        capacity_risk_context.atom_caps,
                    )
                    after_envelope = _capacity_envelope(
                        after_atoms,
                        capacity_risk_context.atom_caps,
                    )
                    query_score = max(0.0, before_envelope - after_envelope)
                    vector_score = 0.0
                elif base_scheme == "voi_sageflowmrr_harmonic_qproject":
                    if edge_flow_risk_context is None:
                        raise AssertionError("edge-flow risk context must be initialized")
                    after_flow_risk = _edge_flow_capacity_risk(clipped_abs, edge_flow_risk_context)
                    query_score = max(0.0, edge_flow_before_risk - after_flow_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sagegraphcapmrr_harmonic_qproject":
                    if edge_flow_risk_context is None:
                        raise AssertionError("graph-capacity risk context must be initialized")
                    after_graph_cap_risk = _edge_flow_capacity_risk(clipped_abs, edge_flow_risk_context)
                    query_score = max(0.0, edge_flow_before_risk - after_graph_cap_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sagegraphcapmassmrr_harmonic_qproject":
                    before_graph_mass_risk = _graph_capacity_mass_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                    )
                    after_graph_mass_risk = _graph_capacity_mass_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                    )
                    query_score = max(0.0, before_graph_mass_risk - after_graph_mass_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sagegraphcapfullmrr_harmonic_qproject":
                    before_graph_full_risk = _graph_capacity_full_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                    )
                    after_graph_full_risk = _graph_capacity_full_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                    )
                    query_score = max(0.0, before_graph_full_risk - after_graph_full_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sageqvcapmrr_harmonic_qproject":
                    before_qv_cap_risk = _query_vector_capacity_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                    )
                    after_qv_cap_risk = _query_vector_capacity_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                    )
                    query_score = max(0.0, before_qv_cap_risk - after_qv_cap_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sageqvcapenvmrr_harmonic_qproject":
                    before_qv_cap_risk = _query_vector_capacity_envelope_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                    )
                    after_qv_cap_risk = _query_vector_capacity_envelope_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                    )
                    query_score = max(0.0, before_qv_cap_risk - after_qv_cap_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sageqvcapcurrmrr_harmonic_qproject":
                    before_qv_curr_risk = _query_vector_curriculum_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        round_id,
                        rounds,
                    )
                    after_qv_curr_risk = _query_vector_curriculum_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        round_id,
                        rounds,
                    )
                    query_score = max(0.0, before_qv_curr_risk - after_qv_curr_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sageqvcapresmrr_harmonic_qproject":
                    before_qv_reserve_risk = _query_vector_reserve_capacity_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        round_id,
                        rounds,
                    )
                    after_qv_reserve_risk = _query_vector_reserve_capacity_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        round_id,
                        rounds,
                    )
                    query_score = max(0.0, before_qv_reserve_risk - after_qv_reserve_risk)
                    vector_score = 0.0
                elif base_scheme == "voi_sageqvcapbandmrr_harmonic_qproject":
                    before_qv_band_risk = _query_vector_capacity_band_risk(
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        round_id,
                        rounds,
                    )
                    after_qv_band_risk = _query_vector_capacity_band_risk(
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_sensitivity,
                        round_id,
                        rounds,
                    )
                    query_score = max(0.0, before_qv_band_risk - after_qv_band_risk)
                    vector_score = 0.0
                elif base_scheme in {
                    "voi_sageqvcalmrr_harmonic_qproject",
                    "voi_sageqvcalcurrmrr_harmonic_qproject",
                    "voi_sageqvcalbandmrr_harmonic_qproject",
                    "voi_sageqvcaldromrr_harmonic_qproject",
                    "voi_sageqvcaltopdromrr_harmonic_qproject",
                    "voi_sageqvcalboxdromrr_harmonic_qproject",
                    "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                    "voi_sageqvcalstaildromrr_harmonic_qproject",
                    "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                    "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                    "voi_sageqvcalmaxdromrr_harmonic_qproject",
                    "voi_sageqvcalfrontdromrr_harmonic_qproject",
                    "voi_sageqvcaljointdromrr_harmonic_qproject",
                    "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                }:
                    if base_scheme == "voi_sageqvcalmaxdromrr_harmonic_qproject":
                        before_qv_calibrated_risk = _query_vector_calibrated_max_frontier_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_max_frontier_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                        )
                    elif base_scheme == "voi_sageqvcalfrontdromrr_harmonic_qproject":
                        before_qv_calibrated_risk = _query_vector_calibrated_target_frontier_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_target_frontier_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                    elif base_scheme == "voi_sageqvcaljointdromrr_harmonic_qproject":
                        before_qv_calibrated_risk = _query_vector_calibrated_joint_tail_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_joint_tail_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                    elif base_scheme == "voi_sageqvcalflowguarddromrr_harmonic_qproject":
                        if edge_flow_risk_context is None:
                            raise AssertionError("edge-flow risk context must be initialized")
                        before_qv_calibrated_risk = _query_vector_calibrated_flow_guard_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            edge_flow_risk_context,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_flow_guard_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            edge_flow_risk_context,
                        )
                    elif base_scheme == "voi_sageqvcaltailcapdromrr_harmonic_qproject":
                        before_qv_calibrated_risk = _query_vector_calibrated_tail_capacity_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_tail_capacity_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                        )
                    elif base_scheme == "voi_sageqvcalatomcapdromrr_harmonic_qproject":
                        before_qv_calibrated_risk = _query_vector_calibrated_atom_capacity_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_atom_capacity_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                        )
                    elif base_scheme == "voi_sageqvcalpairdromrr_harmonic_qproject":
                        before_qv_calibrated_risk = _query_vector_calibrated_paired_frontier_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_paired_frontier_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                    elif base_scheme == "voi_sageqvcalbalfrontdromrr_harmonic_qproject":
                        before_qv_calibrated_risk = _query_vector_calibrated_balanced_frontier_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_balanced_frontier_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            plan.target_qids,
                        )
                    elif base_scheme in {
                        "voi_sageqvcaldromrr_harmonic_qproject",
                        "voi_sageqvcaltopdromrr_harmonic_qproject",
                        "voi_sageqvcalboxdromrr_harmonic_qproject",
                        "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                        "voi_sageqvcalstaildromrr_harmonic_qproject",
                        "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                    }:
                        tail_mode = (
                            "top_sqrt"
                            if base_scheme
                            in {
                                "voi_sageqvcaltopdromrr_harmonic_qproject",
                                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                            }
                            else "max"
                        )
                        constraint_mode = (
                            "uniform_box"
                            if base_scheme
                            in {
                                "voi_sageqvcalboxdromrr_harmonic_qproject",
                                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                            }
                            else "reserved_tail"
                        )
                        scalar_tail_mode = None
                        scalar_tail_indices = None
                        if base_scheme == "voi_sageqvcalstaildromrr_harmonic_qproject":
                            scalar_tail_mode = "max"
                        elif base_scheme == "voi_sageqvcalstailsqrtdromrr_harmonic_qproject":
                            scalar_tail_mode = "top_sqrt"
                        elif base_scheme == "voi_sageqvcalfrontdromrr_harmonic_qproject":
                            scalar_tail_mode = "max"
                            scalar_tail_indices = plan.target_qids
                        before_qv_calibrated_risk = _query_vector_calibrated_dro_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            tail_mode=tail_mode,
                            constraint_mode=constraint_mode,
                            scalar_tail_mode=scalar_tail_mode,
                            scalar_tail_indices=scalar_tail_indices,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_dro_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            tail_mode=tail_mode,
                            constraint_mode=constraint_mode,
                            scalar_tail_mode=scalar_tail_mode,
                            scalar_tail_indices=scalar_tail_indices,
                        )
                    else:
                        if base_scheme == "voi_sageqvcalmrr_harmonic_qproject":
                            calibration_mode = "avg"
                        elif base_scheme == "voi_sageqvcalcurrmrr_harmonic_qproject":
                            calibration_mode = "curr"
                        else:
                            calibration_mode = "band"
                        before_qv_calibrated_risk = _query_vector_calibrated_risk(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            calibration_mode,
                            round_id,
                            rounds,
                        )
                        after_qv_calibrated_risk = _query_vector_calibrated_risk(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                            dual_query_sensitivity,
                            measurement_sigma,
                            row_count,
                            calibration_mode,
                            round_id,
                            rounds,
                        )
                    query_score = max(0.0, before_qv_calibrated_risk - after_qv_calibrated_risk)
                    vector_score = 0.0
                elif base_scheme in {
                    "voi_sagevalidfullmaxmrr_harmonic_qproject",
                    "voi_sagevalidfullcapmrr_harmonic_qproject",
                }:
                    query_score = _valid_full_tvd_score(
                        base_scheme,
                        plan,
                        block,
                        before_abs_all,
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                    )
                    vector_score = 0.0
                elif base_scheme in {
                    "voi_dpstateriskmaxmrr_harmonic_qproject",
                    "voi_dpstateriskenv025mrr_harmonic_qproject",
                    "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
                }:
                    if dual_vector_local_weights is None or dual_query_membership_split is None:
                        raise AssertionError("DP-state risk envelope requires split vector weights")
                    before_state_risk = _dpstate_risk_envelope_value(
                        base_scheme,
                        before_abs_all,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_weights,
                        dual_vector_weights,
                        dual_vector_local_weights,
                        dual_query_membership_split,
                        dual_query_sensitivity,
                        dual_vector_sensitivity,
                    )
                    after_state_risk = _dpstate_risk_envelope_value(
                        base_scheme,
                        clipped_abs,
                        blocks,
                        dual_vector_block_ids,
                        dual_query_weights,
                        dual_vector_weights,
                        dual_vector_local_weights,
                        dual_query_membership_split,
                        dual_query_sensitivity,
                        dual_vector_sensitivity,
                    )
                    query_score = max(0.0, before_state_risk - after_state_risk)
                    vector_score = 0.0
                elif base_scheme in {
                    "voi_sagemetricavgmrr_harmonic_qproject",
                    "voi_sagemetricenvmrr_harmonic_qproject",
                    "voi_sagefloorcapmrr_harmonic_qproject",
                }:
                    before_metric_risk = _normalized_metric_risk_value(
                        base_scheme,
                        _normalized_metric_risk_channels(
                            before_abs_all,
                            blocks,
                            dual_vector_block_ids,
                        ),
                    )
                    after_metric_risk = _normalized_metric_risk_value(
                        base_scheme,
                        _normalized_metric_risk_channels(
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        ),
                    )
                    query_score = max(0.0, before_metric_risk - after_metric_risk)
                    vector_score = 0.0
                elif base_scheme in {
                    "voi_sagelocalblockmrr_harmonic_qproject",
                    "voi_sagelocalvalidtailmrr_harmonic_qproject",
                    "voi_sagelocaltailcap65mrr_harmonic_qproject",
                    "voi_sagelocaltailcap75mrr_harmonic_qproject",
                    "voi_sagelocaltailcap90mrr_harmonic_qproject",
                    "voi_sagelocaltailcap95mrr_harmonic_qproject",
                    "voi_sagegraphcvarmrr_harmonic_qproject",
                    "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
                }:
                    query_score = max(
                        0.0,
                        _harmonic_ordered_sum(before_abs_all[plan.target_qids])
                        - _harmonic_ordered_sum(clipped_abs[plan.target_qids]),
                    )
                elif base_scheme in {
                    "voi_sagenrlocalmrr_harmonic_qproject",
                    "voi_sagecellblockmrr_harmonic_qproject",
                }:
                    reliability = 1.0 / np.sqrt(np.maximum(plan.noise_counts, 1.0e-12))
                    query_score = max(
                        0.0,
                        _weighted_harmonic_ordered_sum(before_abs_all[plan.target_qids], reliability)
                        - _weighted_harmonic_ordered_sum(clipped_abs[plan.target_qids], reliability),
                    )
                else:
                    query_score = max(0.0, dual_query_risk - _harmonic_ordered_sum(clipped_abs))
                if base_scheme in {
                    "voi_sageblockmrr_harmonic_qproject",
                    "voi_sagelocalblockmrr_harmonic_qproject",
                    "voi_sagelocalvalidtailmrr_harmonic_qproject",
                    "voi_sagelocaltailcap65mrr_harmonic_qproject",
                    "voi_sagelocaltailcap75mrr_harmonic_qproject",
                    "voi_sagelocaltailcap90mrr_harmonic_qproject",
                    "voi_sagelocaltailcap95mrr_harmonic_qproject",
                }:
                    vector_gain_values = []
                    target_qids = plan.target_qids
                    source_size = int(block.query_indices.size)
                    for target_block_id in dual_vector_block_ids:
                        target_idx = blocks[target_block_id].query_indices
                        block_gain = max(
                            0.0,
                            0.5 * _harmonic_ordered_sum(before_abs_all[target_idx])
                            - 0.5 * _harmonic_ordered_sum(clipped_abs[target_idx]),
                        )
                        vector_gain_values.append(block_gain)
                        if base_scheme in {
                            "voi_sagelocalvalidtailmrr_harmonic_qproject",
                            "voi_sagelocaltailcap65mrr_harmonic_qproject",
                            "voi_sagelocaltailcap75mrr_harmonic_qproject",
                            "voi_sagelocaltailcap90mrr_harmonic_qproject",
                            "voi_sagelocaltailcap95mrr_harmonic_qproject",
                        }:
                            repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
                            repaired = float(np.count_nonzero(repair_mask))
                            if repaired > 0.0:
                                local_idx = target_idx[repair_mask]
                                breadth = _valid_local_tail_propagation_weight(
                                    repaired,
                                    float(target_idx.size),
                                )
                                source_coverage = _source_cell_propagation_weight(
                                    plan,
                                    local_idx,
                                    source_size,
                                    int(target_idx.size),
                                )
                                block_tvd_gain = 0.5 * float(np.sum(gain_by_qid[target_idx]))
                                valid_tail_gain = max(
                                    valid_tail_gain,
                                    breadth * source_coverage * block_tvd_gain,
                                )
                    vector_score = _harmonic_ordered_sum(np.asarray(vector_gain_values, dtype=np.float64))
                    if base_scheme == "voi_sagelocalvalidtailmrr_harmonic_qproject":
                        vector_score = max(vector_score, valid_tail_gain)
                elif base_scheme in {
                    "voi_sagenrlocalmrr_harmonic_qproject",
                    "voi_sagecellblockmrr_harmonic_qproject",
                }:
                    reliability_by_qid = np.zeros_like(residual, dtype=np.float64)
                    reliability_by_qid[plan.target_qids] = 1.0 / np.sqrt(
                        np.maximum(plan.noise_counts, 1.0e-12)
                    )
                    vector_gain_values = []
                    source_size = int(block.query_indices.size)
                    for target_block_id in dual_vector_block_ids:
                        target_idx = blocks[target_block_id].query_indices
                        repair_mask = reliability_by_qid[target_idx] > 0.0
                        if not np.any(repair_mask):
                            vector_gain_values.append(0.0)
                            continue
                        if base_scheme == "voi_sagenrlocalmrr_harmonic_qproject":
                            local_idx = target_idx[repair_mask]
                            local_reliability = reliability_by_qid[local_idx]
                            block_gain = max(
                                0.0,
                                0.5 * _weighted_harmonic_ordered_sum(before_abs_all[local_idx], local_reliability)
                                - 0.5 * _weighted_harmonic_ordered_sum(clipped_abs[local_idx], local_reliability),
                            )
                        else:
                            local_idx = target_idx[repair_mask]
                            block_gain = max(
                                0.0,
                                0.5 * _harmonic_ordered_sum(before_abs_all[target_idx])
                                - 0.5 * _harmonic_ordered_sum(clipped_abs[target_idx]),
                            )
                            block_gain *= _source_cell_propagation_weight(
                                plan,
                                local_idx,
                                source_size,
                                int(target_idx.size),
                            )
                        vector_gain_values.append(block_gain)
                    vector_score = _harmonic_ordered_sum(np.asarray(vector_gain_values, dtype=np.float64))
                elif base_scheme in {
                    "voi_sagegraphcvarmrr_harmonic_qproject",
                    "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
                }:
                    vector_gain_values = []
                    target_qids = plan.target_qids
                    for target_block_id in dual_vector_block_ids:
                        target_idx = blocks[target_block_id].query_indices
                        repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
                        repaired = float(np.count_nonzero(repair_mask))
                        if repaired <= 0.0:
                            vector_gain_values.append(0.0)
                            continue
                        block_size = max(float(target_idx.size), 1.0)
                        coverage = repaired / block_size
                        if base_scheme == "voi_sagegraphsqrtcvarmrr_harmonic_qproject":
                            coverage = math.sqrt(coverage)
                        tail_k = _cvar_tail_k(base_scheme, int(target_idx.size))
                        before_block_risk = _topk_mean(before_abs_all[target_idx], tail_k)
                        after_block_risk = _topk_mean(clipped_abs[target_idx], tail_k)
                        block_gain = max(0.0, before_block_risk - after_block_risk)
                        vector_gain_values.append(float(coverage) * block_gain)
                    graph_gains = np.asarray(vector_gain_values, dtype=np.float64)
                    vector_score = float(np.max(graph_gains)) if graph_gains.size else 0.0
                elif base_scheme in {
                    "voi_sagerrcvalidfullmrr_harmonic_qproject",
                    "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                    "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                    "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                    "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                    "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                    "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                    "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                    "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
                    "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
                    "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
                    "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
                }:
                    if base_scheme == "voi_sagerrcvalidmaxdropmrr_harmonic_qproject":
                        vector_score = _valid_max_tvd_drop_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcvalidtopkgainmrr_harmonic_qproject":
                        vector_score = _valid_topk_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject":
                        vector_score = _valid_head_topk_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcvalidharmgainmrr_harmonic_qproject":
                        vector_score = _valid_harmonic_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject":
                        vector_score = _valid_lorenz_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcnestedvalidmrr_harmonic_qproject":
                        vector_score = _valid_nested_head_broad_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject":
                        vector_score = _valid_nested_lorenz_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject":
                        vector_score = _valid_nested_topk_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject":
                        vector_score = _valid_nested_lorenz_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcnestedparetovalidmrr_harmonic_qproject":
                        vector_score = _valid_nested_pareto_tvd_gain_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    elif base_scheme == "voi_sagerrcvalidtopkmrr_harmonic_qproject":
                        vector_score = _valid_topk_tvd_drop_score(
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                    else:
                        vector_score = _valid_full_tvd_score(
                            "voi_sagevalidfullmaxmrr_harmonic_qproject",
                            plan,
                            block,
                            before_abs_all,
                            clipped_abs,
                            blocks,
                            dual_vector_block_ids,
                        )
                elif base_scheme not in {
                    "voi_sageenv025mrr_harmonic_qproject",
                    "voi_sageenv05mrr_harmonic_qproject",
                    "voi_sageenv1mrr_harmonic_qproject",
                    "voi_sagecvar4mrr_harmonic_qproject",
                    "voi_sagecvarsqrtmrr_harmonic_qproject",
                    "voi_sagevcvarsqrtmrr_harmonic_qproject",
                    "voi_sagevtopsqrtmrr_harmonic_qproject",
                    "voi_sagestudcvarmrr_harmonic_qproject",
                    "voi_sageflowmrr_harmonic_qproject",
                "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
                    "voi_dpstateriskmaxmrr_harmonic_qproject",
                    "voi_dpstateriskenv025mrr_harmonic_qproject",
                    "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
                }:
                    after_vector_risk = _harmonic_ordered_sum(
                        np.asarray(
                            [
                                0.5 * float(np.sum(clipped_abs[blocks[target_block_id].query_indices]))
                                for target_block_id in dual_vector_block_ids
                            ],
                            dtype=np.float64,
                        )
                    )
                    vector_score = max(0.0, dual_vector_risk - after_vector_risk)
                if base_scheme in {
                    "voi_sagecovmrr_harmonic_qproject",
                    "voi_sagesqcovmrr_harmonic_qproject",
                    "voi_sagesoftcovmrr_harmonic_qproject",
                    "voi_sagebreadthmrr_harmonic_qproject",
                }:
                    vector_gain_values = []
                    target_qids = plan.target_qids
                    for target_block_id in dual_vector_block_ids:
                        target_idx = blocks[target_block_id].query_indices
                        repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
                        repaired = float(np.count_nonzero(repair_mask))
                        block_size = max(float(target_idx.size), 1.0)
                        coverage = repaired / block_size
                        if base_scheme == "voi_sagesqcovmrr_harmonic_qproject":
                            propagation_weight = coverage * coverage
                        elif base_scheme == "voi_sagesoftcovmrr_harmonic_qproject":
                            inverse_size = 1.0 / block_size
                            propagation_weight = coverage * (coverage + inverse_size) / (1.0 + inverse_size)
                        elif base_scheme == "voi_sagebreadthmrr_harmonic_qproject":
                            if block_size <= 1.0:
                                breadth = 1.0
                            else:
                                breadth = max(0.0, (repaired - 1.0) / (block_size - 1.0))
                            propagation_weight = coverage * breadth
                        else:
                            propagation_weight = coverage
                        block_gain = 0.5 * float(np.sum(gain_by_qid[target_idx]))
                        vector_gain_values.append(propagation_weight * block_gain)
                    vector_score = _harmonic_ordered_sum(np.asarray(vector_gain_values, dtype=np.float64))
            elif base_scheme in {
                "voi_dpstatecovsplit_harmonic_qproject",
                "voi_dpstatesqrtcovsplit_harmonic_qproject",
                "voi_dpstatesplittail_harmonic_qproject",
                "voi_dpstatecovtail_harmonic_qproject",
                "voi_dpstatebreadthtail_harmonic_qproject",
            }:
                if dual_vector_local_weights is None or dual_query_membership_split is None:
                    raise AssertionError("split vector weights must be initialized")
                query_score = float(np.sum(dual_query_weights[plan.target_qids] * gain))
                vector_score = 0.0
                target_qids = plan.target_qids
                for block_weight, local_weights, target_block_id in zip(
                    dual_vector_weights,
                    dual_vector_local_weights,
                    dual_vector_block_ids,
                ):
                    target_idx = blocks[target_block_id].query_indices
                    repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
                    repaired = float(np.count_nonzero(repair_mask))
                    if repaired <= 0.0:
                        continue
                    coverage = repaired / max(float(target_idx.size), 1.0)
                    if base_scheme == "voi_dpstatesqrtcovsplit_harmonic_qproject":
                        coverage_weight = math.sqrt(coverage)
                    else:
                        coverage_weight = coverage
                    vector_score += (
                        float(block_weight)
                        * coverage_weight
                        * 0.5
                        * float(
                            np.sum(
                                local_weights
                                * dual_query_membership_split[target_idx]
                                * gain_by_qid[target_idx]
                            )
                        )
                    )
            else:
                query_score = float(np.sum(dual_query_weights[plan.target_qids] * gain))
                vector_score = 0.0
                for vector_weight, target_block_id in zip(dual_vector_weights, dual_vector_block_ids):
                    target_idx = blocks[target_block_id].query_indices
                    vector_score += float(vector_weight) * 0.5 * float(np.sum(gain_by_qid[target_idx]))
            if base_scheme == "voi_dualop_harmonic_qproject":
                score = query_score + vector_score
            elif base_scheme == "voi_nashop_harmonic_qproject":
                # Marginal gain for geometric-mean risk sqrt(R_query * R_vector).
                parity_scale = math.sqrt(dual_query_risk * dual_vector_risk)
                score = (query_score / dual_query_risk + vector_score / dual_vector_risk) * parity_scale
            elif base_scheme == "voi_relmaxscaled_harmonic_qproject":
                parity_scale = min(dual_query_risk, dual_vector_risk)
                score = max(query_score / dual_query_risk, vector_score / dual_vector_risk) * parity_scale
            elif base_scheme == "voi_relmaxposscaled_harmonic_qproject":
                parity_scale = min(dual_query_risk, dual_vector_risk)
                score = max(query_score / dual_query_risk, vector_score / dual_vector_risk) * parity_scale
            elif base_scheme == "voi_relmaxposeff_harmonic_qproject":
                parity_scale = min(dual_query_risk, dual_vector_risk)
                score = max(query_score / dual_query_risk, vector_score / dual_vector_risk) * parity_scale
            elif base_scheme == "voi_relmaxposnorm_harmonic_qproject":
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / dual_query_sensitivity
                vector_score_norm = vector_score / dual_vector_sensitivity
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
            elif base_scheme == "voi_dpstateposnorm_harmonic_qproject":
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / dual_query_sensitivity
                vector_score_norm = vector_score / dual_vector_sensitivity
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
            elif base_scheme in {
                "voi_dpstatecovsplit_harmonic_qproject",
                "voi_dpstatesqrtcovsplit_harmonic_qproject",
                "voi_dpstatesplittail_harmonic_qproject",
                "voi_dpstatecovtail_harmonic_qproject",
                "voi_dpstatebreadthtail_harmonic_qproject",
            }:
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / dual_query_sensitivity
                vector_score_norm = vector_score / dual_vector_sensitivity
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
                if base_scheme in {
                    "voi_dpstatesplittail_harmonic_qproject",
                    "voi_dpstatecovtail_harmonic_qproject",
                    "voi_dpstatebreadthtail_harmonic_qproject",
                }:
                    clipped_abs = before_abs_all.copy()
                    clipped_abs[plan.target_qids] = np.minimum(
                        clipped_abs[plan.target_qids],
                        after,
                    )
                    query_tail_gain = max(0.0, dual_query_tail_risk - float(np.max(clipped_abs)))
                    if base_scheme == "voi_dpstatesplittail_harmonic_qproject":
                        after_vector_tail_risk = max(
                            float(
                                max(
                                    0.5 * float(np.sum(clipped_abs[blocks[target_block_id].query_indices]))
                                    for target_block_id in dual_vector_block_ids
                                )
                            )
                            if dual_vector_block_ids
                            else 0.0,
                            1.0e-12,
                        )
                        vector_tail_gain = max(0.0, dual_vector_tail_risk - after_vector_tail_risk)
                    else:
                        vector_tail_gain = 0.0
                        target_qids = plan.target_qids
                        for target_block_id in dual_vector_block_ids:
                            target_idx = blocks[target_block_id].query_indices
                            repair_mask = np.isin(target_idx, target_qids, assume_unique=False)
                            repaired = float(np.count_nonzero(repair_mask))
                            if repaired <= 0.0:
                                continue
                            before_block_risk = 0.5 * float(np.sum(before_abs_all[target_idx]))
                            after_block_risk = 0.5 * float(np.sum(clipped_abs[target_idx]))
                            block_gain = max(0.0, before_block_risk - after_block_risk)
                            validity = _valid_tail_propagation_weight(base_scheme, repaired, float(target_idx.size))
                            vector_tail_gain = max(vector_tail_gain, validity * block_gain)
                    query_tail_risk_norm = dual_query_tail_risk
                    vector_tail_risk_norm = dual_vector_tail_risk / 0.5
                    query_tail_score_norm = query_tail_gain / 2.0
                    vector_tail_score_norm = vector_tail_gain
                    tail_parity_scale = min(query_tail_risk_norm, vector_tail_risk_norm)
                    tail_score = max(
                        query_tail_score_norm / max(query_tail_risk_norm, 1.0e-12),
                        vector_tail_score_norm / max(vector_tail_risk_norm, 1.0e-12),
                    ) * tail_parity_scale
                    score = max(score, tail_score)
            elif base_scheme == "voi_sageordergain_harmonic_qproject":
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / dual_query_sensitivity
                vector_score_norm = vector_score / dual_vector_sensitivity
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
            elif base_scheme == "voi_sageroutecmrr_harmonic_qproject":
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / dual_query_sensitivity
                vector_score_norm = vector_score / dual_vector_sensitivity
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
            elif base_scheme in {
                "voi_sagemrr_harmonic_qproject",
                "voi_sageblockmrr_harmonic_qproject",
                "voi_sagelocalblockmrr_harmonic_qproject",
                "voi_sagelocalvalidtailmrr_harmonic_qproject",
                "voi_sagelocaltailcap65mrr_harmonic_qproject",
                "voi_sagelocaltailcap75mrr_harmonic_qproject",
                "voi_sagelocaltailcap90mrr_harmonic_qproject",
                "voi_sagelocaltailcap95mrr_harmonic_qproject",
                "voi_sagenrlocalmrr_harmonic_qproject",
                "voi_sagecellblockmrr_harmonic_qproject",
                "voi_sagegraphcvarmrr_harmonic_qproject",
                "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
                "voi_sagerrcvalidfullmrr_harmonic_qproject",
                "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
                "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
            }:
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / (2.0 * dual_query_sensitivity)
                vector_score_norm = vector_score / (2.0 * dual_vector_sensitivity)
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
                if base_scheme in {
                    "voi_sagelocalvalidtailmrr_harmonic_qproject",
                    "voi_sagelocaltailcap65mrr_harmonic_qproject",
                    "voi_sagelocaltailcap75mrr_harmonic_qproject",
                    "voi_sagelocaltailcap90mrr_harmonic_qproject",
                    "voi_sagelocaltailcap95mrr_harmonic_qproject",
                }:
                    vector_tail_risk = max(
                        float(
                            max(
                                0.5 * float(np.sum(before_abs_all[blocks[target_block_id].query_indices]))
                                for target_block_id in dual_vector_block_ids
                            )
                        )
                        if dual_vector_block_ids
                        else 0.0,
                        1.0e-12,
                    )
                    vector_tail_risk_norm = vector_tail_risk / 0.5
                    tail_parity_scale = min(query_risk_norm, vector_tail_risk_norm)
                    tail_score = (
                        valid_tail_gain
                        / max(vector_tail_risk_norm, 1.0e-12)
                        * tail_parity_scale
                    )
                    if base_scheme == "voi_sagelocalvalidtailmrr_harmonic_qproject":
                        score = max(score, tail_score)
                    else:
                        cap = _local_tail_capacity_cap(base_scheme)
                        score = _floored_capacity_envelope(
                            np.asarray([score, tail_score], dtype=np.float64),
                            floor=1.0 - cap,
                            cap=cap,
                        )
            elif base_scheme in {
                "voi_sageenv025mrr_harmonic_qproject",
                "voi_sageenv05mrr_harmonic_qproject",
                "voi_sageenv1mrr_harmonic_qproject",
                "voi_sagecvar4mrr_harmonic_qproject",
                "voi_sagecvarsqrtmrr_harmonic_qproject",
                "voi_sagevcvarsqrtmrr_harmonic_qproject",
                "voi_sagevtopsqrtmrr_harmonic_qproject",
                "voi_sagestudcvarmrr_harmonic_qproject",
                "voi_sagespectralmrr_harmonic_qproject",
                "voi_sageholder15mrr_harmonic_qproject",
                "voi_sageholder2mrr_harmonic_qproject",
                "voi_sagenestedcvarmrr_harmonic_qproject",
                "voi_sagehiercvarmrr_harmonic_qproject",
                "voi_sagerepaircap10mrr_harmonic_qproject",
                "voi_sagerepaircap25mrr_harmonic_qproject",
                "voi_sagerepaircap50mrr_harmonic_qproject",
                "voi_sageflowmrr_harmonic_qproject",
                "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
                "voi_sagemetricavgmrr_harmonic_qproject",
                "voi_sagemetricenvmrr_harmonic_qproject",
                "voi_sagefloorcapmrr_harmonic_qproject",
            }:
                score = query_score
            elif base_scheme == "voi_sagecovmrr_harmonic_qproject":
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / (2.0 * dual_query_sensitivity)
                vector_score_norm = vector_score / dual_vector_sensitivity
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
            elif base_scheme in {
                "voi_sagesqcovmrr_harmonic_qproject",
                "voi_sagesoftcovmrr_harmonic_qproject",
                "voi_sagebreadthmrr_harmonic_qproject",
            }:
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / (2.0 * dual_query_sensitivity)
                vector_score_norm = vector_score / dual_vector_sensitivity
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = max(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
            elif base_scheme == "voi_sageparetomrr_harmonic_qproject":
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / (2.0 * dual_query_sensitivity)
                vector_score_norm = vector_score / (2.0 * dual_vector_sensitivity)
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = min(
                    query_score_norm / max(query_risk_norm, 1.0e-12),
                    vector_score_norm / max(vector_risk_norm, 1.0e-12),
                ) * parity_scale
            elif base_scheme == "voi_sageharmmrr_harmonic_qproject":
                query_risk_norm = dual_query_risk / dual_query_sensitivity
                vector_risk_norm = dual_vector_risk / dual_vector_sensitivity
                query_score_norm = query_score / (2.0 * dual_query_sensitivity)
                vector_score_norm = vector_score / (2.0 * dual_vector_sensitivity)
                query_rel = query_score_norm / max(query_risk_norm, 1.0e-12)
                vector_rel = vector_score_norm / max(vector_risk_norm, 1.0e-12)
                rel_sum = query_rel + vector_rel
                balanced_rel = 0.0 if rel_sum <= 0.0 else 2.0 * query_rel * vector_rel / rel_sum
                parity_scale = min(query_risk_norm, vector_risk_norm)
                score = balanced_rel * parity_scale
            elif base_scheme == "voi_relmaxsqrtlevscaled_harmonic_qproject":
                parity_scale = min(dual_query_risk, dual_vector_risk)
                score = max(query_score / dual_query_risk, vector_score / dual_vector_risk) * parity_scale
            elif base_scheme == "voi_relmaxlevscaled_harmonic_qproject":
                parity_scale = min(dual_query_risk, dual_vector_risk)
                score = max(query_score / dual_query_risk, vector_score / dual_vector_risk) * parity_scale
            else:
                score = max(query_score / dual_query_risk, vector_score / dual_vector_risk)
            normalizer = (
                float(rrc_delta_eff[block_id])
                if base_scheme == "voi_relmaxposeff_harmonic_qproject" and rrc_delta_eff is not None
                else delta
            )
            if base_scheme in {
                "voi_relmaxposnorm_harmonic_qproject",
                "voi_dpstateposnorm_harmonic_qproject",
                "voi_dpstatecovsplit_harmonic_qproject",
                "voi_dpstatesqrtcovsplit_harmonic_qproject",
                "voi_dpstatesplittail_harmonic_qproject",
                "voi_dpstatecovtail_harmonic_qproject",
                "voi_dpstatebreadthtail_harmonic_qproject",
                "voi_dpstateriskmaxmrr_harmonic_qproject",
                "voi_dpstateriskenv025mrr_harmonic_qproject",
                "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
                "voi_sageordergain_harmonic_qproject",
                "voi_sagemrr_harmonic_qproject",
                "voi_sageroutecmrr_harmonic_qproject",
                "voi_sagecovmrr_harmonic_qproject",
                "voi_sagesqcovmrr_harmonic_qproject",
                "voi_sagesoftcovmrr_harmonic_qproject",
                "voi_sagebreadthmrr_harmonic_qproject",
                "voi_sageblockmrr_harmonic_qproject",
                "voi_sagelocalblockmrr_harmonic_qproject",
                "voi_sagelocalvalidtailmrr_harmonic_qproject",
                "voi_sagelocaltailcap65mrr_harmonic_qproject",
                "voi_sagelocaltailcap75mrr_harmonic_qproject",
                "voi_sagelocaltailcap90mrr_harmonic_qproject",
                "voi_sagelocaltailcap95mrr_harmonic_qproject",
                "voi_sagenrlocalmrr_harmonic_qproject",
                "voi_sagecellblockmrr_harmonic_qproject",
                "voi_sagegraphcvarmrr_harmonic_qproject",
                "voi_sagegraphsqrtcvarmrr_harmonic_qproject",
                "voi_sageenv025mrr_harmonic_qproject",
                "voi_sageenv05mrr_harmonic_qproject",
                "voi_sageenv1mrr_harmonic_qproject",
                "voi_sagecvar4mrr_harmonic_qproject",
                "voi_sagecvarsqrtmrr_harmonic_qproject",
                "voi_sagevcvarsqrtmrr_harmonic_qproject",
                "voi_sagevtopsqrtmrr_harmonic_qproject",
                "voi_sagespectralmrr_harmonic_qproject",
                "voi_sagenestedcvarmrr_harmonic_qproject",
                "voi_sagehiercvarmrr_harmonic_qproject",
                "voi_sageflowmrr_harmonic_qproject",
                "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagerrcvalidfullmrr_harmonic_qproject",
                "voi_sagerrcvalidmaxdropmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkmrr_harmonic_qproject",
                "voi_sagerrcvalidtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidheadtopkgainmrr_harmonic_qproject",
                "voi_sagerrcvalidharmgainmrr_harmonic_qproject",
                "voi_sagerrcvalidlorenzgainmrr_harmonic_qproject",
                "voi_sagerrcnestedvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedtopkvalidmrr_harmonic_qproject",
                "voi_sagerrcnestedlorenzbalmrr_harmonic_qproject",
                "voi_sagerrcnestedparetovalidmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
                "voi_sagemetricavgmrr_harmonic_qproject",
                "voi_sagemetricenvmrr_harmonic_qproject",
                "voi_sagefloorcapmrr_harmonic_qproject",
                "voi_sageparetomrr_harmonic_qproject",
                "voi_sageharmmrr_harmonic_qproject",
            }:
                if base_scheme in {
                    "voi_sageenv025mrr_harmonic_qproject",
                    "voi_sageenv05mrr_harmonic_qproject",
                    "voi_sageenv1mrr_harmonic_qproject",
                    "voi_sagecvar4mrr_harmonic_qproject",
                    "voi_sagecvarsqrtmrr_harmonic_qproject",
                    "voi_sagevcvarsqrtmrr_harmonic_qproject",
                    "voi_sagevtopsqrtmrr_harmonic_qproject",
                    "voi_sagestudcvarmrr_harmonic_qproject",
                    "voi_sagespectralmrr_harmonic_qproject",
                    "voi_sageholder15mrr_harmonic_qproject",
                    "voi_sageholder2mrr_harmonic_qproject",
                    "voi_sagenestedcvarmrr_harmonic_qproject",
                    "voi_sagehiercvarmrr_harmonic_qproject",
                    "voi_sagerepaircap10mrr_harmonic_qproject",
                    "voi_sagerepaircap25mrr_harmonic_qproject",
                    "voi_sagerepaircap50mrr_harmonic_qproject",
                    "voi_sageflowmrr_harmonic_qproject",
                "voi_sagegraphcapmrr_harmonic_qproject",
                "voi_sagegraphcapmassmrr_harmonic_qproject",
                "voi_sagegraphcapfullmrr_harmonic_qproject",
                "voi_sageqvcapmrr_harmonic_qproject",
                "voi_sageqvcapenvmrr_harmonic_qproject",
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
                "voi_sageqvcaldromrr_harmonic_qproject",
                "voi_sageqvcaltopdromrr_harmonic_qproject",
                "voi_sageqvcalboxdromrr_harmonic_qproject",
                "voi_sageqvcaltopboxdromrr_harmonic_qproject",
                "voi_sageqvcalstaildromrr_harmonic_qproject",
                "voi_sageqvcalstailsqrtdromrr_harmonic_qproject",
                "voi_sageqvcaltailcapdromrr_harmonic_qproject",
                "voi_sageqvcalatomcapdromrr_harmonic_qproject",
                "voi_sageqvcalpairdromrr_harmonic_qproject",
                "voi_sageqvcalbalfrontdromrr_harmonic_qproject",
                "voi_sageqvcalmaxdromrr_harmonic_qproject",
                "voi_sageqvcalfrontdromrr_harmonic_qproject",
                "voi_sageqvcaljointdromrr_harmonic_qproject",
                "voi_sageqvcalflowguarddromrr_harmonic_qproject",
                "voi_sagevalidfullmaxmrr_harmonic_qproject",
                "voi_sagevalidfullcapmrr_harmonic_qproject",
                "voi_sagecapdromrr_harmonic_qproject",
                    "voi_dpstateriskmaxmrr_harmonic_qproject",
                    "voi_dpstateriskenv025mrr_harmonic_qproject",
                    "voi_dpstatevtailriskenv025mrr_harmonic_qproject",
                    "voi_sagemetricavgmrr_harmonic_qproject",
                    "voi_sagemetricenvmrr_harmonic_qproject",
                    "voi_sagefloorcapmrr_harmonic_qproject",
                }:
                    normalizer = 2.0
                else:
                    normalizer = 8.0 if base_scheme == "voi_sageharmmrr_harmonic_qproject" else 4.0
            scores[block_id] = float(score / max(normalizer, 1.0e-12))
        elif base_scheme in {
            "voi_mixlp2_qproject",
            "voi_mixlp4_qproject",
            "voi_mixlp8_qproject",
        }:
            if mixed_query_weights is None or mixed_vector_weights is None:
                raise AssertionError("mixed Lp weights must be initialized for mixed Lp VOI")
            before = np.abs(target_residual)
            after = _expected_abs_normal(target_residual - projected_repair, np.sqrt(noise_var))
            gain = before - after
            query_score = float(np.sum(mixed_query_weights[plan.target_qids] * gain))
            gain_by_qid = np.zeros_like(residual, dtype=np.float64)
            gain_by_qid[plan.target_qids] = gain
            vector_score = 0.0
            for vector_pos, target_block_id in enumerate(vector_block_ids):
                target_idx = blocks[target_block_id].query_indices
                vector_score += float(mixed_vector_weights[vector_pos]) * 0.5 * float(
                    np.sum(gain_by_qid[target_idx])
                )
            scores[block_id] = float((query_score + vector_score) / delta)
        elif base_scheme in {"voi_l4query_qproject", "voi_l4atom_qproject", "voi_l4dual_qproject"}:
            mean = target_residual - projected_repair
            after4 = mean**4 + 6.0 * mean * mean * noise_var + 3.0 * noise_var * noise_var
            query_gain4 = target_residual**4 - after4
            query_score = float(np.sum(query_gain4))
            if base_scheme == "voi_l4query_qproject":
                scores[block_id] = query_score / max(delta, 1.0e-12)
                continue
            after_abs = _expected_abs_normal(mean, np.sqrt(noise_var))
            abs_gain_by_qid = np.zeros_like(residual, dtype=np.float64)
            abs_gain_by_qid[plan.target_qids] = np.abs(target_residual) - after_abs
            vector_score = 0.0
            for vector_pos, target_block_id in enumerate(vector_block_ids):
                target_idx = blocks[target_block_id].query_indices
                after_tvd = before_vector_tvd[vector_pos] - 0.5 * float(np.sum(abs_gain_by_qid[target_idx]))
                vector_score += float(before_vector_tvd[vector_pos] ** 4 - after_tvd**4)
            if base_scheme == "voi_l4atom_qproject":
                scores[block_id] = (query_score + vector_score) / max(delta, 1.0e-12)
            else:
                normalized_query = query_score / max(1, int(residual.size))
                normalized_vector = vector_score / max(1, len(vector_block_ids))
                scores[block_id] = (normalized_query + normalized_vector) / max(delta, 1.0e-12)
        elif base_scheme == "voi_worstop_qproject":
            after_abs = _expected_abs_normal(target_residual - projected_repair, np.sqrt(noise_var))
            abs_gain_by_qid = np.zeros_like(residual, dtype=np.float64)
            abs_gain_by_qid[plan.target_qids] = np.abs(target_residual) - after_abs
            query_score = float(abs_gain_by_qid[worst_query_qid])
            vector_idx = blocks[worst_vector_block_id].query_indices
            vector_score = 0.5 * float(np.sum(abs_gain_by_qid[vector_idx]))
            parity_scale = min(worst_query_risk, worst_vector_risk)
            score = max(query_score / worst_query_risk, vector_score / worst_vector_risk) * parity_scale
            scores[block_id] = float(score / max(delta, 1.0e-12))
        else:
            raise ValueError(f"Unknown operator value-of-information scheme {scheme!r}")
    if _scheme_has_modifier(scheme, "cover"):
        coverage = np.asarray([block.coverage_weight for block in blocks], dtype=np.float64)
        scores = scores * coverage
    histcover_lambda = _scheme_history_coverage_lambda(scheme)
    if histcover_lambda is not None:
        scores = scores * _history_coverage_attenuation(blocks, selected_block_ids, histcover_lambda)
    return scores


def _block_signal_components(
    blocks: list[AdaptiveBlock],
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    measurement_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = np.zeros(len(blocks), dtype=np.float64)
    tail = np.zeros(len(blocks), dtype=np.float64)
    tvd = np.zeros(len(blocks), dtype=np.float64)
    scan = np.zeros(len(blocks), dtype=np.float64)
    sigma = float(measurement_sigma)
    tau_l1 = math.sqrt(2.0 / math.pi) * sigma
    for block_id, block in enumerate(blocks):
        idx = block.query_indices
        err = true_answers[idx].astype(np.float64, copy=False) - syn_answers[idx].astype(np.float64, copy=False)
        abs_err = np.abs(err)
        delta = max(float(block.delta_l2), 1.0e-12)
        support[block_id] = _adaptive_order_excess(abs_err, sigma, normalize="none") / delta
        scan[block_id] = _adaptive_order_excess(abs_err, sigma, normalize="sqrtm") / delta
        tau_universal = sigma * math.sqrt(2.0 * math.log(float(len(idx) + 1)))
        tail[block_id] = _top_sum(np.maximum(abs_err - tau_universal, 0.0), 1) / delta
        if block.is_vector and len(idx) > 1:
            tvd[block_id] = 0.5 * float(np.sum(np.maximum(abs_err - tau_l1, 0.0))) / delta
    return support, tail, tvd, scan


def _score_blocks_value_of_information(
    scheme: str,
    blocks: list[AdaptiveBlock],
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    measurement_sigma: float,
    overlap: np.ndarray,
    max_overlap_mass: float,
) -> np.ndarray:
    base_scheme = _scheme_base(scheme)
    support, tail, tvd, scan = _block_signal_components(blocks, true_answers, syn_answers, measurement_sigma)
    if base_scheme in {"voi_support_global", "voi_support_project", "voi_support_qproject"}:
        demand = support
        capacity = support
    elif base_scheme in {"voi_scan_global", "voi_scan_project", "voi_scan_qproject"}:
        demand = scan
        capacity = scan
    elif base_scheme in {"voi_envelope_global", "voi_envelope_project", "voi_envelope_qproject"}:
        demand = np.maximum.reduce([support, tail, tvd])
        capacity = support
    elif base_scheme in {"voi_envelope_tailcap_global", "voi_envelope_tailcap_project", "voi_envelope_tailcap_qproject"}:
        demand = np.maximum.reduce([support, tail, tvd])
        capacity = np.maximum(support, tail)
    else:
        raise ValueError(f"Unknown value-of-information scheme {scheme!r}")
    propagated_demand = np.asarray(overlap @ demand, dtype=np.float64) / max(float(max_overlap_mass), 1.0e-12)
    scores = np.minimum(np.maximum(capacity, 0.0), np.maximum(propagated_demand, 0.0))
    if _scheme_has_modifier(scheme, "cover"):
        coverage = np.asarray([block.coverage_weight for block in blocks], dtype=np.float64)
        scores = scores * coverage
    return scores



def _scheme_base(scheme: str) -> str:
    base = scheme[:-7] if scheme.endswith("_repeat") else scheme
    for suffix in (
        "_mdl_family_amortized_entropy",
        "_mdl_family_amortized025",
        "_mdl_family_amortized033",
        "_mdl_family_amortized",
        "_mdl_family",
        "_histcover2",
        "_histcover1",
        "_histcover05",
        "_cover",
    ):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _scheme_allows_repeat(scheme: str) -> bool:
    return scheme.endswith("_repeat")


def _certified_private_selection_sensitivity(scheme: str) -> float:
    base_scheme = _scheme_base(scheme)
    if base_scheme in {
        "voi_sageordergain_harmonic_qproject",
        "aim_l1",
        "aim_l1_floor",
    }:
        return 1.0
    raise ValueError(
        f"Scheme {base_scheme!r} has no registered private-selection sensitivity certificate"
    )


def _scheme_has_modifier(scheme: str, modifier: str) -> bool:
    base = scheme[:-7] if scheme.endswith("_repeat") else scheme
    return base.endswith(f"_{modifier}")


def _sagedp_confirmation_floor_alpha(base_scheme: str) -> float:
    if "floor05" in base_scheme:
        return 0.05
    if "floor10" in base_scheme:
        return 0.10
    if "floor25" in base_scheme:
        return 0.25
    if "floor50" in base_scheme:
        return 0.50
    raise ValueError(f"Unknown confirmation-floor scheme {base_scheme!r}")


def _is_sagedp_stage_switch_floor_scheme(base_scheme: str) -> bool:
    return base_scheme in {
        "voi_sagedpstageswitchfloor05mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor10mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor25mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor50mrr_harmonic_qproject",
    }


def _is_sagedp_stage_hold_floor_scheme(base_scheme: str) -> bool:
    return base_scheme in {
        "voi_sagedpstageholdfloor05mrr_harmonic_qproject",
        "voi_sagedpstageholdfloor10mrr_harmonic_qproject",
    }


def _is_sagedp_stage_floor_scheme(base_scheme: str) -> bool:
    return _is_sagedp_stage_switch_floor_scheme(base_scheme) or _is_sagedp_stage_hold_floor_scheme(base_scheme)


def _is_sagedp_confirmation_floor_scheme(base_scheme: str) -> bool:
    return base_scheme in {
        "voi_sagedpconfirmfloor25mrr_harmonic_qproject",
        "voi_sagedpconfirmfloor50mrr_harmonic_qproject",
    }


def _scheme_history_coverage_lambda(scheme: str) -> float | None:
    base = scheme[:-7] if scheme.endswith("_repeat") else scheme
    if base.endswith("_histcover05"):
        return 0.5
    if base.endswith("_histcover1"):
        return 1.0
    if base.endswith("_histcover2"):
        return 2.0
    return None


def _history_coverage_attenuation(
    blocks: list[AdaptiveBlock],
    selected_block_ids: list[int] | None,
    strength: float,
) -> np.ndarray:
    if not selected_block_ids:
        return np.ones(len(blocks), dtype=np.float64)
    max_attr = max((max(block.scope) if block.scope else -1 for block in blocks), default=-1)
    if max_attr < 0:
        return np.ones(len(blocks), dtype=np.float64)
    attr_counts = np.zeros(max_attr + 1, dtype=np.float64)
    for block_id_raw in selected_block_ids:
        block_id = int(block_id_raw)
        if block_id < 0 or block_id >= len(blocks):
            continue
        for attr in blocks[block_id].scope:
            if 0 <= int(attr) <= max_attr:
                attr_counts[int(attr)] += 1.0
    attenuation = np.ones(len(blocks), dtype=np.float64)
    for block_id, block in enumerate(blocks):
        if not block.scope:
            continue
        repeated_mass = float(np.mean([attr_counts[int(attr)] for attr in block.scope if 0 <= int(attr) <= max_attr]))
        attenuation[block_id] = math.exp(-float(strength) * repeated_mass)
    return attenuation


def _selected_family_counts(blocks: list[AdaptiveBlock], selected_block_ids: list[int] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not selected_block_ids:
        return counts
    for block_id_raw in selected_block_ids:
        block_id = int(block_id_raw)
        if block_id < 0 or block_id >= len(blocks):
            continue
        family = blocks[block_id].family
        counts[family] = counts.get(family, 0) + 1
    return counts


def _sagedp_stage_floor_diagnostics(
    blocks: list[AdaptiveBlock],
    selected_block_ids: list[int] | None,
    state_weight_answers: np.ndarray | None,
    syn_answers: np.ndarray,
    vector_pressure_limit: float | None,
) -> dict[str, Any]:
    family_counts = _selected_family_counts(blocks, selected_block_ids)
    state_diag = _dp_state_tail_pressure_diagnostics(
        blocks,
        state_weight_answers,
        syn_answers,
        root_power=0.5,
        pressure_mode="vector",
    )
    twoway_count = int(family_counts.get("twoway", 0))
    oneway_count = int(family_counts.get("oneway", 0))
    vector_pressure = float(state_diag["dp_state_vector_pressure"])
    active = (
        float(state_diag["dp_state_has_transcript"]) > 0.0
        and twoway_count >= 25
        and oneway_count >= 10
        and math.isfinite(vector_pressure)
        and (vector_pressure_limit is None or vector_pressure <= float(vector_pressure_limit))
    )
    return {
        "sagedp_stage_switch_active": float(int(active)),
        "sagedp_stage_switch_twoway_count": float(twoway_count),
        "sagedp_stage_switch_oneway_count": float(oneway_count),
        "sagedp_stage_switch_vector_pressure": vector_pressure,
    }


def _sagedp_stage_switch_diagnostics(
    blocks: list[AdaptiveBlock],
    selected_block_ids: list[int] | None,
    state_weight_answers: np.ndarray | None,
    syn_answers: np.ndarray,
) -> dict[str, Any]:
    return _sagedp_stage_floor_diagnostics(
        blocks,
        selected_block_ids,
        state_weight_answers,
        syn_answers,
        vector_pressure_limit=1.40,
    )


def _sagedp_stage_hold_diagnostics(
    blocks: list[AdaptiveBlock],
    selected_block_ids: list[int] | None,
    state_weight_answers: np.ndarray | None,
    syn_answers: np.ndarray,
) -> dict[str, Any]:
    return _sagedp_stage_floor_diagnostics(
        blocks,
        selected_block_ids,
        state_weight_answers,
        syn_answers,
        vector_pressure_limit=None,
    )


def _dp_state_tail_pressure_weight(
    blocks: list[AdaptiveBlock],
    state_weight_answers: np.ndarray | None,
    syn_answers: np.ndarray,
    root_power: float = 0.5,
    pressure_mode: str = "max",
) -> float:
    return float(
        _dp_state_tail_pressure_diagnostics(
            blocks,
            state_weight_answers,
            syn_answers,
            root_power=root_power,
            pressure_mode=pressure_mode,
        )["dp_state_gate_weight"]
    )


def _dp_state_tail_pressure_diagnostics(
    blocks: list[AdaptiveBlock],
    state_weight_answers: np.ndarray | None,
    syn_answers: np.ndarray,
    root_power: float = 0.5,
    pressure_mode: str = "max",
) -> dict[str, Any]:
    if state_weight_answers is None:
        return {
            "dp_state_has_transcript": 0.0,
            "dp_state_gate_weight": 0.10,
            "dp_state_pressure_mode": pressure_mode,
            "dp_state_query_pressure": math.nan,
            "dp_state_vector_pressure": math.nan,
            "dp_state_pressure": math.nan,
            "dp_state_tail_fraction": math.nan,
            "dp_state_abs_mean": math.nan,
            "dp_state_abs_max": math.nan,
            "dp_state_vector_tvd_mean": math.nan,
            "dp_state_vector_tvd_max": math.nan,
        }
    state_residual = state_weight_answers.astype(np.float64, copy=False) - syn_answers.astype(np.float64, copy=False)
    abs_state = np.abs(state_residual)
    if abs_state.size == 0 or float(np.sum(abs_state)) <= 0.0:
        return {
            "dp_state_has_transcript": 1.0,
            "dp_state_gate_weight": 0.50,
            "dp_state_pressure_mode": pressure_mode,
            "dp_state_query_pressure": 1.0,
            "dp_state_vector_pressure": 1.0,
            "dp_state_pressure": 1.0,
            "dp_state_tail_fraction": 0.0,
            "dp_state_abs_mean": 0.0,
            "dp_state_abs_max": 0.0,
            "dp_state_vector_tvd_mean": 0.0,
            "dp_state_vector_tvd_max": 0.0,
        }
    abs_mean = float(np.mean(abs_state))
    abs_max = float(np.max(abs_state))
    query_pressure = float(np.max(abs_state)) / max(float(np.mean(abs_state)), 1.0e-12)
    vector_tvd = np.asarray(
        [
            0.5 * float(np.sum(abs_state[block.query_indices]))
            for block in blocks
            if block.is_vector and int(block.query_indices.size) > 1
        ],
        dtype=np.float64,
    )
    vector_tvd_mean = float(np.mean(vector_tvd)) if vector_tvd.size > 0 else 0.0
    vector_tvd_max = float(np.max(vector_tvd)) if vector_tvd.size > 0 else 0.0
    vector_pressure = (
        float(np.max(vector_tvd)) / max(float(np.mean(vector_tvd)), 1.0e-12)
        if vector_tvd.size > 0 and float(np.sum(vector_tvd)) > 0.0
        else 1.0
    )
    if pressure_mode == "query":
        pressure = max(1.0, query_pressure)
    elif pressure_mode == "vector":
        pressure = max(1.0, vector_pressure)
    elif pressure_mode == "geomean":
        pressure = max(1.0, math.sqrt(max(query_pressure, 1.0) * max(vector_pressure, 1.0)))
    elif pressure_mode == "max":
        pressure = max(1.0, query_pressure, vector_pressure)
    else:
        raise ValueError(f"Unknown DP-state pressure mode {pressure_mode!r}")
    tail_fraction = 1.0 - 1.0 / math.pow(pressure, float(root_power))
    weight = float(np.clip(0.50 - 0.40 * tail_fraction, 0.10, 0.50))
    return {
        "dp_state_has_transcript": 1.0,
        "dp_state_gate_weight": weight,
        "dp_state_pressure_mode": pressure_mode,
        "dp_state_query_pressure": float(query_pressure),
        "dp_state_vector_pressure": float(vector_pressure),
        "dp_state_pressure": float(pressure),
        "dp_state_tail_fraction": float(tail_fraction),
        "dp_state_abs_mean": abs_mean,
        "dp_state_abs_max": abs_max,
        "dp_state_vector_tvd_mean": vector_tvd_mean,
        "dp_state_vector_tvd_max": vector_tvd_max,
    }


def _scheme_mdl_family_amortized_alpha(scheme: str, family_counts: dict[str, int] | None = None) -> float | None:
    base = scheme[:-7] if scheme.endswith("_repeat") else scheme
    if base.endswith("_mdl_family_amortized_entropy"):
        if not family_counts:
            raise ValueError("family_counts are required for entropy-amortized MDL prior")
        num_families = max(1, len(family_counts))
        num_blocks = max(num_families + 1, sum(int(count) for count in family_counts.values()))
        return math.log(float(num_families)) / math.log(float(num_blocks))
    if base.endswith("_mdl_family_amortized025"):
        return 0.25
    if base.endswith("_mdl_family_amortized033"):
        return 1.0 / 3.0
    if base.endswith("_mdl_family_amortized"):
        return 0.5
    return None


def _round_score_scheme(scheme: str, round_id: int, rounds: int) -> str:
    base = _scheme_base(scheme)
    suffix = "_repeat" if _scheme_allows_repeat(scheme) else ""
    if base == "hybrid_cover_tail_last1":
        return "tail_guard_noise_power_cover" if round_id > max(0, rounds - 1) else "noise_adaptive_top_linear_cover"
    if base == "hybrid_cover_tail_last2":
        return "tail_guard_noise_power_cover" if round_id > max(0, rounds - 2) else "noise_adaptive_top_linear_cover"
    if base == "hybrid_cover_tail_last3":
        return "tail_guard_noise_power_cover" if round_id > max(0, rounds - 3) else "noise_adaptive_top_linear_cover"
    if base == "hybrid_tophalf2_aim":
        return ("tophalf_pos_floor" if round_id <= 2 else "aim_l1_floor") + suffix
    if base == "hybrid_tophalf3_aim":
        return ("tophalf_pos_floor" if round_id <= 3 else "aim_l1_floor") + suffix
    if base == "hybrid_tophalf4_aim":
        return ("tophalf_pos_floor" if round_id <= 4 else "aim_l1_floor") + suffix
    if base == "hybrid_aim3_tophalf":
        return ("aim_l1_floor" if round_id <= 3 else "tophalf_pos_floor") + suffix
    if base == "hybrid_aim4_l2":
        return ("aim_l1_floor" if round_id <= 4 else "l2_net_clip") + suffix
    if base == "hybrid_tophalf_aim_last":
        switch_round = max(1, int(math.ceil(rounds / 2.0)))
        return ("tophalf_pos_floor" if round_id <= switch_round else "aim_l1_floor") + suffix
    return scheme


def _forced_sequence_names(scheme: str) -> list[str] | None:
    base = _scheme_base(scheme)
    sequences = {
        "oracle_aim7_q6331": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:8:13",
            "twoway:11:12",
            "mixed:13:10:q6331",
        ],
        "oracle_aim6_q6331_1112": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:8:13",
            "mixed:13:10:q6331",
            "twoway:11:12",
        ],
        "oracle_aim5_q6331_813_1112": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "mixed:13:10:q6331",
            "twoway:8:13",
            "twoway:11:12",
        ],
        "oracle_aim7_7351": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:8:13",
            "twoway:11:12",
            "mixed:13:12:q7351",
        ],
        "oracle_aim5_1112_011_813": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:11:12",
            "twoway:0:11",
            "twoway:8:13",
        ],
        "oracle_aim5_011_813_1112": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:0:11",
            "twoway:8:13",
            "twoway:11:12",
        ],
        "oracle_aim5_011_1112_813": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:0:11",
            "twoway:11:12",
            "twoway:8:13",
        ],
        "oracle_aim7_1213": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:8:13",
            "twoway:11:12",
            "twoway:12:13",
        ],
        "oracle_aim5_813_011_1213": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:8:13",
            "twoway:0:11",
            "twoway:12:13",
        ],
        "oracle_aim5_813_1213_011": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:8:13",
            "twoway:12:13",
            "twoway:0:11",
        ],
        "oracle_aim5_011_813_1213": [
            "twoway:3:4",
            "twoway:2:13",
            "twoway:2:10",
            "twoway:5:11",
            "twoway:1:11",
            "twoway:0:11",
            "twoway:8:13",
            "twoway:12:13",
        ],
    }
    return sequences.get(base)


def _block_scope(qcat: QueryCatalogue, query_indices: np.ndarray) -> tuple[int, ...]:
    attrs: set[int] = set()
    for qid_raw in query_indices.tolist():
        qid = int(qid_raw)
        for term_idx in range(int(qcat.num_terms[qid])):
            attr = int(qcat.attrs[qid, term_idx])
            if attr >= 0:
                attrs.add(attr)
        for term_idx in range(int(qcat.linear_num_terms[qid])):
            attr = int(qcat.linear_attrs[qid, term_idx])
            if attr >= 0:
                attrs.add(attr)
    return tuple(sorted(attrs))


def _attach_coverage_weights(blocks: list[AdaptiveBlock]) -> list[AdaptiveBlock]:
    raw_weights: list[float] = []
    scope_sets = [set(block.scope) for block in blocks]
    for scope in scope_sets:
        overlap = sum(len(scope & other) for other in scope_sets)
        raw_weights.append(float(max(1, overlap)))
    max_weight = max(raw_weights) if raw_weights else 1.0
    return [
        AdaptiveBlock(
            name=block.name,
            family=block.family,
            query_indices=block.query_indices,
            delta_l2=block.delta_l2,
            is_vector=block.is_vector,
            scope=block.scope,
            coverage_weight=float(raw_weights[idx] / max_weight),
        )
        for idx, block in enumerate(blocks)
    ]


def _build_blocks(groups: list[WorkloadGroup], qcat: QueryCatalogue) -> list[AdaptiveBlock]:
    blocks: list[AdaptiveBlock] = []
    for group in groups:
        if group.is_partition:
            query_indices = group.query_indices.astype(np.int32, copy=True)
            blocks.append(
                AdaptiveBlock(
                    name=group.name,
                    family=group.family,
                    query_indices=query_indices,
                    delta_l2=float(group.sensitivity_l2),
                    is_vector=True,
                    scope=_block_scope(qcat, query_indices),
                    coverage_weight=1.0,
                )
            )
        else:
            for qid in group.query_indices.tolist():
                query_indices = np.asarray([int(qid)], dtype=np.int32)
                blocks.append(
                    AdaptiveBlock(
                        name=f"{group.name}:q{int(qid)}",
                        family=group.family,
                        query_indices=query_indices,
                        delta_l2=1.0,
                        is_vector=False,
                        scope=_block_scope(qcat, query_indices),
                        coverage_weight=1.0,
                    )
                )
    return _attach_coverage_weights(blocks)


def _sample_exponential(
    scores: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    sensitivity: float = 1.0,
) -> int:
    return sample_exponential_mechanism(
        scores,
        float(epsilon),
        float(sensitivity),
        rng,
    )


def _rank_log_base_measure(scores: np.ndarray, *, max_odds: float) -> np.ndarray:
    """Map released scores to a bounded, scale-free conditional base measure."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Base-measure scores must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("Base-measure scores must be finite")
    if not np.isfinite(max_odds) or float(max_odds) < 1.0:
        raise ValueError("max_odds must be finite and at least one")
    if values.size == 1 or math.isclose(float(max_odds), 1.0):
        return np.zeros(values.size, dtype=np.float64)

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    percentile = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * float(start + stop - 1)
        percentile[order[start:stop]] = average_rank / float(values.size - 1)
        start = stop
    # The additive shift is immaterial to sampling. Keeping the best log weight
    # at zero makes the base-measure odds easy to audit.
    return math.log(float(max_odds)) * (percentile - 1.0)


def _apply_released_score_base_measure(
    private_scores: np.ndarray,
    released_scores: np.ndarray,
    *,
    epsilon: float,
    sensitivity: float,
    max_odds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a released-transcript prior without changing private sensitivity."""
    private = np.asarray(private_scores, dtype=np.float64)
    released = np.asarray(released_scores, dtype=np.float64)
    if private.shape != released.shape or private.ndim != 1:
        raise ValueError("Private and released score vectors must have the same shape")
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(sensitivity) or float(sensitivity) <= 0.0:
        raise ValueError("sensitivity must be positive and finite")
    log_base_measure = _rank_log_base_measure(released, max_odds=float(max_odds))
    adjusted = private + (
        2.0 * float(sensitivity) / float(epsilon)
    ) * log_base_measure
    return adjusted, log_base_measure


def _select_from_scores(
    scores: np.ndarray,
    *,
    rule: str,
    epsilon: float,
    rng: np.random.Generator,
    sensitivity: float = 1.0,
) -> int:
    if rule == "sample":
        return _sample_exponential(scores, epsilon, rng, sensitivity=sensitivity)
    if rule == "argmax":
        return int(np.argmax(scores.astype(np.float64, copy=False)))
    raise ValueError(f"Unknown selection rule {rule!r}")


def _public_bootstrap_scores(
    blocks: list[AdaptiveBlock],
    available_ids: list[int],
    strategy: str,
) -> np.ndarray:
    scores: list[float] = []
    for block_id in available_ids:
        block = blocks[int(block_id)]
        block_size = max(1, int(block.query_indices.size))
        coverage = max(0.0, float(block.coverage_weight))
        scope_size = max(1, len(block.scope))
        if strategy == "coverage":
            score = coverage * math.log1p(float(block_size))
        elif strategy == "low_order_coverage":
            score = coverage * math.log1p(float(block_size)) / math.sqrt(float(scope_size))
        else:
            raise ValueError(f"Unknown public bootstrap strategy {strategy!r}")
        scores.append(float(score))
    return np.asarray(scores, dtype=np.float64)


def _transcript_selector_answers(
    *,
    previous_projected_answers: np.ndarray | None,
    previous_variances: np.ndarray | None,
    syn_answers: np.ndarray,
    untrusted_variance: float,
    reliability_mode: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    if previous_projected_answers is None or previous_variances is None:
        return None, np.zeros_like(syn_answers, dtype=bool)
    projected = previous_projected_answers.astype(np.float64, copy=False)
    variances = previous_variances.astype(np.float64, copy=False)
    syn = syn_answers.astype(np.float64, copy=False)
    if projected.shape[0] != syn.shape[0] or variances.shape[0] != syn.shape[0]:
        raise ValueError("Transcript selector state has incompatible query dimension")
    trusted = np.isfinite(variances) & (variances < float(untrusted_variance))
    if reliability_mode == "all":
        return projected.copy(), np.ones_like(trusted, dtype=bool)
    if reliability_mode != "hard":
        raise ValueError(f"Unknown transcript reliability mode {reliability_mode!r}")
    selector_answers = syn.copy()
    selector_answers[trusted] = projected[trusted]
    return selector_answers, trusted


def _query_error_metrics_with_tvd(
    blocks: list[AdaptiveBlock],
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    n_real: int,
    n_syn: int,
    prefix: str,
) -> dict[str, float]:
    metrics = query_error_metrics(true_answers, syn_answers, n_real, n_syn, prefix=prefix)
    tvds: list[float] = []
    true_rate = true_answers.astype(np.float64) / float(n_real)
    syn_rate = syn_answers.astype(np.float64) / float(n_syn)
    for block in blocks:
        if not block.is_vector:
            continue
        idx = block.query_indices
        if idx.size <= 1:
            continue
        tvds.append(float(0.5 * np.sum(np.abs(syn_rate[idx] - true_rate[idx]))))
    metrics[f"{prefix}_avg_tvd"] = float(np.mean(tvds)) if tvds else 0.0
    metrics[f"{prefix}_max_tvd"] = float(np.max(tvds)) if tvds else 0.0
    return metrics


def _score_block(
    scheme: str,
    block: AdaptiveBlock,
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    n_real: int,
    measurement_sigma: float,
) -> float:
    idx = block.query_indices
    true = true_answers[idx].astype(np.float64, copy=False)
    syn = syn_answers[idx].astype(np.float64, copy=False)
    err = true - syn
    delta = max(float(block.delta_l2), 1.0e-12)
    base_scheme = _scheme_base(scheme)
    k = int(len(idx))
    abs_err = np.abs(err)
    l1 = float(np.sum(abs_err))
    tau_l1 = math.sqrt(2.0 / math.pi) * float(measurement_sigma)
    tau_universal = float(measurement_sigma) * math.sqrt(2.0 * math.log(float(k + 1)))
    tau_universal_plus_mean = tau_l1 + tau_universal
    l1_floor = l1 - tau_l1 * k
    l1_clip_floor = max(0.0, l1_floor)
    positive_excess = np.maximum(abs_err - tau_l1, 0.0)
    l1_pos_floor = float(np.sum(positive_excess))
    l2_pos_floor = float(np.linalg.norm(positive_excess, ord=2))
    l2_net = float(np.linalg.norm(err, ord=2) / delta - float(measurement_sigma) * _chi_mean(k))
    l2_clip = max(0.0, l2_net)
    universal_excess = np.maximum(abs_err - tau_universal, 0.0)
    universal_plus_excess = np.maximum(abs_err - tau_universal_plus_mean, 0.0)
    if base_scheme == "aim_l1":
        score = float(l1 / delta)
    elif base_scheme == "aim_l1_floor":
        score = float(l1_floor)
    elif base_scheme == "aim_l1_clip_floor":
        score = float(l1_clip_floor)
    elif base_scheme == "aim_l1_pos_floor":
        score = float(l1_pos_floor)
    elif base_scheme == "aim_l1_floor_sqrtk":
        score = float(l1_floor / math.sqrt(max(1, k)))
    elif base_scheme == "aim_l1_floor_mean":
        score = float(l1_floor / max(1, k))
    elif base_scheme == "aim_l1_pos_floor_sqrtk":
        score = float(l1_pos_floor / math.sqrt(max(1, k)))
    elif base_scheme == "aim_l1_pos_floor_mean":
        score = float(l1_pos_floor / max(1, k))
    elif base_scheme == "top1_pos_floor":
        score = float(np.max(np.maximum(abs_err - tau_l1, 0.0))) if k > 0 else 0.0
    elif base_scheme == "toplog_pos_floor":
        score = _top_sum(np.maximum(abs_err - tau_l1, 0.0), int(math.ceil(math.log2(k + 1))))
    elif base_scheme == "topsqrt_pos_floor":
        score = _top_sum(np.maximum(abs_err - tau_l1, 0.0), int(math.ceil(math.sqrt(k))))
    elif base_scheme == "tophalf_pos_floor":
        score = _top_sum(np.maximum(abs_err - tau_l1, 0.0), int(math.ceil(k / 2.0)))
    elif base_scheme == "topquarter_pos_floor":
        score = _top_sum(np.maximum(abs_err - tau_l1, 0.0), int(math.ceil(k / 4.0)))
    elif base_scheme == "topthird_pos_floor":
        score = _top_sum(np.maximum(abs_err - tau_l1, 0.0), int(math.ceil(k / 3.0)))
    elif base_scheme == "top2third_pos_floor":
        score = _top_sum(np.maximum(abs_err - tau_l1, 0.0), int(math.ceil(2.0 * k / 3.0)))
    elif base_scheme == "top3quarter_pos_floor":
        score = _top_sum(np.maximum(abs_err - tau_l1, 0.0), int(math.ceil(3.0 * k / 4.0)))
    elif base_scheme == "noise_adaptive_top":
        keep = int(math.ceil(k / (1.0 + float(measurement_sigma) * float(measurement_sigma))))
        score = _top_sum(positive_excess, max(1, keep))
    elif base_scheme == "noise_adaptive_top_linear":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _top_sum(positive_excess, max(1, keep))
    elif base_scheme == "l2_top_linear":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _top_l2(positive_excess, keep) / delta
    elif base_scheme == "prop_l2_top_linear":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = (_top_l2(positive_excess, keep) / delta) * _public_propagation_weight(block)
    elif base_scheme == "l2_bridge_top_linear":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        top_l1 = _top_sum(positive_excess, keep)
        top_l2 = _top_l2(positive_excess, keep)
        normalized_l1 = top_l1 / math.sqrt(max(1, keep))
        score = (0.5 * top_l2 + 0.5 * normalized_l1) / delta
    elif base_scheme == "prop_l2_bridge_top_linear":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        top_l1 = _top_sum(positive_excess, keep)
        top_l2 = _top_l2(positive_excess, keep)
        normalized_l1 = top_l1 / math.sqrt(max(1, keep))
        score = ((0.5 * top_l2 + 0.5 * normalized_l1) / delta) * _public_propagation_weight(block)
    elif base_scheme == "l1_l2mix_top_linear_10":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _l1_l2_mix_top_score(positive_excess, keep, 0.10) / delta
    elif base_scheme == "l1_l2mix_top_linear_25":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _l1_l2_mix_top_score(positive_excess, keep, 0.25) / delta
    elif base_scheme == "l1_l2mix_top_linear_50":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _l1_l2_mix_top_score(positive_excess, keep, 0.50) / delta
    elif base_scheme == "l1_l2mix_noise_linear":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        alpha = 1.0 / (1.0 + float(measurement_sigma))
        score = _l1_l2_mix_top_score(positive_excess, keep, alpha) / delta
    elif base_scheme == "l1_l2mix_noise_power":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        sigma = float(measurement_sigma)
        alpha = 1.0 / (1.0 + sigma * sigma)
        score = _l1_l2_mix_top_score(positive_excess, keep, alpha) / delta
    elif base_scheme == "l1_l2mix_prop_noise_power":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        sigma = float(measurement_sigma)
        alpha = (1.0 / (1.0 + sigma * sigma)) * _public_propagation_weight(block)
        score = _l1_l2_mix_top_score(positive_excess, keep, alpha) / delta
    elif base_scheme == "l1_l2mix_cov_noise_power":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        sigma = float(measurement_sigma)
        alpha = (1.0 / (1.0 + sigma * sigma)) * float(block.coverage_weight)
        score = _l1_l2_mix_top_score(positive_excess, keep, alpha) / delta
    elif base_scheme == "tail_guard_noise_power":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        sigma = float(measurement_sigma)
        alpha = 1.0 / (1.0 + sigma * sigma)
        top_l1 = _top_sum(positive_excess, keep)
        tail = _top_sum(universal_excess, 1)
        score = ((1.0 - alpha) * top_l1 + alpha * tail) / delta
    elif base_scheme == "l1_l2_tail_guard_noise_power":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        sigma = float(measurement_sigma)
        alpha = 1.0 / (1.0 + sigma * sigma)
        top_l1 = _top_sum(positive_excess, keep)
        top_l2 = _top_l2(positive_excess, keep)
        tail = _top_sum(universal_excess, 1)
        score = ((1.0 - alpha) * top_l1 + alpha * 0.5 * (top_l2 + tail)) / delta
    elif base_scheme == "propagated_entropic_excess":
        score = _entropic_excess_risk(positive_excess, measurement_sigma) / delta
    elif base_scheme == "l1_l2cal_top_linear_10":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _l1_l2_calibrated_top_score(positive_excess, keep, 0.10) / delta
    elif base_scheme == "l1_l2cal_top_linear_25":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _l1_l2_calibrated_top_score(positive_excess, keep, 0.25) / delta
    elif base_scheme == "l1_l2cal_top_linear_50":
        keep = _noise_adaptive_keep_linear(k, measurement_sigma)
        score = _l1_l2_calibrated_top_score(positive_excess, keep, 0.50) / delta
    elif base_scheme == "adaptive_order_l1":
        score = _adaptive_order_excess(abs_err, measurement_sigma, normalize="none")
    elif base_scheme == "adaptive_order_l1_sqrtm":
        score = _adaptive_order_excess(abs_err, measurement_sigma, normalize="sqrtm")
    elif base_scheme == "adaptive_order_l1_mdl":
        score = _adaptive_order_excess(abs_err, measurement_sigma, normalize="mdl")
    elif base_scheme == "adaptive_order_tail_power":
        sigma = float(measurement_sigma)
        alpha = 1.0 / (1.0 + sigma * sigma)
        support = _adaptive_order_excess(abs_err, measurement_sigma, normalize="none")
        tail = _top_sum(universal_excess, 1)
        score = ((1.0 - alpha) * support + alpha * tail) / delta
    elif base_scheme == "adaptive_order_tail_linear":
        sigma = float(measurement_sigma)
        alpha = 1.0 / (1.0 + sigma)
        support = _adaptive_order_excess(abs_err, measurement_sigma, normalize="none")
        tail = _top_sum(universal_excess, 1)
        score = ((1.0 - alpha) * support + alpha * tail) / delta
    elif base_scheme == "adaptive_order_tail_half":
        support = _adaptive_order_excess(abs_err, measurement_sigma, normalize="none")
        tail = _top_sum(universal_excess, 1)
        score = 0.5 * (support + tail) / delta
    elif base_scheme == "adaptive_rank_l1":
        score = _adaptive_rank_excess(abs_err, measurement_sigma, norm="l1") / delta
    elif base_scheme == "adaptive_rank_l2":
        score = _adaptive_rank_excess(abs_err, measurement_sigma, norm="l2") / delta
    elif base_scheme == "adaptive_spectral_sqrt":
        score = _adaptive_spectral_excess(abs_err, measurement_sigma, weight="sqrt") / delta
    elif base_scheme == "adaptive_spectral_harmonic":
        score = _adaptive_spectral_excess(abs_err, measurement_sigma, weight="harmonic") / delta
    elif base_scheme == "adaptive_spectral_log":
        score = _adaptive_spectral_excess(abs_err, measurement_sigma, weight="log") / delta
    elif base_scheme == "excess_l2":
        score = l2_pos_floor / delta
    elif base_scheme == "excess_l2_l1blend":
        score = math.sqrt(max(0.0, l1_pos_floor * l2_pos_floor)) / delta
    elif base_scheme == "universal_l1":
        score = float(np.sum(universal_excess)) / delta
    elif base_scheme == "universal_l2":
        score = float(np.linalg.norm(universal_excess, ord=2)) / delta
    elif base_scheme == "universal_plus_l1":
        score = float(np.sum(universal_plus_excess)) / delta
    elif base_scheme == "universal_plus_l2":
        score = float(np.linalg.norm(universal_plus_excess, ord=2)) / delta
    elif base_scheme == "blend_pos_l2_25":
        score = float(0.75 * l1_pos_floor + 0.25 * l2_clip)
    elif base_scheme == "blend_pos_l2_50":
        score = float(0.50 * l1_pos_floor + 0.50 * l2_clip)
    elif base_scheme == "blend_pos_l2_75":
        score = float(0.25 * l1_pos_floor + 0.75 * l2_clip)
    elif base_scheme == "old_support_l1":
        support = float(np.linalg.norm(true, ord=2) / max(1, int(n_real)))
        score = float((support * l1) / 3.0)
    elif base_scheme == "support_true_pos_l1":
        support = float(np.linalg.norm(true, ord=2) / max(1, int(n_real)))
        score = float((support * l1_pos_floor) / 3.0)
    elif base_scheme == "support_union_pos_l1":
        support = float(max(np.linalg.norm(true, ord=2), np.linalg.norm(syn, ord=2)) / max(1, int(n_real)))
        score = float((support * l1_pos_floor) / 3.0)
    elif base_scheme == "l2_net":
        score = float(l2_net)
    elif base_scheme == "l2_net_clip":
        score = float(l2_clip)
    else:
        raise ValueError(f"Unknown adaptive selection scheme {scheme!r}")
    if _scheme_has_modifier(scheme, "cover"):
        score *= float(block.coverage_weight)
    return score


def _selected_block_diagnostics(
    block: AdaptiveBlock,
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    measurement_sigma: float,
) -> dict[str, float | int]:
    idx = block.query_indices
    err = true_answers[idx].astype(np.float64, copy=False) - syn_answers[idx].astype(np.float64, copy=False)
    abs_err = np.abs(err)
    positive_excess = np.maximum(abs_err - math.sqrt(2.0 / math.pi) * float(measurement_sigma), 0.0)
    k = int(len(idx))
    keep = _noise_adaptive_keep_linear(k, measurement_sigma)
    top_l1 = _top_sum(positive_excess, keep)
    top_l2 = _top_l2(positive_excess, keep)
    return {
        "selected_abs_l1": float(np.sum(abs_err)),
        "selected_abs_l2": float(np.linalg.norm(abs_err, ord=2)),
        "selected_positive_excess_l1": float(np.sum(positive_excess)),
        "selected_positive_excess_l2": float(np.linalg.norm(positive_excess, ord=2)),
        "selected_topm_keep_linear": int(keep),
        "selected_topm_excess_l1": float(top_l1),
        "selected_topm_excess_l2": float(top_l2),
        "selected_topm_normalized_l1": float(top_l1 / math.sqrt(max(1, keep))),
        "selected_public_propagation_weight": float(_public_propagation_weight(block)),
    }


def _rank_descending(values: np.ndarray, selected_pos: int) -> int:
    selected_value = float(values[int(selected_pos)])
    return int(1 + np.sum(values > selected_value))


def _dpblend_score_diagnostics(
    *,
    score_scheme: str,
    blocks: list[AdaptiveBlock],
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    measurement_sigma: float,
    repair_plans: list[QueryRepairPlan | None],
    active_query_budget: int,
    kappa_noise: float,
    selected_block_ids: list[int],
    state_weight_answers: np.ndarray | None,
    round_id: int,
    rounds: int,
    row_count: int,
    available_ids: list[int],
    chosen_block_id: int,
) -> dict[str, Any]:
    base_scheme = _scheme_base(score_scheme)
    if base_scheme not in {
        "voi_sagedpblend10mrr_harmonic_qproject",
        "voi_sagedpblend25mrr_harmonic_qproject",
        "voi_sagedpblend50mrr_harmonic_qproject",
        "voi_sagedpblenddecay5010mrr_harmonic_qproject",
        "voi_sagedpblendramp1050mrr_harmonic_qproject",
        "voi_sagedpblendmax1050mrr_harmonic_qproject",
        "voi_sagedpblendstategatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstateqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstatevgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstategate4rtmrr_harmonic_qproject",
        "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpconfirmfloor25mrr_harmonic_qproject",
        "voi_sagedpconfirmfloor50mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor05mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor10mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor25mrr_harmonic_qproject",
        "voi_sagedpstageswitchfloor50mrr_harmonic_qproject",
        "voi_sagedpstageholdfloor05mrr_harmonic_qproject",
        "voi_sagedpstageholdfloor10mrr_harmonic_qproject",
    }:
        return {}

    local_all = _score_blocks_operator_value_of_information(
        "voi_sagelocaltailblend04mrr_harmonic_qproject",
        blocks,
        true_answers,
        syn_answers,
        measurement_sigma,
        repair_plans,
        active_query_budget=active_query_budget,
        kappa_noise=kappa_noise,
        selected_block_ids=selected_block_ids,
        state_weight_answers=state_weight_answers,
        round_id=round_id,
        rounds=rounds,
        row_count=row_count,
    ).astype(np.float64, copy=False)
    dp_all = _score_blocks_operator_value_of_information(
        "voi_dpstatecovsplit_harmonic_qproject",
        blocks,
        true_answers,
        syn_answers,
        measurement_sigma,
        repair_plans,
        active_query_budget=active_query_budget,
        kappa_noise=kappa_noise,
        selected_block_ids=selected_block_ids,
        state_weight_answers=state_weight_answers,
        round_id=round_id,
        rounds=rounds,
        row_count=row_count,
    ).astype(np.float64, copy=False)
    local_scores = np.asarray([local_all[block_id] for block_id in available_ids], dtype=np.float64)
    dp_scores = np.asarray([dp_all[block_id] for block_id in available_ids], dtype=np.float64)
    blend10_scores = (local_scores + 0.10 * dp_scores) / 1.10
    blend50_scores = (local_scores + 0.50 * dp_scores) / 1.50

    chosen_pos = int(available_ids.index(int(chosen_block_id)))
    local_argmax_pos = int(np.argmax(local_scores))
    dp_argmax_pos = int(np.argmax(dp_scores))
    blend10_argmax_pos = int(np.argmax(blend10_scores))
    blend50_argmax_pos = int(np.argmax(blend50_scores))
    stage_diag: dict[str, Any] = {}
    stage_active = False
    if _is_sagedp_stage_floor_scheme(base_scheme):
        if _is_sagedp_stage_hold_floor_scheme(base_scheme):
            stage_diag = _sagedp_stage_hold_diagnostics(
                blocks,
                selected_block_ids,
                state_weight_answers,
                syn_answers,
            )
        else:
            stage_diag = _sagedp_stage_switch_diagnostics(
                blocks,
                selected_block_ids,
                state_weight_answers,
                syn_answers,
            )
        stage_active = float(stage_diag["sagedp_stage_switch_active"]) > 0.0

    if base_scheme == "voi_sagedpblend10mrr_harmonic_qproject":
        dp_weight = 0.10
    elif base_scheme == "voi_sagedpblend25mrr_harmonic_qproject":
        dp_weight = 0.25
    elif base_scheme == "voi_sagedpblend50mrr_harmonic_qproject":
        dp_weight = 0.50
    elif base_scheme == "voi_sagedpblenddecay5010mrr_harmonic_qproject":
        progress = 0.0 if rounds <= 1 else float(round_id - 1) / float(rounds - 1)
        dp_weight = 0.50 - 0.40 * progress
    elif base_scheme == "voi_sagedpblendramp1050mrr_harmonic_qproject":
        progress = 0.0 if rounds <= 1 else float(round_id - 1) / float(rounds - 1)
        dp_weight = 0.10 + 0.40 * progress
    elif base_scheme == "voi_sagedpblendstategate4rtmrr_harmonic_qproject":
        dp_weight = _dp_state_tail_pressure_weight(blocks, state_weight_answers, syn_answers, root_power=0.25)
    elif base_scheme == "voi_sagedpblendstateqvgatesqrtmrr_harmonic_qproject":
        dp_weight = _dp_state_tail_pressure_weight(
            blocks,
            state_weight_answers,
            syn_answers,
            root_power=0.5,
            pressure_mode="geomean",
        )
    elif base_scheme == "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject":
        dp_weight = _dp_state_tail_pressure_weight(
            blocks,
            state_weight_answers,
            syn_answers,
            root_power=0.5,
            pressure_mode="geomean",
        )
    elif base_scheme in {
        "voi_sagedpconfirmfloor25mrr_harmonic_qproject",
        "voi_sagedpconfirmfloor50mrr_harmonic_qproject",
    }:
        dp_weight = math.nan
    elif _is_sagedp_stage_floor_scheme(base_scheme):
        dp_weight = (
            math.nan
            if stage_active
            else _dp_state_tail_pressure_weight(blocks, state_weight_answers, syn_answers, root_power=0.5)
        )
    elif base_scheme == "voi_sagedpblendstatevgatesqrtmrr_harmonic_qproject":
        dp_weight = _dp_state_tail_pressure_weight(
            blocks,
            state_weight_answers,
            syn_answers,
            root_power=0.5,
            pressure_mode="vector",
        )
    elif base_scheme == "voi_sagedpblendstategatesqrtmrr_harmonic_qproject":
        dp_weight = _dp_state_tail_pressure_weight(blocks, state_weight_answers, syn_answers, root_power=0.5)
    else:
        dp_weight = math.nan
    if base_scheme in {
        "voi_sagedpconfirmfloor25mrr_harmonic_qproject",
        "voi_sagedpconfirmfloor50mrr_harmonic_qproject",
    }:
        floor_alpha = _sagedp_confirmation_floor_alpha(base_scheme)
        dp_scores_for_gate = np.minimum(dp_scores, local_scores)
        gate_scores = np.maximum(dp_scores_for_gate, float(floor_alpha) * local_scores)
    elif _is_sagedp_stage_floor_scheme(base_scheme):
        floor_alpha = _sagedp_confirmation_floor_alpha(base_scheme)
        if stage_active:
            dp_scores_for_gate = np.minimum(dp_scores, local_scores)
            gate_scores = np.maximum(dp_scores_for_gate, float(floor_alpha) * local_scores)
        else:
            dp_scores_for_gate = dp_scores
            gate_weight = _dp_state_tail_pressure_weight(blocks, state_weight_answers, syn_answers, root_power=0.5)
            gate_scores = (local_scores + float(gate_weight) * dp_scores_for_gate) / (1.0 + float(gate_weight))
    else:
        floor_alpha = math.nan
        dp_scores_for_gate = np.minimum(dp_scores, local_scores) if base_scheme == "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject" else dp_scores
        gate_scores = (local_scores + float(dp_weight) * dp_scores_for_gate) / (1.0 + float(dp_weight)) if math.isfinite(float(dp_weight)) else np.maximum(blend10_scores, blend50_scores)
    gate_argmax_pos = int(np.argmax(gate_scores))
    state_root_power = 0.25 if base_scheme == "voi_sagedpblendstategate4rtmrr_harmonic_qproject" else 0.5
    if base_scheme in {
        "voi_sagedpblendstateqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject",
    }:
        state_pressure_mode = "geomean"
    elif base_scheme == "voi_sagedpblendstatevgatesqrtmrr_harmonic_qproject":
        state_pressure_mode = "vector"
    else:
        state_pressure_mode = "max"
    state_diag = _dp_state_tail_pressure_diagnostics(
        blocks,
        state_weight_answers,
        syn_answers,
        root_power=state_root_power,
        pressure_mode=state_pressure_mode,
    )
    if base_scheme not in {
        "voi_sagedpblendstategatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstateqvgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstatevgatesqrtmrr_harmonic_qproject",
        "voi_sagedpblendstategate4rtmrr_harmonic_qproject",
        "voi_sagedpconfirmqvgatesqrtmrr_harmonic_qproject",
    } and math.isfinite(float(dp_weight)):
        state_diag["dp_state_gate_weight"] = float(dp_weight)

    selected_local = float(local_scores[chosen_pos])
    selected_dp = float(dp_scores[chosen_pos])
    selected_dp_capped = float(dp_scores_for_gate[chosen_pos])
    selected_blend10 = float(blend10_scores[chosen_pos])
    selected_blend50 = float(blend50_scores[chosen_pos])
    selected_gate = float(gate_scores[chosen_pos])
    local_argmax_block = blocks[int(available_ids[local_argmax_pos])]
    dp_argmax_block = blocks[int(available_ids[dp_argmax_pos])]
    blend10_argmax_block = blocks[int(available_ids[blend10_argmax_pos])]
    blend50_argmax_block = blocks[int(available_ids[blend50_argmax_pos])]
    gate_argmax_block = blocks[int(available_ids[gate_argmax_pos])]
    return {
        **state_diag,
        **stage_diag,
        "score_diag_selected_local_score": selected_local,
        "score_diag_selected_dpstate_score": selected_dp,
        "score_diag_selected_dpstate_capped_score": selected_dp_capped,
        "score_diag_confirmation_floor_alpha": float(floor_alpha),
        "score_diag_selected_dp_minus_local": selected_dp - selected_local,
        "score_diag_selected_dp_over_local": selected_dp / max(abs(selected_local), 1.0e-12),
        "score_diag_selected_blend10_score": selected_blend10,
        "score_diag_selected_blend50_score": selected_blend50,
        "score_diag_selected_gate_score": selected_gate,
        "score_diag_selected_local_rank": _rank_descending(local_scores, chosen_pos),
        "score_diag_selected_dpstate_rank": _rank_descending(dp_scores, chosen_pos),
        "score_diag_selected_blend10_rank": _rank_descending(blend10_scores, chosen_pos),
        "score_diag_selected_blend50_rank": _rank_descending(blend50_scores, chosen_pos),
        "score_diag_selected_gate_rank": _rank_descending(gate_scores, chosen_pos),
        "score_diag_local_argmax_name": local_argmax_block.name,
        "score_diag_dpstate_argmax_name": dp_argmax_block.name,
        "score_diag_blend10_argmax_name": blend10_argmax_block.name,
        "score_diag_blend50_argmax_name": blend50_argmax_block.name,
        "score_diag_gate_argmax_name": gate_argmax_block.name,
        "score_diag_blend10_blend50_disagree": float(
            int(available_ids[blend10_argmax_pos] != available_ids[blend50_argmax_pos])
        ),
        "score_diag_selected_prefers_blend50": float(selected_blend50 > selected_blend10),
        "score_diag_chosen_is_local_argmax": float(int(chosen_pos == local_argmax_pos)),
        "score_diag_chosen_is_dpstate_argmax": float(int(chosen_pos == dp_argmax_pos)),
        "score_diag_chosen_is_blend10_argmax": float(int(chosen_pos == blend10_argmax_pos)),
        "score_diag_chosen_is_blend50_argmax": float(int(chosen_pos == blend50_argmax_pos)),
        "score_diag_chosen_is_gate_argmax": float(int(chosen_pos == gate_argmax_pos)),
    }


def _make_initial_synthetic(n_rows: int, cardinalities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cols = [rng.integers(0, int(card), size=n_rows, dtype=np.int32) for card in cardinalities.tolist()]
    return np.stack(cols, axis=1).astype(np.int32)


def _merge_adaptive_measurement_groups(
    *,
    num_queries: int,
    base_groups: list[MeasurementGroup],
    blocks: list[AdaptiveBlock],
    block_measurement_counts: dict[int, int],
    block_measurement_rho: dict[int, float],
) -> list[MeasurementGroup]:
    """Keep projection groups aligned when a partial base transcript gains new blocks."""
    groups = [
        MeasurementGroup(
            query_indices=group.query_indices.astype(np.int32, copy=True),
            sensitivity_l2=float(group.sensitivity_l2),
            rho=float(group.rho),
            sigma=float(group.sigma),
            noise_std=float(group.noise_std),
            name=str(group.name),
            family=str(group.family),
            is_partition=bool(group.is_partition),
        )
        for group in base_groups
    ]
    for block_id, repeat_count in sorted(block_measurement_counts.items()):
        block = blocks[int(block_id)]
        idx = block.query_indices.astype(np.int32, copy=False)
        idx_set = set(int(value) for value in idx.tolist())
        exact_pos: int | None = None
        overlapping: list[int] = []
        for pos, group in enumerate(groups):
            group_set = set(int(value) for value in group.query_indices.tolist())
            if not idx_set.isdisjoint(group_set):
                overlapping.append(pos)
            if idx_set == group_set:
                exact_pos = pos

        base_rho = 0.0
        if exact_pos is not None and float(groups[exact_pos].rho) > 0.0:
            base_rho = float(groups[exact_pos].rho)
            groups.pop(exact_pos)
        else:
            for pos in reversed(overlapping):
                group = groups[pos]
                if float(group.rho) > 0.0:
                    raise ValueError(
                        f"Adaptive block {block.name!r} partially overlaps measured base group "
                        f"{group.name!r}"
                    )
                keep = np.asarray(
                    [int(qid) for qid in group.query_indices.tolist() if int(qid) not in idx_set],
                    dtype=np.int32,
                )
                if keep.size:
                    groups[pos] = MeasurementGroup(
                        query_indices=keep,
                        sensitivity_l2=float(group.sensitivity_l2),
                        rho=0.0,
                        sigma=float(group.sigma),
                        noise_std=float(group.noise_std),
                        name=str(group.name),
                        family=str(group.family),
                        is_partition=False,
                    )
                else:
                    groups.pop(pos)

        combined_rho = base_rho + float(block_measurement_rho[int(block_id)])
        effective_sigma = math.sqrt(1.0 / (2.0 * combined_rho))
        groups.append(
            MeasurementGroup(
                query_indices=idx.astype(np.int32, copy=True),
                sensitivity_l2=float(block.delta_l2),
                rho=combined_rho,
                sigma=effective_sigma,
                noise_std=float(block.delta_l2) * effective_sigma,
                name=f"adaptive:{block_id}:{block.name}:repeats={repeat_count}",
                family=block.family,
                is_partition=bool(block.is_vector),
            )
        )

    coverage = np.zeros(int(num_queries), dtype=np.int32)
    for group in groups:
        coverage[group.query_indices] += 1
    if not np.all(coverage == 1):
        raise ValueError(
            "Combined base/adaptive measurement groups must cover each query exactly once; "
            f"missing={int(np.sum(coverage == 0))}, overlapping={int(np.sum(coverage > 1))}"
        )
    return groups


def _measurement_artifact(
    *,
    qcat: QueryCatalogue,
    schema: TableSchema,
    blocks: list[AdaptiveBlock],
    measurement_records: list[MeasurementRecord],
    current_syn_answers: np.ndarray,
    total_rows: int,
    cardinalities: np.ndarray,
    projection_cfg: dict[str, Any],
    output_dir: Path,
    epsilon: float,
    measurement_sigma: float,
    delta: float,
    selection_rho_per_round: float | None = None,
    base_measurements: Measurements | None = None,
) -> Path:
    artifact_dir = ensure_dir(output_dir)
    qcat.save_json(artifact_dir / "queries.json")
    schema.save_json(artifact_dir / "schema.json")
    if base_measurements is None:
        target = current_syn_answers.astype(np.float32, copy=True)
        variances = np.full(qcat.m, 1.0e12, dtype=np.float32)
        weighted_targets = np.zeros(qcat.m, dtype=np.float64)
        inv_variances = np.zeros(qcat.m, dtype=np.float64)
        groups: list[MeasurementGroup] = []
        coverage_rho = 0.0
    else:
        if (
            base_measurements.target_noisy.shape != (qcat.m,)
            or base_measurements.variances.shape != (qcat.m,)
        ):
            raise ValueError("Base coverage measurement shape does not match the query catalogue")
        target = base_measurements.target_noisy.astype(np.float32, copy=True)
        variances = np.maximum(
            base_measurements.variances.astype(np.float64, copy=True),
            1.0e-12,
        ).astype(np.float32)
        inv_variances = 1.0 / variances.astype(np.float64)
        weighted_targets = target.astype(np.float64) * inv_variances
        groups = list(base_measurements.groups)
        coverage_rho = float(base_measurements.rho_spent)
    block_measurement_counts: dict[int, int] = {}
    block_measurement_rho: dict[int, float] = {}
    block_measurement_sigmas: dict[int, list[float]] = {}
    rho_select = float(epsilon) ** 2 / 8.0 if selection_rho_per_round is None else float(selection_rho_per_round)
    default_rho_measure = 1.0 / (2.0 * float(measurement_sigma) ** 2)
    selection_rhos: list[float] = []
    selection_epsilons: list[float] = []
    for record in measurement_records:
        block = blocks[int(record.block_id)]
        block_measurement_counts[int(record.block_id)] = (
            block_measurement_counts.get(int(record.block_id), 0) + 1
        )
        record_sigma = (
            float(measurement_sigma)
            if record.measurement_sigma is None
            else float(record.measurement_sigma)
        )
        if not np.isfinite(record_sigma) or record_sigma <= 0.0:
            raise ValueError("Adaptive measurement sigma must be positive and finite")
        record_rho = 1.0 / (2.0 * record_sigma * record_sigma)
        record_selection_rho = (
            float(rho_select) if record.selection_rho is None else float(record.selection_rho)
        )
        if not np.isfinite(record_selection_rho) or record_selection_rho < 0.0:
            raise ValueError("Adaptive selection rho must be finite and non-negative")
        record_selection_epsilon = (
            math.sqrt(8.0 * record_selection_rho)
            if record.selection_epsilon is None
            else float(record.selection_epsilon)
        )
        if not np.isfinite(record_selection_epsilon) or record_selection_epsilon < 0.0:
            raise ValueError("Adaptive selection epsilon must be finite and non-negative")
        if not math.isclose(
            record_selection_rho,
            record_selection_epsilon**2 / 8.0,
            rel_tol=1.0e-10,
            abs_tol=1.0e-15,
        ):
            raise ValueError("Adaptive selection epsilon/rho pair is inconsistent")
        selection_rhos.append(record_selection_rho)
        selection_epsilons.append(record_selection_epsilon)
        block_measurement_rho[int(record.block_id)] = (
            block_measurement_rho.get(int(record.block_id), 0.0) + record_rho
        )
        block_measurement_sigmas.setdefault(int(record.block_id), []).append(record_sigma)
        idx = block.query_indices
        noisy = record.noisy.astype(np.float64, copy=False)
        noise_std = float(block.delta_l2) * record_sigma
        variance = noise_std * noise_std
        inv_var = 1.0 / max(variance, 1.0e-12)
        weighted_targets[idx] += noisy * inv_var
        inv_variances[idx] += inv_var
    if base_measurements is None:
        for block_id, repeat_count in sorted(block_measurement_counts.items()):
            block = blocks[block_id]
            block_rho = float(block_measurement_rho[block_id])
            effective_sigma = math.sqrt(1.0 / (2.0 * block_rho))
            effective_noise_std = float(block.delta_l2) * effective_sigma
            groups.append(
                MeasurementGroup(
                    query_indices=block.query_indices.astype(np.int32, copy=True),
                    sensitivity_l2=float(block.delta_l2),
                    rho=block_rho,
                    sigma=effective_sigma,
                    noise_std=effective_noise_std,
                    name=f"adaptive:{block_id}:{block.name}:repeats={repeat_count}",
                    family=block.family,
                    is_partition=bool(block.is_vector),
                )
            )
    else:
        groups = _merge_adaptive_measurement_groups(
            num_queries=int(qcat.m),
            base_groups=groups,
            blocks=blocks,
            block_measurement_counts=block_measurement_counts,
            block_measurement_rho=block_measurement_rho,
        )
    measured = inv_variances > 0.0
    target[measured] = (weighted_targets[measured] / inv_variances[measured]).astype(np.float32)
    variances[measured] = (1.0 / inv_variances[measured]).astype(np.float32)
    unmeasured = np.flatnonzero(~measured).astype(np.int32)
    if base_measurements is None and unmeasured.size:
        unmeasured_std = float(np.sqrt(1.0e12))
        groups.append(
            MeasurementGroup(
                query_indices=unmeasured,
                sensitivity_l2=1.0,
                rho=0.0,
                sigma=unmeasured_std,
                noise_std=unmeasured_std,
                name="adaptive:unmeasured",
                family="unmeasured",
                is_partition=False,
            )
        )
    projected, projection_diagnostics = _apply_configured_projection(
        target,
        qcat,
        groups,
        int(total_rows),
        projection_cfg,
        variances,
        np.asarray(cardinalities, dtype=np.int32),
    )
    adaptive_measurement_rho = float(sum(block_measurement_rho.values()))
    adaptive_selection_rho = float(sum(selection_rhos))
    adaptive_rho_spent = adaptive_selection_rho + adaptive_measurement_rho
    spent = coverage_rho + adaptive_rho_spent
    epsilon_delta = zcdp_epsilon(spent, float(delta)) if spent > 0.0 else 0.0
    projection_diagnostics["adaptive_selection"] = {
        "enabled": True,
        "epsilon": float(epsilon),
        "measurement_sigma": float(measurement_sigma),
        "rho_select_per_round": float(rho_select),
        "rho_measure_per_round": float(default_rho_measure),
        "measurement_sigma_schedule": [
            float(measurement_sigma) if record.measurement_sigma is None else float(record.measurement_sigma)
            for record in measurement_records
        ],
        "selection_epsilon_schedule": selection_epsilons,
        "selection_rho_schedule": selection_rhos,
        "adaptive_selection_rho": adaptive_selection_rho,
        "adaptive_measurement_rho": adaptive_measurement_rho,
        "coverage_rho": float(coverage_rho),
        "adaptive_rho_spent": float(adaptive_rho_spent),
        "selected_blocks": len(measurement_records),
        "unmeasured_variance": 1.0e12,
        "base_measurements_present": bool(base_measurements is not None),
        "coverage_complete": bool(
            base_measurements is not None
            and np.all(base_measurements.variances.astype(np.float64) < 1.0e11)
        ),
        "adaptive_measurement_ledger": [
            {
                "block_id": int(block_id),
                "name": blocks[block_id].name,
                "family": blocks[block_id].family,
                "repeat_count": int(repeat_count),
                "rho": float(block_measurement_rho[block_id]),
                "measurement_sigmas": [float(value) for value in block_measurement_sigmas[block_id]],
                "num_queries": int(len(blocks[block_id].query_indices)),
            }
            for block_id, repeat_count in sorted(block_measurement_counts.items())
        ],
    }
    write_json(
        {
            "mode": "dp",
            "rho_total": spent,
            "rho_spent": spent,
            "delta": float(delta),
            "epsilon_delta": epsilon_delta,
            "target_noisy": target.tolist(),
            "target_projected": projected.astype(np.float32).tolist(),
            "variances": variances.tolist(),
            "groups": [group.to_dict() for group in groups],
            "projection_diagnostics": projection_diagnostics,
            "num_rows": int(total_rows),
        },
        artifact_dir / "measurements.json",
    )
    return artifact_dir


def _configure_a_generator(
    config: dict[str, Any],
    *,
    output_dir: Path,
    init_path: Path,
    measurement_dir: Path,
    inner_iters: int,
    generator_profile: str = "adaptive_legacy",
    generator_seed: int | None = None,
    stop_patience: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    set_nested(cfg, "run.output_dir", str(output_dir))
    set_nested(cfg, "init.encoded_npy", str(init_path))
    set_nested(cfg, "measurement.reuse_from", str(measurement_dir))
    set_nested(cfg, "measurement.artifact_dir", str(measurement_dir))
    set_nested(cfg, "privacy.measurement_mode", "static_all")
    if generator_profile == "adaptive_legacy":
        set_nested(cfg, "qdte.candidate_compiler", "single_query")
        set_nested(cfg, "qdte.transport_mode", "constructive_pair")
        set_nested(cfg, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(cfg, "qdte.constructive_pair_partner_limit", 16)
        set_nested(cfg, "qdte.constructive_pair_harm_query_limit", 16)
        set_nested(cfg, "qdte.constructive_pair_group_augment", False)
        set_nested(cfg, "qdte.objective_weighting", "variance")
    elif generator_profile == "qdte_standard":
        if str(cfg.get("qdte", {}).get("objective_weighting", "variance")) != "variance":
            raise ValueError("qdte_standard generator profile requires variance objective weighting")
        if str(cfg.get("qdte", {}).get("transport_mode", "")) != "atom_flow":
            raise ValueError("qdte_standard generator profile requires qdte.transport_mode='atom_flow'")
    else:
        raise ValueError(f"Unsupported adaptive generator profile: {generator_profile!r}")
    if generator_seed is not None:
        set_nested(cfg, "run.seed", int(generator_seed))
    set_nested(cfg, "qdte.max_iters", int(inner_iters))
    resolved_stop_patience = int(inner_iters) if stop_patience is None else int(stop_patience)
    if resolved_stop_patience <= 0:
        raise ValueError("generator stop_patience must be positive")
    set_nested(cfg, "qdte.stop_patience", resolved_stop_patience)
    set_nested(cfg, "qdte.kappa_noise", 1.0)
    set_nested(cfg, "qdte.allow_below_noise_fallback", False)
    set_nested(cfg, "evaluation.compute_true_query_error", False)
    set_nested(cfg, "evaluation.compute_heldout_query_error", False)
    set_nested(cfg, "evaluation.save_synthetic_csv", False)
    set_nested(cfg, "runtime.log_measurement_groups", False)
    set_nested(cfg, "runtime.xla_preallocate", False)
    return cfg


def _generator_stage_seed(base_seed: int, stage_id: int) -> int:
    """Derive a reproducible fresh QDTE RNG stream for one adaptive stage."""
    if int(stage_id) <= 0:
        raise ValueError("stage_id must be positive")
    state = np.random.SeedSequence(
        [int(base_seed) & 0xFFFFFFFF, 0x51445445, int(stage_id) & 0xFFFFFFFF]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _run_scheme(
    *,
    base_config: dict[str, Any],
    qcat: QueryCatalogue,
    schema: TableSchema,
    blocks: list[AdaptiveBlock],
    true_answers: np.ndarray,
    initial_syn: np.ndarray,
    cardinalities: np.ndarray,
    output_dir: Path,
    scheme: str,
    rounds: int,
    inner_iters: int,
    budget: BudgetConfig,
    rng: np.random.Generator,
    generator_profile: str = "adaptive_legacy",
    initial_fit_iters: int = 0,
    final_refit_iters: int = 0,
    base_measurements: Measurements | None = None,
    generator_seed_mode: str = "restart",
    initial_fit_stop_patience: int | None = None,
    inner_stop_patience: int | None = None,
    final_stop_patience: int | None = None,
    measurement_schedule: str = "fixed",
    selection_input: str = "oracle",
    bootstrap_strategy: str = "coverage",
    transcript_untrusted_variance: float = 1.0e11,
    transcript_reliability: str = "hard",
    selection_ledger: str = "conservative",
    selection_rule: str = "sample",
    public_bootstrap_rounds: int = 1,
    nonpositive_score_fallback: str = "none",
    transcript_sage_prior_odds: float = 1.0,
) -> dict[str, Any]:
    if generator_seed_mode not in {"restart", "per_stage"}:
        raise ValueError("generator_seed_mode must be 'restart' or 'per_stage'")
    if measurement_schedule not in {"fixed", "aim_released_change"}:
        raise ValueError("measurement_schedule must be 'fixed' or 'aim_released_change'")
    if int(initial_fit_iters) < 0:
        raise ValueError("initial_fit_iters must be non-negative")
    if selection_input not in {"oracle", "transcript"}:
        raise ValueError("selection_input must be 'oracle' or 'transcript'")
    if (
        not np.isfinite(transcript_sage_prior_odds)
        or float(transcript_sage_prior_odds) < 1.0
    ):
        raise ValueError("transcript_sage_prior_odds must be finite and at least one")
    if float(transcript_sage_prior_odds) > 1.0:
        if selection_input != "oracle":
            raise ValueError("The transcript SAGE prior requires private EM selection")
        if _scheme_base(scheme) != "aim_l1_floor":
            raise ValueError("The transcript SAGE prior currently requires aim_l1_floor")
        if base_measurements is None:
            raise ValueError("The transcript SAGE prior requires a released base transcript")
    selection_score_sensitivity = 1.0
    if (
        str(base_config.get("privacy", {}).get("mode", "dp")) == "dp"
        and selection_input == "oracle"
    ):
        if selection_rule != "sample":
            raise ValueError("Private-data selection in DP mode must use the exponential mechanism")
        if selection_ledger == "measurement_only":
            raise ValueError("Private-data selection in DP mode must charge selection privacy")
        selection_score_sensitivity = _certified_private_selection_sensitivity(scheme)
    scheme_dir = ensure_dir(output_dir / scheme)
    current_syn = initial_syn.copy()
    current_syn_path = scheme_dir / "initial_synthetic_encoded.npy"
    np.save(current_syn_path, current_syn)
    selected_block_ids: list[int] = []
    selected_set: set[int] = set()
    measurement_records: list[MeasurementRecord] = []
    rows: list[dict[str, Any]] = []
    n_real = int(initial_syn.shape[0])
    delta = float(base_config.get("privacy", {}).get("delta", 1.0e-9))
    allow_repeat = _scheme_allows_repeat(scheme)
    selection_epsilon = float(budget.epsilon)
    measurement_sigma = float(budget.measurement_sigma)
    coverage_rho = 0.0 if base_measurements is None else float(base_measurements.rho_spent)
    adaptive_rho_limit: float | None = None
    if measurement_schedule == "aim_released_change":
        if not allow_repeat:
            raise ValueError("AIM-style released-change annealing requires a repeat-enabled scheme")
        adaptive_rho_limit = float(budget.rho_total) - coverage_rho
        if adaptive_rho_limit <= 0.0:
            raise ValueError("No adaptive rho remains after coverage")
    family_counts = {
        family: int(sum(1 for block in blocks if block.family == family))
        for family in {block.family for block in blocks}
    }
    propagation_overlap: np.ndarray | None = None
    max_propagation_mass = 1.0
    repair_plans: list[QueryRepairPlan | None] | None = None
    forced_sequence = _forced_sequence_names(scheme)
    name_to_block_id = {block.name: idx for idx, block in enumerate(blocks)}
    if _is_operator_value_of_information_scheme(scheme):
        repair_plans = _query_repair_plans(qcat, blocks)
    elif _is_value_of_information_scheme(scheme):
        if _scheme_base(scheme).endswith("_qproject"):
            propagation_overlap, max_propagation_mass = _query_projection_matrix(qcat, blocks)
        elif _scheme_base(scheme).endswith("_project"):
            propagation_overlap, max_propagation_mass = _scope_projection_matrix(blocks)
        else:
            propagation_overlap, max_propagation_mass = _scope_overlap_matrix(blocks)

    previous_projected_answers = (
        None
        if base_measurements is None
        else base_measurements.target_projected.astype(np.float64, copy=True)
    )
    previous_variances = (
        None
        if base_measurements is None
        else base_measurements.variances.astype(np.float64, copy=True)
    )
    latest_measurement_dir: Path | None = None
    selection_rho_per_round = 0.0 if selection_ledger == "measurement_only" else None
    stage_seed_offset = 0
    warm_start_metadata: dict[str, Any] = {
        "enabled": int(initial_fit_iters) > 0,
        "requested_iters": int(initial_fit_iters),
    }
    if int(initial_fit_iters) > 0:
        if base_measurements is None:
            raise ValueError("An initial transcript fit requires base coverage measurements")
        warm_measurement_dir = _measurement_artifact(
            qcat=qcat,
            schema=schema,
            blocks=blocks,
            measurement_records=[],
            current_syn_answers=answer_queries(
                current_syn,
                qcat,
                batch_size=int(base_config.get("runtime", {}).get("answer_batch_size", 8192)),
            ),
            total_rows=n_real,
            cardinalities=cardinalities,
            projection_cfg=dict(base_config.get("projection", {})),
            output_dir=scheme_dir / "round_000_measurement",
            epsilon=selection_epsilon,
            measurement_sigma=measurement_sigma,
            delta=delta,
            selection_rho_per_round=selection_rho_per_round,
            base_measurements=base_measurements,
        )
        warm_generate_dir = scheme_dir / "round_000_generate"
        warm_config = _configure_a_generator(
            base_config,
            output_dir=warm_generate_dir,
            init_path=current_syn_path,
            measurement_dir=warm_measurement_dir,
            inner_iters=int(initial_fit_iters),
            generator_profile=generator_profile,
            generator_seed=(
                _generator_stage_seed(
                    int(base_config.get("run", {}).get("seed", 0)),
                    1,
                )
                if generator_seed_mode == "per_stage"
                else None
            ),
            stop_patience=initial_fit_stop_patience,
        )
        run_qdte(warm_config)
        current_syn_path = warm_generate_dir / "synthetic_encoded.npy"
        current_syn = np.load(current_syn_path).astype(np.int32)
        latest_measurement_dir = warm_measurement_dir
        stage_seed_offset = 1
        warm_start_metadata.update(
            {
                "output_dir": str(warm_generate_dir),
                "synthetic_encoded": str(current_syn_path),
                "measurement_dir": str(warm_measurement_dir),
            }
        )
    for round_id in range(1, rounds + 1):
        syn_answers = answer_queries(
            current_syn,
            qcat,
            batch_size=int(base_config.get("runtime", {}).get("answer_batch_size", 8192)),
        )
        pre_metrics = _query_error_metrics_with_tvd(
            blocks,
            true_answers,
            syn_answers,
            n_real,
            current_syn.shape[0],
            prefix="full_true",
        )
        round_measurement_sigma = float(measurement_sigma)
        round_measurement_rho = 1.0 / (2.0 * round_measurement_sigma**2)
        round_selection_epsilon = (
            0.0 if selection_ledger == "measurement_only" else float(selection_epsilon)
        )
        round_selection_rho = round_selection_epsilon**2 / 8.0
        round_exhausts_budget = False
        adaptive_measurement_rho_before = float(
            sum(
                1.0
                / (
                    2.0
                    * float(
                        budget.measurement_sigma
                        if record.measurement_sigma is None
                        else record.measurement_sigma
                    )
                    ** 2
                )
                for record in measurement_records
            )
        )
        adaptive_selection_rho_before = float(
            sum(
                0.0 if record.selection_rho is None else float(record.selection_rho)
                for record in measurement_records
            )
        )
        adaptive_rho_before = adaptive_measurement_rho_before + adaptive_selection_rho_before
        if measurement_schedule == "aim_released_change":
            if adaptive_rho_limit is None:
                raise RuntimeError("Missing adaptive rho limit")
            remaining_rho = float(adaptive_rho_limit) - adaptive_rho_before
            tolerance = 1.0e-12 * max(1.0, float(adaptive_rho_limit))
            if remaining_rho <= tolerance:
                break
            round_plan = _plan_annealed_private_round(
                current_measurement_sigma=measurement_sigma,
                current_selection_epsilon=selection_epsilon,
                remaining_rho=remaining_rho,
                remaining_rounds=int(rounds) - int(round_id) + 1,
                charge_selection=selection_ledger != "measurement_only",
            )
            round_measurement_sigma = float(round_plan.measurement_sigma)
            round_measurement_rho = float(round_plan.measurement_rho)
            round_selection_epsilon = float(round_plan.selection_epsilon)
            round_selection_rho = float(round_plan.selection_rho)
            round_exhausts_budget = bool(round_plan.exhausts_budget)
        available_ids = [idx for idx in range(len(blocks)) if allow_repeat or idx not in selected_set]
        if not available_ids:
            break
        score_scheme = _round_score_scheme(scheme, round_id, rounds)
        scoring_answers = true_answers
        trusted_mask = np.ones(int(qcat.m), dtype=bool)
        score_input_source = selection_input
        if selection_input == "transcript":
            transcript_answers, trusted_mask = _transcript_selector_answers(
                previous_projected_answers=previous_projected_answers,
                previous_variances=previous_variances,
                syn_answers=syn_answers,
                untrusted_variance=transcript_untrusted_variance,
                reliability_mode=transcript_reliability,
            )
            if round_id <= int(public_bootstrap_rounds):
                score_input_source = "public_bootstrap"
            elif transcript_answers is not None:
                scoring_answers = transcript_answers
            else:
                score_input_source = "public_bootstrap"
        if forced_sequence is not None and round_id <= len(forced_sequence):
            forced_name = forced_sequence[round_id - 1]
            if forced_name not in name_to_block_id:
                raise ValueError(f"Forced sequence block {forced_name!r} not found")
            chosen_block_id = int(name_to_block_id[forced_name])
            if not allow_repeat and chosen_block_id in selected_set:
                raise ValueError(f"Forced sequence repeats unavailable block {forced_name!r}")
            scores = np.zeros(len(available_ids), dtype=np.float64)
            chosen_local = int(available_ids.index(chosen_block_id))
            score_scheme = f"forced:{forced_name}"
        elif score_input_source == "public_bootstrap":
            scores = _public_bootstrap_scores(blocks, available_ids, bootstrap_strategy)
            chosen_local = _select_from_scores(
                scores,
                rule=selection_rule,
                epsilon=round_selection_epsilon,
                rng=rng,
                sensitivity=selection_score_sensitivity,
            )
            chosen_block_id = int(available_ids[chosen_local])
            score_scheme = f"public_bootstrap:{bootstrap_strategy}"
        elif _is_operator_value_of_information_scheme(score_scheme):
            if repair_plans is None:
                repair_plans = _query_repair_plans(qcat, blocks)
            conditional_context = (
                _build_conditional_propagation_context(
                    qcat,
                    blocks,
                    current_syn,
                    batch_size=int(base_config.get("runtime", {}).get("answer_batch_size", 8192)),
                )
                if _scheme_base(score_scheme)
                in {
                    "voi_condprop_atomlev_harmonic_qproject",
                    "voi_condctx_atomlev_harmonic_qproject",
                    "voi_condbridge_atomlev_harmonic_qproject",
                    "voi_condnorm_atomlev_harmonic_qproject",
                    "voi_condbridgenorm_atomlev_harmonic_qproject",
                    "voi_precinnov_atomlev_harmonic_qproject",
                }
                else None
            )
            all_scores = _score_blocks_operator_value_of_information(
                score_scheme,
                blocks,
                scoring_answers,
                syn_answers,
                round_measurement_sigma,
                repair_plans,
                active_query_budget=int(base_config.get("qdte", {}).get("num_active_targets", 64)),
                kappa_noise=float(base_config.get("qdte", {}).get("kappa_noise", 1.0)),
                selected_block_ids=selected_block_ids,
                conditional_context=conditional_context,
                state_weight_answers=scoring_answers if selection_input == "transcript" else previous_projected_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=n_real,
            )
            scores = np.asarray([all_scores[block_id] for block_id in available_ids], dtype=np.float64)
        elif _is_value_of_information_scheme(score_scheme):
            if propagation_overlap is None:
                if _scheme_base(score_scheme).endswith("_qproject"):
                    propagation_overlap, max_propagation_mass = _query_projection_matrix(qcat, blocks)
                elif _scheme_base(score_scheme).endswith("_project"):
                    propagation_overlap, max_propagation_mass = _scope_projection_matrix(blocks)
                else:
                    propagation_overlap, max_propagation_mass = _scope_overlap_matrix(blocks)
            all_scores = _score_blocks_value_of_information(
                score_scheme,
                blocks,
                scoring_answers,
                syn_answers,
                round_measurement_sigma,
                propagation_overlap,
                max_propagation_mass,
            )
            scores = np.asarray([all_scores[block_id] for block_id in available_ids], dtype=np.float64)
        else:
            scores = np.asarray(
                [
                    _score_block(
                        score_scheme,
                        blocks[block_id],
                        scoring_answers,
                        syn_answers,
                        n_real,
                        round_measurement_sigma,
                    )
                    for block_id in available_ids
                ],
                dtype=np.float64,
            )
        private_scores = scores.astype(np.float64, copy=True)
        sage_prior_scores = np.full(len(available_ids), np.nan, dtype=np.float64)
        sage_prior_log_weights = np.zeros(len(available_ids), dtype=np.float64)
        sage_prior_trusted_queries = 0
        if (
            float(transcript_sage_prior_odds) > 1.0
            and forced_sequence is None
            and score_input_source != "public_bootstrap"
        ):
            prior_answers, prior_trusted_mask = _transcript_selector_answers(
                previous_projected_answers=previous_projected_answers,
                previous_variances=previous_variances,
                syn_answers=syn_answers,
                untrusted_variance=transcript_untrusted_variance,
                reliability_mode="hard",
            )
            if prior_answers is None:
                raise RuntimeError("The transcript SAGE prior has no released transcript")
            if repair_plans is None:
                repair_plans = _query_repair_plans(qcat, blocks)
            prior_all_scores = _score_blocks_operator_value_of_information(
                "voi_sageordergain_harmonic_qproject",
                blocks,
                prior_answers,
                syn_answers,
                round_measurement_sigma,
                repair_plans,
                active_query_budget=int(base_config.get("qdte", {}).get("num_active_targets", 64)),
                kappa_noise=float(base_config.get("qdte", {}).get("kappa_noise", 1.0)),
                selected_block_ids=selected_block_ids,
                conditional_context=None,
                state_weight_answers=prior_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=n_real,
            )
            sage_prior_scores = np.asarray(
                [prior_all_scores[block_id] for block_id in available_ids],
                dtype=np.float64,
            )
            scores, sage_prior_log_weights = _apply_released_score_base_measure(
                private_scores,
                sage_prior_scores,
                epsilon=round_selection_epsilon,
                sensitivity=selection_score_sensitivity,
                max_odds=float(transcript_sage_prior_odds),
            )
            sage_prior_trusted_queries = int(np.sum(prior_trusted_mask))
            score_scheme = (
                f"{score_scheme}|released_sage_rank_prior_odds="
                f"{float(transcript_sage_prior_odds):g}"
            )

        amortized_alpha = _scheme_mdl_family_amortized_alpha(scheme, family_counts)
        if forced_sequence is None and amortized_alpha is not None:
            prior = np.asarray(
                [
                    -math.log(max(1, family_counts[blocks[block_id].family]))
                    + amortized_alpha * math.log(max(1, int(blocks[block_id].query_indices.size)))
                    for block_id in available_ids
                ],
                dtype=np.float64,
            )
            scores = scores + (2.0 / max(float(round_selection_epsilon), 1.0e-12)) * prior
        elif forced_sequence is None and _scheme_has_modifier(scheme, "mdl_family"):
            prior = np.asarray(
                [-math.log(max(1, family_counts[blocks[block_id].family])) for block_id in available_ids],
                dtype=np.float64,
            )
            scores = scores + (2.0 / max(float(round_selection_epsilon), 1.0e-12)) * prior
        if (
            selection_input == "transcript"
            and score_input_source == "transcript"
            and nonpositive_score_fallback == "public_bootstrap"
            and float(np.max(scores)) <= 0.0
        ):
            scores = _public_bootstrap_scores(blocks, available_ids, bootstrap_strategy)
            score_input_source = "transcript_public_fallback"
            score_scheme = f"{score_scheme}|fallback:{bootstrap_strategy}"
        if (
            forced_sequence is None or round_id > len(forced_sequence)
        ) and score_input_source != "public_bootstrap":
            chosen_local = _select_from_scores(
                scores,
                rule=selection_rule,
                epsilon=round_selection_epsilon,
                rng=rng,
                sensitivity=selection_score_sensitivity,
            )
            chosen_block_id = int(available_ids[chosen_local])
        chosen = blocks[chosen_block_id]
        selected_diagnostics = _selected_block_diagnostics(
            chosen,
            true_answers,
            syn_answers,
            round_measurement_sigma,
        )
        score_component_diagnostics: dict[str, Any] = {}
        if _is_operator_value_of_information_scheme(score_scheme):
            if repair_plans is None:
                repair_plans = _query_repair_plans(qcat, blocks)
            score_component_diagnostics = _dpblend_score_diagnostics(
                score_scheme=score_scheme,
                blocks=blocks,
                true_answers=scoring_answers,
                syn_answers=syn_answers,
                measurement_sigma=round_measurement_sigma,
                repair_plans=repair_plans,
                active_query_budget=int(base_config.get("qdte", {}).get("num_active_targets", 64)),
                kappa_noise=float(base_config.get("qdte", {}).get("kappa_noise", 1.0)),
                selected_block_ids=selected_block_ids,
                state_weight_answers=scoring_answers if selection_input == "transcript" else previous_projected_answers,
                round_id=round_id,
                rounds=rounds,
                row_count=n_real,
                available_ids=available_ids,
                chosen_block_id=chosen_block_id,
            )
        selected_set.add(chosen_block_id)
        selected_block_ids.append(chosen_block_id)
        noise_std = float(chosen.delta_l2) * round_measurement_sigma
        measurement_records.append(
            MeasurementRecord(
                block_id=chosen_block_id,
                noisy=true_answers[chosen.query_indices].astype(np.float64)
                + rng.normal(0.0, noise_std, size=len(chosen.query_indices)),
                measurement_sigma=round_measurement_sigma,
                selection_epsilon=round_selection_epsilon,
                selection_rho=round_selection_rho,
            )
        )

        measurement_dir = _measurement_artifact(
            qcat=qcat,
            schema=schema,
            blocks=blocks,
            measurement_records=measurement_records,
            current_syn_answers=syn_answers,
            total_rows=n_real,
            cardinalities=cardinalities,
            projection_cfg=dict(base_config.get("projection", {})),
            output_dir=scheme_dir / f"round_{round_id:03d}_measurement",
            epsilon=selection_epsilon,
            measurement_sigma=float(budget.measurement_sigma),
            delta=delta,
            selection_rho_per_round=selection_rho_per_round,
            base_measurements=base_measurements,
        )
        latest_measurement_dir = measurement_dir
        measurement_payload = read_json(measurement_dir / "measurements.json")
        previous_projected_answers = np.asarray(measurement_payload["target_projected"], dtype=np.float64)
        previous_variances = np.asarray(measurement_payload["variances"], dtype=np.float64)
        round_dir = scheme_dir / f"round_{round_id:03d}_generate"
        generator_config = _configure_a_generator(
            base_config,
            output_dir=round_dir,
            init_path=current_syn_path,
            measurement_dir=measurement_dir,
            inner_iters=inner_iters,
            generator_profile=generator_profile,
            generator_seed=(
                _generator_stage_seed(
                    int(base_config.get("run", {}).get("seed", 0)),
                    round_id + stage_seed_offset,
                )
                if generator_seed_mode == "per_stage"
                else None
            ),
            stop_patience=inner_stop_patience,
        )
        run_qdte(generator_config)
        current_syn_path = round_dir / "synthetic_encoded.npy"
        current_syn = np.load(current_syn_path).astype(np.int32)
        final_answers = answer_queries(
            current_syn,
            qcat,
            batch_size=int(base_config.get("runtime", {}).get("answer_batch_size", 8192)),
        )
        model_change_l1, expected_noise_l1, anneal_signal = (
            _released_model_change_anneal_diagnostic(
                before_answers=syn_answers,
                after_answers=final_answers,
                query_indices=chosen.query_indices,
                noise_std=noise_std,
            )
        )
        annealed_after_round = bool(
            measurement_schedule == "aim_released_change"
            and not round_exhausts_budget
            and anneal_signal
        )
        if annealed_after_round:
            measurement_sigma = round_measurement_sigma / 2.0
            if selection_ledger != "measurement_only":
                selection_epsilon = round_selection_epsilon * 2.0
        elif measurement_schedule == "aim_released_change":
            measurement_sigma = round_measurement_sigma
            if selection_ledger != "measurement_only":
                selection_epsilon = round_selection_epsilon
        metrics = _query_error_metrics_with_tvd(
            blocks,
            true_answers,
            final_answers,
            n_real,
            current_syn.shape[0],
            prefix="full_true",
        )
        qv_curriculum_alpha = (
            _query_vector_curriculum_alpha(round_id, rounds)
            if _scheme_base(scheme)
            in {
                "voi_sageqvcapcurrmrr_harmonic_qproject",
                "voi_sageqvcapresmrr_harmonic_qproject",
                "voi_sageqvcapbandmrr_harmonic_qproject",
                "voi_sageqvcalcurrmrr_harmonic_qproject",
                "voi_sageqvcalbandmrr_harmonic_qproject",
            }
            else np.nan
        )
        rows.append(
            {
                "round": round_id,
                "scheme": scheme,
                "score_scheme": score_scheme,
                "selection_input": selection_input,
                "score_input_source": score_input_source,
                "selection_rule": selection_rule,
                "selection_ledger": selection_ledger,
                "selection_epsilon": round_selection_epsilon,
                "selection_rho": round_selection_rho,
                "adaptive_selection_rho_before": adaptive_selection_rho_before,
                "adaptive_selection_rho_after": adaptive_selection_rho_before
                + round_selection_rho,
                "selection_trusted_queries": int(np.sum(trusted_mask)),
                "selection_untrusted_queries": int(int(qcat.m) - int(np.sum(trusted_mask))),
                "qv_curriculum_alpha": float(qv_curriculum_alpha),
                "selected_block_id": chosen_block_id,
                "selected_name": chosen.name,
                "selected_family": chosen.family,
                "selected_queries": int(len(chosen.query_indices)),
                "selected_delta_l2": float(chosen.delta_l2),
                "measurement_sigma": round_measurement_sigma,
                "measurement_rho": round_measurement_rho,
                "adaptive_measurement_rho_before": adaptive_measurement_rho_before,
                "adaptive_measurement_rho_after": adaptive_measurement_rho_before
                + round_measurement_rho,
                "adaptive_rho_before": adaptive_rho_before,
                "adaptive_rho_after": adaptive_rho_before
                + round_measurement_rho
                + round_selection_rho,
                "adaptive_budget_exhausted": int(round_exhausts_budget),
                "measurement_budget_exhausted": int(round_exhausts_budget),
                "released_model_change_l1": model_change_l1,
                "expected_measurement_noise_l1": expected_noise_l1,
                "anneal_signal": int(anneal_signal),
                "annealed_after_round": int(annealed_after_round),
                "selected_scope": "|".join(str(attr) for attr in chosen.scope),
                "selected_coverage_weight": float(chosen.coverage_weight),
                "selected_score": float(scores[chosen_local]),
                "score_max": float(np.max(scores)),
                "score_mean": float(np.mean(scores)),
                "private_selected_score": float(private_scores[chosen_local]),
                "private_score_max": float(np.max(private_scores)),
                "private_score_mean": float(np.mean(private_scores)),
                "transcript_sage_prior_odds": float(transcript_sage_prior_odds),
                "transcript_sage_prior_trusted_queries": sage_prior_trusted_queries,
                "transcript_sage_prior_selected_score": float(sage_prior_scores[chosen_local]),
                "transcript_sage_prior_score_max": float(np.nanmax(sage_prior_scores))
                if np.any(np.isfinite(sage_prior_scores))
                else np.nan,
                "transcript_sage_prior_selected_log_weight": float(
                    sage_prior_log_weights[chosen_local]
                ),
                "pre_full_true_mae": float(pre_metrics["full_true_mae"]),
                "pre_full_true_rmse": float(pre_metrics["full_true_rmse"]),
                "pre_full_true_max_error": float(pre_metrics["full_true_max_error"]),
                "pre_full_true_avg_tvd": float(pre_metrics["full_true_avg_tvd"]),
                "pre_full_true_max_tvd": float(pre_metrics["full_true_max_tvd"]),
                "full_true_mae": float(metrics["full_true_mae"]),
                "full_true_rmse": float(metrics["full_true_rmse"]),
                "full_true_max_error": float(metrics["full_true_max_error"]),
                "full_true_avg_tvd": float(metrics["full_true_avg_tvd"]),
                "full_true_max_tvd": float(metrics["full_true_max_tvd"]),
                "delta_full_true_mae": float(pre_metrics["full_true_mae"] - metrics["full_true_mae"]),
                "delta_full_true_rmse": float(pre_metrics["full_true_rmse"] - metrics["full_true_rmse"]),
                "delta_full_true_max_error": float(
                    pre_metrics["full_true_max_error"] - metrics["full_true_max_error"]
                ),
                "delta_full_true_avg_tvd": float(pre_metrics["full_true_avg_tvd"] - metrics["full_true_avg_tvd"]),
                "delta_full_true_max_tvd": float(pre_metrics["full_true_max_tvd"] - metrics["full_true_max_tvd"]),
                **selected_diagnostics,
                **score_component_diagnostics,
            }
        )
        pd.DataFrame(rows).to_csv(scheme_dir / "adaptive_timeseries.csv", index=False)
        if round_exhausts_budget:
            break

    pre_refit_answers = answer_queries(
        current_syn,
        qcat,
        batch_size=int(base_config.get("runtime", {}).get("answer_batch_size", 8192)),
    )
    pre_refit_metrics = _query_error_metrics_with_tvd(
        blocks,
        true_answers,
        pre_refit_answers,
        n_real,
        current_syn.shape[0],
        prefix="full_true",
    )
    final_refit_metadata: dict[str, Any] = {
        "enabled": int(final_refit_iters) > 0,
        "requested_iters": int(final_refit_iters),
    }
    if int(final_refit_iters) > 0:
        if latest_measurement_dir is None:
            raise RuntimeError("A final refit requires at least one completed measurement round")
        final_refit_dir = scheme_dir / "final_refit"
        final_refit_config = _configure_a_generator(
            base_config,
            output_dir=final_refit_dir,
            init_path=current_syn_path,
            measurement_dir=latest_measurement_dir,
            inner_iters=int(final_refit_iters),
            generator_profile=generator_profile,
            generator_seed=(
                _generator_stage_seed(
                    int(base_config.get("run", {}).get("seed", 0)),
                    len(rows) + stage_seed_offset + 1,
                )
                if generator_seed_mode == "per_stage"
                else None
            ),
            stop_patience=final_stop_patience,
        )
        run_qdte(final_refit_config)
        current_syn_path = final_refit_dir / "synthetic_encoded.npy"
        current_syn = np.load(current_syn_path).astype(np.int32)
        final_refit_metadata.update(
            {
                "output_dir": str(final_refit_dir),
                "synthetic_encoded": str(current_syn_path),
                "measurement_dir": str(latest_measurement_dir),
            }
        )
    final_answers = answer_queries(
        current_syn,
        qcat,
        batch_size=int(base_config.get("runtime", {}).get("answer_batch_size", 8192)),
    )
    final_metrics = _query_error_metrics_with_tvd(
        blocks,
        true_answers,
        final_answers,
        n_real,
        current_syn.shape[0],
        prefix="full_true",
    )
    summary = {
        "scheme": scheme,
        "rounds_completed": len(rows),
        "rounds_requested": int(rounds),
        "initial_fit": warm_start_metadata,
        "inner_iters": int(inner_iters),
        "inner_stop_patience": int(inner_iters if inner_stop_patience is None else inner_stop_patience),
        "generator_seed_mode": generator_seed_mode,
        "generator_profile": generator_profile,
        "coverage_rho": coverage_rho,
        "measurement_schedule": measurement_schedule,
        "adaptive_rho_limit": adaptive_rho_limit,
        "adaptive_measurement_rho_limit": (
            adaptive_rho_limit if selection_ledger == "measurement_only" else None
        ),
        "selection_epsilon_schedule": [float(row["selection_epsilon"]) for row in rows],
        "selection_rho_schedule": [float(row["selection_rho"]) for row in rows],
        "measurement_sigma_schedule": [float(row["measurement_sigma"]) for row in rows],
        "measurement_rho_schedule": [float(row["measurement_rho"]) for row in rows],
        "final_measurement_dir": None
        if latest_measurement_dir is None
        else str(latest_measurement_dir),
        "final_refit": final_refit_metadata,
        "budget_mode": budget.mode,
        "epsilon": float(budget.epsilon),
        "measurement_sigma": float(budget.measurement_sigma),
        "rho_total": float(budget.rho_total),
        "budget_split_mu": None if budget.budget_split_mu is None else float(budget.budget_split_mu),
        "selection_input": selection_input,
        "selection_score_sensitivity": selection_score_sensitivity,
        "transcript_sage_prior": {
            "enabled": bool(float(transcript_sage_prior_odds) > 1.0),
            "source": "released_projected_transcript_hard_reliability",
            "score": "voi_sageordergain_harmonic_qproject",
            "transform": "midrank_log_base_measure",
            "max_odds": float(transcript_sage_prior_odds),
            "adds_current_round_private_sensitivity": False,
        },
        "selection_mechanism": "exponential"
        if selection_input == "oracle"
        else "postprocessing",
        "bootstrap_strategy": bootstrap_strategy,
        "public_bootstrap_rounds": int(public_bootstrap_rounds),
        "transcript_untrusted_variance": float(transcript_untrusted_variance),
        "transcript_reliability": transcript_reliability,
        "selection_ledger": selection_ledger,
        "selection_rule": selection_rule,
        "nonpositive_score_fallback": nonpositive_score_fallback,
        "num_blocks": len(blocks),
        "num_queries": int(qcat.m),
        "selected_blocks": [
            {
                "block_id": int(block_id),
                "name": blocks[block_id].name,
                "family": blocks[block_id].family,
                "num_queries": int(len(blocks[block_id].query_indices)),
                "delta_l2": float(blocks[block_id].delta_l2),
                "scope": list(blocks[block_id].scope),
                "coverage_weight": float(blocks[block_id].coverage_weight),
            }
            for block_id in selected_block_ids
        ],
        "pre_refit_metrics": {
            key: float(value) for key, value in pre_refit_metrics.items()
        },
        **{key: float(value) for key, value in final_metrics.items()},
    }
    write_json(summary, scheme_dir / "adaptive_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare adaptive selection scores with A/constructive_pair generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--schemes",
        default=(
            "aim_l1_floor,adaptive_order_tail_power_cover,adaptive_order_l1_cover,"
            "noise_adaptive_top_linear,noise_adaptive_top_linear_cover,"
            "propagated_entropic_excess_cover,l1_l2mix_noise_power_cover,tail_guard_noise_power_cover,"
            "l1_l2_tail_guard_noise_power_cover,old_support_l1,l2_net"
        ),
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--inner-iters", type=int, default=30)
    parser.add_argument(
        "--generator-profile",
        choices=["adaptive_legacy", "qdte_standard"],
        default="adaptive_legacy",
        help="Preserve the historical adaptive generator or use the base strong QDTE config.",
    )
    parser.add_argument(
        "--final-refit-iters",
        type=int,
        default=0,
        help="Optional full QDTE refit on the final cumulative measurement artifact.",
    )
    parser.add_argument("--budget-mode", choices=["per_round", "total_zcdp"], default="per_round")
    parser.add_argument("--rho-total", type=float, default=0.0)
    parser.add_argument("--budget-split-mu", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--measurement-sigma", type=float, default=1.0)
    parser.add_argument(
        "--selection-input",
        choices=["oracle", "transcript"],
        default="oracle",
        help="oracle keeps the selector-ablation path; transcript uses only previous noisy/projected transcript state for scoring.",
    )
    parser.add_argument(
        "--bootstrap-strategy",
        choices=["coverage", "low_order_coverage"],
        default="coverage",
        help="Public first-round strategy when --selection-input=transcript.",
    )
    parser.add_argument(
        "--public-bootstrap-rounds",
        type=int,
        default=1,
        help="Number of initial public bootstrap rounds before transcript scoring is enabled.",
    )
    parser.add_argument(
        "--transcript-untrusted-variance",
        type=float,
        default=1.0e11,
        help="Queries with transcript variance above this threshold get zero residual in transcript selection.",
    )
    parser.add_argument(
        "--transcript-reliability",
        choices=["hard", "all"],
        default="hard",
        help="hard gates high-variance queries; all uses the full projected transcript for scoring.",
    )
    parser.add_argument(
        "--selection-ledger",
        choices=["conservative", "measurement_only"],
        default="conservative",
        help="Whether measurement artifacts account a notional EM selection cost or measurement cost only.",
    )
    parser.add_argument(
        "--selection-rule",
        choices=["sample", "argmax"],
        default="sample",
        help="How to choose from scores. Argmax is deterministic post-processing in transcript mode.",
    )
    parser.add_argument(
        "--nonpositive-score-fallback",
        choices=["none", "public_bootstrap"],
        default="none",
        help="Optional public fallback when transcript scores are all non-positive.",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature parameter used as epsilon when transcript selection uses measurement-only total budget.",
    )
    parser.add_argument(
        "--transcript-sage-prior-odds",
        type=float,
        default=1.0,
        help=(
            "Maximum conditional base-measure odds from released-transcript SAGE ranks; "
            "values above one require private aim_l1_floor selection."
        ),
    )
    args, overrides = parser.parse_known_args()
    if int(args.final_refit_iters) < 0:
        raise ValueError("--final-refit-iters must be non-negative")

    config = apply_overrides(load_yaml(args.config), overrides)
    run_cfg = config.get("run", {})
    output_dir = ensure_dir(run_cfg.get("output_dir", "outputs/adaptive_selection_ablation"))
    save_yaml(config, output_dir / "config_base_resolved.yaml")
    budget = _resolve_budget(args, config)

    rng = np.random.default_rng(int(run_cfg.get("seed", 0)))
    preprocess_result = load_and_preprocess_csv(config)
    qcat, groups = build_workload(preprocess_result.schema, config)
    qcat.save_json(output_dir / "queries_full.json")
    preprocess_result.schema.save_json(output_dir / "schema.json")
    blocks = _build_blocks(groups, qcat)
    true_answers = answer_queries(
        preprocess_result.X,
        qcat,
        batch_size=int(config.get("runtime", {}).get("answer_batch_size", 8192)),
    )
    n_syn = preprocess_result.X.shape[0]
    initial_syn = _make_initial_synthetic(n_syn, preprocess_result.schema.cardinalities, rng)
    np.save(output_dir / "initial_synthetic_encoded.npy", initial_syn)

    write_json(
        {
            "num_queries": int(qcat.m),
            "num_blocks": int(len(blocks)),
            "blocks_by_family": {
                family: int(sum(1 for block in blocks if block.family == family))
                for family in sorted({block.family for block in blocks})
            },
            "budget_mode": budget.mode,
            "epsilon": float(budget.epsilon),
            "measurement_sigma": float(budget.measurement_sigma),
            "rho_total": float(budget.rho_total),
            "budget_split_mu": None if budget.budget_split_mu is None else float(budget.budget_split_mu),
            "selection_input": args.selection_input,
            "bootstrap_strategy": args.bootstrap_strategy,
            "public_bootstrap_rounds": int(args.public_bootstrap_rounds),
            "transcript_untrusted_variance": float(args.transcript_untrusted_variance),
            "transcript_reliability": args.transcript_reliability,
            "selection_ledger": args.selection_ledger,
            "selection_rule": args.selection_rule,
            "nonpositive_score_fallback": args.nonpositive_score_fallback,
            "rounds": int(args.rounds),
            "inner_iters": int(args.inner_iters),
            "generator_profile": args.generator_profile,
            "final_refit_iters": int(args.final_refit_iters),
            "transcript_sage_prior_odds": float(args.transcript_sage_prior_odds),
        },
        output_dir / "adaptive_setup.json",
    )

    summaries = []
    for scheme in [part.strip() for part in args.schemes.split(",") if part.strip()]:
        scheme_rng = np.random.default_rng(int(run_cfg.get("seed", 0)) + 10_000)
        summaries.append(
            _run_scheme(
                base_config=config,
                qcat=qcat,
                schema=preprocess_result.schema,
                blocks=blocks,
                true_answers=true_answers,
                initial_syn=initial_syn,
                cardinalities=preprocess_result.schema.cardinalities,
                output_dir=output_dir,
                scheme=scheme,
                rounds=int(args.rounds),
                inner_iters=int(args.inner_iters),
                budget=budget,
                rng=scheme_rng,
                generator_profile=args.generator_profile,
                final_refit_iters=int(args.final_refit_iters),
                selection_input=args.selection_input,
                bootstrap_strategy=args.bootstrap_strategy,
                transcript_untrusted_variance=float(args.transcript_untrusted_variance),
                transcript_reliability=args.transcript_reliability,
                selection_ledger=args.selection_ledger,
                selection_rule=args.selection_rule,
                public_bootstrap_rounds=int(args.public_bootstrap_rounds),
                nonpositive_score_fallback=args.nonpositive_score_fallback,
                transcript_sage_prior_odds=float(args.transcript_sage_prior_odds),
            )
        )
    write_json({"summaries": summaries}, output_dir / "adaptive_comparison.json")


if __name__ == "__main__":
    main()
