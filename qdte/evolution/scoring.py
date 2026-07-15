from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.precision import (
    ActiveSetOrthogonalInteractionPrecision,
    BootstrapDiagonalOrthogonalInteractionPrecision,
    OrthogonalInteractionPrecision,
    PrecisionOperator,
    ShrinkageAnalyticOrthogonalInteractionPrecision,
)
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.eval_jax import eval_records_queries_arrays
from qdte.queries.types import QueryCatalogue


@dataclass(frozen=True)
class ScoreCandidateContext:
    qarrays: tuple[jax.Array, ...]


@dataclass(frozen=True)
class OrthogonalPrecisionScoreContext:
    scorer: Callable[
        [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ]
    coefficient_dimension: int
    row_width: int


@jax.jit
def _score_candidates_jit(
    old_rows: jax.Array,
    new_rows: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    edit_cost: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
    lambda_cost: jax.Array,
) -> jax.Array:
    phi_old = eval_records_queries_arrays(
        old_rows,
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    )
    phi_new = eval_records_queries_arrays(
        new_rows,
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    )
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    enter = phi_new & (~phi_old)
    exit_ = phi_old & (~phi_new)
    linear = enter.astype(jnp.float32) @ weights - exit_.astype(jnp.float32) @ weights
    quad = (enter | exit_).astype(jnp.float32) @ inv_variance.astype(jnp.float32)
    return linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)


@partial(jax.pmap, in_axes=(0, 0, None, None, 0, None, None))
def _score_candidates_pmap(
    old_rows: jax.Array,
    new_rows: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    edit_cost: jax.Array,
    qarrays: tuple[jax.Array, ...],
    lambda_cost: jax.Array,
) -> jax.Array:
    phi_old = eval_records_queries_arrays(old_rows, *qarrays)
    phi_new = eval_records_queries_arrays(new_rows, *qarrays)
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    enter = phi_new & (~phi_old)
    exit_ = phi_old & (~phi_new)
    linear = enter.astype(jnp.float32) @ weights - exit_.astype(jnp.float32) @ weights
    quad = (enter | exit_).astype(jnp.float32) @ inv_variance.astype(jnp.float32)
    return linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)


def _qarrays(qcat: QueryCatalogue) -> tuple[jax.Array, ...]:
    return (
        jnp.asarray(qcat.attrs, dtype=jnp.int32),
        jnp.asarray(qcat.ops, dtype=jnp.int32),
        jnp.asarray(qcat.values, dtype=jnp.int32),
        jnp.asarray(qcat.lows, dtype=jnp.int32),
        jnp.asarray(qcat.highs, dtype=jnp.int32),
        jnp.asarray(qcat.linear_attrs, dtype=jnp.int32),
        jnp.asarray(qcat.linear_weights, dtype=jnp.float32),
        jnp.asarray(qcat.linear_thresholds, dtype=jnp.float32),
        jnp.asarray(qcat.linear_num_terms, dtype=jnp.int32),
    )


def prepare_score_context(qcat: QueryCatalogue) -> ScoreCandidateContext:
    return ScoreCandidateContext(qarrays=tuple(jax.device_put(arr) for arr in _qarrays(qcat)))


def prepare_orthogonal_precision_score_context(
    precision: (
        OrthogonalInteractionPrecision
        | ShrinkageAnalyticOrthogonalInteractionPrecision
    ),
) -> OrthogonalPrecisionScoreContext:
    block_specs: list[
        tuple[
            int,
            int,
            int,
            int,
            float,
            float,
            jax.Array,
            jax.Array,
            jax.Array | None,
        ]
    ] = []
    offset = 0
    for block in precision.blocks:
        end = offset + block.rank
        right_attr = block.scope[1] if len(block.scope) == 2 else -1
        isotropic_precision, directional_precision, direction = (
            precision.block_precision_parameters(block.name)
        )
        if direction.shape != (block.rank,):
            raise ValueError(f"Precision direction has the wrong shape for block {block.name!r}")
        block_specs.append(
            (
                block.scope[0],
                right_attr,
                offset,
                end,
                float(isotropic_precision),
                float(directional_precision),
                jax.device_put(jnp.asarray(direction, dtype=jnp.float32)),
                jax.device_put(jnp.asarray(block.left_contrast, dtype=jnp.float32)),
                (
                    jax.device_put(jnp.asarray(block.right_contrast, dtype=jnp.float32))
                    if block.right_contrast is not None
                    else None
                ),
            )
        )
        offset = end

    @jax.jit
    def scorer(
        old_rows: jax.Array,
        new_rows: jax.Array,
        weighted_coefficient_residual: jax.Array,
        edit_cost: jax.Array,
        lambda_cost: jax.Array,
    ) -> jax.Array:
        linear = jnp.zeros(old_rows.shape[0], dtype=jnp.float32)
        quad = jnp.zeros(old_rows.shape[0], dtype=jnp.float32)
        for (
            left_attr,
            right_attr,
            start,
            end,
            isotropic_precision,
            directional_precision,
            direction,
            left,
            right,
        ) in block_specs:
            left_old = left[old_rows[:, left_attr]]
            left_new = left[new_rows[:, left_attr]]
            if right is None:
                delta = left_new - left_old
            else:
                right_old = right[old_rows[:, right_attr]]
                right_new = right[new_rows[:, right_attr]]
                old_feature = jnp.einsum("ni,nj->nij", left_old, right_old).reshape(
                    old_rows.shape[0],
                    -1,
                )
                new_feature = jnp.einsum("ni,nj->nij", left_new, right_new).reshape(
                    old_rows.shape[0],
                    -1,
                )
                delta = new_feature - old_feature
            linear = linear + delta @ weighted_coefficient_residual[start:end]
            block_quad = jnp.sum(delta * delta, axis=1) * jnp.float32(
                isotropic_precision
            )
            block_quad = block_quad + jnp.float32(directional_precision) * (
                delta @ direction
            ) ** 2
            quad = quad + block_quad
        quad = jnp.maximum(quad, jnp.float32(0.0))
        score = linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)
        return score, quad

    return OrthogonalPrecisionScoreContext(
        scorer=scorer,
        coefficient_dimension=precision.coefficient_dimension,
        row_width=len(precision.cardinalities),
    )


def score_candidates_orthogonal_precision(
    candidates: CandidateBatch,
    residual: np.ndarray,
    precision: (
        OrthogonalInteractionPrecision
        | ShrinkageAnalyticOrthogonalInteractionPrecision
    ),
    lambda_cost: float,
    *,
    chunk_size: int = 4096,
    context: OrthogonalPrecisionScoreContext | None = None,
) -> np.ndarray:
    scores, _ = score_candidates_orthogonal_precision_with_quadratic(
        candidates,
        residual,
        precision,
        lambda_cost,
        chunk_size=chunk_size,
        context=context,
    )
    return scores


def score_candidates_orthogonal_precision_with_quadratic(
    candidates: CandidateBatch,
    residual: np.ndarray,
    precision: (
        OrthogonalInteractionPrecision
        | ShrinkageAnalyticOrthogonalInteractionPrecision
    ),
    lambda_cost: float,
    *,
    chunk_size: int = 4096,
    context: OrthogonalPrecisionScoreContext | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if candidates.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy()
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    active_context = context or prepare_orthogonal_precision_score_context(precision)
    if active_context.coefficient_dimension != precision.coefficient_dimension:
        raise ValueError("Orthogonal precision score context has a mismatched coefficient dimension")
    if candidates.old_rows.shape[1] != active_context.row_width:
        raise ValueError("Candidate row width does not match the orthogonal precision context")
    weighted_residual = jnp.asarray(
        precision.weighted_coefficient_residual(residual),
        dtype=jnp.float32,
    )
    lambda_j = jnp.asarray(lambda_cost, dtype=jnp.float32)
    score_output: list[np.ndarray] = []
    quadratic_output: list[np.ndarray] = []
    for start in range(0, candidates.size, int(chunk_size)):
        end = min(start + int(chunk_size), candidates.size)
        scores, quadratic = active_context.scorer(
            jnp.asarray(candidates.old_rows[start:end], dtype=jnp.int32),
            jnp.asarray(candidates.new_rows[start:end], dtype=jnp.int32),
            weighted_residual,
            jnp.asarray(candidates.edit_cost[start:end], dtype=jnp.float32),
            lambda_j,
        )
        score_output.append(np.asarray(scores, dtype=np.float32))
        quadratic_output.append(np.asarray(quadratic, dtype=np.float32))
    return (
        np.concatenate(score_output, axis=0),
        np.concatenate(quadratic_output, axis=0),
    )


def score_candidates_active_set_orthogonal_precision_with_quadratic(
    candidates: CandidateBatch,
    residual: np.ndarray,
    precision: (
        ActiveSetOrthogonalInteractionPrecision
        | BootstrapDiagonalOrthogonalInteractionPrecision
    ),
    lambda_cost: float,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if candidates.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy()
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    score_output: list[np.ndarray] = []
    quadratic_output: list[np.ndarray] = []
    for start in range(0, candidates.size, int(chunk_size)):
        end = min(start + int(chunk_size), candidates.size)
        feature_deltas = precision.row_feature_deltas(
            candidates.old_rows[start:end],
            candidates.new_rows[start:end],
        )
        scores, quadratic = precision.feature_advantages_with_quadratic(
            residual,
            feature_deltas,
            candidates.edit_cost[start:end],
            lambda_cost,
        )
        score_output.append(scores.astype(np.float32))
        quadratic_output.append(quadratic.astype(np.float32))
    return np.concatenate(score_output), np.concatenate(quadratic_output)


def score_candidates_bootstrap_orthogonal_precision_with_quadratic(
    candidates: CandidateBatch,
    residual: np.ndarray,
    precision: BootstrapDiagonalOrthogonalInteractionPrecision,
    lambda_cost: float,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    return score_candidates_active_set_orthogonal_precision_with_quadratic(
        candidates,
        residual,
        precision,
        lambda_cost,
        chunk_size=chunk_size,
    )


def score_candidates(
    candidates: CandidateBatch,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    qcat: QueryCatalogue,
    lambda_cost: float,
    chunk_size: int = 4096,
    use_pmap: bool = True,
    context: ScoreCandidateContext | None = None,
) -> np.ndarray:
    if candidates.size == 0:
        return np.empty(0, dtype=np.float32)
    qarrays = context.qarrays if context is not None else _qarrays(qcat)
    residual_j = jnp.asarray(residual, dtype=jnp.float32)
    inv_j = jnp.asarray(inv_variance, dtype=jnp.float32)
    lambda_j = jnp.asarray(lambda_cost, dtype=jnp.float32)
    out: list[np.ndarray] = []
    ndev = jax.local_device_count()
    can_pmap = bool(use_pmap and ndev > 1)
    for start in range(0, candidates.size, chunk_size):
        end = min(start + chunk_size, candidates.size)
        old = candidates.old_rows[start:end]
        new = candidates.new_rows[start:end]
        cost = candidates.edit_cost[start:end]
        if can_pmap and len(old) >= ndev:
            pad = (-len(old)) % ndev
            if pad:
                old_p = np.concatenate([old, np.repeat(old[-1:], pad, axis=0)], axis=0)
                new_p = np.concatenate([new, np.repeat(new[-1:], pad, axis=0)], axis=0)
                cost_p = np.concatenate([cost, np.repeat(cost[-1:], pad, axis=0)], axis=0)
            else:
                old_p, new_p, cost_p = old, new, cost
            per_dev = old_p.shape[0] // ndev
            scores = _score_candidates_pmap(
                jnp.asarray(old_p.reshape(ndev, per_dev, old_p.shape[1]), dtype=jnp.int32),
                jnp.asarray(new_p.reshape(ndev, per_dev, new_p.shape[1]), dtype=jnp.int32),
                residual_j,
                inv_j,
                jnp.asarray(cost_p.reshape(ndev, per_dev), dtype=jnp.float32),
                qarrays,
                lambda_j,
            )
            arr = np.asarray(scores).reshape(-1)[: len(old)]
        else:
            scores = _score_candidates_jit(
                jnp.asarray(old, dtype=jnp.int32),
                jnp.asarray(new, dtype=jnp.int32),
                residual_j,
                inv_j,
                jnp.asarray(cost, dtype=jnp.float32),
                *qarrays,
                lambda_j,
            )
            arr = np.asarray(scores)
        out.append(arr.astype(np.float32))
    return np.concatenate(out, axis=0)


def compute_deltas(old_rows: np.ndarray, new_rows: np.ndarray, qcat: QueryCatalogue) -> np.ndarray:
    if len(old_rows) == 0:
        return np.empty((0, qcat.m), dtype=np.int8)
    qarrays = _qarrays(qcat)
    phi_old = eval_records_queries_arrays(jnp.asarray(old_rows, dtype=jnp.int32), *qarrays)
    phi_new = eval_records_queries_arrays(jnp.asarray(new_rows, dtype=jnp.int32), *qarrays)
    delta = phi_new.astype(jnp.int8) - phi_old.astype(jnp.int8)
    return np.asarray(delta, dtype=np.int8)


def compute_deltas_sparse(
    old_rows: np.ndarray,
    new_rows: np.ndarray,
    delta_index: QueryDeltaIndex,
) -> np.ndarray:
    return delta_index.dense_candidate_deltas(old_rows, new_rows)


def score_candidates_precision(
    candidates: CandidateBatch,
    residual: np.ndarray,
    precision: PrecisionOperator,
    qcat: QueryCatalogue,
    lambda_cost: float,
    *,
    chunk_size: int = 512,
    delta_index: QueryDeltaIndex | None = None,
) -> np.ndarray:
    """Reference scorer for a general PSD precision operator.

    Candidate deltas are still evaluated through the frozen query catalogue;
    only the quadratic geometry changes. This path favors auditability over
    throughput and is intentionally opt-in.
    """
    if candidates.size == 0:
        return np.empty(0, dtype=np.float32)
    if precision.dimension != qcat.m:
        raise ValueError("Precision dimension must match the query catalogue")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    output = np.empty(candidates.size, dtype=np.float32)
    for start in range(0, candidates.size, int(chunk_size)):
        end = min(start + int(chunk_size), candidates.size)
        if delta_index is None:
            deltas = compute_deltas(candidates.old_rows[start:end], candidates.new_rows[start:end], qcat)
        else:
            deltas = compute_deltas_sparse(
                candidates.old_rows[start:end],
                candidates.new_rows[start:end],
                delta_index,
            )
        output[start:end] = precision.advantages(
            residual,
            deltas,
            candidates.edit_cost[start:end],
            lambda_cost,
        ).astype(np.float32)
    return output


def edit_advantage_from_delta(
    residual: np.ndarray,
    inv_variance: np.ndarray,
    delta: np.ndarray,
    edit_cost: np.ndarray,
    lambda_cost: float,
) -> np.ndarray:
    d = delta.astype(np.float32)
    linear = d @ (residual.astype(np.float32) * inv_variance.astype(np.float32))
    quad = (d * d) @ inv_variance.astype(np.float32)
    return linear - 0.5 * quad - float(lambda_cost) * edit_cost.astype(np.float32)


def l1_advantage_from_delta(
    residual: np.ndarray,
    weights: np.ndarray,
    delta: np.ndarray,
    edit_cost: np.ndarray,
    lambda_cost: float,
) -> np.ndarray:
    d = delta.astype(np.float32)
    r = residual.astype(np.float32)
    w = weights.astype(np.float32)
    before = np.abs(r).reshape(1, -1)
    after = np.abs(r.reshape(1, -1) - d)
    component = (before - after) @ w
    return component.astype(np.float32) - float(lambda_cost) * edit_cost.astype(np.float32)


@jax.jit
def _score_candidates_l1_jit(
    old_rows: jax.Array,
    new_rows: jax.Array,
    residual: jax.Array,
    weights: jax.Array,
    edit_cost: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
    lambda_cost: jax.Array,
) -> jax.Array:
    phi_old = eval_records_queries_arrays(
        old_rows,
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    )
    phi_new = eval_records_queries_arrays(
        new_rows,
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    )
    delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    component = (jnp.abs(residual).reshape(1, -1) - jnp.abs(residual.reshape(1, -1) - delta)) @ weights
    return component - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)


@partial(jax.pmap, in_axes=(0, 0, None, None, 0, None, None))
def _score_candidates_l1_pmap(
    old_rows: jax.Array,
    new_rows: jax.Array,
    residual: jax.Array,
    weights: jax.Array,
    edit_cost: jax.Array,
    qarrays: tuple[jax.Array, ...],
    lambda_cost: jax.Array,
) -> jax.Array:
    phi_old = eval_records_queries_arrays(old_rows, *qarrays)
    phi_new = eval_records_queries_arrays(new_rows, *qarrays)
    delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    component = (jnp.abs(residual).reshape(1, -1) - jnp.abs(residual.reshape(1, -1) - delta)) @ weights
    return component - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)


def score_candidates_l1(
    candidates: CandidateBatch,
    residual: np.ndarray,
    weights: np.ndarray,
    qcat: QueryCatalogue,
    lambda_cost: float,
    chunk_size: int = 4096,
    use_pmap: bool = True,
    context: ScoreCandidateContext | None = None,
) -> np.ndarray:
    if candidates.size == 0:
        return np.empty(0, dtype=np.float32)
    qarrays = context.qarrays if context is not None else _qarrays(qcat)
    residual_j = jnp.asarray(residual, dtype=jnp.float32)
    weights_j = jnp.asarray(weights, dtype=jnp.float32)
    lambda_j = jnp.asarray(lambda_cost, dtype=jnp.float32)
    out: list[np.ndarray] = []
    ndev = jax.local_device_count()
    can_pmap = bool(use_pmap and ndev > 1)
    for start in range(0, candidates.size, chunk_size):
        end = min(start + chunk_size, candidates.size)
        old = candidates.old_rows[start:end]
        new = candidates.new_rows[start:end]
        cost = candidates.edit_cost[start:end]
        if can_pmap and len(old) >= ndev:
            pad = (-len(old)) % ndev
            if pad:
                old_p = np.concatenate([old, np.repeat(old[-1:], pad, axis=0)], axis=0)
                new_p = np.concatenate([new, np.repeat(new[-1:], pad, axis=0)], axis=0)
                cost_p = np.concatenate([cost, np.repeat(cost[-1:], pad, axis=0)], axis=0)
            else:
                old_p, new_p, cost_p = old, new, cost
            per_dev = old_p.shape[0] // ndev
            scores = _score_candidates_l1_pmap(
                jnp.asarray(old_p.reshape(ndev, per_dev, old_p.shape[1]), dtype=jnp.int32),
                jnp.asarray(new_p.reshape(ndev, per_dev, new_p.shape[1]), dtype=jnp.int32),
                residual_j,
                weights_j,
                jnp.asarray(cost_p.reshape(ndev, per_dev), dtype=jnp.float32),
                qarrays,
                lambda_j,
            )
            arr = np.asarray(scores).reshape(-1)[: len(old)]
        else:
            scores = _score_candidates_l1_jit(
                jnp.asarray(old, dtype=jnp.int32),
                jnp.asarray(new, dtype=jnp.int32),
                residual_j,
                weights_j,
                jnp.asarray(cost, dtype=jnp.float32),
                *qarrays,
                lambda_j,
            )
            arr = np.asarray(scores)
        out.append(arr.astype(np.float32))
    return np.concatenate(out, axis=0)


def score_candidates_sparse(
    candidates: CandidateBatch,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    delta_index: QueryDeltaIndex,
    lambda_cost: float,
) -> np.ndarray:
    scores = np.empty(candidates.size, dtype=np.float32)
    weights = residual.astype(np.float32) * inv_variance.astype(np.float32)
    inv = inv_variance.astype(np.float32)
    for idx in range(candidates.size):
        delta = delta_index.candidate_delta(candidates.old_rows[idx], candidates.new_rows[idx])
        if len(delta.qids) == 0:
            advantage = 0.0
        else:
            signs = delta.values.astype(np.float32)
            qids = delta.qids
            linear = float(signs @ weights[qids])
            quad = float((signs * signs) @ inv[qids])
            advantage = linear - 0.5 * quad
        scores[idx] = np.float32(advantage - float(lambda_cost) * float(candidates.edit_cost[idx]))
    return scores


def score_candidates_target_only(
    candidates: CandidateBatch,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    qcat: QueryCatalogue,
    lambda_cost: float,
) -> np.ndarray:
    scores = np.full(candidates.size, -np.inf, dtype=np.float32)
    if candidates.size == 0:
        return scores
    for qid in sorted(set(int(x) for x in candidates.target_query_ids.tolist() if int(x) >= 0)):
        idx = np.flatnonzero(candidates.target_query_ids == qid)
        if len(idx) == 0:
            continue
        old_sat = qcat.eval_query_np(candidates.old_rows[idx], qid).astype(np.float32)
        new_sat = qcat.eval_query_np(candidates.new_rows[idx], qid).astype(np.float32)
        delta = new_sat - old_sat
        inv = float(inv_variance[qid])
        scores[idx] = (
            float(residual[qid]) * delta * inv
            - 0.5 * delta * delta * inv
            - float(lambda_cost) * candidates.edit_cost[idx]
        ).astype(np.float32)
    random_idx = np.flatnonzero(candidates.target_query_ids < 0)
    if len(random_idx) > 0:
        scores[random_idx] = 0.0 - float(lambda_cost) * candidates.edit_cost[random_idx]
    return scores
