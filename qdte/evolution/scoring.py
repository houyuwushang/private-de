from __future__ import annotations

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from qdte.evolution.candidates import CandidateBatch
from qdte.queries.eval_jax import eval_records_queries_arrays
from qdte.queries.types import QueryCatalogue


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
    lambda_cost: jax.Array,
) -> jax.Array:
    phi_old = eval_records_queries_arrays(old_rows, attrs, ops, values, lows, highs)
    phi_new = eval_records_queries_arrays(new_rows, attrs, ops, values, lows, highs)
    delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    linear = delta @ weights
    quad = (delta * delta) @ inv_variance.astype(jnp.float32)
    return linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)


@partial(jax.pmap, in_axes=(0, 0, None, None, 0, None, None))
def _score_candidates_pmap(
    old_rows: jax.Array,
    new_rows: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    edit_cost: jax.Array,
    qarrays: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    lambda_cost: jax.Array,
) -> jax.Array:
    attrs, ops, values, lows, highs = qarrays
    phi_old = eval_records_queries_arrays(old_rows, attrs, ops, values, lows, highs)
    phi_new = eval_records_queries_arrays(new_rows, attrs, ops, values, lows, highs)
    delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    linear = delta @ weights
    quad = (delta * delta) @ inv_variance.astype(jnp.float32)
    return linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)


def _qarrays(qcat: QueryCatalogue) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        jnp.asarray(qcat.attrs, dtype=jnp.int32),
        jnp.asarray(qcat.ops, dtype=jnp.int32),
        jnp.asarray(qcat.values, dtype=jnp.int32),
        jnp.asarray(qcat.lows, dtype=jnp.int32),
        jnp.asarray(qcat.highs, dtype=jnp.int32),
    )


def score_candidates(
    candidates: CandidateBatch,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    qcat: QueryCatalogue,
    lambda_cost: float,
    chunk_size: int = 4096,
    use_pmap: bool = True,
) -> np.ndarray:
    if candidates.size == 0:
        return np.empty(0, dtype=np.float32)
    qarrays = _qarrays(qcat)
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
