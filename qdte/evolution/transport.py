from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.queries.eval_jax import eval_records_queries_arrays
from qdte.queries.types import QueryCatalogue


@dataclass
class TransportResult:
    accepted_indices: np.ndarray
    delta_sum: np.ndarray
    batch_advantage: float
    mean_advantage: float


def select_top_nonconflicting(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    max_accept: int,
    min_advantage: float,
) -> np.ndarray:
    if max_accept <= 0:
        return np.empty(0, dtype=np.int32)
    valid = np.flatnonzero(np.isfinite(advantages) & (advantages > float(min_advantage)))
    if len(valid) == 0:
        return np.empty(0, dtype=np.int32)
    valid_scores = advantages[valid]
    pool_size = min(len(valid), max(4 * max_accept, max_accept))
    while True:
        if pool_size < len(valid):
            pool = np.argpartition(-valid_scores, pool_size - 1)[:pool_size]
        else:
            pool = np.arange(len(valid))
        order = pool[np.argsort(-valid_scores[pool])]
        seen_rows: set[int] = set()
        selected: list[int] = []
        for local_idx in order.tolist():
            idx = int(valid[local_idx])
            row_id = int(candidates.row_ids[idx])
            if row_id in seen_rows:
                continue
            seen_rows.add(row_id)
            selected.append(idx)
            if len(selected) >= max_accept:
                break
        if len(selected) >= max_accept or pool_size >= len(valid):
            return np.asarray(selected, dtype=np.int32)
        pool_size = min(len(valid), pool_size * 2)


def batch_advantage(
    residual: np.ndarray,
    inv_variance: np.ndarray,
    delta_sum: np.ndarray,
    cost_sum: float,
    lambda_cost: float,
) -> float:
    d = delta_sum.astype(np.float32)
    w = residual.astype(np.float32) * inv_variance.astype(np.float32)
    linear = float(d @ w)
    quad = float((d * d) @ inv_variance.astype(np.float32))
    return linear - 0.5 * quad - float(lambda_cost) * float(cost_sum)


def choose_transport_batch(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    deltas: np.ndarray,
    selected: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    prefix_strategy: str = "largest_positive",
) -> TransportResult:
    if len(selected) == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
        )
    order = np.argsort(-advantages[selected])
    selected = selected[order]
    deltas = deltas[order]
    full_delta_sum = deltas.astype(np.float32).sum(axis=0)
    full_cost_sum = float(candidates.edit_cost[selected].sum())
    full_adv = batch_advantage(residual, inv_variance, full_delta_sum, full_cost_sum, lambda_cost)
    if full_adv > 0.0 and prefix_strategy != "best_advantage":
        best_prefix = len(selected)
        best_adv = float(full_adv)
        delta_sum = full_delta_sum
    else:
        delta_prefix = np.cumsum(deltas.astype(np.float32), axis=0)
        cost_prefix = np.cumsum(candidates.edit_cost[selected].astype(np.float32))
        weights = residual.astype(np.float32) * inv_variance.astype(np.float32)
        batch_advantages = (
            delta_prefix @ weights
            - 0.5 * ((delta_prefix * delta_prefix) @ inv_variance.astype(np.float32))
            - float(lambda_cost) * cost_prefix
        )
        positive = np.flatnonzero(batch_advantages > 0.0)
        if prefix_strategy == "best_advantage":
            best_idx = int(np.argmax(batch_advantages))
            best_adv = float(batch_advantages[best_idx])
            best_prefix = best_idx + 1 if best_adv > 0.0 else 0
            delta_sum = delta_prefix[best_idx] if best_prefix > 0 else np.zeros_like(residual, dtype=np.float32)
        elif len(positive) > 0:
            best_prefix = int(positive[-1]) + 1
            best_adv = float(batch_advantages[best_prefix - 1])
            delta_sum = delta_prefix[best_prefix - 1]
        else:
            best_idx = int(np.argmax(batch_advantages))
            best_prefix = 0
            best_adv = float(batch_advantages[best_idx])
            delta_sum = np.zeros_like(residual, dtype=np.float32)
    if best_prefix == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(best_adv if np.isfinite(best_adv) else 0.0),
            mean_advantage=0.0,
        )
    idx = selected[:best_prefix]
    return TransportResult(
        accepted_indices=idx.astype(np.int32),
        delta_sum=delta_sum.astype(np.float32),
        batch_advantage=float(best_adv),
        mean_advantage=float(advantages[idx].mean()),
    )


@jax.jit
def _choose_transport_prefix_jit(
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
    strategy_code: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    phi_old = eval_records_queries_arrays(old_rows, attrs, ops, values, lows, highs)
    phi_new = eval_records_queries_arrays(new_rows, attrs, ops, values, lows, highs)
    delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    delta_prefix = jnp.cumsum(delta, axis=0)
    cost_prefix = jnp.cumsum(edit_cost.astype(jnp.float32))
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    prefix_advantages = (
        delta_prefix @ weights
        - 0.5 * ((delta_prefix * delta_prefix) @ inv_variance.astype(jnp.float32))
        - lambda_cost.astype(jnp.float32) * cost_prefix
    )
    best_idx = jnp.argmax(prefix_advantages)
    positive = prefix_advantages > 0.0
    last_positive_idx = jnp.max(jnp.where(positive, jnp.arange(prefix_advantages.shape[0], dtype=jnp.int32), -1))
    use_best = strategy_code == jnp.int32(1)
    chosen_idx = jnp.where(use_best, best_idx.astype(jnp.int32), last_positive_idx)
    chosen_adv = jnp.where(chosen_idx >= 0, prefix_advantages[chosen_idx], jnp.max(prefix_advantages))
    chosen_delta = jnp.where(
        chosen_idx >= 0,
        delta_prefix[chosen_idx],
        jnp.zeros((residual.shape[0],), dtype=jnp.float32),
    )
    accepted_count = jnp.maximum(chosen_idx + 1, 0)
    return accepted_count.astype(jnp.int32), chosen_delta.astype(jnp.float32), chosen_adv.astype(jnp.float32)


def choose_transport_batch_jax(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    selected: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    prefix_strategy: str = "largest_positive",
) -> TransportResult:
    if len(selected) == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
        )
    order = np.argsort(-advantages[selected])
    selected = selected[order]
    strategy_code = 1 if prefix_strategy == "best_advantage" else 0
    accepted_count, delta_sum, batch_adv = _choose_transport_prefix_jit(
        jnp.asarray(candidates.old_rows[selected], dtype=jnp.int32),
        jnp.asarray(candidates.new_rows[selected], dtype=jnp.int32),
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(inv_variance, dtype=jnp.float32),
        jnp.asarray(candidates.edit_cost[selected], dtype=jnp.float32),
        jnp.asarray(qcat.attrs, dtype=jnp.int32),
        jnp.asarray(qcat.ops, dtype=jnp.int32),
        jnp.asarray(qcat.values, dtype=jnp.int32),
        jnp.asarray(qcat.lows, dtype=jnp.int32),
        jnp.asarray(qcat.highs, dtype=jnp.int32),
        jnp.asarray(lambda_cost, dtype=jnp.float32),
        jnp.asarray(strategy_code, dtype=jnp.int32),
    )
    count = int(np.asarray(accepted_count))
    if count <= 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(np.asarray(batch_adv)),
            mean_advantage=0.0,
        )
    idx = selected[:count]
    return TransportResult(
        accepted_indices=idx.astype(np.int32),
        delta_sum=np.asarray(delta_sum, dtype=np.float32),
        batch_advantage=float(np.asarray(batch_adv)),
        mean_advantage=float(advantages[idx].mean()),
    )


def apply_edits(X_syn: np.ndarray, candidates: CandidateBatch, accepted_indices: np.ndarray) -> None:
    if len(accepted_indices) == 0:
        return
    X_syn[candidates.row_ids[accepted_indices]] = candidates.new_rows[accepted_indices]
