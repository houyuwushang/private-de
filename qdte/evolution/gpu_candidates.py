from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.queries.eval_jax import eval_records_queries_arrays
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue
from qdte.schema import TableSchema


@dataclass
class GpuCandidateScoreBatch:
    candidates: CandidateBatch
    advantages: np.ndarray


def _randint_mod(key: jax.Array, shape: tuple[int, ...], low: jax.Array, high: jax.Array) -> jax.Array:
    span = jnp.maximum(high - low, 1)
    raw = jax.random.randint(key, shape, minval=0, maxval=jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    return low + (raw % span.astype(jnp.int32))


def _set_column(rows: jax.Array, attrs: jax.Array, values: jax.Array, mask: jax.Array) -> jax.Array:
    n = rows.shape[0]
    row_idx = jnp.arange(n, dtype=jnp.int32)
    return rows.at[row_idx, attrs].set(jnp.where(mask, values, rows[row_idx, attrs]))


def _repair_directed_rows(
    key: jax.Array,
    old_rows: jax.Array,
    qids: jax.Array,
    residual: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    num_terms: jax.Array,
    cardinalities: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    n = old_rows.shape[0]
    max_terms = attrs.shape[1]
    keys = jax.random.split(key, max_terms + 1)
    q_attrs = attrs[qids]
    q_ops = ops[qids]
    q_values = values[qids]
    q_lows = lows[qids]
    q_highs = highs[qids]
    q_num_terms = jnp.maximum(num_terms[qids], 1)
    need_enter = residual[qids] > 0
    break_term = jax.random.randint(keys[0], (n,), minval=0, maxval=max_terms, dtype=jnp.int32) % q_num_terms
    new_rows = old_rows

    for term in range(max_terms):
        attr = jnp.maximum(q_attrs[:, term], 0)
        valid = q_attrs[:, term] >= 0
        op = q_ops[:, term]
        value = q_values[:, term]
        lo = q_lows[:, term]
        hi = q_highs[:, term]
        card = cardinalities[attr]

        enter_low_eq = value
        enter_high_eq = value + 1
        enter_low_le = jnp.zeros_like(value)
        enter_high_le = jnp.minimum(value + 1, card)
        enter_low_ge = jnp.minimum(jnp.maximum(value, 0), jnp.maximum(card - 1, 0))
        enter_high_ge = card
        enter_low_range = jnp.minimum(jnp.maximum(lo, 0), jnp.maximum(card - 1, 0))
        enter_high_range = jnp.maximum(enter_low_range + 1, jnp.minimum(hi + 1, card))

        enter_low = jnp.where(op == OP_EQ, enter_low_eq, enter_low_le)
        enter_high = jnp.where(op == OP_EQ, enter_high_eq, enter_high_le)
        enter_low = jnp.where(op == OP_GE, enter_low_ge, enter_low)
        enter_high = jnp.where(op == OP_GE, enter_high_ge, enter_high)
        enter_low = jnp.where(op == OP_RANGE, enter_low_range, enter_low)
        enter_high = jnp.where(op == OP_RANGE, enter_high_range, enter_high)
        enter_value = _randint_mod(keys[term + 1], (n,), enter_low.astype(jnp.int32), enter_high.astype(jnp.int32))

        raw = jax.random.randint(
            keys[term + 1],
            (n,),
            minval=0,
            maxval=jnp.iinfo(jnp.int32).max,
            dtype=jnp.int32,
        )
        eq_break = raw % jnp.maximum(card - 1, 1)
        eq_break = eq_break + (eq_break >= value)
        le_break = value + 1 + (raw % jnp.maximum(card - value - 1, 1))
        ge_break = raw % jnp.maximum(value, 1)
        range_break = jnp.where(lo > 0, lo - 1, hi + 1)
        range_break = jnp.clip(range_break, 0, card - 1)
        break_value = jnp.where(op == OP_EQ, eq_break, le_break)
        break_value = jnp.where(op == OP_GE, ge_break, break_value)
        break_value = jnp.where(op == OP_RANGE, range_break, break_value)

        can_break = jnp.where(op == OP_EQ, card > 1, value + 1 < card)
        can_break = jnp.where(op == OP_GE, value > 0, can_break)
        can_break = jnp.where(op == OP_RANGE, (lo > 0) | (hi + 1 < card), can_break)
        use_break = (~need_enter) & (break_term == term) & valid & can_break
        use_enter = need_enter & valid
        term_value = jnp.where(use_enter, enter_value, break_value)
        new_rows = _set_column(new_rows, attr, term_value.astype(jnp.int32), use_enter | use_break)

    return new_rows, jnp.where(need_enter, jnp.int32(1), jnp.int32(2))


def _random_mutation_rows(
    key: jax.Array,
    old_rows: jax.Array,
    mutable_attrs: jax.Array,
    cardinalities: jax.Array,
) -> jax.Array:
    n = old_rows.shape[0]
    attr_pos = jax.random.randint(key, (n,), minval=0, maxval=mutable_attrs.shape[0], dtype=jnp.int32)
    attrs = mutable_attrs[attr_pos]
    row_idx = jnp.arange(n, dtype=jnp.int32)
    old = old_rows[row_idx, attrs]
    card = cardinalities[attrs]
    raw = jax.random.randint(key, (n,), minval=0, maxval=jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    vals = raw % jnp.maximum(card - 1, 1)
    vals = vals + (vals >= old)
    return old_rows.at[row_idx, attrs].set(vals.astype(jnp.int32))


def _eval_candidate_source_satisfaction(
    rows: jax.Array,
    qids: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    num_terms: jax.Array,
) -> jax.Array:
    n = rows.shape[0]
    draws = rows.shape[1]
    max_terms = attrs.shape[1]
    q_attrs = attrs[qids]
    q_ops = ops[qids]
    q_values = values[qids]
    q_lows = lows[qids]
    q_highs = highs[qids]
    q_num_terms = num_terms[qids]
    row_idx = jnp.arange(n, dtype=jnp.int32)[:, None]
    draw_idx = jnp.arange(draws, dtype=jnp.int32)[None, :]
    satisfied = jnp.ones((n, draws), dtype=jnp.bool_)
    for term in range(max_terms):
        attr = jnp.maximum(q_attrs[:, term], 0)
        xvals = rows[row_idx, draw_idx, attr[:, None]]
        op = q_ops[:, term]
        value = q_values[:, term]
        lo = q_lows[:, term]
        hi = q_highs[:, term]
        cond = jnp.where(op[:, None] == OP_EQ, xvals == value[:, None], xvals <= value[:, None])
        cond = jnp.where(op[:, None] == OP_GE, xvals >= value[:, None], cond)
        cond = jnp.where(op[:, None] == OP_RANGE, (xvals >= lo[:, None]) & (xvals <= hi[:, None]), cond)
        valid = term < q_num_terms
        satisfied = satisfied & jnp.where(valid[:, None], cond, True)
    return satisfied


@partial(
    jax.pmap,
    in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(15, 16, 17, 18),
)
def _generate_score_candidates_pmap(
    key: jax.Array,
    X_syn: jax.Array,
    active_qids: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    num_terms: jax.Array,
    cardinalities: jax.Array,
    mutable_attrs: jax.Array,
    numeric_weights: jax.Array,
    lambda_cost: jax.Array,
    per_device_total: int,
    random_per_device: int,
    source_draws: int,
    local_top_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    directed_count = per_device_total - random_per_device
    keys = jax.random.split(key, 6)
    n_rows = X_syn.shape[0]

    active_idx = jnp.arange(directed_count, dtype=jnp.int32) % active_qids.shape[0]
    directed_qids = active_qids[active_idx]
    directed_row_draws = jax.random.randint(keys[0], (directed_count, source_draws), 0, n_rows, dtype=jnp.int32)
    old_draws = X_syn[directed_row_draws]
    draw_sat = _eval_candidate_source_satisfaction(
        old_draws,
        directed_qids,
        attrs,
        ops,
        values,
        lows,
        highs,
        num_terms,
    )
    need_source_sat = ~(residual[directed_qids] > 0)
    matches = draw_sat == need_source_sat[:, None]
    chosen_draw = jnp.argmax(matches.astype(jnp.int32), axis=1)
    chosen_draw = jnp.where(jnp.any(matches, axis=1), chosen_draw, 0)
    directed_arange = jnp.arange(directed_count, dtype=jnp.int32)
    directed_row_ids = directed_row_draws[directed_arange, chosen_draw]
    old_directed = old_draws[directed_arange, chosen_draw]
    new_directed, directed_repair = _repair_directed_rows(
        keys[1],
        old_directed,
        directed_qids,
        residual,
        attrs,
        ops,
        values,
        lows,
        highs,
        num_terms,
        cardinalities,
    )

    random_row_ids = jax.random.randint(keys[2], (random_per_device,), 0, n_rows, dtype=jnp.int32)
    old_random = X_syn[random_row_ids]
    new_random = _random_mutation_rows(keys[3], old_random, mutable_attrs, cardinalities)

    row_ids = jnp.concatenate([directed_row_ids, random_row_ids], axis=0)
    old_rows = jnp.concatenate([old_directed, old_random], axis=0)
    new_rows = jnp.concatenate([new_directed, new_random], axis=0)
    target_qids = jnp.concatenate(
        [directed_qids, jnp.full((random_per_device,), -1, dtype=jnp.int32)],
        axis=0,
    )
    repair_type = jnp.concatenate(
        [directed_repair, jnp.zeros((random_per_device,), dtype=jnp.int32)],
        axis=0,
    )

    changed = old_rows != new_rows
    hamming = jnp.sum(changed.astype(jnp.float32), axis=1)
    numeric_cost = jnp.abs(old_rows.astype(jnp.float32) - new_rows.astype(jnp.float32)) @ numeric_weights
    edit_cost = hamming + numeric_cost

    phi_old = eval_records_queries_arrays(old_rows, attrs, ops, values, lows, highs)
    phi_new = eval_records_queries_arrays(new_rows, attrs, ops, values, lows, highs)
    delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    linear = delta @ weights
    quad = (delta * delta) @ inv_variance.astype(jnp.float32)
    scores = linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost
    if 0 < local_top_k < per_device_total:
        scores, top_idx = jax.lax.top_k(scores, local_top_k)
        row_ids = row_ids[top_idx]
        old_rows = old_rows[top_idx]
        new_rows = new_rows[top_idx]
        target_qids = target_qids[top_idx]
        edit_cost = edit_cost[top_idx]
        repair_type = repair_type[top_idx]
    return row_ids, old_rows, new_rows, target_qids, edit_cost, repair_type, scores


def _numeric_weights(schema: TableSchema, numerical_gamma: float) -> np.ndarray:
    weights = np.zeros(schema.d, dtype=np.float32)
    for attr in schema.numerical_indices:
        denom = max(1, int(schema.cardinalities[attr]) - 1)
        weights[int(attr)] = float(numerical_gamma) / float(denom)
    return weights


def replicate_table_to_devices(X: np.ndarray) -> jax.Array:
    ndev = max(1, jax.local_device_count())
    stacked = np.broadcast_to(np.asarray(X, dtype=np.int32), (ndev,) + tuple(X.shape)).copy()
    return jax.device_put(stacked)


@partial(jax.pmap, in_axes=(0, None, None))
def _apply_edits_replicated_pmap(X_repl: jax.Array, row_ids: jax.Array, new_rows: jax.Array) -> jax.Array:
    return X_repl.at[row_ids].set(new_rows)


def apply_edits_to_replicated_table(X_repl: jax.Array, row_ids: np.ndarray, new_rows: np.ndarray) -> jax.Array:
    if len(row_ids) == 0:
        return X_repl
    return _apply_edits_replicated_pmap(
        X_repl,
        jnp.asarray(row_ids, dtype=jnp.int32),
        jnp.asarray(new_rows, dtype=jnp.int32),
    )


def generate_and_score_candidates_gpu(
    X_syn: np.ndarray | jax.Array,
    qcat: QueryCatalogue,
    schema: TableSchema,
    target_query_ids: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> GpuCandidateScoreBatch:
    cfg = config.get("qdte", {})
    total_candidates = int(cfg.get("total_candidates_per_iter", 4096))
    random_fraction = float(cfg.get("random_candidate_fraction", 0.05))
    numerical_gamma = float(cfg.get("numerical_distance_gamma", 0.1))
    lambda_cost = float(cfg.get("lambda_cost", 0.01))
    source_draws = int(cfg.get("gpu_source_draws", 8))
    gpu_return_top_k = int(cfg.get("gpu_return_top_k", 0))
    ndev = max(1, jax.local_device_count())
    per_device_total = int(math.ceil(total_candidates / ndev))
    total_padded = per_device_total * ndev
    local_top_k = max(0, min(per_device_total, gpu_return_top_k))
    random_total = int(round(total_padded * min(1.0, max(0.0, random_fraction))))
    random_per_device = max(0, min(per_device_total, int(round(random_total / ndev))))
    if len(target_query_ids) == 0:
        target_query_ids = np.arange(qcat.m, dtype=np.int32)

    mutable_attrs = np.flatnonzero(schema.cardinalities > 1).astype(np.int32)
    if len(mutable_attrs) == 0:
        mutable_attrs = np.arange(schema.d, dtype=np.int32)
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    keys = jax.random.split(jax.random.PRNGKey(seed), ndev)
    result = _generate_score_candidates_pmap(
        keys,
        X_syn if not isinstance(X_syn, np.ndarray) or X_syn.ndim == 3 else replicate_table_to_devices(X_syn),
        jnp.asarray(target_query_ids, dtype=jnp.int32),
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(inv_variance, dtype=jnp.float32),
        jnp.asarray(qcat.attrs, dtype=jnp.int32),
        jnp.asarray(qcat.ops, dtype=jnp.int32),
        jnp.asarray(qcat.values, dtype=jnp.int32),
        jnp.asarray(qcat.lows, dtype=jnp.int32),
        jnp.asarray(qcat.highs, dtype=jnp.int32),
        jnp.asarray(qcat.num_terms, dtype=jnp.int32),
        jnp.asarray(schema.cardinalities, dtype=jnp.int32),
        jnp.asarray(mutable_attrs, dtype=jnp.int32),
        jnp.asarray(_numeric_weights(schema, numerical_gamma), dtype=jnp.float32),
        jnp.asarray(lambda_cost, dtype=jnp.float32),
        per_device_total,
        random_per_device,
        max(1, source_draws),
        local_top_k,
    )
    row_ids, old_rows, new_rows, target_qids, edit_cost, repair_type, scores = [np.asarray(x).reshape((-1,) + tuple(x.shape[2:])) for x in result]
    returned_count = total_candidates if local_top_k <= 0 else min(total_candidates, local_top_k * ndev)
    row_ids = row_ids.reshape(-1)[:returned_count].astype(np.int32, copy=False)
    old_rows = old_rows.reshape(-1, schema.d)[:returned_count].astype(np.int32, copy=False)
    new_rows = new_rows.reshape(-1, schema.d)[:returned_count].astype(np.int32, copy=False)
    target_qids = target_qids.reshape(-1)[:returned_count].astype(np.int32, copy=False)
    edit_cost = edit_cost.reshape(-1)[:returned_count].astype(np.float32, copy=False)
    repair_type = repair_type.reshape(-1)[:returned_count].astype(np.int32, copy=False)
    scores = scores.reshape(-1)[:returned_count].astype(np.float32, copy=False)
    random_produced = int(np.sum(target_qids < 0))
    candidates = CandidateBatch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        target_query_ids=target_qids,
        edit_cost=edit_cost,
        repair_type=repair_type,
        diagnostics={
            "requested_candidates": float(total_candidates),
            "scored_candidates": float(total_candidates),
            "produced_candidates": float(returned_count),
            "directed_candidates": float(returned_count - random_produced),
            "random_candidates": float(random_produced),
            "candidate_shortfall": 0.0,
            "source_filter_attempts": 0.0,
            "source_filter_failures": 0.0,
            "source_filter_kept": 0.0,
        },
    )
    return GpuCandidateScoreBatch(candidates=candidates, advantages=scores)
