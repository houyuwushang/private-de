from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.eval_jax import eval_records_queries_arrays
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue
from qdte.schema import TableSchema


@dataclass
class GpuCandidateScoreBatch:
    candidates: CandidateBatch
    advantages: np.ndarray


@dataclass
class GpuCandidateContext:
    attrs: jax.Array
    ops: jax.Array
    values: jax.Array
    lows: jax.Array
    highs: jax.Array
    linear_attrs: jax.Array
    linear_weights: jax.Array
    linear_thresholds: jax.Array
    linear_num_terms: jax.Array
    affected_qids: jax.Array
    query_attr_words: jax.Array
    num_terms: jax.Array
    cardinalities: jax.Array
    mutable_attrs: jax.Array
    numeric_weights: jax.Array
    lambda_cost: jax.Array
    per_device_total: int
    random_per_device: int
    source_draws: int
    local_top_k: int
    active_query_capacity: int
    total_candidates: int
    batches_per_iter: int
    score_query_block_size: int
    score_query_block_count: int
    query_count: int
    real_query_count: int
    sparse_score_mode: int
    sparse_query_block_size: int
    sparse_query_block_count: int
    sparse_changed_attr_capacity: int


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


def _score_rows_dense(
    old_rows: jax.Array,
    new_rows: jax.Array,
    edit_cost: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    lambda_cost: jax.Array,
) -> jax.Array:
    phi_old = eval_records_queries_arrays(old_rows, attrs, ops, values, lows, highs)
    phi_new = eval_records_queries_arrays(new_rows, attrs, ops, values, lows, highs)
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    enter = phi_new & (~phi_old)
    exit_ = phi_old & (~phi_new)
    linear = enter.astype(jnp.float32) @ weights - exit_.astype(jnp.float32) @ weights
    quad = (enter | exit_).astype(jnp.float32) @ inv_variance.astype(jnp.float32)
    return linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost


def _score_rows_query_blocks(
    old_rows: jax.Array,
    new_rows: jax.Array,
    edit_cost: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    lambda_cost: jax.Array,
    score_query_block_size: int,
    score_query_block_count: int,
) -> jax.Array:
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    linear = jnp.zeros((old_rows.shape[0],), dtype=jnp.float32)
    quad = jnp.zeros((old_rows.shape[0],), dtype=jnp.float32)
    for block_id in range(score_query_block_count):
        start = block_id * score_query_block_size
        stop = start + score_query_block_size
        block_attrs = attrs[start:stop]
        block_ops = ops[start:stop]
        block_values = values[start:stop]
        block_lows = lows[start:stop]
        block_highs = highs[start:stop]
        phi_old = eval_records_queries_arrays(old_rows, block_attrs, block_ops, block_values, block_lows, block_highs)
        phi_new = eval_records_queries_arrays(new_rows, block_attrs, block_ops, block_values, block_lows, block_highs)
        block_weights = weights[start:stop]
        block_inv = inv_variance[start:stop].astype(jnp.float32)
        enter = phi_new & (~phi_old)
        exit_ = phi_old & (~phi_new)
        linear = linear + enter.astype(jnp.float32) @ block_weights - exit_.astype(jnp.float32) @ block_weights
        quad = quad + (enter | exit_).astype(jnp.float32) @ block_inv
    return linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost


def _eval_rows_query_grid(
    rows: jax.Array,
    qids: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
) -> jax.Array:
    max_terms = attrs.shape[1]
    row_idx = jnp.arange(rows.shape[0], dtype=jnp.int32)[:, None]
    q_attrs = attrs[qids]
    q_ops = ops[qids]
    q_values = values[qids]
    q_lows = lows[qids]
    q_highs = highs[qids]
    satisfied = jnp.ones(qids.shape, dtype=jnp.bool_)
    for term in range(max_terms):
        attr = q_attrs[:, :, term]
        valid = attr >= 0
        xvals = rows[row_idx, jnp.maximum(attr, 0)]
        op = q_ops[:, :, term]
        value = q_values[:, :, term]
        lo = q_lows[:, :, term]
        hi = q_highs[:, :, term]
        cond = jnp.where(op == OP_EQ, xvals == value, xvals <= value)
        cond = jnp.where(op == OP_GE, xvals >= value, cond)
        cond = jnp.where(op == OP_RANGE, (xvals >= lo) & (xvals <= hi), cond)
        satisfied = satisfied & jnp.where(valid, cond, True)

    q_linear_attrs = linear_attrs[qids]
    q_linear_weights = linear_weights[qids]
    linear_scores = jnp.zeros(qids.shape, dtype=jnp.float32)
    for term in range(max_terms):
        attr = q_linear_attrs[:, :, term]
        valid = attr >= 0
        xvals = rows[row_idx, jnp.maximum(attr, 0)].astype(jnp.float32)
        linear_scores = linear_scores + jnp.where(valid, xvals * q_linear_weights[:, :, term], 0.0)
    linear_valid = linear_num_terms[qids] > 0
    linear_cond = linear_scores <= linear_thresholds[qids].astype(jnp.float32)
    return satisfied & jnp.where(linear_valid, linear_cond, True)


def _changed_attrs_sorted(old_rows: jax.Array, new_rows: jax.Array, capacity: int) -> tuple[jax.Array, jax.Array]:
    changed = old_rows != new_rows
    d = old_rows.shape[1]
    attr_ids = jnp.arange(d, dtype=jnp.int32)[None, :]
    keys = jnp.where(changed, attr_ids, jnp.int32(d))
    word_count = max(1, (d + 31) // 32)
    word_ids = attr_ids // jnp.int32(32)
    bit_offsets = attr_ids % jnp.int32(32)
    bit_values = jnp.uint32(1) << bit_offsets.astype(jnp.uint32)
    changed_words = jnp.zeros((old_rows.shape[0], word_count), dtype=jnp.uint32)
    for word in range(word_count):
        word_bits = jnp.sum(
            jnp.where(changed & (word_ids == jnp.int32(word)), bit_values, jnp.uint32(0)),
            axis=1,
        ).astype(jnp.uint32)
        changed_words = changed_words.at[:, word].set(word_bits)
    return jnp.sort(keys, axis=1)[:, :capacity], changed_words


def _score_rows_sparse_delta(
    old_rows: jax.Array,
    new_rows: jax.Array,
    edit_cost: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
    affected_qids: jax.Array,
    query_attr_words: jax.Array,
    lambda_cost: jax.Array,
    sparse_query_block_size: int,
    sparse_query_block_count: int,
    sparse_changed_attr_capacity: int,
    real_query_count: int,
) -> jax.Array:
    changed_attrs, changed_words = _changed_attrs_sorted(old_rows, new_rows, sparse_changed_attr_capacity)
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    linear = jnp.zeros((old_rows.shape[0],), dtype=jnp.float32)
    quad = jnp.zeros((old_rows.shape[0],), dtype=jnp.float32)
    if sparse_query_block_count == 0:
        return -lambda_cost.astype(jnp.float32) * edit_cost

    def attr_body(attr_slot: int, outer_carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        linear_acc, quad_acc = outer_carry
        changed_attr = changed_attrs[:, attr_slot]
        attr_valid = changed_attr < old_rows.shape[1]
        safe_attr = jnp.where(attr_valid, changed_attr, 0)
        safe_word = safe_attr // jnp.int32(32)
        safe_offset = safe_attr % jnp.int32(32)
        word_ids = jnp.arange(query_attr_words.shape[1], dtype=jnp.int32)[None, :]
        lower_word = word_ids < safe_word[:, None]
        same_word = word_ids == safe_word[:, None]
        lower_mask = (jnp.uint32(1) << safe_offset.astype(jnp.uint32)) - jnp.uint32(1)
        earlier_changed_words = jnp.where(
            lower_word,
            changed_words,
            jnp.where(same_word, changed_words & lower_mask[:, None], jnp.uint32(0)),
        )

        def block_body(block_id: int, inner_carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            block_linear, block_quad = inner_carry
            start = block_id * sparse_query_block_size
            qid_block = jax.lax.dynamic_slice(
                affected_qids,
                (0, start),
                (affected_qids.shape[0], sparse_query_block_size),
            )
            qids = qid_block[safe_attr]
            valid_qid = qids < jnp.int32(real_query_count)
            duplicate = jnp.any(
                (query_attr_words[qids] & earlier_changed_words[:, None, :]) != jnp.uint32(0),
                axis=2,
            )
            active = attr_valid[:, None] & valid_qid & (~duplicate)
            phi_old = _eval_rows_query_grid(
                old_rows,
                qids,
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
            phi_new = _eval_rows_query_grid(
                new_rows,
                qids,
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
            enter = active & phi_new & (~phi_old)
            exit_ = active & phi_old & (~phi_new)
            block_weights = weights[qids]
            block_inv = inv_variance[qids].astype(jnp.float32)
            block_linear = block_linear + jnp.sum(enter.astype(jnp.float32) * block_weights, axis=1)
            block_linear = block_linear - jnp.sum(exit_.astype(jnp.float32) * block_weights, axis=1)
            block_quad = block_quad + jnp.sum((enter | exit_).astype(jnp.float32) * block_inv, axis=1)
            return block_linear, block_quad

        return jax.lax.fori_loop(0, sparse_query_block_count, block_body, (linear_acc, quad_acc))

    linear, quad = jax.lax.fori_loop(0, sparse_changed_attr_capacity, attr_body, (linear, quad))
    return linear - 0.5 * quad - lambda_cost.astype(jnp.float32) * edit_cost


@partial(
    jax.pmap,
    in_axes=(
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    static_broadcasted_argnums=(21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
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
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
    affected_qids: jax.Array,
    query_attr_words: jax.Array,
    num_terms: jax.Array,
    cardinalities: jax.Array,
    mutable_attrs: jax.Array,
    numeric_weights: jax.Array,
    lambda_cost: jax.Array,
    per_device_total: int,
    random_per_device: int,
    source_draws: int,
    local_top_k: int,
    score_query_block_size: int,
    score_query_block_count: int,
    sparse_score_mode: int,
    sparse_query_block_size: int,
    sparse_query_block_count: int,
    sparse_changed_attr_capacity: int,
    real_query_count: int,
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

    if sparse_score_mode:
        scores = _score_rows_sparse_delta(
            old_rows,
            new_rows,
            edit_cost,
            residual,
            inv_variance,
            attrs,
            ops,
            values,
            lows,
            highs,
            linear_attrs,
            linear_weights,
            linear_thresholds,
            linear_num_terms,
            affected_qids,
            query_attr_words,
            lambda_cost,
            sparse_query_block_size,
            sparse_query_block_count,
            sparse_changed_attr_capacity,
            real_query_count,
        )
    elif score_query_block_size > 0:
        scores = _score_rows_query_blocks(
            old_rows,
            new_rows,
            edit_cost,
            residual,
            inv_variance,
            attrs,
            ops,
            values,
            lows,
            highs,
            lambda_cost,
            score_query_block_size,
            score_query_block_count,
        )
    else:
        scores = _score_rows_dense(
            old_rows,
            new_rows,
            edit_cost,
            residual,
            inv_variance,
            attrs,
            ops,
            values,
            lows,
            highs,
            lambda_cost,
        )
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


def _pad_active_query_ids(target_query_ids: np.ndarray, active_query_capacity: int) -> np.ndarray:
    qids = np.asarray(target_query_ids, dtype=np.int32)
    if len(qids) == 0 or active_query_capacity <= 0:
        return qids
    if len(qids) == active_query_capacity:
        return qids
    if len(qids) > active_query_capacity:
        return qids[:active_query_capacity]
    repeats = int(math.ceil(active_query_capacity / len(qids)))
    return np.tile(qids, repeats)[:active_query_capacity].astype(np.int32, copy=False)


def _pad_query_arrays_for_block_scoring(
    qcat: QueryCatalogue,
    score_query_block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if score_query_block_size <= 0 or score_query_block_size >= qcat.m:
        return qcat.attrs, qcat.ops, qcat.values, qcat.lows, qcat.highs, 0, qcat.m
    block_count = int(math.ceil(qcat.m / score_query_block_size))
    padded_count = block_count * score_query_block_size
    pad_rows = padded_count - qcat.m
    attrs = np.pad(qcat.attrs, ((0, pad_rows), (0, 0)), constant_values=-1)
    ops = np.pad(qcat.ops, ((0, pad_rows), (0, 0)), constant_values=OP_EQ)
    values = np.pad(qcat.values, ((0, pad_rows), (0, 0)), constant_values=0)
    lows = np.pad(qcat.lows, ((0, pad_rows), (0, 0)), constant_values=0)
    highs = np.pad(qcat.highs, ((0, pad_rows), (0, 0)), constant_values=0)
    return attrs, ops, values, lows, highs, block_count, padded_count


def _query_arrays_with_dummy(
    qcat: QueryCatalogue,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    return (
        np.pad(qcat.attrs, ((0, 1), (0, 0)), constant_values=-1),
        np.pad(qcat.ops, ((0, 1), (0, 0)), constant_values=OP_EQ),
        np.pad(qcat.values, ((0, 1), (0, 0)), constant_values=0),
        np.pad(qcat.lows, ((0, 1), (0, 0)), constant_values=0),
        np.pad(qcat.highs, ((0, 1), (0, 0)), constant_values=0),
        np.pad(qcat.linear_attrs, ((0, 1), (0, 0)), constant_values=-1),
        np.pad(qcat.linear_weights, ((0, 1), (0, 0)), constant_values=0.0),
        np.pad(qcat.linear_thresholds, (0, 1), constant_values=0.0),
        np.pad(qcat.linear_num_terms, (0, 1), constant_values=0),
    )


def _pad_affected_query_ids(values: np.ndarray, block_size: int, pad_value: int) -> tuple[np.ndarray, int]:
    if values.shape[1] == 0:
        return values, 0
    block_count = int(math.ceil(values.shape[1] / block_size))
    padded_width = block_count * block_size
    if padded_width == values.shape[1]:
        return values, block_count
    pad_width = padded_width - values.shape[1]
    return np.pad(values, ((0, 0), (0, pad_width)), constant_values=pad_value), block_count


def _pad_score_vector(values: np.ndarray, target_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == target_size:
        return arr
    if len(arr) > target_size:
        return arr[:target_size]
    return np.pad(arr, (0, target_size - len(arr)), constant_values=0.0).astype(np.float32, copy=False)


def prepare_gpu_candidate_context(qcat: QueryCatalogue, schema: TableSchema, config: dict[str, Any]) -> GpuCandidateContext:
    cfg = config.get("qdte", {})
    total_candidates = int(cfg.get("total_candidates_per_iter", 4096))
    random_fraction = float(cfg.get("random_candidate_fraction", 0.05))
    numerical_gamma = float(cfg.get("numerical_distance_gamma", 0.1))
    lambda_cost = float(cfg.get("lambda_cost", 0.01))
    source_draws = int(cfg.get("gpu_source_draws", 8))
    gpu_return_top_k = int(cfg.get("gpu_return_top_k", 0))
    active_query_capacity = int(cfg.get("num_active_targets", 64))
    batches_per_iter = max(1, int(cfg.get("gpu_batches_per_iter", 1)))
    score_query_block_size = max(0, int(cfg.get("gpu_score_query_block_size", 0)))
    sparse_score_mode = int(str(cfg.get("score_backend", "dense_gpu")) == "sparse_delta_gpu")
    sparse_query_block_size = max(1, int(cfg.get("gpu_sparse_query_block_size", 64)))
    ndev = max(1, jax.local_device_count())
    per_device_total = int(math.ceil(total_candidates / ndev))
    total_padded = per_device_total * ndev
    local_top_k = max(0, min(per_device_total, gpu_return_top_k))
    random_total = int(round(total_padded * min(1.0, max(0.0, random_fraction))))
    random_per_device = max(0, min(per_device_total, int(round(random_total / ndev))))
    mutable_attrs = np.flatnonzero(schema.cardinalities > 1).astype(np.int32)
    if len(mutable_attrs) == 0:
        mutable_attrs = np.arange(schema.d, dtype=np.int32)
    if sparse_score_mode:
        (
            attrs,
            ops,
            values,
            lows,
            highs,
            linear_attrs,
            linear_weights,
            linear_thresholds,
            linear_num_terms,
        ) = _query_arrays_with_dummy(qcat)
        score_query_block_count = 0
        query_count = qcat.m + 1
    else:
        attrs, ops, values, lows, highs, score_query_block_count, query_count = _pad_query_arrays_for_block_scoring(
            qcat,
            score_query_block_size,
        )
        pad_queries = query_count - qcat.m
        linear_attrs = np.pad(qcat.linear_attrs, ((0, pad_queries), (0, 0)), constant_values=-1)
        linear_weights = np.pad(qcat.linear_weights, ((0, pad_queries), (0, 0)), constant_values=0.0)
        linear_thresholds = np.pad(qcat.linear_thresholds, (0, pad_queries), constant_values=0.0)
        linear_num_terms = np.pad(qcat.linear_num_terms, (0, pad_queries), constant_values=0)
    delta_index = QueryDeltaIndex.build(qcat, num_attrs=schema.d)
    affected_qids = delta_index.padded_attr_query_ids(num_attrs=schema.d, pad_value=qcat.m)
    affected_qids, sparse_query_block_count = _pad_affected_query_ids(
        affected_qids,
        sparse_query_block_size,
        pad_value=qcat.m,
    )
    query_attr_words = np.pad(delta_index.query_attr_words(num_attrs=schema.d), ((0, 1), (0, 0)), constant_values=0)
    sparse_changed_attr_capacity = min(
        int(schema.d),
        max(1, int(cfg.get("gpu_sparse_changed_attr_capacity", max(1, qcat.max_terms)))),
    )
    return GpuCandidateContext(
        attrs=jax.device_put(jnp.asarray(attrs, dtype=jnp.int32)),
        ops=jax.device_put(jnp.asarray(ops, dtype=jnp.int32)),
        values=jax.device_put(jnp.asarray(values, dtype=jnp.int32)),
        lows=jax.device_put(jnp.asarray(lows, dtype=jnp.int32)),
        highs=jax.device_put(jnp.asarray(highs, dtype=jnp.int32)),
        linear_attrs=jax.device_put(jnp.asarray(linear_attrs, dtype=jnp.int32)),
        linear_weights=jax.device_put(jnp.asarray(linear_weights, dtype=jnp.float32)),
        linear_thresholds=jax.device_put(jnp.asarray(linear_thresholds, dtype=jnp.float32)),
        linear_num_terms=jax.device_put(jnp.asarray(linear_num_terms, dtype=jnp.int32)),
        affected_qids=jax.device_put(jnp.asarray(affected_qids, dtype=jnp.int32)),
        query_attr_words=jax.device_put(jnp.asarray(query_attr_words, dtype=jnp.uint32)),
        num_terms=jax.device_put(jnp.asarray(qcat.num_terms, dtype=jnp.int32)),
        cardinalities=jax.device_put(jnp.asarray(schema.cardinalities, dtype=jnp.int32)),
        mutable_attrs=jax.device_put(jnp.asarray(mutable_attrs, dtype=jnp.int32)),
        numeric_weights=jax.device_put(jnp.asarray(_numeric_weights(schema, numerical_gamma), dtype=jnp.float32)),
        lambda_cost=jax.device_put(jnp.asarray(lambda_cost, dtype=jnp.float32)),
        per_device_total=per_device_total,
        random_per_device=random_per_device,
        source_draws=max(1, source_draws),
        local_top_k=local_top_k,
        active_query_capacity=active_query_capacity,
        total_candidates=total_candidates,
        batches_per_iter=batches_per_iter,
        score_query_block_size=score_query_block_size if score_query_block_count > 0 else 0,
        score_query_block_count=score_query_block_count,
        query_count=query_count,
        real_query_count=qcat.m,
        sparse_score_mode=sparse_score_mode,
        sparse_query_block_size=sparse_query_block_size,
        sparse_query_block_count=sparse_query_block_count,
        sparse_changed_attr_capacity=sparse_changed_attr_capacity,
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
    context: GpuCandidateContext | None = None,
) -> GpuCandidateScoreBatch:
    if context is None:
        context = prepare_gpu_candidate_context(qcat, schema, config)
    total_candidates = context.total_candidates
    ndev = max(1, jax.local_device_count())
    if len(target_query_ids) == 0:
        target_query_ids = np.arange(qcat.m, dtype=np.int32)
    target_query_ids = _pad_active_query_ids(target_query_ids, context.active_query_capacity)

    returned_count = total_candidates if context.local_top_k <= 0 else min(total_candidates, context.local_top_k * ndev)
    X_repl = X_syn if not isinstance(X_syn, np.ndarray) or X_syn.ndim == 3 else replicate_table_to_devices(X_syn)
    active_j = jnp.asarray(target_query_ids, dtype=jnp.int32)
    residual_j = jnp.asarray(_pad_score_vector(residual, context.query_count), dtype=jnp.float32)
    inv_j = jnp.asarray(_pad_score_vector(inv_variance, context.query_count), dtype=jnp.float32)
    row_id_parts: list[np.ndarray] = []
    old_row_parts: list[np.ndarray] = []
    new_row_parts: list[np.ndarray] = []
    target_qid_parts: list[np.ndarray] = []
    edit_cost_parts: list[np.ndarray] = []
    repair_type_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for _ in range(context.batches_per_iter):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        keys = jax.random.split(jax.random.PRNGKey(seed), ndev)
        result = _generate_score_candidates_pmap(
            keys,
            X_repl,
            active_j,
            residual_j,
            inv_j,
            context.attrs,
            context.ops,
            context.values,
            context.lows,
            context.highs,
            context.linear_attrs,
            context.linear_weights,
            context.linear_thresholds,
            context.linear_num_terms,
            context.affected_qids,
            context.query_attr_words,
            context.num_terms,
            context.cardinalities,
            context.mutable_attrs,
            context.numeric_weights,
            context.lambda_cost,
            context.per_device_total,
            context.random_per_device,
            context.source_draws,
            context.local_top_k,
            context.score_query_block_size,
            context.score_query_block_count,
            context.sparse_score_mode,
            context.sparse_query_block_size,
            context.sparse_query_block_count,
            context.sparse_changed_attr_capacity,
            context.real_query_count,
        )
        row_ids, old_rows, new_rows, target_qids, edit_cost, repair_type, scores = [
            np.asarray(x).reshape((-1,) + tuple(x.shape[2:])) for x in result
        ]
        row_id_parts.append(row_ids.reshape(-1)[:returned_count].astype(np.int32, copy=False))
        old_row_parts.append(old_rows.reshape(-1, schema.d)[:returned_count].astype(np.int32, copy=False))
        new_row_parts.append(new_rows.reshape(-1, schema.d)[:returned_count].astype(np.int32, copy=False))
        target_qid_parts.append(target_qids.reshape(-1)[:returned_count].astype(np.int32, copy=False))
        edit_cost_parts.append(edit_cost.reshape(-1)[:returned_count].astype(np.float32, copy=False))
        repair_type_parts.append(repair_type.reshape(-1)[:returned_count].astype(np.int32, copy=False))
        score_parts.append(scores.reshape(-1)[:returned_count].astype(np.float32, copy=False))
    row_ids = np.concatenate(row_id_parts, axis=0)
    old_rows = np.concatenate(old_row_parts, axis=0)
    new_rows = np.concatenate(new_row_parts, axis=0)
    target_qids = np.concatenate(target_qid_parts, axis=0)
    edit_cost = np.concatenate(edit_cost_parts, axis=0)
    repair_type = np.concatenate(repair_type_parts, axis=0)
    scores = np.concatenate(score_parts, axis=0)
    random_produced = int(np.sum(target_qids < 0))
    scored_per_batch = int(context.per_device_total * ndev)
    effective_requested = int(total_candidates * context.batches_per_iter)
    effective_scored = int(scored_per_batch * context.batches_per_iter)
    candidates = CandidateBatch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        target_query_ids=target_qids,
        edit_cost=edit_cost,
        repair_type=repair_type,
        diagnostics={
            "requested_candidates": float(effective_requested),
            "scored_candidates": float(effective_scored),
            "produced_candidates": float(len(row_ids)),
            "directed_candidates": float(len(row_ids) - random_produced),
            "random_candidates": float(random_produced),
            "candidate_shortfall": 0.0,
            "source_filter_attempts": 0.0,
            "source_filter_failures": 0.0,
            "source_filter_kept": 0.0,
        },
    )
    return GpuCandidateScoreBatch(candidates=candidates, advantages=scores)
