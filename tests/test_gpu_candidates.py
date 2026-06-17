from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from qdte.evolution.gpu_candidates import (
    _pad_active_query_ids,
    _pad_affected_query_ids,
    _pad_query_arrays_for_block_scoring,
    _query_arrays_with_dummy,
    _score_rows_dense,
    _score_rows_query_blocks,
    _score_rows_sparse_delta,
)
from qdte.evolution.scoring import compute_deltas
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, QueryBuilder


def test_pad_active_query_ids_keeps_fixed_gpu_shape_without_changing_cycle_order() -> None:
    active = np.asarray([3, 7, 11], dtype=np.int32)

    padded = _pad_active_query_ids(active, 8)

    assert padded.dtype == np.int32
    assert padded.tolist() == [3, 7, 11, 3, 7, 11, 3, 7]


def test_pad_active_query_ids_leaves_empty_active_empty() -> None:
    padded = _pad_active_query_ids(np.empty(0, dtype=np.int32), 8)

    assert padded.size == 0


def test_query_block_padding_uses_empty_zero_weight_queries() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_LE, 2, 0, 2)], "b<=2", "prefix:1", "prefix")
    builder.add([(1, OP_GE, 3, 3, 3)], "b>=3", "prefix:1", "prefix")
    qcat = builder.build()

    attrs, ops, values, lows, highs, block_count, query_count = _pad_query_arrays_for_block_scoring(qcat, 2)

    assert block_count == 2
    assert query_count == 4
    assert attrs.shape == (4, 1)
    assert np.array_equal(attrs[:3], qcat.attrs)
    assert attrs[3, 0] == -1
    assert ops[3, 0] == OP_EQ
    assert values[3, 0] == 0
    assert lows[3, 0] == 0
    assert highs[3, 0] == 0


def test_query_block_scores_match_dense_scores() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_LE, 2, 0, 2)], "b<=2", "prefix:1", "prefix")
    builder.add([(1, OP_GE, 3, 3, 3)], "b>=3", "prefix:1", "prefix")
    qcat = builder.build()
    attrs, ops, values, lows, highs, block_count, query_count = _pad_query_arrays_for_block_scoring(qcat, 2)
    residual = np.asarray([5.0, -3.0, 2.5, 0.0], dtype=np.float32)
    inv_var = np.asarray([0.5, 2.0, 0.25, 0.0], dtype=np.float32)
    old_rows = np.asarray([[0, 3], [1, 3], [0, 1], [1, 1]], dtype=np.int32)
    new_rows = np.asarray([[1, 3], [1, 1], [1, 1], [0, 4]], dtype=np.int32)
    edit_cost = np.ones(len(old_rows), dtype=np.float32)

    dense = _score_rows_dense(
        jnp.asarray(old_rows),
        jnp.asarray(new_rows),
        jnp.asarray(edit_cost),
        jnp.asarray(residual[: qcat.m]),
        jnp.asarray(inv_var[: qcat.m]),
        jnp.asarray(qcat.attrs),
        jnp.asarray(qcat.ops),
        jnp.asarray(qcat.values),
        jnp.asarray(qcat.lows),
        jnp.asarray(qcat.highs),
        jnp.asarray(0.1, dtype=jnp.float32),
    )
    blocked = _score_rows_query_blocks(
        jnp.asarray(old_rows),
        jnp.asarray(new_rows),
        jnp.asarray(edit_cost),
        jnp.asarray(residual[:query_count]),
        jnp.asarray(inv_var[:query_count]),
        jnp.asarray(attrs),
        jnp.asarray(ops),
        jnp.asarray(values),
        jnp.asarray(lows),
        jnp.asarray(highs),
        jnp.asarray(0.1, dtype=jnp.float32),
        2,
        block_count,
    )

    assert np.allclose(np.asarray(blocked), np.asarray(dense), atol=1e-5)


def test_sparse_gpu_delta_scores_match_dense_delta_formula_for_kway_and_halfspace() -> None:
    builder = QueryBuilder(max_terms=3)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_LE, 2, 0, 2)], "b<=2", "prefix:1", "prefix")
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_LE, 2, 0, 2), (2, OP_GE, 1, 1, 3)],
        "a=1&b<=2&c>=1",
        "threeway",
        "mixed",
    )
    builder.add_halfspace([(0, 1.0), (2, -1.0)], threshold=0.0, name="a-c<=0", group="halfspace")
    qcat = builder.build()
    index = QueryDeltaIndex.build(qcat, num_attrs=3)
    affected = index.padded_attr_query_ids(num_attrs=3, pad_value=qcat.m)
    affected, block_count = _pad_affected_query_ids(affected, block_size=2, pad_value=qcat.m)
    query_attr_words = np.pad(index.query_attr_words(num_attrs=3), ((0, 1), (0, 0)), constant_values=0)
    attrs, ops, values, lows, highs, linear_attrs, linear_weights, linear_thresholds, linear_num_terms = (
        _query_arrays_with_dummy(qcat)
    )
    old_rows = np.asarray([[0, 2, 1], [1, 3, 0], [1, 1, 2], [1, 1, 0]], dtype=np.int32)
    new_rows = np.asarray([[1, 2, 1], [1, 1, 1], [0, 1, 2], [1, 3, 0]], dtype=np.int32)
    edit_cost = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    residual = np.asarray([2.0, -3.0, 5.0, 7.0, 0.0], dtype=np.float32)
    inv_var = np.asarray([1.0, 0.5, 2.0, 3.0, 0.0], dtype=np.float32)
    dense_delta = compute_deltas(old_rows, new_rows, qcat).astype(np.float32)
    expected = (
        dense_delta @ (residual[: qcat.m] * inv_var[: qcat.m])
        - 0.5 * ((dense_delta * dense_delta) @ inv_var[: qcat.m])
        - 0.25 * edit_cost
    )

    sparse = _score_rows_sparse_delta(
        jnp.asarray(old_rows),
        jnp.asarray(new_rows),
        jnp.asarray(edit_cost),
        jnp.asarray(residual),
        jnp.asarray(inv_var),
        jnp.asarray(attrs),
        jnp.asarray(ops),
        jnp.asarray(values),
        jnp.asarray(lows),
        jnp.asarray(highs),
        jnp.asarray(linear_attrs),
        jnp.asarray(linear_weights),
        jnp.asarray(linear_thresholds),
        jnp.asarray(linear_num_terms),
        jnp.asarray(affected),
        jnp.asarray(query_attr_words, dtype=jnp.uint32),
        jnp.asarray(0.25, dtype=jnp.float32),
        2,
        block_count,
        3,
        qcat.m,
    )

    assert np.allclose(np.asarray(sparse), expected, atol=1.0e-5)


def test_sparse_gpu_delta_supports_more_than_31_attrs() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a0=1", "oneway:0", "oneway")
    builder.add([(32, OP_EQ, 1, 1, 1)], "a32=1", "oneway:32", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1), (32, OP_EQ, 1, 1, 1)], "a0=1&a32=1", "cross-word", "mixed")
    builder.add_halfspace([(31, 1.0), (34, -1.0)], threshold=0.0, name="a31-a34<=0", group="halfspace")
    qcat = builder.build()
    num_attrs = 35
    index = QueryDeltaIndex.build(qcat, num_attrs=num_attrs)
    affected = index.padded_attr_query_ids(num_attrs=num_attrs, pad_value=qcat.m)
    affected, block_count = _pad_affected_query_ids(affected, block_size=2, pad_value=qcat.m)
    query_attr_words = np.pad(index.query_attr_words(num_attrs=num_attrs), ((0, 1), (0, 0)), constant_values=0)
    assert query_attr_words.shape == (qcat.m + 1, 2)

    attrs, ops, values, lows, highs, linear_attrs, linear_weights, linear_thresholds, linear_num_terms = (
        _query_arrays_with_dummy(qcat)
    )
    old_rows = np.zeros((3, num_attrs), dtype=np.int32)
    new_rows = old_rows.copy()
    new_rows[0, 0] = 1
    new_rows[0, 32] = 1
    old_rows[1, 0] = 1
    old_rows[1, 32] = 1
    old_rows[2, 31] = 1
    new_rows[2, 34] = 1
    edit_cost = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    residual = np.asarray([2.0, -3.0, 5.0, 7.0, 0.0], dtype=np.float32)
    inv_var = np.asarray([1.0, 0.5, 2.0, 3.0, 0.0], dtype=np.float32)
    dense_delta = compute_deltas(old_rows, new_rows, qcat).astype(np.float32)
    expected = (
        dense_delta @ (residual[: qcat.m] * inv_var[: qcat.m])
        - 0.5 * ((dense_delta * dense_delta) @ inv_var[: qcat.m])
        - 0.25 * edit_cost
    )

    sparse = _score_rows_sparse_delta(
        jnp.asarray(old_rows),
        jnp.asarray(new_rows),
        jnp.asarray(edit_cost),
        jnp.asarray(residual),
        jnp.asarray(inv_var),
        jnp.asarray(attrs),
        jnp.asarray(ops),
        jnp.asarray(values),
        jnp.asarray(lows),
        jnp.asarray(highs),
        jnp.asarray(linear_attrs),
        jnp.asarray(linear_weights),
        jnp.asarray(linear_thresholds),
        jnp.asarray(linear_num_terms),
        jnp.asarray(affected),
        jnp.asarray(query_attr_words, dtype=jnp.uint32),
        jnp.asarray(0.25, dtype=jnp.float32),
        2,
        block_count,
        2,
        qcat.m,
    )

    assert np.allclose(np.asarray(sparse), expected, atol=1.0e-5)
