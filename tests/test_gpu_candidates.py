from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from qdte.evolution.gpu_candidates import (
    generate_and_score_candidates_gpu,
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
from qdte.schema import ColumnSchema, TableSchema


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
        jnp.asarray(qcat.linear_attrs),
        jnp.asarray(qcat.linear_weights),
        jnp.asarray(qcat.linear_thresholds),
        jnp.asarray(qcat.linear_num_terms),
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
        jnp.asarray(np.pad(qcat.linear_attrs, ((0, query_count - qcat.m), (0, 0)), constant_values=-1)),
        jnp.asarray(np.pad(qcat.linear_weights, ((0, query_count - qcat.m), (0, 0)), constant_values=0.0)),
        jnp.asarray(np.pad(qcat.linear_thresholds, (0, query_count - qcat.m), constant_values=0.0)),
        jnp.asarray(np.pad(qcat.linear_num_terms, (0, query_count - qcat.m), constant_values=0)),
        jnp.asarray(0.1, dtype=jnp.float32),
        2,
        block_count,
    )

    assert np.allclose(np.asarray(blocked), np.asarray(dense), atol=1e-5)


def test_dense_and_block_gpu_scores_include_halfspace_terms() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add_halfspace([(0, 1.0), (1, 1.0)], threshold=2.0, name="a+b<=2", group="halfspace")
    qcat = builder.build()
    attrs, ops, values, lows, highs, block_count, query_count = _pad_query_arrays_for_block_scoring(qcat, 1)
    pad_queries = query_count - qcat.m
    linear_attrs = np.pad(qcat.linear_attrs, ((0, pad_queries), (0, 0)), constant_values=-1)
    linear_weights = np.pad(qcat.linear_weights, ((0, pad_queries), (0, 0)), constant_values=0.0)
    linear_thresholds = np.pad(qcat.linear_thresholds, (0, pad_queries), constant_values=0.0)
    linear_num_terms = np.pad(qcat.linear_num_terms, (0, pad_queries), constant_values=0)
    old_rows = np.asarray([[4, 4], [1, 1], [0, 3]], dtype=np.int32)
    new_rows = np.asarray([[1, 1], [4, 4], [2, 0]], dtype=np.int32)
    edit_cost = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    residual = np.asarray([3.0, 5.0], dtype=np.float32)
    inv_var = np.asarray([1.0, 2.0], dtype=np.float32)
    dense_delta = compute_deltas(old_rows, new_rows, qcat).astype(np.float32)
    expected = dense_delta @ (residual * inv_var) - 0.5 * ((dense_delta * dense_delta) @ inv_var) - 0.1

    dense = _score_rows_dense(
        jnp.asarray(old_rows),
        jnp.asarray(new_rows),
        jnp.asarray(edit_cost),
        jnp.asarray(residual),
        jnp.asarray(inv_var),
        jnp.asarray(qcat.attrs),
        jnp.asarray(qcat.ops),
        jnp.asarray(qcat.values),
        jnp.asarray(qcat.lows),
        jnp.asarray(qcat.highs),
        jnp.asarray(qcat.linear_attrs),
        jnp.asarray(qcat.linear_weights),
        jnp.asarray(qcat.linear_thresholds),
        jnp.asarray(qcat.linear_num_terms),
        jnp.asarray(0.1, dtype=jnp.float32),
    )
    blocked = _score_rows_query_blocks(
        jnp.asarray(old_rows),
        jnp.asarray(new_rows),
        jnp.asarray(edit_cost),
        jnp.asarray(np.pad(residual, (0, pad_queries), constant_values=0.0)),
        jnp.asarray(np.pad(inv_var, (0, pad_queries), constant_values=0.0)),
        jnp.asarray(attrs),
        jnp.asarray(ops),
        jnp.asarray(values),
        jnp.asarray(lows),
        jnp.asarray(highs),
        jnp.asarray(linear_attrs),
        jnp.asarray(linear_weights),
        jnp.asarray(linear_thresholds),
        jnp.asarray(linear_num_terms),
        jnp.asarray(0.1, dtype=jnp.float32),
        1,
        block_count,
    )

    assert np.allclose(np.asarray(dense), expected, atol=1.0e-5)
    assert np.allclose(np.asarray(blocked), expected, atol=1.0e-5)


def test_fused_gpu_repair_enters_and_exits_halfspace_queries() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add_halfspace([(0, 1.0), (1, 1.0)], threshold=2.0, name="a+b<=2", group="halfspace")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "numerical_binned", 5),
            ColumnSchema("b", "numerical_binned", 5),
        ]
    )
    X_syn = np.asarray([[4, 4], [1, 1], [3, 0], [0, 3], [2, 0]], dtype=np.int32)
    config = {
        "qdte": {
            "candidate_backend": "jax_repair",
            "score_backend": "sparse_delta_gpu",
            "total_candidates_per_iter": 8,
            "num_active_targets": 1,
            "random_candidate_fraction": 0.0,
            "gpu_source_draws": 8,
            "gpu_batches_per_iter": 1,
            "gpu_sparse_query_block_size": 4,
            "gpu_sparse_changed_attr_capacity": 2,
        }
    }
    inv_var = np.ones(qcat.m, dtype=np.float32)

    enter_batch = generate_and_score_candidates_gpu(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        np.asarray([5.0], dtype=np.float32),
        inv_var,
        config,
        np.random.default_rng(0),
    )
    enter_directed = enter_batch.candidates.target_query_ids == 0
    assert np.any(enter_directed)
    assert np.all(qcat.eval_query_np(enter_batch.candidates.new_rows[enter_directed], 0))

    exit_batch = generate_and_score_candidates_gpu(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        np.asarray([-5.0], dtype=np.float32),
        inv_var,
        config,
        np.random.default_rng(1),
    )
    exit_directed = exit_batch.candidates.target_query_ids == 0
    assert np.any(exit_directed)
    assert not np.any(qcat.eval_query_np(exit_batch.candidates.new_rows[exit_directed], 0))


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
