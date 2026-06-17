from __future__ import annotations

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.scoring import compute_deltas, score_candidates_sparse
from qdte.evolution.transport import choose_atom_flow_batch_transport
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryBuilder


def _kway_catalogue():
    builder = QueryBuilder(max_terms=4)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_RANGE, 1, 1, 3)], "b[1,3]", "range:1", "range")
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_RANGE, 1, 1, 3), (2, OP_GE, 1, 1, 3)],
        "a=1&b[1,3]&c>=1",
        "threeway",
        "mixed",
    )
    builder.add(
        [
            (0, OP_EQ, 1, 1, 1),
            (1, OP_RANGE, 1, 1, 3),
            (2, OP_GE, 1, 1, 3),
            (3, OP_LE, 0, 0, 0),
        ],
        "a=1&b[1,3]&c>=1&d<=0",
        "fourway",
        "mixed",
    )
    builder.add_halfspace([(0, 1.0), (3, 1.0)], threshold=1.0, name="a+d<=1", group="halfspace")
    return builder.build()


def test_query_delta_index_matches_dense_for_kway_range_and_halfspace() -> None:
    qcat = _kway_catalogue()
    index = QueryDeltaIndex.build(qcat, num_attrs=4)
    old_rows = np.asarray(
        [
            [0, 2, 1, 0],
            [1, 2, 1, 1],
            [1, 0, 1, 0],
            [1, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    new_rows = np.asarray(
        [
            [1, 2, 1, 0],
            [1, 2, 1, 0],
            [1, 2, 1, 0],
            [1, 2, 1, 1],
        ],
        dtype=np.int32,
    )

    sparse = index.dense_candidate_deltas(old_rows, new_rows)
    dense = compute_deltas(old_rows, new_rows, qcat)

    assert sparse.dtype == np.int8
    assert np.array_equal(sparse, dense)
    assert sparse.tolist() == [
        [1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, -1, -1],
    ]


def test_query_delta_index_only_checks_queries_touched_by_changed_attrs() -> None:
    qcat = _kway_catalogue()
    index = QueryDeltaIndex.build(qcat, num_attrs=4)

    affected = index.affected_query_ids(
        np.asarray([1, 2, 1, 0], dtype=np.int32),
        np.asarray([1, 0, 1, 0], dtype=np.int32),
    )

    assert affected.tolist() == [1, 2, 3]


def test_sparse_scoring_matches_dense_delta_formula() -> None:
    qcat = _kway_catalogue()
    index = QueryDeltaIndex.build(qcat, num_attrs=4)
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1, 2], dtype=np.int32),
        old_rows=np.asarray([[0, 2, 1, 0], [1, 2, 1, 1], [1, 0, 1, 0]], dtype=np.int32),
        new_rows=np.asarray([[1, 2, 1, 0], [1, 2, 1, 0], [1, 2, 1, 0]], dtype=np.int32),
        target_query_ids=np.asarray([0, 3, 1], dtype=np.int32),
        edit_cost=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        repair_type=np.ones(3, dtype=np.int32),
    )
    residual = np.asarray([2.0, 3.0, 5.0, 7.0, 11.0], dtype=np.float32)
    inv_variance = np.asarray([1.0, 0.5, 2.0, 3.0, 4.0], dtype=np.float32)
    dense_delta = compute_deltas(candidates.old_rows, candidates.new_rows, qcat).astype(np.float32)
    expected = (
        dense_delta @ (residual * inv_variance)
        - 0.5 * ((dense_delta * dense_delta) @ inv_variance)
        - 0.25 * candidates.edit_cost
    )

    sparse_scores = score_candidates_sparse(
        candidates,
        residual,
        inv_variance,
        index,
        lambda_cost=0.25,
    )

    assert np.allclose(sparse_scores, expected)


def test_atom_flow_batch_can_use_sparse_delta_index() -> None:
    builder = QueryBuilder(max_terms=3)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1), (2, OP_EQ, 1, 1, 1)],
        "a=1&b=1&c=1",
        "threeway",
        "mixed",
    )
    qcat = builder.build()
    index = QueryDeltaIndex.build(qcat, num_attrs=3)
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.int32),
        new_rows=np.asarray([[1, 1, 1], [1, 1, 1]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([3.0, 3.0, 3.0], dtype=np.float32)
    inv_variance = np.ones(3, dtype=np.float32)
    advantages = score_candidates_sparse(candidates, residual, inv_variance, index, lambda_cost=0.0)

    dense_result = choose_atom_flow_batch_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        prefix_strategy="best_advantage",
    )
    sparse_result = choose_atom_flow_batch_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        prefix_strategy="best_advantage",
        delta_index=index,
    )

    assert np.array_equal(sparse_result.accepted_indices, dense_result.accepted_indices)
    assert np.allclose(sparse_result.delta_sum, dense_result.delta_sum)
    assert sparse_result.diagnostics["atom_flow_sparse_delta"] == 1
