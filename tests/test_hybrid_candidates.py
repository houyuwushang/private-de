from __future__ import annotations

import numpy as np
import pytest

from qdte.evolution.hybrid_candidates import (
    OPERATOR_GLOBAL_DIRECTED_SWAP,
    OPERATOR_MUTATION,
    OPERATOR_SWAP,
    AdaptiveSwapAttributePolicy,
    HybridCandidateUnitBatch,
    allocate_operator_counts,
    apply_candidate_unit,
    candidate_unit_delta,
    candidate_unit_deltas,
    exact_unit_advantage,
    generate_global_directed_swap_units,
    generate_random_crossover_units,
    generate_directed_crossover_units,
    generate_directed_swap_units,
    generate_random_mutation_units,
    generate_random_swap_units,
    global_residual_swap_attr_probabilities,
    prepare_hybrid_score_context,
    prepare_hybrid_sparse_score_context,
    score_candidate_units,
    score_candidate_units_l1,
    score_candidate_units_optimized,
    score_one_attr_candidate_units_sparse,
)
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


def _schema() -> TableSchema:
    return TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2),
            ColumnSchema(name="b", kind="categorical", cardinality=2),
        ]
    )


def _queries():
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "a", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "b", "oneway")
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1)],
        "a=1,b=1",
        "a,b",
        "twoway",
    )
    return builder.build()


def _manual_unit(old_rows: np.ndarray, new_rows: np.ndarray) -> HybridCandidateUnitBatch:
    return HybridCandidateUnitBatch(
        row_ids=np.asarray([[0, 1]], dtype=np.int32),
        old_rows=np.asarray([old_rows], dtype=np.int32),
        new_rows=np.asarray([new_rows], dtype=np.int32),
        row_mask=np.asarray([[True, True]]),
        operator_ids=np.asarray([OPERATOR_SWAP], dtype=np.int8),
        target_query_ids=np.asarray([-1], dtype=np.int32),
        edit_cost=np.asarray([0.0], dtype=np.float32),
    )


def test_multirow_score_sums_delta_before_quadratic_term() -> None:
    qcat = _queries()
    candidates = _manual_unit(
        old_rows=np.asarray([[0, 0], [0, 0]], dtype=np.int32),
        new_rows=np.asarray([[1, 1], [1, 1]], dtype=np.int32),
    )
    candidates.validate()
    residual = np.asarray([0.0, 0.0, 3.0], dtype=np.float32)
    inv = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    scores = score_candidate_units(candidates, residual, inv, prepare_hybrid_score_context(qcat))
    delta = candidate_unit_delta(candidates, 0, qcat)

    assert delta.tolist() == [2.0, 2.0, 2.0]
    expected = exact_unit_advantage(delta, residual, inv)
    assert np.isclose(scores[0], expected)
    assert expected == 4.0


def test_multirow_l1_score_matches_exact_finite_difference() -> None:
    qcat = _queries()
    candidates = _manual_unit(
        old_rows=np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        new_rows=np.asarray([[1, 1], [0, 0]], dtype=np.int32),
    )
    residual = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
    weights = np.asarray([0.5, 0.5, 2.0], dtype=np.float32)

    scores = score_candidate_units_l1(
        candidates,
        residual,
        weights,
        prepare_hybrid_score_context(qcat),
    )
    delta = candidate_unit_delta(candidates, 0, qcat)
    expected = float(
        (np.abs(residual) - np.abs(residual - delta)) @ weights
    )

    assert scores.shape == (1,)
    assert np.isclose(scores[0], expected)
    assert expected == 2.0


def test_unit_validation_rejects_duplicate_active_rows() -> None:
    candidates = _manual_unit(
        old_rows=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        new_rows=np.asarray([[1, 0], [1, 1]], dtype=np.int32),
    )
    candidates.row_ids[0, 1] = candidates.row_ids[0, 0]

    with pytest.raises(ValueError, match="must be distinct"):
        candidates.validate()


def test_swap_score_matches_full_loss_recomputation_and_preserves_oneway() -> None:
    schema = _schema()
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1]], dtype=np.int32)
    candidates = _manual_unit(
        old_rows=X.copy(),
        new_rows=np.asarray([[1, 1], [0, 0]], dtype=np.int32),
    )
    residual = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    inv = np.ones(3, dtype=np.float32)
    before_hist = [np.bincount(X[:, attr], minlength=2) for attr in range(schema.d)]

    scores = score_candidate_units(candidates, residual, inv, prepare_hybrid_score_context(qcat))
    delta = candidate_unit_delta(candidates, 0, qcat)
    before_loss = 0.5 * float(residual @ residual)
    apply_candidate_unit(X, candidates, 0)
    after_residual = residual - delta
    after_loss = 0.5 * float(after_residual @ after_residual)

    assert np.isclose(scores[0], before_loss - after_loss)
    assert scores[0] == 0.5
    for attr in range(schema.d):
        assert np.array_equal(before_hist[attr], np.bincount(X[:, attr], minlength=2))


def test_random_generators_are_valid_and_deterministic() -> None:
    schema = _schema()
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)

    generators = (
        generate_random_mutation_units,
        generate_random_swap_units,
        generate_random_crossover_units,
    )
    for generator in generators:
        left = generator(X, schema, 32, np.random.default_rng(17))
        right = generator(X, schema, 32, np.random.default_rng(17))
        left.validate()
        assert left.count == 32
        assert np.array_equal(left.row_ids, right.row_ids)
        assert np.array_equal(left.old_rows, right.old_rows)
        assert np.array_equal(left.new_rows, right.new_rows)


def test_generated_swap_preserves_all_oneway_histograms() -> None:
    schema = _schema()
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    candidates = generate_random_swap_units(X, schema, 16, np.random.default_rng(4))

    for idx in range(candidates.count):
        old_rows = candidates.old_rows[idx]
        new_rows = candidates.new_rows[idx]
        for attr, card in enumerate(schema.cardinalities.tolist()):
            assert np.array_equal(
                np.bincount(old_rows[:, attr], minlength=card),
                np.bincount(new_rows[:, attr], minlength=card),
            )


def test_random_mutation_operator_and_full_loss_delta() -> None:
    schema = _schema()
    qcat = _queries()
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    candidates = generate_random_mutation_units(X, schema, 8, np.random.default_rng(8))
    target = answer_queries(X, qcat) + np.asarray([2.0, -1.0, 1.0], dtype=np.float32)
    answer = answer_queries(X, qcat)
    residual = target - answer
    inv = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    scores = score_candidate_units(candidates, residual, inv, prepare_hybrid_score_context(qcat))

    assert np.all(candidates.operator_ids == OPERATOR_MUTATION)
    for idx in range(candidates.count):
        delta = candidate_unit_delta(candidates, idx, qcat)
        expected = exact_unit_advantage(delta, residual, inv, edit_cost=float(candidates.edit_cost[idx]))
        assert np.isclose(scores[idx], expected)


def test_operator_count_allocation_is_exact_and_stable() -> None:
    counts = allocate_operator_counts(11, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    assert counts == {0: 3, 1: 3, 2: 3, 3: 2}
    assert sum(counts.values()) == 11


def test_directed_swap_units_follow_residual_direction() -> None:
    schema = _schema()
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1]], dtype=np.int32)
    residual = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    candidates = generate_directed_swap_units(
        X,
        qcat,
        schema,
        np.asarray([2], dtype=np.int32),
        residual,
        16,
        np.random.default_rng(3),
        inv_variance=np.ones(3, dtype=np.float32),
        over_sample_factor=4,
    )

    candidates.validate()
    assert candidates.count == 16
    for idx in range(candidates.count):
        qid = int(candidates.target_query_ids[idx])
        delta = candidate_unit_delta(candidates, idx, qcat)
        assert residual[qid] * delta[qid] > 0.0


def test_directed_crossover_units_follow_residual_direction() -> None:
    schema = _schema()
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.int32)
    residual = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    candidates = generate_directed_crossover_units(
        X,
        qcat,
        schema,
        np.asarray([2], dtype=np.int32),
        residual,
        16,
        np.random.default_rng(9),
        inv_variance=np.ones(3, dtype=np.float32),
        over_sample_factor=16,
    )

    candidates.validate()
    assert candidates.count == 16
    for idx in range(candidates.count):
        qid = int(candidates.target_query_ids[idx])
        delta = candidate_unit_delta(candidates, idx, qcat)
        assert residual[qid] * delta[qid] > 0.0


def test_global_swap_pressure_uses_multiquery_weighted_residuals() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2),
            ColumnSchema(name="b", kind="categorical", cardinality=2),
            ColumnSchema(name="c", kind="categorical", cardinality=2),
        ]
    )
    builder = QueryBuilder(max_terms=2)
    builder.add([(2, OP_EQ, 1, 1, 1)], "c=1", "c", "oneway")
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1)],
        "a=1,b=1",
        "a,b",
        "twoway",
    )
    qcat = builder.build()
    probabilities = global_residual_swap_attr_probabilities(
        qcat,
        schema,
        residual=np.asarray([100.0, 2.0], dtype=np.float32),
        inv_variance=np.ones(2, dtype=np.float32),
        exploration_floor=0.0,
    )

    assert np.allclose(probabilities, np.asarray([0.5, 0.5, 0.0]))


def test_global_directed_swap_is_deterministic_and_labeled() -> None:
    schema = _schema()
    qcat = _queries()
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    kwargs = dict(
        X_syn=X,
        qcat=qcat,
        schema=schema,
        residual=np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
        inv_variance=np.ones(3, dtype=np.float32),
        count=32,
    )
    left = generate_global_directed_swap_units(rng=np.random.default_rng(13), **kwargs)
    right = generate_global_directed_swap_units(rng=np.random.default_rng(13), **kwargs)

    left.validate()
    assert left.count == 32
    assert np.all(left.operator_ids == OPERATOR_GLOBAL_DIRECTED_SWAP)
    assert np.array_equal(left.row_ids, right.row_ids)
    assert np.array_equal(left.new_rows, right.new_rows)


def test_sparse_one_attr_unit_scores_match_dense_and_full_loss() -> None:
    schema = _schema()
    qcat = _queries()
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    candidates = generate_random_swap_units(X, schema, 32, np.random.default_rng(21))
    residual = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    inv = np.asarray([0.5, 2.0, 4.0], dtype=np.float32)
    dense_context = prepare_hybrid_score_context(qcat)
    sparse_context = prepare_hybrid_sparse_score_context(qcat, num_attrs=schema.d)

    dense = score_candidate_units(candidates, residual, inv, dense_context)
    sparse = score_one_attr_candidate_units_sparse(
        candidates,
        residual,
        inv,
        sparse_context,
        use_pmap=True,
    )
    optimized = score_candidate_units_optimized(
        candidates,
        residual,
        inv,
        dense_context,
        sparse_context,
        use_pmap=True,
    )

    assert np.allclose(sparse, dense, rtol=1.0e-6, atol=1.0e-6)
    assert np.allclose(optimized, dense, rtol=1.0e-6, atol=1.0e-6)
    batched_deltas = candidate_unit_deltas(candidates, dense_context)
    for idx in range(candidates.count):
        delta = candidate_unit_delta(candidates, idx, qcat)
        assert np.array_equal(batched_deltas[idx], delta)
        assert np.isclose(sparse[idx], exact_unit_advantage(delta, residual, inv))


def test_adaptive_swap_policy_learns_exact_advantage_yield_with_exploration() -> None:
    schema = _schema()
    candidates = HybridCandidateUnitBatch(
        row_ids=np.asarray([[0, 1], [2, 3]], dtype=np.int32),
        old_rows=np.asarray(
            [
                [[0, 0], [1, 1]],
                [[0, 0], [1, 1]],
            ],
            dtype=np.int32,
        ),
        new_rows=np.asarray(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ],
            dtype=np.int32,
        ),
        row_mask=np.ones((2, 2), dtype=bool),
        operator_ids=np.full(2, OPERATOR_GLOBAL_DIRECTED_SWAP, dtype=np.int8),
        target_query_ids=np.full(2, -1, dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
    )
    policy = AdaptiveSwapAttributePolicy.create(
        schema,
        exploration_floor=0.20,
        decay=0.0,
        prior_strength=0.0,
    )
    policy.update(candidates, np.asarray([10.0, -1.0], dtype=np.float32))
    probabilities = policy.probabilities()

    assert probabilities[0] > probabilities[1]
    assert probabilities[1] >= 0.10
    assert np.isclose(np.sum(probabilities), 1.0)
