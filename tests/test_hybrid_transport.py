from __future__ import annotations

import numpy as np

from qdte.evolution.hybrid_candidates import (
    OPERATOR_SWAP,
    HybridCandidateUnitBatch,
    prepare_hybrid_score_context,
    score_candidate_units,
    score_candidate_units_l1,
)
from qdte.evolution.hybrid_transport import apply_candidate_units, select_nonconflicting_candidate_units
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import OP_EQ, QueryBuilder


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


def _two_nonconflicting_swaps(X: np.ndarray) -> HybridCandidateUnitBatch:
    old = np.asarray([[X[0], X[1]], [X[2], X[3]]], dtype=np.int32)
    new = np.asarray(
        [
            [[1, 1], [0, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int32,
    )
    return HybridCandidateUnitBatch(
        row_ids=np.asarray([[0, 1], [2, 3]], dtype=np.int32),
        old_rows=old,
        new_rows=new,
        row_mask=np.ones((2, 2), dtype=bool),
        operator_ids=np.full(2, OPERATOR_SWAP, dtype=np.int8),
        target_query_ids=np.full(2, -1, dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
    )


def test_atomic_unit_transport_matches_full_batch_loss_reduction() -> None:
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int32)
    candidates = _two_nonconflicting_swaps(X)
    candidates.validate()
    residual = np.asarray([0.0, 0.0, 3.0], dtype=np.float32)
    inv = np.ones(3, dtype=np.float32)
    scores = score_candidate_units(candidates, residual, inv, prepare_hybrid_score_context(qcat))

    result = select_nonconflicting_candidate_units(
        candidates,
        scores,
        residual,
        inv,
        qcat,
        max_accept=2,
    )
    precomputed = select_nonconflicting_candidate_units(
        candidates,
        scores,
        residual,
        inv,
        qcat,
        max_accept=2,
        precomputed_deltas=np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.int8),
    )
    before = 0.5 * float(np.sum(residual * residual * inv))
    apply_candidate_units(X, candidates, result.accepted_indices)
    after_residual = residual - result.delta_sum
    after = 0.5 * float(np.sum(after_residual * after_residual * inv))

    assert result.accepted_indices.tolist() == [0, 1]
    assert np.array_equal(precomputed.accepted_indices, result.accepted_indices)
    assert np.array_equal(precomputed.deltas, result.deltas)
    assert np.allclose(result.objective_advantages, np.asarray([2.5, 1.5]))
    assert np.isclose(before - after, np.sum(result.objective_advantages))
    assert answer_queries(X, qcat).tolist() == [2.0, 2.0, 2.0]


def test_atomic_unit_transport_matches_full_batch_l1_reduction() -> None:
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int32)
    candidates = _two_nonconflicting_swaps(X)
    residual = np.asarray([0.0, 0.0, 3.0], dtype=np.float32)
    weights = np.asarray([0.5, 0.5, 2.0], dtype=np.float32)
    scores = score_candidate_units_l1(
        candidates,
        residual,
        weights,
        prepare_hybrid_score_context(qcat),
    )

    result = select_nonconflicting_candidate_units(
        candidates,
        scores,
        residual,
        np.ones(3, dtype=np.float32),
        qcat,
        max_accept=2,
        objective="l1",
        objective_weights=weights,
    )
    before = float(np.abs(residual) @ weights)
    after_residual = residual - result.delta_sum
    after = float(np.abs(after_residual) @ weights)

    assert result.accepted_indices.tolist() == [0, 1]
    assert np.allclose(result.objective_advantages, np.asarray([2.0, 2.0]))
    assert np.isclose(before - after, np.sum(result.objective_advantages))


def test_atomic_unit_transport_honors_accept_limit() -> None:
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int32)
    candidates = _two_nonconflicting_swaps(X)
    residual = np.asarray([0.0, 0.0, 3.0], dtype=np.float32)
    inv = np.ones(3, dtype=np.float32)
    scores = score_candidate_units(candidates, residual, inv, prepare_hybrid_score_context(qcat))

    result = select_nonconflicting_candidate_units(
        candidates,
        scores,
        residual,
        inv,
        qcat,
        max_accept=1,
    )

    assert result.count == 1


def test_noise_guard_requires_advantage_above_directional_noise_std() -> None:
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int32)
    candidates = _two_nonconflicting_swaps(X)
    residual = np.asarray([0.0, 0.0, 3.0], dtype=np.float32)
    inv = np.ones(3, dtype=np.float32)
    scores = score_candidate_units(candidates, residual, inv, prepare_hybrid_score_context(qcat))

    accepted = select_nonconflicting_candidate_units(
        candidates,
        scores,
        residual,
        inv,
        qcat,
        max_accept=2,
        noise_guard_kappa=2.0,
    )
    rejected = select_nonconflicting_candidate_units(
        candidates,
        scores,
        residual,
        inv,
        qcat,
        max_accept=2,
        noise_guard_kappa=3.0,
    )

    assert accepted.count == 1
    assert accepted.noise_standard_deviations.tolist() == [1.0]
    assert rejected.count == 0
    assert rejected.num_noise_guard_rejections == 2


def test_noise_guard_accepts_explicit_objective_advantage_noise_variance() -> None:
    qcat = _queries()
    X = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int32)
    candidates = _two_nonconflicting_swaps(X)
    residual = np.asarray([0.0, 0.0, 3.0], dtype=np.float32)
    inv = np.ones(3, dtype=np.float32)
    scores = score_candidate_units(candidates, residual, inv, prepare_hybrid_score_context(qcat))

    result = select_nonconflicting_candidate_units(
        candidates,
        scores,
        residual,
        inv,
        qcat,
        max_accept=2,
        noise_guard_kappa=2.0,
        advantage_noise_variance=np.full(3, 4.0, dtype=np.float32),
    )

    assert result.count == 0
    assert result.num_noise_guard_rejections == 2
