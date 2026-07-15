from __future__ import annotations

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.precision import DiagonalPrecision
from qdte.evolution.transport import (
    batch_advantage,
    batch_advantage_l1,
    choose_atom_flow_batch_transport,
    choose_atom_flow_transport,
    choose_blind_transport,
    choose_constructive_pair_transport,
    choose_directed_group_transport,
    choose_directed_group_transport_jax,
    choose_random_group_transport,
    choose_transport_batch,
    choose_transport_batch_jax,
    select_nonconflicting_in_order,
)
from qdte.queries.types import OP_EQ, QueryBuilder


def test_transport_delta_sum_matches_returned_indices_after_sorting() -> None:
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1, 2], dtype=np.int32),
        old_rows=np.zeros((3, 1), dtype=np.int32),
        new_rows=np.ones((3, 1), dtype=np.int32),
        target_query_ids=np.zeros(3, dtype=np.int32),
        edit_cost=np.zeros(3, dtype=np.float32),
        repair_type=np.ones(3, dtype=np.int32),
    )
    selected = np.asarray([2, 0, 1], dtype=np.int32)
    deltas = np.asarray(
        [
            [0, 0, 1],  # candidate 2
            [1, 0, 0],  # candidate 0
            [0, 1, 0],  # candidate 1
        ],
        dtype=np.int8,
    )
    advantages = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    result = choose_transport_batch(
        candidates,
        advantages,
        deltas,
        selected,
        residual=np.ones(3, dtype=np.float32) * 10.0,
        inv_variance=np.ones(3, dtype=np.float32),
        lambda_cost=0.0,
    )
    expected = np.asarray([1, 1, 1], dtype=np.float32)
    assert result.accepted_indices.tolist() == [0, 2, 1]
    assert np.allclose(result.delta_sum, expected)


def test_batch_advantage_uses_float64_accumulation() -> None:
    rng = np.random.default_rng(7)
    residual = rng.normal(0.0, 1.0e5, size=30_000).astype(np.float32)
    inv_variance = np.exp(rng.normal(0.0, 2.0, size=30_000)).astype(np.float32)
    delta_sum = rng.integers(-8, 9, size=30_000).astype(np.float32)
    expected = float(
        delta_sum.astype(np.float64)
        @ (residual.astype(np.float64) * inv_variance.astype(np.float64))
        - 0.5 * ((delta_sum.astype(np.float64) ** 2) @ inv_variance.astype(np.float64))
        - 0.25 * 17.0
    )

    observed = batch_advantage(
        residual,
        inv_variance,
        delta_sum,
        cost_sum=17.0,
        lambda_cost=0.25,
    )

    assert observed == expected


def test_batch_advantage_matches_realized_loss_with_overlapping_edit_effects() -> None:
    residual = np.asarray([4.0, -3.0, 2.0], dtype=np.float32)
    inv_variance = np.asarray([3.0, 1.0e8, 1.0e-6], dtype=np.float32)
    edit_deltas = np.asarray([[1.0, 1.0, 0.0], [1.0, -1.0, 1.0]], dtype=np.float32)
    delta_sum = edit_deltas.sum(axis=0)
    lambda_cost = 0.25
    cost_sum = 3.0

    loss_before = 0.5 * float(
        residual.astype(np.float64) ** 2 @ inv_variance.astype(np.float64)
    )
    residual_after = residual.astype(np.float64) - delta_sum.astype(np.float64)
    loss_after = 0.5 * float(residual_after**2 @ inv_variance.astype(np.float64))
    expected = loss_before - loss_after - lambda_cost * cost_sum
    observed = batch_advantage(
        residual,
        inv_variance,
        delta_sum,
        cost_sum=cost_sum,
        lambda_cost=lambda_cost,
    )

    individual_sum = sum(
        batch_advantage(
            residual,
            inv_variance,
            delta,
            cost_sum=cost_sum / len(edit_deltas),
            lambda_cost=lambda_cost,
        )
        for delta in edit_deltas
    )
    assert np.isclose(observed, expected, rtol=1.0e-12, atol=1.0e-6)
    assert not np.isclose(individual_sum, observed, rtol=1.0e-6, atol=1.0e-6)


def test_jax_best_advantage_rejects_all_negative_prefixes_like_cpu() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0], dtype=np.int32),
        old_rows=np.asarray([[0]], dtype=np.int32),
        new_rows=np.asarray([[1]], dtype=np.int32),
        target_query_ids=np.asarray([0], dtype=np.int32),
        edit_cost=np.zeros(1, dtype=np.float32),
        repair_type=np.ones(1, dtype=np.int32),
    )
    residual = np.asarray([-0.1], dtype=np.float32)
    inv_variance = np.ones(1, dtype=np.float32)
    advantages = np.asarray([-0.6], dtype=np.float32)
    selected = np.asarray([0], dtype=np.int32)
    deltas = np.asarray([[1]], dtype=np.int8)

    cpu = choose_transport_batch(
        candidates,
        advantages,
        deltas,
        selected,
        residual,
        inv_variance,
        lambda_cost=0.0,
        prefix_strategy="best_advantage",
    )
    jax_result = choose_transport_batch_jax(
        candidates,
        advantages,
        selected,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        prefix_strategy="best_advantage",
    )

    assert cpu.accepted_indices.size == 0
    assert jax_result.accepted_indices.size == 0
    assert jax_result.batch_advantage < 0.0


def test_blind_transport_accepts_nonconflicting_candidates_without_positive_advantage() -> None:
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1, 1, 2], dtype=np.int32),
        old_rows=np.zeros((4, 1), dtype=np.int32),
        new_rows=np.ones((4, 1), dtype=np.int32),
        target_query_ids=np.zeros(4, dtype=np.int32),
        edit_cost=np.zeros(4, dtype=np.float32),
        repair_type=np.ones(4, dtype=np.int32),
    )
    selected = select_nonconflicting_in_order(
        candidates,
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        max_accept=4,
    )
    deltas = np.asarray(
        [
            [-1],
            [-1],
            [-1],
        ],
        dtype=np.int8,
    )

    result = choose_blind_transport(
        candidates,
        deltas,
        selected,
        residual=np.asarray([10.0], dtype=np.float32),
        inv_variance=np.ones(1, dtype=np.float32),
        lambda_cost=0.0,
    )

    assert selected.tolist() == [0, 1, 3]
    assert result.accepted_indices.tolist() == [0, 1, 3]
    assert np.allclose(result.delta_sum, np.asarray([-3.0], dtype=np.float32))
    assert result.batch_advantage < 0.0
    assert result.diagnostics["blind_accept_mode"] == 1


def test_constructive_pair_transport_scores_explicit_attached_partner_pair() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        new_rows=np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.asarray([1, 16], dtype=np.int32),
        attached_pair_indices=np.asarray([[0, 1]], dtype=np.int32),
    )
    residual = np.asarray([2.0, 1.0], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([0.0, 0.5], dtype=np.float32)

    result = choose_constructive_pair_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        pool_multiplier=0,
        partner_limit=0,
        harm_query_limit=0,
        prefix_strategy="best_advantage",
    )

    assert set(result.accepted_indices.tolist()) == {0, 1}
    assert np.allclose(result.delta_sum, np.asarray([1.0, 0.0], dtype=np.float32))
    assert result.batch_advantage > 0.0
    assert result.diagnostics["constructive_pair_explicit_pairs_evaluated"] == 1
    assert result.diagnostics["constructive_pair_explicit_positive_pairs"] == 1
    assert result.diagnostics["constructive_pair_explicit_pair_units"] == 1


def test_atom_flow_avoids_overshooting_duplicate_atom_edge() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1, 2], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 0], [0, 0]], dtype=np.int32),
        new_rows=np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.int32),
        target_query_ids=np.asarray([0, 0, 1], dtype=np.int32),
        edit_cost=np.zeros(3, dtype=np.float32),
        repair_type=np.ones(3, dtype=np.int32),
    )
    residual = np.asarray([1.2, 1.2], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([0.7, 0.7, 0.7], dtype=np.float32)

    result = choose_atom_flow_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=3,
        min_advantage=0.0,
    )

    assert len(result.accepted_indices) == 2
    assert set(result.accepted_indices.tolist()) in ({0, 2}, {1, 2})
    assert np.allclose(result.delta_sum, np.asarray([1.0, 1.0], dtype=np.float32))
    assert result.batch_advantage > batch_advantage(
        residual,
        inv_variance,
        np.asarray([2.0, 1.0], dtype=np.float32),
        cost_sum=0.0,
        lambda_cost=0.0,
    )
    assert result.diagnostics["atom_flow_edges"] == 2
    assert result.diagnostics["atom_flow_augments"] == 2


def test_atom_flow_respects_row_capacity_across_edges() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 0], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 0]], dtype=np.int32),
        new_rows=np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([5.0, 5.0], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([4.5, 4.5], dtype=np.float32)

    result = choose_atom_flow_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
    )

    assert len(result.accepted_indices) == 1
    assert len(set(candidates.row_ids[result.accepted_indices].tolist())) == 1


def test_atom_flow_batch_uses_single_residual_update_with_edge_capacity() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1, 2], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 0], [0, 0]], dtype=np.int32),
        new_rows=np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.int32),
        target_query_ids=np.asarray([0, 0, 1], dtype=np.int32),
        edit_cost=np.zeros(3, dtype=np.float32),
        repair_type=np.ones(3, dtype=np.int32),
    )
    residual = np.asarray([1.2, 1.2], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([0.7, 0.7, 0.7], dtype=np.float32)

    result = choose_atom_flow_batch_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=3,
        min_advantage=0.0,
        prefix_strategy="best_advantage",
    )

    assert len(result.accepted_indices) == 2
    assert set(result.accepted_indices.tolist()) in ({0, 2}, {1, 2})
    assert np.allclose(result.delta_sum, np.asarray([1.0, 1.0], dtype=np.float32))
    assert result.batch_advantage > 0.0
    assert result.diagnostics["atom_flow_batch_mode"] == 1
    assert result.diagnostics["atom_flow_exact_mode"] == 0
    assert result.diagnostics["atom_flow_selected_candidates"] == 2
    assert result.diagnostics["atom_flow_prefix_candidates"] == 2


def test_atom_flow_batch_l1_prefix_uses_l1_objective() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0], [0]], dtype=np.int32),
        new_rows=np.asarray([[1], [1]], dtype=np.int32),
        target_query_ids=np.zeros(2, dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([0.75], dtype=np.float32)
    weights = np.asarray([1.0], dtype=np.float32)
    advantages = np.asarray([0.5, 0.5], dtype=np.float32)

    result = choose_atom_flow_batch_transport(
        candidates,
        advantages,
        residual,
        inv_variance=np.ones(1, dtype=np.float32),
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        pool_multiplier=0,
        prefix_strategy="best_advantage",
        objective="l1",
        objective_weights=weights,
    )

    expected_adv = batch_advantage_l1(residual, weights, result.delta_sum, 0.0, 0.0)
    assert len(result.accepted_indices) == 1
    assert np.allclose(result.batch_advantage, expected_adv)


def test_atom_flow_batch_custom_prefix_uses_exact_callback() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 0]], dtype=np.int32),
        new_rows=np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([2.0, 2.0], dtype=np.float32)
    advantages = np.asarray([1.5, 1.5], dtype=np.float32)
    seen: dict[str, np.ndarray] = {}

    def evaluate_prefixes(
        selected_indices: np.ndarray,
        delta_prefix: np.ndarray,
        cost_prefix: np.ndarray,
    ) -> np.ndarray:
        seen["indices"] = selected_indices.copy()
        seen["delta"] = delta_prefix.copy()
        seen["cost"] = cost_prefix.copy()
        return np.asarray([3.0, -1.0], dtype=np.float64)

    result = choose_atom_flow_batch_transport(
        candidates,
        advantages,
        residual,
        inv_variance=np.ones(2, dtype=np.float32),
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        pool_multiplier=0,
        prefix_strategy="best_advantage",
        precision_operator=DiagonalPrecision(np.ones(2, dtype=np.float32)),
        prefix_advantage_evaluator=evaluate_prefixes,
    )

    assert len(result.accepted_indices) == 1
    assert result.batch_advantage == 3.0
    assert result.diagnostics["atom_flow_custom_prefix"] == 1
    assert seen["indices"].shape == (2,)
    assert seen["delta"].shape == (2, 2)
    assert np.array_equal(seen["cost"], np.zeros(2))


def test_constructive_pair_accepts_compensating_negative_single_edits() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        new_rows=np.asarray([[1, 1], [0, 0]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([0.8, -0.2], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([-0.4, -0.3], dtype=np.float32)

    result = choose_constructive_pair_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        pool_multiplier=0,
        partner_limit=8,
        prefix_strategy="best_advantage",
    )

    assert result.accepted_indices.tolist() == [0, 1]
    assert np.allclose(result.delta_sum, np.asarray([1.0, 0.0], dtype=np.float32))
    assert result.batch_advantage > 0.0
    assert result.diagnostics["constructive_pair_positive_pairs"] == 1
    assert result.diagnostics["constructive_pair_pair_units"] == 1
    assert result.diagnostics["constructive_pair_accepted_candidates"] == 2


def test_random_group_transport_scores_aggregate_group_advantage() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        new_rows=np.asarray([[1, 1], [0, 0]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([0.8, -0.2], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([-0.4, -0.3], dtype=np.float32)

    result = choose_random_group_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        rng=np.random.default_rng(0),
        group_count=8,
        min_group_size=2,
        max_group_size=2,
    )

    assert result.accepted_indices.tolist() == [0, 1]
    assert np.allclose(result.delta_sum, np.asarray([1.0, 0.0], dtype=np.float32))
    assert result.batch_advantage > 0.0
    assert result.diagnostics["random_group_positive_groups"] == 8
    assert result.diagnostics["random_group_groups_with_negative_member"] == 8


def test_directed_group_transport_builds_compensating_group() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        new_rows=np.asarray([[1, 1], [0, 0]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([0.8, -0.2], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([-0.4, -0.3], dtype=np.float32)

    result = choose_directed_group_transport(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        seed_count=2,
        min_group_size=2,
        max_group_size=2,
    )

    assert set(result.accepted_indices.tolist()) == {0, 1}
    assert np.allclose(result.delta_sum, np.asarray([1.0, 0.0], dtype=np.float32))
    assert result.batch_advantage > 0.0
    assert result.diagnostics["directed_group_positive_groups"] >= 1
    assert result.diagnostics["directed_group_accepted_candidates"] == 2


def test_directed_group_transport_jax_builds_compensating_group() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway:1", "oneway")
    qcat = builder.build()
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1], dtype=np.int32),
        old_rows=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        new_rows=np.asarray([[1, 1], [0, 0]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1], dtype=np.int32),
        edit_cost=np.zeros(2, dtype=np.float32),
        repair_type=np.ones(2, dtype=np.int32),
    )
    residual = np.asarray([0.8, -0.2], dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)
    advantages = np.asarray([-0.4, -0.3], dtype=np.float32)

    result = choose_directed_group_transport_jax(
        candidates,
        advantages,
        residual,
        inv_variance,
        lambda_cost=0.0,
        qcat=qcat,
        max_accept=2,
        min_advantage=0.0,
        seed_count=2,
        min_group_size=2,
        max_group_size=2,
    )

    assert set(result.accepted_indices.tolist()) == {0, 1}
    assert np.allclose(result.delta_sum, np.asarray([1.0, 0.0], dtype=np.float32))
    assert result.batch_advantage > 0.0
    assert result.diagnostics["directed_group_jax_mode"] == 1
    assert result.diagnostics["directed_group_positive_groups"] >= 1
    assert result.diagnostics["directed_group_accepted_candidates"] == 2
