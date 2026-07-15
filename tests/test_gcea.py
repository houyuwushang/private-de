from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix

from qdte.measurement.factorization import (
    allocate_strategy_rho,
    compile_hierarchical_pair_strategy,
)
from qdte.measurement.gcea import (
    GCEAProfile,
    RISK_NAMES,
    _RiskEvaluator,
    build_gcea_profile,
    evaluate_gcea_risks,
    optimize_gcea_allocation,
)
from qdte.queries.types import OP_EQ, QueryBuilder


def _eq(attr: int, value: int) -> tuple[int, int, int, int, int]:
    return (attr, OP_EQ, value, 0, 0)


def _complete_pair_fixture() -> tuple:
    cards = (2, 3)
    builder = QueryBuilder(max_terms=2)
    groups: list[dict] = []
    for attr, cardinality in enumerate(cards):
        indices: list[int] = []
        for value in range(cardinality):
            assert builder.add(
                [_eq(attr, value)],
                name=f"oneway:{attr}:{value}",
                group=f"oneway:{attr}",
                family="oneway",
            )
            indices.append(sum(cards[:attr]) + value)
        groups.append(
            {
                "name": f"oneway:{attr}",
                "family": "oneway",
                "query_indices": indices,
                "is_partition": True,
            }
        )
    pair_indices: list[int] = []
    for left_value in range(cards[0]):
        for right_value in range(cards[1]):
            assert builder.add(
                [_eq(0, left_value), _eq(1, right_value)],
                name=f"twoway:{left_value}:{right_value}",
                group="twoway:0:1",
                family="twoway",
            )
            pair_indices.append(sum(cards) + len(pair_indices))
    groups.append(
        {
            "name": "twoway:0:1",
            "family": "twoway",
            "query_indices": pair_indices,
            "is_partition": True,
        }
    )
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1)])
    control = allocate_strategy_rho(strategy, 0.2, "public_optimal")
    return strategy, builder.build(), groups, control


def test_gcea_profile_has_exact_public_reconstruction_and_partitions() -> None:
    strategy, qcat, groups, control = _complete_pair_fixture()
    profile = build_gcea_profile(
        strategy,
        qcat,
        groups,
        public_total=1_000,
        control_rho_by_block=control,
    )

    assert profile.num_queries == qcat.m == 11
    assert profile.num_partitions == 3
    assert profile.unsupported_query_ids == ()
    assert profile.fixed_block_indices.size == 0
    assert profile.reconstruction_max_abs <= 1.0e-12
    assert profile.unit_variance.shape == (11, 3)
    assert profile.unit_variance.nnz > 0
    assert len(profile.profile_sha256()) == 64

    risks = evaluate_gcea_risks(profile, control)
    assert set(risks) == set(RISK_NAMES)
    assert all(math.isfinite(value) and value > 0.0 for value in risks.values())


def test_gcea_analytic_risk_gradients_match_central_differences() -> None:
    strategy, qcat, groups, control = _complete_pair_fixture()
    profile = build_gcea_profile(
        strategy,
        qcat,
        groups,
        public_total=1_000,
        control_rho_by_block=control,
    )
    evaluator = _RiskEvaluator(profile)
    shares = np.asarray([0.24, 0.45, 0.31], dtype=np.float64)
    values, gradients = evaluator.evaluate(shares)
    step = 1.0e-6
    numerical = np.empty_like(gradients)
    for column in range(shares.size):
        left = shares.copy()
        right = shares.copy()
        left[column] -= step
        right[column] += step
        numerical[:, column] = (
            evaluator.evaluate(right)[0] - evaluator.evaluate(left)[0]
        ) / (2.0 * step)
    np.testing.assert_allclose(values, evaluator.evaluate(shares)[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(gradients, numerical, rtol=2.0e-5, atol=1.0e-10)


def test_all_gcea_risks_are_convex_on_positive_allocation_simplex() -> None:
    strategy, qcat, groups, control = _complete_pair_fixture()
    profile = build_gcea_profile(
        strategy,
        qcat,
        groups,
        public_total=1_000,
        control_rho_by_block=control,
    )
    evaluator = _RiskEvaluator(profile)
    left = np.asarray([0.20, 0.55, 0.25])
    right = np.asarray([0.36, 0.34, 0.30])
    weight = 0.37
    middle = weight * left + (1.0 - weight) * right
    left_values, _ = evaluator.evaluate(left)
    right_values, _ = evaluator.evaluate(right)
    middle_values, _ = evaluator.evaluate(middle)
    np.testing.assert_array_less(
        middle_values,
        weight * left_values + (1.0 - weight) * right_values + 1.0e-12,
    )


def _asymmetric_profile() -> GCEAProfile:
    strategy = compile_hierarchical_pair_strategy((2, 2), [])
    return GCEAProfile(
        strategy=strategy,
        public_total=1_000,
        beta_tail=0.05,
        query_ids=np.asarray([0, 1], dtype=np.int64),
        unit_variance=csr_matrix(np.asarray([[9.0, 0.0], [0.0, 1.0]])),
        fixed_variance=np.zeros(2, dtype=np.float64),
        partition_names=("pair",),
        partition_rows=(np.asarray([0, 1], dtype=np.int64),),
        movable_block_indices=np.asarray([0, 1], dtype=np.int64),
        fixed_block_indices=np.asarray([], dtype=np.int64),
        control_rho=np.asarray([0.5, 0.5], dtype=np.float64),
        reconstruction_max_abs=0.0,
        unsupported_query_ids=(),
    )


def test_gcea_optimizer_improves_minimax_risk_without_tail_regression() -> None:
    profile = _asymmetric_profile()
    first = optimize_gcea_allocation(profile)
    second = optimize_gcea_allocation(profile)

    assert first.t_star < 0.97
    assert first.t_final <= first.t_star + 1.1e-10
    assert max(first.primary_ratios.values()) < 1.0
    assert max(first.tail_ratios.values()) <= 1.0 + 1.0e-12
    assert first.kkt_gap <= 1.0e-8
    assert first.sum_rho_relative_error <= 1.0e-15
    assert first.min_movable_share > 1.0e-6
    assert first.stage_one["success"] is True
    assert first.stage_two["success"] is True
    assert first.profile_sha256 == second.profile_sha256
    for name in first.rho_by_block:
        assert math.isclose(
            first.rho_by_block[name],
            second.rho_by_block[name],
            rel_tol=0.0,
            abs_tol=1.0e-13,
        )
