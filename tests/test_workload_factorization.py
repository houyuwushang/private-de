import math

import numpy as np

from qdte.measurement.factorization import compile_hierarchical_pair_strategy
from qdte.measurement.workload_factorization import (
    allocation_risk,
    factorize_public_query,
    factorize_public_workload,
    optimal_rho_for_importance,
    public_query_indicator,
)
from qdte.queries.orthogonal import reconstruct_oneway, reconstruct_pair
from qdte.queries.types import OP_EQ, OP_RANGE, QueryBuilder


def _eq(attr: int, value: int) -> tuple[int, int, int, int, int]:
    return (attr, OP_EQ, value, 0, 0)


def test_public_query_factorization_reconstructs_ordinary_and_linear_queries() -> None:
    cards = (3, 2)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1)])
    builder = QueryBuilder(max_terms=2)
    assert builder.add(
        [(0, OP_RANGE, 0, 1, 2), _eq(1, 1)],
        name="range_pair",
        group="mixed:0:1",
        family="mixed",
    )
    assert builder.add_halfspace(
        [(0, 1.0), (1, 2.0)],
        threshold=2.0,
        name="linear_pair",
        group="linear:0:1",
    )
    qcat = builder.build()

    total = 37.0
    theta_left = np.asarray([1.3, -0.8])
    theta_right = np.asarray([0.7])
    theta_pair = np.asarray([[0.2], [-0.4]])
    table = reconstruct_pair(
        total,
        theta_left,
        theta_right,
        theta_pair,
        cards[0],
        cards[1],
    )
    components = {
        "oneway_contrast:0": theta_left,
        "oneway_contrast:1": theta_right,
        "pair_interaction:0:1": theta_pair,
    }

    for qid in range(qcat.m):
        _, indicator = public_query_indicator(qcat, qid, cards)
        factored = factorize_public_query(strategy, qcat, qid)
        assert factored is not None
        predicted = total * factored.constant_per_row
        predicted += sum(
            float(np.dot(coefficients.reshape(-1), components[name].reshape(-1)))
            for name, coefficients in factored.coefficients.items()
        )
        np.testing.assert_allclose(
            predicted,
            float(np.sum(indicator * table)),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_cell_workload_importance_matches_existing_strategy_formula() -> None:
    cards = (3, 2, 4)
    pairs = [(0, 1), (0, 2), (1, 2)]
    strategy = compile_hierarchical_pair_strategy(cards, pairs)
    builder = QueryBuilder(max_terms=2)
    for attr, cardinality in enumerate(cards):
        for value in range(cardinality):
            assert builder.add(
                [_eq(attr, value)],
                name=f"oneway:{attr}:{value}",
                group=f"oneway:{attr}",
                family="oneway",
            )
    for left, right in pairs:
        for left_value in range(cards[left]):
            for right_value in range(cards[right]):
                assert builder.add(
                    [_eq(left, left_value), _eq(right, right_value)],
                    name=f"twoway:{left}:{right}:{left_value}:{right_value}",
                    group=f"twoway:{left}:{right}",
                    family="twoway",
                )
    factorization = factorize_public_workload(strategy, builder.build())
    assert not factorization.unsupported_query_ids
    for block in strategy.blocks:
        np.testing.assert_allclose(
            factorization.importance_by_block[block.name],
            block.public_importance,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_factorization_records_unsupported_high_order_queries() -> None:
    cards = (2, 2, 2)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1), (0, 2), (1, 2)])
    builder = QueryBuilder(max_terms=3)
    assert builder.add(
        [_eq(0, 1), _eq(1, 1), _eq(2, 1)],
        name="threeway",
        group="threeway",
        family="threeway",
    )
    result = factorize_public_workload(strategy, builder.build())
    assert result.supported_query_ids == ()
    assert result.unsupported_query_ids == (0,)
    assert result.unsupported_scope_orders == {3: 1}


def test_importance_optimal_allocation_reduces_declared_risk() -> None:
    strategy = compile_hierarchical_pair_strategy((2, 3), [(0, 1)])
    importance = {
        "oneway_contrast:0": 1.0,
        "oneway_contrast:1": 7.0,
        "pair_interaction:0:1": 19.0,
    }
    equal = {block.name: 0.3 / len(strategy.blocks) for block in strategy.blocks}
    optimal = optimal_rho_for_importance(strategy, importance, rho_total=0.3)
    assert math.isclose(sum(optimal.values()), 0.3, rel_tol=0.0, abs_tol=1.0e-15)
    assert allocation_risk(strategy, importance, optimal) < allocation_risk(
        strategy,
        importance,
        equal,
    )
