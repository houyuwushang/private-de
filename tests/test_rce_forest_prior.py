from __future__ import annotations

import itertools

import numpy as np

from qdte.evolution.entropy import AtomEntropyState
from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    compile_hierarchical_pair_strategy,
)
from qdte.measurement.support_coarsening import (
    AttributeSupportMap,
    RELEASED_SUPPORT_METHOD,
    SupportCoarsening,
)
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.rce.forest_prior import (
    ReleasedConfidenceForestPrior,
    build_released_confidence_forest_prior,
    solve_ccf_pair,
)


def _catalogue(cardinalities: tuple[int, ...]):
    builder = QueryBuilder(max_terms=2)
    for attr, cardinality in enumerate(cardinalities):
        for value in range(cardinality):
            builder.add(
                [(attr, OP_EQ, value, value, value)],
                name=f"attr{attr}={value}",
                group=f"oneway:{attr}",
                family="oneway",
            )
    for left, right in itertools.combinations(range(len(cardinalities)), 2):
        for left_value, right_value in itertools.product(
            range(cardinalities[left]), range(cardinalities[right])
        ):
            builder.add(
                [
                    (left, OP_EQ, left_value, left_value, left_value),
                    (right, OP_EQ, right_value, right_value, right_value),
                ],
                name=f"attr{left}={left_value}&attr{right}={right_value}",
                group=f"twoway:{left}:{right}",
                family="twoway",
            )
    return builder.build()


def _released_target(qcat, cardinalities: tuple[int, ...], public_total: int) -> np.ndarray:
    target = np.zeros(qcat.m, dtype=np.float64)
    for qid, group in enumerate(qcat.groups):
        if group.startswith("oneway:"):
            attr = int(group.split(":")[1])
            target[qid] = public_total / cardinalities[attr]
        else:
            _, left, right = group.split(":")
            target[qid] = public_total / (
                cardinalities[int(left)] * cardinalities[int(right)]
            )
    return target


def _transcript(
    cardinalities: tuple[int, ...],
    public_total: int,
    interaction_rates: dict[tuple[int, int], np.ndarray],
    variance_rate: float = 1.0e-3,
) -> HierarchicalInteractionTranscript:
    pairs = tuple(itertools.combinations(range(len(cardinalities)), 2))
    strategy = compile_hierarchical_pair_strategy(cardinalities, pairs)
    noisy_components: dict[str, np.ndarray] = {}
    component_variances: dict[str, float] = {}
    rho_by_block: dict[str, float] = {}
    for block in strategy.blocks:
        if block.kind == "oneway_contrast":
            noisy_components[block.name] = np.zeros(block.coefficient_shape)
        else:
            pair = tuple(block.scope)
            noisy_components[block.name] = (
                np.asarray(interaction_rates.get(pair, np.zeros(block.coefficient_shape)))
                * public_total
            )
        variance = variance_rate * public_total**2
        component_variances[block.name] = float(variance)
        rho_by_block[block.name] = 1.0
    return HierarchicalInteractionTranscript(
        strategy=strategy,
        public_total=public_total,
        noisy_components=noisy_components,
        component_variances=component_variances,
        rho_by_block=rho_by_block,
        rho_total=float(len(strategy.blocks)),
        rho_spent=float(len(strategy.blocks)),
        allocation_mode="public_optimal",
    )


def test_pair_program_certifies_product_correlated_and_inconsistent_cases() -> None:
    marginal = np.asarray([0.5, 0.5])

    product = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[0.0]]),
        1.0e-2,
        confidence_radius=5.0,
    )
    assert product.status == "product_in_confidence"
    assert product.kl_upper == 0.0
    assert product.weight_lower_bound == 0.0

    correlated = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[0.3]]),
        1.0e-3,
        confidence_radius=3.84,
    )
    assert correlated.status == "eligible"
    assert correlated.kl_upper > 0.0
    assert correlated.weight_lower_bound > 0.0
    assert correlated.kl_gap < 1.0e-8
    assert correlated.ellipsoid_violation <= 1.0e-8

    inconsistent = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[1.0]]),
        1.0e-3,
        confidence_radius=3.84,
    )
    assert inconsistent.status == "anchor_interaction_inconsistent"
    assert inconsistent.pair_table is None


def test_pair_program_accepts_full_interaction_covariance() -> None:
    marginal = np.asarray([1.0 / 3.0] * 3)
    result = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[0.20, 0.02], [0.01, 0.10]]),
        np.asarray(
            [
                [1.0e-3, 2.0e-4, 0.0, 0.0],
                [2.0e-4, 1.5e-3, 1.0e-4, 0.0],
                [0.0, 1.0e-4, 1.2e-3, 1.0e-4],
                [0.0, 0.0, 1.0e-4, 1.1e-3],
            ]
        ),
        confidence_radius=9.5,
    )

    assert result.rank == 4
    assert result.status in {"eligible", "product_in_confidence"}
    assert result.ellipsoid_violation <= 1.0e-8


def test_pair_program_log_barrier_fallback_preserves_strict_certificate() -> None:
    marginal = np.asarray([1.0 / 3.0] * 3)
    result = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[0.20, 0.02], [0.01, 0.10]]),
        np.asarray(
            [
                [1.0e-3, 2.0e-4, 0.0, 0.0],
                [2.0e-4, 1.5e-3, 1.0e-4, 0.0],
                [0.0, 1.0e-4, 1.2e-3, 1.0e-4],
                [0.0, 0.0, 1.0e-4, 1.1e-3],
            ]
        ),
        confidence_radius=9.5,
        max_iterations=1,
    )

    assert result.status == "eligible"
    assert result.kl_solver_success is True
    assert "log_barrier_candidate_certified" in result.kl_solver_message
    assert result.kl_gap <= 1.0e-8
    assert result.stationarity_residual <= 1.0e-7
    assert result.complementarity_residual <= 1.0e-7
    assert result.ellipsoid_violation <= 1.0e-8


def test_pair_program_supports_exact_zero_variance_interaction() -> None:
    marginal = np.asarray([0.5, 0.5])
    exact = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[0.2]]),
        0.0,
        confidence_radius=0.0,
    )
    inconsistent = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[1.0]]),
        0.0,
        confidence_radius=0.0,
    )

    assert exact.status == "eligible"
    assert exact.rank == 0
    assert exact.pair_table is not None
    assert exact.weight_lower_bound > 0.0
    assert inconsistent.status == "anchor_interaction_inconsistent"
    assert inconsistent.pair_table is None


def test_pair_program_certifies_tiny_positive_kl_edge() -> None:
    marginal = np.asarray([0.5, 0.5])
    result = solve_ccf_pair(
        (0, 1),
        marginal,
        marginal,
        np.asarray([[0.063]]),
        1.0e-3,
        confidence_radius=3.84,
    )

    assert result.status == "eligible"
    assert 0.0 < result.kl_upper < 1.0e-5
    assert result.kl_gap <= 1.0e-10
    assert result.kl_solver_success is True
    assert result.stationarity_residual <= 1.0e-7


def test_pair_program_certifies_transport_boundary_stationarity() -> None:
    left = np.asarray(
        [
            0.11304673930079288,
            0.12514931635722842,
            0.1492668332889846,
            0.029418128077299783,
            0.04086199336050802,
            0.06916547288521072,
            0.11744626827469137,
            0.1406026690291286,
            0.02580930635258175,
            0.11371711678204949,
            0.023812073992563687,
            0.044798213194561134,
            0.006905869104399544,
        ]
    )
    right = np.asarray([0.9525117836294693, 0.047488216370530655])
    released = np.asarray(
        [
            0.010174019857441395,
            0.01780116984957087,
            0.050210951095623915,
            0.03245441348924157,
            0.019696656042058246,
            0.0023443540501176447,
            -0.015405430475694848,
            0.0441045301903284,
            -0.03256945604540949,
            0.02501740847383542,
            0.016515704191852807,
            0.05306147605119033,
        ]
    ).reshape(-1, 1)
    variance = np.asarray(
        [2.869722380859777e-5] * 11 + [5.329484421596729e-5]
    ).reshape(-1, 1)

    result = solve_ccf_pair(
        (6, 11),
        left,
        right,
        released,
        variance,
        confidence_radius=34.954475055225785,
    )

    assert result.status == "eligible"
    assert result.kl_solver_success is True
    assert result.kl_solver_message.startswith("transport_gap_certified_after_")
    assert result.pair_table is not None
    assert float(np.min(result.pair_table)) < 1.0e-12
    assert result.kl_gap <= 1.0e-10
    assert result.stationarity_residual <= 1.0e-7


def test_empty_forest_is_exact_product_prior() -> None:
    cards = (2, 2, 2)
    public_total = 100
    qcat = _catalogue(cards)
    prior = build_released_confidence_forest_prior(
        qcat,
        _released_target(qcat, cards, public_total),
        cards,
        _transcript(cards, public_total, {}),
        public_total=public_total,
    )
    rows = np.asarray(list(itertools.product(range(2), repeat=3)), dtype=np.int32)

    assert isinstance(prior, ReleasedConfidenceForestPrior)
    assert prior.edges == ()
    np.testing.assert_allclose(
        prior.log_probability_rows(rows),
        prior.product_prior.log_probability_rows(rows),
        rtol=0.0,
        atol=0.0,
    )


def test_deterministic_forest_is_normalized_and_acyclic() -> None:
    cards = (2, 2, 2)
    public_total = 100
    qcat = _catalogue(cards)
    interactions = {
        (0, 1): np.asarray([[0.3]]),
        (0, 2): np.asarray([[0.3]]),
        (1, 2): np.asarray([[0.3]]),
    }
    prior = build_released_confidence_forest_prior(
        qcat,
        _released_target(qcat, cards, public_total),
        cards,
        _transcript(cards, public_total, interactions),
        public_total=public_total,
    )
    rows = np.asarray(list(itertools.product(range(2), repeat=3)), dtype=np.int32)
    probabilities = np.exp(prior.log_probability_rows(rows))

    assert [edge.pair for edge in prior.edges] == [(0, 1), (0, 2)]
    assert np.isclose(float(np.sum(probabilities)), 1.0, rtol=1.0e-12, atol=1.0e-12)
    assert np.all(probabilities > 0.0)
    assert prior.diagnostics()["selected_edge_count"] == 2
    assert prior.diagnostics()["eligible_edge_count"] == 3


def test_forest_prior_preserves_exact_empirical_kl_edit_gain() -> None:
    cards = (2, 2)
    public_total = 100
    qcat = _catalogue(cards)
    prior = build_released_confidence_forest_prior(
        qcat,
        _released_target(qcat, cards, public_total),
        cards,
        _transcript(cards, public_total, {(0, 1): np.asarray([[0.3]])}),
        public_total=public_total,
    )
    rows = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int32)
    state = AtomEntropyState.from_rows(rows, prior)
    old = rows[[1]]
    new = np.asarray([[0, 0]], dtype=np.int32)
    gain = float(state.candidate_gains(old, new)[0])
    edited = rows.copy()
    edited[1] = new[0]

    assert np.isclose(
        gain,
        state.regularizer - state.recompute_regularizer(edited),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_coarsened_ccf_lifts_selected_edge_to_original_domain() -> None:
    cards = (3, 3)
    public_total = 100
    qcat = _catalogue(cards)
    support = SupportCoarsening(
        attributes=(
            AttributeSupportMap(
                attribute=0,
                original_cardinality=3,
                retained_categories=(0,),
                rare_categories=(1, 2),
                original_to_coarse=np.asarray([0, 1, 1]),
            ),
            AttributeSupportMap(
                attribute=1,
                original_cardinality=3,
                retained_categories=(0,),
                rare_categories=(1, 2),
                original_to_coarse=np.asarray([0, 1, 1]),
            ),
        ),
        method=RELEASED_SUPPORT_METHOD,
        kappa=3.0,
    )
    prior = build_released_confidence_forest_prior(
        qcat,
        _released_target(qcat, cards, public_total),
        cards,
        _transcript(
            cards,
            public_total,
            {(0, 1): np.asarray([[0.35, 0.0], [0.0, 0.0]])},
            variance_rate=1.0e-3,
        ),
        public_total=public_total,
        support_coarsening=support,
    )

    assert prior.diagnostics()["support_coarsening"]["method"] == (
        RELEASED_SUPPORT_METHOD
    )
    assert len(prior.edges) == 1
    edge = prior.edges[0]
    assert edge.table.shape == (3, 3)
    np.testing.assert_allclose(
        np.sum(edge.unsmoothed_table, axis=1),
        prior.probabilities[0],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.sum(edge.unsmoothed_table, axis=0),
        prior.probabilities[1],
        atol=1.0e-10,
    )
