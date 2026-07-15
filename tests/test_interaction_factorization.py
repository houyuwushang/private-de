from __future__ import annotations

import itertools
import math

import numpy as np

from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    allocate_public_precision,
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.queries.orthogonal import interaction_coefficients, oneway_contrast_from_counts


def _full_factorial(cards: tuple[int, ...]) -> np.ndarray:
    return np.asarray(list(itertools.product(*(range(cardinality) for cardinality in cards))), dtype=np.int32)


def test_public_precision_allocation_matches_closed_form_and_budget() -> None:
    sensitivity = np.asarray([0.5, 1.0, 2.0])
    importance = np.asarray([1.0, 4.0, 9.0])
    allocation = allocate_public_precision(sensitivity, importance, rho_total=0.7)
    expected_weights = sensitivity * np.sqrt(importance)

    np.testing.assert_allclose(allocation / allocation.sum(), expected_weights / expected_weights.sum())
    assert math.isclose(float(allocation.sum()), 0.7, rel_tol=0.0, abs_tol=1.0e-15)


def test_strategy_importance_accounts_for_reused_oneway_contrasts() -> None:
    strategy = compile_hierarchical_pair_strategy((2, 3, 2), [(0, 1), (0, 2), (1, 2)])

    # Attribute 0 contributes to its own one-way table and to pair tables with widths 3 and 2.
    assert math.isclose(strategy.block("oneway_contrast:0").public_importance, 1.0 + 1.0 / 3.0 + 1.0 / 2.0)
    assert strategy.block("pair_interaction:0:1").coefficient_shape == (1, 2)
    assert math.isclose(strategy.block("pair_interaction:0:1").sensitivity_l2, math.sqrt(1.0 / 3.0))


def test_zero_noise_components_reconstruct_exact_shared_marginals() -> None:
    cards = (2, 3, 2)
    rows = np.repeat(_full_factorial(cards), repeats=np.arange(1, 13), axis=0)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1), (0, 2), (1, 2)])
    transcript = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=1.0,
        rng=np.random.default_rng(1),
    )

    # Replace only the released values with exact components to isolate the reconstruction identity.
    for attr, cardinality in enumerate(cards):
        counts = np.bincount(rows[:, attr], minlength=cardinality).astype(np.float64)
        transcript.noisy_components[f"oneway_contrast:{attr}"] = oneway_contrast_from_counts(counts)
    for left, right in strategy.pairs:
        transcript.noisy_components[f"pair_interaction:{left}:{right}"] = interaction_coefficients(
            rows, (left, right), cards
        )

    reconstructed = transcript.reconstruct()
    assert reconstructed.max_consistency_violation() < 1.0e-10
    for attr, cardinality in enumerate(cards):
        expected = np.bincount(rows[:, attr], minlength=cardinality)
        np.testing.assert_allclose(reconstructed.oneway[attr], expected, atol=1.0e-10)
    for left, right in strategy.pairs:
        flat = rows[:, left] * cards[right] + rows[:, right]
        expected = np.bincount(flat, minlength=cards[left] * cards[right]).reshape(cards[left], cards[right])
        np.testing.assert_allclose(reconstructed.pairs[(left, right)], expected, atol=1.0e-10)


def test_transcript_has_exact_rho_ledger_and_correlated_pair_covariance() -> None:
    cards = (2, 2)
    rows = _full_factorial(cards)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1)])
    transcript = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=0.25,
        rng=np.random.default_rng(4),
        allocation_mode="public_optimal",
    )

    assert transcript.adjacency == "add_remove"
    assert math.isclose(transcript.rho_spent, 0.25, abs_tol=1.0e-15)
    covariance = transcript.pair_covariance((0, 1))
    np.testing.assert_allclose(covariance, covariance.T, atol=1.0e-12)
    assert np.any(np.abs(covariance - np.diag(np.diag(covariance))) > 1.0e-12)
    assert float(np.min(np.linalg.eigvalsh(covariance))) >= -1.0e-10


def test_public_transcript_round_trip_is_exact_and_validates_ledger() -> None:
    cards = (3, 2)
    rows = np.repeat(_full_factorial(cards), repeats=3, axis=0)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1)])
    transcript = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=0.7,
        rng=np.random.default_rng(8),
        allocation_mode="public_optimal",
    )

    restored = HierarchicalInteractionTranscript.from_public_dict(transcript.to_public_dict())

    assert restored.strategy.cardinalities == transcript.strategy.cardinalities
    assert restored.strategy.pairs == transcript.strategy.pairs
    assert restored.rho_by_block == transcript.rho_by_block
    assert restored.component_variances == transcript.component_variances
    for name, coefficients in transcript.noisy_components.items():
        np.testing.assert_array_equal(restored.noisy_components[name], coefficients)

    invalid = transcript.to_public_dict()
    invalid["blocks"][0]["coefficient_variance"] *= 2.0
    with np.testing.assert_raises_regex(ValueError, "variance/rho mismatch"):
        HierarchicalInteractionTranscript.from_public_dict(invalid)


def test_workload_rho_override_preserves_coupled_standard_normal_noise() -> None:
    cards = (3, 2)
    rows = np.repeat(_full_factorial(cards), repeats=4, axis=0)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1)])
    current = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=0.6,
        rng=np.random.default_rng(91),
        allocation_mode="public_optimal",
    )
    override = {
        "oneway_contrast:0": 0.3,
        "oneway_contrast:1": 0.2,
        "pair_interaction:0:1": 0.1,
    }
    workload = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=0.6,
        rng=np.random.default_rng(91),
        allocation_mode="workload_optimal",
        rho_by_block_override=override,
    )

    exact = {
        "oneway_contrast:0": oneway_contrast_from_counts(
            np.bincount(rows[:, 0], minlength=cards[0])
        ),
        "oneway_contrast:1": oneway_contrast_from_counts(
            np.bincount(rows[:, 1], minlength=cards[1])
        ),
        "pair_interaction:0:1": interaction_coefficients(rows, (0, 1), cards),
    }
    for block in strategy.blocks:
        current_z = (
            current.noisy_components[block.name] - exact[block.name]
        ) / math.sqrt(current.component_variances[block.name])
        workload_z = (
            workload.noisy_components[block.name] - exact[block.name]
        ) / math.sqrt(workload.component_variances[block.name])
        np.testing.assert_allclose(current_z, workload_z, rtol=1.0e-13, atol=1.0e-13)

    assert workload.allocation_mode == "workload_optimal"
    assert workload.rho_by_block == override
    restored = HierarchicalInteractionTranscript.from_public_dict(workload.to_public_dict())
    assert restored.allocation_mode == "workload_optimal"
    assert restored.rho_by_block == override


def test_rho_override_fails_closed_on_mode_keys_and_budget() -> None:
    cards = (2, 2)
    rows = _full_factorial(cards)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1)])
    valid = {
        "oneway_contrast:0": 0.1,
        "oneway_contrast:1": 0.1,
        "pair_interaction:0:1": 0.1,
    }
    with np.testing.assert_raises_regex(ValueError, "allocation_mode='workload_optimal'"):
        measure_hierarchical_pair_interactions(
            rows,
            strategy,
            public_total=len(rows),
            rho_total=0.3,
            rng=np.random.default_rng(1),
            allocation_mode="public_optimal",
            rho_by_block_override=valid,
        )
    with np.testing.assert_raises_regex(ValueError, "cover every strategy block"):
        measure_hierarchical_pair_interactions(
            rows,
            strategy,
            public_total=len(rows),
            rho_total=0.3,
            rng=np.random.default_rng(1),
            allocation_mode="workload_optimal",
            rho_by_block_override={"oneway_contrast:0": 0.3},
        )
    invalid_budget = dict(valid)
    invalid_budget["pair_interaction:0:1"] = 0.2
    with np.testing.assert_raises_regex(ValueError, "rho override spends"):
        measure_hierarchical_pair_interactions(
            rows,
            strategy,
            public_total=len(rows),
            rho_total=0.3,
            rng=np.random.default_rng(1),
            allocation_mode="workload_optimal",
            rho_by_block_override=invalid_budget,
        )
