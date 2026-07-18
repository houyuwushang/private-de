from __future__ import annotations

import copy

import numpy as np
import pytest

from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.measurement.adaptive_interactions import (
    assemble_adaptive_interaction_transcript,
    combine_coverage_refinements,
    measure_interaction_action,
)
from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import interaction_transcript_to_diagonal_measurements
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import ColumnSchema, TableSchema


def _problem():
    cards = np.asarray([2, 3, 4], dtype=np.int32)
    schema = TableSchema(
        columns=[
            ColumnSchema(name=f"x{attr}", kind="categorical", cardinality=int(cardinality))
            for attr, cardinality in enumerate(cards)
        ]
    )
    rng = np.random.default_rng(20260714)
    rows = np.column_stack(
        [rng.integers(0, cardinality, size=120) for cardinality in cards]
    ).astype(np.int32)
    anchors = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(cards, ()),
        public_total=len(rows),
        rho_total=0.2,
        rng=np.random.default_rng(11),
        allocation_mode="public_optimal",
    )
    return cards, schema, rows, anchors


def test_oneway_only_strategy_round_trips_and_builds_exact_precision() -> None:
    _, schema, _, anchors = _problem()
    transcript = assemble_adaptive_interaction_transcript(
        anchors,
        (),
        rho_total=1.0,
    )
    restored = HierarchicalInteractionTranscript.from_public_dict(transcript.to_public_dict())
    qcat, groups = build_selected_pair_partition_workload(schema, ())
    measurements = interaction_transcript_to_diagonal_measurements(
        restored,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    precision = OrthogonalInteractionPrecision(qcat, groups, restored)

    assert restored.strategy.pairs == ()
    assert restored.allocation_mode == "adaptive_distinct"
    assert qcat.m == sum(int(value) for value in schema.cardinalities)
    assert measurements.target_projected.shape == (qcat.m,)
    assert precision.diagnostics()["num_pair_blocks"] == 0
    assert precision.diagnostics()["num_oneway_blocks"] == schema.d


def test_selected_pair_is_measured_once_and_partial_workload_has_no_fake_blocks() -> None:
    cards, schema, rows, anchors = _problem()
    action = measure_interaction_action(
        rows,
        (2, 0),
        cards,
        rho=0.3,
        rng=np.random.default_rng(17),
    )
    transcript = assemble_adaptive_interaction_transcript(
        anchors,
        [action],
        rho_total=1.0,
    )
    qcat, groups = build_selected_pair_partition_workload(schema, transcript.strategy.pairs)
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    precision = OrthogonalInteractionPrecision(qcat, groups, transcript)

    assert transcript.strategy.pairs == ((0, 2),)
    assert np.isclose(transcript.rho_spent, 0.5)
    assert np.isclose(transcript.rho_total, 1.0)
    assert qcat.m == int(np.sum(cards)) + int(cards[0] * cards[2])
    assert [group.name for group in groups if group.family == "twoway"] == ["twoway:0:2"]
    assert len(measurements.groups) == schema.d + 1
    assert precision.diagnostics()["num_pair_blocks"] == 1
    assert precision.diagnostics()["effective_rank"] == sum(cards - 1) + (cards[0] - 1) * (cards[2] - 1)


def test_action_variance_matches_sensitivity_and_rho() -> None:
    cards, _, rows, _ = _problem()
    action = measure_interaction_action(
        rows,
        (0, 1),
        cards,
        rho=0.25,
        rng=np.random.default_rng(7),
    )

    assert action.pair == (0, 1)
    assert action.noisy_coefficients.shape == (1, 2)
    assert np.isclose(
        action.coefficient_variance,
        action.sensitivity_l2**2 / (2.0 * action.rho),
    )
    assert action.to_public_dict()["pair"] == [0, 1]


def test_adaptive_transcript_rejects_duplicate_or_tampered_actions() -> None:
    cards, _, rows, anchors = _problem()
    action = measure_interaction_action(
        rows,
        (0, 1),
        cards,
        rho=0.25,
        rng=np.random.default_rng(7),
    )
    with pytest.raises(ValueError, match="duplicate"):
        assemble_adaptive_interaction_transcript(
            anchors,
            [action, action],
            rho_total=1.0,
        )

    tampered = copy.copy(action)
    object.__setattr__(tampered, "coefficient_variance", action.coefficient_variance * 2.0)
    with pytest.raises(ValueError, match="variance/rho"):
        assemble_adaptive_interaction_transcript(
            anchors,
            [tampered],
            rho_total=1.0,
        )


def test_selected_partition_builder_rejects_invalid_pairs() -> None:
    _, schema, _, _ = _problem()
    with pytest.raises(ValueError):
        build_selected_pair_partition_workload(schema, [(0, 0)])
    with pytest.raises(ValueError):
        build_selected_pair_partition_workload(schema, [(0, schema.d)])


def test_coverage_refinements_precision_combine_repeated_pair_observations() -> None:
    cards, _, rows, _ = _problem()
    base = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(cards, ((0, 1), (0, 2))),
        public_total=len(rows),
        rho_total=0.4,
        rng=np.random.default_rng(31),
        allocation_mode="public_optimal",
    )
    refinements = [
        measure_interaction_action(
            rows,
            (0, 1),
            cards,
            rho=rho,
            rng=np.random.default_rng(seed),
        )
        for rho, seed in ((0.1, 41), (0.2, 42))
    ]

    combined = combine_coverage_refinements(
        base,
        refinements,
        rho_total=1.0,
    )
    name = "pair_interaction:0:1"
    observations = [base.noisy_components[name]] + [
        action.noisy_coefficients for action in refinements
    ]
    variances = [base.component_variances[name]] + [
        action.coefficient_variance for action in refinements
    ]
    precision = np.sum([1.0 / value for value in variances])
    expected = sum(
        observation / variance
        for observation, variance in zip(observations, variances, strict=True)
    ) / precision

    assert combined.strategy.pairs == base.strategy.pairs
    assert combined.allocation_mode == "coverage_refined"
    assert np.isclose(combined.rho_spent, 0.7)
    assert np.isclose(
        combined.rho_by_block[name],
        base.rho_by_block[name] + 0.3,
    )
    assert np.isclose(combined.component_variances[name], 1.0 / precision)
    np.testing.assert_allclose(combined.noisy_components[name], expected)
    untouched = "pair_interaction:0:2"
    np.testing.assert_array_equal(
        combined.noisy_components[untouched], base.noisy_components[untouched]
    )
    assert combined.refinement_diagnostics is not None
    assert combined.refinement_diagnostics["observation_counts"][name] == 3
    assert combined.refinement_diagnostics["observation_counts"][untouched] == 1

    restored = HierarchicalInteractionTranscript.from_public_dict(
        combined.to_public_dict()
    )
    np.testing.assert_allclose(restored.noisy_components[name], expected)
    assert restored.refinement_diagnostics == combined.refinement_diagnostics


def test_coverage_refinement_rejects_unsupported_or_overspending_actions() -> None:
    cards, _, rows, _ = _problem()
    base = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(cards, ((0, 1),)),
        public_total=len(rows),
        rho_total=0.4,
        rng=np.random.default_rng(51),
    )
    unsupported = measure_interaction_action(
        rows,
        (1, 2),
        cards,
        rho=0.1,
        rng=np.random.default_rng(52),
    )
    with pytest.raises(ValueError, match="outside the broad base"):
        combine_coverage_refinements(base, [unsupported], rho_total=1.0)

    expensive = measure_interaction_action(
        rows,
        (0, 1),
        cards,
        rho=0.7,
        rng=np.random.default_rng(53),
    )
    with pytest.raises(RuntimeError, match="overspends"):
        combine_coverage_refinements(base, [expensive], rho_total=1.0)
