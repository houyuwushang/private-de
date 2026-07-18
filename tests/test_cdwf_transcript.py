from __future__ import annotations

import copy

import numpy as np
import pytest

from qdte.measurement.cdwf import (
    derive_cdwf_budget_plan,
    uniform_cdwf_allocation,
)
from qdte.measurement.cdwf_transcript import (
    CDWFSequentialTranscript,
    ccf_pair_streams_for_history,
    cdwf_transcript_to_measurements,
    combine_cdwf_released_history,
    current_cdwf_rho_by_block,
    measure_cdwf_base,
    measure_cdwf_refinement_stream,
)
from qdte.measurement.factorization import compile_hierarchical_pair_strategy
from qdte.measurement.measure import measurements_from_public_dict
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import ColumnSchema, TableSchema


def _problem() -> tuple[
    np.ndarray,
    TableSchema,
    CDWFSequentialTranscript,
]:
    cards = np.asarray([2, 3, 2], dtype=np.int32)
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name=f"x{attribute}",
                kind="categorical",
                cardinality=int(cardinality),
            )
            for attribute, cardinality in enumerate(cards)
        ]
    )
    rows = np.column_stack(
        [
            np.random.default_rng(100 + attribute).integers(
                0,
                int(cardinality),
                size=120,
            )
            for attribute, cardinality in enumerate(cards)
        ]
    ).astype(np.int32)
    strategy = compile_hierarchical_pair_strategy(
        cards,
        ((0, 1), (0, 2), (1, 2)),
    )
    control = {
        block.name: 0.08 + 0.01 * index
        for index, block in enumerate(strategy.blocks)
    }
    plan = derive_cdwf_budget_plan(strategy, control, epsilon=0.1)
    base = measure_cdwf_base(
        rows,
        strategy,
        plan,
        public_total=len(rows),
        rng=np.random.default_rng(2026071801),
    )
    streams = [base]
    allocation = uniform_cdwf_allocation(plan)
    for round_index in range(plan.rounds):
        streams.append(
            measure_cdwf_refinement_stream(
                rows,
                strategy,
                allocation,
                round_index=round_index,
                rng_by_block={
                    name: np.random.default_rng(
                        2026071810 + 100 * round_index + index
                    )
                    for index, name in enumerate(plan.eligible_interaction_blocks)
                },
            )
        )
    transcript = CDWFSequentialTranscript(
        strategy=strategy,
        public_total=len(rows),
        budget=plan,
        streams=tuple(streams),
    )
    return rows, schema, transcript


def test_cdwf_transcript_preserves_frozen_rho_and_closes_ledger() -> None:
    _, _, transcript = _problem()
    final = transcript.final_rho_by_block()

    for name in transcript.budget.frozen_blocks:
        assert final[name] == pytest.approx(
            transcript.budget.control_rho_by_block[name]
        )
    for name in transcript.budget.eligible_interaction_blocks:
        assert final[name] == pytest.approx(
            transcript.budget.control_rho_by_block[name]
        )
    assert sum(final.values()) == pytest.approx(transcript.budget.rho_total)

    ledger = transcript.privacy_ledger(delta=1.0e-9)
    assert ledger["adjacency"] == "add_remove"
    assert ledger["selection_rho"] == 0.0
    assert ledger["rho_spent"] == pytest.approx(transcript.budget.rho_total)
    assert all(
        entry["public_metadata"]["selection_rho"] == 0.0
        for entry in ledger["entries"]
    )
    assert all(
        entry["public_metadata"]["adjacency"] == "add_remove"
        for entry in ledger["entries"]
    )


def test_cdwf_combination_is_exact_and_streamwise_confidence_is_separate() -> None:
    _, _, transcript = _problem()
    combined = transcript.combined_transcript()
    block_name = transcript.budget.eligible_interaction_blocks[0]
    observations = [
        stream.by_name()[block_name]
        for stream in transcript.streams
        if block_name in stream.by_name()
    ]
    precision = sum(1.0 / observation.coefficient_variance for observation in observations)
    expected = sum(
        observation.noisy_coefficients / observation.coefficient_variance
        for observation in observations
    ) / precision

    np.testing.assert_allclose(combined.noisy_components[block_name], expected)
    assert combined.component_variances[block_name] == pytest.approx(1.0 / precision)
    confidence = transcript.streamwise_confidence()
    assert len(confidence.streams) == transcript.budget.rounds + 1
    assert confidence.confidence_level == pytest.approx(0.95)
    assert confidence.streams[0].name == "base"
    assert confidence.streams[1].coefficient_indices.size < confidence.canonical_dimension


def test_cdwf_partial_history_reconstructs_only_released_rounds() -> None:
    _, _, transcript = _problem()
    partial = transcript.streams[:2]
    combined = combine_cdwf_released_history(
        transcript.strategy,
        transcript.public_total,
        transcript.budget,
        partial,
    )
    current = current_cdwf_rho_by_block(transcript.budget, partial)
    pair_streams = ccf_pair_streams_for_history(
        transcript.strategy,
        transcript.public_total,
        transcript.budget,
        partial,
    )

    assert combined.rho_spent == pytest.approx(
        sum(stream.rho_spent for stream in partial)
    )
    assert sum(current.values()) == pytest.approx(combined.rho_spent)
    assert all(len(streams) in {1, 2} for streams in pair_streams.values())
    assert all(
        stream.covariance_storage == "isotropic_scalar"
        and stream.interaction_covariance_rate.ndim == 0
        for streams in pair_streams.values()
        for stream in streams
    )


def test_cdwf_public_artifact_round_trip_preserves_all_streams() -> None:
    _, schema, transcript = _problem()
    qcat, groups = build_selected_pair_partition_workload(
        schema,
        transcript.strategy.pairs,
    )
    measurements = cdwf_transcript_to_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )
    restored_measurements = measurements_from_public_dict(
        measurements.to_public_dict()
    )
    restored = CDWFSequentialTranscript.from_public_dict(
        restored_measurements.sequential_transcript
    )

    assert restored.to_public_dict() == transcript.to_public_dict()
    assert restored_measurements.privacy_ledger == measurements.privacy_ledger
    restored_combined = restored.combined_transcript()
    original_combined = transcript.combined_transcript()
    np.testing.assert_allclose(
        np.concatenate(
            [
                restored_combined.noisy_components[block.name].reshape(-1)
                for block in restored.strategy.blocks
            ]
        ),
        np.concatenate(
            [
                original_combined.noisy_components[block.name].reshape(-1)
                for block in transcript.strategy.blocks
            ]
        ),
    )


def test_cdwf_transcript_rejects_a_changed_frozen_observation() -> None:
    _, _, transcript = _problem()
    tampered = copy.deepcopy(transcript.to_public_dict())
    frozen_name = transcript.budget.frozen_blocks[0]
    observation = next(
        item
        for item in tampered["streams"][0]["observations"]
        if item["block_name"] == frozen_name
    )
    observation["rho"] *= 0.9
    observation["coefficient_variance"] /= 0.9
    tampered["streams"][0]["rho_spent"] = sum(
        item["rho"] for item in tampered["streams"][0]["observations"]
    )

    with pytest.raises(ValueError, match="base observation mismatch"):
        CDWFSequentialTranscript.from_public_dict(tampered)
