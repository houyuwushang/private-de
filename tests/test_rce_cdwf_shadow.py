from __future__ import annotations

import itertools

import numpy as np

from qdte.measurement.cdwf import derive_cdwf_budget_plan
from qdte.measurement.cdwf_transcript import (
    CDWFSequentialTranscript,
    cdwf_transcript_to_measurements,
    measure_cdwf_base,
)
from qdte.measurement.factorization import compile_hierarchical_pair_strategy
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.rce.cdwf_shadow import (
    CDWFShadowDictionary,
    sample_released_forest_prior,
    solve_cdwf_shadow_rce,
)
from qdte.rce.sequential_forest_prior import (
    build_sequential_confidence_forest_prior,
)
from qdte.schema import ColumnSchema, TableSchema


def _base_problem():
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2),
            ColumnSchema(name="b", kind="categorical", cardinality=2),
            ColumnSchema(name="c", kind="categorical", cardinality=2),
        ]
    )
    rows = np.asarray(list(itertools.product(range(2), repeat=3)) * 20, dtype=np.int32)
    strategy = compile_hierarchical_pair_strategy(
        schema.cardinalities,
        ((0, 1), (0, 2), (1, 2)),
    )
    control = {block.name: 0.25 for block in strategy.blocks}
    plan = derive_cdwf_budget_plan(strategy, control, epsilon=0.1)
    base = measure_cdwf_base(
        rows,
        strategy,
        plan,
        public_total=len(rows),
        rng=np.random.default_rng(17),
    )
    # A partial history is sufficient for the round-0 shadow prior. The final
    # transcript type is used here only to reuse its public CCF stream builder;
    # zero-information test rounds preserve the base centers while spending
    # the declared refinement rho.
    from qdte.measurement.cdwf import uniform_cdwf_allocation
    from qdte.measurement.cdwf_transcript import measure_cdwf_refinement_stream

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
                    name: np.random.default_rng(2000 + 10 * round_index + index)
                    for index, name in enumerate(plan.eligible_interaction_blocks)
                },
            )
        )
    sequential = CDWFSequentialTranscript(
        strategy=strategy,
        public_total=len(rows),
        budget=plan,
        streams=tuple(streams),
    )
    qcat, groups = build_selected_pair_partition_workload(schema, strategy.pairs)
    measurements = cdwf_transcript_to_measurements(
        sequential,
        qcat,
        groups,
        delta=1.0e-9,
    )
    all_pair_streams = sequential.ccf_pair_streams()
    prior = build_sequential_confidence_forest_prior(
        qcat,
        measurements.target_projected,
        schema.cardinalities,
        {pair: (streams[0],) for pair, streams in all_pair_streams.items()},
        public_total=len(rows),
    )
    return rows, qcat, groups, sequential, measurements, prior


def test_shadow_dictionary_is_deterministic_and_released_prior_only() -> None:
    rows, _, _, _, _, prior = _base_problem()
    first = CDWFShadowDictionary.create(
        initial_rows=rows,
        product_prior=prior.product_prior,
        ccf_prior=prior,
        public_seed=77,
    )
    second = CDWFShadowDictionary.create(
        initial_rows=rows,
        product_prior=prior.product_prior,
        ccf_prior=prior,
        public_seed=77,
    )

    assert first.dictionary_hash == second.dictionary_hash
    assert first.to_public_dict()["global_pricing_certificate"] is False
    for left, right in zip(first.tables, second.tables, strict=True):
        np.testing.assert_array_equal(left, right)
    probabilities, support_rows, log_prior = first.component_distributions(prior)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert support_rows.shape[0] == log_prior.shape[0]
    assert np.all(np.isfinite(log_prior))


def test_forest_sampler_returns_schema_valid_rows() -> None:
    _, _, _, _, _, prior = _base_problem()
    sampled = sample_released_forest_prior(
        prior,
        5_000,
        np.random.default_rng(91),
    )

    assert sampled.shape == (5_000, prior.dimension)
    for attribute, cardinality in enumerate(prior.cardinalities):
        assert np.all((0 <= sampled[:, attribute]) & (sampled[:, attribute] < cardinality))
        empirical = np.bincount(
            sampled[:, attribute], minlength=cardinality
        ) / len(sampled)
        np.testing.assert_allclose(
            empirical,
            prior.probabilities[attribute],
            atol=0.035,
        )


def test_shadow_solve_exposes_certified_or_explicit_fallback_state() -> None:
    rows, qcat, groups, sequential, measurements, prior = _base_problem()
    dictionary = CDWFShadowDictionary.create(
        initial_rows=rows,
        product_prior=prior.product_prior,
        ccf_prior=prior,
        public_seed=123,
    )
    combined = sequential.combined_transcript()
    solved = solve_cdwf_shadow_rce(
        dictionary,
        qcat=qcat,
        workload_groups=groups,
        transcript=combined,
        released_target=measurements.target_projected,
        prior=prior,
        rho_by_block=sequential.final_rho_by_block(),
        max_iterations=2_000,
    )

    assert solved.result.support_kind == "convex_hull_of_declared_empirical_tables"
    assert solved.result.globally_certified is False
    assert solved.fallback_required == bool(solved.failure_reasons)
    if solved.fallback_required:
        assert solved.pressure is None or not solved.pressure.autodiff_audit_passed
    else:
        assert solved.pressure is not None
        assert solved.pressure.autodiff_audit_passed
