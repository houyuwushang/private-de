from __future__ import annotations

import math

import numpy as np
import pytest

from qdte.evolution.entropy import ReleasedProductPrior
from qdte.measurement.factorization import compile_hierarchical_pair_strategy
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.forest_prior import (
    ConfidenceForestEdge,
    ReleasedConfidenceForestPrior,
)
from qdte.rce.public_domain import PublicLegalRowDomain
from qdte.rce.rhcg_ccmp import (
    CanonicalConfidenceGeometry,
    RHCGColumnSet,
    solve_enumerated_phase_one,
    solve_enumerated_phase_two,
    solve_phase_one_rhcg,
    solve_phase_two_rhcg,
    solve_rhcg_ccmp,
)
from qdte.rce.row_pricing import (
    ShadowFeatureMap,
    ccf_moment_answer,
    price_legal_row,
)
from qdte.schema import ColumnSchema, TableSchema


def _toy_feature_map(
    cardinalities: tuple[int, ...] = (2, 2, 2),
) -> ShadowFeatureMap:
    schema = TableSchema(
        [
            ColumnSchema(
                name=f"x{attribute}",
                kind="categorical",
                cardinality=cardinality,
                categories=[f"c{attribute}_{value}" for value in range(cardinality)],
            )
            for attribute, cardinality in enumerate(cardinalities)
        ]
    )
    domain = PublicLegalRowDomain.from_schema(schema, public_n=100)
    pairs = [
        (left, right)
        for left in range(len(cardinalities))
        for right in range(left + 1, len(cardinalities))
    ]
    return ShadowFeatureMap(
        compile_hierarchical_pair_strategy(cardinalities, pairs),
        domain,
    )


def _toy_geometry(feature_map: ShadowFeatureMap) -> CanonicalConfidenceGeometry:
    rows = feature_map.domain.enumerate_rows()
    probabilities = np.linspace(1.0, 2.0, len(rows), dtype=np.float64)
    probabilities /= float(np.sum(probabilities))
    truth = probabilities @ feature_map.row_atom_answers(rows)
    unit_covariance = feature_map.unit_rho_covariance_diagonal()
    rho_by_block = {
        name: 0.8 + 0.2 * index
        for index, name in enumerate(feature_map.block_slices)
    }
    variances = np.empty(feature_map.feature_dimension, dtype=np.float64)
    for name, block_slice in feature_map.block_slices.items():
        variances[block_slice] = unit_covariance[block_slice] / rho_by_block[name]
    noise = np.linspace(-0.15, 0.15, feature_map.feature_dimension)
    target = truth + noise * np.sqrt(variances)
    return CanonicalConfidenceGeometry(
        feature_map=feature_map,
        target=target,
        confidence=RCEConfidenceSet.from_diagonal_variances(
            variances,
            alpha_l2=0.1,
            alpha_linf=0.1,
        ),
        rho_by_block=rho_by_block,
    )


def _rescale_block_precision(
    geometry: CanonicalConfidenceGeometry,
    block_name: str,
    log_precision_delta: float,
) -> CanonicalConfidenceGeometry:
    multiplier = math.exp(float(log_precision_delta))
    variances = np.asarray(
        geometry.confidence.marginal_variances,
        dtype=np.float64,
    ).copy()
    variances[geometry.feature_map.block_slices[block_name]] /= multiplier
    rho_by_block = dict(geometry.rho_by_block)
    rho_by_block[block_name] *= multiplier
    return CanonicalConfidenceGeometry(
        feature_map=geometry.feature_map,
        target=geometry.target,
        confidence=RCEConfidenceSet.from_diagonal_variances(
            variances,
            alpha_l2=geometry.confidence.alpha_l2,
            alpha_linf=geometry.confidence.alpha_linf,
        ),
        rho_by_block=rho_by_block,
    )


def test_public_domain_manifest_and_feature_rank_are_deterministic() -> None:
    first = _toy_feature_map((2, 3, 2))
    second = _toy_feature_map((2, 3, 2))
    assert first.domain.manifest_hash == second.domain.manifest_hash
    assert first.feature_hash == second.feature_hash
    assert first.affine_rank == 9
    assert first.atom_cap == 10
    assert first.domain.domain_size == 12


def test_exact_milp_row_pricing_matches_exhaustive_enumeration() -> None:
    feature_map = _toy_feature_map((2, 3, 2))
    coefficients = np.random.default_rng(19).normal(
        size=feature_map.feature_dimension
    )
    exhaustive = price_legal_row(feature_map, coefficients, solver="enumerate")
    milp = price_legal_row(feature_map, coefficients, solver="milp")
    np.testing.assert_array_equal(milp.row, exhaustive.row)
    assert abs(milp.objective - exhaustive.objective) <= 1.0e-10
    assert milp.certified
    assert milp.relative_gap <= 1.0e-9


def test_pricing_energy_tables_match_canonical_row_features() -> None:
    feature_map = _toy_feature_map((2, 3, 4))
    coefficients = np.random.default_rng(29).normal(
        size=feature_map.feature_dimension
    )
    unary, pairwise = feature_map.energy_tables(coefficients)
    for row in feature_map.domain.enumerate_rows():
        table_energy = sum(
            unary[attribute][category]
            for attribute, category in enumerate(row)
        ) + sum(
            table[row[left], row[right]]
            for (left, right), table in pairwise.items()
        )
        assert abs(table_energy - feature_map.row_energy(row, coefficients)) <= 1.0e-12


def test_phase_one_and_two_column_generation_match_full_enumeration() -> None:
    feature_map = _toy_feature_map()
    geometry = _toy_geometry(feature_map)
    initial = RHCGColumnSet.create(
        feature_map,
        initial_rows=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
    )
    phase_one = solve_phase_one_rhcg(
        geometry,
        initial,
        pricing_solver="enumerate",
        maximum_columns=feature_map.atom_cap,
    )
    enumerated_one = solve_enumerated_phase_one(geometry)
    assert phase_one.certified
    assert enumerated_one.certified
    assert abs(phase_one.objective - enumerated_one.objective) <= 1.0e-9
    assert phase_one.global_gap <= 1.0e-8

    prior_moments = np.zeros(feature_map.feature_dimension, dtype=np.float64)
    phase_two = solve_phase_two_rhcg(
        geometry,
        phase_one,
        prior_moments,
        pricing_solver="enumerate",
        maximum_columns=feature_map.atom_cap,
    )
    enumerated_two = solve_enumerated_phase_two(
        geometry,
        enumerated_one,
        prior_moments,
    )
    assert phase_two.certified
    assert enumerated_two.certified
    assert abs(phase_two.objective - enumerated_two.objective) <= 1.0e-9
    assert phase_two.global_gap <= 1.0e-8


def test_rhcg_dictionary_and_pressure_are_deterministic() -> None:
    feature_map = _toy_feature_map()
    geometry = _toy_geometry(feature_map)
    columns = RHCGColumnSet.create(
        feature_map,
        initial_rows=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
    )
    prior_moments = np.zeros(feature_map.feature_dimension, dtype=np.float64)
    first = solve_rhcg_ccmp(
        geometry,
        columns,
        prior_moments,
        pricing_solver="enumerate",
        maximum_columns=feature_map.atom_cap,
    )
    second = solve_rhcg_ccmp(
        geometry,
        columns,
        prior_moments,
        pricing_solver="enumerate",
        maximum_columns=feature_map.atom_cap,
    )
    assert first.certified
    assert second.certified
    assert first.phase_one.columns.dictionary_hash == second.phase_one.columns.dictionary_hash
    assert first.phase_two.columns.dictionary_hash == second.phase_two.columns.dictionary_hash
    assert first.pressure_source == second.pressure_source
    assert first.pressure_by_block == second.pressure_by_block


def test_ccf_moment_projection_matches_direct_forest_enumeration() -> None:
    feature_map = _toy_feature_map()
    node_probabilities = (
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
    )
    product = ReleasedProductPrior(
        cardinalities=feature_map.domain.cardinalities,
        probabilities=node_probabilities,
        public_total=feature_map.domain.public_n,
        smoothing=1.0,
    )
    first = np.asarray([[0.40, 0.10], [0.10, 0.40]], dtype=np.float64)
    second = np.asarray([[0.35, 0.15], [0.15, 0.35]], dtype=np.float64)
    prior = ReleasedConfidenceForestPrior(
        product_prior=product,
        edges=(
            ConfidenceForestEdge((0, 1), first, first, 0.1, 0.09),
            ConfidenceForestEdge((1, 2), second, second, 0.05, 0.04),
        ),
        pair_results=(),
    )
    rows = feature_map.domain.enumerate_rows()
    probabilities = np.exp(prior.log_probability_rows(rows))
    assert abs(float(np.sum(probabilities)) - 1.0) <= 1.0e-12
    enumerated = probabilities @ feature_map.row_atom_answers(rows)
    np.testing.assert_allclose(
        ccf_moment_answer(feature_map, prior),
        enumerated,
        rtol=0.0,
        atol=1.0e-10,
    )


def test_phase_two_pressure_matches_log_precision_finite_difference() -> None:
    feature_map = _toy_feature_map()
    geometry = _toy_geometry(feature_map)
    prior_moments = np.zeros(feature_map.feature_dimension, dtype=np.float64)
    phase_one = solve_enumerated_phase_one(geometry)
    phase_two = solve_enumerated_phase_two(geometry, phase_one, prior_moments)
    block_name = next(iter(feature_map.block_slices))
    analytic = geometry.block_pressures(
        phase_two.answer,
        phase_two.dual.multipliers,
    )[block_name]
    step = 1.0e-4

    def value(offset: float) -> float:
        local_geometry = _rescale_block_precision(geometry, block_name, offset)
        local_one = solve_enumerated_phase_one(local_geometry)
        local_two = solve_enumerated_phase_two(
            local_geometry,
            local_one,
            prior_moments,
        )
        assert local_one.certified and local_two.certified
        return local_two.objective

    finite_difference = (value(step) - value(-step)) / (2.0 * step)
    relative_error = abs(finite_difference - analytic) / max(
        1.0,
        abs(finite_difference),
        abs(analytic),
    )
    assert relative_error <= 1.0e-5


def test_positive_phase_one_inflation_is_certified_and_pressure_audited() -> None:
    feature_map = _toy_feature_map()
    base = _toy_geometry(feature_map)
    shifted_target = np.asarray(base.target, dtype=np.float64).copy()
    shifted_target[0] = 200.0
    geometry = CanonicalConfidenceGeometry(
        feature_map=feature_map,
        target=shifted_target,
        confidence=base.confidence,
        rho_by_block=base.rho_by_block,
    )
    phase_one = solve_enumerated_phase_one(geometry)
    assert phase_one.certified
    assert phase_one.objective > 0.0
    assert phase_one.inflation == phase_one.objective
    assert phase_one.global_gap <= 1.0e-8
    block_name = next(iter(feature_map.block_slices))
    analytic = geometry.block_pressures(
        phase_one.answer,
        phase_one.dual.multipliers,
    )[block_name]
    step = 1.0e-4

    def value(offset: float) -> float:
        local = _rescale_block_precision(geometry, block_name, offset)
        result = solve_enumerated_phase_one(local)
        assert result.certified
        return result.objective

    finite_difference = (value(step) - value(-step)) / (2.0 * step)
    relative_error = abs(finite_difference - analytic) / max(
        1.0,
        abs(finite_difference),
        abs(analytic),
    )
    assert relative_error <= 1.0e-5


@pytest.mark.parametrize(
    "cardinalities",
    (
        (2, 3, 4),
        (2, 2, 3, 2),
        (2, 2, 2, 2, 2),
    ),
)
def test_exhaustive_stage_zero_certification_across_public_domains(
    cardinalities: tuple[int, ...],
) -> None:
    feature_map = _toy_feature_map(cardinalities)
    geometry = _toy_geometry(feature_map)
    initial = RHCGColumnSet.create(
        feature_map,
        initial_rows=np.asarray(
            (
                [0] * len(cardinalities),
                [value - 1 for value in cardinalities],
            ),
            dtype=np.int32,
        ),
    )
    phase_one = solve_phase_one_rhcg(
        geometry,
        initial,
        pricing_solver="enumerate",
        maximum_columns=feature_map.atom_cap,
        max_iterations=10_000,
    )
    enumerated_one = solve_enumerated_phase_one(geometry)
    assert phase_one.certified and enumerated_one.certified
    assert abs(phase_one.objective - enumerated_one.objective) <= 1.0e-9
    assert phase_one.global_gap <= 1.0e-8
    assert phase_one.columns.column_count <= feature_map.atom_cap

    prior_moments = np.zeros(feature_map.feature_dimension, dtype=np.float64)
    phase_two = solve_phase_two_rhcg(
        geometry,
        phase_one,
        prior_moments,
        pricing_solver="enumerate",
        maximum_columns=feature_map.atom_cap,
        max_iterations=10_000,
    )
    enumerated_two = solve_enumerated_phase_two(
        geometry,
        enumerated_one,
        prior_moments,
    )
    assert phase_two.certified and enumerated_two.certified
    assert abs(phase_two.objective - enumerated_two.objective) <= 1.0e-9
    assert phase_two.global_gap <= 1.0e-8
    assert phase_two.columns.column_count <= feature_map.atom_cap
