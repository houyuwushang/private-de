from __future__ import annotations

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet


def test_rce_mixed_confidence_set_has_nominal_simulated_coverage() -> None:
    rng = np.random.default_rng(20260715)
    variances = np.asarray([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    confidence = RCEConfidenceSet.from_diagonal_variances(variances)
    draws = rng.normal(size=(20_000, len(variances))) * np.sqrt(variances)
    covered = np.fromiter(
        (confidence.contains(draw) for draw in draws),
        dtype=np.bool_,
        count=len(draws),
    )
    assert float(np.mean(covered)) >= 0.945
    assert confidence.effective_rank == 4
    assert confidence.tube_dimension == 4


def test_rce_diagonal_zero_precision_coordinate_is_excluded() -> None:
    confidence = RCEConfidenceSet.from_diagonal_variances(
        np.asarray([1.0, 0.0, 4.0], dtype=np.float64)
    )
    baseline = confidence.evaluate(np.asarray([0.2, 0.0, -0.5]))
    null_shift = confidence.evaluate(np.asarray([0.2, 1.0e9, -0.5]))
    assert confidence.effective_rank == 2
    assert confidence.tube_dimension == 2
    assert baseline.squared_discrepancy == null_shift.squared_discrepancy
    assert baseline.max_standardized_coordinate == null_shift.max_standardized_coordinate
    assert np.array_equal(confidence.tube_mask, np.asarray([True, False, True]))


def test_rce_singular_covariance_uses_pseudoinverse_support() -> None:
    covariance = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    confidence = RCEConfidenceSet.from_covariance(covariance)
    assert confidence.effective_rank == 1
    assert np.isclose(confidence.squared_discrepancy(np.asarray([1.0, 1.0])), 1.0)
    assert np.isclose(confidence.squared_discrepancy(np.asarray([1.0, -1.0])), 0.0)


def test_rce_final_coordinate_envelope_follows_from_two_feasible_points() -> None:
    confidence = RCEConfidenceSet.from_diagonal_variances(
        np.asarray([1.0, 4.0], dtype=np.float64)
    )
    bounds = confidence.coordinate_bounds
    truth_residual = np.asarray([0.4 * bounds[0], -0.3 * bounds[1]])
    synthetic_residual = np.asarray([-0.5 * bounds[0], 0.6 * bounds[1]])
    assert confidence.contains(truth_residual)
    assert confidence.contains(synthetic_residual)
    answer_difference = truth_residual - synthetic_residual
    assert np.all(np.abs(answer_difference) <= 2.0 * bounds + 1.0e-12)
