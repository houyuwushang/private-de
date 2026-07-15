from __future__ import annotations

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.dual import RCEDualState


def _dual_state() -> RCEDualState:
    confidence = RCEConfidenceSet.from_diagonal_variances(
        np.asarray([2.0, 0.5, 1.5], dtype=np.float64)
    )
    state = RCEDualState.create(confidence, max_iterations=100)
    state.ellipsoid_weight = 1.7
    state.tube_positive[:] = np.asarray([0.4, 0.0, 0.8])
    state.tube_negative[:] = np.asarray([0.0, 0.6, 0.0])
    return state


def test_rce_single_edit_lagrangian_gain_matches_full_recomputation() -> None:
    state = _dual_state()
    residual = np.asarray([1.2, -0.7, 0.3], dtype=np.float64)
    delta = np.asarray([0.5, -0.25, 1.0], dtype=np.float64)
    regularizer = 12.3
    regularizer_gain = 0.41
    score = state.batch_gain(
        residual=residual,
        delta_sum=delta,
        regularizer_gain=regularizer_gain,
    )
    before = state.lagrangian(residual, regularizer).value
    after = state.lagrangian(residual - delta, regularizer - regularizer_gain).value
    assert np.isclose(score, before - after, rtol=1.0e-12, atol=1.0e-12)


def test_rce_batch_uses_aggregate_delta_cross_terms() -> None:
    state = _dual_state()
    residual = np.asarray([1.2, -0.7, 0.3], dtype=np.float64)
    deltas = np.asarray(
        [[0.5, -0.25, 1.0], [-0.2, 0.4, -0.5]],
        dtype=np.float64,
    )
    aggregate = np.sum(deltas, axis=0)
    regularizer = 12.3
    aggregate_regularizer_gain = 0.27
    score = state.batch_gain(
        residual=residual,
        delta_sum=aggregate,
        regularizer_gain=aggregate_regularizer_gain,
    )
    before = state.lagrangian(residual, regularizer).value
    after = state.lagrangian(
        residual - aggregate,
        regularizer - aggregate_regularizer_gain,
    ).value
    individual = state.candidate_gains(
        residual=residual,
        deltas=deltas,
        regularizer_gains=np.asarray([0.1, 0.17]),
        edit_costs=np.zeros(2),
        lambda_cost=0.0,
    )
    assert np.isclose(score, before - after, rtol=1.0e-12, atol=1.0e-12)
    assert not np.isclose(score, float(np.sum(individual)), rtol=1.0e-8, atol=1.0e-8)


def test_rce_projected_dual_update_and_certificate_are_nonnegative() -> None:
    confidence = RCEConfidenceSet.from_diagonal_variances(np.ones(2))
    state = RCEDualState.create(confidence, max_iterations=25)
    residual = 2.0 * confidence.coordinate_bounds
    state.update(residual)
    certificate = state.certificate(residual)
    assert state.ellipsoid_weight >= 0.0
    assert np.all(state.tube_positive >= 0.0)
    assert np.all(state.tube_negative >= 0.0)
    assert certificate["primal_violation"] > 0.0
    assert certificate["dual_violation"] == 0.0
    assert certificate["stationarity_certified"] is False
