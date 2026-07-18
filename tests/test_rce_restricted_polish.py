from __future__ import annotations

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.restricted_polish import polish_interior_restricted_mixture


def test_restricted_newton_polish_closes_interior_kl_certificate() -> None:
    components = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    residuals = np.zeros((2, 1), dtype=np.float64)
    prior = np.asarray([0.75, 0.25], dtype=np.float64)
    confidence = RCEConfidenceSet.from_diagonal_variances(np.asarray([1.0]))
    result = polish_interior_restricted_mixture(
        components,
        residuals,
        np.log(prior),
        confidence,
        np.asarray([0.74, 0.26]),
        face_slack=1.0e-10,
    )
    assert result.diagnostics["strictly_interior_at_start"]
    assert result.diagnostics["objective_improvement"] > 0.0
    assert np.allclose(result.component_weights, prior, atol=1.0e-9)
    assert result.certificate["relative_primal_dual_gap"] <= 1.0e-9
    assert result.certificate["global_pricing_gap"] is None


def test_restricted_newton_polish_does_not_move_boundary_problem() -> None:
    components = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    residuals = np.asarray([[-10.0], [10.0]], dtype=np.float64)
    confidence = RCEConfidenceSet.from_diagonal_variances(np.asarray([1.0e-2]))
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    result = polish_interior_restricted_mixture(
        components,
        residuals,
        np.log(np.asarray([0.9, 0.1])),
        confidence,
        weights,
        face_slack=1.0e-10,
        interior_margin=2.0,
    )
    assert not result.diagnostics["strictly_interior_at_start"]
    assert np.allclose(result.component_weights, weights)
