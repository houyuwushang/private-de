from __future__ import annotations

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.relaxed import solve_relaxed_rce


def test_relaxed_rce_returns_product_prior_when_it_is_feasible() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=np.float64)
    target = np.asarray([0.8], dtype=np.float64)
    prior = np.asarray([0.5, 0.5], dtype=np.float64)
    confidence = RCEConfidenceSet.from_diagonal_variances(np.asarray([1.0]))
    result = solve_relaxed_rce(features, target, prior, confidence)
    assert result.stage_one["success"]
    assert result.stage_two["success"]
    assert result.globally_certified
    assert result.unique_on_declared_support
    assert result.slack_star <= 1.0e-9
    assert np.allclose(result.probabilities, prior, atol=1.0e-7)


def test_relaxed_rce_is_deterministic_and_moves_only_as_far_as_required() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=np.float64)
    target = np.asarray([1.0], dtype=np.float64)
    prior = np.asarray([0.9, 0.1], dtype=np.float64)
    confidence = RCEConfidenceSet.from_diagonal_variances(np.asarray([1.0e-4]))
    first = solve_relaxed_rce(features, target, prior, confidence)
    second = solve_relaxed_rce(features, target, prior, confidence)
    assert first.stage_one["success"]
    assert first.stage_two["success"]
    assert first.confidence["inside"]
    assert first.probabilities[1] > prior[1]
    assert np.allclose(first.probabilities, second.probabilities, atol=1.0e-10)
    assert np.isclose(first.kl_objective, second.kl_objective, atol=1.0e-12)


def test_restricted_relaxed_rce_never_claims_global_certificate() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=np.float64)
    confidence = RCEConfidenceSet.from_diagonal_variances(np.asarray([1.0]))
    result = solve_relaxed_rce(
        features,
        np.asarray([0.5]),
        np.asarray([0.25, 0.25]),
        confidence,
        support_kind="discovered_atoms",
        globally_certified=False,
    )
    assert result.support_kind == "discovered_atoms"
    assert result.globally_certified is False
