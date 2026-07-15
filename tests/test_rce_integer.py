from __future__ import annotations

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.integer import RCEIntegerIncumbent


def _consider(
    incumbent: RCEIntegerIncumbent,
    residual: float,
    regularizer: float,
    iteration: int,
) -> bool:
    return incumbent.consider(
        rows=np.asarray([[iteration % 2]], dtype=np.int32),
        answers=np.asarray([float(iteration)], dtype=np.float32),
        coefficient_residual=np.asarray([residual], dtype=np.float64),
        regularizer=regularizer,
        iteration=iteration,
    )


def test_integer_incumbent_uses_slack_then_kl_lexicographic_order() -> None:
    confidence = RCEConfidenceSet.from_diagonal_variances(np.asarray([1.0]))
    incumbent = RCEIntegerIncumbent(confidence)

    assert _consider(incumbent, residual=10.0, regularizer=0.1, iteration=0)
    assert _consider(incumbent, residual=5.0, regularizer=100.0, iteration=1)
    assert _consider(incumbent, residual=0.0, regularizer=50.0, iteration=2)
    assert _consider(incumbent, residual=0.1, regularizer=20.0, iteration=3)
    assert not _consider(incumbent, residual=0.0, regularizer=30.0, iteration=4)

    assert incumbent.best.iteration == 3
    diagnostics = incumbent.diagnostics()
    assert diagnostics["scope"] == "qdte_visited_integer_tables"
    assert diagnostics["globally_certified"] is False
    assert diagnostics["regularizer_definition"] == "D_KL(p_empirical||p0)"
    assert diagnostics["best_kl_per_row"] == diagnostics["best_regularizer"]
    assert diagnostics["best_confidence"]["inside"]
