from __future__ import annotations

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.entropy import AtomEntropyState
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.evolution.scoring import (
    OrthogonalPrecisionScoreContext,
    score_candidates_orthogonal_precision_with_quadratic,
)
from qdte.rce.dual import RCEDualState


def rce_coefficient_linear_weights(
    residual: np.ndarray,
    precision: OrthogonalInteractionPrecision,
    dual: RCEDualState,
) -> np.ndarray:
    """Return the exact coefficient-linear part of an RCE edit gain."""

    coefficient_residual = precision.coefficient_coordinates(residual)
    if coefficient_residual.shape != (dual.confidence.dimension,):
        raise ValueError("RCE confidence dimension does not match orthogonal coefficients")
    ellipsoid = (
        2.0
        * float(dual.ellipsoid_weight)
        / dual.confidence.squared_discrepancy_threshold
        * dual.confidence.precision_matvec(coefficient_residual)
    )
    return ellipsoid + dual.signed_tube_weights()


def score_candidates_rce_with_quadratic(
    candidates: CandidateBatch,
    residual: np.ndarray,
    precision: OrthogonalInteractionPrecision,
    dual: RCEDualState,
    entropy_state: AtomEntropyState,
    *,
    lambda_cost: float,
    chunk_size: int = 4096,
    context: OrthogonalPrecisionScoreContext | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score exact RCE Lagrangian decreases for single-row candidates."""

    if candidates.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy()
    if dual.confidence.dimension != precision.coefficient_dimension:
        raise ValueError("RCE confidence dimension does not match orthogonal precision")
    if not np.allclose(
        dual.confidence.marginal_variances,
        precision.coefficient_variances,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("RCE confidence covariance must match the raw orthogonal transcript")

    linear_weights = rce_coefficient_linear_weights(residual, precision, dual)
    pseudo_coefficients = linear_weights * precision.coefficient_variances
    pseudo_residual = precision.coefficient_adjoint(pseudo_coefficients)
    base_scores, base_quadratic = score_candidates_orthogonal_precision_with_quadratic(
        candidates,
        pseudo_residual,
        precision,
        lambda_cost=0.0,
        chunk_size=chunk_size,
        context=context,
    )
    ellipsoid_scale = (
        float(dual.ellipsoid_weight)
        / dual.confidence.squared_discrepancy_threshold
    )
    n_rows = float(entropy_state.n_rows)
    regularizer_gains = entropy_state.candidate_gains(
        candidates.old_rows,
        candidates.new_rows,
    ) / n_rows
    scores = (
        np.asarray(base_scores, dtype=np.float64)
        + (0.5 - ellipsoid_scale)
        * np.asarray(base_quadratic, dtype=np.float64)
        + regularizer_gains
        - (float(lambda_cost) / n_rows)
        * candidates.edit_cost.astype(np.float64, copy=False)
    )
    quadratic = (
        2.0 * ellipsoid_scale * np.asarray(base_quadratic, dtype=np.float64)
    )
    return scores.astype(np.float32), quadratic.astype(np.float32)


__all__ = [
    "rce_coefficient_linear_weights",
    "score_candidates_rce_with_quadratic",
]
