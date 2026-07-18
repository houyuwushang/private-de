from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.coarse_precision import CoarsenedInteractionPrecision
from qdte.evolution.entropy import AtomEntropyState
from qdte.evolution.scoring import OrthogonalPrecisionScoreContext
from qdte.rce.dual import RCEDualState


def score_candidates_coarsened_rce_with_quadratic(
    candidates: CandidateBatch,
    residual: np.ndarray,
    precision: CoarsenedInteractionPrecision,
    dual: RCEDualState,
    entropy_state: AtomEntropyState,
    *,
    lambda_cost: float,
    chunk_size: int,
    context: OrthogonalPrecisionScoreContext,
) -> tuple[np.ndarray, np.ndarray]:
    if candidates.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy()
    if int(chunk_size) <= 0:
        raise ValueError("C2 RCE chunk_size must be positive")
    if dual.confidence.dimension != precision.coefficient_dimension:
        raise ValueError("C2 RCE confidence and precision dimensions do not match")
    if context.coefficient_dimension != precision.coefficient_dimension:
        raise ValueError("C2 RCE score context has the wrong dimension")
    if not np.allclose(
        dual.confidence.marginal_variances,
        precision.coefficient_variances,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("C2 RCE confidence marginal variances do not match")

    coefficient_residual = precision.coefficient_coordinates(residual)
    linear_weights = (
        2.0
        * float(dual.ellipsoid_weight)
        / dual.confidence.squared_discrepancy_threshold
        * dual.confidence.precision_matvec(coefficient_residual)
        + dual.signed_tube_weights()
    )
    ellipsoid_scale = (
        float(dual.ellipsoid_weight)
        / dual.confidence.squared_discrepancy_threshold
    )
    n_rows = float(entropy_state.n_rows)
    score_output: list[np.ndarray] = []
    quadratic_output: list[np.ndarray] = []
    for start in range(0, candidates.size, int(chunk_size)):
        end = min(start + int(chunk_size), candidates.size)
        base_scores, quadratic = context.scorer(
            jnp.asarray(candidates.old_rows[start:end], dtype=jnp.int32),
            jnp.asarray(candidates.new_rows[start:end], dtype=jnp.int32),
            jnp.asarray(linear_weights, dtype=jnp.float32),
            jnp.asarray(candidates.edit_cost[start:end], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        regularizer_gains = entropy_state.candidate_gains(
            candidates.old_rows[start:end],
            candidates.new_rows[start:end],
        ) / n_rows
        local_quadratic = np.asarray(quadratic, dtype=np.float64)
        scores = (
            np.asarray(base_scores, dtype=np.float64)
            + (0.5 - ellipsoid_scale) * local_quadratic
            + regularizer_gains
            - (float(lambda_cost) / n_rows)
            * candidates.edit_cost[start:end].astype(np.float64, copy=False)
        )
        score_output.append(scores.astype(np.float32))
        quadratic_output.append(
            (2.0 * ellipsoid_scale * local_quadratic).astype(np.float32)
        )
    return np.concatenate(score_output), np.concatenate(quadratic_output)


__all__ = ["score_candidates_coarsened_rce_with_quadratic"]
