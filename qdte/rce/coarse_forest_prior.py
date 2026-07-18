from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2

from qdte.evolution.entropy import ReleasedProductPrior
from qdte.measurement.direct_coarse import CoarseInteractionTranscript
from qdte.measurement.support_coarsening import lift_coarse_pair_distribution
from qdte.queries.types import QueryCatalogue
from qdte.rce.forest_prior import (
    CCF_ALPHA_STRUCT,
    ConfidenceForestEdge,
    ReleasedConfidenceForestPrior,
    _maximum_weight_forest,
    solve_ccf_pair,
)


def build_coarsened_confidence_forest_prior(
    qcat: QueryCatalogue,
    released_target: np.ndarray,
    cardinalities: np.ndarray | tuple[int, ...],
    transcript: CoarseInteractionTranscript,
    *,
    public_total: int,
    smoothing: float = 1.0,
    alpha_struct: float = CCF_ALPHA_STRUCT,
) -> ReleasedConfidenceForestPrior:
    """Build CCF directly from a C2 coarse-interaction transcript."""

    if not np.isclose(float(alpha_struct), CCF_ALPHA_STRUCT, rtol=0.0, atol=0.0):
        raise ValueError("C2 CCF requires alpha_struct=0.05")
    cards = tuple(int(value) for value in np.asarray(cardinalities, dtype=np.int64))
    if cards != transcript.support.original_cardinalities:
        raise ValueError("C2 CCF cardinalities do not match the support map")
    if int(public_total) != int(transcript.public_total):
        raise ValueError("C2 CCF public_total does not match the transcript")
    product_prior = ReleasedProductPrior.from_released_oneway(
        qcat,
        released_target,
        cards,
        public_total=int(public_total),
        smoothing=float(smoothing),
    )
    if not transcript.pairs:
        return ReleasedConfidenceForestPrior(
            product_prior=product_prior,
            edges=(),
            pair_results=(),
            support_coarsening=transcript.support,
        )

    simultaneous_probability = 1.0 - float(alpha_struct) / float(
        len(transcript.pairs)
    )
    pair_results = []
    n_squared = float(public_total) ** 2
    for pair in transcript.pairs:
        left, right = pair
        observation = transcript.pair(pair)
        left_map = transcript.support.attributes[left]
        right_map = transcript.support.attributes[right]
        left_marginal = left_map.aggregate_probabilities(
            product_prior.probabilities[left]
        )
        right_marginal = right_map.aggregate_probabilities(
            product_prior.probabilities[right]
        )
        covariance = observation.interaction_covariance / n_squared
        eigenvalues = np.linalg.eigvalsh(covariance)
        tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
        rank = int(np.sum(eigenvalues > tolerance))
        if rank <= 0:
            raise ValueError("C2 CCF interaction covariance must have positive rank")
        radius = float(chi2.ppf(simultaneous_probability, rank))
        if not math.isfinite(radius) or radius <= 0.0:
            raise RuntimeError("C2 CCF produced an invalid confidence radius")
        pair_results.append(
            solve_ccf_pair(
                pair,
                left_marginal,
                right_marginal,
                observation.interaction_center / float(public_total),
                covariance,
                confidence_radius=radius,
            )
        )

    frozen_results = tuple(pair_results)
    selected_results = _maximum_weight_forest(frozen_results, len(cards))
    lambda_n = 1.0 / float(public_total + 1)
    edges: list[ConfidenceForestEdge] = []
    for result in selected_results:
        if result.pair_table is None:
            raise AssertionError("Selected C2 CCF edge is missing its pair table")
        left, right = result.pair
        unsmoothed = lift_coarse_pair_distribution(
            result.pair_table,
            product_prior.probabilities[left],
            product_prior.probabilities[right],
            transcript.support.attributes[left],
            transcript.support.attributes[right],
        )
        product = np.outer(
            product_prior.probabilities[left],
            product_prior.probabilities[right],
        )
        smoothed = (1.0 - lambda_n) * unsmoothed + lambda_n * product
        edges.append(
            ConfidenceForestEdge(
                pair=result.pair,
                table=smoothed,
                unsmoothed_table=unsmoothed,
                weight=result.kl_upper,
                weight_lower_bound=result.weight_lower_bound,
            )
        )
    return ReleasedConfidenceForestPrior(
        product_prior=product_prior,
        edges=tuple(edges),
        pair_results=frozen_results,
        support_coarsening=transcript.support,
    )


__all__ = ["build_coarsened_confidence_forest_prior"]
