from __future__ import annotations

import numpy as np

from qdte.evolution.coarse_precision import CoarsenedInteractionPrecision
from qdte.evolution.entropy import AtomEntropyState
from qdte.evolution.scoring import OrthogonalPrecisionScoreContext
from qdte.measurement.direct_coarse import CoarseInteractionTranscript
from qdte.queries.types import QueryCatalogue
from qdte.rce.coarse_forest_prior import build_coarsened_confidence_forest_prior
from qdte.rce.coarse_scoring import (
    score_candidates_coarsened_rce_with_quadratic,
)
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.controller import RCEQDTEController
from qdte.rce.dual import RCEDualState
from qdte.rce.integer import RCEIntegerIncumbent
from qdte.evolution.candidates import CandidateBatch


C2_RCE_METHOD = "row_realizable_confidence_set_coarsened_full_covariance_v1"


class CoarsenedRCEQDTEController(RCEQDTEController):
    precision: CoarsenedInteractionPrecision

    @classmethod
    def create(
        cls,
        *,
        qcat: QueryCatalogue,
        released_target: np.ndarray,
        cardinalities: np.ndarray | tuple[int, ...],
        public_total: int,
        initial_rows: np.ndarray,
        initial_answers: np.ndarray,
        initial_residual: np.ndarray,
        precision: CoarsenedInteractionPrecision,
        interaction_transcript: CoarseInteractionTranscript,
        max_iterations: int,
        alpha_l2: float = 0.025,
        alpha_linf: float = 0.025,
        product_prior_smoothing: float = 1.0,
    ) -> CoarsenedRCEQDTEController:
        confidence = RCEConfidenceSet.from_covariance(
            precision.coefficient_covariance,
            alpha_l2=float(alpha_l2),
            alpha_linf=float(alpha_linf),
        )
        prior = build_coarsened_confidence_forest_prior(
            qcat,
            released_target,
            cardinalities,
            interaction_transcript,
            public_total=int(public_total),
            smoothing=float(product_prior_smoothing),
        )
        entropy_state = AtomEntropyState.from_rows(initial_rows, prior)
        normalized_regularizer = float(
            entropy_state.regularizer / entropy_state.n_rows
        )
        dual = RCEDualState.create(confidence, max_iterations=int(max_iterations))
        incumbent = RCEIntegerIncumbent(confidence)
        coefficient_residual = precision.coefficient_coordinates(initial_residual)
        incumbent.consider(
            rows=initial_rows,
            answers=initial_answers,
            coefficient_residual=coefficient_residual,
            regularizer=normalized_regularizer,
            iteration=0,
        )
        return cls(
            precision=precision,
            confidence=confidence,
            prior=prior,
            entropy_state=entropy_state,
            dual=dual,
            incumbent=incumbent,
            initial_regularizer=normalized_regularizer,
            initial_confidence=confidence.evaluate(
                coefficient_residual
            ).to_dict(),
            method=C2_RCE_METHOD,
        )

    def candidate_scores(
        self,
        candidates: CandidateBatch,
        query_residual: np.ndarray,
        *,
        lambda_cost: float,
        chunk_size: int,
        context: OrthogonalPrecisionScoreContext | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if context is None:
            raise ValueError("C2 RCE requires a compiled coarse score context")
        return score_candidates_coarsened_rce_with_quadratic(
            candidates,
            query_residual,
            self.precision,
            self.dual,
            self.entropy_state,
            lambda_cost=float(lambda_cost),
            chunk_size=int(chunk_size),
            context=context,
        )

    def prefix_advantages_from_rows(
        self,
        *,
        query_residual: np.ndarray,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        edit_cost_prefix: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        """Evaluate cumulative prefixes under the exact full covariance."""

        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        costs = np.asarray(edit_cost_prefix, dtype=np.float64)
        if old.shape != new.shape or old.ndim != 2:
            raise ValueError("C2 prefix rows must be matching row matrices")
        if costs.shape != (len(old),):
            raise ValueError("C2 prefix edit costs must match the row prefixes")
        edit_deltas = self.precision.row_feature_deltas(old, new)
        prefix_deltas = np.cumsum(edit_deltas, axis=0, dtype=np.float64)
        regularizer_gains = self.entropy_state.prefix_gains(old, new)
        coefficient_residual = self.coefficient_residual(query_residual)
        precision_residual = self.precision.coefficient_precision_matvec(
            coefficient_residual
        )
        quadratic = self.precision.feature_quadratic_many(prefix_deltas)
        ellipsoid_scale = (
            float(self.dual.ellipsoid_weight)
            / self.confidence.squared_discrepancy_threshold
        )
        ellipsoid_gain = ellipsoid_scale * (
            2.0 * (prefix_deltas @ precision_residual) - quadratic
        )
        tube_gain = prefix_deltas @ self.dual.signed_tube_weights()
        return (
            regularizer_gains / self.entropy_state.n_rows
            + ellipsoid_gain
            + tube_gain
            - self.effective_lambda_cost(lambda_cost) * costs
        )


__all__ = ["C2_RCE_METHOD", "CoarsenedRCEQDTEController"]
