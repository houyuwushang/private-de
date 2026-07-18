from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.entropy import (
    AtomEntropyState,
    ReleasedProductPrior,
    RowReferencePrior,
)
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.evolution.rce_scoring import (
    RCE_SCORING_METHOD,
    rce_coefficient_linear_weights,
    score_candidates_rce_with_quadratic,
)
from qdte.evolution.scoring import OrthogonalPrecisionScoreContext
from qdte.queries.types import QueryCatalogue
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.support_coarsening import SupportCoarsening
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.dual import RCEDualState
from qdte.rce.forest_prior import build_released_confidence_forest_prior
from qdte.rce.integer import RCEIntegerIncumbent, RCEIntegerSnapshot


RCE_METHOD = "row_realizable_confidence_set_entropic_primal_dual_v1"
RCE_PRODUCT_PRIOR = "released_product"
RCE_CCF_PRIOR = "released_confidence_forest_v1"


@dataclass
class RCEQDTEController:
    precision: OrthogonalInteractionPrecision
    confidence: RCEConfidenceSet
    prior: RowReferencePrior
    entropy_state: AtomEntropyState
    dual: RCEDualState
    incumbent: RCEIntegerIncumbent
    initial_regularizer: float
    initial_confidence: dict[str, Any]
    method: str = RCE_METHOD

    @property
    def regularizer(self) -> float:
        """Return D_KL(p_empirical || p0), not the count-scaled n * D_KL."""

        return float(self.entropy_state.regularizer / self.entropy_state.n_rows)

    def effective_lambda_cost(self, lambda_cost: float) -> float:
        return float(lambda_cost) / float(self.entropy_state.n_rows)

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
        precision: OrthogonalInteractionPrecision,
        max_iterations: int,
        alpha_l2: float = 0.025,
        alpha_linf: float = 0.025,
        product_prior_smoothing: float = 1.0,
        reference_prior: str = RCE_PRODUCT_PRIOR,
        interaction_transcript: HierarchicalInteractionTranscript | None = None,
        support_coarsening: SupportCoarsening | None = None,
    ) -> RCEQDTEController:
        confidence = RCEConfidenceSet.from_diagonal_variances(
            precision.coefficient_variances,
            alpha_l2=float(alpha_l2),
            alpha_linf=float(alpha_linf),
            enforce_zero_variance_equalities=True,
            exact_dual_scale=1.0,
        )
        prior_name = str(reference_prior)
        if support_coarsening is not None and prior_name != RCE_CCF_PRIOR:
            raise ValueError("Support coarsening is defined only for the CCF prior")
        if prior_name == RCE_PRODUCT_PRIOR:
            prior: RowReferencePrior = ReleasedProductPrior.from_released_oneway(
                qcat,
                released_target,
                cardinalities,
                public_total=int(public_total),
                smoothing=float(product_prior_smoothing),
            )
        elif prior_name == RCE_CCF_PRIOR:
            if interaction_transcript is None:
                raise ValueError("CCF RCE requires the released interaction transcript")
            prior = build_released_confidence_forest_prior(
                qcat,
                released_target,
                cardinalities,
                interaction_transcript,
                public_total=int(public_total),
                smoothing=float(product_prior_smoothing),
                support_coarsening=support_coarsening,
            )
        else:
            raise ValueError(f"Unsupported RCE reference prior {prior_name!r}")
        entropy_state = AtomEntropyState.from_rows(initial_rows, prior)
        normalized_regularizer = float(entropy_state.regularizer / entropy_state.n_rows)
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
        initial_confidence = confidence.evaluate(coefficient_residual).to_dict()
        return cls(
            precision=precision,
            confidence=confidence,
            prior=prior,
            entropy_state=entropy_state,
            dual=dual,
            incumbent=incumbent,
            initial_regularizer=normalized_regularizer,
            initial_confidence=initial_confidence,
        )

    def coefficient_residual(self, query_residual: np.ndarray) -> np.ndarray:
        return self.precision.coefficient_coordinates(query_residual)

    def update_dual(self, query_residual: np.ndarray) -> dict[str, float | int]:
        return self.dual.update(self.coefficient_residual(query_residual))

    def candidate_scores(
        self,
        candidates: CandidateBatch,
        query_residual: np.ndarray,
        *,
        lambda_cost: float,
        chunk_size: int,
        context: OrthogonalPrecisionScoreContext | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return score_candidates_rce_with_quadratic(
            candidates,
            query_residual,
            self.precision,
            self.dual,
            self.entropy_state,
            lambda_cost=float(lambda_cost),
            chunk_size=int(chunk_size),
            context=context,
        )

    def prefix_advantages(
        self,
        *,
        query_residual: np.ndarray,
        query_delta_prefix: np.ndarray,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        edit_cost_prefix: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        coefficient_residual = self.coefficient_residual(query_residual)
        coefficient_deltas = self.precision.coefficient_coordinates_many(
            np.asarray(query_delta_prefix, dtype=np.float64)
        )
        regularizer_gains = self.entropy_state.prefix_gains(old_rows, new_rows)
        return self.dual.candidate_gains(
            residual=coefficient_residual,
            deltas=coefficient_deltas,
            regularizer_gains=(regularizer_gains / self.entropy_state.n_rows),
            edit_costs=np.asarray(edit_cost_prefix, dtype=np.float64),
            lambda_cost=self.effective_lambda_cost(lambda_cost),
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
        """Evaluate every edit prefix directly in coefficient feature space.

        Query-space prefix deltas are unnecessary for an orthogonal precision
        objective: a row edit already has an exact coefficient delta.  Keeping
        the cumulative calculation in that space avoids materializing and then
        re-projecting a ``num_prefixes x num_queries`` matrix.
        """

        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        costs = np.asarray(edit_cost_prefix, dtype=np.float64)
        if old.shape != new.shape or old.ndim != 2:
            raise ValueError("RCE prefix rows must be matching two-dimensional arrays")
        if costs.shape != (len(old),):
            raise ValueError("RCE prefix edit costs must match the row prefixes")
        precision_diagonal = self.confidence.precision_diagonal
        if precision_diagonal is None:
            raise ValueError(
                "Feature-space RCE prefix scoring requires diagonal confidence precision"
            )
        linear_weights = rce_coefficient_linear_weights(
            query_residual,
            self.precision,
            self.dual,
        )
        linear_values, quadratic_values = (
            self.precision.prefix_linear_and_diagonal_quadratic(
                old,
                new,
                linear_weights,
                precision_diagonal,
            )
        )
        regularizer_gains = self.entropy_state.prefix_gains(old, new)
        ellipsoid_scale = (
            float(self.dual.ellipsoid_weight)
            / self.confidence.squared_discrepancy_threshold
        )
        return (
            regularizer_gains / self.entropy_state.n_rows
            + linear_values
            - ellipsoid_scale * quadratic_values
            - self.effective_lambda_cost(lambda_cost) * costs
        )

    def prefix_advantages_from_rows_gpu(
        self,
        *,
        query_residual: np.ndarray,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        edit_cost_prefix: np.ndarray,
        lambda_cost: float,
        context: OrthogonalPrecisionScoreContext,
        static_capacity: int,
    ) -> np.ndarray:
        """Evaluate coefficient prefix terms on GPU with a static row shape."""

        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        costs = np.asarray(edit_cost_prefix, dtype=np.float64)
        if old.shape != new.shape or old.ndim != 2 or len(old) == 0:
            raise ValueError("GPU RCE prefix rows must be nonempty matching matrices")
        if costs.shape != (len(old),):
            raise ValueError("GPU RCE prefix edit costs must match the row prefixes")
        capacity = int(static_capacity)
        if capacity < len(old):
            raise ValueError("GPU RCE prefix capacity is smaller than the selected prefix")
        if context.prefix_scorer is None:
            raise ValueError("GPU RCE prefix scoring is unavailable in this score context")
        if context.coefficient_dimension != self.confidence.dimension:
            raise ValueError("GPU RCE prefix context has a mismatched coefficient dimension")
        precision_diagonal = self.confidence.precision_diagonal
        if precision_diagonal is None:
            raise ValueError("GPU RCE prefix scoring currently requires diagonal confidence precision")

        padded_old = np.repeat(old[:1], capacity, axis=0)
        padded_new = padded_old.copy()
        padded_old[: len(old)] = old
        padded_new[: len(new)] = new
        linear_weights = rce_coefficient_linear_weights(
            query_residual,
            self.precision,
            self.dual,
        )
        linear, quadratic = context.prefix_scorer(
            padded_old,
            padded_new,
            np.asarray(linear_weights, dtype=np.float32),
            np.asarray(precision_diagonal, dtype=np.float32),
        )
        linear_values = np.asarray(linear, dtype=np.float32)[: len(old)].astype(
            np.float64
        )
        quadratic_values = np.asarray(quadratic, dtype=np.float32)[: len(old)].astype(
            np.float64
        )
        regularizer_gains = self.entropy_state.prefix_gains(old, new)
        ellipsoid_scale = (
            float(self.dual.ellipsoid_weight)
            / self.confidence.squared_discrepancy_threshold
        )
        return (
            regularizer_gains / self.entropy_state.n_rows
            + linear_values
            - ellipsoid_scale * quadratic_values
            - self.effective_lambda_cost(lambda_cost) * costs
        )

    def batch_gain(
        self,
        *,
        query_residual: np.ndarray,
        query_delta_sum: np.ndarray,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        edit_cost: float,
        lambda_cost: float,
    ) -> tuple[float, float]:
        coefficient_residual = self.coefficient_residual(query_residual)
        coefficient_delta = self.precision.coefficient_coordinates(query_delta_sum)
        regularizer_gain = float(
            self.entropy_state.batch_gain(old_rows, new_rows)
            / self.entropy_state.n_rows
        )
        total = self.dual.batch_gain(
            residual=coefficient_residual,
            delta_sum=coefficient_delta,
            regularizer_gain=regularizer_gain,
            edit_cost=float(edit_cost),
            lambda_cost=self.effective_lambda_cost(lambda_cost),
        )
        return total, regularizer_gain

    def apply_batch(self, old_rows: np.ndarray, new_rows: np.ndarray) -> float:
        return float(
            self.entropy_state.apply_batch(old_rows, new_rows)
            / self.entropy_state.n_rows
        )

    def consider(
        self,
        *,
        rows: np.ndarray,
        answers: np.ndarray,
        query_residual: np.ndarray,
        iteration: int,
    ) -> bool:
        return self.incumbent.consider(
            rows=rows,
            answers=answers,
            coefficient_residual=self.coefficient_residual(query_residual),
            regularizer=self.regularizer,
            iteration=int(iteration),
        )

    def lagrangian_value(self, query_residual: np.ndarray) -> float:
        return self.dual.lagrangian(
            self.coefficient_residual(query_residual),
            self.regularizer,
        ).value

    def current_confidence(self, query_residual: np.ndarray) -> dict[str, Any]:
        return self.confidence.evaluate(
            self.coefficient_residual(query_residual)
        ).to_dict()

    def restore_incumbent(self) -> RCEIntegerSnapshot:
        best = self.incumbent.best
        self.entropy_state = AtomEntropyState.from_rows(best.rows, self.prior)
        return best

    def assert_matches(self, rows: np.ndarray) -> None:
        self.entropy_state.assert_matches(rows)

    def diagnostics(self, query_residual: np.ndarray) -> dict[str, Any]:
        coefficient_residual = self.coefficient_residual(query_residual)
        return {
            "enabled": True,
            "method": self.method,
            "scoring_method": RCE_SCORING_METHOD,
            "confidence_set": self.confidence.diagnostics(),
            "initial_confidence": dict(self.initial_confidence),
            "final_confidence": self.confidence.evaluate(coefficient_residual).to_dict(),
            "prior": self.prior.diagnostics(),
            "state": self.entropy_state.diagnostics(),
            "regularizer_definition": "D_KL(p_empirical||p0)",
            "count_scaled_regularizer": float(self.entropy_state.regularizer),
            "dual": self.dual.diagnostics(),
            "certificate": self.dual.certificate(coefficient_residual),
            "integer_incumbent": self.incumbent.diagnostics(),
            "initial_regularizer": self.initial_regularizer,
            "final_regularizer": self.regularizer,
        }


__all__ = [
    "RCE_CCF_PRIOR",
    "RCE_METHOD",
    "RCE_PRODUCT_PRIOR",
    "RCEQDTEController",
]
