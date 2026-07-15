from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.entropy import AtomEntropyState, ReleasedProductPrior
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.evolution.rce_scoring import score_candidates_rce_with_quadratic
from qdte.evolution.scoring import OrthogonalPrecisionScoreContext
from qdte.queries.types import QueryCatalogue
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.dual import RCEDualState
from qdte.rce.integer import RCEIntegerIncumbent, RCEIntegerSnapshot


RCE_METHOD = "row_realizable_confidence_set_entropic_primal_dual_v1"


@dataclass
class RCEQDTEController:
    precision: OrthogonalInteractionPrecision
    confidence: RCEConfidenceSet
    prior: ReleasedProductPrior
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
    ) -> RCEQDTEController:
        confidence = RCEConfidenceSet.from_diagonal_variances(
            precision.coefficient_variances,
            alpha_l2=float(alpha_l2),
            alpha_linf=float(alpha_linf),
        )
        prior = ReleasedProductPrior.from_released_oneway(
            qcat,
            released_target,
            cardinalities,
            public_total=int(public_total),
            smoothing=float(product_prior_smoothing),
        )
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


__all__ = ["RCE_METHOD", "RCEQDTEController"]
