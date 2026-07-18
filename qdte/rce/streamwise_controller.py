from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.entropy import AtomEntropyState, RowReferencePrior
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.evolution.scoring import (
    OrthogonalPrecisionScoreContext,
    score_candidates_orthogonal_dynamic_diagonal_with_quadratic,
)
from qdte.rce.streamwise import StreamwiseRCEConfidenceSet
from qdte.rce.streamwise_dual import StreamwiseRCEDualState


STREAMWISE_RCE_CONTROLLER_METHOD = (
    "adaptive_safe_streamwise_entropic_primal_dual_qdte_v1"
)


@dataclass(frozen=True)
class StreamwiseRCEIntegerSnapshot:
    rows: np.ndarray
    answers: np.ndarray
    combined_coefficient_residual: np.ndarray
    regularizer: float
    iteration: int
    confidence: dict[str, Any]


class StreamwiseRCEIntegerIncumbent:
    def __init__(
        self,
        confidence: StreamwiseRCEConfidenceSet,
        *,
        feasibility_tolerance: float = 1.0e-9,
        comparison_tolerance: float = 1.0e-12,
    ) -> None:
        self.confidence = confidence
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.comparison_tolerance = float(comparison_tolerance)
        self._best: StreamwiseRCEIntegerSnapshot | None = None
        self.num_considered = 0
        self.num_improvements = 0

    @property
    def best(self) -> StreamwiseRCEIntegerSnapshot:
        if self._best is None:
            raise RuntimeError("Streamwise RCE incumbent has not observed a table")
        return self._best

    def _strictly_better(
        self,
        candidate: StreamwiseRCEIntegerSnapshot,
        current: StreamwiseRCEIntegerSnapshot,
    ) -> bool:
        candidate_feasible = bool(candidate.confidence["inside"])
        current_feasible = bool(current.confidence["inside"])
        if candidate_feasible != current_feasible:
            return candidate_feasible
        tolerance = self.comparison_tolerance
        if not candidate_feasible:
            candidate_slack = float(candidate.confidence["slack"])
            current_slack = float(current.confidence["slack"])
            scale = max(1.0, candidate_slack, current_slack)
            if candidate_slack < current_slack - tolerance * scale:
                return True
            if candidate_slack > current_slack + tolerance * scale:
                return False
        scale = max(1.0, candidate.regularizer, current.regularizer)
        return candidate.regularizer < current.regularizer - tolerance * scale

    def consider(
        self,
        *,
        rows: np.ndarray,
        answers: np.ndarray,
        combined_coefficient_residual: np.ndarray,
        regularizer: float,
        iteration: int,
    ) -> bool:
        records = np.asarray(rows, dtype=np.int32)
        answer_vector = np.asarray(answers, dtype=np.float32)
        residual = np.asarray(combined_coefficient_residual, dtype=np.float64)
        if records.ndim != 2 or len(records) == 0:
            raise ValueError("Streamwise incumbent rows must be a non-empty matrix")
        if answer_vector.ndim != 1 or not np.all(np.isfinite(answer_vector)):
            raise ValueError("Streamwise incumbent answers must be finite")
        if residual.shape != (self.confidence.canonical_dimension,):
            raise ValueError("Streamwise incumbent residual dimension mismatch")
        if not np.isfinite(regularizer) or regularizer < 0.0:
            raise ValueError("Streamwise incumbent regularizer must be nonnegative")
        evaluation = self.confidence.evaluate_combined_residual(
            residual,
            tolerance=self.feasibility_tolerance,
        )
        snapshot = StreamwiseRCEIntegerSnapshot(
            rows=records.copy(),
            answers=answer_vector.copy(),
            combined_coefficient_residual=residual.copy(),
            regularizer=float(regularizer),
            iteration=int(iteration),
            confidence=evaluation,
        )
        self.num_considered += 1
        if self._best is None or self._strictly_better(snapshot, self._best):
            self._best = snapshot
            self.num_improvements += 1
            return True
        return False

    def diagnostics(self) -> dict[str, Any]:
        best = self.best
        return {
            "method": "visited_integer_streamwise_slack_then_kl_v1",
            "scope": "qdte_visited_integer_tables",
            "globally_certified": False,
            "best_iteration": best.iteration,
            "best_regularizer": best.regularizer,
            "best_confidence": dict(best.confidence),
            "num_considered": self.num_considered,
            "num_improvements": self.num_improvements,
            "feasibility_tolerance": self.feasibility_tolerance,
            "comparison_tolerance": self.comparison_tolerance,
        }


@dataclass
class StreamwiseRCEQDTEController:
    precision: OrthogonalInteractionPrecision
    confidence: StreamwiseRCEConfidenceSet
    prior: RowReferencePrior
    entropy_state: AtomEntropyState
    dual: StreamwiseRCEDualState
    incumbent: StreamwiseRCEIntegerIncumbent
    initial_regularizer: float
    initial_confidence: dict[str, Any]
    combined_target_roundtrip_error: float
    method: str = STREAMWISE_RCE_CONTROLLER_METHOD

    @classmethod
    def create(
        cls,
        *,
        released_target: np.ndarray,
        initial_rows: np.ndarray,
        initial_answers: np.ndarray,
        initial_residual: np.ndarray,
        precision: OrthogonalInteractionPrecision,
        confidence: StreamwiseRCEConfidenceSet,
        prior: RowReferencePrior,
        max_iterations: int,
    ) -> StreamwiseRCEQDTEController:
        canonical_target = precision.coefficient_coordinates(released_target)
        roundtrip_error = float(
            np.max(np.abs(canonical_target - confidence.combined_target))
        )
        roundtrip_tolerance = 1.0e-5 * max(
            1.0,
            float(np.max(np.abs(confidence.combined_target), initial=0.0)),
        )
        if roundtrip_error > roundtrip_tolerance:
            raise ValueError("Streamwise RCE combined target differs from the QDTE target")
        # Measurements are serialized as float32 query answers while the
        # sequential Gaussian streams retain float64 coefficient centers.
        # The combined target is only the linear origin used to recover the
        # canonical answer from a query-space residual; align that origin to
        # the actual released QDTE target without changing any stream center.
        confidence = replace(confidence, combined_target=canonical_target)
        if precision.coefficient_dimension != confidence.canonical_dimension:
            raise ValueError("Streamwise RCE confidence dimension differs from precision")
        entropy_state = AtomEntropyState.from_rows(initial_rows, prior)
        regularizer = float(entropy_state.regularizer / entropy_state.n_rows)
        dual = StreamwiseRCEDualState.create(
            confidence,
            max_iterations=int(max_iterations),
        )
        incumbent = StreamwiseRCEIntegerIncumbent(confidence)
        coefficient_residual = precision.coefficient_coordinates(initial_residual)
        incumbent.consider(
            rows=initial_rows,
            answers=initial_answers,
            combined_coefficient_residual=coefficient_residual,
            regularizer=regularizer,
            iteration=0,
        )
        return cls(
            precision=precision,
            confidence=confidence,
            prior=prior,
            entropy_state=entropy_state,
            dual=dual,
            incumbent=incumbent,
            initial_regularizer=regularizer,
            initial_confidence=confidence.evaluate_combined_residual(
                coefficient_residual
            ),
            combined_target_roundtrip_error=roundtrip_error,
        )

    @property
    def regularizer(self) -> float:
        return float(self.entropy_state.regularizer / self.entropy_state.n_rows)

    def effective_lambda_cost(self, lambda_cost: float) -> float:
        return float(lambda_cost) / float(self.entropy_state.n_rows)

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
        coefficient_residual = self.coefficient_residual(query_residual)
        linear, quadratic = self.dual.coefficient_terms(coefficient_residual)
        scores, quadratic_values = (
            score_candidates_orthogonal_dynamic_diagonal_with_quadratic(
                candidates,
                linear,
                quadratic,
                self.precision,
                self.effective_lambda_cost(lambda_cost),
                chunk_size=int(chunk_size),
                context=context,
            )
        )
        regularizer = self.entropy_state.candidate_gains(
            candidates.old_rows,
            candidates.new_rows,
        ) / self.entropy_state.n_rows
        return (
            (np.asarray(scores, dtype=np.float64) + regularizer).astype(np.float32),
            quadratic_values,
        )

    def _prefix_terms(
        self,
        query_residual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.dual.coefficient_terms(self.coefficient_residual(query_residual))

    def prefix_advantages_from_rows(
        self,
        *,
        query_residual: np.ndarray,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        edit_cost_prefix: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        linear, quadratic = self._prefix_terms(query_residual)
        linear_values, quadratic_values = self.precision.prefix_linear_and_diagonal_quadratic(
            old_rows,
            new_rows,
            linear,
            quadratic,
        )
        regularizer = self.entropy_state.prefix_gains(old_rows, new_rows)
        return (
            linear_values
            - 0.5 * quadratic_values
            + regularizer / self.entropy_state.n_rows
            - self.effective_lambda_cost(lambda_cost)
            * np.asarray(edit_cost_prefix, dtype=np.float64)
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
        if context.prefix_scorer is None:
            raise ValueError("Streamwise GPU prefix scorer is unavailable")
        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        capacity = int(static_capacity)
        if len(old) == 0 or old.shape != new.shape or capacity < len(old):
            raise ValueError("Streamwise GPU prefix rows or capacity are invalid")
        padded_old = np.repeat(old[:1], capacity, axis=0)
        padded_new = padded_old.copy()
        padded_old[: len(old)] = old
        padded_new[: len(new)] = new
        linear, quadratic = self._prefix_terms(query_residual)
        local_linear, local_quadratic = context.prefix_scorer(
            padded_old,
            padded_new,
            np.asarray(linear, dtype=np.float32),
            np.asarray(quadratic, dtype=np.float32),
        )
        regularizer = self.entropy_state.prefix_gains(old, new)
        return (
            np.asarray(local_linear, dtype=np.float64)[: len(old)]
            - 0.5 * np.asarray(local_quadratic, dtype=np.float64)[: len(old)]
            + regularizer / self.entropy_state.n_rows
            - self.effective_lambda_cost(lambda_cost)
            * np.asarray(edit_cost_prefix, dtype=np.float64)
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
        regularizer_gain = float(
            self.entropy_state.batch_gain(old_rows, new_rows)
            / self.entropy_state.n_rows
        )
        value = self.dual.batch_gain(
            combined_residual=self.coefficient_residual(query_residual),
            canonical_delta_sum=self.precision.coefficient_coordinates(
                query_delta_sum
            ),
            regularizer_gain=regularizer_gain,
            edit_cost=float(edit_cost),
            lambda_cost=self.effective_lambda_cost(lambda_cost),
        )
        return value, regularizer_gain

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
            combined_coefficient_residual=self.coefficient_residual(query_residual),
            regularizer=self.regularizer,
            iteration=int(iteration),
        )

    def lagrangian_value(self, query_residual: np.ndarray) -> float:
        return self.dual.lagrangian(
            self.coefficient_residual(query_residual),
            self.regularizer,
        )

    def current_confidence(self, query_residual: np.ndarray) -> dict[str, Any]:
        return self.confidence.evaluate_combined_residual(
            self.coefficient_residual(query_residual)
        )

    def restore_incumbent(self) -> StreamwiseRCEIntegerSnapshot:
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
            "scoring_method": "streamwise_dynamic_diagonal_exact_v1",
            "confidence_set": self.confidence.to_public_dict(),
            "initial_confidence": dict(self.initial_confidence),
            "final_confidence": self.confidence.evaluate_combined_residual(
                coefficient_residual
            ),
            "prior": self.prior.diagnostics(),
            "state": self.entropy_state.diagnostics(),
            "regularizer_definition": "D_KL(p_empirical||p0)",
            "count_scaled_regularizer": float(self.entropy_state.regularizer),
            "dual": self.dual.diagnostics(),
            "certificate": self.dual.certificate(coefficient_residual),
            "integer_incumbent": self.incumbent.diagnostics(),
            "initial_regularizer": self.initial_regularizer,
            "final_regularizer": self.regularizer,
            "combined_target_roundtrip_error": self.combined_target_roundtrip_error,
        }


__all__ = [
    "STREAMWISE_RCE_CONTROLLER_METHOD",
    "StreamwiseRCEIntegerIncumbent",
    "StreamwiseRCEIntegerSnapshot",
    "StreamwiseRCEQDTEController",
]
