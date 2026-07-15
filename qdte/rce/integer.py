from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet, RCEConstraintEvaluation


@dataclass(frozen=True)
class RCEIntegerSnapshot:
    rows: np.ndarray
    answers: np.ndarray
    coefficient_residual: np.ndarray
    regularizer: float
    iteration: int
    confidence: RCEConstraintEvaluation


class RCEIntegerIncumbent:
    """Lexicographic incumbent over the integer tables visited by QDTE."""

    def __init__(
        self,
        confidence: RCEConfidenceSet,
        *,
        feasibility_tolerance: float = 1.0e-9,
        comparison_tolerance: float = 1.0e-12,
    ) -> None:
        if feasibility_tolerance < 0.0 or not np.isfinite(feasibility_tolerance):
            raise ValueError("feasibility_tolerance must be finite and nonnegative")
        if comparison_tolerance < 0.0 or not np.isfinite(comparison_tolerance):
            raise ValueError("comparison_tolerance must be finite and nonnegative")
        self.confidence = confidence
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.comparison_tolerance = float(comparison_tolerance)
        self._best: RCEIntegerSnapshot | None = None
        self.num_considered = 0
        self.num_improvements = 0

    @property
    def best(self) -> RCEIntegerSnapshot:
        if self._best is None:
            raise RuntimeError("RCE integer incumbent has not observed a table")
        return self._best

    def _strictly_better(
        self,
        candidate: RCEIntegerSnapshot,
        current: RCEIntegerSnapshot,
    ) -> bool:
        candidate_feasible = candidate.confidence.inside
        current_feasible = current.confidence.inside
        if candidate_feasible != current_feasible:
            return candidate_feasible
        tolerance = self.comparison_tolerance
        if not candidate_feasible:
            slack_scale = max(1.0, candidate.confidence.slack, current.confidence.slack)
            slack_difference = candidate.confidence.slack - current.confidence.slack
            if slack_difference < -tolerance * slack_scale:
                return True
            if slack_difference > tolerance * slack_scale:
                return False
        regularizer_scale = max(1.0, candidate.regularizer, current.regularizer)
        return candidate.regularizer < current.regularizer - tolerance * regularizer_scale

    def consider(
        self,
        *,
        rows: np.ndarray,
        answers: np.ndarray,
        coefficient_residual: np.ndarray,
        regularizer: float,
        iteration: int,
    ) -> bool:
        records = np.asarray(rows, dtype=np.int32)
        answer_vector = np.asarray(answers, dtype=np.float32)
        residual = np.asarray(coefficient_residual, dtype=np.float64)
        if records.ndim != 2 or records.shape[0] <= 0:
            raise ValueError("rows must be a non-empty matrix")
        if answer_vector.ndim != 1 or not np.all(np.isfinite(answer_vector)):
            raise ValueError("answers must be a finite vector")
        if residual.shape != (self.confidence.dimension,) or not np.all(np.isfinite(residual)):
            raise ValueError("coefficient_residual must match the confidence dimension")
        if not np.isfinite(regularizer) or regularizer < 0.0:
            raise ValueError("regularizer must be finite and nonnegative")
        if int(iteration) < 0:
            raise ValueError("iteration must be nonnegative")
        evaluation = self.confidence.evaluate(
            residual,
            tolerance=self.feasibility_tolerance,
        )
        snapshot = RCEIntegerSnapshot(
            rows=records.copy(),
            answers=answer_vector.copy(),
            coefficient_residual=residual.copy(),
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
            "method": "visited_integer_lexicographic_slack_then_kl_v1",
            "scope": "qdte_visited_integer_tables",
            "globally_certified": False,
            "best_iteration": best.iteration,
            "best_regularizer": best.regularizer,
            "best_kl_per_row": best.regularizer,
            "regularizer_definition": "D_KL(p_empirical||p0)",
            "best_confidence": best.confidence.to_dict(),
            "num_considered": self.num_considered,
            "num_improvements": self.num_improvements,
            "feasibility_tolerance": self.feasibility_tolerance,
            "comparison_tolerance": self.comparison_tolerance,
        }


__all__ = ["RCEIntegerIncumbent", "RCEIntegerSnapshot"]
