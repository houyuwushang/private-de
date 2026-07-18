from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from qdte.rce.streamwise import StreamwiseRCEConfidenceSet


STREAMWISE_DUAL_METHOD = "streamwise_normalized_projected_dual_v1"


@dataclass
class StreamwiseRCEDualState:
    confidence: StreamwiseRCEConfidenceSet
    max_iterations: int
    ellipsoid_weights: np.ndarray
    tube_positive_by_stream: tuple[np.ndarray, ...]
    tube_negative_by_stream: tuple[np.ndarray, ...]
    step_scale: float = 1.0
    dual_max: float = 1.0e8
    num_updates: int = 0
    method: str = STREAMWISE_DUAL_METHOD

    @classmethod
    def create(
        cls,
        confidence: StreamwiseRCEConfidenceSet,
        *,
        max_iterations: int,
        step_scale: float = 1.0,
        dual_max: float = 1.0e8,
    ) -> StreamwiseRCEDualState:
        return cls(
            confidence=confidence,
            max_iterations=int(max_iterations),
            ellipsoid_weights=np.zeros(len(confidence.streams), dtype=np.float64),
            tube_positive_by_stream=tuple(
                np.zeros(stream.dimension, dtype=np.float64)
                for stream in confidence.streams
            ),
            tube_negative_by_stream=tuple(
                np.zeros(stream.dimension, dtype=np.float64)
                for stream in confidence.streams
            ),
            step_scale=float(step_scale),
            dual_max=float(dual_max),
        )

    def __post_init__(self) -> None:
        if self.method != STREAMWISE_DUAL_METHOD:
            raise ValueError(f"Unsupported streamwise dual method {self.method!r}")
        if int(self.max_iterations) <= 0:
            raise ValueError("Streamwise dual max_iterations must be positive")
        if not math.isfinite(self.step_scale) or self.step_scale <= 0.0:
            raise ValueError("Streamwise dual step_scale must be finite and positive")
        if not math.isfinite(self.dual_max) or self.dual_max <= 0.0:
            raise ValueError("Streamwise dual_max must be finite and positive")
        weights = np.asarray(self.ellipsoid_weights, dtype=np.float64)
        if weights.shape != (len(self.confidence.streams),):
            raise ValueError("Streamwise ellipsoid weights must match the streams")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Streamwise ellipsoid weights must be finite and nonnegative")
        if len(self.tube_positive_by_stream) != len(self.confidence.streams) or len(
            self.tube_negative_by_stream
        ) != len(self.confidence.streams):
            raise ValueError("Streamwise tube multipliers must match the streams")
        positive: list[np.ndarray] = []
        negative: list[np.ndarray] = []
        for stream, pos, neg in zip(
            self.confidence.streams,
            self.tube_positive_by_stream,
            self.tube_negative_by_stream,
            strict=True,
        ):
            local_pos = np.asarray(pos, dtype=np.float64).copy()
            local_neg = np.asarray(neg, dtype=np.float64).copy()
            if local_pos.shape != (stream.dimension,) or local_neg.shape != (
                stream.dimension,
            ):
                raise ValueError("Streamwise tube multiplier dimension mismatch")
            if (
                not np.all(np.isfinite(local_pos))
                or not np.all(np.isfinite(local_neg))
                or np.any(local_pos < 0.0)
                or np.any(local_neg < 0.0)
            ):
                raise ValueError("Streamwise tube multipliers must be finite and nonnegative")
            positive.append(local_pos)
            negative.append(local_neg)
        self.ellipsoid_weights = weights.copy()
        self.tube_positive_by_stream = tuple(positive)
        self.tube_negative_by_stream = tuple(negative)

    @property
    def step_size(self) -> float:
        return self.step_scale / math.sqrt(float(self.max_iterations))

    @property
    def ellipsoid_weight(self) -> float:
        return float(np.sum(self.ellipsoid_weights))

    @property
    def tube_positive(self) -> np.ndarray:
        result = np.zeros(self.confidence.canonical_dimension, dtype=np.float64)
        for stream, local in zip(
            self.confidence.streams,
            self.tube_positive_by_stream,
            strict=True,
        ):
            result[stream.coefficient_indices] += local
        return result

    @property
    def tube_negative(self) -> np.ndarray:
        result = np.zeros(self.confidence.canonical_dimension, dtype=np.float64)
        for stream, local in zip(
            self.confidence.streams,
            self.tube_negative_by_stream,
            strict=True,
        ):
            result[stream.coefficient_indices] += local
        return result

    def _canonical_answer(self, combined_residual: np.ndarray) -> np.ndarray:
        return self.confidence.canonical_answer_from_combined_residual(
            combined_residual
        )

    def normalized_constraints(
        self,
        combined_residual: np.ndarray,
    ) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
        answer = self._canonical_answer(combined_residual)
        records: list[tuple[float, np.ndarray, np.ndarray]] = []
        for stream in self.confidence.streams:
            residual = stream.residual(answer)
            local = stream.confidence
            ellipsoid = (
                local.squared_discrepancy(residual)
                / local.squared_discrepancy_threshold
                - 1.0
            )
            bounds = local.coordinate_bounds
            positive = residual / bounds - 1.0
            negative = -residual / bounds - 1.0
            records.append((float(ellipsoid), positive, negative))
        return tuple(records)

    def update(self, combined_residual: np.ndarray) -> dict[str, float | int]:
        constraints = self.normalized_constraints(combined_residual)
        step = self.step_size
        for index, (ellipsoid, positive, negative) in enumerate(constraints):
            self.ellipsoid_weights[index] = np.clip(
                self.ellipsoid_weights[index] + step * ellipsoid,
                0.0,
                self.dual_max,
            )
            self.tube_positive_by_stream[index][:] = np.clip(
                self.tube_positive_by_stream[index] + step * positive,
                0.0,
                self.dual_max,
            )
            self.tube_negative_by_stream[index][:] = np.clip(
                self.tube_negative_by_stream[index] + step * negative,
                0.0,
                self.dual_max,
            )
        self.num_updates += 1
        return {
            "ellipsoid_constraint": max(value[0] for value in constraints),
            "tube_positive_max": max(float(np.max(value[1])) for value in constraints),
            "tube_negative_max": max(float(np.max(value[2])) for value in constraints),
            "ellipsoid_weight": self.ellipsoid_weight,
            "active_positive_duals": int(np.sum(self.tube_positive > 0.0)),
            "active_negative_duals": int(np.sum(self.tube_negative > 0.0)),
            "num_streams": len(constraints),
        }

    def coefficient_terms(
        self,
        combined_residual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        answer = self._canonical_answer(combined_residual)
        linear = np.zeros(self.confidence.canonical_dimension, dtype=np.float64)
        quadratic = np.zeros(self.confidence.canonical_dimension, dtype=np.float64)
        for index, stream in enumerate(self.confidence.streams):
            residual = stream.residual(answer)
            local_confidence = stream.confidence
            precision = local_confidence.precision_diagonal
            if precision is None:
                raise ValueError("CDWF streamwise scoring requires diagonal stream precision")
            scale = (
                self.ellipsoid_weights[index]
                / local_confidence.squared_discrepancy_threshold
            )
            local_linear = 2.0 * scale * precision * residual
            local_linear += (
                self.tube_positive_by_stream[index]
                - self.tube_negative_by_stream[index]
            ) / local_confidence.coordinate_bounds
            local_quadratic = 2.0 * scale * precision
            linear[stream.coefficient_indices] += local_linear
            quadratic[stream.coefficient_indices] += local_quadratic
        return linear, quadratic

    def candidate_gains(
        self,
        *,
        combined_residual: np.ndarray,
        canonical_deltas: np.ndarray,
        regularizer_gains: np.ndarray,
        edit_costs: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        deltas = np.asarray(canonical_deltas, dtype=np.float64)
        regularizer = np.asarray(regularizer_gains, dtype=np.float64)
        costs = np.asarray(edit_costs, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.confidence.canonical_dimension:
            raise ValueError("Canonical deltas must match the streamwise dimension")
        if regularizer.shape != (len(deltas),) or costs.shape != (len(deltas),):
            raise ValueError("Streamwise regularizer and edit costs must match deltas")
        linear, quadratic = self.coefficient_terms(combined_residual)
        return (
            deltas @ linear
            - 0.5 * ((deltas * deltas) @ quadratic)
            + regularizer
            - float(lambda_cost) * costs
        )

    def batch_gain(
        self,
        *,
        combined_residual: np.ndarray,
        canonical_delta_sum: np.ndarray,
        regularizer_gain: float,
        edit_cost: float,
        lambda_cost: float,
    ) -> float:
        result = self.candidate_gains(
            combined_residual=combined_residual,
            canonical_deltas=np.asarray(canonical_delta_sum, dtype=np.float64).reshape(1, -1),
            regularizer_gains=np.asarray([regularizer_gain], dtype=np.float64),
            edit_costs=np.asarray([edit_cost], dtype=np.float64),
            lambda_cost=float(lambda_cost),
        )
        return float(result[0])

    def lagrangian(self, combined_residual: np.ndarray, regularizer: float) -> float:
        constraints = self.normalized_constraints(combined_residual)
        value = float(regularizer)
        for index, (ellipsoid, positive, negative) in enumerate(constraints):
            value += self.ellipsoid_weights[index] * ellipsoid
            value += float(self.tube_positive_by_stream[index] @ positive)
            value += float(self.tube_negative_by_stream[index] @ negative)
        return value

    def certificate(self, combined_residual: np.ndarray) -> dict[str, Any]:
        constraints = self.normalized_constraints(combined_residual)
        primal = 0.0
        complementarity = 0.0
        for index, (ellipsoid, positive, negative) in enumerate(constraints):
            primal = max(
                primal,
                ellipsoid,
                float(np.max(positive)),
                float(np.max(negative)),
            )
            complementarity = max(
                complementarity,
                abs(self.ellipsoid_weights[index] * ellipsoid),
                float(
                    np.max(
                        np.abs(self.tube_positive_by_stream[index] * positive)
                    )
                ),
                float(
                    np.max(
                        np.abs(self.tube_negative_by_stream[index] * negative)
                    )
                ),
            )
        evaluation = self.confidence.evaluate_combined_residual(combined_residual)
        return {
            "primal_violation": max(0.0, primal),
            "dual_violation": 0.0,
            "complementarity_residual": complementarity,
            "stationarity_certified": False,
            "stationarity_reason": "row-atom restricted oracle required",
            "confidence": evaluation,
            "num_streams": len(self.confidence.streams),
            "dual_updates": self.num_updates,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "max_iterations": self.max_iterations,
            "step_scale": self.step_scale,
            "step_size": self.step_size,
            "dual_max": self.dual_max,
            "ellipsoid_weights": self.ellipsoid_weights.tolist(),
            "ellipsoid_weight": self.ellipsoid_weight,
            "active_positive_duals": int(np.sum(self.tube_positive > 0.0)),
            "active_negative_duals": int(np.sum(self.tube_negative > 0.0)),
            "dual_updates": self.num_updates,
        }


__all__ = ["STREAMWISE_DUAL_METHOD", "StreamwiseRCEDualState"]
