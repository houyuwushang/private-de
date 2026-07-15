from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet


@dataclass(frozen=True)
class RCELagrangianEvaluation:
    value: float
    regularizer: float
    ellipsoid_constraint: float
    tube_positive_max: float
    tube_negative_max: float
    ellipsoid_term: float
    tube_term: float

    def to_dict(self) -> dict[str, float]:
        return {
            "value": self.value,
            "regularizer": self.regularizer,
            "ellipsoid_constraint": self.ellipsoid_constraint,
            "tube_positive_max": self.tube_positive_max,
            "tube_negative_max": self.tube_negative_max,
            "ellipsoid_term": self.ellipsoid_term,
            "tube_term": self.tube_term,
        }


@dataclass
class RCEDualState:
    confidence: RCEConfidenceSet
    max_iterations: int
    ellipsoid_weight: float
    tube_positive: np.ndarray
    tube_negative: np.ndarray
    step_scale: float = 1.0
    dual_max: float = 1.0e8
    num_updates: int = 0

    @classmethod
    def create(
        cls,
        confidence: RCEConfidenceSet,
        *,
        max_iterations: int,
        step_scale: float = 1.0,
        dual_max: float = 1.0e8,
    ) -> RCEDualState:
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(float(step_scale)) or float(step_scale) <= 0.0:
            raise ValueError("step_scale must be finite and positive")
        if not math.isfinite(float(dual_max)) or float(dual_max) <= 0.0:
            raise ValueError("dual_max must be finite and positive")
        return cls(
            confidence=confidence,
            max_iterations=int(max_iterations),
            ellipsoid_weight=0.0,
            tube_positive=np.zeros(confidence.dimension, dtype=np.float64),
            tube_negative=np.zeros(confidence.dimension, dtype=np.float64),
            step_scale=float(step_scale),
            dual_max=float(dual_max),
        )

    def __post_init__(self) -> None:
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        for name, values in (
            ("tube_positive", self.tube_positive),
            ("tube_negative", self.tube_negative),
        ):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (self.confidence.dimension,):
                raise ValueError(f"{name} must match the confidence dimension")
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, array.copy())
        if not math.isfinite(float(self.ellipsoid_weight)) or self.ellipsoid_weight < 0.0:
            raise ValueError("ellipsoid_weight must be finite and nonnegative")
        if not math.isfinite(float(self.step_scale)) or self.step_scale <= 0.0:
            raise ValueError("step_scale must be finite and positive")
        if not math.isfinite(float(self.dual_max)) or self.dual_max <= 0.0:
            raise ValueError("dual_max must be finite and positive")
        self.tube_positive[~self.confidence.tube_mask] = 0.0
        self.tube_negative[~self.confidence.tube_mask] = 0.0

    @property
    def step_size(self) -> float:
        return self.step_scale / math.sqrt(float(self.max_iterations))

    def normalized_constraints(
        self,
        residual: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        vector = np.asarray(residual, dtype=np.float64)
        if vector.shape != (self.confidence.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError("residual must match the confidence dimension and be finite")
        ellipsoid = (
            self.confidence.squared_discrepancy(vector)
            / self.confidence.squared_discrepancy_threshold
            - 1.0
        )
        positive = np.full(vector.shape, -1.0, dtype=np.float64)
        negative = np.full(vector.shape, -1.0, dtype=np.float64)
        mask = self.confidence.tube_mask
        bounds = self.confidence.coordinate_bounds[mask]
        positive[mask] = vector[mask] / bounds - 1.0
        negative[mask] = -vector[mask] / bounds - 1.0
        return float(ellipsoid), positive, negative

    def update(self, residual: np.ndarray) -> dict[str, float | int]:
        ellipsoid, positive, negative = self.normalized_constraints(residual)
        step = self.step_size
        self.ellipsoid_weight = float(
            np.clip(self.ellipsoid_weight + step * ellipsoid, 0.0, self.dual_max)
        )
        mask = self.confidence.tube_mask
        self.tube_positive[mask] = np.clip(
            self.tube_positive[mask] + step * positive[mask],
            0.0,
            self.dual_max,
        )
        self.tube_negative[mask] = np.clip(
            self.tube_negative[mask] + step * negative[mask],
            0.0,
            self.dual_max,
        )
        self.num_updates += 1
        return {
            "ellipsoid_constraint": ellipsoid,
            "tube_positive_max": float(np.max(positive[mask])),
            "tube_negative_max": float(np.max(negative[mask])),
            "ellipsoid_weight": self.ellipsoid_weight,
            "active_positive_duals": int(np.sum(self.tube_positive[mask] > 0.0)),
            "active_negative_duals": int(np.sum(self.tube_negative[mask] > 0.0)),
        }

    def signed_tube_weights(self) -> np.ndarray:
        weights = np.zeros(self.confidence.dimension, dtype=np.float64)
        mask = self.confidence.tube_mask
        weights[mask] = (
            self.tube_positive[mask] - self.tube_negative[mask]
        ) / self.confidence.coordinate_bounds[mask]
        return weights

    def candidate_gains(
        self,
        *,
        residual: np.ndarray,
        deltas: np.ndarray,
        regularizer_gains: np.ndarray,
        edit_costs: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64)
        d = np.asarray(deltas, dtype=np.float64)
        regularizer = np.asarray(regularizer_gains, dtype=np.float64)
        costs = np.asarray(edit_costs, dtype=np.float64)
        if r.shape != (self.confidence.dimension,) or not np.all(np.isfinite(r)):
            raise ValueError("residual must match the confidence dimension")
        if d.ndim != 2 or d.shape[1] != self.confidence.dimension or not np.all(np.isfinite(d)):
            raise ValueError("deltas must be a finite matrix matching the confidence dimension")
        if regularizer.shape != (d.shape[0],) or costs.shape != (d.shape[0],):
            raise ValueError("regularizer_gains and edit_costs must match deltas")
        if not np.all(np.isfinite(regularizer)) or not np.all(np.isfinite(costs)):
            raise ValueError("regularizer_gains and edit_costs must be finite")
        if not math.isfinite(float(lambda_cost)) or float(lambda_cost) < 0.0:
            raise ValueError("lambda_cost must be finite and nonnegative")

        precision_residual = self.confidence.precision_matvec(r)
        precision_deltas = self.confidence.precision_matvec_many(d)
        ellipsoid_gain = (
            self.ellipsoid_weight
            / self.confidence.squared_discrepancy_threshold
            * (
                2.0 * (d @ precision_residual)
                - np.einsum("ij,ij->i", d, precision_deltas, optimize=True)
            )
        )
        tube_gain = d @ self.signed_tube_weights()
        return (
            regularizer
            + ellipsoid_gain
            + tube_gain
            - float(lambda_cost) * costs
        )

    def batch_gain(
        self,
        *,
        residual: np.ndarray,
        delta_sum: np.ndarray,
        regularizer_gain: float,
        edit_cost: float = 0.0,
        lambda_cost: float = 0.0,
    ) -> float:
        value = self.candidate_gains(
            residual=residual,
            deltas=np.asarray(delta_sum, dtype=np.float64).reshape(1, -1),
            regularizer_gains=np.asarray([regularizer_gain], dtype=np.float64),
            edit_costs=np.asarray([edit_cost], dtype=np.float64),
            lambda_cost=lambda_cost,
        )
        return float(value[0])

    def lagrangian(self, residual: np.ndarray, regularizer: float) -> RCELagrangianEvaluation:
        if not math.isfinite(float(regularizer)) or float(regularizer) < 0.0:
            raise ValueError("regularizer must be finite and nonnegative")
        ellipsoid, positive, negative = self.normalized_constraints(residual)
        mask = self.confidence.tube_mask
        ellipsoid_term = self.ellipsoid_weight * ellipsoid
        tube_term = float(
            self.tube_positive[mask] @ positive[mask]
            + self.tube_negative[mask] @ negative[mask]
        )
        return RCELagrangianEvaluation(
            value=float(regularizer + ellipsoid_term + tube_term),
            regularizer=float(regularizer),
            ellipsoid_constraint=ellipsoid,
            tube_positive_max=float(np.max(positive[mask])),
            tube_negative_max=float(np.max(negative[mask])),
            ellipsoid_term=float(ellipsoid_term),
            tube_term=tube_term,
        )

    def certificate(self, residual: np.ndarray) -> dict[str, Any]:
        ellipsoid, positive, negative = self.normalized_constraints(residual)
        mask = self.confidence.tube_mask
        primal_violation = max(
            0.0,
            ellipsoid,
            float(np.max(positive[mask])),
            float(np.max(negative[mask])),
        )
        complementarity = max(
            abs(self.ellipsoid_weight * ellipsoid),
            float(np.max(np.abs(self.tube_positive[mask] * positive[mask]))),
            float(np.max(np.abs(self.tube_negative[mask] * negative[mask]))),
        )
        dual_violation = max(
            0.0,
            -self.ellipsoid_weight,
            float(np.max(-self.tube_positive[mask])),
            float(np.max(-self.tube_negative[mask])),
        )
        evaluation = self.confidence.evaluate(residual)
        return {
            "primal_violation": float(primal_violation),
            "dual_violation": float(dual_violation),
            "complementarity_residual": float(complementarity),
            "stationarity_certified": False,
            "stationarity_reason": "row-atom restricted oracle required",
            "confidence": evaluation.to_dict(),
            "ellipsoid_weight": float(self.ellipsoid_weight),
            "active_positive_duals": int(np.sum(self.tube_positive[mask] > 0.0)),
            "active_negative_duals": int(np.sum(self.tube_negative[mask] > 0.0)),
            "dual_updates": int(self.num_updates),
            "dual_step_size": float(self.step_size),
        }

    def diagnostics(self) -> dict[str, Any]:
        mask = self.confidence.tube_mask
        return {
            "method": "rce_normalized_projected_dual_v1",
            "max_iterations": int(self.max_iterations),
            "step_scale": float(self.step_scale),
            "step_size": float(self.step_size),
            "dual_max": float(self.dual_max),
            "ellipsoid_weight": float(self.ellipsoid_weight),
            "active_positive_duals": int(np.sum(self.tube_positive[mask] > 0.0)),
            "active_negative_duals": int(np.sum(self.tube_negative[mask] > 0.0)),
            "max_positive_dual": float(np.max(self.tube_positive[mask])),
            "max_negative_dual": float(np.max(self.tube_negative[mask])),
            "dual_updates": int(self.num_updates),
        }
