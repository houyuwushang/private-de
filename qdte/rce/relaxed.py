from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qdte.rce.confidence_set import RCEConfidenceSet


@dataclass(frozen=True)
class RelaxedRCEResult:
    probabilities: np.ndarray
    synthetic_answer: np.ndarray
    residual: np.ndarray
    slack_star: float
    kl_objective: float
    confidence: dict[str, Any]
    stage_one: dict[str, Any]
    stage_two: dict[str, Any]
    support_kind: str
    support_size: int
    globally_certified: bool
    unique_on_declared_support: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities.tolist(),
            "synthetic_answer": self.synthetic_answer.tolist(),
            "residual": self.residual.tolist(),
            "slack_star": self.slack_star,
            "kl_objective": self.kl_objective,
            "confidence": self.confidence,
            "stage_one": self.stage_one,
            "stage_two": self.stage_two,
            "support_kind": self.support_kind,
            "support_size": self.support_size,
            "globally_certified": self.globally_certified,
            "unique_on_declared_support": self.unique_on_declared_support,
        }


def _solver_record(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "function_evaluations": int(getattr(result, "nfev", -1)),
        "jacobian_evaluations": int(getattr(result, "njev", -1)),
        "objective": float(result.fun),
    }


def solve_relaxed_rce(
    row_features: np.ndarray,
    target: np.ndarray,
    prior_probabilities: np.ndarray,
    confidence: RCEConfidenceSet,
    *,
    support_kind: str = "full_enumerated_domain",
    globally_certified: bool = True,
    max_iterations: int = 5_000,
    ftol: float = 1.0e-12,
    optimal_face_tolerance: float = 1.0e-10,
    feasibility_tolerance: float = 1.0e-9,
) -> RelaxedRCEResult:
    """Solve exact or restricted relaxed RCE over a declared finite atom set."""

    features = np.asarray(row_features, dtype=np.float64)
    released = np.asarray(target, dtype=np.float64)
    prior = np.asarray(prior_probabilities, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("row_features must be a non-empty matrix")
    if features.shape[1] != confidence.dimension:
        raise ValueError("row_features width must match the confidence dimension")
    if released.shape != (confidence.dimension,) or not np.all(np.isfinite(released)):
        raise ValueError("target must be finite and match the confidence dimension")
    if prior.shape != (features.shape[0],):
        raise ValueError("prior_probabilities must contain one value per atom")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(prior)):
        raise ValueError("row_features and prior_probabilities must be finite")
    if np.any(prior <= 0.0) or float(np.sum(prior)) > 1.0 + 1.0e-9:
        raise ValueError("prior_probabilities must be positive with total mass at most one")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(float(ftol)) or float(ftol) <= 0.0:
        raise ValueError("ftol must be finite and positive")
    if not math.isfinite(float(optimal_face_tolerance)) or optimal_face_tolerance < 0.0:
        raise ValueError("optimal_face_tolerance must be finite and nonnegative")
    if not math.isfinite(float(feasibility_tolerance)) or feasibility_tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be finite and nonnegative")

    num_atoms = features.shape[0]
    normalized_prior = prior / float(np.sum(prior))
    tube_indices = np.flatnonzero(confidence.tube_mask)
    bounds = confidence.coordinate_bounds[tube_indices]

    def answer(probabilities: np.ndarray) -> np.ndarray:
        return probabilities @ features

    def residual(probabilities: np.ndarray) -> np.ndarray:
        return released - answer(probabilities)

    def stage_one_constraints(values: np.ndarray) -> np.ndarray:
        probabilities = values[:-1]
        slack = float(values[-1])
        current = residual(probabilities)
        ellipsoid = (
            1.0
            + slack
            - confidence.squared_discrepancy(current)
            / confidence.squared_discrepancy_threshold
        )
        local = current[tube_indices]
        positive = bounds * (1.0 + slack) - local
        negative = bounds * (1.0 + slack) + local
        return np.concatenate(([ellipsoid], positive, negative))

    def stage_one_jacobian(values: np.ndarray) -> np.ndarray:
        probabilities = values[:-1]
        current = residual(probabilities)
        precision_residual = confidence.precision_matvec(current)
        jacobian = np.zeros((1 + 2 * len(tube_indices), num_atoms + 1), dtype=np.float64)
        jacobian[0, :-1] = (
            2.0
            * (features @ precision_residual)
            / confidence.squared_discrepancy_threshold
        )
        jacobian[0, -1] = 1.0
        jacobian[1 : 1 + len(tube_indices), :-1] = features[:, tube_indices].T
        jacobian[1 : 1 + len(tube_indices), -1] = bounds
        jacobian[1 + len(tube_indices) :, :-1] = -features[:, tube_indices].T
        jacobian[1 + len(tube_indices) :, -1] = bounds
        return jacobian

    initial_slack = confidence.evaluate(residual(normalized_prior)).slack
    initial_stage_one = np.concatenate((normalized_prior, [initial_slack]))
    stage_one = minimize(
        lambda values: float(values[-1]),
        initial_stage_one,
        jac=lambda values: np.concatenate((np.zeros(num_atoms), [1.0])),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * num_atoms + [(0.0, None)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda values: float(np.sum(values[:-1]) - 1.0),
                "jac": lambda values: np.concatenate((np.ones(num_atoms), [0.0])),
            },
            {
                "type": "ineq",
                "fun": stage_one_constraints,
                "jac": stage_one_jacobian,
            },
        ],
        options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
    )
    stage_one_probabilities = np.asarray(stage_one.x[:-1], dtype=np.float64)
    stage_one_probabilities = np.maximum(stage_one_probabilities, 0.0)
    stage_one_probabilities /= float(np.sum(stage_one_probabilities))
    measured_slack = confidence.evaluate(residual(stage_one_probabilities)).slack
    slack_star = max(0.0, float(stage_one.x[-1]), measured_slack)
    face_slack = slack_star + float(optimal_face_tolerance)

    probability_floor = min(1.0e-14, 0.1 / float(num_atoms))
    stage_two_initial = np.maximum(stage_one_probabilities, probability_floor)
    stage_two_initial /= float(np.sum(stage_two_initial))

    def kl_objective(probabilities: np.ndarray) -> float:
        values = np.asarray(probabilities, dtype=np.float64)
        return float(np.sum(values * np.log(values / prior), dtype=np.float64))

    def kl_gradient(probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        return np.log(values / prior) + 1.0

    def stage_two_constraints(probabilities: np.ndarray) -> np.ndarray:
        values = np.concatenate((np.asarray(probabilities, dtype=np.float64), [face_slack]))
        return stage_one_constraints(values)

    def stage_two_jacobian(probabilities: np.ndarray) -> np.ndarray:
        values = np.concatenate((np.asarray(probabilities, dtype=np.float64), [face_slack]))
        return stage_one_jacobian(values)[:, :-1]

    stage_two = minimize(
        kl_objective,
        stage_two_initial,
        jac=kl_gradient,
        method="SLSQP",
        bounds=[(probability_floor, 1.0)] * num_atoms,
        constraints=[
            {
                "type": "eq",
                "fun": lambda values: float(np.sum(values) - 1.0),
                "jac": lambda values: np.ones(num_atoms, dtype=np.float64),
            },
            {
                "type": "ineq",
                "fun": stage_two_constraints,
                "jac": stage_two_jacobian,
            },
        ],
        options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
    )
    probabilities = np.asarray(stage_two.x, dtype=np.float64)
    probabilities = np.maximum(probabilities, probability_floor)
    probabilities /= float(np.sum(probabilities))
    synthetic = answer(probabilities)
    final_residual = released - synthetic
    evaluation = confidence.evaluate(
        final_residual,
        tolerance=float(feasibility_tolerance),
    )
    unique = bool(stage_two.success and np.all(prior > 0.0))
    return RelaxedRCEResult(
        probabilities=probabilities,
        synthetic_answer=synthetic,
        residual=final_residual,
        slack_star=max(slack_star, evaluation.slack),
        kl_objective=kl_objective(probabilities),
        confidence=evaluation.to_dict(),
        stage_one=_solver_record(stage_one),
        stage_two=_solver_record(stage_two),
        support_kind=str(support_kind),
        support_size=int(num_atoms),
        globally_certified=bool(
            globally_certified
            and stage_one.success
            and stage_two.success
            and evaluation.inside
        ),
        unique_on_declared_support=unique,
    )
