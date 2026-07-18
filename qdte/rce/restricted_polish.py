from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.special import logsumexp

from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.relaxed import _restricted_supporting_hyperplane_certificate


@dataclass(frozen=True)
class PolishedRestrictedMixture:
    component_weights: np.ndarray
    mixture_probabilities: np.ndarray
    residual: np.ndarray
    kl_objective: float
    confidence: dict[str, Any]
    certificate: dict[str, Any]
    diagnostics: dict[str, Any]


def polish_interior_restricted_mixture(
    component_probabilities: np.ndarray,
    component_residuals: np.ndarray,
    log_prior_probabilities: np.ndarray,
    confidence: RCEConfidenceSet,
    initial_weights: np.ndarray,
    *,
    face_slack: float,
    weight_floor: float = 1.0e-12,
    maximum_iterations: int = 100,
    feasibility_tolerance: float = 1.0e-9,
    interior_margin: float = 1.0e-7,
) -> PolishedRestrictedMixture:
    """Polish an interior finite-hull KL solution with constrained Newton.

    The Newton system enforces the simplex equality. A backtracking line
    search preserves the floored simplex and every confidence inequality. If
    the starting point is not a strict confidence-set interior point, the
    routine returns it unchanged; active confidence constraints require a
    separate constrained solver and are never silently approximated here.
    """

    probabilities = np.asarray(component_probabilities, dtype=np.float64)
    residuals = np.asarray(component_residuals, dtype=np.float64)
    log_prior = np.asarray(log_prior_probabilities, dtype=np.float64)
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    if probabilities.ndim != 2 or residuals.shape[0] != probabilities.shape[0]:
        raise ValueError("Component probabilities and residuals must align")
    if residuals.shape[1] != confidence.dimension:
        raise ValueError("Component residuals must match confidence dimension")
    if log_prior.shape != (probabilities.shape[1],):
        raise ValueError("log_prior_probabilities must match component support")
    if weights.shape != (probabilities.shape[0],):
        raise ValueError("initial_weights must match component count")
    floor = float(weight_floor)
    if floor <= 0.0 or floor * len(weights) >= 1.0:
        raise ValueError("weight_floor must leave a non-empty simplex")
    if int(maximum_iterations) <= 0:
        raise ValueError("maximum_iterations must be positive")
    weights = np.maximum(weights, floor)
    weights /= float(np.sum(weights))

    tube_indices = np.flatnonzero(confidence.tube_mask)
    bounds = confidence.coordinate_bounds[tube_indices]

    def mixture(local_weights: np.ndarray) -> np.ndarray:
        return np.asarray(local_weights, dtype=np.float64) @ probabilities

    def residual(local_weights: np.ndarray) -> np.ndarray:
        return np.asarray(local_weights, dtype=np.float64) @ residuals

    def objective(local_weights: np.ndarray) -> float:
        mixed = mixture(local_weights)
        positive = mixed > 0.0
        return float(
            np.sum(
                mixed[positive]
                * (np.log(mixed[positive]) - log_prior[positive]),
                dtype=np.float64,
            )
        )

    def gradient(local_weights: np.ndarray) -> np.ndarray:
        mixed = np.maximum(mixture(local_weights), np.finfo(np.float64).tiny)
        return probabilities @ (np.log(mixed) - log_prior + 1.0)

    def confidence_constraints(local_weights: np.ndarray) -> np.ndarray:
        current = residual(local_weights)
        ellipsoid = (
            1.0
            + float(face_slack)
            - confidence.squared_discrepancy(current)
            / confidence.squared_discrepancy_threshold
        )
        local = current[tube_indices]
        positive = bounds * (1.0 + float(face_slack)) - local
        negative = bounds * (1.0 + float(face_slack)) + local
        return np.concatenate(([ellipsoid], positive, negative))

    def confidence_jacobian(local_weights: np.ndarray) -> np.ndarray:
        current = residual(local_weights)
        precision_residual = confidence.precision_matvec(current)
        jacobian = np.zeros(
            (1 + 2 * len(tube_indices), len(weights)),
            dtype=np.float64,
        )
        jacobian[0] = (
            -2.0
            * (residuals @ precision_residual)
            / confidence.squared_discrepancy_threshold
        )
        jacobian[1 : 1 + len(tube_indices)] = -residuals[:, tube_indices].T
        jacobian[1 + len(tube_indices) :] = residuals[:, tube_indices].T
        return jacobian

    initial_objective = objective(weights)
    initial_constraints = confidence_constraints(weights)
    strictly_interior = bool(
        float(np.min(initial_constraints)) > float(interior_margin)
    )
    iterations = 0
    accepted_steps = 0
    if strictly_interior:
        for iteration in range(int(maximum_iterations)):
            iterations = iteration + 1
            mixed = np.maximum(mixture(weights), np.finfo(np.float64).tiny)
            local_gradient = probabilities @ (np.log(mixed) - log_prior + 1.0)
            hessian = (probabilities / mixed.reshape(1, -1)) @ probabilities.T
            hessian = 0.5 * (hessian + hessian.T)

            free = weights > floor + 1.0e-10
            if np.sum(free) < 2:
                break
            free_indices = np.flatnonzero(free)
            free_hessian = hessian[np.ix_(free_indices, free_indices)]
            free_gradient = local_gradient[free_indices]
            kkt = np.block(
                [
                    [
                        free_hessian,
                        np.ones((len(free_indices), 1), dtype=np.float64),
                    ],
                    [
                        np.ones((1, len(free_indices)), dtype=np.float64),
                        np.zeros((1, 1), dtype=np.float64),
                    ],
                ]
            )
            rhs = np.concatenate((-free_gradient, [0.0]))
            solution = np.linalg.lstsq(kkt, rhs, rcond=1.0e-12)[0]
            direction = np.zeros_like(weights)
            direction[free_indices] = solution[:-1]
            directional_derivative = float(local_gradient @ direction)
            if (
                float(np.max(np.abs(direction))) <= 1.0e-13
                or directional_derivative >= -1.0e-14
            ):
                break

            negative = direction < 0.0
            maximum_step = 1.0
            if np.any(negative):
                maximum_step = min(
                    maximum_step,
                    0.99
                    * float(
                        np.min(
                            (weights[negative] - floor) / (-direction[negative])
                        )
                    ),
                )
            step = max(0.0, maximum_step)
            current_objective = objective(weights)
            accepted = False
            for _ in range(60):
                candidate = weights + step * direction
                if (
                    step > 0.0
                    and float(np.min(candidate)) >= floor
                    and abs(float(np.sum(candidate)) - 1.0) <= 1.0e-10
                    and float(np.min(confidence_constraints(candidate)))
                    >= -float(feasibility_tolerance)
                    and objective(candidate)
                    <= current_objective + 1.0e-4 * step * directional_derivative
                ):
                    weights = candidate
                    accepted = True
                    accepted_steps += 1
                    break
                step *= 0.5
            if not accepted:
                break

    final_objective = objective(weights)
    final_residual = residual(weights)
    constraint_values = -confidence_constraints(weights)
    constraint_jacobian = -confidence_jacobian(weights)
    certificate = _restricted_supporting_hyperplane_certificate(
        weights=weights,
        weight_floor=floor,
        objective=final_objective,
        objective_gradient=gradient(weights),
        unconstrained_objective_lower_bound=-float(logsumexp(log_prior)),
        constraint_values=constraint_values,
        constraint_jacobian=constraint_jacobian,
    )
    confidence_evaluation = confidence.evaluate(
        final_residual,
        tolerance=float(feasibility_tolerance),
    )
    if final_objective > initial_objective + 1.0e-10:
        raise RuntimeError("Restricted Newton polishing increased the KL objective")
    if not confidence_evaluation.inside:
        raise RuntimeError("Restricted Newton polishing left the confidence set")
    return PolishedRestrictedMixture(
        component_weights=weights,
        mixture_probabilities=mixture(weights),
        residual=final_residual,
        kl_objective=final_objective,
        confidence=confidence_evaluation.to_dict(),
        certificate=certificate,
        diagnostics={
            "method": "interior_equality_constrained_newton_v1",
            "strictly_interior_at_start": strictly_interior,
            "iterations": iterations,
            "accepted_steps": accepted_steps,
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            "objective_improvement": initial_objective - final_objective,
            "initial_minimum_confidence_margin": float(
                np.min(initial_constraints)
            ),
            "final_minimum_confidence_margin": float(
                np.min(confidence_constraints(weights))
            ),
        },
    )
