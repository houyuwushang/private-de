from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    least_squares,
    linprog,
    minimize,
)
from scipy.special import logsumexp

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


@dataclass(frozen=True)
class RestrictedMixtureRCEResult:
    component_weights: np.ndarray
    mixture_probabilities: np.ndarray
    residual: np.ndarray
    slack_star: float
    kl_objective: float
    confidence: dict[str, Any]
    stage_one: dict[str, Any]
    stage_two: dict[str, Any]
    component_names: tuple[str, ...]
    support_size: int
    support_kind: str
    restricted_dual_certificate: dict[str, Any]
    globally_certified: bool = False

    def to_dict(self, *, include_mixture_probabilities: bool = False) -> dict[str, Any]:
        certificate = self.restricted_dual_certificate
        payload: dict[str, Any] = {
            "component_weights": {
                name: float(weight)
                for name, weight in zip(
                    self.component_names,
                    self.component_weights,
                    strict=True,
                )
            },
            "residual": self.residual.tolist(),
            "slack_star": self.slack_star,
            "kl_objective": self.kl_objective,
            "confidence": self.confidence,
            "stage_one": self.stage_one,
            "stage_two": self.stage_two,
            "component_names": list(self.component_names),
            "support_size": self.support_size,
            "support_kind": self.support_kind,
            "restricted_dual_certificate": self.restricted_dual_certificate,
            "globally_certified": self.globally_certified,
            "exact_on_declared_convex_hull": bool(
                self.stage_one["success"]
                and self.stage_two["success"]
                and certificate.get("success")
                and certificate.get("relative_primal_dual_gap", math.inf) <= 1.0e-7
                and certificate.get("maximum_constraint_violation", math.inf)
                <= 1.0e-8
            ),
        }
        if include_mixture_probabilities:
            payload["mixture_probabilities"] = self.mixture_probabilities.tolist()
        return payload


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


def _restricted_supporting_hyperplane_certificate(
    *,
    weights: np.ndarray,
    weight_floor: float,
    objective: float,
    objective_gradient: np.ndarray,
    unconstrained_objective_lower_bound: float,
    constraint_values: np.ndarray,
    constraint_jacobian: np.ndarray,
    floor_active_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Certify a lower bound on the declared finite convex-hull problem.

    The stage-two problem has convex objective ``f`` and convex inequalities
    ``h_j <= 0`` over the floored simplex. For any nonnegative multipliers,
    supporting hyperplanes of ``f + lambda @ h`` give an affine lower bound
    whose minimum over that simplex is analytic. Maximizing that bound is a
    small linear program. The certificate is restricted to the supplied
    components; it is not a pricing certificate over all legal rows.
    """

    current = np.asarray(weights, dtype=np.float64)
    gradient = np.asarray(objective_gradient, dtype=np.float64)
    values = np.asarray(constraint_values, dtype=np.float64)
    jacobian = np.asarray(constraint_jacobian, dtype=np.float64)
    if current.ndim != 1 or gradient.shape != current.shape:
        raise ValueError("weights and objective_gradient must be matching vectors")
    if values.ndim != 1 or jacobian.shape != (len(values), len(current)):
        raise ValueError("constraint Jacobian must match values and weights")
    declared_floor_active: np.ndarray | None = None
    if floor_active_mask is not None:
        declared_floor_active = np.asarray(floor_active_mask, dtype=bool)
        if declared_floor_active.shape != current.shape:
            raise ValueError("floor_active_mask must match the mixture weights")

    num_components = len(current)
    floor = float(weight_floor)
    remaining_mass = 1.0 - floor * num_components
    if remaining_mass <= 0.0:
        raise ValueError("weight_floor leaves no free simplex mass")

    # Let g(lambda) = grad f(x) + J_h(x)^T lambda. The minimum of the
    # supporting affine function over {w >= floor, 1^T w = 1} puts all
    # remaining mass on a minimum coordinate of g. Variable t represents
    # min_i g_i, turning maximization of the lower bound into an LP.
    lambda_coefficients = (
        values
        - jacobian @ current
        + floor * np.sum(jacobian, axis=1)
    )
    constant = (
        float(objective)
        - float(gradient @ current)
        + floor * float(np.sum(gradient))
    )
    objective_coefficients = np.concatenate(
        (-lambda_coefficients, [-remaining_mass])
    )
    inequalities = np.zeros(
        (num_components, len(values) + 1),
        dtype=np.float64,
    )
    inequalities[:, : len(values)] = -jacobian.T
    inequalities[:, -1] = 1.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*")
        lp = linprog(
            objective_coefficients,
            A_ub=inequalities,
            b_ub=gradient,
            bounds=[(0.0, None)] * len(values) + [(None, None)],
            method="highs",
            options={"threads": 1, "random_seed": 0},
        )
    if not lp.success:
        return {
            "certificate_kind": "restricted_supporting_hyperplane_dual",
            "support_scope": "declared_empirical_table_convex_hull_only",
            "success": False,
            "status": int(lp.status),
            "message": str(lp.message),
            "primal_objective": float(objective),
            "dual_lower_bound": None,
            "relative_primal_dual_gap": None,
            "global_pricing_gap": None,
            "globally_certified": False,
        }

    multipliers = np.maximum(np.asarray(lp.x[:-1], dtype=np.float64), 0.0)
    minimum_gradient = float(lp.x[-1])
    lagrangian_gradient = gradient + jacobian.T @ multipliers
    supporting_lower_bound = float(
        constant
        + lambda_coefficients @ multipliers
        + remaining_mass * minimum_gradient
    )
    raw_lower_bound = max(
        supporting_lower_bound,
        float(unconstrained_objective_lower_bound),
    )
    # Floating-point LP and nonlinear evaluations can differ by a few ulps.
    # Subtract an explicit allowance so the recorded value remains a lower
    # bound under the declared numerical model.
    roundoff_allowance = 1.0e-10 * max(
        1.0,
        abs(float(objective)),
        abs(raw_lower_bound),
        float(np.sum(np.abs(multipliers * values))),
    )
    lower_bound = raw_lower_bound - roundoff_allowance
    absolute_gap = max(0.0, float(objective) - lower_bound)
    free = (
        ~declared_floor_active
        if declared_floor_active is not None
        else current > floor + 1.0e-8
    )
    stationarity_residual = (
        float(np.max(np.abs(lagrangian_gradient[free] - minimum_gradient)))
        if np.any(free)
        else 0.0
    )
    dual_face_scale = max(1.0, abs(supporting_lower_bound))
    declared_dual_face_gap = 1.0e-8
    numerical_dual_face_gap = 9.9e-9
    minimum_norm_target = (
        supporting_lower_bound - numerical_dual_face_gap * dual_face_scale
    )
    minimum_norm_success = False
    minimum_norm_message = "minimum_norm_kkt_not_solved"
    minimum_norm_multipliers = multipliers.copy()
    minimum_norm_lower_bound = supporting_lower_bound
    minimum_norm_status = -1
    active_constraint_tolerance = 1.0e-7 * max(
        1.0,
        float(np.max(np.abs(values), initial=0.0)),
    )
    certificate_active = np.flatnonzero(
        (values >= -active_constraint_tolerance)
        | (multipliers > 1.0e-12)
    )
    zero_lower_bound = float(constant + remaining_mass * float(np.min(gradient)))
    if zero_lower_bound >= minimum_norm_target - 1.0e-10:
        minimum_norm_multipliers = np.zeros_like(multipliers)
        minimum_norm_lower_bound = zero_lower_bound
        minimum_norm_status = 0
        minimum_norm_message = "analytic_zero_minimum_norm_dual_face_solution"
        minimum_norm_success = True
        certificate_active = np.empty(0, dtype=np.int64)
    elif len(certificate_active) == 0:
        minimum_norm_message = "no_certificate_active_dual_constraints"
    else:
        active_initial = multipliers[certificate_active]
        active_jacobian = jacobian[certificate_active]
        active_lambda_coefficients = lambda_coefficients[certificate_active]
        initial_gradient = gradient + active_jacobian.T @ active_initial
        initial_level = float(np.min(initial_gradient))
        initial = np.concatenate((active_initial, [initial_level]))

        def norm_objective(local_values: np.ndarray) -> float:
            return 0.5 * float(local_values[:-1] @ local_values[:-1])

        def norm_gradient(local_values: np.ndarray) -> np.ndarray:
            return np.concatenate((local_values[:-1], [0.0]))

        dual_feasibility_matrix = np.column_stack(
            (
                active_jacobian.T,
                -np.ones(num_components, dtype=np.float64),
            )
        )
        dual_face_jacobian = np.concatenate(
            (active_lambda_coefficients, [remaining_mass])
        )
        minimum_norm = minimize(
            norm_objective,
            initial,
            jac=norm_gradient,
            method="SLSQP",
            bounds=[(0.0, None)] * len(certificate_active)
            + [(None, None)],
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda local: float(
                        constant
                        + active_lambda_coefficients @ local[:-1]
                        + remaining_mass * local[-1]
                        - minimum_norm_target
                    ),
                    "jac": lambda local: dual_face_jacobian,
                },
                {
                    "type": "ineq",
                    "fun": lambda local: gradient
                    + active_jacobian.T @ local[:-1]
                    - local[-1],
                    "jac": lambda local: dual_feasibility_matrix,
                },
            ],
            options={"maxiter": 10_000, "ftol": 1.0e-13, "disp": False},
        )
        candidate_multipliers = np.zeros_like(multipliers)
        candidate_multipliers[certificate_active] = np.maximum(
            np.asarray(minimum_norm.x[:-1], dtype=np.float64),
            0.0,
        )
        candidate_gradient = gradient + jacobian.T @ candidate_multipliers
        candidate_minimum = float(np.min(candidate_gradient))
        candidate_lower_bound = float(
            constant
            + active_lambda_coefficients @ minimum_norm.x[:-1]
            + remaining_mass * candidate_minimum
        )
        dual_face_relative_gap = max(
            0.0,
            supporting_lower_bound - candidate_lower_bound,
        ) / dual_face_scale
        candidate_floor_multipliers = np.maximum(
            candidate_gradient - candidate_minimum,
            0.0,
        )
        candidate_bound_complementarity = float(
            np.max(
                candidate_floor_multipliers * (current - floor),
                initial=0.0,
            )
        )
        candidate_dual_violation = max(
            0.0,
            -float(
                np.min(
                    candidate_gradient - float(minimum_norm.x[-1]),
                    initial=0.0,
                )
            ),
        )
        candidate_feasible = bool(
            candidate_dual_violation <= 1.0e-7
            and dual_face_relative_gap <= declared_dual_face_gap + 1.0e-12
            and candidate_bound_complementarity <= 1.0e-7
        )
        minimum_norm_success = candidate_feasible
        minimum_norm_status = int(minimum_norm.status)
        minimum_norm_message = (
            str(minimum_norm.message)
            if minimum_norm.success
            else f"certificate-accepted candidate: {minimum_norm.message}"
        )
        minimum_norm_multipliers = candidate_multipliers
        minimum_norm_lower_bound = candidate_lower_bound

    minimum_norm_gradient = gradient + jacobian.T @ minimum_norm_multipliers
    minimum_norm_level = float(np.min(minimum_norm_gradient))
    minimum_norm_floor_multipliers = np.maximum(
        minimum_norm_gradient - minimum_norm_level,
        0.0,
    )
    minimum_norm_stationarity = float(
        np.max(
            np.abs(
                minimum_norm_gradient
                - minimum_norm_level
                - minimum_norm_floor_multipliers
            ),
            initial=0.0,
        )
    )
    minimum_norm_bound_complementarity = float(
        np.max(
            minimum_norm_floor_multipliers * (current - floor),
            initial=0.0,
        )
    )
    minimum_norm_confidence_complementarity = float(
        np.max(
            np.abs(minimum_norm_multipliers * values),
            initial=0.0,
        )
    )
    minimum_norm_success = bool(
        minimum_norm_success
        and minimum_norm_stationarity <= 1.0e-7
        and minimum_norm_bound_complementarity <= 1.0e-7
        and minimum_norm_confidence_complementarity <= 1.0e-7
    )
    return {
        "certificate_kind": "restricted_supporting_hyperplane_dual",
        "support_scope": "declared_empirical_table_convex_hull_only",
        "success": True,
        "status": int(lp.status),
        "message": str(lp.message),
        "primal_objective": float(objective),
        "supporting_hyperplane_lower_bound": supporting_lower_bound,
        "unconstrained_kl_lower_bound": float(
            unconstrained_objective_lower_bound
        ),
        "raw_dual_lower_bound": raw_lower_bound,
        "roundoff_allowance": roundoff_allowance,
        "dual_lower_bound": lower_bound,
        "absolute_primal_dual_gap": absolute_gap,
        "relative_primal_dual_gap": absolute_gap
        / max(1.0, abs(float(objective))),
        "maximum_constraint_violation": max(
            0.0,
            float(np.max(values, initial=-np.inf)),
        ),
        "simplex_residual": abs(float(np.sum(current)) - 1.0),
        "floor_violation": max(0.0, floor - float(np.min(current))),
        "stationarity_residual": stationarity_residual,
        "complementarity_residual": float(
            np.max(np.abs(multipliers * values), initial=0.0)
        ),
        "dual_multiplier_l1": float(np.sum(multipliers)),
        "dual_multipliers": multipliers.tolist(),
        "minimum_norm_dual": {
            "success": minimum_norm_success,
            "status": minimum_norm_status,
            "message": minimum_norm_message,
            "target_gap": declared_dual_face_gap,
            "numerical_target_gap": numerical_dual_face_gap,
            "multipliers": minimum_norm_multipliers.tolist(),
            "multiplier_l2": float(np.linalg.norm(minimum_norm_multipliers)),
            "active_constraint_count": int(len(certificate_active)),
            "total_constraint_count": int(len(values)),
            "active_constraint_tolerance": active_constraint_tolerance,
            "inactive_multipliers_fixed_by_complementarity": True,
            "floor_dual_source": "analytic_simplex_lower_bound_kkt",
            "floor_active_count": int(
                np.sum(minimum_norm_floor_multipliers > 1.0e-12)
            ),
            "supporting_lower_bound": minimum_norm_lower_bound,
            "dual_face_relative_gap": max(
                0.0,
                supporting_lower_bound - minimum_norm_lower_bound,
            )
            / dual_face_scale,
            "relative_primal_dual_gap": max(
                0.0,
                float(objective) - minimum_norm_lower_bound,
            )
            / max(1.0, abs(float(objective))),
            "stationarity_residual": minimum_norm_stationarity,
            "confidence_complementarity_residual": (
                minimum_norm_confidence_complementarity
            ),
            "bound_complementarity_residual": (
                minimum_norm_bound_complementarity
            ),
            "complementarity_residual": max(
                float(
                    np.max(
                        np.abs(minimum_norm_multipliers * values),
                        initial=0.0,
                    )
                ),
                minimum_norm_bound_complementarity,
            ),
            "dual_negativity": max(
                0.0,
                -float(np.min(minimum_norm_multipliers, initial=0.0)),
            ),
        },
        "global_pricing_gap": None,
        "global_pricing_status": "unavailable_no_full_row_pricing_oracle",
        "globally_certified": False,
    }


def solve_restricted_mixture_rce(
    component_probabilities: np.ndarray,
    component_residuals: np.ndarray,
    log_prior_probabilities: np.ndarray,
    confidence: RCEConfidenceSet,
    *,
    component_names: tuple[str, ...] | None = None,
    max_iterations: int = 5_000,
    ftol: float = 1.0e-12,
    optimal_face_tolerance: float = 1.0e-10,
    feasibility_tolerance: float = 1.0e-9,
    weight_floor: float = 1.0e-12,
) -> RestrictedMixtureRCEResult:
    """Solve RCE exactly on the convex hull of declared empirical tables.

    This is a released-only restricted diagnostic. It removes the integer
    interpolation barrier between the supplied tables, but it is not a global
    relaxed-RCE certificate over the full row domain.
    """

    probabilities = np.asarray(component_probabilities, dtype=np.float64)
    residuals = np.asarray(component_residuals, dtype=np.float64)
    log_prior = np.asarray(log_prior_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] < 2:
        raise ValueError("component_probabilities must contain at least two components")
    num_components, support_size = probabilities.shape
    if support_size <= 0:
        raise ValueError("component_probabilities must have non-empty support")
    if residuals.shape != (num_components, confidence.dimension):
        raise ValueError("component_residuals must match components and confidence dimension")
    if log_prior.shape != (support_size,):
        raise ValueError("log_prior_probabilities must match the declared support")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("component_probabilities must be finite and nonnegative")
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(log_prior)):
        raise ValueError("component_residuals and log_prior_probabilities must be finite")
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("Every component distribution must sum to one")
    if np.any(probabilities.sum(axis=0) <= 0.0):
        raise ValueError("The declared support contains an unused atom")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(float(ftol)) or float(ftol) <= 0.0:
        raise ValueError("ftol must be finite and positive")
    if not math.isfinite(float(optimal_face_tolerance)) or optimal_face_tolerance < 0.0:
        raise ValueError("optimal_face_tolerance must be finite and nonnegative")
    if not math.isfinite(float(feasibility_tolerance)) or feasibility_tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be finite and nonnegative")
    floor = float(weight_floor)
    if not math.isfinite(floor) or floor <= 0.0 or floor * num_components >= 1.0:
        raise ValueError("weight_floor must be positive and leave a non-empty simplex")

    def normalize_floored_simplex(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        excess = np.maximum(vector - floor, 0.0)
        available = 1.0 - floor * num_components
        total_excess = float(np.sum(excess))
        if total_excess <= 0.0:
            excess = np.zeros(num_components, dtype=np.float64)
            excess[int(np.argmax(vector))] = 1.0
            total_excess = 1.0
        return floor + available * excess / total_excess

    names = (
        tuple(f"component_{index}" for index in range(num_components))
        if component_names is None
        else tuple(str(name) for name in component_names)
    )
    if len(names) != num_components or len(set(names)) != num_components:
        raise ValueError("component_names must be unique and match the components")

    tube_indices = np.flatnonzero(confidence.tube_mask)
    bounds = confidence.coordinate_bounds[tube_indices]

    def mixed_residual(weights: np.ndarray) -> np.ndarray:
        return np.asarray(weights, dtype=np.float64) @ residuals

    def constraints_for(weights: np.ndarray, slack: float) -> np.ndarray:
        current = mixed_residual(weights)
        ellipsoid = (
            1.0
            + float(slack)
            - confidence.squared_discrepancy(current)
            / confidence.squared_discrepancy_threshold
        )
        local = current[tube_indices]
        positive = bounds * (1.0 + float(slack)) - local
        negative = bounds * (1.0 + float(slack)) + local
        return np.concatenate(([ellipsoid], positive, negative))

    def constraints_jacobian(weights: np.ndarray, *, include_slack: bool) -> np.ndarray:
        current = mixed_residual(weights)
        precision_residual = confidence.precision_matvec(current)
        width = num_components + int(include_slack)
        jacobian = np.zeros((1 + 2 * len(tube_indices), width), dtype=np.float64)
        jacobian[0, :num_components] = (
            -2.0
            * (residuals @ precision_residual)
            / confidence.squared_discrepancy_threshold
        )
        jacobian[1 : 1 + len(tube_indices), :num_components] = -residuals[
            :, tube_indices
        ].T
        jacobian[1 + len(tube_indices) :, :num_components] = residuals[
            :, tube_indices
        ].T
        if include_slack:
            jacobian[0, -1] = 1.0
            jacobian[1 : 1 + len(tube_indices), -1] = bounds
            jacobian[1 + len(tube_indices) :, -1] = bounds
        return jacobian

    component_slacks = np.asarray(
        [confidence.evaluate(row).slack for row in residuals],
        dtype=np.float64,
    )
    best_component = int(np.argmin(component_slacks))
    initial_weights = np.zeros(num_components, dtype=np.float64)
    initial_weights[best_component] = 1.0
    if component_slacks[best_component] <= float(feasibility_tolerance):
        # Slack is constrained to be nonnegative. A feasible declared component
        # therefore certifies the exact stage-one optimum s*=0 without asking a
        # boundary-sensitive numerical solver to rediscover that fact.
        stage_one_weights = initial_weights
        slack_star = 0.0
        stage_one_record = {
            "success": True,
            "status": 0,
            "message": "analytic feasible-component certificate",
            "iterations": 0,
            "function_evaluations": 0,
            "jacobian_evaluations": 0,
            "objective": 0.0,
        }
    else:
        initial_stage_one = np.concatenate(
            (initial_weights, [float(component_slacks[best_component])])
        )
        stage_one = minimize(
            lambda values: float(values[-1]),
            initial_stage_one,
            jac=lambda values: np.concatenate((np.zeros(num_components), [1.0])),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * num_components + [(0.0, None)],
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda values: float(np.sum(values[:-1]) - 1.0),
                    "jac": lambda values: np.concatenate(
                        (np.ones(num_components), [0.0])
                    ),
                },
                {
                    "type": "ineq",
                    "fun": lambda values: constraints_for(
                        values[:-1], float(values[-1])
                    ),
                    "jac": lambda values: constraints_jacobian(
                        values[:-1], include_slack=True
                    ),
                },
            ],
            options={
                "maxiter": int(max_iterations),
                "ftol": float(ftol),
                "disp": False,
            },
        )
        stage_one_weights = np.maximum(
            np.asarray(stage_one.x[:-1], dtype=np.float64),
            0.0,
        )
        stage_one_weights /= float(np.sum(stage_one_weights))
        measured_slack = confidence.evaluate(mixed_residual(stage_one_weights)).slack
        slack_star = max(0.0, float(stage_one.x[-1]), measured_slack)
        stage_one_record = _solver_record(stage_one)
    face_slack = slack_star + float(optimal_face_tolerance)

    stage_two_initial = normalize_floored_simplex(stage_one_weights)

    def mixture_probabilities(weights: np.ndarray) -> np.ndarray:
        return np.asarray(weights, dtype=np.float64) @ probabilities

    def kl_objective(weights: np.ndarray) -> float:
        mixed = mixture_probabilities(weights)
        positive = mixed > 0.0
        return float(
            np.sum(
                mixed[positive] * (np.log(mixed[positive]) - log_prior[positive]),
                dtype=np.float64,
            )
        )

    def kl_gradient(weights: np.ndarray) -> np.ndarray:
        mixed = mixture_probabilities(weights)
        stabilized = np.maximum(mixed, np.finfo(np.float64).tiny)
        return probabilities @ (np.log(stabilized) - log_prior + 1.0)

    def kl_hessian(weights: np.ndarray) -> np.ndarray:
        mixed = mixture_probabilities(weights)
        stabilized = np.maximum(mixed, np.finfo(np.float64).tiny)
        return (probabilities / stabilized[None, :]) @ probabilities.T

    stage_two = minimize(
        kl_objective,
        stage_two_initial,
        jac=kl_gradient,
        method="SLSQP",
        bounds=[(floor, 1.0)] * num_components,
        constraints=[
            {
                "type": "eq",
                "fun": lambda values: float(np.sum(values) - 1.0),
                "jac": lambda values: np.ones(num_components, dtype=np.float64),
            },
            {
                "type": "ineq",
                "fun": lambda values: constraints_for(values, face_slack),
                "jac": lambda values: constraints_jacobian(
                    values, include_slack=False
                ),
            },
        ],
        options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
    )
    initial_stage_two_record = _solver_record(stage_two)
    polished_weights = normalize_floored_simplex(
        np.asarray(stage_two.x, dtype=np.float64)
    )
    initial_margins = constraints_for(polished_weights, face_slack)
    scaled_margins = initial_margins.copy()
    if len(tube_indices):
        scaled_margins[1 : 1 + len(tube_indices)] /= bounds
        scaled_margins[1 + len(tube_indices) :] /= bounds
    active_constraints = set(
        np.flatnonzero(scaled_margins <= 1.0e-8).astype(int).tolist()
    )
    active_set_passes = 0
    active_set_polish_attempts = 0
    active_set_polish_accepted = 0
    last_polish_record: dict[str, Any] | None = None
    last_weight_slsqp_polish_record: dict[str, Any] | None = None
    last_logit_polish_record: dict[str, Any] | None = None
    last_kkt_polish_record: dict[str, Any] | None = None
    certificate_floor_active_mask: np.ndarray | None = None
    full_constraint_violation = max(
        0.0,
        -float(np.min(initial_margins, initial=math.inf)),
    )
    for active_set_passes in (range(1, 11) if active_constraints else ()):
        candidate_rows = constraints_jacobian(
            polished_weights,
            include_slack=False,
        )
        independent_rows = np.ones((1, num_components), dtype=np.float64)
        independent_rank = int(np.linalg.matrix_rank(independent_rows))
        selected_list: list[int] = []
        for constraint_index in sorted(active_constraints):
            trial = np.vstack(
                (independent_rows, candidate_rows[int(constraint_index)])
            )
            trial_rank = int(np.linalg.matrix_rank(trial))
            if trial_rank > independent_rank:
                selected_list.append(int(constraint_index))
                independent_rows = trial
                independent_rank = trial_rank
        selected = np.asarray(selected_list, dtype=np.int64)
        if not len(selected):
            break
        precision_residuals = np.vstack(
            [confidence.precision_matvec(row) for row in residuals]
        )
        ellipsoid_hessian = (
            -2.0
            * (residuals @ precision_residuals.T)
            / confidence.squared_discrepancy_threshold
        )
        ellipsoid_positions = np.flatnonzero(selected == 0)

        def active_constraint_hessian(
            _weights: np.ndarray,
            multipliers: np.ndarray,
        ) -> np.ndarray:
            if not len(ellipsoid_positions):
                return np.zeros(
                    (num_components, num_components),
                    dtype=np.float64,
                )
            scale = float(np.sum(np.asarray(multipliers)[ellipsoid_positions]))
            return scale * ellipsoid_hessian

        polish_constraints = [
            LinearConstraint(
                np.ones((1, num_components), dtype=np.float64),
                np.ones(1, dtype=np.float64),
                np.ones(1, dtype=np.float64),
            ),
            NonlinearConstraint(
                lambda values: constraints_for(values, face_slack)[selected],
                np.zeros(len(selected), dtype=np.float64),
                np.full(len(selected), np.inf, dtype=np.float64),
                jac=lambda values: constraints_jacobian(
                    values,
                    include_slack=False,
                )[selected],
                hess=active_constraint_hessian,
            ),
        ]
        polish_start = polished_weights
        interior_anchor = normalize_floored_simplex(stage_one_weights)
        for blend in (
            1.0e-8,
            1.0e-7,
            1.0e-6,
            1.0e-5,
            1.0e-4,
            1.0e-3,
            1.0e-2,
            1.0e-1,
        ):
            trial = normalize_floored_simplex(
                (1.0 - blend) * polished_weights
                + blend * interior_anchor
            )
            if float(
                np.min(
                    constraints_for(trial, face_slack)[selected],
                    initial=math.inf,
                )
            ) > 1.0e-10:
                polish_start = trial
                break
        active_set_polish_attempts += 1
        polished = minimize(
            kl_objective,
            polish_start,
            jac=kl_gradient,
            hess=kl_hessian,
            method="trust-constr",
            bounds=Bounds(
                np.full(num_components, floor, dtype=np.float64),
                np.ones(num_components, dtype=np.float64),
                keep_feasible=True,
            ),
            constraints=polish_constraints,
            options={
                "maxiter": min(int(max_iterations), 2_000),
                "gtol": 1.0e-10,
                "xtol": 1.0e-12,
                "barrier_tol": 1.0e-12,
                "initial_barrier_parameter": 1.0e-10,
                "initial_barrier_tolerance": 1.0e-10,
                "factorization_method": "SVDFactorization",
                "verbose": 0,
            },
        )
        trust_multiplier_blocks = list(getattr(polished, "v", ()))
        trust_bound_multipliers = (
            np.asarray(trust_multiplier_blocks[-1], dtype=np.float64)
            if len(trust_multiplier_blocks)
            and np.asarray(trust_multiplier_blocks[-1]).shape
            == (num_components,)
            else np.zeros(num_components, dtype=np.float64)
        )
        last_polish_record = {
            **_solver_record(polished),
            "optimality": float(getattr(polished, "optimality", math.inf)),
            "constraint_violation": float(
                getattr(polished, "constr_violation", math.inf)
            ),
            "barrier_parameter": float(
                getattr(polished, "barrier_parameter", math.inf)
            ),
            "lower_bound_active_count": int(
                np.sum(trust_bound_multipliers < -1.0e-7)
            ),
            "minimum_bound_multiplier": float(
                np.min(trust_bound_multipliers, initial=0.0)
            ),
            "maximum_bound_multiplier": float(
                np.max(trust_bound_multipliers, initial=0.0)
            ),
            "simplex_multiplier": (
                float(np.asarray(trust_multiplier_blocks[0]).reshape(-1)[0])
                if len(trust_multiplier_blocks)
                and np.asarray(trust_multiplier_blocks[0]).size == 1
                else None
            ),
            "active_confidence_multipliers": (
                np.asarray(trust_multiplier_blocks[1], dtype=np.float64).tolist()
                if len(trust_multiplier_blocks) >= 2
                else None
            ),
            "bound_multipliers": trust_bound_multipliers.tolist(),
        }
        candidate_weights = normalize_floored_simplex(
            np.asarray(polished.x, dtype=np.float64)
        )

        # The interior-point solver can terminate on ``xtol`` while its bound
        # barrier still spreads small positive mass over components that should
        # lie on the simplex floor.  Re-solve the already identified convex
        # active-set problem directly in weight space.  SLSQP can hit those
        # bounds exactly; acceptance below still depends on the full original
        # confidence family and the independent convex dual certificate.
        if num_components > 16:
            weight_slsqp_scale = 1.0
            weight_slsqp = minimize(
                lambda values: weight_slsqp_scale * kl_objective(values),
                candidate_weights,
                jac=lambda values: weight_slsqp_scale * kl_gradient(values),
                method="SLSQP",
                bounds=[(floor, 1.0)] * num_components,
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda values: float(np.sum(values) - 1.0),
                        "jac": lambda values: np.ones(
                            num_components,
                            dtype=np.float64,
                        ),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda values: constraints_for(
                            values,
                            face_slack,
                        )[selected]
                        * weight_slsqp_scale,
                        "jac": lambda values: constraints_jacobian(
                            values,
                            include_slack=False,
                        )[selected]
                        * weight_slsqp_scale,
                    },
                ],
                options={
                    "maxiter": min(int(max_iterations), 2_000),
                    "ftol": min(float(ftol), 1.0e-14),
                    "disp": False,
                },
            )
            weight_slsqp_weights = normalize_floored_simplex(
                np.asarray(weight_slsqp.x, dtype=np.float64)
            )
            weight_slsqp_margins = constraints_for(
                weight_slsqp_weights,
                face_slack,
            )
            weight_slsqp_violation = max(
                0.0,
                -float(np.min(weight_slsqp_margins, initial=math.inf)),
            )
            last_weight_slsqp_polish_record = {
                **_solver_record(weight_slsqp),
                "scaled_solver_objective": float(weight_slsqp.fun),
                "objective": kl_objective(weight_slsqp_weights),
                "numerical_scale": weight_slsqp_scale,
                "full_constraint_violation": weight_slsqp_violation,
                "minimum_weight": float(np.min(weight_slsqp_weights)),
            }
            weight_slsqp_acceptable = bool(
                weight_slsqp.success
                and weight_slsqp_violation <= float(feasibility_tolerance)
                and kl_objective(weight_slsqp_weights)
                <= kl_objective(candidate_weights)
                + max(
                    1.0e-10,
                    1.0e-9 * max(1.0, abs(kl_objective(candidate_weights))),
                )
            )
            if weight_slsqp_acceptable:
                candidate_weights = weight_slsqp_weights
                polished = OptimizeResult(
                    x=candidate_weights,
                    fun=kl_objective(candidate_weights),
                    success=True,
                    status=int(weight_slsqp.status),
                    message="weight-space active-set SLSQP polish certified",
                    nit=int(getattr(weight_slsqp, "nit", -1)),
                    nfev=int(getattr(weight_slsqp, "nfev", -1)),
                    njev=int(getattr(weight_slsqp, "njev", -1)),
                    optimality=0.0,
                    constr_violation=weight_slsqp_violation,
                )

        # The trust-constr pass already identifies the active simplex face.
        # A second dense logit solve did not improve the certificate on the
        # paper-scale 31-component dictionaries and dominated CPU time.  Keep
        # it only as a small-problem diagnostic; the explicit active-support
        # KKT solve below is the certificate-producing refinement.
        if 16 < num_components <= 24:
            logit_numerical_scale = 1.0e6
            pivot = int(np.argmax(candidate_weights))
            logit_indices = np.asarray(
                [index for index in range(num_components) if index != pivot],
                dtype=np.int64,
            )
            excess_mass = 1.0 - floor * num_components
            excess_probabilities = np.maximum(
                (candidate_weights - floor) / excess_mass,
                np.finfo(np.float64).tiny,
            )
            logits_start = (
                np.log(excess_probabilities[logit_indices])
                - math.log(excess_probabilities[pivot])
            )

            def weights_from_logits(logits: np.ndarray) -> np.ndarray:
                full = np.zeros(num_components, dtype=np.float64)
                full[logit_indices] = np.asarray(logits, dtype=np.float64)
                full -= float(np.max(full))
                exponentials = np.exp(full)
                probabilities_local = exponentials / float(
                    np.sum(exponentials)
                )
                return floor + excess_mass * probabilities_local

            def weight_logit_jacobian(weights_local: np.ndarray) -> np.ndarray:
                probabilities_local = (weights_local - floor) / excess_mass
                return excess_mass * (
                    np.diag(probabilities_local)
                    - np.outer(probabilities_local, probabilities_local)
                )[:, logit_indices]

            def logit_objective(logits: np.ndarray) -> float:
                return logit_numerical_scale * kl_objective(
                    weights_from_logits(logits)
                )

            def logit_gradient(logits: np.ndarray) -> np.ndarray:
                weights_local = weights_from_logits(logits)
                return logit_numerical_scale * (
                    weight_logit_jacobian(weights_local).T
                    @ kl_gradient(weights_local)
                )

            logit_polish = minimize(
                logit_objective,
                logits_start,
                jac=logit_gradient,
                method="SLSQP",
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda logits: constraints_for(
                            weights_from_logits(logits),
                            face_slack,
                        )[selected]
                        * logit_numerical_scale,
                        "jac": lambda logits: (
                            constraints_jacobian(
                                weights_from_logits(logits),
                                include_slack=False,
                            )[selected]
                            @ weight_logit_jacobian(
                                weights_from_logits(logits)
                            )
                            * logit_numerical_scale
                        ),
                    }
                ],
                options={
                    "maxiter": min(int(max_iterations), 100),
                    "ftol": min(float(ftol), 1.0e-14),
                    "disp": False,
                },
            )
            logit_weights = weights_from_logits(
                np.asarray(logit_polish.x, dtype=np.float64)
            )
            logit_margins = constraints_for(logit_weights, face_slack)
            logit_violation = max(
                0.0,
                -float(np.min(logit_margins, initial=math.inf)),
            )
            last_logit_polish_record = {
                **_solver_record(logit_polish),
                "scaled_solver_objective": float(logit_polish.fun),
                "objective": kl_objective(logit_weights),
                "numerical_scale": logit_numerical_scale,
                "full_constraint_violation": logit_violation,
                "minimum_weight": float(np.min(logit_weights)),
            }
            logit_acceptable = bool(
                logit_polish.success
                and logit_violation <= float(feasibility_tolerance)
                and kl_objective(logit_weights)
                <= kl_objective(candidate_weights)
                + max(
                    1.0e-10,
                    1.0e-9 * max(1.0, abs(kl_objective(candidate_weights))),
                )
            )
            if logit_acceptable:
                candidate_weights = logit_weights
                polished = OptimizeResult(
                    x=candidate_weights,
                    fun=kl_objective(candidate_weights),
                    success=True,
                    status=int(logit_polish.status),
                    message="logit-simplex active-face polish certified",
                    nit=int(getattr(logit_polish, "nit", -1)),
                    nfev=int(getattr(logit_polish, "nfev", -1)),
                    njev=int(getattr(logit_polish, "njev", -1)),
                    optimality=0.0,
                    constr_violation=logit_violation,
                )

        # The interior-point solution supplies a stable active face, but its
        # barrier can leave numerically positive weights that should be at the
        # simplex floor. Polish the square KKT system on an explicit support,
        # adding any omitted coordinate whose reduced gradient violates the
        # lower-bound condition. This produces the primal accuracy required by
        # the independently computed minimum-norm dual certificate.
        certificate_hessian = -ellipsoid_hessian
        trust_multipliers = trust_multiplier_blocks
        bound_multipliers = (
            np.asarray(trust_multipliers[-1], dtype=np.float64)
            if len(trust_multipliers)
            and np.asarray(trust_multipliers[-1]).shape == (num_components,)
            else np.zeros(num_components, dtype=np.float64)
        )
        support = set(
            np.flatnonzero(bound_multipliers >= -1.0e-7)
            .astype(int)
            .tolist()
        )
        support.add(int(np.argmax(candidate_weights)))
        kkt_candidate: np.ndarray | None = None
        kkt_working_weights = candidate_weights.copy()
        kkt_history: list[dict[str, Any]] = []
        kkt_record: dict[str, Any] = {
            "success": False,
            "message": "active_support_kkt_not_attempted",
        }
        # The explicit active-support root solve is useful for small theorem
        # fixtures.  At paper scale the independent convex dual certificate is
        # both stronger and substantially cheaper; avoid a dense CPU polish
        # whose cost grows sharply once the dictionary exceeds 16 tables.
        kkt_support_pass_limit = min(16, num_components) if num_components <= 16 else 0
        for support_pass in range(1, kkt_support_pass_limit + 1):
            if len(support) > 64:
                kkt_record = {
                    "success": False,
                    "message": "active_support_exceeds_polish_cap_64",
                    "support_passes": support_pass,
                    "support_size": len(support),
                    "history": kkt_history,
                }
                break
            free_indices = np.asarray(sorted(support), dtype=np.int64)
            fixed_mask = np.ones(num_components, dtype=bool)
            fixed_mask[free_indices] = False
            free_mass = 1.0 - floor * int(np.sum(fixed_mask))
            if free_mass <= floor * len(free_indices):
                kkt_record = {
                    "success": False,
                    "message": "active_support_has_no_free_simplex_mass",
                    "support_passes": support_pass,
                }
                break
            if len(free_indices) == 1 and np.any(fixed_mask):
                omitted = np.flatnonzero(fixed_mask)
                support.add(
                    int(omitted[np.argmax(kkt_working_weights[omitted])])
                )
                continue
            start_weights = np.full(num_components, floor, dtype=np.float64)
            local_excess = np.maximum(
                kkt_working_weights[free_indices] - floor,
                0.0,
            )
            if float(np.sum(local_excess)) <= 0.0:
                local_excess[
                    int(np.argmax(kkt_working_weights[free_indices]))
                ] = 1.0
            # A newly admitted support coordinate otherwise starts one ulp
            # above the lower bound.  The KL KKT system is badly conditioned
            # there and the bounded least-squares solver cannot discover its
            # interior mass.  Seed every declared free coordinate with a tiny
            # positive excess, then renormalize exactly below.
            interior_seed = max(
                1.0e-10,
                1.0e-6 * float(np.max(local_excess, initial=1.0)),
            )
            local_excess = np.maximum(local_excess, interior_seed)
            free_excess_mass = free_mass - floor * len(free_indices)
            local_start = (
                floor
                + free_excess_mass
                * local_excess
                / float(np.sum(local_excess))
            )
            start_weights[free_indices] = local_start
            pivot = int(free_indices[np.argmax(local_start)])
            nonpivot_indices = free_indices[free_indices != pivot]
            h_jacobian = -constraints_jacobian(
                start_weights,
                include_slack=False,
            )[selected]
            start_gradient = kl_gradient(start_weights)
            dual_system = (
                h_jacobian[:, nonpivot_indices].T
                - h_jacobian[:, pivot][None, :]
            )
            if (
                len(trust_multipliers) >= 2
                and np.asarray(trust_multipliers[1]).shape == (len(selected),)
            ):
                dual_start = np.maximum(
                    -np.asarray(trust_multipliers[1], dtype=np.float64),
                    0.0,
                )
            else:
                dual_start = np.linalg.lstsq(
                    dual_system,
                    -(
                        start_gradient[nonpivot_indices]
                        - start_gradient[pivot]
                    ),
                    rcond=None,
                )[0]
            dual_start = np.maximum(dual_start, 0.0)
            root_start = np.concatenate(
                (start_weights[nonpivot_indices], dual_start)
            )

            def kkt_equations(local: np.ndarray) -> np.ndarray:
                trial_weights = np.full(
                    num_components,
                    floor,
                    dtype=np.float64,
                )
                nonpivot_count = len(nonpivot_indices)
                trial_weights[nonpivot_indices] = local[:nonpivot_count]
                trial_weights[pivot] = free_mass - float(
                    np.sum(local[:nonpivot_count])
                )
                if trial_weights[pivot] <= 0.0:
                    return np.full(
                        nonpivot_count + len(selected),
                        1.0e6 * (1.0 - trial_weights[pivot]),
                        dtype=np.float64,
                    )
                local_multipliers = local[
                    nonpivot_count : nonpivot_count + len(selected)
                ]
                h_values = -constraints_for(trial_weights, face_slack)[selected]
                local_jacobian = -constraints_jacobian(
                    trial_weights,
                    include_slack=False,
                )[selected]
                lagrangian_gradient = (
                    kl_gradient(trial_weights)
                    + local_jacobian.T @ local_multipliers
                )
                return np.concatenate(
                    (
                        lagrangian_gradient[nonpivot_indices]
                        - lagrangian_gradient[pivot],
                        h_values,
                    )
                )

            def kkt_jacobian(local: np.ndarray) -> np.ndarray:
                trial_weights = np.full(
                    num_components,
                    floor,
                    dtype=np.float64,
                )
                nonpivot_count = len(nonpivot_indices)
                trial_weights[nonpivot_indices] = local[:nonpivot_count]
                trial_weights[pivot] = free_mass - float(
                    np.sum(local[:nonpivot_count])
                )
                local_multipliers = local[
                    nonpivot_count : nonpivot_count + len(selected)
                ]
                local_jacobian = -constraints_jacobian(
                    trial_weights,
                    include_slack=False,
                )[selected]
                lagrangian_hessian = kl_hessian(trial_weights)
                if len(ellipsoid_positions):
                    lagrangian_hessian = (
                        lagrangian_hessian
                        + float(
                            np.sum(local_multipliers[ellipsoid_positions])
                        )
                        * certificate_hessian
                    )
                active_count = len(selected)
                output = np.zeros(
                    (nonpivot_count + active_count,) * 2,
                    dtype=np.float64,
                )
                local_hessian = lagrangian_hessian[
                    np.ix_(nonpivot_indices, nonpivot_indices)
                ]
                local_hessian = (
                    local_hessian
                    - lagrangian_hessian[nonpivot_indices, pivot][:, None]
                    - lagrangian_hessian[pivot, nonpivot_indices][None, :]
                    + lagrangian_hessian[pivot, pivot]
                )
                output[:nonpivot_count, :nonpivot_count] = local_hessian
                output[
                    :nonpivot_count,
                    nonpivot_count : nonpivot_count + active_count,
                ] = (
                    local_jacobian[:, nonpivot_indices].T
                    - local_jacobian[:, pivot][None, :]
                )
                output[
                    nonpivot_count : nonpivot_count + active_count,
                    :nonpivot_count,
                ] = (
                    local_jacobian[:, nonpivot_indices]
                    - local_jacobian[:, pivot][:, None]
                )
                return output

            nonpivot_count = len(nonpivot_indices)
            active_count = len(selected)
            kkt_lower = np.concatenate(
                (
                    np.full(nonpivot_count, floor, dtype=np.float64),
                    np.zeros(active_count, dtype=np.float64),
                )
            )
            nonpivot_upper = free_mass - floor * (len(free_indices) - 1)
            kkt_upper = np.concatenate(
                (
                    np.full(
                        nonpivot_count,
                        nonpivot_upper,
                        dtype=np.float64,
                    ),
                    np.full(active_count, np.inf, dtype=np.float64),
                )
            )
            finite_lower = np.isfinite(kkt_lower)
            finite_upper = np.isfinite(kkt_upper)
            root_start[finite_lower] = np.maximum(
                root_start[finite_lower],
                np.nextafter(kkt_lower[finite_lower], np.inf),
            )
            root_start[finite_upper] = np.minimum(
                root_start[finite_upper],
                np.nextafter(kkt_upper[finite_upper], -np.inf),
            )
            kkt = least_squares(
                kkt_equations,
                root_start,
                jac=kkt_jacobian,
                bounds=(kkt_lower, kkt_upper),
                method="trf",
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=500,
                x_scale="jac",
            )
            root_residual = float(
                np.max(np.abs(kkt_equations(np.asarray(kkt.x))), initial=0.0)
            )
            kkt_history.append(
                {
                    "support_pass": support_pass,
                    "support_size": len(free_indices),
                    "root_residual": root_residual,
                    "solver_status": int(kkt.status),
                }
            )
            solved_weights = np.full(num_components, floor, dtype=np.float64)
            solved_weights[nonpivot_indices] = np.asarray(kkt.x)[
                :nonpivot_count
            ]
            solved_weights[pivot] = free_mass - float(
                np.sum(np.asarray(kkt.x)[:nonpivot_count])
            )
            solved_multipliers = np.asarray(kkt.x)[
                nonpivot_count : nonpivot_count + len(selected)
            ]
            if root_residual > 9.0e-8 and np.any(fixed_mask):
                kkt_history[-1]["expansion_reason"] = "root_residual"
                omitted = np.flatnonzero(fixed_mask)
                next_index = int(
                    omitted[np.argmax(kkt_working_weights[omitted])]
                )
                support.add(next_index)
                continue
            too_small = free_indices[
                solved_weights[free_indices] <= floor + 1.0e-9
            ]
            if len(too_small) and len(free_indices) > 1:
                kkt_history[-1]["expansion_reason"] = "drop_floor_weights"
                support.difference_update(int(index) for index in too_small)
                kkt_working_weights = solved_weights
                continue
            solved_jacobian = -constraints_jacobian(
                solved_weights,
                include_slack=False,
            )[selected]
            solved_lagrangian_gradient = (
                kl_gradient(solved_weights)
                + solved_jacobian.T @ solved_multipliers
            )
            solved_level = float(solved_lagrangian_gradient[pivot])
            reduced_gradient = solved_lagrangian_gradient - solved_level
            omitted_violations = np.flatnonzero(
                fixed_mask & (reduced_gradient < -1.0e-7)
            )
            if len(omitted_violations):
                kkt_history[-1]["expansion_reason"] = (
                    "omitted_reduced_gradient"
                )
                kkt_history[-1]["minimum_omitted_reduced_gradient"] = float(
                    np.min(reduced_gradient[omitted_violations])
                )
                worst = int(
                    omitted_violations[
                        np.argmin(reduced_gradient[omitted_violations])
                    ]
                )
                support.add(worst)
                kkt_working_weights = solved_weights
                continue
            kkt_success = bool(
                root_residual <= 9.0e-8
                and np.min(solved_weights) >= floor - 1.0e-10
                and np.min(solved_multipliers, initial=0.0) >= -1.0e-10
                and np.max(
                    np.abs(float(np.sum(solved_weights)) - 1.0),
                    initial=0.0,
                )
                <= 1.0e-10
            )
            kkt_record = {
                "success": kkt_success,
                "status": int(kkt.status),
                "message": str(kkt.message),
                "support_passes": support_pass,
                "support_size": len(free_indices),
                "root_residual": root_residual,
                "minimum_multiplier": float(
                    np.min(solved_multipliers, initial=0.0)
                ),
                "minimum_omitted_reduced_gradient": float(
                    np.min(reduced_gradient[fixed_mask], initial=0.0)
                ),
                "trust_lower_bound_active_count": int(
                    np.sum(bound_multipliers < -1.0e-7)
                ),
                "history": kkt_history,
            }
            if kkt_success:
                kkt_candidate = solved_weights
            break
        if (
            kkt_candidate is None
            and kkt_history
            and kkt_record.get("message") == "active_support_kkt_not_attempted"
        ):
            kkt_record = {
                "success": False,
                "message": "active_support_kkt_pass_limit_exhausted",
                "support_passes": len(kkt_history),
                "support_size": len(support),
                "history": kkt_history,
            }
        last_kkt_polish_record = kkt_record
        if kkt_candidate is not None:
            candidate_weights = kkt_candidate
            polished = OptimizeResult(
                x=candidate_weights,
                fun=kl_objective(candidate_weights),
                success=True,
                status=0,
                message="active-support KKT polish certified",
                nit=int(kkt_record.get("support_passes", -1)),
                nfev=-1,
                njev=-1,
                optimality=float(kkt_record.get("root_residual", math.inf)),
                constr_violation=max(
                    0.0,
                    -float(
                        np.min(
                            constraints_for(candidate_weights, face_slack),
                            initial=math.inf,
                        )
                    ),
                ),
            )
        all_margins = constraints_for(candidate_weights, face_slack)
        full_constraint_violation = max(
            0.0,
            -float(np.min(all_margins, initial=math.inf)),
        )
        scaled_all_margins = all_margins.copy()
        if len(tube_indices):
            scaled_all_margins[1 : 1 + len(tube_indices)] /= bounds
            scaled_all_margins[1 + len(tube_indices) :] /= bounds
        newly_relevant = set(
            np.flatnonzero(scaled_all_margins < -float(feasibility_tolerance))
            .astype(int)
            .tolist()
        )
        candidate_objective = kl_objective(candidate_weights)
        current_objective = kl_objective(polished_weights)
        objective_tolerance = max(
            1.0e-10,
            1.0e-9 * max(1.0, abs(current_objective)),
        )
        solver_certified = bool(polished.success) or (
            float(getattr(polished, "optimality", math.inf)) <= 1.0e-7
            and float(getattr(polished, "constr_violation", math.inf)) <= 1.0e-8
        )
        candidate_acceptable = bool(
            solver_certified
            and full_constraint_violation <= float(feasibility_tolerance)
            and candidate_objective <= current_objective + objective_tolerance
        )
        if candidate_acceptable:
            polished_weights = candidate_weights
            stage_two = polished
            active_set_polish_accepted += 1
            certificate_floor_active_mask = (
                candidate_weights <= floor + 1.0e-9
                if kkt_candidate is not None
                else trust_bound_multipliers < -1.0e-7
            )
        if candidate_acceptable and newly_relevant.issubset(active_constraints):
            break
        active_constraints.update(newly_relevant)
        if not solver_certified or not newly_relevant:
            break
    accepted_margins = constraints_for(polished_weights, face_slack)
    full_constraint_violation = max(
        0.0,
        -float(np.min(accepted_margins, initial=math.inf)),
    )
    weights = polished_weights
    final_residual = mixed_residual(weights)
    evaluation = confidence.evaluate(
        final_residual,
        tolerance=float(feasibility_tolerance),
    )
    certificate = _restricted_supporting_hyperplane_certificate(
        weights=weights,
        weight_floor=floor,
        objective=kl_objective(weights),
        objective_gradient=kl_gradient(weights),
        unconstrained_objective_lower_bound=-float(logsumexp(log_prior)),
        constraint_values=-constraints_for(weights, face_slack),
        constraint_jacobian=-constraints_jacobian(
            weights,
            include_slack=False,
        ),
        floor_active_mask=certificate_floor_active_mask,
    )
    certificate["constraint_order"] = {
        "ellipsoid": 0,
        "tube_positive_start": 1,
        "tube_negative_start": 1 + len(tube_indices),
        "tube_coordinate_indices": tube_indices.tolist(),
        "normalized_constraints": True,
        "face_slack": face_slack,
    }
    return RestrictedMixtureRCEResult(
        component_weights=weights,
        mixture_probabilities=mixture_probabilities(weights),
        residual=final_residual,
        slack_star=max(slack_star, evaluation.slack),
        kl_objective=kl_objective(weights),
        confidence=evaluation.to_dict(),
        stage_one=stage_one_record,
        stage_two={
            **_solver_record(stage_two),
            "initial_full_solver": initial_stage_two_record,
            "active_set_polish_passes": int(active_set_passes),
            "active_set_polish_attempts": int(active_set_polish_attempts),
            "active_set_polish_accepted": int(active_set_polish_accepted),
            "last_active_set_polish_solver": last_polish_record,
            "last_weight_slsqp_polish_solver": (
                last_weight_slsqp_polish_record
            ),
            "last_logit_polish_solver": last_logit_polish_record,
            "last_active_support_kkt_solver": last_kkt_polish_record,
            "active_constraint_count": len(active_constraints),
            "total_constraint_count": 1 + 2 * len(tube_indices),
            "full_constraint_violation": full_constraint_violation,
        },
        component_names=names,
        support_size=int(support_size),
        support_kind="convex_hull_of_declared_empirical_tables",
        restricted_dual_certificate=certificate,
        globally_certified=False,
    )


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
