from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import xlogy

from qdte.evolution.entropy import ReleasedProductPrior
from qdte.queries.types import QueryCatalogue
from qdte.rce.forest_prior import (
    CCF_ALPHA_STRUCT,
    CCF_ELLIPSOID_TOLERANCE,
    CCF_KL_ABSOLUTE_GAP_TOLERANCE,
    CCF_KL_RELATIVE_GAP_TOLERANCE,
    CCF_MARGINAL_TOLERANCE,
    CCF_NONNEGATIVITY_TOLERANCE,
    CCF_STATIONARITY_TOLERANCE,
    ConfidenceForestEdge,
    ReleasedConfidenceForestPrior,
    _certified_transport_linear_minimum,
    _maximum_weight_forest,
    _pair_geometry,
    _pair_residuals,
    _transport_equalities,
    solve_ccf_pair,
)


SEQUENTIAL_CCF_PRIOR_METHOD = "released_sequential_confidence_calibrated_forest_v1"


class SequentialCCFCertificationError(RuntimeError):
    pass


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CCFInteractionStream:
    name: str
    released_interaction_rate: np.ndarray
    interaction_covariance_rate: float | np.ndarray
    confidence_radius: float
    alpha_struct: float
    round_index: int

    def __post_init__(self) -> None:
        center = np.asarray(self.released_interaction_rate, dtype=np.float64)
        covariance = np.asarray(self.interaction_covariance_rate, dtype=np.float64)
        if not self.name or center.ndim != 2 or min(center.shape) <= 0:
            raise ValueError("Sequential CCF stream must have a named matrix center")
        dimension = int(center.size)
        if covariance.ndim == 0:
            if not math.isfinite(float(covariance)) or float(covariance) <= 0.0:
                raise ValueError(
                    "Sequential CCF streams require positive isotropic covariance"
                )
        elif covariance.shape == center.shape:
            if not np.all(np.isfinite(covariance)) or np.any(covariance <= 0.0):
                raise ValueError(
                    "Sequential CCF streams require positive diagonal covariance"
                )
        elif covariance.shape == (dimension, dimension):
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if (
                not np.all(np.isfinite(covariance))
                or float(np.min(eigenvalues)) <= 0.0
            ):
                raise ValueError(
                    "Sequential CCF streams require positive-definite covariance"
                )
        else:
            raise ValueError(
                "Sequential CCF covariance must be scalar, coefficient-shaped, or dense"
            )
        if not np.all(np.isfinite(center)):
            raise ValueError("Sequential CCF stream center must be finite")
        radius = float(self.confidence_radius)
        alpha = float(self.alpha_struct)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Sequential CCF confidence radius must be finite and positive")
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("Sequential CCF alpha must lie in (0, 1)")
        if int(self.round_index) < -1:
            raise ValueError("Sequential CCF round index must be at least -1")
        object.__setattr__(self, "released_interaction_rate", _readonly(center))
        object.__setattr__(self, "interaction_covariance_rate", _readonly(covariance))

    @property
    def rank(self) -> int:
        return int(self.released_interaction_rate.size)

    @property
    def covariance_storage(self) -> str:
        covariance = self.interaction_covariance_rate
        if covariance.ndim == 0:
            return "isotropic_scalar"
        if covariance.shape == self.released_interaction_rate.shape:
            return "diagonal"
        return "dense"

    def precision_matvec(self, values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (self.rank,):
            raise ValueError("Sequential CCF precision vector has the wrong dimension")
        covariance = self.interaction_covariance_rate
        if covariance.ndim == 0:
            return vector / float(covariance)
        if covariance.shape == self.released_interaction_rate.shape:
            return vector / covariance.reshape(-1)
        return np.linalg.solve(covariance, vector)

    def quadratic(self, values: np.ndarray) -> float:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        return float(vector @ self.precision_matvec(vector))

    def add_scaled_precision(
        self,
        destination: np.ndarray,
        scale: float,
    ) -> None:
        matrix = np.asarray(destination)
        if matrix.shape != (self.rank, self.rank):
            raise ValueError("Sequential CCF Hessian has the wrong dimension")
        covariance = self.interaction_covariance_rate
        factor = float(scale)
        diagonal = np.diag_indices(self.rank)
        if covariance.ndim == 0:
            matrix[diagonal] += factor / float(covariance)
        elif covariance.shape == self.released_interaction_rate.shape:
            matrix[diagonal] += factor / covariance.reshape(-1)
        else:
            matrix += factor * np.linalg.inv(covariance)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "round_index": self.round_index,
            "rank": self.rank,
            "alpha_struct": self.alpha_struct,
            "confidence_radius": self.confidence_radius,
            "released_interaction_rate": self.released_interaction_rate.tolist(),
            "interaction_covariance_rate": self.interaction_covariance_rate.tolist(),
            "interaction_covariance_storage": self.covariance_storage,
        }


@dataclass(frozen=True)
class SequentialCCFPairResult:
    pair: tuple[int, int]
    status: str
    streams: tuple[CCFInteractionStream, ...]
    product_discrepancies: tuple[float, ...]
    product_in_confidence: bool
    feasibility_upper: float
    feasibility_lower: float
    feasibility_gap: float
    pair_table: np.ndarray | None
    kl_upper: float
    kl_lower: float
    kl_gap: float
    weight_lower_bound: float
    nonnegativity_violation: float
    row_marginal_residual: float
    column_marginal_residual: float
    maximum_ellipsoid_violation: float
    stationarity_residual: float
    complementarity_residual: float
    solver_success: bool
    solver_status: int
    solver_message: str
    dual_multipliers: tuple[float, ...]
    certificate_kind: str = "certified_primal_dual_optimum"

    @property
    def eligible(self) -> bool:
        return self.status == "eligible" and self.weight_lower_bound > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": list(self.pair),
            "status": self.status,
            "eligible": self.eligible,
            "num_streams": len(self.streams),
            "streams": [stream.to_public_dict() for stream in self.streams],
            "product_discrepancies": list(self.product_discrepancies),
            "product_in_confidence": self.product_in_confidence,
            "feasibility_upper": self.feasibility_upper,
            "feasibility_lower": self.feasibility_lower,
            "feasibility_gap": self.feasibility_gap,
            "kl_upper": self.kl_upper,
            "kl_lower": self.kl_lower,
            "kl_gap": self.kl_gap,
            "weight_lower_bound": self.weight_lower_bound,
            "nonnegativity_violation": self.nonnegativity_violation,
            "row_marginal_residual": self.row_marginal_residual,
            "column_marginal_residual": self.column_marginal_residual,
            "maximum_ellipsoid_violation": self.maximum_ellipsoid_violation,
            "stationarity_residual": self.stationarity_residual,
            "complementarity_residual": self.complementarity_residual,
            "solver_success": self.solver_success,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
            "dual_multipliers": list(self.dual_multipliers),
            "certificate_kind": self.certificate_kind,
        }


def _supporting_transport_dual(
    *,
    table: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    objective: float,
    objective_gradient: np.ndarray,
    constraint_values: np.ndarray,
    constraint_gradients: np.ndarray,
) -> dict[str, Any]:
    gradient_f = np.asarray(objective_gradient, dtype=np.float64).reshape(-1)
    values = np.asarray(constraint_values, dtype=np.float64)
    gradients = np.asarray(constraint_gradients, dtype=np.float64).reshape(
        len(values), -1
    )
    current = np.asarray(table, dtype=np.float64).reshape(-1)
    equality, rhs = _transport_equalities(left, right)
    num_constraints = len(values)
    num_potentials = equality.shape[0]
    # f(mu) + lambda h(mu) is minorized by its tangent. The transportation
    # dual supplies the exact minimum of that affine minorant.
    lambda_coefficients = values - gradients @ current
    constant = float(objective) - float(gradient_f @ current)
    objective_coefficients = -np.concatenate((lambda_coefficients, rhs))
    inequalities = np.zeros(
        (len(current), num_constraints + num_potentials),
        dtype=np.float64,
    )
    inequalities[:, :num_constraints] = -gradients.T
    inequalities[:, num_constraints:] = equality.T.toarray()
    result = linprog(
        objective_coefficients,
        A_ub=inequalities,
        b_ub=gradient_f,
        bounds=[(0.0, None)] * num_constraints
        + [(None, None)] * num_potentials,
        method="highs",
    )
    if not result.success or result.x is None:
        raise SequentialCCFCertificationError(
            f"Sequential CCF supporting dual failed: {result.status} {result.message}"
        )
    multipliers = np.maximum(result.x[:num_constraints], 0.0)
    lagrangian_gradient = gradient_f + gradients.T @ multipliers
    transport = _certified_transport_linear_minimum(
        lagrangian_gradient.reshape(table.shape),
        left,
        right,
    )
    lower = float(
        constant
        + lambda_coefficients @ multipliers
        + transport.lower_bound
    )
    allowance = 1.0e-10 * max(1.0, abs(float(objective)), abs(lower))
    lower -= allowance
    gap = max(0.0, float(objective) - lower)
    linear_value = float(lagrangian_gradient @ current)
    return {
        "lower_bound": lower,
        "gap": gap,
        "relative_gap": gap / max(1.0e-12, abs(float(objective))),
        "multipliers": multipliers,
        "stationarity": max(0.0, linear_value - transport.lower_bound)
        / max(1.0, abs(float(objective)), abs(linear_value)),
        "complementarity": float(
            np.max(np.abs(multipliers * values), initial=0.0)
        )
        / max(1.0, abs(float(objective))),
        "transport": transport,
    }


def solve_sequential_ccf_pair(
    pair: tuple[int, int],
    left_marginal: np.ndarray,
    right_marginal: np.ndarray,
    streams: Sequence[CCFInteractionStream],
    *,
    max_iterations: int = 5_000,
) -> SequentialCCFPairResult:
    left = np.asarray(left_marginal, dtype=np.float64)
    right = np.asarray(right_marginal, dtype=np.float64)
    local_streams = tuple(streams)
    if not local_streams:
        raise ValueError("Sequential CCF pair requires at least one observation stream")
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("Sequential CCF marginals must be strictly positive")
    if not np.isclose(np.sum(left), 1.0) or not np.isclose(np.sum(right), 1.0):
        raise ValueError("Sequential CCF marginals must sum to one")
    product, c_left, c_right, basis, product_theta = _pair_geometry(left, right)
    shape = product_theta.shape
    if any(stream.released_interaction_rate.shape != shape for stream in local_streams):
        raise ValueError("Sequential CCF stream shape does not match the pair marginals")
    dimension = int(product_theta.size)

    def table_from_x(x: np.ndarray) -> np.ndarray:
        return (
            product.reshape(-1) + basis @ np.asarray(x, dtype=np.float64)
        ).reshape(product.shape)

    def theta_from_x(x: np.ndarray) -> np.ndarray:
        return product_theta.reshape(-1) + np.asarray(x, dtype=np.float64)

    if len(local_streams) == 1:
        stream = local_streams[0]
        single = solve_ccf_pair(
            pair,
            left,
            right,
            stream.released_interaction_rate,
            stream.interaction_covariance_rate,
            confidence_radius=stream.confidence_radius,
            max_iterations=int(max_iterations),
        )
        multiplier = 0.0
        if single.pair_table is not None and not single.product_in_confidence:
            table = np.asarray(single.pair_table, dtype=np.float64)
            theta = c_left.T @ table @ c_right
            safe = np.maximum(table, np.finfo(np.float64).tiny)
            gradient_f = basis.T @ (np.log(safe / product) + 1.0).reshape(-1)
            gradient_g = 2.0 * stream.precision_matvec(
                theta.reshape(-1)
                - stream.released_interaction_rate.reshape(-1)
            )
            denominator = float(gradient_g @ gradient_g)
            if denominator > 0.0:
                multiplier = max(
                    0.0,
                    -float(gradient_f @ gradient_g) / denominator,
                )
        radius = float(stream.confidence_radius)
        return SequentialCCFPairResult(
            pair=pair,
            status=single.status,
            streams=local_streams,
            product_discrepancies=(float(single.product_discrepancy),),
            product_in_confidence=single.product_in_confidence,
            feasibility_upper=float(single.eta_upper / radius),
            feasibility_lower=float(single.eta_lower / radius),
            feasibility_gap=float(single.eta_gap / radius),
            pair_table=single.pair_table,
            kl_upper=single.kl_upper,
            kl_lower=single.kl_lower,
            kl_gap=single.kl_gap,
            weight_lower_bound=single.weight_lower_bound,
            nonnegativity_violation=single.nonnegativity_violation,
            row_marginal_residual=single.row_marginal_residual,
            column_marginal_residual=single.column_marginal_residual,
            maximum_ellipsoid_violation=single.ellipsoid_violation,
            stationarity_residual=single.stationarity_residual,
            complementarity_residual=single.complementarity_residual,
            solver_success=bool(
                single.eta_solver_success and single.kl_solver_success
            ),
            solver_status=max(single.eta_solver_status, single.kl_solver_status),
            solver_message=(
                "single_stream_certified_solver; "
                f"eta={single.eta_solver_message}; kl={single.kl_solver_message}"
            ),
            dual_multipliers=(multiplier,),
        )

    def discrepancies(x: np.ndarray) -> np.ndarray:
        theta = theta_from_x(x)
        return np.asarray(
            [
                stream.quadratic(
                    theta - stream.released_interaction_rate.reshape(-1)
                )
                for stream in local_streams
            ],
            dtype=np.float64,
        )

    def discrepancy_gradients_x(x: np.ndarray) -> np.ndarray:
        theta = theta_from_x(x)
        return np.stack(
            [
                2.0
                * stream.precision_matvec(
                    theta - stream.released_interaction_rate.reshape(-1)
                )
                for stream in local_streams
            ],
            axis=0,
        )

    radii = np.asarray(
        [stream.confidence_radius for stream in local_streams],
        dtype=np.float64,
    )
    product_discrepancies = discrepancies(np.zeros(dimension, dtype=np.float64))
    product_in_set = bool(
        np.all(product_discrepancies <= radii + CCF_ELLIPSOID_TOLERANCE)
    )
    if product_in_set:
        nonnegative, row_residual, column_residual = _pair_residuals(
            product,
            left,
            right,
        )
        return SequentialCCFPairResult(
            pair=pair,
            status="product_in_confidence",
            streams=local_streams,
            product_discrepancies=tuple(product_discrepancies.tolist()),
            product_in_confidence=True,
            feasibility_upper=float(np.max(product_discrepancies / radii)),
            feasibility_lower=0.0,
            feasibility_gap=float(np.max(product_discrepancies / radii)),
            pair_table=_readonly(product),
            kl_upper=0.0,
            kl_lower=0.0,
            kl_gap=0.0,
            weight_lower_bound=0.0,
            nonnegativity_violation=nonnegative,
            row_marginal_residual=row_residual,
            column_marginal_residual=column_residual,
            maximum_ellipsoid_violation=0.0,
            stationarity_residual=0.0,
            complementarity_residual=0.0,
            solver_success=True,
            solver_status=0,
            solver_message="exact_product_feasibility_witness",
            dual_multipliers=tuple(0.0 for _ in local_streams),
        )

    initial_ratio = float(np.max(product_discrepancies / radii))
    initial = np.concatenate((np.zeros(dimension, dtype=np.float64), [initial_ratio]))

    def feasibility_objective(values: np.ndarray) -> float:
        return float(values[-1])

    def feasibility_gradient(values: np.ndarray) -> np.ndarray:
        return np.concatenate((np.zeros(dimension, dtype=np.float64), [1.0]))

    def table_constraint(values: np.ndarray) -> np.ndarray:
        return table_from_x(values[:-1]).reshape(-1)

    def table_jacobian(values: np.ndarray) -> np.ndarray:
        return np.column_stack((basis, np.zeros(product.size, dtype=np.float64)))

    def ratio_constraint(values: np.ndarray) -> np.ndarray:
        return float(values[-1]) - discrepancies(values[:-1]) / radii

    def ratio_jacobian(values: np.ndarray) -> np.ndarray:
        gradients = -discrepancy_gradients_x(values[:-1]) / radii.reshape(-1, 1)
        return np.column_stack((gradients, np.ones(len(local_streams), dtype=np.float64)))

    feasibility = minimize(
        feasibility_objective,
        initial,
        jac=feasibility_gradient,
        method="SLSQP",
        bounds=[(None, None)] * dimension + [(0.0, None)],
        constraints=[
            {"type": "ineq", "fun": table_constraint, "jac": table_jacobian},
            {"type": "ineq", "fun": ratio_constraint, "jac": ratio_jacobian},
        ],
        options={"maxiter": int(max_iterations), "ftol": 1.0e-13, "disp": False},
    )
    x_feasible = np.asarray(feasibility.x[:-1], dtype=np.float64)
    table_feasible = table_from_x(x_feasible)
    ratios_feasible = discrepancies(x_feasible) / radii
    feasibility_upper = max(float(feasibility.x[-1]), float(np.max(ratios_feasible)))
    feasibility_lower = 0.0
    if feasibility_upper > 1.0 + CCF_ELLIPSOID_TOLERANCE:
        raw_multipliers = np.asarray(
            getattr(feasibility, "multipliers", np.empty(0)),
            dtype=np.float64,
        )
        ratio_multipliers = raw_multipliers[-len(local_streams) :]
        ratio_multipliers = np.maximum(ratio_multipliers, 0.0)
        if float(np.sum(ratio_multipliers)) <= 0.0:
            raise SequentialCCFCertificationError(
                f"Sequential CCF pair {pair} could not certify an empty intersection"
            )
        ratio_multipliers /= float(np.sum(ratio_multipliers))
        gradients_x = discrepancy_gradients_x(x_feasible) / radii.reshape(-1, 1)
        gradients_table = np.stack(
            [
                (c_left @ gradient.reshape(shape) @ c_right.T).reshape(product.shape)
                for gradient in gradients_x
            ],
            axis=0,
        )
        combined_gradient = np.tensordot(
            ratio_multipliers,
            gradients_table,
            axes=(0, 0),
        )
        transport = _certified_transport_linear_minimum(
            combined_gradient,
            left,
            right,
        )
        weighted_value = float(ratio_multipliers @ ratios_feasible)
        feasibility_lower = float(
            weighted_value
            - np.sum(combined_gradient * table_feasible, dtype=np.float64)
            + transport.lower_bound
        )
        feasibility_lower -= 1.0e-10 * max(1.0, abs(feasibility_lower))
        if feasibility_lower <= 1.0 + CCF_ELLIPSOID_TOLERANCE:
            raise SequentialCCFCertificationError(
                f"Sequential CCF pair {pair} intersection classification is unresolved: "
                f"[{feasibility_lower:.12g}, {feasibility_upper:.12g}]"
            )
        nonnegative, row_residual, column_residual = _pair_residuals(
            table_feasible,
            left,
            right,
        )
        return SequentialCCFPairResult(
            pair=pair,
            status="anchor_interaction_inconsistent",
            streams=local_streams,
            product_discrepancies=tuple(product_discrepancies.tolist()),
            product_in_confidence=False,
            feasibility_upper=feasibility_upper,
            feasibility_lower=feasibility_lower,
            feasibility_gap=max(0.0, feasibility_upper - feasibility_lower),
            pair_table=None,
            kl_upper=0.0,
            kl_lower=0.0,
            kl_gap=0.0,
            weight_lower_bound=0.0,
            nonnegativity_violation=nonnegative,
            row_marginal_residual=row_residual,
            column_marginal_residual=column_residual,
            maximum_ellipsoid_violation=max(0.0, feasibility_upper - 1.0),
            stationarity_residual=0.0,
            complementarity_residual=0.0,
            solver_success=bool(feasibility.success),
            solver_status=int(feasibility.status),
            solver_message=str(feasibility.message),
            dual_multipliers=tuple(ratio_multipliers.tolist()),
        )
    if float(np.min(table_feasible)) < -CCF_NONNEGATIVITY_TOLERANCE:
        raise SequentialCCFCertificationError(
            f"Sequential CCF pair {pair} feasibility solve violated nonnegativity"
        )

    low = 0.0
    high = 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if np.all(discrepancies(middle * x_feasible) <= radii):
            high = middle
        else:
            low = middle
    interpolation = min(1.0, high + 1.0e-5 * (1.0 - high))
    x_start = interpolation * x_feasible
    if np.any(discrepancies(x_start) > radii + CCF_ELLIPSOID_TOLERANCE):
        x_start = x_feasible.copy()

    def kl_objective(x: np.ndarray) -> float:
        table = table_from_x(x)
        if float(np.min(table)) < -CCF_NONNEGATIVITY_TOLERANCE:
            return 1.0e100
        clipped = np.maximum(table, 0.0)
        return float(np.sum(xlogy(clipped, clipped / product), dtype=np.float64))

    solver_table_floor = max(
        1.0e-12,
        1.0e-8 * float(np.min(product)),
    )

    def kl_gradient_x(x: np.ndarray) -> np.ndarray:
        table = np.maximum(table_from_x(x), solver_table_floor)
        return basis.T @ (np.log(table / product) + 1.0).reshape(-1)

    kl_solve = minimize(
        kl_objective,
        x_start,
        jac=kl_gradient_x,
        method="SLSQP",
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: table_from_x(x).reshape(-1),
                "jac": lambda x: basis,
            },
            {
                "type": "ineq",
                "fun": lambda x: 1.0 - discrepancies(x) / radii,
                "jac": lambda x: (
                    -discrepancy_gradients_x(x) / radii.reshape(-1, 1)
                ),
            },
        ],
        options={"maxiter": int(max_iterations), "ftol": 1.0e-13, "disp": False},
    )

    def numerically_feasible(candidate: np.ndarray) -> bool:
        local = np.asarray(candidate, dtype=np.float64)
        return bool(
            float(np.min(table_from_x(local))) >= -CCF_NONNEGATIVITY_TOLERANCE
            and float(np.max(discrepancies(local) - radii))
            <= CCF_ELLIPSOID_TOLERANCE
        )

    feasible_candidates = [
        candidate
        for candidate in (
            np.asarray(kl_solve.x, dtype=np.float64),
            x_start,
            x_feasible,
        )
        if numerically_feasible(candidate)
    ]
    if not feasible_candidates:
        raise SequentialCCFCertificationError(
            f"Sequential CCF pair {pair} lost every certified feasibility witness"
        )
    x_star = min(feasible_candidates, key=kl_objective).copy()
    table_star = table_from_x(x_star)
    discrepancy_star = discrepancies(x_star)
    h_values = discrepancy_star - radii
    objective = kl_objective(x_star)
    gradient_f_table = np.log(
        np.maximum(table_star, np.finfo(np.float64).tiny) / product
    ) + 1.0
    gradients_x = discrepancy_gradients_x(x_star)
    gradients_table = np.stack(
        [
            c_left @ gradient.reshape(shape) @ c_right.T
            for gradient in gradients_x
        ],
        axis=0,
    )
    dual = _supporting_transport_dual(
        table=table_star,
        left=left,
        right=right,
        objective=objective,
        objective_gradient=gradient_f_table,
        constraint_values=h_values,
        constraint_gradients=gradients_table,
    )

    def certificate_passes(
        local_table: np.ndarray,
        local_values: np.ndarray,
        local_dual: dict[str, Any],
    ) -> bool:
        local_nonnegative, local_row, local_column = _pair_residuals(
            local_table,
            left,
            right,
        )
        return bool(
            local_nonnegative <= CCF_NONNEGATIVITY_TOLERANCE
            and local_row <= CCF_MARGINAL_TOLERANCE
            and local_column <= CCF_MARGINAL_TOLERANCE
            and max(0.0, float(np.max(local_values)))
            <= CCF_ELLIPSOID_TOLERANCE
            and (
                local_dual["gap"] <= CCF_KL_ABSOLUTE_GAP_TOLERANCE
                or local_dual["relative_gap"] <= CCF_KL_RELATIVE_GAP_TOLERANCE
            )
            and local_dual["stationarity"] <= CCF_STATIONARITY_TOLERANCE
            and local_dual["complementarity"] <= 1.0e-7
        )

    # SLSQP occasionally stops a few 1e-8 away from the certified optimum on
    # general-cardinality pairs. Polish the active stream constraints with a
    # damped primal-dual Newton solve, then rerun the independent transport
    # lower-bound certificate. This never weakens a tolerance or accepts the
    # optimizer status by itself.
    if not certificate_passes(table_star, h_values, dual):
        active = np.flatnonzero(
            np.asarray(dual["multipliers"], dtype=np.float64) > 1.0e-12
        )
        if len(active) == 0:
            active = np.asarray([int(np.argmax(h_values))], dtype=np.int64)
        polish_x = x_star.copy()
        polish_multipliers = np.maximum(
            np.asarray(dual["multipliers"], dtype=np.float64)[active],
            1.0e-15,
        )
        for _ in range(min(int(max_iterations), 200)):
            polish_table = table_from_x(polish_x)
            if float(np.min(polish_table)) <= 0.0:
                break
            local_gradients = discrepancy_gradients_x(polish_x)[active]
            local_values = discrepancies(polish_x)[active] - radii[active]
            stationarity = (
                kl_gradient_x(polish_x)
                + local_gradients.T @ polish_multipliers
            )
            kkt_residual = np.concatenate((stationarity, local_values))
            if float(np.max(np.abs(kkt_residual))) <= 1.0e-12:
                break
            flat_table = polish_table.reshape(-1)
            hessian = basis.T @ (basis / flat_table.reshape(-1, 1))
            for multiplier, stream_index in zip(
                polish_multipliers,
                active,
                strict=True,
            ):
                local_streams[int(stream_index)].add_scaled_precision(
                    hessian,
                    2.0 * float(multiplier),
                )
            kkt = np.block(
                [
                    [
                        hessian,
                        local_gradients.T,
                    ],
                    [
                        local_gradients,
                        np.zeros((len(active), len(active)), dtype=np.float64),
                    ],
                ]
            )
            try:
                direction = np.linalg.solve(kkt, -kkt_residual)
            except np.linalg.LinAlgError:
                direction = np.linalg.lstsq(kkt, -kkt_residual, rcond=1.0e-12)[0]
            delta_x = direction[:dimension]
            delta_multipliers = direction[dimension:]
            delta_table = basis @ delta_x
            step = 1.0
            decreasing = delta_table < 0.0
            if np.any(decreasing):
                step = min(
                    step,
                    0.99
                    * float(
                        np.min(
                            -flat_table[decreasing] / delta_table[decreasing]
                        )
                    ),
                )
            decreasing_dual = delta_multipliers < 0.0
            if np.any(decreasing_dual):
                step = min(
                    step,
                    0.99
                    * float(
                        np.min(
                            -polish_multipliers[decreasing_dual]
                            / delta_multipliers[decreasing_dual]
                        )
                    ),
                )
            merit = float(np.linalg.norm(kkt_residual))
            accepted = False
            for _ in range(80):
                candidate_x = polish_x + step * delta_x
                candidate_multipliers = (
                    polish_multipliers + step * delta_multipliers
                )
                candidate_table = table_from_x(candidate_x)
                candidate_all_values = discrepancies(candidate_x) - radii
                if (
                    float(np.min(candidate_table)) > 0.0
                    and float(np.min(candidate_multipliers)) >= 0.0
                    and float(np.max(candidate_all_values))
                    <= CCF_ELLIPSOID_TOLERANCE
                ):
                    candidate_gradients = discrepancy_gradients_x(candidate_x)[
                        active
                    ]
                    candidate_residual = np.concatenate(
                        (
                            kl_gradient_x(candidate_x)
                            + candidate_gradients.T @ candidate_multipliers,
                            candidate_all_values[active],
                        )
                    )
                    if float(np.linalg.norm(candidate_residual)) <= (
                        1.0 - 1.0e-4 * step
                    ) * merit:
                        accepted = True
                        break
                step *= 0.5
            if not accepted:
                break
            polish_x = candidate_x
            polish_multipliers = candidate_multipliers

        polished_table = table_from_x(polish_x)
        polished_values = discrepancies(polish_x) - radii
        polished_objective = kl_objective(polish_x)
        polished_gradient_table = np.log(
            np.maximum(polished_table, np.finfo(np.float64).tiny) / product
        ) + 1.0
        polished_gradients_x = discrepancy_gradients_x(polish_x)
        polished_gradients_table = np.stack(
            [
                c_left @ gradient.reshape(shape) @ c_right.T
                for gradient in polished_gradients_x
            ],
            axis=0,
        )
        polished_dual = _supporting_transport_dual(
            table=polished_table,
            left=left,
            right=right,
            objective=polished_objective,
            objective_gradient=polished_gradient_table,
            constraint_values=polished_values,
            constraint_gradients=polished_gradients_table,
        )
        if (
            polished_objective <= objective + 1.0e-10
            and polished_dual["gap"] < dual["gap"]
        ):
            x_star = polish_x
            table_star = polished_table
            discrepancy_star = discrepancies(x_star)
            h_values = polished_values
            objective = polished_objective
            gradients_x = polished_gradients_x
            gradients_table = polished_gradients_table
            dual = polished_dual
    nonnegative, row_residual, column_residual = _pair_residuals(
        table_star,
        left,
        right,
    )
    maximum_ellipsoid = max(0.0, float(np.max(h_values)))
    # SLSQP can report a boundary line-search status after reaching a point
    # whose independently reconstructed primal/dual certificate is already
    # valid.  Promotion is based on that certificate, not the optimizer's
    # advisory status bit; the raw status and message remain in the artifact.
    certified = certificate_passes(table_star, h_values, dual)
    if not certified:
        # The feasible witness certifies a finite upper bound, while KL >= 0
        # is a global lower bound. A zero certified edge weight is therefore
        # exact enough to exclude this edge from the deterministic forest;
        # no unverified optimizer output enters the prior.
        return SequentialCCFPairResult(
            pair=pair,
            status="zero_certified_weight",
            streams=local_streams,
            product_discrepancies=tuple(product_discrepancies.tolist()),
            product_in_confidence=False,
            feasibility_upper=feasibility_upper,
            feasibility_lower=feasibility_lower,
            feasibility_gap=max(0.0, feasibility_upper - feasibility_lower),
            pair_table=_readonly(np.maximum(table_star, 0.0)),
            kl_upper=objective,
            kl_lower=0.0,
            kl_gap=objective,
            weight_lower_bound=0.0,
            nonnegativity_violation=nonnegative,
            row_marginal_residual=row_residual,
            column_marginal_residual=column_residual,
            maximum_ellipsoid_violation=maximum_ellipsoid,
            stationarity_residual=float(dual["stationarity"]),
            complementarity_residual=float(dual["complementarity"]),
            solver_success=True,
            solver_status=int(kl_solve.status),
            solver_message=(
                "certified_feasible_upper_and_universal_kl_lower_bound; "
                f"raw_solver_success={bool(kl_solve.success)}; "
                f"raw_message={kl_solve.message}"
            ),
            dual_multipliers=tuple(0.0 for _ in local_streams),
            certificate_kind="feasible_upper_universal_kl_lower_bound",
        )
    lower = float(dual["lower_bound"])
    weight_lower = max(0.0, lower)
    return SequentialCCFPairResult(
        pair=pair,
        status="eligible" if weight_lower > 0.0 else "zero_certified_weight",
        streams=local_streams,
        product_discrepancies=tuple(product_discrepancies.tolist()),
        product_in_confidence=False,
        feasibility_upper=feasibility_upper,
        feasibility_lower=feasibility_lower,
        feasibility_gap=max(0.0, feasibility_upper - feasibility_lower),
        pair_table=_readonly(np.maximum(table_star, 0.0)),
        kl_upper=objective,
        kl_lower=lower,
        kl_gap=float(dual["gap"]),
        weight_lower_bound=weight_lower,
        nonnegativity_violation=nonnegative,
        row_marginal_residual=row_residual,
        column_marginal_residual=column_residual,
        maximum_ellipsoid_violation=maximum_ellipsoid,
        stationarity_residual=float(dual["stationarity"]),
        complementarity_residual=float(dual["complementarity"]),
        solver_success=True,
        solver_status=int(kl_solve.status),
        solver_message=(
            f"certificate_passed; raw_solver_success={bool(kl_solve.success)}; "
            f"raw_message={kl_solve.message}"
        ),
        dual_multipliers=tuple(np.asarray(dual["multipliers"]).tolist()),
    )


def build_sequential_confidence_forest_prior(
    qcat: QueryCatalogue,
    released_target: np.ndarray,
    cardinalities: np.ndarray | tuple[int, ...],
    pair_streams: Mapping[tuple[int, int], Sequence[CCFInteractionStream]],
    *,
    public_total: int,
    smoothing: float = 1.0,
) -> ReleasedConfidenceForestPrior:
    cards = tuple(int(value) for value in np.asarray(cardinalities, dtype=np.int64))
    product_prior = ReleasedProductPrior.from_released_oneway(
        qcat,
        released_target,
        cards,
        public_total=int(public_total),
        smoothing=float(smoothing),
    )
    pair_results: list[SequentialCCFPairResult] = []
    for pair in sorted(pair_streams):
        left, right = int(pair[0]), int(pair[1])
        if left >= right or min(left, right) < 0 or right >= len(cards):
            raise ValueError(f"Sequential CCF pair {pair!r} is not canonical")
        pair_results.append(
            solve_sequential_ccf_pair(
                (left, right),
                product_prior.probabilities[left],
                product_prior.probabilities[right],
                pair_streams[pair],
            )
        )
    frozen_results = tuple(pair_results)
    selected = _maximum_weight_forest(frozen_results, len(cards))
    lambda_n = 1.0 / float(public_total + 1)
    edges: list[ConfidenceForestEdge] = []
    for result in selected:
        if result.pair_table is None:
            raise AssertionError("Selected sequential CCF edge is missing its pair table")
        left, right = result.pair
        product = np.outer(
            product_prior.probabilities[left],
            product_prior.probabilities[right],
        )
        smoothed = (1.0 - lambda_n) * result.pair_table + lambda_n * product
        edges.append(
            ConfidenceForestEdge(
                pair=result.pair,
                table=smoothed,
                unsmoothed_table=result.pair_table,
                weight=result.kl_upper,
                weight_lower_bound=result.weight_lower_bound,
            )
        )
    return ReleasedConfidenceForestPrior(
        product_prior=product_prior,
        edges=tuple(edges),
        pair_results=frozen_results,
        alpha_struct=CCF_ALPHA_STRUCT,
        method=SEQUENTIAL_CCF_PRIOR_METHOD,
    )


__all__ = [
    "CCFInteractionStream",
    "SEQUENTIAL_CCF_PRIOR_METHOD",
    "SequentialCCFCertificationError",
    "SequentialCCFPairResult",
    "build_sequential_confidence_forest_prior",
    "solve_sequential_ccf_pair",
]
