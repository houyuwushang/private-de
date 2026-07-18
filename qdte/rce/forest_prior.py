from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.special import xlogy
from scipy.stats import chi2

from qdte.evolution.entropy import ReleasedProductPrior
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.support_coarsening import (
    SupportCoarsening,
    lift_coarse_pair_distribution,
)
from qdte.queries.orthogonal import helmert_contrast
from qdte.queries.types import QueryCatalogue


CCF_PRIOR_METHOD = "released_confidence_calibrated_forest_v1"
SEQUENTIAL_CCF_PRIOR_METHOD = (
    "released_sequential_confidence_calibrated_forest_v1"
)
CCF_ALPHA_STRUCT = 0.05
CCF_NONNEGATIVITY_TOLERANCE = 1.0e-10
CCF_MARGINAL_TOLERANCE = 1.0e-10
CCF_ELLIPSOID_TOLERANCE = 1.0e-8
CCF_ETA_RELATIVE_GAP_TOLERANCE = 1.0e-8
CCF_KL_RELATIVE_GAP_TOLERANCE = 1.0e-7
CCF_KL_ABSOLUTE_GAP_TOLERANCE = 1.0e-10
CCF_STATIONARITY_TOLERANCE = 1.0e-7
CCF_COMPLEMENTARITY_TOLERANCE = 1.0e-7
CCF_TRANSPORT_SOLVER_TOLERANCE = 1.0e-10
CCF_NEWTON_KKT_TOLERANCE = 1.0e-12


class CCFCertificationError(RuntimeError):
    pass


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class TransportLinearCertificate:
    lower_bound: float
    solver_objective: float
    equality_residual: float
    dual_feasibility_residual: float
    solver_status: int
    solver_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_bound": self.lower_bound,
            "solver_objective": self.solver_objective,
            "equality_residual": self.equality_residual,
            "dual_feasibility_residual": self.dual_feasibility_residual,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
        }


def _transport_equalities(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    d_left = int(len(left))
    d_right = int(len(right))
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    equation = 0
    for i in range(d_left):
        for j in range(d_right):
            row_indices.append(equation)
            col_indices.append(i * d_right + j)
            values.append(1.0)
        equation += 1
    # The final column-marginal equality is redundant.
    for j in range(d_right - 1):
        for i in range(d_left):
            row_indices.append(equation)
            col_indices.append(i * d_right + j)
            values.append(1.0)
        equation += 1
    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(d_left + d_right - 1, d_left * d_right),
        dtype=np.float64,
    )
    rhs = np.concatenate([left, right[:-1]]).astype(np.float64, copy=False)
    return matrix, rhs


def _certified_transport_linear_minimum(
    cost: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> TransportLinearCertificate:
    vector = np.asarray(cost, dtype=np.float64).reshape(-1)
    equality, rhs = _transport_equalities(left, right)
    result = linprog(
        vector,
        A_eq=equality,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
        options={
            "dual_feasibility_tolerance": CCF_TRANSPORT_SOLVER_TOLERANCE,
            "primal_feasibility_tolerance": CCF_TRANSPORT_SOLVER_TOLERANCE,
        },
    )
    if not result.success or result.x is None:
        raise CCFCertificationError(
            f"Transportation certificate LP failed: {result.status} {result.message}"
        )
    marginals = getattr(getattr(result, "eqlin", None), "marginals", None)
    if marginals is None:
        raise CCFCertificationError("HiGHS did not return equality dual multipliers")
    dual = np.asarray(marginals, dtype=np.float64).copy()
    dual_lhs = np.asarray(equality.T @ dual, dtype=np.float64).reshape(-1)
    violation = max(0.0, float(np.max(dual_lhs - vector)))
    safety = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(vector))))
    )
    # Every cell belongs to exactly one retained row-marginal equality. Lowering
    # all row potentials makes the LP dual feasible without changing support.
    dual[: len(left)] -= violation + safety
    adjusted_lhs = np.asarray(equality.T @ dual, dtype=np.float64).reshape(-1)
    dual_residual = max(0.0, float(np.max(adjusted_lhs - vector)))
    lower_bound = float(rhs @ dual)
    equality_residual = float(
        np.max(np.abs(np.asarray(equality @ result.x).reshape(-1) - rhs))
    )
    if dual_residual > 1.0e-10 or equality_residual > 1.0e-9:
        raise CCFCertificationError(
            "Transportation LP failed its primal/dual residual certificate"
        )
    return TransportLinearCertificate(
        lower_bound=lower_bound,
        solver_objective=float(result.fun),
        equality_residual=equality_residual,
        dual_feasibility_residual=dual_residual,
        solver_status=int(result.status),
        solver_message=str(result.message),
    )


@dataclass(frozen=True)
class CCFPairResult:
    pair: tuple[int, int]
    status: str
    rank: int
    confidence_radius: float
    product_discrepancy: float
    eta_upper: float
    eta_lower: float
    eta_gap: float
    product_in_confidence: bool
    pair_table: np.ndarray | None
    kl_upper: float
    kl_lower: float
    kl_gap: float
    weight_lower_bound: float
    nonnegativity_violation: float
    row_marginal_residual: float
    column_marginal_residual: float
    ellipsoid_violation: float
    stationarity_residual: float
    complementarity_residual: float
    eta_solver_success: bool
    eta_solver_status: int
    eta_solver_message: str
    kl_solver_success: bool
    kl_solver_status: int
    kl_solver_message: str
    eta_lp: TransportLinearCertificate
    kl_lp: TransportLinearCertificate | None

    @property
    def eligible(self) -> bool:
        return self.status == "eligible" and self.weight_lower_bound > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": list(self.pair),
            "status": self.status,
            "eligible": self.eligible,
            "rank": self.rank,
            "confidence_radius": self.confidence_radius,
            "product_discrepancy": self.product_discrepancy,
            "eta_upper": self.eta_upper,
            "eta_lower": self.eta_lower,
            "eta_gap": self.eta_gap,
            "product_in_confidence": self.product_in_confidence,
            "kl_upper": self.kl_upper,
            "kl_lower": self.kl_lower,
            "kl_gap": self.kl_gap,
            "weight_lower_bound": self.weight_lower_bound,
            "nonnegativity_violation": self.nonnegativity_violation,
            "row_marginal_residual": self.row_marginal_residual,
            "column_marginal_residual": self.column_marginal_residual,
            "ellipsoid_violation": self.ellipsoid_violation,
            "stationarity_residual": self.stationarity_residual,
            "complementarity_residual": self.complementarity_residual,
            "eta_solver_success": self.eta_solver_success,
            "eta_solver_status": self.eta_solver_status,
            "eta_solver_message": self.eta_solver_message,
            "kl_solver_success": self.kl_solver_success,
            "kl_solver_status": self.kl_solver_status,
            "kl_solver_message": self.kl_solver_message,
            "pair_table_sha256": (
                _array_sha256(self.pair_table) if self.pair_table is not None else None
            ),
            "eta_transport_certificate": self.eta_lp.to_dict(),
            "kl_transport_certificate": (
                self.kl_lp.to_dict() if self.kl_lp is not None else None
            ),
        }


def _pair_geometry(
    left_marginal: np.ndarray,
    right_marginal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(left_marginal, dtype=np.float64)
    right = np.asarray(right_marginal, dtype=np.float64)
    product = np.outer(left, right)
    left_contrast = helmert_contrast(len(left))
    right_contrast = helmert_contrast(len(right))
    basis = np.einsum(
        "ia,jb->ijab",
        left_contrast,
        right_contrast,
        optimize=True,
    ).reshape(product.size, (len(left) - 1) * (len(right) - 1))
    product_theta = left_contrast.T @ product @ right_contrast
    return product, left_contrast, right_contrast, basis, product_theta


def _pair_residuals(
    table: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float, float]:
    return (
        max(0.0, -float(np.min(table))),
        float(np.max(np.abs(np.sum(table, axis=1) - left))),
        float(np.max(np.abs(np.sum(table, axis=0) - right))),
    )


def _solve_eta_active_set(
    hessian_matrix: np.ndarray,
    linear_term: np.ndarray,
    basis: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, bool, int, str]:
    hessian = np.asarray(hessian_matrix, dtype=np.float64)
    linear = np.asarray(linear_term, dtype=np.float64)
    matrix = np.asarray(basis, dtype=np.float64)
    base = np.asarray(offset, dtype=np.float64)
    dimension = int(len(linear))
    if hessian.shape == (dimension,):
        hessian = np.diag(hessian)
    if hessian.shape != (dimension, dimension):
        raise ValueError("Eta QP Hessian shape mismatch")
    if not np.allclose(hessian, hessian.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Eta QP Hessian must be symmetric")
    eigenvalues = np.linalg.eigvalsh(hessian)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("Eta QP requires a positive-definite Hessian")
    if matrix.shape != (len(base), dimension):
        raise ValueError("Eta QP basis shape mismatch")

    x = np.zeros(dimension, dtype=np.float64)
    active: list[int] = []
    max_iterations = 10 * (dimension + len(base)) + 100
    for iteration in range(max_iterations):
        gradient = hessian @ x + linear
        if active:
            active_array = np.asarray(active, dtype=np.int64)
            active_matrix = matrix[active_array]
            _, triangular, pivots = qr(
                active_matrix.T,
                mode="economic",
                pivoting=True,
            )
            diagonal = np.abs(np.diag(triangular))
            rank_tolerance = (
                1.0e-11 * max(active_matrix.shape) * float(np.max(diagonal))
                if diagonal.size
                else 0.0
            )
            rank = int(np.sum(diagonal > rank_tolerance))
            if rank < len(active):
                active = [active[int(index)] for index in pivots[:rank]]
                active_array = np.asarray(active, dtype=np.int64)
                active_matrix = matrix[active_array]
            unconstrained_direction = np.linalg.solve(hessian, -gradient)
            weighted_transpose = np.linalg.solve(hessian, active_matrix.T)
            schur = active_matrix @ weighted_transpose
            schur_rhs = active_matrix @ unconstrained_direction
            try:
                equality_multipliers = np.linalg.solve(schur, schur_rhs)
            except np.linalg.LinAlgError:
                equality_multipliers = np.linalg.lstsq(
                    schur,
                    schur_rhs,
                    rcond=1.0e-12,
                )[0]
            direction = (
                unconstrained_direction
                - weighted_transpose @ equality_multipliers
            )
        else:
            direction = np.linalg.solve(hessian, -gradient)
            equality_multipliers = np.empty(0, dtype=np.float64)

        # At this scale a smaller direction is dominated by the dependent
        # active-cell equalities. The subsequent affine-minorant LP supplies
        # the actual optimality certificate, so no KKT claim relies on this stop.
        if float(np.linalg.norm(direction, ord=np.inf)) <= 1.0e-7:
            return (
                x,
                True,
                0,
                f"active_set_interval_certified_after_{iteration + 1}_iterations",
            )

        slack = base + matrix @ x
        slack_direction = matrix @ direction
        active_set = set(active)
        step = 1.0
        blocker: int | None = None
        for constraint_index in range(len(base)):
            if constraint_index in active_set or slack_direction[constraint_index] >= -1.0e-13:
                continue
            ratio = slack[constraint_index] / (-slack_direction[constraint_index])
            candidate_step = max(0.0, float(ratio))
            if candidate_step < step - 1.0e-13 or (
                abs(candidate_step - step) <= 1.0e-13
                and (blocker is None or constraint_index < blocker)
            ):
                step = candidate_step
                blocker = constraint_index
        x = x + step * direction
        if blocker is not None and step < 1.0 - 1.0e-10:
            active.append(blocker)

        if active:
            active_array = np.asarray(active, dtype=np.int64)
            active_matrix = matrix[active_array]
            correction_rhs = -base[active_array] - active_matrix @ x
            gram = active_matrix @ active_matrix.T
            try:
                correction_weights = np.linalg.solve(gram, correction_rhs)
            except np.linalg.LinAlgError:
                correction_weights = np.linalg.lstsq(
                    gram,
                    correction_rhs,
                    rcond=1.0e-12,
                )[0]
            correction = active_matrix.T @ correction_weights
            x = x + correction

        if step == 1.0 and blocker is None and equality_multipliers.size:
            # If the equality-constrained point is numerically stationary but
            # has a negative inequality multiplier, release that active cell.
            inequality_multipliers = -equality_multipliers
            if float(np.min(inequality_multipliers)) < -1.0e-8:
                active.pop(int(np.argmin(inequality_multipliers)))

    return x, False, 1, f"active_set_iteration_limit_{max_iterations}"


def solve_ccf_pair(
    pair: tuple[int, int],
    left_marginal: np.ndarray,
    right_marginal: np.ndarray,
    released_interaction_rate: np.ndarray,
    interaction_variance_rate: float | np.ndarray,
    *,
    confidence_radius: float,
    max_iterations: int = 2_000,
) -> CCFPairResult:
    left = np.asarray(left_marginal, dtype=np.float64)
    right = np.asarray(right_marginal, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or min(len(left), len(right)) < 2:
        raise ValueError("CCF pair marginals must be one-dimensional with size at least two")
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("CCF pair marginals must be strictly positive")
    if not np.isclose(float(np.sum(left)), 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Left CCF marginal must sum to one")
    if not np.isclose(float(np.sum(right)), 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Right CCF marginal must sum to one")
    radius = float(confidence_radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("confidence_radius must be finite and nonnegative")

    product, c_left, c_right, basis, product_theta = _pair_geometry(left, right)
    shape = product_theta.shape
    released = np.asarray(released_interaction_rate, dtype=np.float64)
    if released.shape != shape or not np.all(np.isfinite(released)):
        raise ValueError(f"released interaction must have shape {shape} and be finite")
    dimension = int(product_theta.size)
    raw_variance = np.asarray(interaction_variance_rate, dtype=np.float64)
    exact_covariance = bool(raw_variance.size > 0 and np.all(raw_variance == 0.0))
    if exact_covariance:
        if radius != 0.0:
            raise ValueError("Exact interaction covariance requires zero confidence radius")
        x_exact = (released - product_theta).reshape(-1)
        table_exact = (
            product.reshape(-1) + basis @ x_exact
        ).reshape(product.shape)
        nonnegative, row_residual, column_residual = _pair_residuals(
            table_exact,
            left,
            right,
        )
        zero_transport = _certified_transport_linear_minimum(
            np.zeros_like(product),
            left,
            right,
        )
        product_discrepancy = float(
            np.sum((product_theta - released) ** 2, dtype=np.float64)
        )
        if (
            nonnegative > CCF_NONNEGATIVITY_TOLERANCE
            or row_residual > CCF_MARGINAL_TOLERANCE
            or column_residual > CCF_MARGINAL_TOLERANCE
        ):
            exact_violation = max(nonnegative, row_residual, column_residual)
            return CCFPairResult(
                pair=pair,
                status="anchor_interaction_inconsistent",
                rank=0,
                confidence_radius=0.0,
                product_discrepancy=product_discrepancy,
                eta_upper=exact_violation,
                eta_lower=exact_violation,
                eta_gap=0.0,
                product_in_confidence=False,
                pair_table=None,
                kl_upper=0.0,
                kl_lower=0.0,
                kl_gap=0.0,
                weight_lower_bound=0.0,
                nonnegativity_violation=nonnegative,
                row_marginal_residual=row_residual,
                column_marginal_residual=column_residual,
                ellipsoid_violation=exact_violation,
                stationarity_residual=0.0,
                complementarity_residual=0.0,
                eta_solver_success=True,
                eta_solver_status=0,
                eta_solver_message="exact_zero_variance_unique_table_infeasible",
                kl_solver_success=True,
                kl_solver_status=0,
                kl_solver_message="not_run_inconsistent",
                eta_lp=zero_transport,
                kl_lp=None,
            )
        table_nonnegative = np.maximum(table_exact, 0.0)
        kl = float(
            np.sum(
                xlogy(table_nonnegative, table_nonnegative / product),
                dtype=np.float64,
            )
        )
        product_in_confidence = bool(
            np.max(np.abs(x_exact)) <= CCF_ELLIPSOID_TOLERANCE
        )
        lower = max(0.0, kl - 1.0e-12 * max(1.0, abs(kl)))
        return CCFPairResult(
            pair=pair,
            status=(
                "product_in_confidence"
                if product_in_confidence
                else ("eligible" if lower > 0.0 else "zero_certified_weight")
            ),
            rank=0,
            confidence_radius=0.0,
            product_discrepancy=product_discrepancy,
            eta_upper=0.0,
            eta_lower=0.0,
            eta_gap=0.0,
            product_in_confidence=product_in_confidence,
            pair_table=_readonly(table_nonnegative),
            kl_upper=kl,
            kl_lower=lower,
            kl_gap=max(0.0, kl - lower),
            weight_lower_bound=lower,
            nonnegativity_violation=nonnegative,
            row_marginal_residual=row_residual,
            column_marginal_residual=column_residual,
            ellipsoid_violation=0.0,
            stationarity_residual=0.0,
            complementarity_residual=0.0,
            eta_solver_success=True,
            eta_solver_status=0,
            eta_solver_message="exact_zero_variance_unique_table",
            kl_solver_success=True,
            kl_solver_status=0,
            kl_solver_message="exact_zero_variance_unique_optimum",
            eta_lp=zero_transport,
            kl_lp=None,
        )
    if raw_variance.ndim <= 1 or raw_variance.shape == shape:
        variance = np.broadcast_to(raw_variance, shape).copy().reshape(-1)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
            raise ValueError("interaction variance must be finite and positive")
        precision_matrix = np.diag(1.0 / variance)
        effective_rank = dimension
    elif raw_variance.shape == (dimension, dimension):
        covariance = 0.5 * (raw_variance + raw_variance.T)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("interaction covariance must be finite")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
        support = eigenvalues > tolerance
        if not np.any(support) or float(np.min(eigenvalues)) < -tolerance:
            raise ValueError("interaction covariance must be positive semidefinite")
        precision_matrix = (
            eigenvectors[:, support]
            * (1.0 / eigenvalues[support]).reshape(1, -1)
        ) @ eigenvectors[:, support].T
        precision_matrix = 0.5 * (precision_matrix + precision_matrix.T)
        effective_rank = int(np.sum(support))
    else:
        raise ValueError(
            "interaction variance must be scalar, coefficient-shaped, or a covariance matrix"
        )
    hessian_g = 2.0 * precision_matrix

    def table_from_x(x: np.ndarray) -> np.ndarray:
        return (product.reshape(-1) + basis @ np.asarray(x, dtype=np.float64)).reshape(
            product.shape
        )

    def theta_from_x(x: np.ndarray) -> np.ndarray:
        return product_theta + np.asarray(x, dtype=np.float64).reshape(shape)

    def discrepancy(x: np.ndarray) -> float:
        residual = (theta_from_x(x) - released).reshape(-1)
        return float(residual @ precision_matrix @ residual)

    def discrepancy_gradient_x(x: np.ndarray) -> np.ndarray:
        residual = (theta_from_x(x) - released).reshape(-1)
        return 2.0 * (precision_matrix @ residual)

    def discrepancy_gradient_table(x: np.ndarray) -> np.ndarray:
        local = discrepancy_gradient_x(x).reshape(shape)
        return c_left @ local @ c_right.T

    x_zero = np.zeros(dimension, dtype=np.float64)
    product_discrepancy = discrepancy(x_zero)
    product_in_confidence = bool(
        product_discrepancy <= radius + CCF_ELLIPSOID_TOLERANCE
    )
    if product_in_confidence:
        zero_transport = _certified_transport_linear_minimum(
            np.zeros_like(product),
            left,
            right,
        )
        nonnegative, row_residual, column_residual = _pair_residuals(
            product, left, right
        )
        return CCFPairResult(
            pair=pair,
            status="product_in_confidence",
            rank=effective_rank,
            confidence_radius=radius,
            product_discrepancy=product_discrepancy,
            eta_upper=product_discrepancy,
            eta_lower=0.0,
            eta_gap=product_discrepancy,
            product_in_confidence=True,
            pair_table=_readonly(product),
            kl_upper=0.0,
            kl_lower=0.0,
            kl_gap=0.0,
            weight_lower_bound=0.0,
            nonnegativity_violation=nonnegative,
            row_marginal_residual=row_residual,
            column_marginal_residual=column_residual,
            ellipsoid_violation=0.0,
            stationarity_residual=0.0,
            complementarity_residual=0.0,
            eta_solver_success=True,
            eta_solver_status=0,
            eta_solver_message="exact_product_feasibility_witness",
            kl_solver_success=True,
            kl_solver_status=0,
            kl_solver_message="exact_product_optimum",
            eta_lp=zero_transport,
            kl_lp=None,
        )

    x_unconstrained = (released - product_theta).reshape(-1)
    if float(np.min(table_from_x(x_unconstrained))) >= 0.0:
        x_eta = x_unconstrained
        eta_solver_success = True
        eta_solver_status = 0
        eta_solver_message = "exact_unconstrained_interaction_is_nonnegative"
    else:
        x_eta, eta_solver_success, eta_solver_status, eta_solver_message = (
            _solve_eta_active_set(
                hessian_g,
                2.0
                * (
                    precision_matrix
                    @ (product_theta - released).reshape(-1)
                ),
                basis,
                product.reshape(-1),
            )
        )
    table_eta = table_from_x(x_eta)
    eta_upper = discrepancy(x_eta)
    eta_gradient = discrepancy_gradient_table(x_eta)
    eta_transport = _certified_transport_linear_minimum(eta_gradient, left, right)
    eta_lower = float(
        eta_upper
        - np.sum(eta_gradient * table_eta, dtype=np.float64)
        + eta_transport.lower_bound
    )
    eta_lower -= 1.0e-12 * max(1.0, abs(eta_upper), abs(eta_lower))
    eta_gap = max(0.0, eta_upper - eta_lower)
    eta_nonnegative, eta_row_residual, eta_column_residual = _pair_residuals(
        table_eta, left, right
    )
    eta_certified = (
        eta_nonnegative <= CCF_NONNEGATIVITY_TOLERANCE
        and eta_row_residual <= CCF_MARGINAL_TOLERANCE
        and eta_column_residual <= CCF_MARGINAL_TOLERANCE
        and math.isfinite(eta_lower)
        and eta_lower
        <= eta_upper
        + CCF_ETA_RELATIVE_GAP_TOLERANCE * max(1.0, abs(eta_upper))
    )
    if not eta_certified:
        raise CCFCertificationError(
            f"CCF pair {pair} eta solve failed its convex certificate: "
            f"upper={eta_upper:.12g}, lower={eta_lower:.12g}, gap={eta_gap:.3g}"
        )

    if eta_lower > radius + CCF_ELLIPSOID_TOLERANCE:
        return CCFPairResult(
            pair=pair,
            status="anchor_interaction_inconsistent",
            rank=effective_rank,
            confidence_radius=radius,
            product_discrepancy=product_discrepancy,
            eta_upper=eta_upper,
            eta_lower=eta_lower,
            eta_gap=eta_gap,
            product_in_confidence=False,
            pair_table=None,
            kl_upper=0.0,
            kl_lower=0.0,
            kl_gap=0.0,
            weight_lower_bound=0.0,
            nonnegativity_violation=eta_nonnegative,
            row_marginal_residual=eta_row_residual,
            column_marginal_residual=eta_column_residual,
            ellipsoid_violation=max(0.0, eta_upper - radius),
            stationarity_residual=0.0,
            complementarity_residual=0.0,
            eta_solver_success=eta_solver_success,
            eta_solver_status=eta_solver_status,
            eta_solver_message=eta_solver_message,
            kl_solver_success=True,
            kl_solver_status=0,
            kl_solver_message="not_run_inconsistent",
            eta_lp=eta_transport,
            kl_lp=None,
        )
    if eta_upper > radius + CCF_ELLIPSOID_TOLERANCE:
        raise CCFCertificationError(
            f"CCF pair {pair} consistency classification is unresolved: "
            f"eta=[{eta_lower:.12g}, {eta_upper:.12g}], radius={radius:.12g}"
        )

    # Find a strictly positive point inside the confidence slice on the segment
    # from the product table to the certified eta solution.
    low = 0.0
    high = 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if discrepancy(middle * x_eta) <= radius:
            high = middle
        else:
            low = middle
    x_start = high * x_eta
    if float(np.min(table_from_x(x_start))) <= 0.0:
        raise CCFCertificationError(
            f"CCF pair {pair} has no numerically certified strictly positive "
            "Slater point for the KL program"
        )

    def kl_objective(x: np.ndarray) -> float:
        table = table_from_x(x)
        if np.min(table) < -CCF_NONNEGATIVITY_TOLERANCE:
            return 1.0e100
        clipped = np.maximum(table, 0.0)
        return float(np.sum(xlogy(clipped, clipped / product), dtype=np.float64))

    def kl_gradient_x(x: np.ndarray) -> np.ndarray:
        table = table_from_x(x)
        safe = np.maximum(table, np.finfo(np.float64).tiny)
        gradient_table = np.log(safe / product) + 1.0
        return np.asarray(basis.T @ gradient_table.reshape(-1), dtype=np.float64)

    # The product is outside the ellipsoid, so the unique KL optimum has an
    # active ellipsoid constraint. KL's derivative diverges at zero and the
    # segment above supplies a positive Slater point, hence the optimum is
    # interior to the transportation polytope. Solve its smooth KKT system with
    # damped Newton steps, then independently certify it with the LP lower bound.
    x_star = np.asarray(x_start, dtype=np.float64).copy()
    initial_grad_f = kl_gradient_x(x_star)
    initial_grad_g = discrepancy_gradient_x(x_star)
    denominator = float(initial_grad_g @ initial_grad_g)
    multiplier = max(
        1.0e-15,
        -float(initial_grad_f @ initial_grad_g) / denominator,
    )
    kl_solver_success = False
    kl_solver_status = 1
    kl_solver_message = "maximum_kkt_iterations_reached"
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    kl_iterations = int(max_iterations)
    for iteration in range(kl_iterations):
        table_current = table_from_x(x_star)
        if float(np.min(table_current)) <= 0.0:
            raise CCFCertificationError("CCF KL Newton iterate left the positive domain")
        grad_f = kl_gradient_x(x_star)
        grad_g = discrepancy_gradient_x(x_star)
        constraint_residual = discrepancy(x_star) - radius
        stationarity_vector = grad_f + multiplier * grad_g
        kkt_residual = np.concatenate(
            [stationarity_vector, np.asarray([constraint_residual])]
        )
        if float(np.max(np.abs(kkt_residual))) <= CCF_NEWTON_KKT_TOLERANCE:
            kl_solver_success = True
            kl_solver_status = 0
            kl_solver_message = "damped_newton_kkt_converged"
            break

        table_flat = table_current.reshape(-1)
        hessian_f = basis.T @ (basis / table_flat.reshape(-1, 1))
        hessian = np.asarray(hessian_f, dtype=np.float64)
        hessian += multiplier * hessian_g
        kkt_matrix = np.empty((dimension + 1, dimension + 1), dtype=np.float64)
        kkt_matrix[:-1, :-1] = hessian
        kkt_matrix[:-1, -1] = grad_g
        kkt_matrix[-1, :-1] = grad_g
        kkt_matrix[-1, -1] = 0.0
        try:
            direction = np.linalg.solve(kkt_matrix, -kkt_residual)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(kkt_matrix, -kkt_residual, rcond=None)[0]
        delta_x = direction[:-1]
        delta_multiplier = float(direction[-1])
        delta_table = basis @ delta_x
        step = 1.0
        decreasing_cells = delta_table < 0.0
        if np.any(decreasing_cells):
            step = min(
                step,
                0.99
                * float(
                    np.min(
                        -table_flat[decreasing_cells]
                        / delta_table[decreasing_cells]
                    )
                ),
            )
        if delta_multiplier < 0.0:
            step = min(step, 0.99 * (-multiplier / delta_multiplier))
        merit = float(np.linalg.norm(kkt_residual))
        accepted = False
        for _ in range(80):
            candidate_x = x_star + step * delta_x
            candidate_multiplier = multiplier + step * delta_multiplier
            if (
                candidate_multiplier >= 0.0
                and float(np.min(table_from_x(candidate_x))) > 0.0
            ):
                candidate_residual = np.concatenate(
                    [
                        kl_gradient_x(candidate_x)
                        + candidate_multiplier
                        * discrepancy_gradient_x(candidate_x),
                        np.asarray([discrepancy(candidate_x) - radius]),
                    ]
                )
                if float(np.linalg.norm(candidate_residual)) <= (
                    1.0 - 1.0e-4 * step
                ) * merit:
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            kl_solver_status = 2
            kl_solver_message = "damped_newton_line_search_failed"
            break
        x_star = candidate_x
        multiplier = candidate_multiplier

    def kl_certificate_at(
        candidate_x: np.ndarray,
        candidate_multiplier: float,
    ) -> dict[str, Any]:
        local_x = np.asarray(candidate_x, dtype=np.float64)
        local_multiplier = max(0.0, float(candidate_multiplier))
        local_table = table_from_x(local_x)
        local_upper = kl_objective(local_x)
        local_discrepancy = discrepancy(local_x)
        gradient_f_table = np.log(
            np.maximum(local_table, np.finfo(np.float64).tiny) / product
        ) + 1.0
        gradient_g_table = discrepancy_gradient_table(local_x)
        lagrangian_gradient = (
            gradient_f_table + local_multiplier * gradient_g_table
        )
        transport = _certified_transport_linear_minimum(
            lagrangian_gradient,
            left,
            right,
        )
        linear_value = float(
            np.sum(lagrangian_gradient * local_table, dtype=np.float64)
        )
        local_lower = float(
            local_upper
            + local_multiplier * (local_discrepancy - radius)
            - linear_value
            + transport.lower_bound
        )
        local_lower -= 1.0e-12 * max(
            1.0,
            abs(local_upper),
            abs(local_lower),
        )
        local_gap = max(0.0, local_upper - local_lower)
        local_nonnegative, local_row, local_column = _pair_residuals(
            local_table,
            left,
            right,
        )
        local_ellipsoid = max(0.0, local_discrepancy - radius)
        local_stationarity = float(
            max(0.0, linear_value - transport.lower_bound)
            / max(1.0, abs(local_upper), abs(linear_value))
        )
        local_complementarity = float(
            abs(local_multiplier * (local_discrepancy - radius))
            / max(1.0, abs(local_upper))
        )
        local_relative_gap = local_gap / max(abs(local_upper), 1.0e-12)
        certified = (
            local_nonnegative <= CCF_NONNEGATIVITY_TOLERANCE
            and local_row <= CCF_MARGINAL_TOLERANCE
            and local_column <= CCF_MARGINAL_TOLERANCE
            and local_ellipsoid <= CCF_ELLIPSOID_TOLERANCE
            and (
                local_gap <= CCF_KL_ABSOLUTE_GAP_TOLERANCE
                or local_relative_gap <= CCF_KL_RELATIVE_GAP_TOLERANCE
            )
            and local_stationarity <= CCF_STATIONARITY_TOLERANCE
            and local_complementarity <= CCF_COMPLEMENTARITY_TOLERANCE
        )
        return {
            "x": local_x,
            "multiplier": local_multiplier,
            "table": local_table,
            "kl_upper": local_upper,
            "kl_lower": local_lower,
            "kl_gap": local_gap,
            "weight_lower_bound": max(0.0, local_lower),
            "nonnegative": local_nonnegative,
            "row_residual": local_row,
            "column_residual": local_column,
            "ellipsoid_violation": local_ellipsoid,
            "stationarity": local_stationarity,
            "complementarity": local_complementarity,
            "transport": transport,
            "certified": certified,
        }

    def refine_certificate_multiplier(
        candidate_x: np.ndarray,
        initial_multiplier_value: float,
    ) -> tuple[float, dict[str, Any]]:
        local_x = np.asarray(candidate_x, dtype=np.float64)
        gradient_f = kl_gradient_x(local_x)
        gradient_g = discrepancy_gradient_x(local_x)
        gradient_scale = max(
            1.0e-12,
            float(np.linalg.norm(gradient_f))
            / max(float(np.linalg.norm(gradient_g)), 1.0e-12),
            float(initial_multiplier_value),
        )
        log_grid = np.linspace(-10.0, 10.0, 41) + math.log(gradient_scale)
        candidates: list[tuple[float, dict[str, Any]]] = [
            (0.0, kl_certificate_at(local_x, 0.0))
        ]
        for log_multiplier in log_grid:
            local_value = float(math.exp(float(log_multiplier)))
            candidates.append(
                (local_value, kl_certificate_at(local_x, local_value))
            )
        best_index = max(
            range(len(candidates)),
            key=lambda index: candidates[index][1]["kl_lower"],
        )
        best_multiplier, best_certificate = candidates[best_index]
        grid_index = best_index - 1
        if 0 < grid_index < len(log_grid) - 1:
            refined = minimize_scalar(
                lambda log_value: -kl_certificate_at(
                    local_x,
                    math.exp(float(log_value)),
                )["kl_lower"],
                bounds=(
                    float(log_grid[grid_index - 1]),
                    float(log_grid[grid_index + 1]),
                ),
                method="bounded",
                options={"xatol": 1.0e-10, "maxiter": 100},
            )
            refined_multiplier = float(math.exp(float(refined.x)))
            refined_certificate = kl_certificate_at(
                local_x,
                refined_multiplier,
            )
            if refined_certificate["kl_lower"] > best_certificate["kl_lower"]:
                best_multiplier = refined_multiplier
                best_certificate = refined_certificate
        return best_multiplier, best_certificate

    certificate = kl_certificate_at(x_star, multiplier)
    if not certificate["certified"]:
        refined_multiplier, refined_certificate = refine_certificate_multiplier(
            x_star,
            multiplier,
        )
        if refined_certificate["kl_lower"] > certificate["kl_lower"]:
            multiplier = refined_multiplier
            certificate = refined_certificate
            kl_solver_message = f"{kl_solver_message}_dual_multiplier_refined"
    slsqp_diagnostic = "not_run"
    if not certificate["certified"]:
        precision_diagonal = np.maximum(
            np.diag(precision_matrix),
            np.finfo(np.float64).tiny,
        )
        coordinate_scale = 1.0 / np.sqrt(precision_diagonal)

        def x_from_scaled(value: np.ndarray) -> np.ndarray:
            return x_start + coordinate_scale * np.asarray(value, dtype=np.float64)

        slsqp = minimize(
            lambda value: kl_objective(x_from_scaled(value)),
            np.zeros(dimension, dtype=np.float64),
            jac=lambda value: (
                coordinate_scale * kl_gradient_x(x_from_scaled(value))
            ),
            method="SLSQP",
            constraints=(
                {
                    "type": "ineq",
                    "fun": lambda value: table_from_x(
                        x_from_scaled(value)
                    ).reshape(-1),
                    "jac": lambda value: basis * coordinate_scale.reshape(1, -1),
                },
                {
                    "type": "ineq",
                    "fun": lambda value: radius
                    - discrepancy(x_from_scaled(value)),
                    "jac": lambda value: -coordinate_scale
                    * discrepancy_gradient_x(x_from_scaled(value)),
                },
            ),
            options={
                "maxiter": int(max_iterations),
                "ftol": 1.0e-13,
                "disp": False,
            },
        )
        candidate_x = x_from_scaled(np.asarray(slsqp.x, dtype=np.float64))
        candidate_table = table_from_x(candidate_x)
        slsqp_diagnostic = (
            f"status={int(slsqp.status)},success={bool(slsqp.success)},"
            f"message={slsqp.message},min_cell={float(np.min(candidate_table)):.3g},"
            f"ellipsoid={discrepancy(candidate_x) - radius:.3g},"
            f"objective={kl_objective(candidate_x):.12g}"
        )
        if (
            np.all(np.isfinite(candidate_x))
            and float(np.min(candidate_table)) >= -CCF_NONNEGATIVITY_TOLERANCE
            and discrepancy(candidate_x) <= radius + CCF_ELLIPSOID_TOLERANCE
        ):
            gradient_f = kl_gradient_x(candidate_x)
            gradient_g = discrepancy_gradient_x(candidate_x)
            denominator = float(gradient_g @ gradient_g)
            candidate_multiplier = (
                max(0.0, -float(gradient_f @ gradient_g) / denominator)
                if denominator > 0.0
                else 0.0
            )
            candidate_certificate = kl_certificate_at(
                candidate_x,
                candidate_multiplier,
            )
            if not candidate_certificate["certified"]:
                candidate_multiplier, candidate_certificate = (
                    refine_certificate_multiplier(
                        candidate_x,
                        candidate_multiplier,
                    )
                )
            slsqp_diagnostic += (
                f",dual_lower={candidate_certificate['kl_lower']:.12g},"
                f"gap={candidate_certificate['kl_gap']:.3g},"
                f"stationarity={candidate_certificate['stationarity']:.3g},"
                f"complementarity={candidate_certificate['complementarity']:.3g}"
            )
            if (
                candidate_certificate["certified"]
                or candidate_certificate["kl_lower"] > certificate["kl_lower"]
            ):
                certificate = candidate_certificate
                x_star = candidate_x
                multiplier = candidate_multiplier
                kl_solver_status = int(slsqp.status)
                kl_solver_message = (
                    "slsqp_candidate_certified"
                    if candidate_certificate["certified"]
                    else f"slsqp_candidate_{slsqp.message}"
                )

    barrier_diagnostic = "not_run"
    if not certificate["certified"]:
        barrier_x = x_start + 0.01 * (x_eta - x_start)

        def barrier_value(value: np.ndarray, barrier_weight: float) -> float:
            local_table = table_from_x(value).reshape(-1)
            local_slack = radius - discrepancy(value)
            if float(np.min(local_table)) <= 0.0 or local_slack <= 0.0:
                return math.inf
            return float(
                kl_objective(value)
                - barrier_weight * np.sum(np.log(local_table), dtype=np.float64)
                - barrier_weight * math.log(local_slack)
            )

        barrier_iterations = 0
        barrier_line_search_failed = False
        for barrier_weight in np.geomspace(1.0e-2, 1.0e-12, 21):
            for _ in range(100):
                barrier_iterations += 1
                local_table = table_from_x(barrier_x).reshape(-1)
                local_slack = radius - discrepancy(barrier_x)
                if float(np.min(local_table)) <= 0.0 or local_slack <= 0.0:
                    barrier_line_search_failed = True
                    break
                gradient_g = discrepancy_gradient_x(barrier_x)
                gradient = (
                    kl_gradient_x(barrier_x)
                    - barrier_weight * (basis.T @ (1.0 / local_table))
                    + barrier_weight * gradient_g / local_slack
                )
                hessian_f = basis.T @ (
                    basis / local_table.reshape(-1, 1)
                )
                hessian_barrier = basis.T @ (
                    basis / (local_table * local_table).reshape(-1, 1)
                )
                hessian = (
                    hessian_f
                    + barrier_weight * hessian_barrier
                    + barrier_weight * hessian_g / local_slack
                    + barrier_weight
                    * np.outer(gradient_g, gradient_g)
                    / (local_slack * local_slack)
                )
                try:
                    direction = np.linalg.solve(hessian, -gradient)
                except np.linalg.LinAlgError:
                    direction = np.linalg.lstsq(hessian, -gradient, rcond=1.0e-12)[0]
                directional_derivative = float(gradient @ direction)
                if directional_derivative >= 0.0:
                    barrier_line_search_failed = True
                    break
                if -0.5 * directional_derivative <= max(
                    1.0e-14,
                    1.0e-3 * float(barrier_weight),
                ):
                    break
                current_value = barrier_value(barrier_x, float(barrier_weight))
                table_direction = basis @ direction
                step = 1.0
                decreasing = table_direction < 0.0
                if np.any(decreasing):
                    step = min(
                        step,
                        0.99
                        * float(
                            np.min(
                                -local_table[decreasing]
                                / table_direction[decreasing]
                            )
                        ),
                    )
                accepted = False
                for _ in range(100):
                    candidate_x = barrier_x + step * direction
                    candidate_value = barrier_value(
                        candidate_x,
                        float(barrier_weight),
                    )
                    if candidate_value <= (
                        current_value
                        + 1.0e-4 * step * directional_derivative
                    ):
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    barrier_line_search_failed = True
                    break
                barrier_x = candidate_x
            if barrier_line_search_failed:
                break

        barrier_table = table_from_x(barrier_x)
        barrier_diagnostic = (
            f"iterations={barrier_iterations},"
            f"line_search_failed={barrier_line_search_failed},"
            f"min_cell={float(np.min(barrier_table)):.3g},"
            f"ellipsoid={discrepancy(barrier_x) - radius:.3g},"
            f"objective={kl_objective(barrier_x):.12g}"
        )
        if (
            np.all(np.isfinite(barrier_x))
            and float(np.min(barrier_table)) >= -CCF_NONNEGATIVITY_TOLERANCE
            and discrepancy(barrier_x) <= radius + CCF_ELLIPSOID_TOLERANCE
        ):
            barrier_gradient_f = kl_gradient_x(barrier_x)
            barrier_gradient_g = discrepancy_gradient_x(barrier_x)
            barrier_denominator = float(barrier_gradient_g @ barrier_gradient_g)
            barrier_multiplier = (
                max(
                    0.0,
                    -float(barrier_gradient_f @ barrier_gradient_g)
                    / barrier_denominator,
                )
                if barrier_denominator > 0.0
                else 0.0
            )
            barrier_multiplier, barrier_certificate = refine_certificate_multiplier(
                barrier_x,
                barrier_multiplier,
            )
            barrier_diagnostic += (
                f",dual_lower={barrier_certificate['kl_lower']:.12g},"
                f"gap={barrier_certificate['kl_gap']:.3g},"
                f"stationarity={barrier_certificate['stationarity']:.3g},"
                f"complementarity={barrier_certificate['complementarity']:.3g}"
            )
            if (
                barrier_certificate["certified"]
                or barrier_certificate["kl_lower"] > certificate["kl_lower"]
            ):
                certificate = barrier_certificate
                x_star = barrier_x
                multiplier = barrier_multiplier
                kl_solver_status = 0 if barrier_certificate["certified"] else 1
                kl_solver_message = (
                    "log_barrier_candidate_certified"
                    if barrier_certificate["certified"]
                    else "log_barrier_candidate_uncertified"
                )

    if not certificate["certified"]:
        local_x = np.asarray(certificate["x"], dtype=np.float64)
        best_multiplier, best_certificate = refine_certificate_multiplier(
            local_x,
            float(certificate["multiplier"]),
        )
        if best_certificate["kl_lower"] > certificate["kl_lower"]:
            certificate = best_certificate
            multiplier = best_multiplier
            kl_solver_message = f"{kl_solver_message}_dual_multiplier_refined"

    table_star = np.asarray(certificate["table"], dtype=np.float64)
    kl_upper = float(certificate["kl_upper"])
    kl_lower = float(certificate["kl_lower"])
    kl_gap = float(certificate["kl_gap"])
    weight_lower_bound = float(certificate["weight_lower_bound"])
    nonnegative = float(certificate["nonnegative"])
    row_residual = float(certificate["row_residual"])
    column_residual = float(certificate["column_residual"])
    ellipsoid_violation = float(certificate["ellipsoid_violation"])
    stationarity = float(certificate["stationarity"])
    complementarity = float(certificate["complementarity"])
    kl_transport = certificate["transport"]
    kl_certified = bool(certificate["certified"])
    if not kl_certified:
        raise CCFCertificationError(
            f"CCF pair {pair} KL solve failed its convex certificate: "
            f"kl={kl_upper:.12g}, lower={kl_lower:.12g}, gap={kl_gap:.3g}, "
            f"stationarity={stationarity:.3g}, complementarity={complementarity:.3g}, "
            f"solver={kl_solver_message}, slsqp=({slsqp_diagnostic}), "
            f"barrier=({barrier_diagnostic})"
        )
    if not kl_solver_success:
        kl_solver_message = (
            f"transport_gap_certified_after_{kl_solver_message}"
        )
        kl_solver_success = True
        kl_solver_status = 0
    status = "eligible" if weight_lower_bound > 0.0 else "zero_certified_weight"
    return CCFPairResult(
        pair=pair,
        status=status,
        rank=effective_rank,
        confidence_radius=radius,
        product_discrepancy=product_discrepancy,
        eta_upper=eta_upper,
        eta_lower=eta_lower,
        eta_gap=eta_gap,
        product_in_confidence=False,
        pair_table=_readonly(table_star),
        kl_upper=kl_upper,
        kl_lower=kl_lower,
        kl_gap=kl_gap,
        weight_lower_bound=weight_lower_bound,
        nonnegativity_violation=nonnegative,
        row_marginal_residual=row_residual,
        column_marginal_residual=column_residual,
        ellipsoid_violation=ellipsoid_violation,
        stationarity_residual=stationarity,
        complementarity_residual=complementarity,
        eta_solver_success=eta_solver_success,
        eta_solver_status=eta_solver_status,
        eta_solver_message=eta_solver_message,
        kl_solver_success=kl_solver_success,
        kl_solver_status=kl_solver_status,
        kl_solver_message=kl_solver_message,
        eta_lp=eta_transport,
        kl_lp=kl_transport,
    )


@dataclass(frozen=True)
class ConfidenceForestEdge:
    pair: tuple[int, int]
    table: np.ndarray
    unsmoothed_table: np.ndarray
    weight: float
    weight_lower_bound: float

    def __post_init__(self) -> None:
        if self.pair[0] >= self.pair[1]:
            raise ValueError("Forest edges must use canonical attribute order")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("Forest edge weight must be finite and positive")
        if not math.isfinite(float(self.weight_lower_bound)) or float(
            self.weight_lower_bound
        ) <= 0.0:
            raise ValueError("Forest edge certified weight must be finite and positive")
        object.__setattr__(self, "table", _readonly(self.table))
        object.__setattr__(self, "unsmoothed_table", _readonly(self.unsmoothed_table))


def _maximum_weight_forest(
    pair_results: tuple[CCFPairResult, ...],
    dimension: int,
) -> tuple[CCFPairResult, ...]:
    parent = list(range(int(dimension)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    selected: list[CCFPairResult] = []
    eligible = sorted(
        (result for result in pair_results if result.eligible),
        key=lambda result: (
            -float(result.weight_lower_bound),
            int(result.pair[0]),
            int(result.pair[1]),
        ),
    )
    for result in eligible:
        left_root = find(result.pair[0])
        right_root = find(result.pair[1])
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        selected.append(result)
    return tuple(selected)


@dataclass(frozen=True)
class ReleasedConfidenceForestPrior:
    product_prior: ReleasedProductPrior
    edges: tuple[ConfidenceForestEdge, ...]
    pair_results: tuple[CCFPairResult, ...]
    support_coarsening: SupportCoarsening | None = None
    alpha_struct: float = CCF_ALPHA_STRUCT
    method: str = CCF_PRIOR_METHOD
    _forest_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {CCF_PRIOR_METHOD, SEQUENTIAL_CCF_PRIOR_METHOD}:
            raise ValueError(f"Unsupported CCF prior method {self.method!r}")
        if not np.isclose(
            float(self.alpha_struct), CCF_ALPHA_STRUCT, rtol=0.0, atol=0.0
        ):
            raise ValueError("CCF-v1 requires alpha_struct=0.05")
        seen: set[tuple[int, int]] = set()
        parent = list(range(self.product_prior.dimension))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for edge in self.edges:
            if edge.pair in seen:
                raise ValueError("CCF forest contains a duplicate edge")
            seen.add(edge.pair)
            left, right = edge.pair
            if min(left, right) < 0 or max(left, right) >= self.product_prior.dimension:
                raise ValueError("CCF forest edge is out of range")
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                raise ValueError("CCF selected edges must form a forest")
            parent[right_root] = left_root
            expected_shape = (
                self.cardinalities[left],
                self.cardinalities[right],
            )
            if edge.table.shape != expected_shape:
                raise ValueError("CCF edge table shape does not match cardinalities")
            q_left = self.probabilities[left]
            q_right = self.probabilities[right]
            if np.any(edge.table <= 0.0) or not np.isclose(
                float(np.sum(edge.table)), 1.0, rtol=1.0e-12, atol=1.0e-12
            ):
                raise ValueError("CCF smoothed edge tables must be positive distributions")
            if np.max(np.abs(np.sum(edge.table, axis=1) - q_left)) > 1.0e-10:
                raise ValueError("CCF edge row marginal does not match its node marginal")
            if np.max(np.abs(np.sum(edge.table, axis=0) - q_right)) > 1.0e-10:
                raise ValueError("CCF edge column marginal does not match its node marginal")
        payload = [
            {
                "pair": list(edge.pair),
                "weight_lower_bound": edge.weight_lower_bound,
                "table_sha256": _array_sha256(edge.table),
            }
            for edge in self.edges
        ]
        forest_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "_forest_hash", forest_hash)

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return self.product_prior.cardinalities

    @property
    def probabilities(self) -> tuple[np.ndarray, ...]:
        return self.product_prior.probabilities

    @property
    def public_total(self) -> int:
        return self.product_prior.public_total

    @property
    def smoothing(self) -> float:
        return self.product_prior.smoothing

    @property
    def dimension(self) -> int:
        return self.product_prior.dimension

    @property
    def domain_size(self) -> int:
        return self.product_prior.domain_size

    @property
    def forest_hash(self) -> str:
        return self._forest_hash

    def encode_rows(self, rows: np.ndarray) -> np.ndarray:
        return self.product_prior.encode_rows(rows)

    def log_probability_rows(self, rows: np.ndarray) -> np.ndarray:
        array = np.asarray(rows, dtype=np.int64)
        result = self.product_prior.log_probability_rows(array)
        for edge in self.edges:
            left, right = edge.pair
            result += np.log(edge.table[array[:, left], array[:, right]])
            result -= np.log(self.probabilities[left][array[:, left]])
            result -= np.log(self.probabilities[right][array[:, right]])
        if not np.all(np.isfinite(result)):
            raise RuntimeError("CCF prior produced a non-finite row log probability")
        return result

    def diagnostics(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for result in self.pair_results:
            statuses[result.status] = statuses.get(result.status, 0) + 1
        return {
            "method": self.method,
            "public_total": self.public_total,
            "smoothing": self.smoothing,
            "domain_size": self.domain_size,
            "alpha_struct": self.alpha_struct,
            "candidate_edge_count": len(self.pair_results),
            "eligible_edge_count": sum(result.eligible for result in self.pair_results),
            "selected_edge_count": len(self.edges),
            "edge_status_counts": statuses,
            "product_in_set_fraction": (
                float(np.mean([result.product_in_confidence for result in self.pair_results]))
                if self.pair_results
                else 1.0
            ),
            "selected_sum_kl": float(sum(edge.weight for edge in self.edges)),
            "selected_sum_weight_lower_bound": float(
                sum(edge.weight_lower_bound for edge in self.edges)
            ),
            "forest_hash": self.forest_hash,
            "selected_edges": [
                {
                    "pair": list(edge.pair),
                    "kl": edge.weight,
                    "weight_lower_bound": edge.weight_lower_bound,
                    "table_sha256": _array_sha256(edge.table),
                }
                for edge in self.edges
            ],
            "pair_programs": [result.to_dict() for result in self.pair_results],
            "support_coarsening": (
                self.support_coarsening.to_dict()
                if self.support_coarsening is not None
                else None
            ),
            "product_prior": self.product_prior.diagnostics(),
        }


def _coarsened_pair_problem(
    product_prior: ReleasedProductPrior,
    transcript: HierarchicalInteractionTranscript,
    pair: tuple[int, int],
    support_coarsening: SupportCoarsening,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right = pair
    left_map = support_coarsening.attributes[left]
    right_map = support_coarsening.attributes[right]
    q_left = product_prior.probabilities[left]
    q_right = product_prior.probabilities[right]
    coarse_left = left_map.aggregate_probabilities(q_left)
    coarse_right = right_map.aggregate_probabilities(q_right)

    original_left_contrast = helmert_contrast(len(q_left))
    original_right_contrast = helmert_contrast(len(q_right))
    coarse_left_contrast = helmert_contrast(len(coarse_left))
    coarse_right_contrast = helmert_contrast(len(coarse_right))
    aggregation_left = left_map.aggregation_matrix()
    aggregation_right = right_map.aggregation_matrix()

    name = f"pair_interaction:{left}:{right}"
    released = transcript.noisy_components[name] / float(transcript.public_total)
    original_product = np.outer(q_left, q_right)
    product_theta = (
        original_left_contrast.T
        @ original_product
        @ original_right_contrast
    )
    # Anchor the released interaction center to the released one-way marginals,
    # aggregate categories, and express the result in the coarse contrast basis.
    anchored_center = original_product + (
        original_left_contrast
        @ (released - product_theta)
        @ original_right_contrast.T
    )
    coarse_center = aggregation_left @ anchored_center @ aggregation_right.T
    coarse_released = (
        coarse_left_contrast.T @ coarse_center @ coarse_right_contrast
    )

    left_operator = (
        coarse_left_contrast.T @ aggregation_left @ original_left_contrast
    )
    right_operator = (
        original_right_contrast.T
        @ aggregation_right.T
        @ coarse_right_contrast
    )
    row_major_operator = np.kron(left_operator, right_operator.T)
    variance_rate = (
        float(transcript.component_variances[name])
        / float(transcript.public_total) ** 2
    )
    coarse_covariance = variance_rate * (
        row_major_operator @ row_major_operator.T
    )
    return coarse_left, coarse_right, coarse_released, coarse_covariance


def build_released_confidence_forest_prior(
    qcat: QueryCatalogue,
    released_target: np.ndarray,
    cardinalities: np.ndarray | tuple[int, ...],
    transcript: HierarchicalInteractionTranscript,
    *,
    public_total: int,
    smoothing: float = 1.0,
    alpha_struct: float = CCF_ALPHA_STRUCT,
    support_coarsening: SupportCoarsening | None = None,
) -> ReleasedConfidenceForestPrior:
    if not np.isclose(float(alpha_struct), CCF_ALPHA_STRUCT, rtol=0.0, atol=0.0):
        raise ValueError("CCF-v1 requires alpha_struct=0.05")
    if int(public_total) != int(transcript.public_total):
        raise ValueError("CCF public_total must match the released transcript")
    cards = tuple(int(value) for value in np.asarray(cardinalities, dtype=np.int64))
    if cards != transcript.strategy.cardinalities:
        raise ValueError("CCF cardinalities must match the released transcript")
    if (
        support_coarsening is not None
        and support_coarsening.original_cardinalities != cards
    ):
        raise ValueError("CCF support coarsening must match the public cardinalities")
    product_prior = ReleasedProductPrior.from_released_oneway(
        qcat,
        released_target,
        cards,
        public_total=int(public_total),
        smoothing=float(smoothing),
    )
    pairs = tuple(transcript.strategy.pairs)
    if not pairs:
        return ReleasedConfidenceForestPrior(
            product_prior=product_prior,
            edges=(),
            pair_results=(),
            support_coarsening=support_coarsening,
        )
    simultaneous_probability = 1.0 - float(alpha_struct) / float(len(pairs))
    pair_results: list[CCFPairResult] = []
    n_squared = float(public_total) ** 2
    for left, right in pairs:
        name = f"pair_interaction:{left}:{right}"
        block = transcript.strategy.block(name)
        if support_coarsening is None:
            left_marginal = product_prior.probabilities[left]
            right_marginal = product_prior.probabilities[right]
            interaction_center = (
                transcript.noisy_components[name] / float(public_total)
            )
            interaction_covariance: float | np.ndarray = (
                transcript.component_variances[name] / n_squared
            )
            expected_rank = (
                0
                if float(interaction_covariance) == 0.0
                else int(np.prod(block.coefficient_shape, dtype=np.int64))
            )
        else:
            (
                left_marginal,
                right_marginal,
                interaction_center,
                interaction_covariance,
            ) = _coarsened_pair_problem(
                product_prior,
                transcript,
                (left, right),
                support_coarsening,
            )
            expected_rank = int(
                np.linalg.matrix_rank(
                    interaction_covariance,
                    tol=1.0e-12
                    * max(
                        1.0,
                        float(np.max(np.abs(interaction_covariance))),
                    ),
                )
            )
            if expected_rank <= 0 and not np.all(
                np.asarray(interaction_covariance, dtype=np.float64) == 0.0
            ):
                raise RuntimeError(f"CCF support map erased pair block {name}")
        radius = (
            0.0
            if expected_rank == 0
            else float(chi2.ppf(simultaneous_probability, expected_rank))
        )
        if expected_rank > 0 and (not math.isfinite(radius) or radius <= 0.0):
            raise RuntimeError(f"CCF produced an invalid confidence radius for {name}")
        result = solve_ccf_pair(
            (left, right),
            left_marginal,
            right_marginal,
            interaction_center,
            interaction_covariance,
            confidence_radius=radius,
        )
        pair_results.append(result)

    frozen_results = tuple(pair_results)
    selected_results = _maximum_weight_forest(frozen_results, len(cards))
    lambda_n = 1.0 / float(public_total + 1)
    edges: list[ConfidenceForestEdge] = []
    for result in selected_results:
        if result.pair_table is None:
            raise AssertionError("Selected CCF edge is missing its pair table")
        left, right = result.pair
        unsmoothed = result.pair_table
        if support_coarsening is not None:
            unsmoothed = lift_coarse_pair_distribution(
                unsmoothed,
                product_prior.probabilities[left],
                product_prior.probabilities[right],
                support_coarsening.attributes[left],
                support_coarsening.attributes[right],
            )
        product = np.outer(
            product_prior.probabilities[left],
            product_prior.probabilities[right],
        )
        smoothed = (1.0 - lambda_n) * unsmoothed + lambda_n * product
        edges.append(
            ConfidenceForestEdge(
                pair=result.pair,
                table=smoothed,
                unsmoothed_table=unsmoothed,
                weight=result.kl_upper,
                weight_lower_bound=result.weight_lower_bound,
            )
        )
    return ReleasedConfidenceForestPrior(
        product_prior=product_prior,
        edges=tuple(edges),
        pair_results=frozen_results,
        support_coarsening=support_coarsening,
    )


__all__ = [
    "CCF_ALPHA_STRUCT",
    "CCF_PRIOR_METHOD",
    "CCFCertificationError",
    "CCFPairResult",
    "ConfidenceForestEdge",
    "ReleasedConfidenceForestPrior",
    "SEQUENTIAL_CCF_PRIOR_METHOD",
    "build_released_confidence_forest_prior",
    "solve_ccf_pair",
]
