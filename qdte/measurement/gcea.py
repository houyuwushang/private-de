from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.sparse import csr_matrix

from qdte.measurement.factorization import (
    HierarchicalPairStrategy,
    validate_strategy_rho_override,
)
from qdte.measurement.workload_factorization import (
    factorize_public_query,
    public_query_indicator,
)
from qdte.queries.orthogonal import helmert_contrast, pair_reconstruction_maps
from qdte.queries.types import QueryCatalogue


PRIMARY_NAMES = ("mae", "rmse", "avg_tvd")
TAIL_NAMES = ("max_error", "max_tvd")
RISK_NAMES = (*PRIMARY_NAMES, *TAIL_NAMES)


@dataclass(frozen=True)
class GCEAProfile:
    strategy: HierarchicalPairStrategy
    public_total: int
    beta_tail: float
    query_ids: np.ndarray
    unit_variance: csr_matrix
    fixed_variance: np.ndarray
    partition_names: tuple[str, ...]
    partition_rows: tuple[np.ndarray, ...]
    movable_block_indices: np.ndarray
    fixed_block_indices: np.ndarray
    control_rho: np.ndarray
    reconstruction_max_abs: float
    unsupported_query_ids: tuple[int, ...]

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.strategy.blocks)

    @property
    def movable_block_names(self) -> tuple[str, ...]:
        return tuple(self.block_names[int(index)] for index in self.movable_block_indices)

    @property
    def rho_movable(self) -> float:
        return float(np.sum(self.control_rho[self.movable_block_indices]))

    @property
    def control_shares(self) -> np.ndarray:
        values = self.control_rho[self.movable_block_indices] / self.rho_movable
        return np.asarray(values, dtype=np.float64)

    @property
    def num_queries(self) -> int:
        return int(self.query_ids.size)

    @property
    def num_partitions(self) -> int:
        return len(self.partition_rows)

    def profile_sha256(self) -> str:
        digest = hashlib.sha256()
        header = {
            "public_total": self.public_total,
            "beta_tail": self.beta_tail,
            "block_names": self.block_names,
            "partition_names": self.partition_names,
            "reconstruction_max_abs": self.reconstruction_max_abs,
            "unsupported_query_ids": self.unsupported_query_ids,
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for array in (
            self.query_ids,
            self.unit_variance.data,
            self.unit_variance.indices,
            self.unit_variance.indptr,
            self.fixed_variance,
            self.movable_block_indices,
            self.fixed_block_indices,
            self.control_rho,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        for rows in self.partition_rows:
            digest.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class GCEARiskSnapshot:
    values: dict[str, float]
    primary_ratios: dict[str, float]
    tail_ratios: dict[str, float]


@dataclass(frozen=True)
class GCEAOptimizationResult:
    rho_by_block: dict[str, float]
    control_rho_by_block: dict[str, float]
    control_risks: dict[str, float]
    candidate_risks: dict[str, float]
    primary_ratios: dict[str, float]
    tail_ratios: dict[str, float]
    t_star: float
    t_final: float
    stage_one: dict[str, Any]
    stage_two: dict[str, Any]
    kkt_gap: float
    sum_rho_relative_error: float
    min_movable_share: float
    profile_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho_by_block": self.rho_by_block,
            "control_rho_by_block": self.control_rho_by_block,
            "control_risks": self.control_risks,
            "candidate_risks": self.candidate_risks,
            "primary_ratios": self.primary_ratios,
            "tail_ratios": self.tail_ratios,
            "t_star": self.t_star,
            "t_final": self.t_final,
            "stage_one": self.stage_one,
            "stage_two": self.stage_two,
            "kkt_gap": self.kkt_gap,
            "sum_rho_relative_error": self.sum_rho_relative_error,
            "min_movable_share": self.min_movable_share,
            "profile_sha256": self.profile_sha256,
        }


def _reconstruction_residual(
    strategy: HierarchicalPairStrategy,
    qcat: QueryCatalogue,
    qid: int,
) -> float:
    factored = factorize_public_query(strategy, qcat, qid)
    if factored is None:
        raise ValueError("Cannot certify an unsupported public query")
    scope, indicator = public_query_indicator(qcat, qid, strategy.cardinalities)
    flat = indicator.reshape(-1)
    reconstructed = np.full(flat.shape, factored.constant_per_row, dtype=np.float64)
    if len(scope) == 1:
        attr = scope[0]
        name = f"oneway_contrast:{attr}"
        reconstructed += helmert_contrast(strategy.cardinalities[attr]) @ factored.coefficients[name]
    else:
        left, right = scope
        left_map, right_map, interaction_map = pair_reconstruction_maps(
            strategy.cardinalities[left],
            strategy.cardinalities[right],
        )
        reconstructed += (
            float(strategy.cardinalities[right])
            * left_map
            @ factored.coefficients[f"oneway_contrast:{left}"]
        )
        reconstructed += (
            float(strategy.cardinalities[left])
            * right_map
            @ factored.coefficients[f"oneway_contrast:{right}"]
        )
        reconstructed += interaction_map @ factored.coefficients[
            f"pair_interaction:{left}:{right}"
        ].reshape(-1)
    return float(np.max(np.abs(reconstructed - flat)))


def _validate_workload_groups(
    groups: Sequence[Mapping[str, Any]],
    query_count: int,
) -> None:
    coverage = np.zeros(int(query_count), dtype=np.int64)
    for group_id, raw in enumerate(groups):
        indices = np.asarray(raw.get("query_indices", []), dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"Workload group {group_id} must have non-empty query_indices")
        if np.any(indices < 0) or np.any(indices >= int(query_count)):
            raise ValueError(f"Workload group {group_id} has an out-of-range query index")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Workload group {group_id} contains duplicate query indices")
        coverage[indices] += 1
    if np.any(coverage != 1):
        raise ValueError(
            "Workload groups must cover every evaluator query exactly once: "
            f"missing={int(np.sum(coverage == 0))}, overlapping={int(np.sum(coverage > 1))}"
        )


def build_gcea_profile(
    strategy: HierarchicalPairStrategy,
    qcat: QueryCatalogue,
    workload_groups: Sequence[Mapping[str, Any]],
    *,
    public_total: int,
    control_rho_by_block: Mapping[str, float],
    beta_tail: float = 0.05,
) -> GCEAProfile:
    qcat.validate(np.asarray(strategy.cardinalities, dtype=np.int64))
    if int(public_total) <= 0:
        raise ValueError("GCEA requires a positive explicitly public row count")
    beta = float(beta_tail)
    if not np.isfinite(beta) or not 0.0 < beta < 1.0:
        raise ValueError("beta_tail must lie strictly between zero and one")
    _validate_workload_groups(workload_groups, int(qcat.m))

    rho_total = float(sum(float(value) for value in control_rho_by_block.values()))
    control_mapping = validate_strategy_rho_override(
        strategy,
        rho_total,
        control_rho_by_block,
    )
    control_rho = np.asarray(
        [control_mapping[block.name] for block in strategy.blocks],
        dtype=np.float64,
    )
    block_index = {block.name: index for index, block in enumerate(strategy.blocks)}

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    query_ids: list[int] = []
    unsupported: list[int] = []
    max_residual = 0.0
    for qid in range(int(qcat.m)):
        factored = factorize_public_query(strategy, qcat, qid)
        if factored is None:
            unsupported.append(qid)
            continue
        row = len(query_ids)
        query_ids.append(qid)
        max_residual = max(max_residual, _reconstruction_residual(strategy, qcat, qid))
        for name, coefficients in factored.coefficients.items():
            column = block_index[name]
            coefficient_norm_sq = float(
                np.dot(coefficients.reshape(-1), coefficients.reshape(-1))
            )
            unit_variance = (
                0.5
                * strategy.blocks[column].sensitivity_l2**2
                * coefficient_norm_sq
            )
            if unit_variance > 0.0:
                row_indices.append(row)
                column_indices.append(column)
                values.append(unit_variance)

    if not query_ids:
        raise ValueError("GCEA requires at least one exactly reconstructable evaluator query")
    full_matrix = csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(column_indices, dtype=np.int64),
            ),
        ),
        shape=(len(query_ids), len(strategy.blocks)),
        dtype=np.float64,
    )
    column_mass = np.asarray(full_matrix.sum(axis=0), dtype=np.float64).reshape(-1)
    movable = np.flatnonzero(column_mass > 0.0).astype(np.int64)
    fixed = np.flatnonzero(column_mass == 0.0).astype(np.int64)
    if movable.size == 0:
        raise ValueError("GCEA found no movable strategy blocks")
    fixed_variance = np.zeros(len(query_ids), dtype=np.float64)
    if fixed.size:
        fixed_variance = np.asarray(
            full_matrix[:, fixed] @ (1.0 / control_rho[fixed]),
            dtype=np.float64,
        ).reshape(-1)
    unit_variance = full_matrix[:, movable].tocsr()

    row_by_qid = {qid: row for row, qid in enumerate(query_ids)}
    supported = set(query_ids)
    partition_names: list[str] = []
    partition_rows: list[np.ndarray] = []
    for group_id, raw in enumerate(workload_groups):
        indices = np.asarray(raw["query_indices"], dtype=np.int64)
        if indices.size <= 1 or not bool(raw.get("is_partition", False)):
            continue
        if not all(int(qid) in supported for qid in indices.tolist()):
            continue
        partition_names.append(str(raw.get("name", f"group:{group_id}")))
        partition_rows.append(
            np.asarray([row_by_qid[int(qid)] for qid in indices.tolist()], dtype=np.int64)
        )
    if not partition_rows:
        raise ValueError("GCEA requires at least one complete reconstructable TVD partition")

    return GCEAProfile(
        strategy=strategy,
        public_total=int(public_total),
        beta_tail=beta,
        query_ids=np.asarray(query_ids, dtype=np.int64),
        unit_variance=unit_variance,
        fixed_variance=fixed_variance,
        partition_names=tuple(partition_names),
        partition_rows=tuple(partition_rows),
        movable_block_indices=movable,
        fixed_block_indices=fixed,
        control_rho=control_rho,
        reconstruction_max_abs=max_residual,
        unsupported_query_ids=tuple(unsupported),
    )


class _RiskEvaluator:
    def __init__(self, profile: GCEAProfile) -> None:
        self.profile = profile
        self.rho_movable = profile.rho_movable
        self.scaled_variance = profile.unit_variance * (1.0 / self.rho_movable)
        self.column_mass = np.asarray(
            self.scaled_variance.sum(axis=0), dtype=np.float64
        ).reshape(-1)
        self.tvd_weights = np.zeros(profile.num_queries, dtype=np.float64)
        for rows in profile.partition_rows:
            self.tvd_weights[rows] += 1.0
        self.c_beta = math.sqrt(
            2.0 * math.log(2.0 * profile.num_queries / profile.beta_tail)
        )
        self._cache_p: np.ndarray | None = None
        self._cache_values: np.ndarray | None = None
        self._cache_gradients: np.ndarray | None = None

    def _weighted_s_gradient(
        self,
        shares: np.ndarray,
        standard_deviation: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        scaled = np.divide(
            weights,
            standard_deviation,
            out=np.zeros_like(weights, dtype=np.float64),
            where=standard_deviation > 0.0,
        )
        return (
            -0.5
            * np.asarray(self.scaled_variance.T @ scaled, dtype=np.float64).reshape(-1)
            / np.square(shares)
        )

    def evaluate(self, shares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(shares, dtype=np.float64)
        if p.shape != (self.profile.movable_block_indices.size,):
            raise ValueError("GCEA share vector has the wrong shape")
        if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
            raise ValueError("GCEA shares must be finite and positive")
        if self._cache_p is not None and np.array_equal(p, self._cache_p):
            assert self._cache_values is not None and self._cache_gradients is not None
            return self._cache_values, self._cache_gradients

        variance = self.profile.fixed_variance + np.asarray(
            self.scaled_variance @ (1.0 / p), dtype=np.float64
        ).reshape(-1)
        if np.any(variance < -1.0e-12) or not np.all(np.isfinite(variance)):
            raise FloatingPointError("GCEA produced invalid public variances")
        variance = np.maximum(variance, 0.0)
        standard_deviation = np.sqrt(variance)
        n = float(self.profile.public_total)
        q_count = float(self.profile.num_queries)
        b_count = float(self.profile.num_partitions)

        mae_scale = math.sqrt(2.0 / math.pi) / (n * q_count)
        rmse_scale = 1.0 / (n * math.sqrt(q_count))
        tvd_scale = math.sqrt(2.0 / math.pi) / (2.0 * n * b_count)
        max_error_scale = self.c_beta / n
        max_tvd_scale = self.c_beta / (2.0 * n)

        mae = mae_scale * float(np.sum(standard_deviation))
        variance_sum = float(np.sum(variance))
        rmse = rmse_scale * math.sqrt(variance_sum)
        avg_tvd = tvd_scale * float(np.dot(self.tvd_weights, standard_deviation))

        max_query_row = int(np.argmax(standard_deviation))
        max_error = max_error_scale * float(standard_deviation[max_query_row])
        block_sums = np.asarray(
            [float(np.sum(standard_deviation[rows])) for rows in self.profile.partition_rows],
            dtype=np.float64,
        )
        max_block = int(np.argmax(block_sums))
        max_tvd = max_tvd_scale * float(block_sums[max_block])

        ones = np.ones(self.profile.num_queries, dtype=np.float64)
        mae_gradient = mae_scale * self._weighted_s_gradient(
            p, standard_deviation, ones
        )
        rmse_gradient = (
            -0.5
            * rmse_scale
            * self.column_mass
            / (math.sqrt(variance_sum) * np.square(p))
        )
        avg_tvd_gradient = tvd_scale * self._weighted_s_gradient(
            p, standard_deviation, self.tvd_weights
        )
        max_query_weight = np.zeros(self.profile.num_queries, dtype=np.float64)
        max_query_weight[max_query_row] = 1.0
        max_error_gradient = max_error_scale * self._weighted_s_gradient(
            p, standard_deviation, max_query_weight
        )
        max_block_weight = np.zeros(self.profile.num_queries, dtype=np.float64)
        max_block_weight[self.profile.partition_rows[max_block]] = 1.0
        max_tvd_gradient = max_tvd_scale * self._weighted_s_gradient(
            p, standard_deviation, max_block_weight
        )

        values = np.asarray([mae, rmse, avg_tvd, max_error, max_tvd], dtype=np.float64)
        gradients = np.vstack(
            (
                mae_gradient,
                rmse_gradient,
                avg_tvd_gradient,
                max_error_gradient,
                max_tvd_gradient,
            )
        )
        self._cache_p = p.copy()
        self._cache_values = values
        self._cache_gradients = gradients
        return values, gradients


def evaluate_gcea_risks(
    profile: GCEAProfile,
    rho_by_block: Mapping[str, float],
) -> dict[str, float]:
    rho_total = float(sum(float(value) for value in rho_by_block.values()))
    mapping = validate_strategy_rho_override(profile.strategy, rho_total, rho_by_block)
    rho = np.asarray([mapping[name] for name in profile.block_names], dtype=np.float64)
    if profile.fixed_block_indices.size and not np.allclose(
        rho[profile.fixed_block_indices],
        profile.control_rho[profile.fixed_block_indices],
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("GCEA fixed-support blocks must retain their control allocation")
    movable_total = float(np.sum(rho[profile.movable_block_indices]))
    if not math.isclose(movable_total, profile.rho_movable, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("GCEA movable allocation must retain rho_movable")
    evaluator = _RiskEvaluator(profile)
    values, _ = evaluator.evaluate(rho[profile.movable_block_indices] / profile.rho_movable)
    return {name: float(value) for name, value in zip(RISK_NAMES, values, strict=True)}


def _constraint_kkt(
    result: Any,
    objective_gradient: np.ndarray,
    equality_value: np.ndarray,
    equality_jacobian: np.ndarray,
    inequality_value: np.ndarray,
    inequality_jacobian: np.ndarray,
) -> dict[str, Any]:
    multipliers = np.asarray(getattr(result, "multipliers", []), dtype=np.float64)
    expected = equality_value.size + inequality_value.size
    if multipliers.shape != (expected,):
        return {
            "gap": float("inf"),
            "reason": "missing_or_invalid_solver_multipliers",
            "multipliers": multipliers.tolist(),
        }
    equality_multipliers = multipliers[: equality_value.size]
    inequality_multipliers = multipliers[equality_value.size :]
    stationarity = (
        objective_gradient
        - equality_jacobian.T @ equality_multipliers
        - inequality_jacobian.T @ inequality_multipliers
    )
    equality_residual = float(np.max(np.abs(equality_value), initial=0.0))
    inequality_violation = float(np.max(np.maximum(-inequality_value, 0.0), initial=0.0))
    dual_violation = float(
        np.max(np.maximum(-inequality_multipliers, 0.0), initial=0.0)
    )
    complementarity = float(
        np.max(np.abs(inequality_multipliers * inequality_value), initial=0.0)
    )
    stationarity_residual = float(np.max(np.abs(stationarity), initial=0.0))
    gap = max(
        equality_residual,
        inequality_violation,
        dual_violation,
        complementarity,
        stationarity_residual,
    )
    return {
        "gap": gap,
        "equality_residual": equality_residual,
        "inequality_violation": inequality_violation,
        "dual_violation": dual_violation,
        "complementarity_residual": complementarity,
        "stationarity_residual": stationarity_residual,
        "equality_multipliers": equality_multipliers.tolist(),
        "inequality_multipliers": inequality_multipliers.tolist(),
    }


def _solver_summary(result: Any, kkt: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "jacobian_evaluations": int(result.njev),
        "objective": float(result.fun),
        "kkt": kkt,
    }


def optimize_gcea_allocation(
    profile: GCEAProfile,
    *,
    share_floor: float = 1.0e-14,
    optimal_face_tolerance: float = 1.0e-10,
    ftol: float = 1.0e-13,
    max_iterations: int = 5_000,
) -> GCEAOptimizationResult:
    floor = float(share_floor)
    face_tolerance = float(optimal_face_tolerance)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("share_floor must be finite and positive")
    if not np.isfinite(face_tolerance) or face_tolerance < 0.0:
        raise ValueError("optimal_face_tolerance must be finite and nonnegative")
    p0 = profile.control_shares
    if np.any(p0 <= floor):
        raise ValueError("Control allocation is too close to the numerical share floor")

    evaluator = _RiskEvaluator(profile)
    control_values, _ = evaluator.evaluate(p0)
    if np.any(control_values <= 0.0) or not np.all(np.isfinite(control_values)):
        raise ValueError("Control GCEA risks must be finite and positive")

    def ratios_and_gradients(shares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, gradients = evaluator.evaluate(shares)
        return values / control_values, gradients / control_values[:, None]

    movable_count = p0.size
    stage_one_initial = np.concatenate((p0, np.asarray([1.0], dtype=np.float64)))
    stage_one_bounds = Bounds(
        np.concatenate((np.full(movable_count, floor), np.asarray([0.0]))),
        np.concatenate((np.ones(movable_count), np.asarray([1.0]))),
    )

    def stage_one_objective(vector: np.ndarray) -> float:
        return float(vector[-1])

    def stage_one_objective_jacobian(vector: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(vector, dtype=np.float64)
        gradient[-1] = 1.0
        return gradient

    def stage_one_equality(vector: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.sum(vector[:-1]) - 1.0)], dtype=np.float64)

    def stage_one_equality_jacobian(vector: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((1, vector.size), dtype=np.float64)
        jacobian[0, :-1] = 1.0
        return jacobian

    def stage_one_inequality(vector: np.ndarray) -> np.ndarray:
        ratios, _ = ratios_and_gradients(vector[:-1])
        return np.concatenate((vector[-1] - ratios[:3], 1.0 - ratios[3:]))

    def stage_one_inequality_jacobian(vector: np.ndarray) -> np.ndarray:
        _, gradients = ratios_and_gradients(vector[:-1])
        jacobian = np.zeros((5, vector.size), dtype=np.float64)
        jacobian[:, :-1] = -gradients
        jacobian[:3, -1] = 1.0
        return jacobian

    stage_one_result = minimize(
        stage_one_objective,
        stage_one_initial,
        jac=stage_one_objective_jacobian,
        method="SLSQP",
        bounds=stage_one_bounds,
        constraints=(
            {
                "type": "eq",
                "fun": stage_one_equality,
                "jac": stage_one_equality_jacobian,
            },
            {
                "type": "ineq",
                "fun": stage_one_inequality,
                "jac": stage_one_inequality_jacobian,
            },
        ),
        options={"ftol": float(ftol), "maxiter": int(max_iterations), "disp": False},
    )
    stage_one_vector = np.asarray(stage_one_result.x, dtype=np.float64)
    stage_one_ratios, _ = ratios_and_gradients(stage_one_vector[:-1])
    t_star = max(float(stage_one_vector[-1]), float(np.max(stage_one_ratios[:3])))
    stage_one_eq = stage_one_equality(stage_one_vector)
    stage_one_eq_jac = stage_one_equality_jacobian(stage_one_vector)
    stage_one_ineq = stage_one_inequality(stage_one_vector)
    stage_one_ineq_jac = stage_one_inequality_jacobian(stage_one_vector)
    stage_one_kkt = _constraint_kkt(
        stage_one_result,
        stage_one_objective_jacobian(stage_one_vector),
        stage_one_eq,
        stage_one_eq_jac,
        stage_one_ineq,
        stage_one_ineq_jac,
    )
    stage_one_summary = _solver_summary(stage_one_result, stage_one_kkt)
    stage_one_summary["t_star"] = t_star
    stage_one_summary["primary_ratios"] = {
        name: float(value)
        for name, value in zip(PRIMARY_NAMES, stage_one_ratios[:3], strict=True)
    }
    stage_one_summary["tail_ratios"] = {
        name: float(value)
        for name, value in zip(TAIL_NAMES, stage_one_ratios[3:], strict=True)
    }

    face_limit = t_star + face_tolerance

    def stage_two_objective(shares: np.ndarray) -> float:
        return float(np.dot(shares, np.log(shares / p0)))

    def stage_two_objective_jacobian(shares: np.ndarray) -> np.ndarray:
        return np.log(shares / p0) + 1.0

    def stage_two_equality(shares: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.sum(shares) - 1.0)], dtype=np.float64)

    def stage_two_equality_jacobian(shares: np.ndarray) -> np.ndarray:
        return np.ones((1, shares.size), dtype=np.float64)

    def stage_two_inequality(shares: np.ndarray) -> np.ndarray:
        ratios, _ = ratios_and_gradients(shares)
        return np.concatenate((face_limit - ratios[:3], 1.0 - ratios[3:]))

    def stage_two_inequality_jacobian(shares: np.ndarray) -> np.ndarray:
        _, gradients = ratios_and_gradients(shares)
        return -gradients

    stage_two_result = minimize(
        stage_two_objective,
        stage_one_vector[:-1],
        jac=stage_two_objective_jacobian,
        method="SLSQP",
        bounds=Bounds(np.full(movable_count, floor), np.ones(movable_count)),
        constraints=(
            {
                "type": "eq",
                "fun": stage_two_equality,
                "jac": stage_two_equality_jacobian,
            },
            {
                "type": "ineq",
                "fun": stage_two_inequality,
                "jac": stage_two_inequality_jacobian,
            },
        ),
        options={"ftol": float(ftol), "maxiter": int(max_iterations), "disp": False},
    )
    final_shares = np.asarray(stage_two_result.x, dtype=np.float64)
    final_shares[-1] += 1.0 - float(np.sum(final_shares))
    final_ratios, _ = ratios_and_gradients(final_shares)
    stage_two_eq = stage_two_equality(final_shares)
    stage_two_eq_jac = stage_two_equality_jacobian(final_shares)
    stage_two_ineq = stage_two_inequality(final_shares)
    stage_two_ineq_jac = stage_two_inequality_jacobian(final_shares)
    stage_two_kkt = _constraint_kkt(
        stage_two_result,
        stage_two_objective_jacobian(final_shares),
        stage_two_eq,
        stage_two_eq_jac,
        stage_two_ineq,
        stage_two_ineq_jac,
    )
    stage_two_summary = _solver_summary(stage_two_result, stage_two_kkt)
    stage_two_summary["face_limit"] = face_limit
    stage_two_summary["primary_ratios"] = {
        name: float(value)
        for name, value in zip(PRIMARY_NAMES, final_ratios[:3], strict=True)
    }
    stage_two_summary["tail_ratios"] = {
        name: float(value)
        for name, value in zip(TAIL_NAMES, final_ratios[3:], strict=True)
    }

    final_rho = profile.control_rho.copy()
    final_rho[profile.movable_block_indices] = profile.rho_movable * final_shares
    final_rho[profile.movable_block_indices[-1]] += float(np.sum(profile.control_rho)) - float(
        np.sum(final_rho)
    )
    total = float(np.sum(profile.control_rho))
    sum_error = abs(float(np.sum(final_rho)) - total) / total
    rho_by_block = {
        name: float(value)
        for name, value in zip(profile.block_names, final_rho, strict=True)
    }
    control_mapping = {
        name: float(value)
        for name, value in zip(profile.block_names, profile.control_rho, strict=True)
    }
    control_risks = {
        name: float(value) for name, value in zip(RISK_NAMES, control_values, strict=True)
    }
    candidate_values, _ = evaluator.evaluate(final_shares)
    candidate_risks = {
        name: float(value) for name, value in zip(RISK_NAMES, candidate_values, strict=True)
    }
    return GCEAOptimizationResult(
        rho_by_block=rho_by_block,
        control_rho_by_block=control_mapping,
        control_risks=control_risks,
        candidate_risks=candidate_risks,
        primary_ratios={
            name: float(value)
            for name, value in zip(PRIMARY_NAMES, final_ratios[:3], strict=True)
        },
        tail_ratios={
            name: float(value)
            for name, value in zip(TAIL_NAMES, final_ratios[3:], strict=True)
        },
        t_star=t_star,
        t_final=float(np.max(final_ratios[:3])),
        stage_one=stage_one_summary,
        stage_two=stage_two_summary,
        kkt_gap=max(float(stage_one_kkt["gap"]), float(stage_two_kkt["gap"])),
        sum_rho_relative_error=sum_error,
        min_movable_share=float(np.min(final_shares)),
        profile_sha256=profile.profile_sha256(),
    )
