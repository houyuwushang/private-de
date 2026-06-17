from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.measurement.consistency import (
    project_consistent_targets,
    project_local_table_feasible_lsq,
    project_local_table_feasible_jax,
    project_query_space_feasible_lsq,
    project_query_space_lsq,
)
from qdte.measurement.projection import clip_counts, project_non_decreasing, project_simplex
from qdte.privacy.accountant import zcdp_epsilon
from qdte.privacy.gaussian import add_zcdp_gaussian_noise, sigma_from_rho
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup


@dataclass
class MeasurementGroup:
    query_indices: np.ndarray
    sensitivity_l2: float
    rho: float
    sigma: float
    noise_std: float
    name: str
    family: str
    is_partition: bool

    def to_dict(self) -> dict:
        return {
            "query_indices": self.query_indices.tolist(),
            "sensitivity_l2": self.sensitivity_l2,
            "rho": self.rho,
            "sigma": self.sigma,
            "noise_std": self.noise_std,
            "name": self.name,
            "family": self.family,
            "is_partition": self.is_partition,
        }


@dataclass
class Measurements:
    target_noisy: np.ndarray
    target_projected: np.ndarray
    variances: np.ndarray
    inv_variances: np.ndarray
    groups: list[MeasurementGroup]
    mode: str
    rho_total: float
    rho_spent: float
    epsilon_delta: float
    delta: float
    projection_diagnostics: dict[str, Any] | None = None

    def to_public_dict(self) -> dict:
        return {
            "mode": self.mode,
            "rho_total": self.rho_total,
            "rho_spent": self.rho_spent,
            "delta": self.delta,
            "epsilon_delta": self.epsilon_delta,
            "target_noisy": self.target_noisy.tolist(),
            "target_projected": self.target_projected.tolist(),
            "variances": self.variances.tolist(),
            "groups": [g.to_dict() for g in self.groups],
            "projection_diagnostics": self.projection_diagnostics or {},
        }


def _family_counts(groups: list[WorkloadGroup]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.family] = counts.get(group.family, 0) + 1
    return counts


def _allocate_group_budgets(groups: list[WorkloadGroup], privacy_cfg: dict[str, Any]) -> dict[str, float]:
    rho_total = float(privacy_cfg.get("rho_total", 1.0))
    allocation = privacy_cfg.get("measurement_allocation", {})
    counts = _family_counts(groups)
    observed_families = sorted(counts)
    if not observed_families:
        return {}
    if not allocation:
        allocation = {family: 1.0 for family in observed_families}
    family_weights: dict[str, float] = {}
    for family in observed_families:
        if family not in allocation:
            raise ValueError(f"privacy.measurement_allocation is missing observed workload family {family!r}")
        weight = float(allocation[family])
        if weight <= 0.0:
            raise ValueError(f"privacy.measurement_allocation[{family!r}] must be positive")
        family_weights[family] = weight
    alloc_sum = float(sum(family_weights.values()))
    if alloc_sum <= 0:
        raise ValueError("measurement_allocation must have positive sum")
    budgets: dict[str, float] = {}
    for family, count in counts.items():
        family_rho = rho_total * family_weights[family] / alloc_sum
        budgets[family] = family_rho / max(1, count)
    return budgets


def project_targets(
    noisy: np.ndarray,
    groups: list[MeasurementGroup],
    total: int,
    project_partitions: bool,
    clip_nonpartition: bool,
    prefix_monotonicity: bool = False,
    variances: np.ndarray | None = None,
) -> np.ndarray:
    projected = noisy.astype(np.float32).copy()
    for group in groups:
        idx = group.query_indices
        if group.is_partition and project_partitions:
            projected[idx] = project_simplex(projected[idx], float(total))
        elif group.family == "prefix" and prefix_monotonicity:
            values = projected[idx]
            if clip_nonpartition:
                values = clip_counts(values, float(total))
            weights = None
            if variances is not None:
                weights = 1.0 / np.maximum(np.asarray(variances[idx], dtype=np.float64), 1.0e-12)
            values = project_non_decreasing(values, weights)
            if clip_nonpartition:
                values = clip_counts(values, float(total))
            projected[idx] = values
        elif clip_nonpartition:
            projected[idx] = clip_counts(projected[idx], float(total))
    return projected.astype(np.float32)


def measure_real_dataset(
    X_real: np.ndarray,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    config: dict[str, Any],
    rng: np.random.Generator,
    batch_size: int = 8192,
    cardinalities: np.ndarray | None = None,
) -> Measurements:
    privacy_cfg = config.get("privacy", {})
    projection_cfg = config.get("projection", {})
    mode = str(privacy_cfg.get("mode", "dp")).lower()
    measurement_mode = str(privacy_cfg.get("measurement_mode", "static_all")).lower()
    if measurement_mode != "static_all":
        raise NotImplementedError(
            "Only privacy.measurement_mode=static_all is implemented. "
            "adaptive_select_measure/select-measure-generate is not implemented in this version."
        )
    rho_total = float(privacy_cfg.get("rho_total", 1.0))
    delta = float(privacy_cfg.get("delta", 1.0e-9))
    epsilon_delta = zcdp_epsilon(rho_total, delta)

    true_answers = answer_queries(X_real, qcat, batch_size=batch_size)
    target = np.zeros(qcat.m, dtype=np.float32)
    variances = np.ones(qcat.m, dtype=np.float32)
    measurement_groups: list[MeasurementGroup] = []
    rho_spent = 0.0

    if mode == "oracle":
        target = true_answers.astype(np.float32).copy()
        variances.fill(float(privacy_cfg.get("oracle_variance", 1.0)))
        for wg in workload_groups:
            measurement_groups.append(
                MeasurementGroup(
                    query_indices=wg.query_indices,
                    sensitivity_l2=wg.sensitivity_l2,
                    rho=0.0,
                    sigma=0.0,
                    noise_std=0.0,
                    name=wg.name,
                    family=wg.family,
                    is_partition=wg.is_partition,
                )
            )
    elif mode == "dp":
        budgets = _allocate_group_budgets(workload_groups, privacy_cfg)
        for wg in workload_groups:
            rho_g = float(budgets[wg.family])
            rho_spent += rho_g
            idx = wg.query_indices
            noisy, sigma, noise_std = add_zcdp_gaussian_noise(true_answers[idx], rho_g, wg.sensitivity_l2, rng)
            target[idx] = noisy
            variances[idx] = np.float32(noise_std * noise_std)
            measurement_groups.append(
                MeasurementGroup(
                    query_indices=idx,
                    sensitivity_l2=wg.sensitivity_l2,
                    rho=rho_g,
                    sigma=sigma,
                    noise_std=noise_std,
                    name=wg.name,
                    family=wg.family,
                    is_partition=wg.is_partition,
                )
            )
    else:
        raise ValueError(f"privacy.mode must be 'dp' or 'oracle', got {mode!r}")

    min_variance = float(privacy_cfg.get("min_variance", 1.0e-6))
    variances = np.maximum(variances, min_variance).astype(np.float32)
    projected = project_targets(
        target,
        measurement_groups,
        X_real.shape[0],
        project_partitions=bool(projection_cfg.get("project_partitions", True)),
        clip_nonpartition=bool(projection_cfg.get("clip_nonpartition", True)),
        prefix_monotonicity=bool(projection_cfg.get("prefix_monotonicity", False)),
        variances=variances,
    )
    projection_diagnostics: dict[str, Any] = {
        "consistency": {"enabled": False},
    }
    consistency_cfg = projection_cfg.get("consistency", {})
    if consistency_cfg is None:
        consistency_cfg = {}
    if not isinstance(consistency_cfg, dict):
        raise ValueError("projection.consistency must be a mapping")
    if bool(consistency_cfg.get("enabled", False)):
        cards = (
            np.asarray(cardinalities, dtype=np.int32)
            if cardinalities is not None
            else (np.max(X_real, axis=0).astype(np.int32) + 1)
        )
        method = str(consistency_cfg.get("method", "local_marginal_ipf"))
        if method == "local_marginal_ipf":
            consistency_result = project_consistent_targets(
                target,
                qcat,
                cards,
                X_real.shape[0],
                variances=variances,
                max_scope_cells=int(consistency_cfg.get("max_scope_cells", 200_000)),
                max_iterations=int(consistency_cfg.get("max_iterations", 100)),
                tolerance=float(consistency_cfg.get("tolerance", 1.0e-2)),
                max_lsq_iterations=int(consistency_cfg.get("max_lsq_iterations", 100)),
            )
        elif method == "query_space_lsq":
            consistency_result = project_query_space_lsq(
                target,
                qcat,
                cards,
                X_real.shape[0],
                variances=variances,
                max_constraints=int(consistency_cfg.get("max_constraints", 200_000)),
                solver_atol=float(consistency_cfg.get("solver_atol", 1.0e-10)),
                solver_btol=float(consistency_cfg.get("solver_btol", 1.0e-10)),
                solver_max_iterations=int(consistency_cfg.get("solver_max_iterations", 10_000)),
            )
        elif method == "query_space_feasible_lsq":
            consistency_result = project_query_space_feasible_lsq(
                target,
                qcat,
                cards,
                X_real.shape[0],
                variances=variances,
                max_constraints=int(consistency_cfg.get("max_constraints", 200_000)),
                solver_ftol=float(consistency_cfg.get("solver_ftol", 1.0e-9)),
                solver_max_iterations=int(consistency_cfg.get("solver_max_iterations", 1_000)),
                max_dense_constraint_cells=int(consistency_cfg.get("max_dense_constraint_cells", 20_000_000)),
            )
        elif method == "local_table_feasible_lsq":
            consistency_result = project_local_table_feasible_lsq(
                target,
                qcat,
                cards,
                X_real.shape[0],
                variances=variances,
                max_scope_cells=int(consistency_cfg.get("max_scope_cells", 200_000)),
                solver_ftol=float(consistency_cfg.get("solver_ftol", 1.0e-9)),
                solver_max_iterations=int(consistency_cfg.get("solver_max_iterations", 1_000)),
                max_dense_constraint_cells=int(consistency_cfg.get("max_dense_constraint_cells", 20_000_000)),
            )
        elif method == "local_table_feasible_jax":
            consistency_result = project_local_table_feasible_jax(
                target,
                qcat,
                cards,
                X_real.shape[0],
                variances=variances,
                max_scope_cells=int(consistency_cfg.get("max_scope_cells", 200_000)),
                jax_iterations=int(consistency_cfg.get("jax_iterations", 1_000)),
                jax_active_set_tolerance=float(consistency_cfg.get("jax_active_set_tolerance", 1.0e-8)),
                jax_kkt_ridge=float(consistency_cfg.get("jax_kkt_ridge", 1.0e-10)),
                max_dense_constraint_cells=int(consistency_cfg.get("max_dense_constraint_cells", 20_000_000)),
            )
        else:
            raise ValueError(
                "projection.consistency.method must be one of: "
                "local_marginal_ipf, query_space_lsq, query_space_feasible_lsq, "
                "local_table_feasible_lsq, local_table_feasible_jax; "
                f"got {method!r}"
            )
        projected = consistency_result.projected
        projection_diagnostics["consistency"] = consistency_result.diagnostics
    inv_variances = (1.0 / variances).astype(np.float32)
    return Measurements(
        target_noisy=target.astype(np.float32),
        target_projected=projected,
        variances=variances,
        inv_variances=inv_variances,
        groups=measurement_groups,
        mode=mode,
        rho_total=rho_total,
        rho_spent=float(rho_spent),
        epsilon_delta=epsilon_delta,
        delta=delta,
        projection_diagnostics=projection_diagnostics,
    )
