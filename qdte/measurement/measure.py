from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.measurement.projection import clip_counts, project_simplex
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
    epsilon_delta: float
    delta: float

    def to_public_dict(self) -> dict:
        return {
            "mode": self.mode,
            "rho_total": self.rho_total,
            "delta": self.delta,
            "epsilon_delta": self.epsilon_delta,
            "target_noisy": self.target_noisy.tolist(),
            "target_projected": self.target_projected.tolist(),
            "variances": self.variances.tolist(),
            "groups": [g.to_dict() for g in self.groups],
        }


def _family_counts(groups: list[WorkloadGroup]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.family] = counts.get(group.family, 0) + 1
    return counts


def _allocate_group_budgets(groups: list[WorkloadGroup], privacy_cfg: dict[str, Any]) -> dict[str, float]:
    rho_total = float(privacy_cfg.get("rho_total", 1.0))
    allocation = privacy_cfg.get("measurement_allocation", {})
    if not allocation:
        families = sorted(set(g.family for g in groups))
        allocation = {family: 1.0 / len(families) for family in families}
    alloc_sum = float(sum(float(v) for v in allocation.values()))
    if alloc_sum <= 0:
        raise ValueError("measurement_allocation must have positive sum")
    normalized = {str(k): float(v) / alloc_sum for k, v in allocation.items()}
    counts = _family_counts(groups)
    budgets: dict[str, float] = {}
    for family, count in counts.items():
        family_rho = rho_total * normalized.get(family, 0.0)
        if family_rho <= 0:
            family_rho = rho_total * 1.0e-6 / max(1, len(groups))
        budgets[family] = family_rho / max(1, count)
    return budgets


def project_targets(
    noisy: np.ndarray,
    groups: list[MeasurementGroup],
    total: int,
    project_partitions: bool,
    clip_nonpartition: bool,
) -> np.ndarray:
    projected = noisy.astype(np.float32).copy()
    for group in groups:
        idx = group.query_indices
        if group.is_partition and project_partitions:
            projected[idx] = project_simplex(projected[idx], float(total))
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
    )
    inv_variances = (1.0 / variances).astype(np.float32)
    return Measurements(
        target_noisy=target.astype(np.float32),
        target_projected=projected,
        variances=variances,
        inv_variances=inv_variances,
        groups=measurement_groups,
        mode=mode,
        rho_total=rho_total,
        epsilon_delta=epsilon_delta,
        delta=delta,
    )
