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
from qdte.privacy.accountant import ZCDPPrivacyFilter, zcdp_epsilon
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
    projection_uncertainty_bias: np.ndarray | None = None
    num_rows: int | None = None
    strategy_transcript: dict[str, Any] | None = None
    privacy_ledger: dict[str, Any] | None = None

    def to_public_dict(self) -> dict:
        payload = {
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
            "num_rows": self.num_rows,
        }
        if self.strategy_transcript is not None:
            payload["strategy_transcript"] = self.strategy_transcript
        if self.privacy_ledger is not None:
            payload["privacy_ledger"] = self.privacy_ledger
        return payload


def measurement_group_from_dict(data: dict[str, Any]) -> MeasurementGroup:
    return MeasurementGroup(
        query_indices=np.asarray(data["query_indices"], dtype=np.int32),
        sensitivity_l2=float(data["sensitivity_l2"]),
        rho=float(data["rho"]),
        sigma=float(data["sigma"]),
        noise_std=float(data["noise_std"]),
        name=str(data["name"]),
        family=str(data["family"]),
        is_partition=bool(data["is_partition"]),
    )


def measurements_from_public_dict(data: dict[str, Any]) -> Measurements:
    target_noisy = np.asarray(data["target_noisy"], dtype=np.float32)
    target_projected = np.asarray(data["target_projected"], dtype=np.float32)
    variances = np.asarray(data["variances"], dtype=np.float32)
    if target_noisy.ndim != 1 or target_projected.shape != target_noisy.shape or variances.shape != target_noisy.shape:
        raise ValueError("Serialized measurement targets and variances must be one-dimensional with matching shapes")
    if not np.all(np.isfinite(target_noisy)) or not np.all(np.isfinite(target_projected)):
        raise ValueError("Serialized measurement targets must be finite")
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("Serialized measurement variances must be finite and positive")
    inv_variances = 1.0 / np.maximum(variances, 1.0e-12)
    groups = [measurement_group_from_dict(group) for group in data.get("groups", [])]
    _validate_measurement_groups(len(target_noisy), groups)
    raw_ledger = data.get("privacy_ledger")
    if raw_ledger is not None and not isinstance(raw_ledger, dict):
        raise ValueError("Serialized privacy_ledger must be a mapping")
    return Measurements(
        target_noisy=target_noisy,
        target_projected=target_projected,
        variances=variances,
        inv_variances=inv_variances.astype(np.float32),
        groups=groups,
        mode=str(data["mode"]),
        rho_total=float(data["rho_total"]),
        rho_spent=float(data["rho_spent"]),
        epsilon_delta=float(data["epsilon_delta"]),
        delta=float(data["delta"]),
        projection_diagnostics=dict(data.get("projection_diagnostics", {})),
        num_rows=int(data["num_rows"]) if data.get("num_rows") is not None else None,
        strategy_transcript=(
            dict(data["strategy_transcript"])
            if data.get("strategy_transcript") is not None
            else None
        ),
        privacy_ledger=dict(raw_ledger) if raw_ledger is not None else None,
    )


def _family_counts(groups: list[WorkloadGroup]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.family] = counts.get(group.family, 0) + 1
    return counts


def _validate_measurement_groups(num_queries: int, groups: list[WorkloadGroup | MeasurementGroup]) -> None:
    coverage = np.zeros(int(num_queries), dtype=np.int32)
    for group in groups:
        idx = np.asarray(group.query_indices, dtype=np.int32)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"Measurement group {group.name!r} must contain a non-empty 1D query index vector")
        if np.any(idx < 0) or np.any(idx >= int(num_queries)):
            raise ValueError(f"Measurement group {group.name!r} has out-of-range query indices")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"Measurement group {group.name!r} contains duplicate query indices")
        sensitivity = float(group.sensitivity_l2)
        if not np.isfinite(sensitivity) or sensitivity <= 0.0:
            raise ValueError(f"Measurement group {group.name!r} must have positive finite L2 sensitivity")
        coverage[idx] += 1
    missing = np.flatnonzero(coverage == 0)
    overlapping = np.flatnonzero(coverage > 1)
    if len(missing) or len(overlapping):
        raise ValueError(
            "Static measurement groups must cover every query exactly once; "
            f"missing={len(missing)}, overlapping={len(overlapping)}"
        )


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


def _cardinalities_for_projection(X_real: np.ndarray, cardinalities: np.ndarray | None) -> np.ndarray:
    if cardinalities is not None:
        cards = np.asarray(cardinalities, dtype=np.int32)
    else:
        cards = np.max(X_real, axis=0).astype(np.int32) + 1
    if cards.shape != (X_real.shape[1],) or np.any(cards <= 0):
        raise ValueError(f"cardinalities must have shape ({X_real.shape[1]},) with positive entries")
    for attr, cardinality in enumerate(cards.tolist()):
        values = X_real[:, attr]
        if np.any(values < 0) or np.any(values >= int(cardinality)):
            raise ValueError(f"X_real has values outside the public domain for attribute {attr}")
    return cards


def _apply_configured_projection(
    noisy: np.ndarray,
    qcat: QueryCatalogue,
    groups: list[MeasurementGroup],
    total: int,
    projection_cfg: dict[str, Any],
    variances: np.ndarray,
    cardinalities: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    projected = project_targets(
        noisy,
        groups,
        total,
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
        method = str(consistency_cfg.get("method", "local_marginal_ipf"))
        if method == "local_marginal_ipf":
            consistency_result = project_consistent_targets(
                noisy,
                qcat,
                cardinalities,
                total,
                variances=variances,
                max_scope_cells=int(consistency_cfg.get("max_scope_cells", 200_000)),
                max_iterations=int(consistency_cfg.get("max_iterations", 100)),
                tolerance=float(consistency_cfg.get("tolerance", 1.0e-2)),
                max_lsq_iterations=int(consistency_cfg.get("max_lsq_iterations", 100)),
            )
        elif method == "query_space_lsq":
            consistency_result = project_query_space_lsq(
                noisy,
                qcat,
                cardinalities,
                total,
                variances=variances,
                max_constraints=int(consistency_cfg.get("max_constraints", 200_000)),
                solver_atol=float(consistency_cfg.get("solver_atol", 1.0e-10)),
                solver_btol=float(consistency_cfg.get("solver_btol", 1.0e-10)),
                solver_max_iterations=int(consistency_cfg.get("solver_max_iterations", 10_000)),
            )
        elif method == "query_space_feasible_lsq":
            consistency_result = project_query_space_feasible_lsq(
                noisy,
                qcat,
                cardinalities,
                total,
                variances=variances,
                max_constraints=int(consistency_cfg.get("max_constraints", 200_000)),
                solver_ftol=float(consistency_cfg.get("solver_ftol", 1.0e-9)),
                solver_max_iterations=int(consistency_cfg.get("solver_max_iterations", 1_000)),
                max_dense_constraint_cells=int(consistency_cfg.get("max_dense_constraint_cells", 20_000_000)),
                certificate_feasibility_tolerance=float(
                    consistency_cfg.get("certificate_feasibility_tolerance", 1.0e-6)
                ),
                certificate_gap_absolute_tolerance=float(
                    consistency_cfg.get("certificate_gap_absolute_tolerance", 1.0e-7)
                ),
                certificate_gap_relative_tolerance=float(
                    consistency_cfg.get("certificate_gap_relative_tolerance", 1.0e-8)
                ),
                certificate_max_iterations=int(
                    consistency_cfg.get("certificate_max_iterations", 1_000)
                ),
            )
        elif method == "local_table_feasible_lsq":
            consistency_result = project_local_table_feasible_lsq(
                noisy,
                qcat,
                cardinalities,
                total,
                variances=variances,
                max_scope_cells=int(consistency_cfg.get("max_scope_cells", 200_000)),
                solver_ftol=float(consistency_cfg.get("solver_ftol", 1.0e-9)),
                solver_max_iterations=int(consistency_cfg.get("solver_max_iterations", 1_000)),
                max_dense_constraint_cells=int(consistency_cfg.get("max_dense_constraint_cells", 20_000_000)),
            )
        elif method == "local_table_feasible_jax":
            consistency_result = project_local_table_feasible_jax(
                noisy,
                qcat,
                cardinalities,
                total,
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
    return projected.astype(np.float32), projection_diagnostics


def _projection_uncertainty_cfg(projection_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = projection_cfg.get("uncertainty", {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("projection.uncertainty must be a mapping")
    return cfg


def _apply_projection_aware_uncertainty(
    noisy: np.ndarray,
    projected: np.ndarray,
    qcat: QueryCatalogue,
    groups: list[MeasurementGroup],
    total: int,
    projection_cfg: dict[str, Any],
    variances: np.ndarray,
    cardinalities: np.ndarray,
    rng: np.random.Generator,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]:
    cfg = _projection_uncertainty_cfg(projection_cfg)
    if not bool(cfg.get("enabled", False)):
        return projected.astype(np.float32), variances.astype(np.float32), {"enabled": False}, None
    method = str(cfg.get("method", "bootstrap_diagonal"))
    if method != "bootstrap_diagonal":
        raise ValueError("projection.uncertainty.method must be 'bootstrap_diagonal'")
    num_samples = int(cfg.get("num_samples", 32))
    if num_samples <= 1:
        raise ValueError("projection.uncertainty.num_samples must be greater than 1")
    center_name = str(cfg.get("center", "projected")).lower()
    if center_name == "projected":
        center = np.asarray(projected, dtype=np.float64)
    elif center_name == "noisy":
        center = np.asarray(noisy, dtype=np.float64)
    else:
        raise ValueError("projection.uncertainty.center must be 'projected' or 'noisy'")
    raw_var = np.maximum(np.asarray(variances, dtype=np.float64), float(min_variance))
    raw_std = np.sqrt(raw_var)
    mean = np.zeros(qcat.m, dtype=np.float64)
    m2 = np.zeros(qcat.m, dtype=np.float64)
    for sample_idx in range(1, num_samples + 1):
        boot_noisy = center + rng.normal(loc=0.0, scale=raw_std, size=qcat.m)
        boot_projected, _ = _apply_configured_projection(
            boot_noisy.astype(np.float32),
            qcat,
            groups,
            total,
            projection_cfg,
            raw_var.astype(np.float32),
            cardinalities,
        )
        x = boot_projected.astype(np.float64)
        delta = x - mean
        mean += delta / float(sample_idx)
        m2 += delta * (x - mean)
    boot_var = m2 / float(num_samples - 1)
    min_var = float(cfg.get("min_variance", min_variance))
    if min_var <= 0.0:
        raise ValueError("projection.uncertainty.min_variance must be positive")
    min_raw_fraction = float(cfg.get("min_raw_variance_fraction", 0.0))
    if min_raw_fraction < 0.0:
        raise ValueError("projection.uncertainty.min_raw_variance_fraction must be non-negative")
    effective_var = np.maximum(boot_var, min_var)
    if min_raw_fraction > 0.0:
        effective_var = np.maximum(effective_var, min_raw_fraction * raw_var)
    debias_target = bool(cfg.get("debias_target", False))
    debias_alpha = float(cfg.get("debias_alpha", 1.0))
    if debias_alpha < 0.0 or debias_alpha > 1.0:
        raise ValueError("projection.uncertainty.debias_alpha must be in [0, 1]")
    effective_debias_alpha = debias_alpha if debias_target else 0.0
    reproject_debiased_target = bool(cfg.get("reproject_debiased_target", False))
    bias = mean - center
    target = np.asarray(projected, dtype=np.float64)
    pre_reproject_min = float(np.min(target)) if target.size else 0.0
    pre_reproject_negative_count = int(np.sum(target < -1.0e-9))
    if debias_target:
        target = target - effective_debias_alpha * bias
        pre_reproject_min = float(np.min(target)) if target.size else 0.0
        pre_reproject_negative_count = int(np.sum(target < -1.0e-9))
        if reproject_debiased_target:
            target, _ = _apply_configured_projection(
                target.astype(np.float32),
                qcat,
                groups,
                total,
                projection_cfg,
                raw_var.astype(np.float32),
                cardinalities,
            )
            target = target.astype(np.float64)
    diagnostics = {
        "enabled": True,
        "method": method,
        "num_samples": int(num_samples),
        "center": center_name,
        "debias_target": bool(debias_target),
        "debias_alpha": float(debias_alpha),
        "effective_debias_alpha": float(effective_debias_alpha),
        "reproject_debiased_target": bool(reproject_debiased_target),
        "target_reprojected_after_debias": bool(debias_target and reproject_debiased_target),
        "pre_reproject_target_min": float(pre_reproject_min),
        "pre_reproject_negative_target_count": int(pre_reproject_negative_count),
        "final_target_min": float(np.min(target)) if target.size else 0.0,
        "final_negative_target_count": int(np.sum(target < -1.0e-9)),
        "min_variance": float(min_var),
        "min_raw_variance_fraction": float(min_raw_fraction),
        "raw_variance_mean": float(np.mean(raw_var)),
        "raw_variance_min": float(np.min(raw_var)),
        "raw_variance_max": float(np.max(raw_var)),
        "effective_variance_mean": float(np.mean(effective_var)),
        "effective_variance_min": float(np.min(effective_var)),
        "effective_variance_max": float(np.max(effective_var)),
        "effective_to_raw_variance_mean": float(np.mean(effective_var / raw_var)),
        "bias_l2": float(np.linalg.norm(bias)),
        "bias_linf": float(np.max(np.abs(bias))) if bias.size else 0.0,
        "mean_projected_bootstrap_l2_from_center": float(np.linalg.norm(mean - center)),
    }
    return target.astype(np.float32), effective_var.astype(np.float32), diagnostics, bias.astype(np.float32)


def measure_real_dataset(
    X_real: np.ndarray,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    config: dict[str, Any],
    rng: np.random.Generator,
    batch_size: int = 8192,
    cardinalities: np.ndarray | None = None,
) -> Measurements:
    X_real = np.asarray(X_real)
    if X_real.ndim != 2 or X_real.shape[0] <= 0 or X_real.shape[1] <= 0:
        raise ValueError("X_real must be a non-empty two-dimensional encoded table")
    if qcat.m <= 0:
        raise ValueError("Static measurement requires at least one query")
    privacy_cfg = config.get("privacy", {})
    projection_cfg = config.get("projection", {})
    mode = str(privacy_cfg.get("mode", "dp")).lower()
    adjacency = str(privacy_cfg.get("adjacency", "add_remove"))
    if adjacency != "add_remove":
        raise ValueError("Static measurement currently requires privacy.adjacency='add_remove'")
    dp_release_mode = bool(privacy_cfg.get("dp_release_mode", False))
    measurement_mode = str(privacy_cfg.get("measurement_mode", "static_all")).lower()
    if measurement_mode != "static_all":
        raise NotImplementedError(
            "Only privacy.measurement_mode=static_all is implemented. "
            "adaptive_select_measure/select-measure-generate is not implemented in this version."
        )
    rho_total = float(privacy_cfg.get("rho_total", 1.0))
    delta = float(privacy_cfg.get("delta", 1.0e-9))
    if mode == "dp" and rho_total <= 0.0:
        raise ValueError("privacy.rho_total must be positive in DP mode")

    if dp_release_mode and mode == "dp":
        public_total = privacy_cfg.get("public_n_rows")
        if (
            not isinstance(public_total, int)
            or isinstance(public_total, bool)
            or public_total <= 0
        ):
            raise ValueError(
                "privacy.dp_release_mode=true requires positive integer privacy.public_n_rows"
            )
        if int(X_real.shape[0]) != public_total:
            raise ValueError(
                "Private input row count does not match declared privacy.public_n_rows: "
                f"input={int(X_real.shape[0])}, public={public_total}"
            )
        projection_total = int(public_total)
    else:
        projection_total = int(X_real.shape[0])

    _validate_measurement_groups(qcat.m, workload_groups)
    if mode == "dp" and cardinalities is None:
        raise ValueError("DP measurement requires public schema cardinalities; private-data inference is disabled")
    cards = _cardinalities_for_projection(X_real, cardinalities)
    qcat.validate(cards)

    true_answers = answer_queries(X_real, qcat, batch_size=batch_size)
    target = np.zeros(qcat.m, dtype=np.float32)
    variances = np.ones(qcat.m, dtype=np.float32)
    measurement_groups: list[MeasurementGroup] = []
    rho_spent = 0.0
    privacy_filter = ZCDPPrivacyFilter(rho_total) if mode == "dp" else None

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
            if privacy_filter is None:
                raise RuntimeError("DP measurement is missing its privacy filter")
            privacy_filter.spend(
                label=wg.name,
                mechanism="gaussian_vector",
                rho=rho_g,
                public_metadata={
                    "accounting": "gaussian_zcdp_exact_v1",
                    "adjacency": adjacency,
                    "family": wg.family,
                    "num_queries": int(len(idx)),
                    "sensitivity_l2": float(wg.sensitivity_l2),
                    "sigma_multiplier": float(sigma),
                    "noise_std": float(noise_std),
                },
            )
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
    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError("privacy.min_variance must be positive and finite")
    variances = np.maximum(variances, min_variance).astype(np.float32)
    projected, projection_diagnostics = _apply_configured_projection(
        target,
        qcat,
        measurement_groups,
        projection_total,
        projection_cfg,
        variances,
        cards,
    )
    projected, variances, uncertainty_diagnostics, uncertainty_bias = _apply_projection_aware_uncertainty(
        target,
        projected,
        qcat,
        measurement_groups,
        projection_total,
        projection_cfg,
        variances,
        cards,
        rng,
        min_variance=min_variance,
    )
    projection_diagnostics["uncertainty"] = uncertainty_diagnostics
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(projected)):
        raise RuntimeError("Measurement or projection produced non-finite targets")
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise RuntimeError("Measurement or projection produced invalid variances")
    if privacy_filter is not None:
        rho_spent = privacy_filter.rho_spent
    if rho_spent > rho_total + 1.0e-10 * max(1.0, abs(rho_total)):
        raise RuntimeError(f"Measurement spent rho={rho_spent} above configured rho_total={rho_total}")
    epsilon_delta = zcdp_epsilon(float(rho_spent), delta)
    privacy_ledger = None
    if privacy_filter is not None:
        privacy_ledger = privacy_filter.to_public_dict(delta=delta)
        privacy_ledger.update(
            {
                "accounting_theorem": "gaussian_zcdp_rho_equals_delta2_over_2_noise_variance",
                "accounting_version": "static_gaussian_vector_v1",
                "adjacency": adjacency,
                "postprocessing": [
                    "configured_projection",
                    "projection_aware_uncertainty",
                    "qdte_generation",
                ],
            }
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
        rho_spent=float(rho_spent),
        epsilon_delta=epsilon_delta,
        delta=delta,
        projection_diagnostics=projection_diagnostics,
        projection_uncertainty_bias=uncertainty_bias,
        num_rows=projection_total,
        privacy_ledger=privacy_ledger,
    )
