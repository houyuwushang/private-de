from __future__ import annotations

from typing import Any, Literal

import numpy as np

from qdte.measurement.covariance import (
    CoefficientBootstrapDiagonal,
    bootstrap_local_polytope_coefficient_diagonal,
)
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.local_polytope import (
    LocalPolytopeProjectionResult,
    project_hierarchical_local_polytope,
)
from qdte.measurement.measure import MeasurementGroup, Measurements
from qdte.measurement.projection import project_simplex
from qdte.measurement.shrinkage import (
    InteractionShrinkageResult,
    positive_part_block_james_stein,
)
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup


def _scope_from_group(group: WorkloadGroup) -> tuple[int, ...]:
    parts = str(group.name).split(":")
    if group.family == "oneway" and len(parts) == 2:
        return (int(parts[1]),)
    if group.family == "twoway" and len(parts) == 3:
        left, right = int(parts[1]), int(parts[2])
        return (left, right) if left < right else (right, left)
    raise ValueError(
        "Interaction adapter requires canonical complete-partition groups named "
        "oneway:<attr> or twoway:<left>:<right>"
    )


def _consistency_linf(
    oneway: dict[int, np.ndarray],
    pairs: dict[tuple[int, int], np.ndarray],
) -> float:
    violation = 0.0
    for (left, right), table in pairs.items():
        violation = max(
            violation,
            float(np.max(np.abs(np.sum(table, axis=1) - oneway[left]))),
            float(np.max(np.abs(np.sum(table, axis=0) - oneway[right]))),
        )
    return violation


def _interaction_privacy_ledger(
    transcript: HierarchicalInteractionTranscript,
    *,
    delta: float,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for block in transcript.strategy.blocks:
        variance = float(transcript.component_variances[block.name])
        sensitivity = float(block.sensitivity_l2)
        entries.append(
            {
                "label": block.name,
                "mechanism": "gaussian_vector",
                "rho": float(transcript.rho_by_block[block.name]),
                "public_metadata": {
                    "accounting": "gaussian_zcdp_exact_v1",
                    "adjacency": transcript.adjacency,
                    "kind": block.kind,
                    "scope": list(block.scope),
                    "num_coefficients": int(np.prod(block.coefficient_shape)),
                    "sensitivity_l2": sensitivity,
                    "sigma_multiplier": float(np.sqrt(variance) / sensitivity),
                    "noise_std": float(np.sqrt(variance)),
                },
            }
        )
    return {
        "accounting": "zcdp_actual_spend_v1",
        "accounting_theorem": "gaussian_zcdp_rho_equals_delta2_over_2_noise_variance",
        "accounting_version": "hierarchical_interaction_gaussian_v1",
        "adjacency": transcript.adjacency,
        "rho_limit": float(transcript.rho_total),
        "rho_spent": float(transcript.rho_spent),
        "rho_remaining": float(max(0.0, transcript.rho_total - transcript.rho_spent)),
        "delta": float(delta),
        "epsilon_from_actual_spend": zcdp_epsilon(
            float(transcript.rho_spent),
            float(delta),
        ),
        "entries": entries,
        "postprocessing": [
            "hierarchical_marginal_reconstruction",
            "configured_projection",
            "qdte_generation",
        ],
    }


def interaction_transcript_to_diagonal_measurements(
    transcript: HierarchicalInteractionTranscript,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    *,
    delta: float,
    min_variance: float = 1.0e-12,
    target_projection: Literal["per_scope_simplex", "raw_reconstruction"] = "per_scope_simplex",
) -> Measurements:
    """Adapt a correlated interaction transcript to the legacy diagonal QDTE interface.

    This adapter is intentionally explicit: it preserves marginal variances but drops
    cross-cell and cross-scope covariance. It is suitable for the Static-ICE Phase-A
    transfer smoke, not for the final covariance-aware ICE profile.
    """
    if not np.isfinite(delta) or not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not np.isfinite(min_variance) or float(min_variance) <= 0.0:
        raise ValueError("min_variance must be finite and positive")
    if qcat.m <= 0:
        raise ValueError("Interaction adapter requires a non-empty query catalogue")
    if target_projection not in {"per_scope_simplex", "raw_reconstruction"}:
        raise ValueError(
            "target_projection must be 'per_scope_simplex' or 'raw_reconstruction'"
        )

    reconstructed = transcript.reconstruct()
    raw_target = np.zeros(qcat.m, dtype=np.float64)
    projected_target = np.zeros(qcat.m, dtype=np.float64)
    variances = np.zeros(qcat.m, dtype=np.float64)
    coverage = np.zeros(qcat.m, dtype=np.int32)
    measurement_groups: list[MeasurementGroup] = []
    projected_oneway: dict[int, np.ndarray] = {}
    projected_pairs: dict[tuple[int, int], np.ndarray] = {}

    for group in workload_groups:
        if not group.is_partition:
            raise ValueError("Interaction adapter supports complete partition groups only")
        idx = np.asarray(group.query_indices, dtype=np.int32)
        if idx.ndim != 1 or idx.size == 0 or np.any(idx < 0) or np.any(idx >= qcat.m):
            raise ValueError(f"Invalid query indices in group {group.name!r}")
        scope = _scope_from_group(group)
        if len(scope) == 1:
            attr = scope[0]
            raw = reconstructed.oneway[attr]
            variance = reconstructed.oneway_variances[attr]
            projected = (
                project_simplex(raw, float(transcript.public_total)).astype(np.float64)
                if target_projection == "per_scope_simplex"
                else raw.astype(np.float64, copy=True)
            )
            projected_oneway[attr] = projected
        else:
            pair = (scope[0], scope[1])
            raw = reconstructed.pairs[pair]
            variance = reconstructed.pair_variances[pair]
            projected = (
                project_simplex(raw.ravel(), float(transcript.public_total)).reshape(raw.shape)
                if target_projection == "per_scope_simplex"
                else raw.astype(np.float64, copy=True)
            )
            projected_pairs[pair] = projected.astype(np.float64)
        if raw.size != idx.size:
            raise ValueError(
                f"Group {group.name!r} has {idx.size} queries but reconstructed scope has {raw.size} cells"
            )
        raw_target[idx] = raw.ravel()
        projected_target[idx] = projected.ravel()
        variances[idx] = variance.ravel()
        coverage[idx] += 1
        measurement_groups.append(
            MeasurementGroup(
                query_indices=idx.copy(),
                sensitivity_l2=1.0,
                rho=0.0,
                sigma=0.0,
                noise_std=0.0,
                name=group.name,
                family=group.family,
                is_partition=True,
            )
        )

    if not np.all(coverage == 1):
        raise ValueError(
            "Interaction adapter workload groups must cover every reconstructed query exactly once"
        )
    variances = np.maximum(variances, float(min_variance))
    raw_values = np.concatenate(
        [value.ravel() for value in reconstructed.oneway.values()]
        + [value.ravel() for value in reconstructed.pairs.values()]
    )
    diagnostics: dict[str, Any] = {
        "measurement_strategy": {
            "name": "hierarchical_helmert_interactions",
            "adjacency": transcript.adjacency,
            "allocation_mode": transcript.allocation_mode,
            "num_strategy_blocks": len(transcript.strategy.blocks),
            "num_pair_scopes": len(transcript.strategy.pairs),
            "covariance_adapter": "marginal_diagonal_only",
            "covariance_approximation": True,
            "target_projection": target_projection,
            "raw_negative_cells": int(np.sum(raw_values < 0.0)),
            "raw_min_cell": float(np.min(raw_values)),
            "raw_consistency_linf": reconstructed.max_consistency_violation(),
            "p1_consistency_linf": _consistency_linf(projected_oneway, projected_pairs),
        },
        "consistency": {
            "enabled": False,
            "method": (
                "per_scope_simplex_p1"
                if target_projection == "per_scope_simplex"
                else "raw_hierarchical_reconstruction"
            ),
        },
        "uncertainty": {
            "enabled": True,
            "method": "analytic_reconstruction_marginal_diagonal",
            "drops_cross_cell_covariance": True,
            "drops_cross_scope_covariance": True,
        },
    }
    if transcript.adjacency != "add_remove":
        raise ValueError("Static-ICE interaction adapter currently requires add_remove adjacency")
    if transcript.rho_spent > transcript.rho_total + 1.0e-12 * max(1.0, transcript.rho_total):
        raise RuntimeError("Interaction transcript overspent its declared rho_total")
    return Measurements(
        target_noisy=raw_target.astype(np.float32),
        target_projected=projected_target.astype(np.float32),
        variances=variances.astype(np.float32),
        inv_variances=(1.0 / variances).astype(np.float32),
        groups=measurement_groups,
        mode="dp",
        rho_total=float(transcript.rho_total),
        rho_spent=float(transcript.rho_spent),
        epsilon_delta=zcdp_epsilon(float(transcript.rho_spent), float(delta)),
        delta=float(delta),
        projection_diagnostics=diagnostics,
        num_rows=int(transcript.public_total),
        strategy_transcript=transcript.to_public_dict(),
        privacy_ledger=_interaction_privacy_ledger(transcript, delta=float(delta)),
    )


def local_polytope_projection_to_query_target(
    projection: LocalPolytopeProjectionResult,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
) -> np.ndarray:
    target = np.zeros(qcat.m, dtype=np.float64)
    coverage = np.zeros(qcat.m, dtype=np.int32)
    for group in workload_groups:
        if not group.is_partition:
            raise ValueError("Local-polytope adapter supports complete partition groups only")
        indices = np.asarray(group.query_indices, dtype=np.int32)
        scope = _scope_from_group(group)
        if len(scope) == 1:
            values = projection.marginals.oneway[scope[0]]
        else:
            values = projection.marginals.pairs[(scope[0], scope[1])]
        if values.size != indices.size:
            raise ValueError(f"Local-polytope target size mismatch for group {group.name!r}")
        target[indices] = values.reshape(-1)
        coverage[indices] += 1
    if not np.all(coverage == 1):
        raise ValueError("Local-polytope groups must cover every query exactly once")
    return target


def interaction_transcript_to_local_polytope_measurements(
    transcript: HierarchicalInteractionTranscript,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    *,
    delta: float,
) -> tuple[Measurements, LocalPolytopeProjectionResult]:
    """Create a P3 target while retaining raw uncertainty as an explicit control arm."""
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        workload_groups,
        delta=delta,
        target_projection="raw_reconstruction",
    )
    projection = project_hierarchical_local_polytope(transcript)
    projected_target = local_polytope_projection_to_query_target(
        projection,
        qcat,
        workload_groups,
    )
    measurements.target_projected = projected_target.astype(np.float32)
    diagnostics = dict(measurements.projection_diagnostics or {})
    diagnostics["consistency"] = {
        "enabled": True,
        "method": "hierarchical_orthogonal_local_polytope_p3",
        "certificate_passed": bool(projection.diagnostics["certificate_passed"]),
    }
    diagnostics["local_polytope_p3"] = dict(projection.diagnostics)
    diagnostics["uncertainty"] = {
        "enabled": True,
        "method": "raw_coefficient_covariance_control",
        "projection_covariance_propagated": False,
    }
    measurements.projection_diagnostics = diagnostics
    return measurements, projection


def interaction_transcript_to_shrunk_measurements(
    transcript: HierarchicalInteractionTranscript,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    *,
    delta: float,
) -> tuple[Measurements, InteractionShrinkageResult]:
    """Create a hierarchy-consistent target with pair interactions shrunk before P3."""
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        workload_groups,
        delta=delta,
        target_projection="raw_reconstruction",
    )
    shrinkage = positive_part_block_james_stein(transcript)
    shrunk_transcript = shrinkage.shrunk_transcript(transcript)
    shrunk_measurements = interaction_transcript_to_diagonal_measurements(
        shrunk_transcript,
        qcat,
        workload_groups,
        delta=delta,
        target_projection="raw_reconstruction",
    )
    measurements.target_projected = shrunk_measurements.target_projected.copy()
    shrunk_reconstruction = shrunk_transcript.reconstruct()
    shrunk_values = np.concatenate(
        [value.ravel() for value in shrunk_reconstruction.oneway.values()]
        + [value.ravel() for value in shrunk_reconstruction.pairs.values()]
    )
    diagnostics = dict(measurements.projection_diagnostics or {})
    strategy_diagnostics = dict(diagnostics.get("measurement_strategy", {}))
    strategy_diagnostics.update(
        {
            "target_projection": "raw_reconstruction_after_interaction_shrinkage",
            "shrunk_negative_cells": int(np.sum(shrunk_values < 0.0)),
            "shrunk_min_cell": float(np.min(shrunk_values)),
            "shrunk_consistency_linf": shrunk_reconstruction.max_consistency_violation(),
        }
    )
    diagnostics["measurement_strategy"] = strategy_diagnostics
    diagnostics["consistency"] = {
        "enabled": False,
        "method": "raw_hierarchical_reconstruction_after_interaction_shrinkage",
    }
    diagnostics["interaction_shrinkage"] = shrinkage.to_public_dict()
    diagnostics["uncertainty"] = {
        "enabled": True,
        "method": "raw_coefficient_covariance_control",
        "shrinkage_covariance_propagated": False,
    }
    measurements.projection_diagnostics = diagnostics
    return measurements, shrinkage


def interaction_transcript_to_local_polytope_bootdiag_measurements(
    transcript: HierarchicalInteractionTranscript,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    *,
    delta: float,
    rng: np.random.Generator,
    num_samples: int = 16,
    min_variance: float = 1.0e-6,
    min_raw_variance_fraction: float = 0.02,
) -> tuple[
    Measurements,
    LocalPolytopeProjectionResult,
    CoefficientBootstrapDiagonal,
]:
    measurements, projection = interaction_transcript_to_local_polytope_measurements(
        transcript,
        qcat,
        workload_groups,
        delta=delta,
    )
    bootstrap = bootstrap_local_polytope_coefficient_diagonal(
        transcript,
        projection,
        rng=rng,
        num_samples=int(num_samples),
        min_variance=float(min_variance),
        min_raw_variance_fraction=float(min_raw_variance_fraction),
    )
    diagnostics = dict(measurements.projection_diagnostics or {})
    diagnostics["coefficient_bootstrap_diagonal"] = bootstrap.to_public_dict()
    diagnostics["uncertainty"] = {
        "enabled": True,
        "method": "coefficient_space_local_polytope_bootstrap_diagonal",
        "projection_covariance_propagated": True,
        "num_samples": int(num_samples),
        "center": "projected",
        "debias_target": False,
        "min_variance": float(min_variance),
        "min_raw_variance_fraction": float(min_raw_variance_fraction),
    }
    measurements.projection_diagnostics = diagnostics
    return measurements, projection, bootstrap
