from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from qdte.measurement.factorization import (
    ORACLE_PRECISION_HOMOTOPY_METHOD,
    HierarchicalInteractionTranscript,
)
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
)
from qdte.measurement.measure import Measurements
from qdte.queries.orthogonal import interaction_coefficients
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup


PRECISION_HOMOTOPY_PROTOCOL = "SAGE-QDTE-RCE-C3-PRECISION-HOMOTOPY-20260717-v3"
FINITE_PRECISION_GAMMAS = (1.0, 2.0, 4.0, 8.0)
INFINITY_GAMMA = "infinity"


def normalize_precision_gamma(value: float | int | str) -> float | str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            return INFINITY_GAMMA
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported precision homotopy gamma {value!r}") from exc
    gamma = float(value)
    if not math.isfinite(gamma) or gamma not in FINITE_PRECISION_GAMMAS:
        raise ValueError(
            "Finite precision homotopy gamma must be one of "
            f"{FINITE_PRECISION_GAMMAS}"
        )
    return gamma


def _transcript_sha256(transcript: HierarchicalInteractionTranscript) -> str:
    payload = json.dumps(
        transcript.to_public_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def precision_homotopy_transcript(
    released: HierarchicalInteractionTranscript,
    private_rows: np.ndarray,
    gamma: float | int | str,
) -> HierarchicalInteractionTranscript:
    """Construct an offline oracle interaction precision ceiling.

    One-way blocks remain byte-identical. Pair centers share the released
    Gaussian residual and pair covariance is divided by ``gamma``. Infinity
    uses exact zero-variance pair coordinates and is not approximated by a
    variance floor.
    """

    normalized_gamma = normalize_precision_gamma(gamma)
    rows = np.asarray(private_rows, dtype=np.int64)
    cards = released.strategy.cardinalities
    if rows.ndim != 2 or rows.shape[1] != len(cards):
        raise ValueError("Precision homotopy private rows do not match the strategy")
    if rows.shape[0] != int(released.public_total):
        raise ValueError("Precision homotopy private rows must match public_total")
    for attribute, cardinality in enumerate(cards):
        values = rows[:, attribute]
        if np.any(values < 0) or np.any(values >= cardinality):
            raise ValueError("Precision homotopy private rows are outside the public domain")

    components: dict[str, np.ndarray] = {}
    variances: dict[str, float] = {}
    exact = normalized_gamma == INFINITY_GAMMA
    scale = 0.0 if exact else 1.0 / math.sqrt(float(normalized_gamma))
    for block in released.strategy.blocks:
        name = block.name
        released_values = np.asarray(released.noisy_components[name], dtype=np.float64)
        if block.kind == "oneway_contrast":
            components[name] = released_values.copy()
            variances[name] = float(released.component_variances[name])
            continue
        if block.kind != "pair_interaction" or len(block.scope) != 2:
            raise ValueError(f"Unsupported homotopy block {name!r}")
        truth = interaction_coefficients(rows, block.scope, cards)
        components[name] = truth + scale * (released_values - truth)
        variances[name] = (
            0.0
            if exact
            else float(released.component_variances[name]) / float(normalized_gamma)
        )

    diagnostics: dict[str, Any] = {
        "method": ORACLE_PRECISION_HOMOTOPY_METHOD,
        "protocol_id": PRECISION_HOMOTOPY_PROTOCOL,
        "gamma": normalized_gamma,
        "exact_interaction_equality": exact,
        "oneway_center": "released_unchanged",
        "oneway_covariance": "released_unchanged",
        "interaction_center": "truth_plus_shared_dp_residual_over_sqrt_gamma",
        "interaction_covariance": (
            "exact_zero"
            if exact
            else "released_covariance_divided_by_gamma"
        ),
        "source_allocation_mode": released.allocation_mode,
        "source_transcript_sha256": _transcript_sha256(released),
        "private_truth_used": True,
        "promotion_eligible": False,
        "privacy_claim": "none_offline_oracle_ceiling",
    }
    result = HierarchicalInteractionTranscript(
        strategy=released.strategy,
        public_total=int(released.public_total),
        noisy_components=components,
        component_variances=variances,
        rho_by_block=dict(released.rho_by_block),
        rho_total=float(released.rho_total),
        rho_spent=float(released.rho_spent),
        allocation_mode="oracle_precision_homotopy",
        adjacency=released.adjacency,
        refinement_diagnostics=diagnostics,
    )
    # Exercise the serialized validation path used by the engine.
    HierarchicalInteractionTranscript.from_public_dict(result.to_public_dict())
    return result


def precision_homotopy_measurements(
    released_measurements: Measurements,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    private_rows: np.ndarray,
    gamma: float | int | str,
) -> tuple[Measurements, HierarchicalInteractionTranscript]:
    if released_measurements.strategy_transcript is None:
        raise ValueError("Precision homotopy requires an interaction transcript")
    released = HierarchicalInteractionTranscript.from_public_dict(
        released_measurements.strategy_transcript
    )
    transformed = precision_homotopy_transcript(released, private_rows, gamma)
    measurements = interaction_transcript_to_diagonal_measurements(
        transformed,
        qcat,
        workload_groups,
        delta=float(released_measurements.delta),
        target_projection="raw_reconstruction",
    )
    measurements.mode = "oracle"
    measurements.rho_total = float(released_measurements.rho_total)
    measurements.rho_spent = float(released_measurements.rho_spent)
    measurements.epsilon_delta = float(released_measurements.epsilon_delta)
    measurements.privacy_ledger = {
        "accounting": "offline_oracle_diagnostic_no_privacy_claim",
        "protocol_id": PRECISION_HOMOTOPY_PROTOCOL,
        "source_rho_total": float(released_measurements.rho_total),
        "source_rho_spent": float(released_measurements.rho_spent),
        "additional_privacy_claim": None,
        "private_truth_used": True,
        "promotion_eligible": False,
    }
    projection = dict(measurements.projection_diagnostics or {})
    projection["offline_precision_homotopy"] = dict(
        transformed.refinement_diagnostics or {}
    )
    measurements.projection_diagnostics = projection
    return measurements, transformed


__all__ = [
    "FINITE_PRECISION_GAMMAS",
    "INFINITY_GAMMA",
    "PRECISION_HOMOTOPY_PROTOCOL",
    "normalize_precision_gamma",
    "precision_homotopy_measurements",
    "precision_homotopy_transcript",
]
