from __future__ import annotations

from typing import Literal

import numpy as np

from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
)
from qdte.measurement.measure import Measurements, measurements_from_public_dict
from qdte.measurement.support_coarsening import (
    SupportCoarsening,
    oracle_support_coarsening,
    released_support_coarsening,
)
from qdte.queries.orthogonal import interaction_coefficients
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup


C0_PROTOCOL = "SAGE-QDTE-RCE-C0-CCF-20260715-v3"
C0_SUPPORT_MODES = {"released", "oracle"}
C0_INTERACTION_CENTERS = {"dp", "clean"}


def clean_interaction_transcript(
    transcript: HierarchicalInteractionTranscript,
    private_rows: np.ndarray,
) -> HierarchicalInteractionTranscript:
    """Replace only interaction centers with truth for an offline C0 diagnostic."""

    rows = np.asarray(private_rows, dtype=np.int64)
    cards = transcript.strategy.cardinalities
    if rows.ndim != 2 or rows.shape[1] != len(cards):
        raise ValueError("C0 private rows do not match the public interaction strategy")
    if len(rows) != int(transcript.public_total):
        raise ValueError("C0 private rows must match the declared public row count")
    for attribute, cardinality in enumerate(cards):
        values = rows[:, attribute]
        if np.any(values < 0) or np.any(values >= cardinality):
            raise ValueError("C0 private rows contain an out-of-domain category")

    components = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in transcript.noisy_components.items()
    }
    for pair in transcript.strategy.pairs:
        name = f"pair_interaction:{pair[0]}:{pair[1]}"
        components[name] = interaction_coefficients(rows, pair, cards)
    return HierarchicalInteractionTranscript(
        strategy=transcript.strategy,
        public_total=int(transcript.public_total),
        noisy_components=components,
        component_variances=dict(transcript.component_variances),
        rho_by_block=dict(transcript.rho_by_block),
        rho_total=float(transcript.rho_total),
        rho_spent=float(transcript.rho_spent),
        allocation_mode=transcript.allocation_mode,
        adjacency=transcript.adjacency,
        refinement_diagnostics=(
            dict(transcript.refinement_diagnostics)
            if transcript.refinement_diagnostics is not None
            else None
        ),
    )


def c0_support_coarsening(
    released_transcript: HierarchicalInteractionTranscript,
    private_rows: np.ndarray,
    support_mode: Literal["released", "oracle"],
) -> SupportCoarsening:
    mode = str(support_mode)
    if mode not in C0_SUPPORT_MODES:
        raise ValueError(f"Unsupported C0 support mode {mode!r}")
    released = released_support_coarsening(released_transcript)
    if mode == "released":
        return released
    return oracle_support_coarsening(released, private_rows)


def c0_diagnostic_measurements(
    measurements: Measurements,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    private_rows: np.ndarray,
    interaction_center: Literal["dp", "clean"],
) -> tuple[Measurements, HierarchicalInteractionTranscript]:
    """Build an explicitly non-DP C0 artifact while preserving DP covariance."""

    center = str(interaction_center)
    if center not in C0_INTERACTION_CENTERS:
        raise ValueError(f"Unsupported C0 interaction center {center!r}")
    if measurements.strategy_transcript is None:
        raise ValueError("C0 requires the sealed hierarchical interaction transcript")
    released_transcript = HierarchicalInteractionTranscript.from_public_dict(
        measurements.strategy_transcript
    )
    source_reference = interaction_transcript_to_diagonal_measurements(
        released_transcript,
        qcat,
        workload_groups,
        delta=float(measurements.delta),
        target_projection="raw_reconstruction",
    )
    if not np.array_equal(
        measurements.target_projected,
        source_reference.target_projected,
    ):
        maximum = float(
            np.max(
                np.abs(
                    measurements.target_projected.astype(np.float64)
                    - source_reference.target_projected.astype(np.float64)
                )
            )
        )
        raise ValueError(
            "C0 requires the frozen raw orthogonal target; "
            f"max_abs_difference={maximum:.6g}"
        )

    if center == "dp":
        diagnostic = measurements_from_public_dict(measurements.to_public_dict())
        selected_transcript = released_transcript
    else:
        selected_transcript = clean_interaction_transcript(
            released_transcript,
            private_rows,
        )
        diagnostic = interaction_transcript_to_diagonal_measurements(
            selected_transcript,
            qcat,
            workload_groups,
            delta=float(measurements.delta),
            target_projection="raw_reconstruction",
        )
        if not np.array_equal(diagnostic.variances, source_reference.variances):
            raise RuntimeError("C0 clean center changed the frozen DP variances")
        for attribute in range(len(selected_transcript.strategy.cardinalities)):
            name = f"oneway_contrast:{attribute}"
            if not np.array_equal(
                selected_transcript.noisy_components[name],
                released_transcript.noisy_components[name],
            ):
                raise RuntimeError("C0 clean center changed a released one-way anchor")

    diagnostic.mode = "oracle"
    projection_diagnostics = dict(diagnostic.projection_diagnostics or {})
    projection_diagnostics["offline_c0_diagnostic"] = {
        "protocol_id": C0_PROTOCOL,
        "private_truth_used": center == "clean",
        "interaction_center": center,
        "oneway_anchors": "released_unchanged",
        "covariance": "frozen_dp_covariance",
        "promotion_eligible": False,
    }
    diagnostic.projection_diagnostics = projection_diagnostics
    return diagnostic, selected_transcript


__all__ = [
    "C0_INTERACTION_CENTERS",
    "C0_PROTOCOL",
    "C0_SUPPORT_MODES",
    "c0_diagnostic_measurements",
    "c0_support_coarsening",
    "clean_interaction_transcript",
]
