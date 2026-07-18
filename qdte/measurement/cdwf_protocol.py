from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np

from qdte.evolution.entropy import ReleasedProductPrior
from qdte.measurement.cdwf import (
    CDWFAllocationResult,
    CDWFBudgetPlan,
    pressure_concentration,
    solve_cdwf_water_filling,
    uniform_cdwf_allocation,
)
from qdte.measurement.cdwf_transcript import (
    CDWFReleasedStream,
    CDWFSequentialTranscript,
    ccf_pair_streams_for_history,
    combine_cdwf_released_history,
    current_cdwf_rho_by_block,
    measure_cdwf_base,
    measure_cdwf_refinement_stream,
)
from qdte.measurement.factorization import HierarchicalPairStrategy
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
)
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.cdwf_shadow import (
    CDWFShadowDictionary,
    build_released_directed_shadow_dictionary,
    sample_released_forest_prior,
)
from qdte.rce.public_domain import PublicLegalRowDomain
from qdte.rce.rhcg_ccmp import (
    CanonicalConfidenceGeometry,
    RHCGColumnSet,
    solve_rhcg_ccmp,
)
from qdte.rce.row_pricing import ShadowFeatureMap, ccf_moment_answer
from qdte.rce.sequential_forest_prior import (
    build_sequential_confidence_forest_prior,
)
from qdte.schema import TableSchema


CDWF_PROTOCOL_METHOD = "sage_qdte_rce_c3_cdwf_rhcg_ccmp_v2"
CDWF_ARMS = ("split_uniform", "dual_water_fill")


@dataclass(frozen=True)
class CDWFRoundRecord:
    round_index: int
    allocation_mode: str
    rho_before: dict[str, float]
    allocation: dict[str, float]
    rho_after: dict[str, float]
    dual_fallback: bool
    fallback_reasons: tuple[str, ...]
    shadow: dict[str, Any] | None
    allocation_solver: dict[str, Any] | None
    ccf_prior: dict[str, Any]
    pressure_concentration_9_24: float | None
    pressure_concentration_23_40: float | None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "allocation_mode": self.allocation_mode,
            "rho_before": dict(self.rho_before),
            "allocation": dict(self.allocation),
            "rho_after": dict(self.rho_after),
            "dual_fallback": self.dual_fallback,
            "fallback_reasons": list(self.fallback_reasons),
            "shadow": self.shadow,
            "allocation_solver": self.allocation_solver,
            "ccf_prior": self.ccf_prior,
            "pressure_concentration_9_24": self.pressure_concentration_9_24,
            "pressure_concentration_23_40": self.pressure_concentration_23_40,
        }


@dataclass(frozen=True)
class CDWFMeasurementRun:
    arm: str
    transcript: CDWFSequentialTranscript
    public_row_domain: PublicLegalRowDomain
    rhcg_columns: RHCGColumnSet
    rhcg_warm_start: dict[str, Any]
    rounds: tuple[CDWFRoundRecord, ...]
    base_ccf_prior: dict[str, Any]
    method: str = CDWF_PROTOCOL_METHOD

    def __post_init__(self) -> None:
        if self.method != CDWF_PROTOCOL_METHOD or self.arm not in CDWF_ARMS:
            raise ValueError("Unsupported CDWF measurement run")
        if len(self.rounds) != self.transcript.budget.rounds:
            raise ValueError("CDWF measurement run omitted a declared round")
        if self.public_row_domain.public_n != self.transcript.public_total:
            raise ValueError("CDWF public row domain and transcript disagree on public n")

    @property
    def dual_fallback_count(self) -> int:
        return sum(record.dual_fallback for record in self.rounds)

    @property
    def promotion_eligible_dual(self) -> bool:
        return self.arm == "dual_water_fill" and self.dual_fallback_count == 0

    def concentration_diagnostics(self) -> dict[str, Any]:
        final_gamma = self.transcript.final_gamma_by_block()
        control = self.transcript.budget.control_rho_by_block
        eligible = self.transcript.budget.eligible_interaction_blocks
        total_mass = math.fsum(control[name] for name in eligible)
        mass_at = {
            threshold: math.fsum(
                control[name]
                for name in eligible
                if final_gamma[name] >= threshold - 1.0e-12
            )
            / total_mass
            for threshold in (2.0, 3.0, 4.0)
        }
        refined = [
            name
            for name in eligible
            if final_gamma[name] > self.transcript.budget.base_fraction + 1.0e-12
        ]
        return {
            "final_gamma_by_block": final_gamma,
            "number_of_refined_blocks": len(refined),
            "control_rho_mass_gamma_ge_2": mass_at[2.0],
            "control_rho_mass_gamma_ge_3": mass_at[3.0],
            "control_rho_mass_gamma_ge_4": mass_at[4.0],
            "dual_fallback_count": self.dual_fallback_count,
            "promotion_eligible_dual": self.promotion_eligible_dual,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "arm": self.arm,
            "selection_rho": 0.0,
            "truth_accessed_by_allocation": False,
            "per_round_full_qdte": False,
            "final_qdte_runs": 1,
            "public_row_domain": {
                **self.public_row_domain.manifest,
                "manifest_sha256": self.public_row_domain.manifest_hash,
            },
            "rhcg_columns": self.rhcg_columns.to_public_dict(),
            "rhcg_warm_start": dict(self.rhcg_warm_start),
            "base_ccf_prior": dict(self.base_ccf_prior),
            "rounds": [record.to_public_dict() for record in self.rounds],
            "concentration": self.concentration_diagnostics(),
        }


def _round_rngs(
    public_seed: int,
    round_index: int,
    eligible_blocks: Sequence[str],
) -> dict[str, np.random.Generator]:
    return {
        name: np.random.default_rng(
            np.random.SeedSequence(
                [int(public_seed), 0x43445746, 0x52454649, int(round_index), index]
            )
        )
        for index, name in enumerate(eligible_blocks)
    }


def _released_state(
    *,
    strategy: HierarchicalPairStrategy,
    plan: CDWFBudgetPlan,
    streams: Sequence[CDWFReleasedStream],
    public_total: int,
    qcat: QueryCatalogue,
    workload_groups: Sequence[WorkloadGroup],
    schema: TableSchema,
):
    combined = combine_cdwf_released_history(
        strategy,
        int(public_total),
        plan,
        streams,
    )
    measurements = interaction_transcript_to_diagonal_measurements(
        combined,
        qcat,
        list(workload_groups),
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    prior = build_sequential_confidence_forest_prior(
        qcat,
        measurements.target_projected,
        schema.cardinalities,
        ccf_pair_streams_for_history(
            strategy,
            int(public_total),
            plan,
            streams,
        ),
        public_total=int(public_total),
        smoothing=1.0,
    )
    return combined, measurements, prior


def _rhcg_shadow_geometry(
    *,
    feature_map: ShadowFeatureMap,
    combined,
    rho_by_block: dict[str, float],
) -> CanonicalConfidenceGeometry:
    target_parts: list[np.ndarray] = []
    variance_parts: list[np.ndarray] = []
    tolerance = 1.0e-10 * max(1.0, math.fsum(rho_by_block.values()))
    for block in feature_map.strategy.blocks:
        if block.name not in combined.noisy_components:
            raise ValueError(f"Combined C3 transcript omitted block {block.name!r}")
        target_parts.append(
            np.asarray(combined.noisy_components[block.name], dtype=np.float64).reshape(-1)
        )
        variance = float(combined.component_variances[block.name])
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("Combined C3 covariance must be finite and positive")
        variance_parts.append(np.full(block.dimension, variance, dtype=np.float64))
        if abs(float(combined.rho_by_block[block.name]) - rho_by_block[block.name]) > tolerance:
            raise ValueError("Combined C3 precision disagrees with the privacy ledger")
    target = np.concatenate(target_parts)
    variances = np.concatenate(variance_parts)
    return CanonicalConfidenceGeometry(
        feature_map=feature_map,
        target=target,
        confidence=RCEConfidenceSet.from_diagonal_variances(
            variances,
            alpha_l2=0.025,
            alpha_linf=0.025,
        ),
        rho_by_block=rho_by_block,
    )


def run_cdwf_adaptive_measurement(
    rows: np.ndarray,
    schema: TableSchema,
    qcat: QueryCatalogue,
    workload_groups: Sequence[WorkloadGroup],
    strategy: HierarchicalPairStrategy,
    plan: CDWFBudgetPlan,
    *,
    public_total: int,
    arm: str,
    base_noise_seed: int,
    refinement_noise_seed: int,
    generation_seed: int,
    shadow_dictionary_rounds: int = 512,
    shadow_max_iterations: int = 5_000,
    answer_batch_size: int = 8192,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> CDWFMeasurementRun:
    def progress(stage: str, **values: Any) -> None:
        if progress_callback is not None:
            progress_callback(stage, values)

    mode = str(arm)
    if mode not in CDWF_ARMS:
        raise ValueError(f"CDWF arm must be one of {CDWF_ARMS}")
    public_domain = PublicLegalRowDomain.from_schema(
        schema,
        public_n=int(public_total),
    )
    feature_map = ShadowFeatureMap(strategy, public_domain)
    private_rows = np.asarray(rows, dtype=np.int32)
    if private_rows.shape != (int(public_total), schema.d):
        raise ValueError("CDWF private rows do not match the public schema and row count")
    base = measure_cdwf_base(
        private_rows,
        strategy,
        plan,
        public_total=int(public_total),
        rng=np.random.default_rng(int(base_noise_seed)),
    )
    progress("base_measurement_complete", num_observations=len(base.observations))
    streams: list[CDWFReleasedStream] = [base]
    combined, measurements, prior = _released_state(
        strategy=strategy,
        plan=plan,
        streams=streams,
        public_total=int(public_total),
        qcat=qcat,
        workload_groups=workload_groups,
        schema=schema,
    )
    progress(
        "base_released_state_complete",
        selected_forest_edges=len(prior.edges),
    )
    base_ccf_diagnostics = prior.diagnostics()
    product_prior = ReleasedProductPrior.from_released_oneway(
        qcat,
        measurements.target_projected,
        schema.cardinalities,
        public_total=int(public_total),
        smoothing=1.0,
    )
    initial_rows = sample_released_forest_prior(
        prior,
        int(public_total),
        np.random.default_rng(
            np.random.SeedSequence(
                [int(generation_seed), 0x52484347, 0x5741524D]
            )
        ),
    )
    warm_dictionary = CDWFShadowDictionary.create(
        initial_rows=initial_rows,
        product_prior=product_prior,
        ccf_prior=prior,
        public_seed=int(generation_seed),
    )
    progress(
        "rhcg_warm_start_started",
        anchor_tables=len(warm_dictionary.tables),
        directed_rounds=int(shadow_dictionary_rounds),
    )
    warm_dictionary = build_released_directed_shadow_dictionary(
        warm_dictionary,
        schema=schema,
        qcat=qcat,
        workload_groups=workload_groups,
        transcript=combined,
        released_target=measurements.target_projected,
        max_rounds=int(shadow_dictionary_rounds),
        feature_batch_size=int(answer_batch_size),
    )
    if warm_dictionary.coefficient_answers is None:
        raise RuntimeError("Released RHCG warm start omitted coefficient answers")
    warm_indices = tuple(
        dict.fromkeys((0, 1, 2, len(warm_dictionary.names) - 1))
    )
    warm_names = tuple(warm_dictionary.names[index] for index in warm_indices)
    warm_answers = np.stack(
        [warm_dictionary.coefficient_answers[index] for index in warm_indices],
        axis=0,
    )
    if warm_answers.shape[1] != feature_map.feature_dimension:
        raise RuntimeError("Released RHCG warm start uses a different shadow feature map")
    columns = RHCGColumnSet.create(
        feature_map,
        warm_names=warm_names,
        warm_answers=warm_answers,
    )
    warm_start_diagnostics = {
        **warm_dictionary.to_public_dict(),
        "role": "aggregate_warm_start_only",
        "global_pricing_certificate": False,
        "retained_column_names": list(warm_names),
        "retained_column_count": len(warm_names),
    }
    progress(
        "rhcg_warm_start_complete",
        aggregate_columns=columns.column_count,
        path_snapshot_columns=len(warm_dictionary.names),
        path_rounds=int(
            (warm_dictionary.path_diagnostics or {}).get("rounds_executed", 0)
        ),
        dictionary_sha256=columns.dictionary_hash,
    )
    progress(
        "rhcg_public_domain_sealed",
        schema_sha256=public_domain.schema_hash,
        manifest_sha256=public_domain.manifest_hash,
        domain_size=public_domain.domain_size,
        shadow_affine_rank=feature_map.affine_rank,
        column_cap=feature_map.atom_cap,
    )

    records: list[CDWFRoundRecord] = []
    for round_index in range(plan.rounds):
        rho_before = current_cdwf_rho_by_block(plan, streams)
        shadow_payload: dict[str, Any] | None = None
        allocation_result: CDWFAllocationResult | None = None
        fallback = False
        reasons: tuple[str, ...] = ()
        concentration_9: float | None = None
        concentration_23: float | None = None
        try:
            progress("rhcg_ccmp_started", round_index=round_index)
            geometry = _rhcg_shadow_geometry(
                feature_map=feature_map,
                combined=combined,
                rho_by_block=rho_before,
            )
            shadow = solve_rhcg_ccmp(
                geometry,
                columns,
                ccf_moment_answer(feature_map, prior),
                pricing_solver="auto",
                global_gap_tolerance=1.0e-8,
                maximum_columns=feature_map.atom_cap,
                max_iterations=int(shadow_max_iterations),
                progress_callback=lambda stage, values: progress(
                    f"rhcg_{stage}",
                    round_index=round_index,
                    **values,
                ),
            )
            shadow_payload = shadow.to_public_dict()
            fallback = not shadow.certified
            reasons = shadow.fallback_reasons
            if shadow.certified:
                columns = shadow.phase_two.columns
            progress(
                "rhcg_ccmp_complete",
                round_index=round_index,
                certified=shadow.certified,
                fallback_reasons=list(reasons),
                phase_one_inflation=shadow.phase_one.inflation,
                column_count=shadow.phase_two.columns.column_count,
            )
            if mode == "dual_water_fill" and not fallback:
                pressures = {
                    name: shadow.pressure_by_block[name]
                    for name in plan.eligible_interaction_blocks
                }
                allocation_result = solve_cdwf_water_filling(
                    plan,
                    rho_before,
                    pressures,
                )
                concentration_9 = pressure_concentration(
                    pressures,
                    {
                        name: plan.control_rho_by_block[name]
                        for name in plan.eligible_interaction_blocks
                    },
                    mass_fraction=11.0 / 119.0,
                )
                concentration_23 = pressure_concentration(
                    pressures,
                    {
                        name: plan.control_rho_by_block[name]
                        for name in plan.eligible_interaction_blocks
                    },
                    mass_fraction=11.0 / 47.0,
                )
        except (RuntimeError, ValueError) as error:
            fallback = True
            reasons = (
                f"{error.__class__.__name__}:{str(error)}",
            )
            progress(
                "rhcg_ccmp_failed_closed",
                round_index=round_index,
                fallback_reasons=list(reasons),
                column_count=columns.column_count,
            )
        allocation = (
            allocation_result.allocation
            if allocation_result is not None
            else uniform_cdwf_allocation(plan)
        )
        refinement = measure_cdwf_refinement_stream(
            private_rows,
            strategy,
            allocation,
            round_index=round_index,
            rng_by_block=_round_rngs(
                int(refinement_noise_seed),
                round_index,
                plan.eligible_interaction_blocks,
            ),
        )
        progress(
            "refinement_measurement_complete",
            round_index=round_index,
            measured_blocks=len(refinement.observations),
        )
        streams.append(refinement)
        rho_after = current_cdwf_rho_by_block(plan, streams)
        records.append(
            CDWFRoundRecord(
                round_index=round_index,
                allocation_mode=(
                    "dual_water_filling"
                    if allocation_result is not None
                    else "public_uniform"
                ),
                rho_before=rho_before,
                allocation=allocation,
                rho_after=rho_after,
                dual_fallback=bool(fallback),
                fallback_reasons=reasons,
                shadow=shadow_payload,
                allocation_solver=(
                    allocation_result.to_public_dict()
                    if allocation_result is not None
                    else None
                ),
                ccf_prior=prior.diagnostics(),
                pressure_concentration_9_24=concentration_9,
                pressure_concentration_23_40=concentration_23,
            )
        )
        combined, measurements, prior = _released_state(
            strategy=strategy,
            plan=plan,
            streams=streams,
            public_total=int(public_total),
            qcat=qcat,
            workload_groups=workload_groups,
            schema=schema,
        )
        progress(
            "round_released_state_complete",
            round_index=round_index,
            selected_forest_edges=len(prior.edges),
        )

    transcript = CDWFSequentialTranscript(
        strategy=strategy,
        public_total=int(public_total),
        budget=plan,
        streams=tuple(streams),
    )
    return CDWFMeasurementRun(
        arm=mode,
        transcript=transcript,
        public_row_domain=public_domain,
        rhcg_columns=columns,
        rhcg_warm_start=warm_start_diagnostics,
        rounds=tuple(records),
        base_ccf_prior=base_ccf_diagnostics,
    )


__all__ = [
    "CDWF_ARMS",
    "CDWF_PROTOCOL_METHOD",
    "CDWFMeasurementRun",
    "CDWFRoundRecord",
    "run_cdwf_adaptive_measurement",
]
