from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np

from qdte.evolution.initialization import initialize_independent_oneway
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
from qdte.rce.cdwf_shadow import (
    CDWFShadowDictionary,
    build_released_directed_shadow_dictionary,
    solve_cdwf_shadow_rce,
)
from qdte.rce.sequential_forest_prior import (
    build_sequential_confidence_forest_prior,
)
from qdte.schema import TableSchema


CDWF_PROTOCOL_METHOD = "sage_qdte_rce_c3_cdwf_measurement_v1"
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
    shadow_dictionary: CDWFShadowDictionary | None
    rounds: tuple[CDWFRoundRecord, ...]
    base_ccf_prior: dict[str, Any]
    method: str = CDWF_PROTOCOL_METHOD

    def __post_init__(self) -> None:
        if self.method != CDWF_PROTOCOL_METHOD or self.arm not in CDWF_ARMS:
            raise ValueError("Unsupported CDWF measurement run")
        if len(self.rounds) != self.transcript.budget.rounds:
            raise ValueError("CDWF measurement run omitted a declared round")
        if self.arm == "dual_water_fill" and self.shadow_dictionary is None:
            raise ValueError("Dual CDWF requires its frozen shadow dictionary")

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
            "shadow_dictionary": (
                self.shadow_dictionary.to_public_dict()
                if self.shadow_dictionary is not None
                else None
            ),
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
    dictionary: CDWFShadowDictionary | None = None
    if mode == "dual_water_fill":
        initial_rows = initialize_independent_oneway(
            qcat,
            measurements.target_projected,
            schema,
            int(public_total),
            np.random.default_rng(int(generation_seed)),
        )
        dictionary = CDWFShadowDictionary.create(
            initial_rows=initial_rows,
            product_prior=prior.product_prior,
            ccf_prior=prior,
            public_seed=int(generation_seed),
        )
        progress("shadow_anchor_dictionary_complete", num_tables=len(dictionary.tables))
        dictionary = build_released_directed_shadow_dictionary(
            dictionary,
            schema=schema,
            qcat=qcat,
            workload_groups=workload_groups,
            transcript=combined,
            released_target=measurements.target_projected,
            max_rounds=int(shadow_dictionary_rounds),
            candidates_per_round=1_024,
            accepted_per_round=64,
        )
        progress(
            "shadow_directed_dictionary_complete",
            num_tables=len(dictionary.tables),
            entered_confidence_set=bool(
                (dictionary.path_diagnostics or {}).get(
                    "entered_confidence_set",
                    False,
                )
            ),
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
        if mode == "dual_water_fill":
            if dictionary is None:
                raise AssertionError("CDWF dual dictionary was not frozen")
            try:
                progress("shadow_solve_started", round_index=round_index)
                shadow = solve_cdwf_shadow_rce(
                    dictionary,
                    qcat=qcat,
                    workload_groups=workload_groups,
                    transcript=combined,
                    released_target=measurements.target_projected,
                    prior=prior,
                    rho_by_block=rho_before,
                    answer_batch_size=int(answer_batch_size),
                    max_iterations=int(shadow_max_iterations),
                )
                shadow_payload = shadow.to_public_dict()
                progress(
                    "shadow_solve_complete",
                    round_index=round_index,
                    certified=not shadow.fallback_required,
                    fallback_reasons=list(shadow.failure_reasons),
                )
                fallback = shadow.fallback_required
                reasons = shadow.failure_reasons
                if not fallback and shadow.pressure is not None:
                    pressures = {
                        name: shadow.pressure.pressure_by_block[name]
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
                dual_fallback=bool(mode == "dual_water_fill" and fallback),
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
        shadow_dictionary=dictionary,
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
