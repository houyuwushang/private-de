from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

from qdte.measurement.adaptive_interactions import (
    InteractionActionRelease,
    combine_coverage_refinements,
    measure_interaction_action,
)
from qdte.measurement.cdwf import (
    CDWF_METHOD,
    CDWFBudgetPlan,
    coefficient_block_slices,
)
from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    HierarchicalPairStrategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
)
from qdte.measurement.measure import Measurements
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup
from qdte.rce.sequential_forest_prior import CCFInteractionStream
from qdte.rce.streamwise import StreamwiseRCEConfidenceSet, build_cdwf_streamwise_confidence


CDWF_TRANSCRIPT_METHOD = "adaptive_gaussian_interaction_streams_v1"


@dataclass(frozen=True)
class CDWFBlockObservation:
    block_name: str
    kind: str
    scope: tuple[int, ...]
    noisy_coefficients: np.ndarray
    coefficient_variance: float
    rho: float
    sensitivity_l2: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.noisy_coefficients, dtype=np.float64)
        if not self.block_name or self.kind not in {"oneway_contrast", "pair_interaction"}:
            raise ValueError("CDWF observation has invalid block metadata")
        if coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
            raise ValueError("CDWF observation coefficients must be finite and non-empty")
        variance = float(self.coefficient_variance)
        rho = float(self.rho)
        sensitivity = float(self.sensitivity_l2)
        if min(variance, rho, sensitivity) <= 0.0 or not all(
            math.isfinite(value) for value in (variance, rho, sensitivity)
        ):
            raise ValueError("CDWF observation variance, rho, and sensitivity must be positive")
        expected = sensitivity * sensitivity / (2.0 * rho)
        if not math.isclose(variance, expected, rel_tol=1.0e-11, abs_tol=1.0e-14):
            raise ValueError("CDWF observation variance does not match its rho")
        coefficients = coefficients.copy()
        coefficients.setflags(write=False)
        object.__setattr__(self, "noisy_coefficients", coefficients)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "block_name": self.block_name,
            "kind": self.kind,
            "scope": list(self.scope),
            "coefficient_shape": list(self.noisy_coefficients.shape),
            "noisy_coefficients": self.noisy_coefficients.tolist(),
            "coefficient_variance": self.coefficient_variance,
            "rho": self.rho,
            "sensitivity_l2": self.sensitivity_l2,
        }

    @classmethod
    def from_public_dict(cls, data: Mapping[str, Any]) -> CDWFBlockObservation:
        return cls(
            block_name=str(data["block_name"]),
            kind=str(data["kind"]),
            scope=tuple(int(value) for value in data["scope"]),
            noisy_coefficients=np.asarray(data["noisy_coefficients"], dtype=np.float64),
            coefficient_variance=float(data["coefficient_variance"]),
            rho=float(data["rho"]),
            sensitivity_l2=float(data["sensitivity_l2"]),
        )


@dataclass(frozen=True)
class CDWFReleasedStream:
    name: str
    stage: str
    round_index: int
    observations: tuple[CDWFBlockObservation, ...]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        if not self.name or not observations:
            raise ValueError("CDWF released stream must be named and non-empty")
        if self.stage == "base":
            if self.round_index != -1:
                raise ValueError("CDWF base stream must use round_index=-1")
        elif self.stage == "refinement":
            if self.round_index < 0:
                raise ValueError("CDWF refinement streams need a nonnegative round index")
            if any(observation.kind != "pair_interaction" for observation in observations):
                raise ValueError("CDWF refinement streams may contain only interactions")
        else:
            raise ValueError("CDWF stream stage must be base or refinement")
        names = [observation.block_name for observation in observations]
        if len(set(names)) != len(names):
            raise ValueError("CDWF stream contains duplicate block observations")
        object.__setattr__(self, "observations", observations)

    @property
    def rho_spent(self) -> float:
        return float(math.fsum(observation.rho for observation in self.observations))

    def by_name(self) -> dict[str, CDWFBlockObservation]:
        return {observation.block_name: observation for observation in self.observations}

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "round_index": self.round_index,
            "rho_spent": self.rho_spent,
            "observations": [observation.to_public_dict() for observation in self.observations],
        }

    @classmethod
    def from_public_dict(cls, data: Mapping[str, Any]) -> CDWFReleasedStream:
        result = cls(
            name=str(data["name"]),
            stage=str(data["stage"]),
            round_index=int(data["round_index"]),
            observations=tuple(
                CDWFBlockObservation.from_public_dict(observation)
                for observation in data["observations"]
            ),
        )
        if not math.isclose(
            result.rho_spent,
            float(data["rho_spent"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        ):
            raise ValueError("Serialized CDWF stream spend is inconsistent")
        return result


def _plan_from_public_dict(data: Mapping[str, Any]) -> CDWFBudgetPlan:
    return CDWFBudgetPlan(
        method=str(data["method"]),
        control_rho_by_block={
            str(name): float(value)
            for name, value in data["control_rho_by_block"].items()
        },
        base_rho_by_block={
            str(name): float(value)
            for name, value in data["base_rho_by_block"].items()
        },
        eligible_interaction_blocks=tuple(data["eligible_interaction_blocks"]),
        frozen_blocks=tuple(data["frozen_blocks"]),
        rounds=int(data["rounds"]),
        rho_total=float(data["rho_total"]),
        rho_interaction_control=float(data["rho_interaction_control"]),
        rho_refinement=float(data["rho_refinement"]),
        rho_refinement_per_round=float(data["rho_refinement_per_round"]),
        base_fraction=float(data["base_fraction"]),
        refinement_fraction=float(data["refinement_fraction"]),
        maximum_gamma=float(data["maximum_gamma"]),
    )


def _validate_released_history(
    strategy: HierarchicalPairStrategy,
    budget: CDWFBudgetPlan,
    streams: Sequence[CDWFReleasedStream],
) -> tuple[CDWFReleasedStream, ...]:
    released = tuple(streams)
    if not released or released[0].stage != "base":
        raise ValueError("CDWF released history must begin with the base stream")
    if len(released) > budget.rounds + 1:
        raise ValueError("CDWF released history has more rounds than declared")
    if [stream.round_index for stream in released[1:]] != list(
        range(len(released) - 1)
    ):
        raise ValueError("CDWF released history rounds must be contiguous and ordered")
    expected = {block.name for block in strategy.blocks}
    base = released[0].by_name()
    if set(base) != expected:
        raise ValueError("CDWF released history base must cover every strategy block")
    tolerance = 1.0e-12 * max(1.0, budget.rho_total)
    for name, observation in base.items():
        block = strategy.block(name)
        if (
            observation.kind != block.kind
            or observation.scope != block.scope
            or observation.noisy_coefficients.shape != block.coefficient_shape
            or not math.isclose(
                observation.rho,
                budget.base_rho_by_block[name],
                rel_tol=1.0e-11,
                abs_tol=tolerance,
            )
        ):
            raise ValueError(f"CDWF released history base mismatch for {name}")
    eligible = set(budget.eligible_interaction_blocks)
    for stream in released[1:]:
        if not set(stream.by_name()) <= eligible:
            raise ValueError("CDWF released history contains an ineligible refinement")
        if not math.isclose(
            stream.rho_spent,
            budget.rho_refinement_per_round,
            rel_tol=1.0e-10,
            abs_tol=tolerance,
        ):
            raise ValueError("CDWF released history did not spend the round budget")
    return released


def current_cdwf_rho_by_block(
    budget: CDWFBudgetPlan,
    streams: Sequence[CDWFReleasedStream],
) -> dict[str, float]:
    result = dict(budget.base_rho_by_block)
    for stream in tuple(streams)[1:]:
        for observation in stream.observations:
            result[observation.block_name] += observation.rho
    return result


def combine_cdwf_released_history(
    strategy: HierarchicalPairStrategy,
    public_total: int,
    budget: CDWFBudgetPlan,
    streams: Sequence[CDWFReleasedStream],
) -> HierarchicalInteractionTranscript:
    released = _validate_released_history(strategy, budget, streams)
    base = _base_hierarchical_transcript(
        strategy,
        int(public_total),
        budget,
        released[0],
    )
    actions = [
        InteractionActionRelease(
            pair=(int(observation.scope[0]), int(observation.scope[1])),
            noisy_coefficients=observation.noisy_coefficients,
            coefficient_variance=observation.coefficient_variance,
            rho=observation.rho,
            sensitivity_l2=observation.sensitivity_l2,
        )
        for stream in released[1:]
        for observation in stream.observations
    ]
    current_spend = math.fsum(stream.rho_spent for stream in released)
    return combine_coverage_refinements(
        base,
        actions,
        rho_total=current_spend,
    )


def ccf_pair_streams_for_history(
    strategy: HierarchicalPairStrategy,
    public_total: int,
    budget: CDWFBudgetPlan,
    streams: Sequence[CDWFReleasedStream],
) -> dict[tuple[int, int], tuple[CCFInteractionStream, ...]]:
    released = _validate_released_history(strategy, budget, streams)
    candidate_count = len(strategy.pairs)
    if candidate_count <= 0:
        raise ValueError("Sequential CCF requires at least one candidate pair")
    result: dict[tuple[int, int], list[CCFInteractionStream]] = {
        pair: [] for pair in strategy.pairs
    }
    n_squared = float(public_total) ** 2
    for stream in released:
        alpha = 0.025 if stream.stage == "base" else 0.025 / budget.rounds
        for observation in stream.observations:
            if observation.kind != "pair_interaction":
                continue
            pair = (int(observation.scope[0]), int(observation.scope[1]))
            rank = int(observation.noisy_coefficients.size)
            result[pair].append(
                CCFInteractionStream(
                    name=stream.name,
                    released_interaction_rate=(
                        observation.noisy_coefficients / int(public_total)
                    ),
                    interaction_covariance_rate=(
                        observation.coefficient_variance / n_squared
                    ),
                    confidence_radius=float(
                        chi2.ppf(1.0 - alpha / candidate_count, rank)
                    ),
                    alpha_struct=alpha,
                    round_index=stream.round_index,
                )
            )
    return {pair: tuple(local) for pair, local in result.items()}


@dataclass(frozen=True)
class CDWFSequentialTranscript:
    strategy: HierarchicalPairStrategy
    public_total: int
    budget: CDWFBudgetPlan
    streams: tuple[CDWFReleasedStream, ...]
    method: str = CDWF_TRANSCRIPT_METHOD

    def __post_init__(self) -> None:
        streams = tuple(self.streams)
        if self.method != CDWF_TRANSCRIPT_METHOD:
            raise ValueError(f"Unsupported CDWF transcript method {self.method!r}")
        if int(self.public_total) <= 0:
            raise ValueError("CDWF public_total must be positive")
        if len(streams) != self.budget.rounds + 1 or streams[0].stage != "base":
            raise ValueError("CDWF transcript must contain one base and every refinement round")
        if [stream.round_index for stream in streams[1:]] != list(
            range(self.budget.rounds)
        ):
            raise ValueError("CDWF refinement streams must be contiguous and ordered")
        expected = {block.name for block in self.strategy.blocks}
        if set(streams[0].by_name()) != expected:
            raise ValueError("CDWF base stream must cover every strategy block")
        tolerance = 1.0e-12 * max(1.0, self.budget.rho_total)
        for name, observation in streams[0].by_name().items():
            block = self.strategy.block(name)
            if (
                observation.kind != block.kind
                or observation.scope != block.scope
                or observation.noisy_coefficients.shape != block.coefficient_shape
                or not math.isclose(
                    observation.rho,
                    self.budget.base_rho_by_block[name],
                    rel_tol=1.0e-11,
                    abs_tol=tolerance,
                )
            ):
                raise ValueError(f"CDWF base observation mismatch for {name}")
        eligible = set(self.budget.eligible_interaction_blocks)
        for stream in streams[1:]:
            if not set(stream.by_name()) <= eligible:
                raise ValueError("CDWF refinement stream contains an ineligible block")
            if not math.isclose(
                stream.rho_spent,
                self.budget.rho_refinement_per_round,
                rel_tol=1.0e-10,
                abs_tol=tolerance,
            ):
                raise ValueError("CDWF refinement stream does not spend its round budget")
        final_rho = self.final_rho_by_block()
        if not math.isclose(
            math.fsum(final_rho.values()),
            self.budget.rho_total,
            rel_tol=1.0e-11,
            abs_tol=tolerance,
        ):
            raise ValueError("CDWF streams do not close the control rho ledger")
        for name in self.budget.frozen_blocks:
            if not math.isclose(
                final_rho[name],
                self.budget.control_rho_by_block[name],
                rel_tol=1.0e-11,
                abs_tol=tolerance,
            ):
                raise ValueError("CDWF changed a frozen block")
        for name in self.budget.eligible_interaction_blocks:
            gamma = final_rho[name] / self.budget.control_rho_by_block[name]
            if not self.budget.base_fraction - 1.0e-11 <= gamma <= self.budget.maximum_gamma + 1.0e-11:
                raise ValueError("CDWF final interaction gamma is outside its public bounds")
        object.__setattr__(self, "streams", streams)

    def final_rho_by_block(self) -> dict[str, float]:
        return current_cdwf_rho_by_block(self.budget, self.streams)

    def final_gamma_by_block(self) -> dict[str, float]:
        final = self.final_rho_by_block()
        return {
            name: final[name] / self.budget.control_rho_by_block[name]
            for name in self.budget.eligible_interaction_blocks
        }

    def combined_transcript(self) -> HierarchicalInteractionTranscript:
        return combine_cdwf_released_history(
            self.strategy,
            self.public_total,
            self.budget,
            self.streams,
        )

    def streamwise_confidence(self) -> StreamwiseRCEConfidenceSet:
        slices = coefficient_block_slices(self.strategy)
        combined = self.combined_transcript()
        combined_target = np.concatenate(
            [
                np.asarray(combined.noisy_components[block.name], dtype=np.float64).reshape(-1)
                for block in self.strategy.blocks
            ]
        )
        base_by_name = self.streams[0].by_name()
        base_target = np.concatenate(
            [base_by_name[block.name].noisy_coefficients.reshape(-1) for block in self.strategy.blocks]
        )
        base_variances = np.concatenate(
            [
                np.full(block.dimension, base_by_name[block.name].coefficient_variance)
                for block in self.strategy.blocks
            ]
        )
        refinement_targets: list[np.ndarray] = []
        refinement_variances: list[np.ndarray] = []
        refinement_indices: list[np.ndarray] = []
        for stream in self.streams[1:]:
            observations = stream.by_name()
            ordered = [block for block in self.strategy.blocks if block.name in observations]
            refinement_targets.append(
                np.concatenate([observations[block.name].noisy_coefficients.reshape(-1) for block in ordered])
            )
            refinement_variances.append(
                np.concatenate(
                    [
                        np.full(block.dimension, observations[block.name].coefficient_variance)
                        for block in ordered
                    ]
                )
            )
            refinement_indices.append(
                np.concatenate(
                    [
                        np.arange(slices[block.name].start, slices[block.name].stop, dtype=np.int64)
                        for block in ordered
                    ]
                )
            )
        return build_cdwf_streamwise_confidence(
            combined_target=combined_target,
            base_target=base_target,
            base_variances=base_variances,
            refinement_targets=tuple(refinement_targets),
            refinement_variances=tuple(refinement_variances),
            refinement_indices=tuple(refinement_indices),
        )

    def ccf_pair_streams(self) -> dict[tuple[int, int], tuple[CCFInteractionStream, ...]]:
        return ccf_pair_streams_for_history(
            self.strategy,
            self.public_total,
            self.budget,
            self.streams,
        )

    def privacy_ledger(self, *, delta: float) -> dict[str, Any]:
        entries = []
        for stream in self.streams:
            for observation in stream.observations:
                entries.append(
                    {
                        "label": f"{stream.name}:{observation.block_name}",
                        "mechanism": "gaussian_vector",
                        "rho": observation.rho,
                        "public_metadata": {
                            "adjacency": "add_remove",
                            "stage": stream.stage,
                            "round_index": stream.round_index,
                            "block_name": observation.block_name,
                            "kind": observation.kind,
                            "scope": list(observation.scope),
                            "sensitivity_l2": observation.sensitivity_l2,
                            "selection_rho": 0.0,
                            "allocation_depends_on": (
                                "public_control_allocation"
                                if stream.stage == "base"
                                else "released_transcript_only"
                            ),
                        },
                    }
                )
        spent = math.fsum(float(entry["rho"]) for entry in entries)
        if not math.isclose(spent, self.budget.rho_total, rel_tol=1.0e-11, abs_tol=1.0e-14):
            raise RuntimeError("CDWF privacy ledger does not equal the control spend")
        return {
            "accounting": "zcdp_actual_spend_v1",
            "accounting_theorem": "adaptive_gaussian_zcdp_composition_v1",
            "accounting_version": "c3_cdwf_streamwise_v1",
            "adjacency": "add_remove",
            "rho_limit": self.budget.rho_total,
            "rho_spent": spent,
            "rho_remaining": 0.0,
            "delta": float(delta),
            "epsilon_from_actual_spend": zcdp_epsilon(spent, float(delta)),
            "selection_rho": 0.0,
            "entries": entries,
            "postprocessing": [
                "released_dual_water_filling",
                "precision_combination",
                "sequential_ccf",
                "streamwise_rce",
                "qdte_generation",
            ],
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "adjacency": "add_remove",
            "selection_rho": 0.0,
            "public_total": self.public_total,
            "cardinalities": list(self.strategy.cardinalities),
            "pairs": [list(pair) for pair in self.strategy.pairs],
            "budget": self.budget.to_public_dict(),
            "streams": [stream.to_public_dict() for stream in self.streams],
            "final_rho_by_block": self.final_rho_by_block(),
            "final_gamma_by_block": self.final_gamma_by_block(),
            "streamwise_confidence": self.streamwise_confidence().to_public_dict(),
        }

    @classmethod
    def from_public_dict(cls, data: Mapping[str, Any]) -> CDWFSequentialTranscript:
        from qdte.measurement.factorization import compile_hierarchical_pair_strategy

        strategy = compile_hierarchical_pair_strategy(
            tuple(int(value) for value in data["cardinalities"]),
            tuple(tuple(int(value) for value in pair) for pair in data["pairs"]),
        )
        result = cls(
            method=str(data["method"]),
            strategy=strategy,
            public_total=int(data["public_total"]),
            budget=_plan_from_public_dict(data["budget"]),
            streams=tuple(
                CDWFReleasedStream.from_public_dict(stream) for stream in data["streams"]
            ),
        )
        expected = result.streamwise_confidence().stream_hash
        observed = str(data["streamwise_confidence"]["stream_hash"])
        if expected != observed:
            raise ValueError("Serialized CDWF streamwise confidence hash changed")
        return result


def _base_hierarchical_transcript(
    strategy: HierarchicalPairStrategy,
    public_total: int,
    budget: CDWFBudgetPlan,
    stream: CDWFReleasedStream,
) -> HierarchicalInteractionTranscript:
    by_name = stream.by_name()
    return HierarchicalInteractionTranscript(
        strategy=strategy,
        public_total=int(public_total),
        noisy_components={name: observation.noisy_coefficients.copy() for name, observation in by_name.items()},
        component_variances={name: observation.coefficient_variance for name, observation in by_name.items()},
        rho_by_block={name: observation.rho for name, observation in by_name.items()},
        rho_total=math.fsum(budget.base_rho_by_block.values()),
        rho_spent=stream.rho_spent,
        allocation_mode="workload_optimal",
        adjacency="add_remove",
    )


def released_stream_from_base(
    transcript: HierarchicalInteractionTranscript,
) -> CDWFReleasedStream:
    observations = []
    for block in transcript.strategy.blocks:
        observations.append(
            CDWFBlockObservation(
                block_name=block.name,
                kind=block.kind,
                scope=block.scope,
                noisy_coefficients=transcript.noisy_components[block.name],
                coefficient_variance=transcript.component_variances[block.name],
                rho=transcript.rho_by_block[block.name],
                sensitivity_l2=block.sensitivity_l2,
            )
        )
    return CDWFReleasedStream(
        name="base",
        stage="base",
        round_index=-1,
        observations=tuple(observations),
    )


def measure_cdwf_base(
    rows: np.ndarray,
    strategy: HierarchicalPairStrategy,
    plan: CDWFBudgetPlan,
    *,
    public_total: int,
    rng: np.random.Generator,
) -> CDWFReleasedStream:
    transcript = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=int(public_total),
        rho_total=math.fsum(plan.base_rho_by_block.values()),
        rng=rng,
        allocation_mode="workload_optimal",
        rho_by_block_override=plan.base_rho_by_block,
    )
    return released_stream_from_base(transcript)


def measure_cdwf_refinement_stream(
    rows: np.ndarray,
    strategy: HierarchicalPairStrategy,
    allocation: Mapping[str, float],
    *,
    round_index: int,
    rng_by_block: Mapping[str, np.random.Generator],
) -> CDWFReleasedStream:
    positive = {
        str(name): float(value)
        for name, value in allocation.items()
        if float(value) > 0.0
    }
    if not positive:
        raise ValueError("CDWF refinement stream must measure at least one block")
    observations = []
    for block in strategy.blocks:
        if block.name not in positive:
            continue
        if block.kind != "pair_interaction" or block.name not in rng_by_block:
            raise ValueError("CDWF refinement allocation contains an invalid block or RNG")
        action = measure_interaction_action(
            rows,
            block.scope,
            strategy.cardinalities,
            rho=positive[block.name],
            rng=rng_by_block[block.name],
        )
        observations.append(
            CDWFBlockObservation(
                block_name=block.name,
                kind=block.kind,
                scope=block.scope,
                noisy_coefficients=action.noisy_coefficients,
                coefficient_variance=action.coefficient_variance,
                rho=action.rho,
                sensitivity_l2=action.sensitivity_l2,
            )
        )
    return CDWFReleasedStream(
        name=f"refinement_{int(round_index):02d}",
        stage="refinement",
        round_index=int(round_index),
        observations=tuple(observations),
    )


def cdwf_transcript_to_measurements(
    transcript: CDWFSequentialTranscript,
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    *,
    delta: float,
) -> Measurements:
    combined = transcript.combined_transcript()
    measurements = interaction_transcript_to_diagonal_measurements(
        combined,
        qcat,
        workload_groups,
        delta=float(delta),
        target_projection="raw_reconstruction",
    )
    measurements.sequential_transcript = transcript.to_public_dict()
    measurements.privacy_ledger = transcript.privacy_ledger(delta=float(delta))
    diagnostics = dict(measurements.projection_diagnostics or {})
    diagnostics["c3_cdwf"] = {
        "method": CDWF_METHOD,
        "selection_rho": 0.0,
        "adaptive_safe_confidence": "streamwise_intersection",
        "naive_fixed_design_combined_certificate_used": False,
        "final_gamma_by_block": transcript.final_gamma_by_block(),
    }
    measurements.projection_diagnostics = diagnostics
    return measurements


__all__ = [
    "CDWFBlockObservation",
    "CDWFReleasedStream",
    "CDWFSequentialTranscript",
    "CDWF_TRANSCRIPT_METHOD",
    "ccf_pair_streams_for_history",
    "cdwf_transcript_to_measurements",
    "combine_cdwf_released_history",
    "current_cdwf_rho_by_block",
    "measure_cdwf_base",
    "measure_cdwf_refinement_stream",
    "released_stream_from_base",
]
