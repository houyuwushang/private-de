from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from qdte.measurement.factorization import HierarchicalPairStrategy


CDWF_METHOD = "confidence_dual_water_filling_v1"
CDWF_BASE_FRACTION = 25.0 / 36.0
CDWF_REFINEMENT_FRACTION = 11.0 / 36.0
CDWF_MAX_GAMMA = 4.0


def cdwf_rounds_for_epsilon(epsilon: float) -> int:
    value = float(epsilon)
    for declared, rounds in ((0.1, 4), (0.3, 6)):
        if math.isclose(value, declared, rel_tol=0.0, abs_tol=1.0e-12):
            return rounds
    raise ValueError("C3-CDWF epsilon must be one of {0.1, 0.3}")


def coefficient_block_slices(
    strategy: HierarchicalPairStrategy,
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for block in strategy.blocks:
        end = offset + int(block.dimension)
        result[block.name] = slice(offset, end)
        offset = end
    return result


@dataclass(frozen=True)
class CDWFBudgetPlan:
    control_rho_by_block: dict[str, float]
    base_rho_by_block: dict[str, float]
    eligible_interaction_blocks: tuple[str, ...]
    frozen_blocks: tuple[str, ...]
    rounds: int
    rho_total: float
    rho_interaction_control: float
    rho_refinement: float
    rho_refinement_per_round: float
    base_fraction: float = CDWF_BASE_FRACTION
    refinement_fraction: float = CDWF_REFINEMENT_FRACTION
    maximum_gamma: float = CDWF_MAX_GAMMA
    method: str = CDWF_METHOD

    def __post_init__(self) -> None:
        if self.method != CDWF_METHOD:
            raise ValueError(f"Unsupported CDWF method {self.method!r}")
        control = {str(name): float(value) for name, value in self.control_rho_by_block.items()}
        base = {str(name): float(value) for name, value in self.base_rho_by_block.items()}
        if not control or set(control) != set(base):
            raise ValueError("CDWF control and base allocations must cover the same blocks")
        if any(not math.isfinite(value) or value <= 0.0 for value in control.values()):
            raise ValueError("CDWF control rho values must be finite and positive")
        if any(not math.isfinite(value) or value <= 0.0 for value in base.values()):
            raise ValueError("CDWF base rho values must be finite and positive")
        eligible = tuple(str(name) for name in self.eligible_interaction_blocks)
        frozen = tuple(str(name) for name in self.frozen_blocks)
        if len(set(eligible)) != len(eligible) or len(set(frozen)) != len(frozen):
            raise ValueError("CDWF block lists must not contain duplicates")
        if set(eligible) & set(frozen) or set(eligible) | set(frozen) != set(control):
            raise ValueError("CDWF eligible and frozen blocks must partition the strategy")
        if not eligible:
            raise ValueError("CDWF requires at least one eligible interaction block")
        if int(self.rounds) <= 0:
            raise ValueError("CDWF rounds must be positive")
        if not math.isclose(
            float(self.base_fraction) + float(self.refinement_fraction),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("CDWF base and refinement fractions must sum to one")
        interaction_control = math.fsum(control[name] for name in eligible)
        refinement = float(self.refinement_fraction) * interaction_control
        total = math.fsum(control.values())
        tolerance = 1.0e-12 * max(1.0, total)
        checks = (
            abs(float(self.rho_total) - total),
            abs(float(self.rho_interaction_control) - interaction_control),
            abs(float(self.rho_refinement) - refinement),
            abs(float(self.rho_refinement_per_round) - refinement / int(self.rounds)),
        )
        if max(checks) > tolerance:
            raise ValueError("CDWF budget totals are inconsistent")
        for name in frozen:
            if not math.isclose(base[name], control[name], rel_tol=0.0, abs_tol=tolerance):
                raise ValueError("CDWF must preserve every frozen block at control precision")
        for name in eligible:
            expected = float(self.base_fraction) * control[name]
            if not math.isclose(base[name], expected, rel_tol=1.0e-12, abs_tol=tolerance):
                raise ValueError("CDWF must split only eligible interaction blocks")
        base_plus_refinement = math.fsum(base.values()) + refinement
        if not math.isclose(base_plus_refinement, total, rel_tol=1.0e-12, abs_tol=tolerance):
            raise ValueError("CDWF base and refinement spend must equal control spend")
        object.__setattr__(self, "control_rho_by_block", control)
        object.__setattr__(self, "base_rho_by_block", base)
        object.__setattr__(self, "eligible_interaction_blocks", eligible)
        object.__setattr__(self, "frozen_blocks", frozen)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "base_fraction": self.base_fraction,
            "refinement_fraction": self.refinement_fraction,
            "maximum_gamma": self.maximum_gamma,
            "rounds": self.rounds,
            "rho_total": self.rho_total,
            "rho_interaction_control": self.rho_interaction_control,
            "rho_refinement": self.rho_refinement,
            "rho_refinement_per_round": self.rho_refinement_per_round,
            "eligible_interaction_blocks": list(self.eligible_interaction_blocks),
            "frozen_blocks": list(self.frozen_blocks),
            "control_rho_by_block": dict(self.control_rho_by_block),
            "base_rho_by_block": dict(self.base_rho_by_block),
        }


def derive_cdwf_budget_plan(
    strategy: HierarchicalPairStrategy,
    control_rho_by_block: Mapping[str, float],
    *,
    epsilon: float,
    eligible_interaction_blocks: Sequence[str] | None = None,
) -> CDWFBudgetPlan:
    expected = {block.name for block in strategy.blocks}
    control = {str(name): float(value) for name, value in control_rho_by_block.items()}
    if set(control) != expected:
        raise ValueError("CDWF control allocation must cover every strategy block exactly")
    pair_names = {
        block.name for block in strategy.blocks if block.kind == "pair_interaction"
    }
    if eligible_interaction_blocks is None:
        eligible = tuple(
            block.name for block in strategy.blocks if block.kind == "pair_interaction"
        )
    else:
        eligible = tuple(str(name) for name in eligible_interaction_blocks)
        if len(set(eligible)) != len(eligible) or not set(eligible) <= pair_names:
            raise ValueError("CDWF eligible blocks must be distinct pair-interaction blocks")
    frozen = tuple(block.name for block in strategy.blocks if block.name not in set(eligible))
    base = dict(control)
    for name in eligible:
        base[name] = CDWF_BASE_FRACTION * control[name]
    interaction_control = math.fsum(control[name] for name in eligible)
    refinement = CDWF_REFINEMENT_FRACTION * interaction_control
    rounds = cdwf_rounds_for_epsilon(epsilon)
    return CDWFBudgetPlan(
        control_rho_by_block=control,
        base_rho_by_block=base,
        eligible_interaction_blocks=eligible,
        frozen_blocks=frozen,
        rounds=rounds,
        rho_total=math.fsum(control.values()),
        rho_interaction_control=interaction_control,
        rho_refinement=refinement,
        rho_refinement_per_round=refinement / rounds,
    )


@dataclass(frozen=True)
class CDWFAllocationResult:
    allocation: dict[str, float]
    objective: float
    uniform_objective: float
    eta: float | None
    all_pressures_zero: bool
    budget_residual: float
    maximum_cap_violation: float

    def to_public_dict(self) -> dict[str, object]:
        return {
            "allocation": dict(self.allocation),
            "objective": self.objective,
            "uniform_objective": self.uniform_objective,
            "eta": self.eta,
            "all_pressures_zero": self.all_pressures_zero,
            "budget_residual": self.budget_residual,
            "maximum_cap_violation": self.maximum_cap_violation,
        }


def _weighted_uniform_projection(
    names: tuple[str, ...],
    control: Mapping[str, float],
    capacities: Mapping[str, float],
    amount: float,
) -> dict[str, float]:
    if amount <= 0.0:
        return {name: 0.0 for name in names}
    total_capacity = math.fsum(capacities[name] for name in names)
    tolerance = 1.0e-13 * max(1.0, amount, total_capacity)
    if amount > total_capacity + tolerance:
        raise ValueError("CDWF tie-break allocation exceeds available capacity")
    weights = np.asarray([control[name] for name in names], dtype=np.float64)
    caps = np.asarray([capacities[name] for name in names], dtype=np.float64)
    if np.any(weights <= 0.0) or np.any(caps < 0.0):
        raise ValueError("CDWF tie-break weights and capacities are invalid")
    lower = 0.0
    upper = amount / float(np.min(weights)) + 1.0
    for _ in range(160):
        scale = 0.5 * (lower + upper)
        values = np.minimum(caps, scale * weights)
        if float(np.sum(values)) < amount:
            lower = scale
        else:
            upper = scale
    values = np.minimum(caps, upper * weights)
    residual = amount - float(np.sum(values))
    if abs(residual) > tolerance:
        free = np.flatnonzero(values < caps - tolerance)
        if residual > 0.0 and len(free):
            room = caps[free] - values[free]
            increment = residual * weights[free] / float(np.sum(weights[free]))
            increment = np.minimum(increment, room)
            values[free] += increment
    residual = amount - float(np.sum(values))
    if abs(residual) > tolerance:
        for index in range(len(values) - 1, -1, -1):
            candidate = values[index] + residual
            if -tolerance <= candidate <= caps[index] + tolerance:
                values[index] = min(caps[index], max(0.0, candidate))
                break
    if not math.isclose(float(np.sum(values)), amount, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError("CDWF weighted-uniform tie-break did not close its budget")
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def uniform_cdwf_allocation(
    plan: CDWFBudgetPlan,
) -> dict[str, float]:
    scale = plan.rho_refinement_per_round / plan.rho_interaction_control
    allocation = {
        name: scale * plan.control_rho_by_block[name]
        for name in plan.eligible_interaction_blocks
    }
    last = plan.eligible_interaction_blocks[-1]
    allocation[last] += plan.rho_refinement_per_round - math.fsum(allocation.values())
    return allocation


def _water_filling_objective(
    names: tuple[str, ...],
    allocation: Mapping[str, float],
    pressures: Mapping[str, float],
    current: Mapping[str, float],
) -> float:
    return float(
        math.fsum(
            float(pressures[name])
            * math.log1p(float(allocation[name]) / float(current[name]))
            for name in names
        )
    )


def solve_cdwf_water_filling(
    plan: CDWFBudgetPlan,
    current_rho_by_block: Mapping[str, float],
    pressures: Mapping[str, float],
) -> CDWFAllocationResult:
    names = plan.eligible_interaction_blocks
    if set(current_rho_by_block) != set(plan.control_rho_by_block):
        raise ValueError("CDWF current allocation must cover every control block")
    if set(pressures) != set(names):
        raise ValueError("CDWF pressure must cover every eligible interaction block")
    current = {name: float(current_rho_by_block[name]) for name in names}
    pressure = {name: float(pressures[name]) for name in names}
    if any(not math.isfinite(value) or value <= 0.0 for value in current.values()):
        raise ValueError("CDWF current rho values must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in pressure.values()):
        raise ValueError("CDWF pressures must be finite and nonnegative")
    capacities = {
        name: max(
            0.0,
            plan.maximum_gamma * plan.control_rho_by_block[name] - current[name],
        )
        for name in names
    }
    amount = plan.rho_refinement_per_round
    tolerance = 1.0e-12 * max(1.0, amount)
    if math.fsum(capacities.values()) < amount - tolerance:
        raise RuntimeError("CDWF gamma caps cannot absorb the declared round budget")

    positive = tuple(name for name in names if pressure[name] > 0.0)
    allocation = {name: 0.0 for name in names}
    eta: float | None = None
    positive_capacity = math.fsum(capacities[name] for name in positive)
    positive_amount = min(amount, positive_capacity)
    if positive and positive_amount > tolerance:
        def allocated(candidate_eta: float) -> float:
            return math.fsum(
                min(
                    capacities[name],
                    max(0.0, pressure[name] / candidate_eta - current[name]),
                )
                for name in positive
            )

        lower = np.finfo(np.float64).tiny
        upper = max(pressure[name] / current[name] for name in positive)
        for _ in range(220):
            middle = math.sqrt(lower * upper) if lower > 0.0 else 0.5 * upper
            if allocated(middle) > positive_amount:
                lower = middle
            else:
                upper = middle
        eta = upper
        for name in positive:
            allocation[name] = min(
                capacities[name],
                max(0.0, pressure[name] / eta - current[name]),
            )
        positive_residual = positive_amount - math.fsum(
            allocation[name] for name in positive
        )
        if abs(positive_residual) > tolerance:
            free = tuple(
                name
                for name in positive
                if allocation[name] < capacities[name] - tolerance
            )
            if free:
                correction = _weighted_uniform_projection(
                    free,
                    plan.control_rho_by_block,
                    {name: capacities[name] - allocation[name] for name in free},
                    max(0.0, positive_residual),
                )
                for name in free:
                    allocation[name] += correction[name]

    remaining = amount - math.fsum(allocation.values())
    zero_names = tuple(name for name in names if pressure[name] == 0.0)
    if remaining > tolerance:
        candidates = zero_names or tuple(
            name for name in names if allocation[name] < capacities[name] - tolerance
        )
        tie = _weighted_uniform_projection(
            candidates,
            plan.control_rho_by_block,
            {name: capacities[name] - allocation[name] for name in candidates},
            remaining,
        )
        for name, value in tie.items():
            allocation[name] += value
    residual = amount - math.fsum(allocation.values())
    if abs(residual) > tolerance:
        for name in reversed(names):
            candidate = allocation[name] + residual
            if -tolerance <= candidate <= capacities[name] + tolerance:
                allocation[name] = min(capacities[name], max(0.0, candidate))
                break

    uniform = uniform_cdwf_allocation(plan)
    objective = _water_filling_objective(names, allocation, pressure, current)
    uniform_objective = _water_filling_objective(names, uniform, pressure, current)
    budget_residual = math.fsum(allocation.values()) - amount
    cap_violation = max(
        0.0,
        max(
            current[name]
            + allocation[name]
            - plan.maximum_gamma * plan.control_rho_by_block[name]
            for name in names
        ),
    )
    if abs(budget_residual) > tolerance or cap_violation > tolerance:
        raise RuntimeError("CDWF water-filling failed its budget or cap invariant")
    if objective + 1.0e-11 * max(1.0, abs(uniform_objective)) < uniform_objective:
        raise RuntimeError("CDWF water-filling objective is worse than public uniform")
    return CDWFAllocationResult(
        allocation=allocation,
        objective=objective,
        uniform_objective=uniform_objective,
        eta=eta,
        all_pressures_zero=not positive,
        budget_residual=budget_residual,
        maximum_cap_violation=cap_violation,
    )


def pressure_concentration(
    pressures: Mapping[str, float],
    control_rho_by_block: Mapping[str, float],
    *,
    mass_fraction: float,
) -> float:
    fraction = float(mass_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("mass_fraction must lie in [0, 1]")
    if set(pressures) != set(control_rho_by_block) or not pressures:
        raise ValueError("pressure and control-rho maps must have the same non-empty support")
    total_pressure = math.fsum(max(0.0, float(value)) for value in pressures.values())
    if total_pressure <= 0.0:
        return 0.0
    total_mass = math.fsum(float(value) for value in control_rho_by_block.values())
    capacity = fraction * total_mass
    ordered = sorted(
        pressures,
        key=lambda name: (
            -float(pressures[name]) / float(control_rho_by_block[name]),
            str(name),
        ),
    )
    selected_pressure = 0.0
    used = 0.0
    for name in ordered:
        mass = float(control_rho_by_block[name])
        if used >= capacity:
            break
        taken = min(mass, capacity - used)
        selected_pressure += float(pressures[name]) * taken / mass
        used += taken
    return float(selected_pressure / total_pressure)


__all__ = [
    "CDWF_BASE_FRACTION",
    "CDWF_MAX_GAMMA",
    "CDWF_METHOD",
    "CDWF_REFINEMENT_FRACTION",
    "CDWFAllocationResult",
    "CDWFBudgetPlan",
    "cdwf_rounds_for_epsilon",
    "coefficient_block_slices",
    "derive_cdwf_budget_plan",
    "pressure_concentration",
    "solve_cdwf_water_filling",
    "uniform_cdwf_allocation",
]
