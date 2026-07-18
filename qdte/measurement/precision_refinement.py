from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qdte.measurement.factorization import HierarchicalPairStrategy


@dataclass(frozen=True)
class CoverageRefinementBudget:
    rho_total: float
    rounds: int
    beta: float
    beta_per_round: float
    coverage_std_inflation_cap: float
    base_fraction: float
    adaptive_fraction: float
    selection_proxy: float
    refinement_proxy: float
    raw_selection_fraction: float
    selection_fraction: float
    rho_base: float
    rho_selection: float
    rho_refinement: float
    rho_selection_per_round: float
    rho_refinement_per_round: float
    candidate_count: int
    max_action_dimension: int

    def to_public_dict(self) -> dict[str, float | int]:
        return {
            "rho_total": self.rho_total,
            "rounds": self.rounds,
            "beta": self.beta,
            "beta_per_round": self.beta_per_round,
            "coverage_std_inflation_cap": self.coverage_std_inflation_cap,
            "base_fraction": self.base_fraction,
            "adaptive_fraction": self.adaptive_fraction,
            "selection_proxy": self.selection_proxy,
            "refinement_proxy": self.refinement_proxy,
            "raw_selection_fraction": self.raw_selection_fraction,
            "selection_fraction": self.selection_fraction,
            "rho_base": self.rho_base,
            "rho_selection": self.rho_selection,
            "rho_refinement": self.rho_refinement,
            "rho_selection_per_round": self.rho_selection_per_round,
            "rho_refinement_per_round": self.rho_refinement_per_round,
            "candidate_count": self.candidate_count,
            "max_action_dimension": self.max_action_dimension,
        }


def coverage_rounds_for_epsilon(epsilon: float) -> int:
    value = float(epsilon)
    for declared, rounds in (
        (0.1, 4),
        (0.3, 6),
        (1.0, 8),
        (3.0, 12),
        (10.0, 12),
    ):
        if math.isclose(value, declared, rel_tol=0.0, abs_tol=1.0e-12):
            return rounds
    raise ValueError("epsilon must be one of {0.1, 0.3, 1, 3, 10}")


def derive_coverage_refinement_budget(
    strategy: HierarchicalPairStrategy,
    *,
    rho_total: float,
    rounds: int,
    beta: float = 0.05,
    coverage_std_inflation_cap: float = 1.20,
    min_selection_fraction: float = 0.15,
    max_selection_fraction: float = 0.40,
) -> CoverageRefinementBudget:
    """Derive the WP8a split using public strategy metadata only."""
    total = float(rho_total)
    num_rounds = int(rounds)
    failure_probability = float(beta)
    inflation_cap = float(coverage_std_inflation_cap)
    lower = float(min_selection_fraction)
    upper = float(max_selection_fraction)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("rho_total must be finite and positive")
    if num_rounds <= 0:
        raise ValueError("rounds must be positive")
    if not math.isfinite(failure_probability) or not 0.0 < failure_probability < 1.0:
        raise ValueError("beta must lie strictly between zero and one")
    if not math.isfinite(inflation_cap) or inflation_cap <= 1.0:
        raise ValueError("coverage_std_inflation_cap must exceed one")
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError("selection fraction bounds must satisfy 0 < min <= max < 1")

    pair_blocks = tuple(
        block for block in strategy.blocks if block.kind == "pair_interaction"
    )
    if not pair_blocks:
        raise ValueError("coverage refinement requires at least one pair action")

    beta_per_round = failure_probability / float(num_rounds)
    log_tail = math.log(1.0 / beta_per_round)
    selection_proxy = math.log(float(len(pair_blocks)) / beta_per_round)
    refinement_terms = []
    for block in pair_blocks:
        dimension = float(block.dimension)
        chi_radius = math.sqrt(
            dimension
            + 2.0 * math.sqrt(dimension * log_tail)
            + 2.0 * log_tail
        )
        refinement_terms.append(block.sensitivity_l2 * chi_radius)
    refinement_proxy = max(refinement_terms)

    selection_power = selection_proxy ** (2.0 / 3.0)
    refinement_power = refinement_proxy ** (2.0 / 3.0)
    raw_selection_fraction = selection_power / (
        selection_power + refinement_power
    )
    selection_fraction = float(
        np.clip(raw_selection_fraction, lower, upper)
    )

    base_fraction = 1.0 / (inflation_cap * inflation_cap)
    adaptive_fraction = 1.0 - base_fraction
    rho_base = base_fraction * total
    rho_selection = adaptive_fraction * selection_fraction * total
    rho_refinement = total - rho_base - rho_selection
    if min(rho_base, rho_selection, rho_refinement) <= 0.0:
        raise RuntimeError("derived coverage-refinement budget is not positive")

    return CoverageRefinementBudget(
        rho_total=total,
        rounds=num_rounds,
        beta=failure_probability,
        beta_per_round=beta_per_round,
        coverage_std_inflation_cap=inflation_cap,
        base_fraction=base_fraction,
        adaptive_fraction=adaptive_fraction,
        selection_proxy=selection_proxy,
        refinement_proxy=refinement_proxy,
        raw_selection_fraction=raw_selection_fraction,
        selection_fraction=selection_fraction,
        rho_base=rho_base,
        rho_selection=rho_selection,
        rho_refinement=rho_refinement,
        rho_selection_per_round=rho_selection / float(num_rounds),
        rho_refinement_per_round=rho_refinement / float(num_rounds),
        candidate_count=len(pair_blocks),
        max_action_dimension=max(block.dimension for block in pair_blocks),
    )
