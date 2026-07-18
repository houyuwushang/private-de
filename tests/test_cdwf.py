import math

import numpy as np

from qdte.measurement.cdwf import (
    CDWF_BASE_FRACTION,
    CDWF_REFINEMENT_FRACTION,
    derive_cdwf_budget_plan,
    pressure_concentration,
    solve_cdwf_water_filling,
    uniform_cdwf_allocation,
)
from qdte.measurement.factorization import compile_hierarchical_pair_strategy


def _plan():
    strategy = compile_hierarchical_pair_strategy(
        (2, 3, 2, 2),
        ((0, 1), (0, 2), (2, 3)),
    )
    control = {
        block.name: float(index + 1) / 100.0
        for index, block in enumerate(strategy.blocks)
    }
    return strategy, control, derive_cdwf_budget_plan(
        strategy,
        control,
        epsilon=0.1,
    )


def test_cdwf_budget_splits_only_interactions_and_closes_rho() -> None:
    strategy, control, plan = _plan()
    assert plan.rounds == 4
    assert math.isclose(
        CDWF_BASE_FRACTION + CDWF_REFINEMENT_FRACTION,
        1.0,
        abs_tol=1.0e-15,
    )
    for block in strategy.blocks:
        if block.kind == "oneway_contrast":
            assert plan.base_rho_by_block[block.name] == control[block.name]
        else:
            assert math.isclose(
                plan.base_rho_by_block[block.name],
                CDWF_BASE_FRACTION * control[block.name],
            )
    assert math.isclose(
        math.fsum(plan.base_rho_by_block.values()) + plan.rho_refinement,
        math.fsum(control.values()),
        rel_tol=1.0e-12,
    )


def test_uniform_refinement_restores_every_interaction_control_precision() -> None:
    _, _, plan = _plan()
    per_round = uniform_cdwf_allocation(plan)
    for name in plan.eligible_interaction_blocks:
        final = plan.base_rho_by_block[name] + plan.rounds * per_round[name]
        assert math.isclose(
            final,
            plan.control_rho_by_block[name],
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )


def test_dual_water_filling_closes_budget_and_beats_uniform_surrogate() -> None:
    _, _, plan = _plan()
    current = dict(plan.base_rho_by_block)
    pressures = {
        name: value
        for name, value in zip(
            plan.eligible_interaction_blocks,
            (100.0, 1.0, 0.1),
            strict=True,
        )
    }
    result = solve_cdwf_water_filling(plan, current, pressures)
    assert math.isclose(
        math.fsum(result.allocation.values()),
        plan.rho_refinement_per_round,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert result.objective >= result.uniform_objective - 1.0e-12
    assert result.allocation[plan.eligible_interaction_blocks[0]] > (
        uniform_cdwf_allocation(plan)[plan.eligible_interaction_blocks[0]]
    )
    assert result.maximum_cap_violation <= 1.0e-12


def test_zero_pressure_uses_the_unique_public_uniform_tie_break() -> None:
    _, _, plan = _plan()
    current = dict(plan.base_rho_by_block)
    pressures = {name: 0.0 for name in plan.eligible_interaction_blocks}
    result = solve_cdwf_water_filling(plan, current, pressures)
    assert result.all_pressures_zero
    assert all(
        math.isclose(
            result.allocation[name],
            uniform_cdwf_allocation(plan)[name],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for name in plan.eligible_interaction_blocks
    )


def test_pressure_concentration_allows_fractional_boundary_block() -> None:
    pressures = {"a": 8.0, "b": 2.0}
    rho = {"a": 1.0, "b": 1.0}
    assert math.isclose(
        pressure_concentration(pressures, rho, mass_fraction=0.25),
        0.4,
        abs_tol=1.0e-12,
    )
