import numpy as np

from qdte.rce.cdwf_dual import (
    compute_cdwf_block_pressures,
    extract_certified_restricted_dual,
)
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.relaxed import solve_restricted_mixture_rce


def _active_restricted_problem():
    confidence = RCEConfidenceSet.from_diagonal_variances(
        np.asarray([1.0e-2, 4.0e-2])
    )
    result = solve_restricted_mixture_rce(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[-10.0, -5.0], [10.0, 5.0]]),
        np.log(np.asarray([0.9, 0.1])),
        confidence,
        component_names=("left", "right"),
    )
    return confidence, result


def test_restricted_solver_exposes_certified_minimum_norm_dual() -> None:
    confidence, result = _active_restricted_problem()
    dual = extract_certified_restricted_dual(result, confidence)
    assert dual.eligible, dual.failure_reasons
    assert dual.ellipsoid_weight >= 0.0
    assert dual.certificate["stationarity_infinity_norm"] <= 1.0e-7
    assert dual.certificate["relative_primal_dual_gap"] <= 1.0e-6
    assert result.restricted_dual_certificate["minimum_norm_dual"]["success"]


def test_cdwf_pressure_formula_matches_constraint_autodiff() -> None:
    confidence, result = _active_restricted_problem()
    dual = extract_certified_restricted_dual(result, confidence)
    pressure = compute_cdwf_block_pressures(
        residual=result.residual,
        confidence=confidence,
        dual=dual,
        block_slices={"pair_a": slice(0, 1), "pair_b": slice(1, 2)},
        rho_by_block={"pair_a": 1.0, "pair_b": 2.0},
    )
    assert pressure.autodiff_audit_passed
    assert pressure.maximum_autodiff_error <= 1.0e-9
    assert all(value >= 0.0 for value in pressure.pressure_by_block.values())
