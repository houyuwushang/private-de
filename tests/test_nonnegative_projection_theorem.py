from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from qdte.measurement.consistency import (
    _solve_box_qp_slsqp_reference,
    offline_query_space_projection_dominance_diagnostics,
    project_query_space_feasible_lsq,
    project_query_space_lsq,
)
from qdte.queries.types import OP_EQ, OP_LE, QueryBuilder


def _complete_oneway(cardinality: int):
    builder = QueryBuilder(max_terms=1)
    for value in range(cardinality):
        builder.add(
            [(0, OP_EQ, value, value, value)],
            f"x={value}",
            "oneway:0",
            "oneway",
        )
    return builder.build()


def _project_pair(
    noisy: np.ndarray,
    variances: np.ndarray,
    *,
    total: int,
):
    qcat = _complete_oneway(len(noisy))
    cards = np.asarray([len(noisy)], dtype=np.int32)
    equality = project_query_space_lsq(
        noisy,
        qcat,
        cards,
        total=total,
        variances=variances,
    )
    feasible = project_query_space_feasible_lsq(
        noisy,
        qcat,
        cards,
        total=total,
        variances=variances,
    )
    return qcat, cards, equality, feasible


def test_dense_reference_fallback_solves_degenerate_box_qp() -> None:
    result = _solve_box_qp_slsqp_reference(
        noisy=np.asarray([-1.0, 2.0], dtype=np.float64),
        variance=np.ones(2, dtype=np.float64),
        constraints=csr_matrix(np.asarray([[1.0, 1.0]], dtype=np.float64)),
        rhs=np.asarray([1.0], dtype=np.float64),
        upper=1.0,
        initial=np.asarray([0.5, 0.5], dtype=np.float64),
        feasibility_tolerance=1.0e-8,
        solver_ftol=1.0e-12,
        solver_max_iterations=100,
    )

    assert result["success"] is True
    assert result["solver"] == "slsqp_dense_reference_fallback"
    assert np.allclose(result["projected"], np.asarray([0.0, 1.0]), atol=1.0e-8)


def test_nonnegative_projection_strictly_dominates_when_equality_is_negative() -> None:
    noisy = np.asarray([-10.0, 25.0], dtype=np.float64)
    truth = np.asarray([7.0, 3.0], dtype=np.float64)
    variances = np.asarray([1.0, 9.0], dtype=np.float64)
    qcat, cards, equality, feasible = _project_pair(noisy, variances, total=10)

    diagnostics = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=truth,
        qcat=qcat,
        cardinalities=cards,
        total=10,
        variances=variances,
        equality_result=equality,
        feasible_result=feasible,
    )

    assert equality.projected.dtype == np.float64
    assert feasible.projected.dtype == np.float64
    assert diagnostics["theorem_qualified"] is True
    assert diagnostics["fingerprints_match"] is True
    assert diagnostics["equality_outside_K"] is True
    assert diagnostics["strict_dominance_observed"] is True
    assert diagnostics["dominance_slack"] >= -diagnostics["comparison_tolerance"]
    assert feasible.diagnostics["feasible_projection_certificate_passed"] is True
    assert feasible.diagnostics["certificate_objective_suboptimality_upper_bound"] >= 0.0


def test_nonnegative_projection_matches_equality_when_bounds_are_inactive() -> None:
    noisy = np.asarray([8.0, 10.0], dtype=np.float64)
    truth = np.asarray([7.0, 3.0], dtype=np.float64)
    variances = np.asarray([1.0, 9.0], dtype=np.float64)
    qcat, cards, equality, feasible = _project_pair(noisy, variances, total=10)

    diagnostics = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=truth,
        qcat=qcat,
        cardinalities=cards,
        total=10,
        variances=variances,
        equality_result=equality,
        feasible_result=feasible,
    )

    assert diagnostics["theorem_qualified"] is True
    assert diagnostics["equality_outside_K"] is False
    assert np.allclose(feasible.projected, equality.projected, atol=1.0e-7)
    assert diagnostics["equality_to_nonnegative_omega_squared"] <= 1.0e-12


def test_nonnegative_projection_handles_redundant_and_prefix_constraints() -> None:
    builder = QueryBuilder(max_terms=2)
    for value in range(2):
        builder.add(
            [(0, OP_EQ, value, value, value)],
            f"a={value}",
            "oneway:0",
            "oneway",
        )
    for a in range(2):
        for b in range(2):
            builder.add(
                [(0, OP_EQ, a, a, a), (1, OP_EQ, b, b, b)],
                f"a={a}&b={b}",
                "twoway:0:1",
                "twoway",
            )
    builder.add(
        [(0, OP_LE, 0, 0, 0)],
        "a<=0",
        "prefix:0",
        "prefix",
    )
    qcat = builder.build()
    cards = np.asarray([2, 2], dtype=np.int32)
    truth = np.asarray([6.0, 4.0, 3.0, 3.0, 1.0, 3.0, 6.0], dtype=np.float64)
    noisy = np.asarray([-4.0, 17.0, 8.0, -2.0, 5.0, 4.0, -3.0], dtype=np.float64)
    variances = np.asarray([1.0, 4.0, 2.0, 7.0, 3.0, 9.0, 5.0], dtype=np.float64)

    equality = project_query_space_lsq(noisy, qcat, cards, total=10, variances=variances)
    feasible = project_query_space_feasible_lsq(noisy, qcat, cards, total=10, variances=variances)
    diagnostics = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=truth,
        qcat=qcat,
        cardinalities=cards,
        total=10,
        variances=variances,
        equality_result=equality,
        feasible_result=feasible,
    )

    assert feasible.diagnostics["num_redundant_constraints"] > 0
    assert diagnostics["theorem_qualified"] is True
    assert diagnostics["dominance_holds_within_tolerance"] is True


def test_random_nonnegative_projection_dominance_property() -> None:
    rng = np.random.default_rng(20260712)
    qcat = _complete_oneway(4)
    cards = np.asarray([4], dtype=np.int32)
    total = 50

    for _ in range(20):
        truth = rng.multinomial(total, rng.dirichlet(np.ones(4))).astype(np.float64)
        variances = np.exp(rng.uniform(np.log(0.25), np.log(25.0), size=4))
        noisy = truth + rng.normal(0.0, np.sqrt(variances) * 8.0, size=4)
        equality = project_query_space_lsq(noisy, qcat, cards, total=total, variances=variances)
        feasible = project_query_space_feasible_lsq(
            noisy,
            qcat,
            cards,
            total=total,
            variances=variances,
        )
        diagnostics = offline_query_space_projection_dominance_diagnostics(
            noisy=noisy,
            truth=truth,
            qcat=qcat,
            cardinalities=cards,
            total=total,
            variances=variances,
            equality_result=equality,
            feasible_result=feasible,
            tolerance=5.0e-7,
        )

        assert diagnostics["theorem_qualified"] is True, diagnostics
        assert diagnostics["dominance_holds_within_tolerance"] is True


def test_dominance_audit_rejects_infeasible_truth() -> None:
    noisy = np.asarray([-10.0, 25.0], dtype=np.float64)
    variances = np.asarray([1.0, 9.0], dtype=np.float64)
    qcat, cards, equality, feasible = _project_pair(noisy, variances, total=10)

    diagnostics = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=np.asarray([-1.0, 11.0], dtype=np.float64),
        qcat=qcat,
        cardinalities=cards,
        total=10,
        variances=variances,
        equality_result=equality,
        feasible_result=feasible,
    )

    assert diagnostics["theorem_qualified"] is False
    assert "truth_not_in_K" in diagnostics["failure_reasons"]


def test_dominance_audit_rejects_different_weights() -> None:
    noisy = np.asarray([-10.0, 25.0], dtype=np.float64)
    truth = np.asarray([7.0, 3.0], dtype=np.float64)
    qcat = _complete_oneway(2)
    cards = np.asarray([2], dtype=np.int32)
    equality_variances = np.asarray([1.0, 9.0], dtype=np.float64)
    feasible_variances = np.asarray([2.0, 9.0], dtype=np.float64)
    equality = project_query_space_lsq(
        noisy,
        qcat,
        cards,
        total=10,
        variances=equality_variances,
    )
    feasible = project_query_space_feasible_lsq(
        noisy,
        qcat,
        cards,
        total=10,
        variances=feasible_variances,
    )

    diagnostics = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=truth,
        qcat=qcat,
        cardinalities=cards,
        total=10,
        variances=equality_variances,
        equality_result=equality,
        feasible_result=feasible,
    )

    assert diagnostics["theorem_qualified"] is False
    assert "identity_fingerprint_mismatch" in diagnostics["failure_reasons"]


def test_dominance_audit_rejects_different_constraint_caps() -> None:
    builder = QueryBuilder(max_terms=1)
    for value in range(3):
        builder.add(
            [(0, OP_EQ, value, value, value)],
            f"x={value}",
            "oneway:0",
            "oneway",
        )
    builder.add([(0, OP_LE, 1, 0, 1)], "x<=1", "prefix:0", "prefix")
    qcat = builder.build()
    cards = np.asarray([3], dtype=np.int32)
    noisy = np.asarray([-5.0, 8.0, 12.0, 10.0], dtype=np.float64)
    truth = np.asarray([2.0, 3.0, 5.0, 5.0], dtype=np.float64)
    variances = np.ones(qcat.m, dtype=np.float64)
    equality = project_query_space_lsq(
        noisy,
        qcat,
        cards,
        total=10,
        variances=variances,
        max_constraints=1,
    )
    feasible = project_query_space_feasible_lsq(
        noisy,
        qcat,
        cards,
        total=10,
        variances=variances,
        max_constraints=10,
    )

    diagnostics = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=truth,
        qcat=qcat,
        cardinalities=cards,
        total=10,
        variances=variances,
        equality_result=equality,
        feasible_result=feasible,
        max_constraints=10,
    )

    assert diagnostics["theorem_qualified"] is False
    assert "identity_fingerprint_mismatch" in diagnostics["failure_reasons"]


def test_dominance_audit_rejects_uncertified_loose_dual_solve() -> None:
    qcat = _complete_oneway(3)
    cards = np.asarray([3], dtype=np.int32)
    noisy = np.asarray([-20.0, 5.0, 40.0], dtype=np.float64)
    truth = np.asarray([8.0, 7.0, 5.0], dtype=np.float64)
    variances = np.asarray([1.0, 3.0, 10.0], dtype=np.float64)
    equality = project_query_space_lsq(
        noisy,
        qcat,
        cards,
        total=20,
        variances=variances,
    )
    feasible = project_query_space_feasible_lsq(
        noisy,
        qcat,
        cards,
        total=20,
        variances=variances,
        certificate_max_iterations=0,
        certificate_gap_absolute_tolerance=0.0,
        certificate_gap_relative_tolerance=0.0,
    )

    diagnostics = offline_query_space_projection_dominance_diagnostics(
        noisy=noisy,
        truth=truth,
        qcat=qcat,
        cardinalities=cards,
        total=20,
        variances=variances,
        equality_result=equality,
        feasible_result=feasible,
    )

    assert feasible.diagnostics["feasible_projection_certificate_passed"] is False
    assert diagnostics["theorem_qualified"] is False
    assert "feasible_solver_certificate_failed" in diagnostics["failure_reasons"]


def test_projection_functions_do_not_accept_true_answers() -> None:
    for projector in (project_query_space_lsq, project_query_space_feasible_lsq):
        parameters = inspect.signature(projector).parameters
        assert "truth" not in parameters
        assert "true_answers" not in parameters

    measure_source = (
        Path(__file__).resolve().parents[1] / "qdte" / "measurement" / "measure.py"
    ).read_text(encoding="utf-8")
    assert "offline_query_space_projection_dominance_diagnostics" not in measure_source
