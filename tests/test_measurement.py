from __future__ import annotations

import math

import numpy as np
import pytest

from qdte.measurement.measure import _allocate_group_budgets, measure_real_dataset
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.queries.workload import WorkloadGroup


def _allocation_groups() -> list[WorkloadGroup]:
    return [
        WorkloadGroup("oneway:0", "oneway", np.asarray([0], dtype=np.int32), 1.0, True),
        WorkloadGroup("oneway:1", "oneway", np.asarray([1], dtype=np.int32), 1.0, True),
        WorkloadGroup("twoway:0:1", "twoway", np.asarray([2], dtype=np.int32), 1.0, True),
    ]


def _spent_rho(groups: list[WorkloadGroup], budgets: dict[str, float]) -> float:
    return float(sum(budgets[group.family] for group in groups))


def test_measurement_noise_parameters() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0, 1], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
    cfg = {
        "privacy": {
            "mode": "dp",
            "rho_total": 2.0,
            "delta": 1e-9,
            "measurement_allocation": {"oneway": 1.0},
        },
        "projection": {"project_partitions": False, "clip_nonpartition": False},
    }
    m = measure_real_dataset(X, qcat, [group], cfg, np.random.default_rng(0), batch_size=4)
    expected_sigma = 1.0 / math.sqrt(2.0 * 2.0)
    assert math.isclose(m.groups[0].sigma, expected_sigma)
    assert math.isclose(m.groups[0].noise_std, expected_sigma)
    assert np.allclose(m.variances, expected_sigma**2)
    assert math.isclose(m.rho_spent, 2.0)
    assert math.isclose(m.to_public_dict()["rho_spent"], 2.0)
    assert not hasattr(m, "true_answers_debug")
    assert "true_answers_debug" not in m.to_public_dict()


def test_oracle_measurement_is_exact() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0, 1], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
    cfg = {"privacy": {"mode": "oracle"}, "projection": {"project_partitions": False}}
    m = measure_real_dataset(X, qcat, [group], cfg, np.random.default_rng(0), batch_size=4)
    assert m.target_noisy.tolist() == [2.0, 2.0]


def test_measurement_can_apply_consistency_projection_with_known_total() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0, 1], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
    cfg = {
        "privacy": {"mode": "oracle"},
        "projection": {
            "project_partitions": False,
            "clip_nonpartition": False,
            "consistency": {"enabled": True, "max_iterations": 10},
        },
    }
    m = measure_real_dataset(
        X,
        qcat,
        [group],
        cfg,
        np.random.default_rng(0),
        batch_size=4,
        cardinalities=np.asarray([2], dtype=np.int32),
    )

    assert np.isclose(float(m.target_projected.sum()), 4.0, atol=1.0e-5)
    assert m.to_public_dict()["projection_diagnostics"]["consistency"]["enabled"] is True


def test_measurement_can_apply_query_space_lsq_consistency_projection() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0, 1], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
    cfg = {
        "privacy": {"mode": "oracle"},
        "projection": {
            "project_partitions": False,
            "clip_nonpartition": False,
            "consistency": {"enabled": True, "method": "query_space_lsq"},
        },
    }
    m = measure_real_dataset(
        X,
        qcat,
        [group],
        cfg,
        np.random.default_rng(0),
        batch_size=4,
        cardinalities=np.asarray([2], dtype=np.int32),
    )

    diagnostics = m.to_public_dict()["projection_diagnostics"]["consistency"]
    assert diagnostics["enabled"] is True
    assert diagnostics["method"] == "query_space_lsq"
    assert np.isclose(float(m.target_projected.sum()), 4.0, atol=1.0e-5)


def test_measurement_can_apply_query_space_feasible_lsq_consistency_projection() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0, 1], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
    cfg = {
        "privacy": {"mode": "oracle"},
        "projection": {
            "project_partitions": False,
            "clip_nonpartition": False,
            "consistency": {"enabled": True, "method": "query_space_feasible_lsq"},
        },
    }
    m = measure_real_dataset(
        X,
        qcat,
        [group],
        cfg,
        np.random.default_rng(0),
        batch_size=4,
        cardinalities=np.asarray([2], dtype=np.int32),
    )

    diagnostics = m.to_public_dict()["projection_diagnostics"]["consistency"]
    assert diagnostics["enabled"] is True
    assert diagnostics["method"] == "query_space_feasible_lsq"
    assert np.all(m.target_projected >= -1.0e-6)
    assert np.all(m.target_projected <= 4.0 + 1.0e-6)
    assert np.isclose(float(m.target_projected.sum()), 4.0, atol=1.0e-5)


def test_measurement_can_apply_local_table_feasible_lsq_consistency_projection() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0, 1], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
    cfg = {
        "privacy": {"mode": "oracle"},
        "projection": {
            "project_partitions": False,
            "clip_nonpartition": False,
            "consistency": {"enabled": True, "method": "local_table_feasible_lsq"},
        },
    }
    m = measure_real_dataset(
        X,
        qcat,
        [group],
        cfg,
        np.random.default_rng(0),
        batch_size=4,
        cardinalities=np.asarray([2], dtype=np.int32),
    )

    diagnostics = m.to_public_dict()["projection_diagnostics"]["consistency"]
    assert diagnostics["enabled"] is True
    assert diagnostics["method"] == "local_table_feasible_lsq"
    assert np.all(m.target_projected >= -1.0e-6)
    assert np.all(m.target_projected <= 4.0 + 1.0e-6)
    assert np.isclose(float(m.target_projected.sum()), 4.0, atol=1.0e-5)


def test_measurement_can_apply_local_table_feasible_jax_consistency_projection() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0, 1], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
    cfg = {
        "privacy": {"mode": "oracle"},
        "projection": {
            "project_partitions": False,
            "clip_nonpartition": False,
            "consistency": {
                "enabled": True,
                "method": "local_table_feasible_jax",
                "jax_iterations": 100,
            },
        },
    }
    m = measure_real_dataset(
        X,
        qcat,
        [group],
        cfg,
        np.random.default_rng(0),
        batch_size=4,
        cardinalities=np.asarray([2], dtype=np.int32),
    )

    diagnostics = m.to_public_dict()["projection_diagnostics"]["consistency"]
    assert diagnostics["enabled"] is True
    assert diagnostics["method"] == "local_table_feasible_jax"
    assert np.all(m.target_projected >= -1.0e-5)
    assert np.all(m.target_projected <= 4.0 + 1.0e-5)
    assert np.isclose(float(m.target_projected.sum()), 4.0, atol=1.0e-4)


def test_adaptive_measurement_mode_fails_fast() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    qcat = builder.build()
    group = WorkloadGroup("oneway:0", "oneway", np.asarray([0], dtype=np.int32), 1.0, True)
    X = np.asarray([[0], [1]], dtype=np.int32)
    cfg = {"privacy": {"mode": "dp", "measurement_mode": "adaptive_select_measure"}}
    try:
        measure_real_dataset(X, qcat, [group], cfg, np.random.default_rng(0), batch_size=2)
    except NotImplementedError as exc:
        assert "static_all" in str(exc)
    else:
        raise AssertionError("adaptive_select_measure should fail fast")


def test_all_family_measurement_allocation_spends_rho_total() -> None:
    groups = _allocation_groups()
    budgets = _allocate_group_budgets(
        groups,
        {"rho_total": 2.0, "measurement_allocation": {"oneway": 1.0, "twoway": 3.0, "unused": 100.0}},
    )

    assert math.isclose(_spent_rho(groups, budgets), 2.0, abs_tol=1.0e-12)
    assert math.isclose(budgets["oneway"], 0.25)
    assert math.isclose(budgets["twoway"], 1.5)


def test_missing_observed_family_measurement_allocation_raises() -> None:
    with pytest.raises(ValueError, match="twoway"):
        _allocate_group_budgets(
            _allocation_groups(),
            {"rho_total": 2.0, "measurement_allocation": {"oneway": 1.0}},
        )


def test_empty_measurement_allocation_distributes_uniformly_over_observed_families() -> None:
    groups = _allocation_groups()
    budgets = _allocate_group_budgets(groups, {"rho_total": 2.0})

    assert math.isclose(_spent_rho(groups, budgets), 2.0, abs_tol=1.0e-12)
    assert math.isclose(budgets["oneway"], 0.5)
    assert math.isclose(budgets["twoway"], 1.0)
