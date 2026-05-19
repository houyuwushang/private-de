from __future__ import annotations

import math

import numpy as np

from qdte.measurement.measure import measure_real_dataset
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.queries.workload import WorkloadGroup


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
