from __future__ import annotations

import math

import numpy as np

from qdte.schema import ColumnSchema, TableSchema
from scripts.run_orthogonal_low_budget_pilot import (
    _all_groups_are_complete_partitions,
    _measure_oneway_warmup,
    build_complete_low_order_workload,
    rho_for_epsilon,
)


def _schema() -> TableSchema:
    return TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"]),
            ColumnSchema(name="b", kind="categorical", cardinality=3, categories=["0", "1", "2"]),
            ColumnSchema(name="c", kind="categorical", cardinality=2, categories=["0", "1"]),
        ],
        label_column=None,
    )


def test_complete_low_order_workload_is_partitioned_and_sensitivity_one() -> None:
    schema = _schema()
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=100)

    assert qcat.m == (2 + 3 + 2) + (2 * 3 + 2 * 2 + 3 * 2)
    assert len(groups) == 3 + 3
    assert _all_groups_are_complete_partitions(groups, qcat)
    assert all(group.is_partition for group in groups)
    assert all(math.isclose(group.sensitivity_l2, 1.0) for group in groups)

    rows = np.asarray([[a, b, c] for a in range(2) for b in range(3) for c in range(2)], dtype=np.int32)
    for group in groups:
        hits = np.zeros(rows.shape[0], dtype=np.int32)
        for qid in group.query_indices.tolist():
            hits += qcat.eval_query_np(rows, int(qid)).astype(np.int32)
        assert np.all(hits == 1)


def test_pair_cell_cap_is_public_and_deterministic() -> None:
    schema = _schema()
    first, first_groups = build_complete_low_order_workload(schema, max_pair_cells=4)
    second, second_groups = build_complete_low_order_workload(schema, max_pair_cells=4)

    # Only the (a, c) pair has at most four cells.
    assert [group.name for group in first_groups if group.family == "twoway"] == ["twoway:0:2"]
    assert first.to_dict() == second.to_dict()
    assert [group.to_dict() for group in first_groups] == [group.to_dict() for group in second_groups]


def test_epsilon_conversion_matches_known_low_budget_values() -> None:
    assert math.isclose(rho_for_epsilon(0.1, 1.0e-9), 0.00012034716353620051, rel_tol=1.0e-12)
    assert math.isclose(rho_for_epsilon(0.3, 1.0e-9), 0.0010779477762902546, rel_tol=1.0e-12)


def test_oneway_warmup_measures_only_oneway_partitions() -> None:
    schema = _schema()
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=100)
    rows = np.asarray(
        [[a, b, c] for a in range(2) for b in range(3) for c in range(2)],
        dtype=np.int32,
    )
    true_answers = np.asarray(
        [np.sum(qcat.eval_query_np(rows, qid)) for qid in range(qcat.m)],
        dtype=np.float64,
    )

    measurements = _measure_oneway_warmup(
        true_answers=true_answers,
        qcat=qcat,
        groups=groups,
        schema=schema,
        total_rows=int(rows.shape[0]),
        rho_total=0.3,
        delta=1.0e-9,
        projection_cfg={"project_partitions": True, "clip_nonpartition": True},
        rng=np.random.default_rng(5),
    )

    oneway_qids = np.concatenate(
        [group.query_indices for group in groups if group.family == "oneway"]
    )
    twoway_qids = np.concatenate(
        [group.query_indices for group in groups if group.family == "twoway"]
    )
    assert np.all(measurements.variances[oneway_qids] < 1.0e12)
    assert np.all(measurements.variances[twoway_qids] == np.float32(1.0e12))
    assert np.isclose(measurements.rho_spent, 0.3)
    assert sum(group.family == "oneway" for group in measurements.groups) == schema.d
    assert sum(group.family == "unmeasured" for group in measurements.groups) == 1
