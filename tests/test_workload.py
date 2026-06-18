from __future__ import annotations

import math

from qdte.queries.workload import build_workload
from qdte.queries.types import OP_EQ, OP_LE, OP_RANGE
from qdte.schema import ColumnSchema, TableSchema


def test_build_workload_can_include_configurable_kway_family() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 3),
            ColumnSchema("b", "categorical", 4),
            ColumnSchema("c", "numerical_binned", 5),
            ColumnSchema("d", "numerical_binned", 6),
        ]
    )

    qcat, groups = build_workload(
        schema,
        {
            "workload": {
                "include_oneway": False,
                "include_2way_cat": False,
                "include_prefix": False,
                "include_range": False,
                "include_mixed": False,
                "include_kway": True,
                "kway_orders": [3],
                "kway_queries_per_order": 9,
                "max_queries": 20,
                "max_terms": 4,
                "random_seed": 11,
            }
        },
    )

    assert qcat.m == 9
    assert set(qcat.families) == {"kway"}
    assert qcat.num_terms.tolist() == [3] * qcat.m
    assert set(qcat.ops[:, :3].ravel().tolist()) == {OP_EQ}
    assert [group.family for group in groups] == ["kway"]


def test_build_workload_uses_unit_sensitivity_for_orthogonal_kway_cells() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
            ColumnSchema("c", "categorical", 2),
        ]
    )

    qcat, groups = build_workload(
        schema,
        {
            "workload": {
                "include_oneway": False,
                "include_2way_cat": False,
                "include_prefix": False,
                "include_range": False,
                "include_mixed": False,
                "include_kway": True,
                "kway_orders": [3],
                "kway_queries_per_order": 8,
                "max_queries": 20,
                "max_terms": 3,
                "random_seed": 23,
            }
        },
    )

    assert qcat.m == 8
    assert set(qcat.families) == {"kway"}
    assert len(groups) == 1
    assert math.isclose(groups[0].sensitivity_l2, 1.0)


def test_build_workload_can_include_configurable_kway_prefix_and_range_families() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 3),
            ColumnSchema("b", "categorical", 4),
            ColumnSchema("c", "numerical_binned", 5),
            ColumnSchema("d", "numerical_binned", 6),
        ]
    )

    qcat, groups = build_workload(
        schema,
        {
            "workload": {
                "include_oneway": False,
                "include_2way_cat": False,
                "include_prefix": False,
                "include_range": False,
                "include_mixed": False,
                "include_kway_prefix": True,
                "include_kway_range": True,
                "kway_prefix_orders": [3],
                "kway_range_orders": [4],
                "kway_prefix_queries_per_order": 7,
                "kway_range_queries_per_order": 6,
                "max_queries": 20,
                "max_terms": 4,
                "random_seed": 13,
            }
        },
    )

    assert qcat.m == 13
    assert qcat.families.count("kway_prefix") == 7
    assert qcat.families.count("kway_range") == 6
    prefix_qids = [qid for qid, family in enumerate(qcat.families) if family == "kway_prefix"]
    range_qids = [qid for qid, family in enumerate(qcat.families) if family == "kway_range"]
    assert all(int(qcat.num_terms[qid]) == 3 for qid in prefix_qids)
    assert all(int(qcat.num_terms[qid]) == 4 for qid in range_qids)
    assert all(int((qcat.ops[qid, : qcat.num_terms[qid]] == OP_LE).sum()) == 1 for qid in prefix_qids)
    assert all(int((qcat.ops[qid, : qcat.num_terms[qid]] == OP_RANGE).sum()) == 1 for qid in range_qids)
    assert all(
        set(qcat.ops[qid, : qcat.num_terms[qid]].tolist()).issubset({OP_EQ, OP_LE})
        for qid in prefix_qids
    )
    assert all(
        set(qcat.ops[qid, : qcat.num_terms[qid]].tolist()).issubset({OP_EQ, OP_RANGE})
        for qid in range_qids
    )
    assert [group.family for group in groups] == ["kway_prefix", "kway_range"]


def test_build_workload_can_include_configurable_kway_mixed_family() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 3),
            ColumnSchema("b", "categorical", 4),
            ColumnSchema("c", "numerical_binned", 5),
            ColumnSchema("d", "numerical_binned", 6),
        ]
    )

    qcat, groups = build_workload(
        schema,
        {
            "workload": {
                "include_oneway": False,
                "include_2way_cat": False,
                "include_prefix": False,
                "include_range": False,
                "include_mixed": False,
                "include_kway_mixed": True,
                "kway_mixed_orders": [4],
                "kway_mixed_queries_per_order": 8,
                "max_queries": 20,
                "max_terms": 4,
                "random_seed": 17,
            }
        },
    )

    assert qcat.m == 8
    assert set(qcat.families) == {"kway_mixed"}
    assert qcat.num_terms.tolist() == [4] * qcat.m
    assert all(
        set(qcat.ops[qid, : int(qcat.num_terms[qid])].tolist()).issubset({OP_EQ, OP_LE, OP_RANGE})
        for qid in range(qcat.m)
    )
    assert all(
        int((qcat.ops[qid, : int(qcat.num_terms[qid])] != OP_EQ).sum()) == 2
        for qid in range(qcat.m)
    )
    assert [group.family for group in groups] == ["kway_mixed"]


def test_build_workload_can_include_orthogonal_kway_mixed_family() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema("cat", "categorical", 2),
            ColumnSchema("num", "numerical_binned", 5),
        ]
    )

    qcat, groups = build_workload(
        schema,
        {
            "workload": {
                "include_oneway": False,
                "include_2way_cat": False,
                "include_prefix": False,
                "include_range": False,
                "include_mixed": False,
                "include_orthogonal_kway_mixed": True,
                "orthogonal_kway_mixed_orders": [2],
                "orthogonal_kway_mixed_scopes_per_order": 1,
                "orthogonal_kway_mixed_range_bins": 2,
                "max_queries": 20,
                "max_terms": 2,
                "random_seed": 19,
            }
        },
    )

    assert qcat.m == 4
    assert set(qcat.families) == {"orthogonal_kway_mixed"}
    assert len(groups) == 1
    assert groups[0].is_partition is True
    assert math.isclose(groups[0].sensitivity_l2, 1.0)
    assert qcat.num_terms.tolist() == [2] * qcat.m
    assert all(set(qcat.ops[qid, :2].tolist()) == {OP_EQ, OP_RANGE} for qid in range(qcat.m))
    ranges = set()
    for qid in range(qcat.m):
        range_pos = int((qcat.ops[qid, :2] == OP_RANGE).argmax())
        ranges.add((int(qcat.lows[qid, range_pos]), int(qcat.highs[qid, range_pos])))
    assert ranges == {(0, 1), (2, 4)}


def test_build_workload_can_include_halfspace_family() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema("cat", "categorical", 3),
            ColumnSchema("x", "numerical_binned", 4),
            ColumnSchema("y", "numerical_binned", 5),
        ]
    )

    qcat, groups = build_workload(
        schema,
        {
            "workload": {
                "include_oneway": False,
                "include_2way_cat": False,
                "include_prefix": False,
                "include_range": False,
                "include_mixed": False,
                "include_halfspace": True,
                "halfspace_queries": 7,
                "max_queries": 20,
                "max_terms": 2,
                "random_seed": 7,
            }
        },
    )

    assert qcat.m == 7
    assert set(qcat.families) == {"halfspace"}
    assert qcat.linear_num_terms.min() >= 1
    assert qcat.linear_num_terms.max() <= 2
    assert [group.family for group in groups] == ["halfspace"]
