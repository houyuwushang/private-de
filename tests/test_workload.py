from __future__ import annotations

from qdte.queries.workload import build_workload
from qdte.schema import ColumnSchema, TableSchema


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
