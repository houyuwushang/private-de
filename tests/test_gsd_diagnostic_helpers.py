from __future__ import annotations

import numpy as np

from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.queries.workload import WorkloadGroup
from scripts.materialize_gsd_measurement import _remap_groups


def test_gsd_measurement_does_not_mark_truncated_histogram_as_partition() -> None:
    builder = QueryBuilder(max_terms=1)
    for value in range(3):
        builder.add([(0, OP_EQ, value, value, value)], f"a={value}", "a", "twoway")
    qcat = builder.build()
    group = WorkloadGroup(
        name="a",
        family="twoway",
        query_indices=np.arange(qcat.m, dtype=np.int32),
        sensitivity_l2=1.0,
        is_partition=True,
    )

    truncated = _remap_groups([group], np.asarray([0, 2], dtype=np.int32))
    complete = _remap_groups([group], np.asarray([2, 0, 1], dtype=np.int32))

    assert len(truncated) == 1
    assert truncated[0].is_partition is False
    assert complete[0].is_partition is True
