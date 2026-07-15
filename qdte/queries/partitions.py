from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from qdte.queries.types import OP_EQ, QueryBuilder, QueryCatalogue
from qdte.queries.workload import WorkloadGroup
from qdte.schema import TableSchema


def build_selected_pair_partition_workload(
    schema: TableSchema,
    pairs: Sequence[Sequence[int]],
) -> tuple[QueryCatalogue, list[WorkloadGroup]]:
    """Build all one-way partitions and exactly the declared pair partitions."""
    schema.validate()
    canonical_pairs: set[tuple[int, int]] = set()
    for raw_pair in pairs:
        values = tuple(int(value) for value in raw_pair)
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError("each pair must contain two distinct attributes")
        left, right = sorted(values)
        if left < 0 or right >= schema.d:
            raise ValueError("pair contains an out-of-range attribute")
        canonical_pairs.add((left, right))

    builder = QueryBuilder(max_terms=2)
    groups: list[WorkloadGroup] = []

    def add_scope(scope: tuple[int, ...], family: str) -> None:
        start = len(builder.names)
        axes = [range(int(schema.columns[attr].cardinality)) for attr in scope]
        group_name = f"{family}:" + ":".join(str(attr) for attr in scope)
        for values in itertools.product(*axes):
            terms = [
                (int(attr), OP_EQ, int(value), int(value), int(value))
                for attr, value in zip(scope, values, strict=True)
            ]
            name = "&".join(
                f"{schema.columns[attr].name}={value}"
                for attr, value in zip(scope, values, strict=True)
            )
            if not builder.add(terms, name=name, group=group_name, family=family):
                raise RuntimeError(f"duplicate query in partition {group_name}")
        stop = len(builder.names)
        groups.append(
            WorkloadGroup(
                name=group_name,
                family=family,
                query_indices=np.arange(start, stop, dtype=np.int32),
                sensitivity_l2=1.0,
                is_partition=True,
            )
        )

    for attr in range(schema.d):
        add_scope((attr,), "oneway")
    for pair in sorted(canonical_pairs):
        add_scope(pair, "twoway")

    qcat = builder.build()
    qcat.validate(schema.cardinalities)
    return qcat, groups
