from __future__ import annotations

import numpy as np

from qdte.queries.eval_jax import answer_queries, eval_records_queries
from qdte.queries.types import OP_EQ, OP_LE, OP_RANGE, QueryBuilder, filter_query_catalogue, query_key


def test_query_eval_shapes_and_answers() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_LE, 2, 0, 2)], "b<=2", "prefix:1", "prefix")
    builder.add([(0, OP_EQ, 1, 1, 1), (1, OP_RANGE, 1, 1, 3)], "mixed", "mixed", "mixed")
    qcat = builder.build()
    X = np.asarray([[1, 2], [0, 3], [1, 4], [1, 1]], dtype=np.int32)
    phi = np.asarray(eval_records_queries(X, qcat))
    assert phi.shape == (4, 3)
    assert phi.astype(int).tolist() == [[1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1]]
    assert answer_queries(X, qcat).tolist() == [3.0, 2.0, 2.0]


def test_query_key_and_filter_catalogue_remove_exact_duplicates() -> None:
    measured_builder = QueryBuilder(max_terms=2)
    measured_builder.add([(0, OP_EQ, 1, 1, 1), (1, OP_RANGE, 0, 0, 2)], "dup", "g0", "mixed")
    measured = measured_builder.build()

    heldout_builder = QueryBuilder(max_terms=2)
    heldout_builder.add([(1, OP_RANGE, 0, 0, 2), (0, OP_EQ, 1, 1, 1)], "dup-reordered", "g0", "mixed")
    heldout_builder.add([(1, OP_LE, 1, 0, 1)], "fresh", "g1", "prefix")
    heldout = heldout_builder.build()

    measured_keys = {query_key(measured, qid) for qid in range(measured.m)}
    keep_indices = np.asarray(
        [qid for qid in range(heldout.m) if query_key(heldout, qid) not in measured_keys],
        dtype=np.int32,
    )
    filtered = filter_query_catalogue(heldout, keep_indices)

    assert filtered.m == 1
    assert filtered.names == ["fresh"]
    assert query_key(filtered, 0) not in measured_keys
