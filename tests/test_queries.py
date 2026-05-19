from __future__ import annotations

import numpy as np

from qdte.queries.eval_jax import answer_queries, eval_records_queries
from qdte.queries.types import OP_EQ, OP_LE, OP_RANGE, QueryBuilder


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
