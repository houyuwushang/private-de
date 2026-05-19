from __future__ import annotations

import numpy as np

from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema


def initialize_independent_oneway(
    qcat: QueryCatalogue,
    target: np.ndarray,
    schema: TableSchema,
    n_syn: int,
    rng: np.random.Generator,
) -> np.ndarray:
    X = np.zeros((n_syn, schema.d), dtype=np.int32)
    for attr, col in enumerate(schema.columns):
        indices = [i for i, group in enumerate(qcat.groups) if group == f"oneway:{attr}"]
        if len(indices) == col.cardinality:
            counts = np.maximum(target[np.asarray(indices, dtype=np.int32)], 0.0).astype(np.float64)
            if counts.sum() <= 0:
                probs = np.ones(col.cardinality, dtype=np.float64) / col.cardinality
            else:
                probs = counts / counts.sum()
        else:
            probs = np.ones(col.cardinality, dtype=np.float64) / col.cardinality
        X[:, attr] = rng.choice(col.cardinality, size=n_syn, p=probs)
    return X.astype(np.int32)
