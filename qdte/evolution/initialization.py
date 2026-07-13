from __future__ import annotations

import numpy as np

from qdte.queries.types import OP_EQ, QueryCatalogue
from qdte.schema import TableSchema


def split_run_rng_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    # Preserve the historical generation seed mapping while isolating measurement draws.
    measurement_seed = np.random.SeedSequence([int(seed), 0x4D454153])
    return np.random.default_rng(measurement_seed), np.random.default_rng(int(seed))


def initialize_independent_oneway(
    qcat: QueryCatalogue,
    target: np.ndarray,
    schema: TableSchema,
    n_syn: int,
    rng: np.random.Generator,
) -> np.ndarray:
    schema.validate()
    qcat.validate(schema.cardinalities)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (qcat.m,):
        raise ValueError(f"target must have shape ({qcat.m},), got {target.shape}")
    if not np.all(np.isfinite(target)):
        raise ValueError("target must be finite")
    if int(n_syn) <= 0:
        raise ValueError("n_syn must be positive")
    X = np.zeros((n_syn, schema.d), dtype=np.int32)
    for attr, col in enumerate(schema.columns):
        indices = [i for i, group in enumerate(qcat.groups) if group == f"oneway:{attr}"]
        counts = np.zeros(int(col.cardinality), dtype=np.float64)
        seen_values: set[int] = set()
        for qid in indices:
            terms = qcat.query_terms(qid)
            if len(terms) != 1 or int(qcat.linear_num_terms[qid]) != 0:
                continue
            query_attr, op, value, _, _ = terms[0]
            if query_attr != attr or op != OP_EQ or not 0 <= value < int(col.cardinality) or value in seen_values:
                continue
            seen_values.add(value)
            counts[value] = max(float(target[qid]), 0.0)
        if len(seen_values) == int(col.cardinality):
            if counts.sum() <= 0:
                probs = np.ones(col.cardinality, dtype=np.float64) / col.cardinality
            else:
                probs = counts / counts.sum()
        else:
            probs = np.ones(col.cardinality, dtype=np.float64) / col.cardinality
        X[:, attr] = rng.choice(col.cardinality, size=n_syn, p=probs)
    return X.astype(np.int32)
