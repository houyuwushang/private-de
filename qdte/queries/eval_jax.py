from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue


@jax.jit
def _eval_records_queries_arrays_impl(
    X_batch: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
) -> jax.Array:
    m = attrs.shape[0]
    max_terms = attrs.shape[1]
    satisfied = jnp.ones((X_batch.shape[0], m), dtype=jnp.bool_)
    for term in range(max_terms):
        attr = attrs[:, term]
        valid = attr >= 0
        attr_clipped = jnp.maximum(attr, 0)
        xvals = X_batch[:, attr_clipped]
        op = ops[:, term]
        cond_eq = xvals == values[:, term]
        cond_le = xvals <= values[:, term]
        cond_ge = xvals >= values[:, term]
        cond_range = (xvals >= lows[:, term]) & (xvals <= highs[:, term])
        cond = jnp.where(op[None, :] == OP_EQ, cond_eq, cond_le)
        cond = jnp.where(op[None, :] == OP_GE, cond_ge, cond)
        cond = jnp.where(op[None, :] == OP_RANGE, cond_range, cond)
        cond = jnp.where(valid[None, :], cond, True)
        satisfied = satisfied & cond
    linear_scores = jnp.zeros((X_batch.shape[0], m), dtype=jnp.float32)
    for term in range(max_terms):
        attr = linear_attrs[:, term]
        valid = attr >= 0
        attr_clipped = jnp.maximum(attr, 0)
        xvals = X_batch[:, attr_clipped].astype(jnp.float32)
        linear_scores = linear_scores + jnp.where(
            valid[None, :],
            xvals * linear_weights[:, term].astype(jnp.float32),
            0.0,
        )
    linear_valid = linear_num_terms > 0
    linear_cond = linear_scores <= linear_thresholds.astype(jnp.float32)
    satisfied = satisfied & jnp.where(linear_valid[None, :], linear_cond, True)
    return satisfied


def eval_records_queries_arrays(
    X_batch: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array | None = None,
    linear_weights: jax.Array | None = None,
    linear_thresholds: jax.Array | None = None,
    linear_num_terms: jax.Array | None = None,
) -> jax.Array:
    if linear_attrs is None:
        linear_attrs = jnp.full(attrs.shape, -1, dtype=jnp.int32)
    if linear_weights is None:
        linear_weights = jnp.zeros(attrs.shape, dtype=jnp.float32)
    if linear_thresholds is None:
        linear_thresholds = jnp.zeros((attrs.shape[0],), dtype=jnp.float32)
    if linear_num_terms is None:
        linear_num_terms = jnp.zeros((attrs.shape[0],), dtype=jnp.int32)
    return _eval_records_queries_arrays_impl(
        X_batch,
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    )


def eval_records_queries(X_batch: np.ndarray | jax.Array, qcat: QueryCatalogue) -> jax.Array:
    return eval_records_queries_arrays(
        jnp.asarray(X_batch, dtype=jnp.int32),
        *(
            jnp.asarray(arr, dtype=jnp.float32)
            if idx == 6 or idx == 7
            else jnp.asarray(arr, dtype=jnp.int32)
            for idx, arr in enumerate(qcat.eval_arrays())
        ),
    )


def answer_queries(X: np.ndarray, qcat: QueryCatalogue, batch_size: int = 8192) -> np.ndarray:
    total = np.zeros(qcat.m, dtype=np.float32)
    for start in range(0, X.shape[0], batch_size):
        batch = X[start : start + batch_size]
        phi = eval_records_queries(batch, qcat)
        total += np.asarray(jnp.sum(phi, axis=0), dtype=np.float32)
    return total


def answer_queries_jax(X: np.ndarray | jax.Array, qcat: QueryCatalogue, batch_size: int = 8192) -> np.ndarray:
    return answer_queries(np.asarray(X, dtype=np.int32), qcat, batch_size=batch_size)
