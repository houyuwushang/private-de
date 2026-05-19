from __future__ import annotations

import numpy as np

import jax

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.scoring import edit_advantage_from_delta, score_candidates
from qdte.queries.types import OP_EQ, OP_LE, QueryBuilder


def test_edit_advantage_identity() -> None:
    rng = np.random.default_rng(0)
    residual = rng.normal(size=20).astype(np.float32)
    inv_var = rng.uniform(0.1, 2.0, size=20).astype(np.float32)
    delta = rng.integers(-1, 2, size=(5, 20), dtype=np.int8)
    cost = np.zeros(5, dtype=np.float32)
    adv = edit_advantage_from_delta(residual, inv_var, delta, cost, lambda_cost=0.0)
    for i in range(5):
        loss_before = 0.5 * np.sum(residual**2 * inv_var)
        residual_after = residual - delta[i].astype(np.float32)
        loss_after = 0.5 * np.sum(residual_after**2 * inv_var)
        assert np.allclose(loss_after - loss_before, -adv[i], atol=1e-5)


def test_sign_convention_enter_advantage() -> None:
    residual = np.asarray([10.0], dtype=np.float32)
    inv_var = np.asarray([1.0], dtype=np.float32)
    delta = np.asarray([[1]], dtype=np.int8)
    adv = edit_advantage_from_delta(residual, inv_var, delta, np.zeros(1, dtype=np.float32), 0.0)
    assert adv[0] > 0


def test_pmap_and_jit_candidate_scores_match() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    builder.add([(1, OP_LE, 2, 0, 2)], "b<=2", "prefix:1", "prefix")
    qcat = builder.build()
    old_rows = np.asarray([[0, 3], [1, 3], [0, 1], [1, 1]], dtype=np.int32)
    new_rows = np.asarray([[1, 3], [1, 1], [1, 1], [0, 4]], dtype=np.int32)
    candidates = CandidateBatch(
        row_ids=np.arange(len(old_rows), dtype=np.int32),
        old_rows=old_rows,
        new_rows=new_rows,
        target_query_ids=np.zeros(len(old_rows), dtype=np.int32),
        edit_cost=np.ones(len(old_rows), dtype=np.float32),
        repair_type=np.ones(len(old_rows), dtype=np.int32),
    )
    residual = np.asarray([5.0, -3.0], dtype=np.float32)
    inv_var = np.asarray([0.5, 2.0], dtype=np.float32)
    jit_scores = score_candidates(candidates, residual, inv_var, qcat, lambda_cost=0.1, chunk_size=4, use_pmap=False)
    if jax.local_device_count() > 1:
        pmap_scores = score_candidates(candidates, residual, inv_var, qcat, lambda_cost=0.1, chunk_size=4, use_pmap=True)
        assert np.allclose(pmap_scores, jit_scores, atol=1e-5)
