from __future__ import annotations

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.entropy import AtomEntropyState, ReleasedProductPrior
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.evolution.rce_scoring import score_candidates_rce_with_quadratic
from qdte.evolution.scoring import prepare_orthogonal_precision_score_context
from qdte.measurement.factorization import (
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.dual import RCEDualState
from qdte.schema import ColumnSchema, TableSchema
from scripts.run_orthogonal_low_budget_pilot import build_complete_low_order_workload


def _problem():
    cardinalities = np.asarray([2, 2], dtype=np.int32)
    schema = TableSchema(
        columns=[
            ColumnSchema(name=f"x{index}", kind="categorical", cardinality=2)
            for index in range(2)
        ]
    )
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=16)
    strategy = compile_hierarchical_pair_strategy(cardinalities, ((0, 1),))
    real = np.asarray(
        [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1]],
        dtype=np.int32,
    )
    transcript = measure_hierarchical_pair_interactions(
        real,
        strategy,
        public_total=len(real),
        rho_total=2.0,
        rng=np.random.default_rng(17),
        allocation_mode="public_optimal",
    )
    precision = OrthogonalInteractionPrecision(qcat, groups, transcript)
    reconstructed = transcript.reconstruct()
    target = np.zeros(qcat.m, dtype=np.float64)
    for group in groups:
        parts = group.name.split(":")
        if group.family == "oneway":
            values = reconstructed.oneway[int(parts[1])]
        else:
            values = reconstructed.pairs[(int(parts[1]), int(parts[2]))]
        target[group.query_indices] = values.reshape(-1)
    return qcat, precision, target


def test_rce_gpu_candidate_scores_match_direct_finite_difference_terms() -> None:
    qcat, precision, target = _problem()
    rows = np.asarray(
        [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1]],
        dtype=np.int32,
    )
    prior = ReleasedProductPrior.from_released_oneway(
        qcat,
        target,
        (2, 2),
        public_total=len(rows),
        smoothing=1.0,
    )
    entropy = AtomEntropyState.from_rows(rows, prior)
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 2, 4], dtype=np.int32),
        old_rows=rows[[0, 2, 4]],
        new_rows=np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.int32),
        target_query_ids=np.asarray([0, 1, 2], dtype=np.int32),
        edit_cost=np.asarray([1.0, 2.0, 1.0], dtype=np.float32),
        repair_type=np.zeros(3, dtype=np.int8),
    )
    synthetic_answer = np.asarray(
        [np.sum(qcat.eval_query_np(rows, qid)) for qid in range(qcat.m)],
        dtype=np.float64,
    )
    residual = target - synthetic_answer
    confidence = RCEConfidenceSet.from_diagonal_variances(
        precision.coefficient_variances
    )
    dual = RCEDualState.create(confidence, max_iterations=5000)
    dual.ellipsoid_weight = 1.7
    dual.tube_positive[:] = np.linspace(0.1, 0.3, confidence.dimension)
    dual.tube_negative[:] = np.linspace(0.05, 0.15, confidence.dimension)

    scores, quadratic = score_candidates_rce_with_quadratic(
        candidates,
        residual,
        precision,
        dual,
        entropy,
        lambda_cost=0.07,
        chunk_size=8,
        context=prepare_orthogonal_precision_score_context(precision),
    )

    coefficient_residual = precision.coefficient_coordinates(residual)
    feature_deltas = precision.row_feature_deltas(
        candidates.old_rows,
        candidates.new_rows,
    )
    expected_scores = dual.candidate_gains(
        residual=coefficient_residual,
        deltas=feature_deltas,
        regularizer_gains=entropy.candidate_gains(
            candidates.old_rows,
            candidates.new_rows,
        )
        / entropy.n_rows,
        edit_costs=candidates.edit_cost,
        lambda_cost=0.07 / entropy.n_rows,
    )
    expected_quadratic = (
        2.0
        * dual.ellipsoid_weight
        / confidence.squared_discrepancy_threshold
        * precision.feature_quadratic_many(feature_deltas)
    )
    assert np.allclose(scores, expected_scores, rtol=2.0e-5, atol=2.0e-5)
    assert np.allclose(quadratic, expected_quadratic, rtol=2.0e-5, atol=2.0e-5)


def test_rce_kl_edit_gain_uses_distribution_scale() -> None:
    qcat, _, target = _problem()
    rows = np.asarray(
        [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1]],
        dtype=np.int32,
    )
    prior = ReleasedProductPrior.from_released_oneway(
        qcat,
        target,
        (2, 2),
        public_total=len(rows),
        smoothing=1.0,
    )
    state = AtomEntropyState.from_rows(rows, prior)
    old = rows[[0]]
    new = np.asarray([[1, 0]], dtype=np.int32)
    before = state.regularizer / state.n_rows
    expected_gain = state.candidate_gains(old, new)[0] / state.n_rows
    applied_gain = state.apply_batch(old, new) / state.n_rows
    after = state.regularizer / state.n_rows

    assert np.isclose(applied_gain, expected_gain, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(applied_gain, before - after, rtol=1.0e-12, atol=1.0e-12)
