from __future__ import annotations

import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.evolution.transport import choose_transport_batch


def test_transport_delta_sum_matches_returned_indices_after_sorting() -> None:
    candidates = CandidateBatch(
        row_ids=np.asarray([0, 1, 2], dtype=np.int32),
        old_rows=np.zeros((3, 1), dtype=np.int32),
        new_rows=np.ones((3, 1), dtype=np.int32),
        target_query_ids=np.zeros(3, dtype=np.int32),
        edit_cost=np.zeros(3, dtype=np.float32),
        repair_type=np.ones(3, dtype=np.int32),
    )
    selected = np.asarray([2, 0, 1], dtype=np.int32)
    deltas = np.asarray(
        [
            [0, 0, 1],  # candidate 2
            [1, 0, 0],  # candidate 0
            [0, 1, 0],  # candidate 1
        ],
        dtype=np.int8,
    )
    advantages = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    result = choose_transport_batch(
        candidates,
        advantages,
        deltas,
        selected,
        residual=np.ones(3, dtype=np.float32) * 10.0,
        inv_variance=np.ones(3, dtype=np.float32),
        lambda_cost=0.0,
    )
    expected = np.asarray([1, 1, 1], dtype=np.float32)
    assert result.accepted_indices.tolist() == [0, 2, 1]
    assert np.allclose(result.delta_sum, expected)
