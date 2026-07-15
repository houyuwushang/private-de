from __future__ import annotations

import numpy as np

from scripts.evaluate_rce_c1_restricted_mixture import _metrics_from_answers


def test_fractional_answer_metrics_match_expected_rates() -> None:
    true_answers = np.asarray([8.0, 2.0, 4.0, 6.0])
    synthetic_answers = np.asarray([7.0, 3.0, 5.0, 5.0])
    metrics = _metrics_from_answers(
        true_answers,
        synthetic_answers,
        10,
        [np.asarray([0, 1]), np.asarray([2, 3])],
    )
    assert np.isclose(metrics["full_true_mae"], 0.1)
    assert np.isclose(metrics["full_true_rmse"], 0.1)
    assert np.isclose(metrics["full_true_max_error"], 0.1)
    assert np.isclose(metrics["full_true_avg_tvd"], 0.1)
    assert np.isclose(metrics["full_true_max_tvd"], 0.1)
