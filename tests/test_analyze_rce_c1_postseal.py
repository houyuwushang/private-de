from __future__ import annotations

import math

from scripts.analyze_rce_c1_postseal import _primary_ratio_rows, _ratio_summary


def _row(dataset: str, epsilon: float, seed: int, arm: str, value: float) -> dict:
    return {
        "dataset": dataset,
        "epsilon": epsilon,
        "seed": seed,
        "arm": arm,
        "full_true_mae": value,
        "full_true_rmse": value * 2.0,
        "full_true_avg_tvd": value * 3.0,
    }


def test_primary_ratio_rows_uses_paired_metric_geometric_mean() -> None:
    rows = [
        _row("adult", 0.1, 0, "left", 2.0),
        _row("adult", 0.1, 0, "right", 1.0),
        _row("adult", 0.1, 1, "left", 8.0),
        _row("adult", 0.1, 1, "right", 2.0),
    ]
    assert math.isclose(_primary_ratio_rows(rows, "left", "right"), math.sqrt(8.0))


def test_ratio_summary_keeps_dataset_and_epsilon_cells_separate() -> None:
    rows = [
        _row(dataset, epsilon, 0, arm, value)
        for dataset, epsilon, arm, value in (
            ("adult", 0.1, "left", 2.0),
            ("adult", 0.1, "right", 1.0),
            ("br2000", 0.1, "left", 1.0),
            ("br2000", 0.1, "right", 2.0),
        )
    ]
    summary = _ratio_summary(rows, "left", "right")
    assert math.isclose(summary["overall"], 1.0)
    assert math.isclose(summary["dataset"]["adult"], 2.0)
    assert math.isclose(summary["dataset"]["br2000"], 0.5)
    assert math.isclose(summary["cell"]["adult|0.1"], 2.0)
