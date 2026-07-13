from __future__ import annotations

import numpy as np

from qdte.eval.gsd_progress import GSDProgressObserver


def test_progress_observer_converts_rate_fitness_to_count_loss() -> None:
    observer = GSDProgressObserver(
        n_synthetic=100,
        started_at=0.0,
        min_generation=0,
        relative_improvement_threshold=1.0e-3,
        plateau_checkpoints=2,
    )

    observer.record(10, 0.04, 0.05, np.asarray([0.25, 0.75]), 1.0)

    assert observer.rows[0]["target_count_loss"] == 200.0
    assert np.isclose(observer.rows[0]["relative_improvement_from_previous_checkpoint"], 0.2)


def test_progress_observer_stops_after_predeclared_consecutive_plateaus() -> None:
    observer = GSDProgressObserver(
        n_synthetic=100,
        started_at=0.0,
        min_generation=100,
        relative_improvement_threshold=1.0e-3,
        plateau_checkpoints=2,
    )

    assert observer.should_stop(50, 0.9, 1.0) is False
    assert observer.should_stop(100, 0.8995, 0.9) is False
    assert observer.consecutive_plateau_checkpoints == 1
    assert observer.should_stop(150, 0.8991, 0.8995) is True
    assert observer.stop_generation == 151


def test_progress_observer_resets_plateau_after_material_improvement() -> None:
    observer = GSDProgressObserver(
        n_synthetic=100,
        started_at=0.0,
        min_generation=0,
        relative_improvement_threshold=1.0e-3,
        plateau_checkpoints=2,
    )

    assert observer.should_stop(10, 0.9995, 1.0) is False
    assert observer.consecutive_plateau_checkpoints == 1
    assert observer.should_stop(20, 0.98, 0.9995) is False
    assert observer.consecutive_plateau_checkpoints == 0
