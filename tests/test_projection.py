from __future__ import annotations

import numpy as np

from qdte.measurement.measure import MeasurementGroup, project_targets
from qdte.measurement.projection import project_non_decreasing


def test_project_non_decreasing_uses_pava() -> None:
    projected = project_non_decreasing(np.asarray([5.0, 3.0, 10.0, 8.0], dtype=np.float32))

    assert np.all(np.diff(projected) >= -1.0e-6)
    assert np.allclose(projected, np.asarray([4.0, 4.0, 9.0, 9.0], dtype=np.float32))


def test_prefix_projection_clips_to_total_and_is_non_decreasing() -> None:
    group = MeasurementGroup(
        query_indices=np.asarray([0, 1, 2, 3], dtype=np.int32),
        sensitivity_l2=2.0,
        rho=1.0,
        sigma=1.0,
        noise_std=1.0,
        name="prefix:0",
        family="prefix",
        is_partition=False,
    )

    projected = project_targets(
        np.asarray([5.0, -3.0, 12.0, 8.0], dtype=np.float32),
        [group],
        total=10,
        project_partitions=False,
        clip_nonpartition=True,
        prefix_monotonicity=True,
    )

    assert np.all(np.diff(projected) >= -1.0e-6)
    assert np.all(projected >= 0.0)
    assert np.all(projected <= 10.0)


def test_prefix_projection_leaves_monotone_vector_unchanged() -> None:
    y = np.asarray([1.0, 2.0, 2.0, 5.0], dtype=np.float32)

    projected = project_non_decreasing(y)

    assert np.allclose(projected, y)
