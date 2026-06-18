from __future__ import annotations

import math

import numpy as np

from qdte.eval.metrics import rms_standardized_residual, rms_unweighted_residual, unweighted_measured_loss


def test_rms_standardized_residual_from_loss() -> None:
    loss = 18.0
    num_queries = 9

    assert math.isclose(rms_standardized_residual(loss, num_queries), math.sqrt(2.0 * loss / num_queries))


def test_unweighted_measured_loss_and_rms_residual() -> None:
    residual = np.asarray([3.0, 4.0], dtype=np.float32)

    assert math.isclose(unweighted_measured_loss(residual), 12.5)
    assert math.isclose(rms_unweighted_residual(residual), math.sqrt(12.5))
