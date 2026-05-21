from __future__ import annotations

import math

from qdte.eval.metrics import rms_standardized_residual


def test_rms_standardized_residual_from_loss() -> None:
    loss = 18.0
    num_queries = 9

    assert math.isclose(rms_standardized_residual(loss, num_queries), math.sqrt(2.0 * loss / num_queries))
