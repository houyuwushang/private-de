from __future__ import annotations

import math


def zcdp_epsilon(rho: float, delta: float) -> float:
    if rho <= 0:
        return 0.0
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))
