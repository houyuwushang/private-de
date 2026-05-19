from __future__ import annotations

import numpy as np


def project_simplex(y: np.ndarray, total: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return y.astype(np.float32)
    if total < 0:
        raise ValueError("total must be non-negative")
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - total
    ind = np.arange(1, len(y) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        theta = cssv[-1] / len(y)
    else:
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
    return np.maximum(y - theta, 0.0).astype(np.float32)


def clip_counts(y: np.ndarray, total: float) -> np.ndarray:
    return np.clip(np.asarray(y, dtype=np.float32), 0.0, float(total)).astype(np.float32)
