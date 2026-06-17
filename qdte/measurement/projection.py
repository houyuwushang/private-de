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


def project_non_decreasing(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    if weights is None:
        w = np.ones(values.shape, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != values.shape:
            raise ValueError("weights must have the same shape as y")
        w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)

    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    block_ends: list[int] = []

    for idx, (value, weight) in enumerate(zip(values, w, strict=True)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(idx)
        block_ends.append(idx + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / merged_weight
            block_values[-2] = merged_value
            block_weights[-2] = merged_weight
            block_ends[-2] = block_ends[-1]
            block_values.pop()
            block_weights.pop()
            block_starts.pop()
            block_ends.pop()

    projected = np.empty_like(values)
    for value, start, end in zip(block_values, block_starts, block_ends, strict=True):
        projected[start:end] = value
    return projected.astype(np.float32)
