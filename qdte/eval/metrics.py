from __future__ import annotations

import math

import numpy as np


def measured_loss(residual: np.ndarray, inv_variance: np.ndarray) -> float:
    r = residual.astype(np.float64)
    inv = inv_variance.astype(np.float64)
    return float(0.5 * np.sum(r * r * inv))


def rms_standardized_residual(loss: float, num_queries: int) -> float:
    if num_queries <= 0:
        return 0.0
    return float(math.sqrt(max(0.0, 2.0 * float(loss) / float(num_queries))))


def query_error_metrics(
    true_answers: np.ndarray,
    syn_answers: np.ndarray,
    n_real: int,
    n_syn: int,
    prefix: str = "true_query",
) -> dict[str, float]:
    true_rate = true_answers.astype(np.float64) / float(n_real)
    syn_rate = syn_answers.astype(np.float64) / float(n_syn)
    diff = syn_rate - true_rate
    return {
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_rmse": float(math.sqrt(float(np.mean(diff * diff)))),
        f"{prefix}_max_error": float(np.max(np.abs(diff))),
    }
