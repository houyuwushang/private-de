from __future__ import annotations

import numpy as np


def select_active_queries(
    residual: np.ndarray,
    sigma: np.ndarray,
    debt: np.ndarray,
    num_active_targets: int,
    kappa_noise: float,
    debt_alpha: float = 0.0,
    importance: np.ndarray | None = None,
    importance_beta: float = 0.0,
) -> np.ndarray:
    safe_sigma = np.maximum(sigma.astype(np.float32), 1.0e-6)
    priority = (np.abs(residual) - float(kappa_noise) * safe_sigma) / safe_sigma
    priority = np.where(priority > 0.0, priority, -np.inf)
    if debt_alpha:
        priority = priority + float(debt_alpha) * debt
    if importance is not None and importance_beta:
        priority = priority + float(importance_beta) * importance
    if not np.isfinite(priority).any():
        priority = np.abs(residual) / safe_sigma
    k = min(int(num_active_targets), len(priority))
    if k <= 0:
        return np.empty(0, dtype=np.int32)
    idx = np.argpartition(-priority, kth=k - 1)[:k]
    return idx[np.argsort(-priority[idx])].astype(np.int32)
