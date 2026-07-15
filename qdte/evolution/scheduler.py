from __future__ import annotations

import numpy as np


def debt_diagnostics(debt: np.ndarray, collateral_damage: np.ndarray | None = None) -> dict[str, float | int]:
    debt64 = np.asarray(debt, dtype=np.float64)
    damage64 = np.zeros_like(debt64) if collateral_damage is None else np.asarray(collateral_damage, dtype=np.float64)
    return {
        "mean_debt": float(np.mean(debt64)) if debt64.size else 0.0,
        "max_debt": float(np.max(debt64)) if debt64.size else 0.0,
        "num_positive_debt_queries": int(np.sum(debt64 > 0.0)),
        "mean_collateral_damage": float(np.mean(damage64)) if damage64.size else 0.0,
        "max_collateral_damage": float(np.max(damage64)) if damage64.size else 0.0,
    }


def update_query_debt_from_loss_vectors(
    debt: np.ndarray,
    loss_vec_before: np.ndarray,
    loss_vec_after: np.ndarray,
    debt_decay: float = 0.95,
    debt_repay: float = 1.0,
    debt_cap: float = 1.0e6,
) -> tuple[np.ndarray, dict[str, float | int]]:
    before = np.asarray(loss_vec_before, dtype=np.float64)
    after = np.asarray(loss_vec_after, dtype=np.float64)
    damage = np.maximum(after - before, 0.0)
    improvement = np.maximum(before - after, 0.0)
    updated = float(debt_decay) * np.asarray(debt, dtype=np.float64) + damage - float(debt_repay) * improvement
    updated = np.clip(updated, 0.0, float(debt_cap)).astype(np.float32)
    return updated, debt_diagnostics(updated, damage)


def update_query_debt(
    debt: np.ndarray,
    residual_before: np.ndarray,
    residual_after: np.ndarray,
    inv_variance: np.ndarray,
    debt_decay: float = 0.95,
    debt_repay: float = 1.0,
    debt_cap: float = 1.0e6,
) -> tuple[np.ndarray, dict[str, float | int]]:
    loss_vec_before = 0.5 * np.asarray(residual_before, dtype=np.float64) ** 2 * np.asarray(
        inv_variance, dtype=np.float64
    )
    loss_vec_after = 0.5 * np.asarray(residual_after, dtype=np.float64) ** 2 * np.asarray(
        inv_variance, dtype=np.float64
    )
    return update_query_debt_from_loss_vectors(
        debt,
        loss_vec_before,
        loss_vec_after,
        debt_decay=debt_decay,
        debt_repay=debt_repay,
        debt_cap=debt_cap,
    )


def select_active_queries(
    residual: np.ndarray,
    sigma: np.ndarray,
    debt: np.ndarray,
    num_active_targets: int,
    kappa_noise: float,
    debt_alpha: float = 0.0,
    importance: np.ndarray | None = None,
    importance_beta: float = 0.0,
    allow_below_noise_fallback: bool = False,
) -> np.ndarray:
    safe_sigma = np.maximum(sigma.astype(np.float32), 1.0e-6)
    above_noise = np.abs(residual) > float(kappa_noise) * safe_sigma
    priority = np.full_like(safe_sigma, -np.inf, dtype=np.float32)
    priority[above_noise] = (np.abs(residual[above_noise]) - float(kappa_noise) * safe_sigma[above_noise]) / safe_sigma[
        above_noise
    ]
    if debt_alpha:
        priority = priority + float(debt_alpha) * debt
    if importance is not None and importance_beta:
        priority = priority + float(importance_beta) * importance
    if not np.isfinite(priority).any():
        if not allow_below_noise_fallback:
            return np.empty(0, dtype=np.int32)
        priority = np.abs(residual) / safe_sigma
    finite = np.isfinite(priority)
    k = min(int(num_active_targets), int(np.sum(finite)))
    if k <= 0:
        return np.empty(0, dtype=np.int32)
    finite_idx = np.flatnonzero(finite)
    finite_priority = priority[finite_idx]
    idx = finite_idx[np.argpartition(-finite_priority, kth=k - 1)[:k]]
    return idx[np.argsort(-priority[idx])].astype(np.int32)
