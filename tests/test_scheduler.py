from __future__ import annotations

import numpy as np

from qdte.evolution.scheduler import select_active_queries, update_query_debt


def test_select_active_queries_respects_noise_threshold_without_fallback() -> None:
    residual = np.asarray([0.5, 0.9], dtype=np.float32)
    sigma = np.ones(2, dtype=np.float32)
    debt = np.zeros(2, dtype=np.float32)

    active = select_active_queries(
        residual,
        sigma,
        debt,
        num_active_targets=2,
        kappa_noise=1.0,
        allow_below_noise_fallback=False,
    )

    assert active.size == 0


def test_select_active_queries_can_use_below_noise_fallback() -> None:
    residual = np.asarray([0.5, 0.9], dtype=np.float32)
    sigma = np.ones(2, dtype=np.float32)
    debt = np.zeros(2, dtype=np.float32)

    active = select_active_queries(
        residual,
        sigma,
        debt,
        num_active_targets=1,
        kappa_noise=1.0,
        allow_below_noise_fallback=True,
    )

    assert active.tolist() == [1]


def test_debt_alpha_changes_active_query_priority() -> None:
    residual = np.asarray([10.0, 9.0], dtype=np.float32)
    sigma = np.ones(2, dtype=np.float32)
    debt = np.asarray([0.0, 10.0], dtype=np.float32)

    without_debt = select_active_queries(residual, sigma, debt, 1, kappa_noise=0.0, debt_alpha=0.0)
    with_debt = select_active_queries(residual, sigma, debt, 1, kappa_noise=0.0, debt_alpha=0.5)

    assert without_debt.tolist() == [0]
    assert with_debt.tolist() == [1]


def test_update_query_debt_damage_and_repayment() -> None:
    debt = np.zeros(2, dtype=np.float32)
    inv_variance = np.ones(2, dtype=np.float32)

    damaged, damage_diag = update_query_debt(
        debt,
        residual_before=np.asarray([4.0, 1.0], dtype=np.float32),
        residual_after=np.asarray([2.0, 3.0], dtype=np.float32),
        inv_variance=inv_variance,
        debt_decay=1.0,
        debt_repay=1.0,
    )
    assert damaged[0] == 0.0
    assert damaged[1] > 0.0
    assert damage_diag["max_collateral_damage"] > 0.0

    repaid, _ = update_query_debt(
        damaged,
        residual_before=np.asarray([2.0, 3.0], dtype=np.float32),
        residual_after=np.asarray([2.0, 1.0], dtype=np.float32),
        inv_variance=inv_variance,
        debt_decay=1.0,
        debt_repay=1.0,
    )
    assert repaid[1] < damaged[1]
