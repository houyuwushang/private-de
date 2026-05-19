from __future__ import annotations

import math

import numpy as np


def sigma_from_rho(rho: float) -> float:
    if rho <= 0:
        raise ValueError("rho must be positive")
    return float(1.0 / math.sqrt(2.0 * rho))


def add_zcdp_gaussian_noise(
    answers: np.ndarray,
    rho: float,
    sensitivity_l2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    sigma = sigma_from_rho(rho)
    noise_std = float(sensitivity_l2 * sigma)
    noisy = answers.astype(np.float64) + rng.normal(0.0, noise_std, size=answers.shape)
    return noisy.astype(np.float32), sigma, noise_std
