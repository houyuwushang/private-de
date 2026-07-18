from __future__ import annotations

import numpy as np


def exponential_mechanism_probabilities(
    scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
    *,
    log_base_measure: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    eps = float(epsilon)
    delta = float(sensitivity)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite vector")
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("sensitivity must be finite and positive")
    logits = (eps * values) / (2.0 * delta)
    if log_base_measure is not None:
        base = np.asarray(log_base_measure, dtype=np.float64)
        if base.shape != values.shape or not np.all(np.isfinite(base)):
            raise ValueError("log_base_measure must be finite and match scores")
        logits = logits + base
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    normalizer = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ArithmeticError("exponential mechanism normalization failed")
    return weights / normalizer


def sample_exponential_mechanism(
    scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
    rng: np.random.Generator,
    *,
    log_base_measure: np.ndarray | None = None,
) -> int:
    probabilities = exponential_mechanism_probabilities(
        scores,
        epsilon,
        sensitivity,
        log_base_measure=log_base_measure,
    )
    return int(rng.choice(probabilities.size, p=probabilities))
