from __future__ import annotations

import numpy as np


def sample_exponential_mechanism(scores: np.ndarray, epsilon: float, sensitivity: float, rng: np.random.Generator) -> int:
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    scaled = (epsilon * scores.astype(np.float64)) / (2.0 * sensitivity)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return int(rng.choice(len(scores), p=probs))
