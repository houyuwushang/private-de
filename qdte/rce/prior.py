from __future__ import annotations

import itertools

import numpy as np

from qdte.evolution.entropy import ReleasedProductPrior


def enumerate_domain_rows(
    cardinalities: tuple[int, ...] | np.ndarray,
    *,
    max_atoms: int = 100_000,
) -> np.ndarray:
    cards = tuple(int(value) for value in np.asarray(cardinalities, dtype=np.int64))
    if not cards or any(value <= 0 for value in cards):
        raise ValueError("cardinalities must be non-empty and positive")
    domain_size = int(np.prod(np.asarray(cards, dtype=object), dtype=object))
    if domain_size > int(max_atoms):
        raise ValueError(
            f"Full RCE domain has {domain_size} atoms, exceeding cap {int(max_atoms)}"
        )
    return np.asarray(list(itertools.product(*(range(value) for value in cards))), dtype=np.int32)


def product_prior_probabilities(
    prior: ReleasedProductPrior,
    rows: np.ndarray,
) -> np.ndarray:
    log_probabilities = prior.log_probability_rows(rows)
    probabilities = np.exp(log_probabilities)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise RuntimeError("Released product prior produced invalid atom probabilities")
    return probabilities


__all__ = [
    "ReleasedProductPrior",
    "enumerate_domain_rows",
    "product_prior_probabilities",
]
