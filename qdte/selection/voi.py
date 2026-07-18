from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from qdte.queries.orthogonal import interaction_coefficients, interaction_sensitivity


def _validated_pair(
    pair: Sequence[int],
    cardinalities: Sequence[int],
    width: int,
) -> tuple[int, int, np.ndarray]:
    cards = np.asarray(cardinalities, dtype=np.int64)
    if cards.shape != (int(width),) or np.any(cards < 2):
        raise ValueError("cardinalities must match the table width and be at least two")
    attrs = tuple(int(value) for value in pair)
    if len(attrs) != 2 or attrs[0] == attrs[1]:
        raise ValueError("pair must contain two distinct attributes")
    left, right = sorted(attrs)
    if left < 0 or right >= int(width):
        raise ValueError("pair contains an out-of-range attribute")
    return left, right, cards


def _validated_rows(rows: np.ndarray, cardinalities: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    table = np.asarray(rows, dtype=np.int64)
    cards = np.asarray(cardinalities, dtype=np.int64)
    if table.ndim != 2 or cards.shape != (table.shape[1],) or np.any(cards < 2):
        raise ValueError("rows and cardinalities have incompatible shapes")
    for attr, cardinality in enumerate(cards.tolist()):
        if np.any(table[:, attr] < 0) or np.any(table[:, attr] >= cardinality):
            raise ValueError(f"rows contain out-of-domain values for attribute {attr}")
    return table, cards


def pair_partition_counts(
    rows: np.ndarray,
    pair: Sequence[int],
    cardinalities: Sequence[int],
) -> np.ndarray:
    table, cards = _validated_rows(rows, cardinalities)
    left, right, _ = _validated_pair(pair, cards, table.shape[1])
    flat = table[:, left] * int(cards[right]) + table[:, right]
    return np.bincount(
        flat,
        minlength=int(cards[left] * cards[right]),
    ).astype(np.float64).reshape(int(cards[left]), int(cards[right]))


def partition_l1_score(
    private_rows: np.ndarray,
    synthetic_rows: np.ndarray,
    pair: Sequence[int],
    cardinalities: Sequence[int],
) -> float:
    """Count-space partition L1 score with add/remove sensitivity one."""
    private, cards = _validated_rows(private_rows, cardinalities)
    synthetic, synthetic_cards = _validated_rows(synthetic_rows, cardinalities)
    if not np.array_equal(cards, synthetic_cards):
        raise ValueError("private and synthetic cardinalities differ")
    discrepancy = pair_partition_counts(private, pair, cards) - pair_partition_counts(
        synthetic,
        pair,
        cards,
    )
    return float(np.sum(np.abs(discrepancy), dtype=np.float64))


def _chi_mean(dimension: int) -> float:
    dim = int(dimension)
    if dim <= 0:
        raise ValueError("dimension must be positive")
    return float(
        math.sqrt(2.0)
        * math.exp(math.lgamma((dim + 1.0) / 2.0) - math.lgamma(dim / 2.0))
    )


def expected_normalized_interaction_noise(dimension: int, measurement_rho: float) -> float:
    rho = float(measurement_rho)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("measurement_rho must be finite and positive")
    return float(_chi_mean(int(dimension)) / math.sqrt(2.0 * rho))


def orthogonal_interaction_score(
    private_rows: np.ndarray,
    synthetic_rows: np.ndarray,
    pair: Sequence[int],
    cardinalities: Sequence[int],
    *,
    measurement_rho: float,
    noise_floor_multiplier: float = 1.0,
) -> float:
    """Noise-floored OI score with conditional add/remove sensitivity one."""
    private, cards = _validated_rows(private_rows, cardinalities)
    synthetic, synthetic_cards = _validated_rows(synthetic_rows, cardinalities)
    if not np.array_equal(cards, synthetic_cards):
        raise ValueError("private and synthetic cardinalities differ")
    left, right, _ = _validated_pair(pair, cards, private.shape[1])
    multiplier = float(noise_floor_multiplier)
    if not math.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("noise_floor_multiplier must be finite and non-negative")
    discrepancy = interaction_coefficients(private, (left, right), cards) - interaction_coefficients(
        synthetic,
        (left, right),
        cards,
    )
    sensitivity = interaction_sensitivity((int(cards[left]), int(cards[right])))
    normalized_signal = float(np.linalg.norm(discrepancy.reshape(-1), ord=2) / sensitivity)
    floor = multiplier * expected_normalized_interaction_noise(
        int(discrepancy.size),
        float(measurement_rho),
    )
    return float(max(0.0, normalized_signal - floor))


def public_pair_action_order(
    pairs: Sequence[Sequence[int]],
    cardinalities: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Public low-noise ordering used by the matched static-R control."""
    cards = np.asarray(cardinalities, dtype=np.int64)
    if cards.ndim != 1 or np.any(cards < 2):
        raise ValueError("cardinalities must be a vector with values at least two")
    canonical = {
        _validated_pair(pair, cards, len(cards))[:2]
        for pair in pairs
    }

    def risk(pair: tuple[int, int]) -> tuple[float, int, int, int]:
        left, right = pair
        dimension = (int(cards[left]) - 1) * (int(cards[right]) - 1)
        sensitivity = interaction_sensitivity((int(cards[left]), int(cards[right])))
        coefficient_noise_trace = float(dimension) * sensitivity * sensitivity
        cells = int(cards[left] * cards[right])
        return coefficient_noise_trace, cells, left, right

    return tuple(sorted(canonical, key=risk))
