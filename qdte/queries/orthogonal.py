from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def helmert_contrast(cardinality: int) -> np.ndarray:
    """Return a deterministic orthonormal contrast basis with shape ``(d, d - 1)``."""
    d = int(cardinality)
    if d < 2:
        raise ValueError("Orthogonal contrasts require cardinality >= 2")
    contrast = np.zeros((d, d - 1), dtype=np.float64)
    for column in range(d - 1):
        width = column + 1
        scale = np.sqrt(float(width * (width + 1)))
        contrast[:width, column] = 1.0 / scale
        contrast[width, column] = -float(width) / scale
    return contrast


def orthogonal_basis(cardinality: int) -> np.ndarray:
    """Return ``[1 / sqrt(d), C]`` for the Helmert contrast matrix ``C``."""
    d = int(cardinality)
    contrast = helmert_contrast(d)
    constant = np.full((d, 1), 1.0 / np.sqrt(float(d)), dtype=np.float64)
    return np.concatenate((constant, contrast), axis=1)


def interaction_sensitivity(scope_cardinalities: Sequence[int]) -> float:
    """Exact add/remove-one L2 sensitivity of a pure interaction block."""
    cards = tuple(int(value) for value in scope_cardinalities)
    if not cards:
        raise ValueError("Interaction sensitivity requires a non-empty scope")
    if any(cardinality < 2 for cardinality in cards):
        raise ValueError("Interaction cardinalities must all be >= 2")
    squared = np.prod([1.0 - 1.0 / float(cardinality) for cardinality in cards])
    return float(np.sqrt(squared))


def contrast_features(values: np.ndarray, cardinality: int) -> np.ndarray:
    """Evaluate ``C.T @ e_x`` for an encoded categorical value vector."""
    encoded = np.asarray(values, dtype=np.int64)
    if encoded.ndim != 1:
        raise ValueError("Encoded values must be a one-dimensional vector")
    d = int(cardinality)
    if np.any(encoded < 0) or np.any(encoded >= d):
        raise ValueError("Encoded values fall outside the public cardinality")
    return helmert_contrast(d)[encoded]


def interaction_features(
    rows: np.ndarray,
    scope: Sequence[int],
    cardinalities: Sequence[int],
) -> np.ndarray:
    """Evaluate flattened pure-interaction features for every input row."""
    X = np.asarray(rows, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("rows must be a two-dimensional encoded table")
    attrs = tuple(int(attr) for attr in scope)
    if not attrs or len(set(attrs)) != len(attrs):
        raise ValueError("scope must contain distinct attributes")
    cards = np.asarray(cardinalities, dtype=np.int64)
    if cards.shape != (X.shape[1],) or np.any(cards <= 0):
        raise ValueError("cardinalities must match the encoded table width")
    if any(attr < 0 or attr >= X.shape[1] for attr in attrs):
        raise ValueError("scope contains an out-of-range attribute")

    features = np.ones((X.shape[0], 1), dtype=np.float64)
    for attr in attrs:
        local = contrast_features(X[:, attr], int(cards[attr]))
        next_width = int(features.shape[1] * local.shape[1])
        features = np.einsum("ni,nj->nij", features, local, optimize=True).reshape(
            X.shape[0], next_width
        )
    return features


def interaction_coefficients(
    rows: np.ndarray,
    scope: Sequence[int],
    cardinalities: Sequence[int],
) -> np.ndarray:
    """Sum pure-interaction row features and restore their tensor shape."""
    attrs = tuple(int(attr) for attr in scope)
    cards = np.asarray(cardinalities, dtype=np.int64)
    features = interaction_features(rows, attrs, cards)
    shape = tuple(int(cards[attr]) - 1 for attr in attrs)
    return np.sum(features, axis=0, dtype=np.float64).reshape(shape)


def oneway_contrast_from_counts(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("One-way counts must be a finite vector with at least two cells")
    return helmert_contrast(len(values)).T @ values


def pair_interaction_from_table(table: np.ndarray) -> np.ndarray:
    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Pair table must be a finite matrix with both dimensions >= 2")
    left = helmert_contrast(values.shape[0])
    right = helmert_contrast(values.shape[1])
    return left.T @ values @ right


def reconstruct_oneway(
    total: float,
    contrast: np.ndarray,
    cardinality: int,
) -> np.ndarray:
    d = int(cardinality)
    theta = np.asarray(contrast, dtype=np.float64)
    if theta.shape != (d - 1,) or not np.all(np.isfinite(theta)):
        raise ValueError(f"One-way contrast must have shape ({d - 1},)")
    if not np.isfinite(total):
        raise ValueError("total must be finite")
    return np.full(d, float(total) / float(d), dtype=np.float64) + helmert_contrast(d) @ theta


def pair_reconstruction_maps(
    left_cardinality: int,
    right_cardinality: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return maps from left, right and interaction coefficients to row-major cells."""
    d_left = int(left_cardinality)
    d_right = int(right_cardinality)
    C_left = helmert_contrast(d_left)
    C_right = helmert_contrast(d_right)
    left_map = np.kron(C_left, np.ones((d_right, 1), dtype=np.float64) / float(d_right))
    right_map = np.kron(
        np.ones((d_left, 1), dtype=np.float64) / float(d_left),
        C_right,
    )
    interaction_map = np.kron(C_left, C_right)
    return left_map, right_map, interaction_map


def reconstruct_pair(
    total: float,
    left_contrast: np.ndarray,
    right_contrast: np.ndarray,
    interaction: np.ndarray,
    left_cardinality: int,
    right_cardinality: int,
) -> np.ndarray:
    d_left = int(left_cardinality)
    d_right = int(right_cardinality)
    theta_left = np.asarray(left_contrast, dtype=np.float64)
    theta_right = np.asarray(right_contrast, dtype=np.float64)
    theta_interaction = np.asarray(interaction, dtype=np.float64)
    if theta_left.shape != (d_left - 1,):
        raise ValueError(f"left_contrast must have shape ({d_left - 1},)")
    if theta_right.shape != (d_right - 1,):
        raise ValueError(f"right_contrast must have shape ({d_right - 1},)")
    expected_interaction = (d_left - 1, d_right - 1)
    if theta_interaction.shape != expected_interaction:
        raise ValueError(f"interaction must have shape {expected_interaction}")
    if not all(
        np.all(np.isfinite(values))
        for values in (theta_left, theta_right, theta_interaction)
    ) or not np.isfinite(total):
        raise ValueError("Pair reconstruction inputs must be finite")

    C_left = helmert_contrast(d_left)
    C_right = helmert_contrast(d_right)
    reconstructed = np.full(
        (d_left, d_right),
        float(total) / float(d_left * d_right),
        dtype=np.float64,
    )
    reconstructed += np.outer(C_left @ theta_left, np.ones(d_right)) / float(d_right)
    reconstructed += np.outer(np.ones(d_left), C_right @ theta_right) / float(d_left)
    reconstructed += C_left @ theta_interaction @ C_right.T
    return reconstructed


def decompose_pair_table(
    table: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Pair table must be a finite matrix with both dimensions >= 2")
    total = float(np.sum(values))
    left = oneway_contrast_from_counts(np.sum(values, axis=1))
    right = oneway_contrast_from_counts(np.sum(values, axis=0))
    interaction = pair_interaction_from_table(values)
    return total, left, right, interaction
