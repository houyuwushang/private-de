from __future__ import annotations

import itertools
import math

import numpy as np

from qdte.queries.orthogonal import (
    decompose_pair_table,
    helmert_contrast,
    interaction_features,
    interaction_sensitivity,
    orthogonal_basis,
    reconstruct_pair,
)


def test_helmert_basis_is_orthonormal_and_has_constant_first_column() -> None:
    for cardinality in range(2, 9):
        basis = orthogonal_basis(cardinality)
        contrast = helmert_contrast(cardinality)

        np.testing.assert_allclose(basis.T @ basis, np.eye(cardinality), atol=1.0e-12)
        np.testing.assert_allclose(
            basis[:, 0],
            np.full(cardinality, 1.0 / math.sqrt(cardinality)),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(np.sum(contrast, axis=0), 0.0, atol=1.0e-12)


def test_exhaustive_row_norm_matches_add_remove_sensitivity() -> None:
    cardinalities = np.asarray([2, 3, 4], dtype=np.int32)
    rows = np.asarray(list(itertools.product(*(range(int(d)) for d in cardinalities))), dtype=np.int32)

    for scope in ((0,), (1,), (0, 1), (0, 1, 2)):
        features = interaction_features(rows, scope, cardinalities)
        expected = interaction_sensitivity([cardinalities[attr] for attr in scope])
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), expected, atol=1.0e-12)


def test_pair_decomposition_reconstructs_arbitrary_table() -> None:
    rng = np.random.default_rng(713)
    for shape in ((2, 2), (2, 5), (4, 3)):
        table = rng.normal(loc=10.0, scale=4.0, size=shape)
        total, left, right, interaction = decompose_pair_table(table)
        reconstructed = reconstruct_pair(total, left, right, interaction, *shape)
        np.testing.assert_allclose(reconstructed, table, atol=1.0e-11)


def test_binary_pair_sensitivity_is_one_half() -> None:
    assert math.isclose(interaction_sensitivity((2, 2)), 0.5, abs_tol=1.0e-15)
