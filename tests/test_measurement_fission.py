from __future__ import annotations

import numpy as np

from qdte.measurement.fission import gaussian_fission_split, select_fission_checkpoint


def test_gaussian_fission_reconstructs_observed_measurement() -> None:
    noisy = np.asarray([2.0, -1.0, 7.5], dtype=np.float32)
    variances = np.asarray([1.0, 4.0, 9.0], dtype=np.float32)
    split = gaussian_fission_split(
        noisy,
        variances,
        train_fraction=0.8,
        rng=np.random.default_rng(17),
    )

    reconstructed = 0.8 * split.train_noisy + 0.2 * split.validation_noisy

    assert np.allclose(reconstructed, noisy, atol=1.0e-6)
    assert np.allclose(split.train_variances, variances / 0.8)
    assert np.allclose(split.validation_variances, variances / 0.2)
    assert np.isclose(split.gamma, 0.5)


def test_gaussian_fission_train_and_validation_are_empirically_independent() -> None:
    size = 200_000
    rng = np.random.default_rng(23)
    original_noise = rng.normal(size=size)
    split = gaussian_fission_split(
        original_noise,
        np.ones(size, dtype=np.float64),
        train_fraction=0.8,
        rng=rng,
    )

    covariance = np.cov(split.train_noisy, split.validation_noisy, ddof=1)[0, 1]

    assert abs(float(covariance)) < 0.03
    assert np.isclose(np.var(split.train_noisy, ddof=1), 1.25, rtol=0.03)
    assert np.isclose(np.var(split.validation_noisy, ddof=1), 5.0, rtol=0.03)


def test_checkpoint_selection_uses_earliest_within_one_standard_error() -> None:
    validation = np.asarray([10.0, 10.0], dtype=np.float32)
    variances = np.ones(2, dtype=np.float32)
    answers = np.asarray(
        [
            [7.0, 7.0],
            [9.0, 9.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    selection = select_fission_checkpoint(
        answers,
        validation,
        variances,
        one_se_multiplier=1.0,
    )

    assert selection.best_index == 2
    assert selection.selected_index == 1
    assert selection.eligible.tolist() == [False, True, True]


def test_checkpoint_selection_can_use_validation_minimum() -> None:
    validation = np.asarray([10.0, 10.0], dtype=np.float32)
    variances = np.ones(2, dtype=np.float32)
    answers = np.asarray(
        [
            [7.0, 7.0],
            [9.0, 9.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    selection = select_fission_checkpoint(
        answers,
        validation,
        variances,
        selection_rule="validation_minimum",
        one_se_multiplier=1.0,
    )

    assert selection.best_index == 2
    assert selection.selected_index == 2


def test_checkpoint_selection_applies_public_query_weight_multipliers() -> None:
    validation = np.zeros(2, dtype=np.float32)
    variances = np.ones(2, dtype=np.float32)
    answers = np.asarray([[0.0, 2.0], [1.5, 0.0]], dtype=np.float32)

    unweighted = select_fission_checkpoint(
        answers,
        validation,
        variances,
        selection_rule="validation_minimum",
    )
    weighted = select_fission_checkpoint(
        answers,
        validation,
        variances,
        selection_rule="validation_minimum",
        query_weight_multipliers=np.asarray([10.0, 1.0], dtype=np.float32),
    )

    assert unweighted.selected_index == 1
    assert weighted.selected_index == 0


def test_checkpoint_selection_can_break_l2_confidence_set_by_partition_l1() -> None:
    validation = np.zeros(3, dtype=np.float32)
    variances = np.ones(3, dtype=np.float32)
    answers = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    selection = select_fission_checkpoint(
        answers,
        validation,
        variances,
        selection_rule="l2_one_se_tvd_minimum",
        one_se_multiplier=1.0,
        partition_blocks=[np.asarray([0, 1], dtype=np.int32)],
    )

    assert selection.best_index == 1
    assert selection.eligible.tolist() == [True, True]
    assert selection.validation_avg_partition_l1.tolist() == [0.0, 0.5]
    assert selection.selected_index == 0


def test_checkpoint_selection_can_use_partition_tvd_quadratic_upper_bound() -> None:
    validation = np.zeros(4, dtype=np.float32)
    variances = np.ones(4, dtype=np.float32)
    answers = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0],
            [0.5, 0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    selection = select_fission_checkpoint(
        answers,
        validation,
        variances,
        selection_rule="l2_one_se_tvd_upper_minimum",
        one_se_multiplier=1.0,
        partition_blocks=[np.asarray([0, 1, 2], dtype=np.int32)],
    )

    assert selection.best_index == 1
    assert selection.eligible.tolist() == [True, True]
    assert selection.validation_avg_partition_l2_upper.tolist() == [0.0, 0.5625]
    assert selection.selected_index == 0
