from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianFissionSplit:
    train_noisy: np.ndarray
    validation_noisy: np.ndarray
    train_variances: np.ndarray
    validation_variances: np.ndarray
    train_fraction: float
    gamma: float


@dataclass(frozen=True)
class FissionCheckpointSelection:
    selected_index: int
    best_index: int
    validation_losses: np.ndarray
    validation_avg_partition_l1: np.ndarray
    validation_avg_partition_l2_upper: np.ndarray
    standard_errors_to_best: np.ndarray
    eligible: np.ndarray


def gaussian_fission_split(
    noisy: np.ndarray,
    variances: np.ndarray,
    *,
    train_fraction: float,
    rng: np.random.Generator,
) -> GaussianFissionSplit:
    observed = np.asarray(noisy, dtype=np.float64)
    variance = np.asarray(variances, dtype=np.float64)
    if observed.ndim != 1 or variance.shape != observed.shape:
        raise ValueError("noisy and variances must be one-dimensional with matching shapes")
    if not np.all(np.isfinite(observed)):
        raise ValueError("noisy must be finite")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("variances must be finite and positive")
    fraction = float(train_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    gamma = float(np.sqrt((1.0 - fraction) / fraction))
    auxiliary_noise = rng.normal(loc=0.0, scale=np.sqrt(variance), size=observed.shape)
    train = observed + gamma * auxiliary_noise
    validation = observed - auxiliary_noise / gamma
    return GaussianFissionSplit(
        train_noisy=train.astype(np.float32),
        validation_noisy=validation.astype(np.float32),
        train_variances=(variance / fraction).astype(np.float32),
        validation_variances=(variance / (1.0 - fraction)).astype(np.float32),
        train_fraction=fraction,
        gamma=gamma,
    )


def select_fission_checkpoint(
    checkpoint_answers: np.ndarray,
    validation_noisy: np.ndarray,
    validation_variances: np.ndarray,
    *,
    selection_rule: str = "earliest_within_one_se",
    one_se_multiplier: float = 1.0,
    partition_blocks: list[np.ndarray] | None = None,
    query_weight_multipliers: np.ndarray | None = None,
) -> FissionCheckpointSelection:
    answers = np.asarray(checkpoint_answers, dtype=np.float64)
    target = np.asarray(validation_noisy, dtype=np.float64)
    variance = np.asarray(validation_variances, dtype=np.float64)
    if answers.ndim != 2:
        raise ValueError("checkpoint_answers must be a two-dimensional array")
    if target.ndim != 1 or variance.shape != target.shape or answers.shape[1] != len(target):
        raise ValueError("checkpoint answers, validation target, and variances must have matching query width")
    if answers.shape[0] == 0:
        raise ValueError("at least one checkpoint is required")
    if not np.all(np.isfinite(answers)) or not np.all(np.isfinite(target)):
        raise ValueError("checkpoint answers and validation target must be finite")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("validation variances must be finite and positive")
    multiplier = float(one_se_multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("one_se_multiplier must be finite and non-negative")
    rule = str(selection_rule)
    if rule not in {
        "earliest_within_one_se",
        "validation_minimum",
        "l2_one_se_tvd_minimum",
        "l2_one_se_tvd_upper_minimum",
    }:
        raise ValueError(f"Unsupported fission selection_rule: {rule!r}")

    if query_weight_multipliers is None:
        multipliers = np.ones_like(variance, dtype=np.float64)
    else:
        multipliers = np.asarray(query_weight_multipliers, dtype=np.float64)
        if multipliers.shape != variance.shape:
            raise ValueError("query_weight_multipliers must match the query width")
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
            raise ValueError("query_weight_multipliers must be finite and positive")
    inv = multipliers / variance
    residual = target[None, :] - answers
    losses = 0.5 * np.sum(residual * residual * inv[None, :], axis=1, dtype=np.float64)
    best_index = int(np.argmin(losses))
    answer_delta = answers - answers[best_index][None, :]
    standard_errors = np.sqrt(
        np.maximum(
            0.0,
            np.sum(
                answer_delta
                * answer_delta
                * (inv * inv * variance)[None, :],
                axis=1,
                dtype=np.float64,
            ),
        )
    )
    eligible = losses <= losses[best_index] + multiplier * standard_errors + 1.0e-12
    block_losses: list[np.ndarray] = []
    block_l2_upper_losses: list[np.ndarray] = []
    for raw_indices in list(partition_blocks or []):
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("partition_blocks must contain non-empty one-dimensional indices")
        if np.any(indices < 0) or np.any(indices >= answers.shape[1]):
            raise ValueError("partition block query index is out of bounds")
        block_losses.append(
            0.5 * np.sum(np.abs(residual[:, indices]), axis=1, dtype=np.float64)
        )
        block_l2_upper_losses.append(
            0.25
            * float(indices.size)
            * np.sum(residual[:, indices] ** 2, axis=1, dtype=np.float64)
        )
    if block_losses:
        validation_avg_partition_l1 = np.mean(
            np.stack(block_losses, axis=1), axis=1, dtype=np.float64
        )
    else:
        validation_avg_partition_l1 = np.zeros(answers.shape[0], dtype=np.float64)
    if block_l2_upper_losses:
        validation_avg_partition_l2_upper = np.mean(
            np.stack(block_l2_upper_losses, axis=1), axis=1, dtype=np.float64
        )
    else:
        validation_avg_partition_l2_upper = np.zeros(
            answers.shape[0], dtype=np.float64
        )
    if rule == "validation_minimum":
        selected_index = best_index
    elif rule == "l2_one_se_tvd_minimum":
        if not block_losses:
            raise ValueError("l2_one_se_tvd_minimum requires partition_blocks")
        eligible_indices = np.flatnonzero(eligible)
        selected_index = int(
            eligible_indices[
                np.argmin(validation_avg_partition_l1[eligible_indices])
            ]
        )
    elif rule == "l2_one_se_tvd_upper_minimum":
        if not block_l2_upper_losses:
            raise ValueError("l2_one_se_tvd_upper_minimum requires partition_blocks")
        eligible_indices = np.flatnonzero(eligible)
        selected_index = int(
            eligible_indices[
                np.argmin(validation_avg_partition_l2_upper[eligible_indices])
            ]
        )
    else:
        selected_index = int(np.flatnonzero(eligible)[0])
    return FissionCheckpointSelection(
        selected_index=selected_index,
        best_index=best_index,
        validation_losses=losses,
        validation_avg_partition_l1=validation_avg_partition_l1,
        validation_avg_partition_l2_upper=validation_avg_partition_l2_upper,
        standard_errors_to_best=standard_errors,
        eligible=eligible,
    )
