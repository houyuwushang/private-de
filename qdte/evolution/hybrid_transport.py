from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qdte.evolution.hybrid_candidates import HybridCandidateUnitBatch, exact_unit_advantage
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.types import QueryCatalogue


@dataclass(frozen=True)
class HybridTransportResult:
    accepted_indices: np.ndarray
    deltas: np.ndarray
    objective_advantages: np.ndarray
    noise_standard_deviations: np.ndarray
    num_noise_guard_rejections: int = 0

    @property
    def count(self) -> int:
        return int(len(self.accepted_indices))

    @property
    def delta_sum(self) -> np.ndarray:
        if self.deltas.shape[0] == 0:
            return np.zeros(self.deltas.shape[1], dtype=np.float32)
        return np.sum(self.deltas, axis=0, dtype=np.float32)


def select_nonconflicting_candidate_units(
    candidates: HybridCandidateUnitBatch,
    scores: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    qcat: QueryCatalogue,
    *,
    max_accept: int,
    min_advantage: float = 1.0e-6,
    lambda_cost: float = 0.0,
    noise_guard_kappa: float = 0.0,
    advantage_noise_variance: np.ndarray | None = None,
    delta_index: QueryDeltaIndex | None = None,
    precomputed_deltas: np.ndarray | None = None,
    objective: str = "quadratic",
    objective_weights: np.ndarray | None = None,
) -> HybridTransportResult:
    """Select an exact-positive, row-nonconflicting unit prefix.

    Initial scores only determine scan order. Every accepted unit is rescored
    against the virtual residual left by earlier units, preserving the QDTE
    finite-difference objective for the whole atomic batch.
    """
    candidates.validate()
    score_arr = np.asarray(scores, dtype=np.float64)
    residual_arr = np.asarray(residual, dtype=np.float32)
    inv_arr = np.asarray(inv_variance, dtype=np.float32)
    if score_arr.shape != (candidates.count,):
        raise ValueError("scores must contain one value per candidate unit")
    if residual_arr.shape != (qcat.m,) or inv_arr.shape != (qcat.m,):
        raise ValueError("Objective vectors must match the query catalogue")
    objective_name = str(objective)
    if objective_name not in {"quadratic", "l1"}:
        raise ValueError("objective must be quadratic or l1")
    l1_weights = (
        inv_arr
        if objective_weights is None
        else np.asarray(objective_weights, dtype=np.float32)
    )
    if l1_weights.shape != (qcat.m,):
        raise ValueError("objective_weights must match the query catalogue")
    limit = max(0, int(max_accept))
    kappa = float(noise_guard_kappa)
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("noise_guard_kappa must be finite and non-negative")
    if limit == 0 or candidates.count == 0:
        return HybridTransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            deltas=np.empty((0, qcat.m), dtype=np.float32),
            objective_advantages=np.empty(0, dtype=np.float64),
            noise_standard_deviations=np.empty(0, dtype=np.float64),
        )

    valid = np.flatnonzero(np.isfinite(score_arr) & (score_arr > float(min_advantage)))
    order = valid[np.argsort(-score_arr[valid], kind="stable")]
    dense_deltas = None
    if precomputed_deltas is not None:
        dense_deltas = np.asarray(precomputed_deltas, dtype=np.float32)
        if dense_deltas.shape != (candidates.count, qcat.m):
            raise ValueError(
                "precomputed_deltas must have shape "
                f"({candidates.count}, {qcat.m}), got {dense_deltas.shape}"
            )
    index = (
        None
        if dense_deltas is not None
        else (delta_index or QueryDeltaIndex.build(qcat, num_attrs=candidates.width))
    )
    virtual_residual = residual_arr.copy()
    inv64 = inv_arr.astype(np.float64)
    if advantage_noise_variance is None:
        noise_variance_per_query = inv64
    else:
        noise_variance_per_query = np.asarray(
            advantage_noise_variance, dtype=np.float64
        )
        if noise_variance_per_query.shape != (qcat.m,):
            raise ValueError("advantage_noise_variance must match the query catalogue")
        if (
            not np.all(np.isfinite(noise_variance_per_query))
            or np.any(noise_variance_per_query < 0.0)
        ):
            raise ValueError("advantage_noise_variance must be finite and non-negative")
    used_rows: set[int] = set()
    accepted: list[int] = []
    dense_accepted_deltas: list[np.ndarray] = []
    sparse_accepted_qids: list[np.ndarray] = []
    sparse_accepted_values: list[np.ndarray] = []
    advantages: list[float] = []
    noise_standard_deviations: list[float] = []
    noise_guard_rejections = 0
    for idx_raw in order.tolist():
        idx = int(idx_raw)
        mask = candidates.row_mask[idx]
        row_ids = candidates.row_ids[idx, mask]
        if any(int(row_id) in used_rows for row_id in row_ids.tolist()):
            continue
        sparse_qids: np.ndarray | None = None
        sparse_values: np.ndarray | None = None
        if dense_deltas is not None:
            delta = dense_deltas[idx]
            if objective_name == "l1":
                advantage = float(
                    (
                        np.abs(virtual_residual.astype(np.float64))
                        - np.abs(
                            virtual_residual.astype(np.float64)
                            - np.asarray(delta, dtype=np.float64)
                        )
                    )
                    @ l1_weights.astype(np.float64)
                    - float(lambda_cost) * float(candidates.edit_cost[idx])
                )
            else:
                advantage = exact_unit_advantage(
                    delta,
                    virtual_residual,
                    inv_arr,
                    lambda_cost=lambda_cost,
                    edit_cost=float(candidates.edit_cost[idx]),
                )
            delta64 = np.asarray(delta, dtype=np.float64)
            noise_variance = float(
                delta64 @ (delta64 * noise_variance_per_query)
            )
        else:
            sparse_delta = index.sparse_delta_sum(  # type: ignore[union-attr]
                candidates.old_rows[idx, mask],
                candidates.new_rows[idx, mask],
            )
            sparse_qids = sparse_delta.qids
            sparse_values = sparse_delta.values.astype(np.float32, copy=False)
            values64 = sparse_values.astype(np.float64)
            local_inv64 = inv64[sparse_qids]
            if objective_name == "l1":
                local_residual = virtual_residual[sparse_qids].astype(np.float64)
                advantage = float(
                    (
                        np.abs(local_residual)
                        - np.abs(local_residual - values64)
                    )
                    @ l1_weights[sparse_qids].astype(np.float64)
                    - float(lambda_cost) * float(candidates.edit_cost[idx])
                )
            else:
                advantage = float(
                    values64
                    @ (virtual_residual[sparse_qids].astype(np.float64) * local_inv64)
                    - 0.5 * ((values64 * values64) @ local_inv64)
                    - float(lambda_cost) * float(candidates.edit_cost[idx])
                )
            noise_variance = float(
                (values64 * values64)
                @ noise_variance_per_query[sparse_qids]
            )
        noise_std = float(np.sqrt(np.maximum(0.0, noise_variance)))
        if not np.isfinite(advantage) or advantage <= float(min_advantage) + kappa * noise_std:
            if np.isfinite(advantage) and advantage > float(min_advantage):
                noise_guard_rejections += 1
            continue
        accepted.append(idx)
        advantages.append(float(advantage))
        noise_standard_deviations.append(noise_std)
        if dense_deltas is not None:
            dense_delta = delta.astype(np.float32, copy=False)
            dense_accepted_deltas.append(dense_delta)
            virtual_residual -= dense_delta
        else:
            if sparse_qids is None or sparse_values is None:
                raise RuntimeError("Sparse structured delta was not initialized")
            sparse_accepted_qids.append(sparse_qids)
            sparse_accepted_values.append(sparse_values)
            virtual_residual[sparse_qids] -= sparse_values
        used_rows.update(int(row_id) for row_id in row_ids.tolist())
        if len(accepted) >= limit:
            break

    if dense_deltas is not None:
        accepted_deltas = np.asarray(dense_accepted_deltas, dtype=np.float32).reshape(-1, qcat.m)
    else:
        accepted_deltas = np.zeros((len(accepted), qcat.m), dtype=np.float32)
        for row, (qids, values) in enumerate(zip(sparse_accepted_qids, sparse_accepted_values)):
            accepted_deltas[row, qids] = values

    return HybridTransportResult(
        accepted_indices=np.asarray(accepted, dtype=np.int32),
        deltas=accepted_deltas,
        objective_advantages=np.asarray(advantages, dtype=np.float64),
        noise_standard_deviations=np.asarray(noise_standard_deviations, dtype=np.float64),
        num_noise_guard_rejections=int(noise_guard_rejections),
    )


def apply_candidate_units(
    X_syn: np.ndarray,
    candidates: HybridCandidateUnitBatch,
    accepted_indices: np.ndarray,
) -> None:
    used_rows: set[int] = set()
    for idx_raw in np.asarray(accepted_indices, dtype=np.int32).tolist():
        idx = int(idx_raw)
        mask = candidates.row_mask[idx]
        row_ids = candidates.row_ids[idx, mask]
        if any(int(row_id) in used_rows for row_id in row_ids.tolist()):
            raise ValueError("Accepted candidate units contain a row conflict")
        X_syn[row_ids] = candidates.new_rows[idx, mask]
        used_rows.update(int(row_id) for row_id in row_ids.tolist())
