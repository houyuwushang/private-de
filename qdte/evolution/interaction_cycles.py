from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from qdte.evolution.candidates import compute_edit_cost
from qdte.evolution.hybrid_candidates import (
    OPERATOR_INTERACTION_CYCLE,
    HybridCandidateUnitBatch,
)
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.schema import TableSchema


INTERACTION_CYCLE_METHOD = "interaction_rectangle_v1"
INTERACTION_CYCLE_SCOPE_LIMIT = 16


def _empty_batch(width: int) -> HybridCandidateUnitBatch:
    return HybridCandidateUnitBatch(
        row_ids=np.empty((0, 2), dtype=np.int32),
        old_rows=np.empty((0, 2, width), dtype=np.int32),
        new_rows=np.empty((0, 2, width), dtype=np.int32),
        row_mask=np.empty((0, 2), dtype=bool),
        operator_ids=np.empty(0, dtype=np.int8),
        target_query_ids=np.empty(0, dtype=np.int32),
        edit_cost=np.empty(0, dtype=np.float32),
    )


@dataclass(frozen=True)
class InteractionCycleCompilation:
    candidates: HybridCandidateUnitBatch
    target_scopes: np.ndarray
    target_linear_gains: np.ndarray
    diagnostics: dict[str, Any]
    method: str = INTERACTION_CYCLE_METHOD

    def __post_init__(self) -> None:
        self.candidates.validate()
        scopes = np.asarray(self.target_scopes, dtype=np.int32)
        gains = np.asarray(self.target_linear_gains, dtype=np.float64)
        if scopes.shape != (self.candidates.count, 2):
            raise ValueError("target_scopes must have shape (candidate_count, 2)")
        if gains.shape != (self.candidates.count,) or not np.all(np.isfinite(gains)):
            raise ValueError("target_linear_gains must be finite for every candidate")
        if np.any(gains <= 0.0):
            raise ValueError("Every interaction-cycle target linear gain must be positive")
        if self.method != INTERACTION_CYCLE_METHOD:
            raise ValueError(f"Unsupported interaction cycle method {self.method!r}")
        scopes = scopes.copy()
        gains = gains.copy()
        scopes.setflags(write=False)
        gains.setflags(write=False)
        object.__setattr__(self, "target_scopes", scopes)
        object.__setattr__(self, "target_linear_gains", gains)


@dataclass(frozen=True)
class _Rectangle:
    scope: tuple[int, int]
    first_code: int
    second_code: int
    linear_gain: float


def _occupied_buckets(
    rows: np.ndarray,
    scope: tuple[int, int],
    right_cardinality: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    left, right = scope
    codes = (
        rows[:, left].astype(np.int64) * int(right_cardinality)
        + rows[:, right].astype(np.int64)
    )
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    unique, starts = np.unique(sorted_codes, return_index=True)
    ends = np.r_[starts[1:], len(order)]
    result: dict[int, np.ndarray] = {}
    for code, start, end in zip(unique.tolist(), starts.tolist(), ends.tolist(), strict=True):
        row_ids = order[int(start) : int(end)].astype(np.int32, copy=True)
        if len(row_ids) > 1:
            row_ids = row_ids[rng.permutation(len(row_ids))]
        result[int(code)] = row_ids
    return result


def _top_rectangles(
    scope: tuple[int, int],
    potential: np.ndarray,
    buckets: dict[int, np.ndarray],
    *,
    limit: int,
) -> list[_Rectangle]:
    codes = np.asarray(sorted(buckets), dtype=np.int64)
    if len(codes) < 2:
        return []
    right_cardinality = int(potential.shape[1])
    left_values = codes // right_cardinality
    right_values = codes % right_cardinality
    first, second = np.triu_indices(len(codes), k=1)
    valid = (left_values[first] != left_values[second]) & (
        right_values[first] != right_values[second]
    )
    if not np.any(valid):
        return []
    first = first[valid]
    second = second[valid]
    a = left_values[first]
    c = right_values[first]
    b = left_values[second]
    d = right_values[second]
    gains = potential[b, c] + potential[a, d] - potential[a, c] - potential[b, d]
    positive = np.flatnonzero(np.isfinite(gains) & (gains > 0.0))
    if len(positive) == 0:
        return []
    take = min(max(1, int(limit)), len(positive))
    if take < len(positive):
        local = np.argpartition(-gains[positive], take - 1)[:take]
        selected = positive[local]
    else:
        selected = positive
    selected = selected[np.argsort(-gains[selected], kind="stable")]
    return [
        _Rectangle(
            scope=scope,
            first_code=int(codes[first[index]]),
            second_code=int(codes[second[index]]),
            linear_gain=float(gains[index]),
        )
        for index in selected.tolist()
    ]


def compile_interaction_cycle_candidates(
    X_syn: np.ndarray,
    schema: TableSchema,
    residual: np.ndarray,
    precision: OrthogonalInteractionPrecision,
    count: int,
    rng: np.random.Generator,
    *,
    numerical_gamma: float = 0.1,
    scope_limit: int = INTERACTION_CYCLE_SCOPE_LIMIT,
) -> InteractionCycleCompilation:
    """Compile feasible two-row cycles from released pair-interaction residuals."""
    rows = np.asarray(X_syn, dtype=np.int32)
    requested = max(0, int(count))
    if rows.ndim != 2 or rows.shape[1] != schema.d:
        raise ValueError(f"X_syn must have shape (n, {schema.d})")
    if rows.shape[0] < 2 or requested == 0:
        return InteractionCycleCompilation(
            candidates=_empty_batch(schema.d),
            target_scopes=np.empty((0, 2), dtype=np.int32),
            target_linear_gains=np.empty(0, dtype=np.float64),
            diagnostics={
                "method": INTERACTION_CYCLE_METHOD,
                "requested_candidates": requested,
                "compiled_candidates": 0,
                "selected_scopes": 0,
                "positive_rectangles": 0,
            },
        )
    if tuple(int(value) for value in schema.cardinalities) != precision.cardinalities:
        raise ValueError("Schema cardinalities must match orthogonal precision")
    if not np.all(np.isfinite(np.asarray(residual, dtype=np.float64))):
        raise ValueError("residual must be finite")
    if int(scope_limit) != INTERACTION_CYCLE_SCOPE_LIMIT:
        raise ValueError(
            f"The frozen first cycle profile requires scope_limit={INTERACTION_CYCLE_SCOPE_LIMIT}"
        )

    potentials = precision.interaction_cell_potentials(residual)
    ranked_scopes = sorted(
        potentials,
        key=lambda scope: (
            -float(np.linalg.norm(potentials[scope])),
            scope,
        ),
    )[: min(int(scope_limit), len(potentials))]
    rectangles_per_scope = max(
        8,
        2 * int(math.ceil(requested / max(1, 2 * len(ranked_scopes)))),
    )
    buckets_by_scope: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    rectangles: list[_Rectangle] = []
    for scope in ranked_scopes:
        right_cardinality = int(schema.cardinalities[scope[1]])
        buckets = _occupied_buckets(rows, scope, right_cardinality, rng)
        buckets_by_scope[scope] = buckets
        rectangles.extend(
            _top_rectangles(
                scope,
                potentials[scope],
                buckets,
                limit=rectangles_per_scope,
            )
        )
    rectangles.sort(
        key=lambda item: (
            -item.linear_gain,
            item.scope,
            item.first_code,
            item.second_code,
        )
    )

    row_ids_out: list[np.ndarray] = []
    old_rows_out: list[np.ndarray] = []
    new_rows_out: list[np.ndarray] = []
    scopes_out: list[tuple[int, int]] = []
    gains_out: list[float] = []
    seen: set[tuple[int, int, int]] = set()
    rectangle_uses = np.zeros(len(rectangles), dtype=np.int64)
    max_passes = max(1, requested * 4)
    passes = 0
    while len(row_ids_out) < requested and rectangles and passes < max_passes:
        additions_before = len(row_ids_out)
        for rectangle_index, rectangle in enumerate(rectangles):
            buckets = buckets_by_scope[rectangle.scope]
            first_rows = buckets[rectangle.first_code]
            second_rows = buckets[rectangle.second_code]
            use = int(rectangle_uses[rectangle_index])
            first_id = int(first_rows[(use // len(second_rows)) % len(first_rows)])
            second_id = int(second_rows[use % len(second_rows)])
            rectangle_uses[rectangle_index] += 1
            if first_id == second_id:
                raise AssertionError("Distinct occupied pair cells produced the same row id")
            old = rows[np.asarray([first_id, second_id], dtype=np.int32)].copy()
            for swap_attr in rectangle.scope:
                signature = (min(first_id, second_id), max(first_id, second_id), int(swap_attr))
                if signature in seen:
                    continue
                new = old.copy()
                new[0, swap_attr], new[1, swap_attr] = (
                    old[1, swap_attr],
                    old[0, swap_attr],
                )
                if np.array_equal(new, old):
                    continue
                seen.add(signature)
                row_ids_out.append(np.asarray([first_id, second_id], dtype=np.int32))
                old_rows_out.append(old.copy())
                new_rows_out.append(new)
                scopes_out.append(rectangle.scope)
                gains_out.append(rectangle.linear_gain)
                if len(row_ids_out) >= requested:
                    break
            if len(row_ids_out) >= requested:
                break
        passes += 1
        if len(row_ids_out) == additions_before:
            break

    if not row_ids_out:
        candidates = _empty_batch(schema.d)
        scopes_array = np.empty((0, 2), dtype=np.int32)
        gains_array = np.empty(0, dtype=np.float64)
    else:
        row_ids = np.stack(row_ids_out).astype(np.int32)
        old_rows = np.stack(old_rows_out).astype(np.int32)
        new_rows = np.stack(new_rows_out).astype(np.int32)
        changed_attrs = np.any(old_rows != new_rows, axis=1)
        if np.any(np.sum(changed_attrs, axis=1) != 1):
            raise AssertionError("An interaction cycle must swap exactly one attribute")
        if not np.array_equal(np.sort(old_rows, axis=1), np.sort(new_rows, axis=1)):
            raise AssertionError("Interaction cycle failed its one-way preservation certificate")
        flat_cost = compute_edit_cost(
            old_rows.reshape(-1, schema.d),
            new_rows.reshape(-1, schema.d),
            schema,
            numerical_gamma,
        ).reshape(len(row_ids), 2)
        candidates = HybridCandidateUnitBatch(
            row_ids=row_ids,
            old_rows=old_rows,
            new_rows=new_rows,
            row_mask=np.ones((len(row_ids), 2), dtype=bool),
            operator_ids=np.full(
                len(row_ids),
                OPERATOR_INTERACTION_CYCLE,
                dtype=np.int8,
            ),
            target_query_ids=np.full(len(row_ids), -1, dtype=np.int32),
            edit_cost=np.sum(flat_cost, axis=1, dtype=np.float32),
        )
        scopes_array = np.asarray(scopes_out, dtype=np.int32)
        gains_array = np.asarray(gains_out, dtype=np.float64)

    return InteractionCycleCompilation(
        candidates=candidates,
        target_scopes=scopes_array,
        target_linear_gains=gains_array,
        diagnostics={
            "method": INTERACTION_CYCLE_METHOD,
            "requested_candidates": requested,
            "compiled_candidates": candidates.count,
            "available_pair_scopes": len(potentials),
            "selected_scopes": len(ranked_scopes),
            "scope_limit": int(scope_limit),
            "positive_rectangles": len(rectangles),
            "materialization_passes": passes,
            "target_linear_gain_min": (
                float(np.min(gains_array)) if len(gains_array) else 0.0
            ),
            "target_linear_gain_mean": (
                float(np.mean(gains_array)) if len(gains_array) else 0.0
            ),
            "target_linear_gain_max": (
                float(np.max(gains_array)) if len(gains_array) else 0.0
            ),
        },
    )
