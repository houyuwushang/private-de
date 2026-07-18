from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Literal, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize
from scipy.sparse import csr_matrix, issparse

from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.row_pricing import (
    RowPricingResult,
    ShadowFeatureMap,
    price_legal_row,
)


RHCG_CCMP_METHOD = "released_history_column_generation_ccf_moment_projection_v1"
PHASE_ONE_METHOD = "global_row_polytope_confidence_margin_v1"
PHASE_TWO_METHOD = "global_row_polytope_ccf_moment_projection_v1"


def _readonly_float(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("RHCG arrays must be finite")
    result.setflags(write=False)
    return result


def _readonly_int(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int32).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CanonicalConfidenceGeometry:
    feature_map: ShadowFeatureMap
    target: np.ndarray
    confidence: RCEConfidenceSet
    rho_by_block: dict[str, float]

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=np.float64)
        if target.shape != (self.feature_map.feature_dimension,) or not np.all(
            np.isfinite(target)
        ):
            raise ValueError("RHCG target must match the shadow feature map")
        if self.confidence.dimension != self.feature_map.feature_dimension:
            raise ValueError("RHCG confidence dimension differs from its feature map")
        expected = set(self.feature_map.block_slices)
        if set(self.rho_by_block) != expected:
            raise ValueError("RHCG rho ledger must match the shadow strategy blocks")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.rho_by_block.values()
        ):
            raise ValueError("RHCG rho values must be finite and positive")
        object.__setattr__(self, "target", _readonly_float(target))
        object.__setattr__(
            self,
            "rho_by_block",
            {name: float(value) for name, value in self.rho_by_block.items()},
        )

    @property
    def tube_indices(self) -> np.ndarray:
        return np.flatnonzero(self.confidence.tube_mask)

    @property
    def constraint_count(self) -> int:
        return 1 + 2 * len(self.tube_indices)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        indices = self.tube_indices.tolist()
        return (
            "ellipsoid",
            *(f"tube_positive:{index}" for index in indices),
            *(f"tube_negative:{index}" for index in indices),
        )

    def residual(self, answer: np.ndarray) -> np.ndarray:
        values = np.asarray(answer, dtype=np.float64)
        if values.shape != self.target.shape or not np.all(np.isfinite(values)):
            raise ValueError("RHCG answer must match the canonical target")
        return values - self.target

    def constraint_values(self, answer: np.ndarray) -> np.ndarray:
        residual = self.residual(answer)
        local = residual[self.tube_indices]
        bounds = self.confidence.coordinate_bounds[self.tube_indices]
        return np.concatenate(
            (
                [
                    self.confidence.squared_discrepancy(residual)
                    / self.confidence.squared_discrepancy_threshold
                    - 1.0
                ],
                local / bounds - 1.0,
                -local / bounds - 1.0,
            )
        )

    def constraint_gradients(self, answer: np.ndarray) -> csr_matrix:
        residual = self.residual(answer)
        ellipsoid = (
            2.0
            * self.confidence.precision_matvec(residual)
            / self.confidence.squared_discrepancy_threshold
        )
        bounds = self.confidence.coordinate_bounds[self.tube_indices]
        tube_count = len(self.tube_indices)
        row_indices = np.concatenate(
            (
                np.zeros(self.feature_map.feature_dimension, dtype=np.int64),
                1 + np.arange(tube_count, dtype=np.int64),
                1 + tube_count + np.arange(tube_count, dtype=np.int64),
            )
        )
        column_indices = np.concatenate(
            (
                np.arange(self.feature_map.feature_dimension, dtype=np.int64),
                self.tube_indices,
                self.tube_indices,
            )
        )
        data = np.concatenate((ellipsoid, 1.0 / bounds, -1.0 / bounds))
        return csr_matrix(
            (data, (row_indices, column_indices)),
            shape=(self.constraint_count, self.feature_map.feature_dimension),
        )

    def block_pressures(
        self,
        answer: np.ndarray,
        multipliers: np.ndarray,
    ) -> dict[str, float]:
        dual = np.asarray(multipliers, dtype=np.float64)
        if dual.shape != (self.constraint_count,) or np.any(dual < -1.0e-10):
            raise ValueError("RHCG pressure multipliers have an invalid shape or sign")
        residual = self.residual(answer)
        precision_residual = self.confidence.precision_matvec(residual)
        tube_count = len(self.tube_indices)
        positive = dual[1 : 1 + tube_count]
        negative = dual[1 + tube_count :]
        tube_position = {
            int(coordinate): index
            for index, coordinate in enumerate(self.tube_indices.tolist())
        }
        result: dict[str, float] = {}
        for name, block_slice in self.feature_map.block_slices.items():
            ellipsoid = (
                dual[0]
                * float(
                    residual[block_slice]
                    @ precision_residual[block_slice]
                )
                / self.confidence.squared_discrepancy_threshold
            )
            tubes = 0.0
            for coordinate in range(block_slice.start, block_slice.stop):
                position = tube_position.get(coordinate)
                if position is None:
                    continue
                bound = self.confidence.coordinate_bounds[coordinate]
                scaled = residual[coordinate] / bound
                tubes += 0.5 * (
                    positive[position] * scaled - negative[position] * scaled
                )
            result[name] = max(0.0, float(ellipsoid + tubes))
        return result

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "constraint_normalization": "dimensionless_ellipsoid_and_tubes_v1",
            "feature_hash": self.feature_map.feature_hash,
            "target_sha256": hashlib.sha256(self.target.astype("<f8").tobytes()).hexdigest(),
            "constraint_count": self.constraint_count,
            "constraint_names": list(self.constraint_names),
            "confidence": self.confidence.diagnostics(),
            "rho_by_block": dict(self.rho_by_block),
        }


@dataclass(frozen=True)
class RHCGColumnSet:
    names: tuple[str, ...]
    answers: np.ndarray
    atom_rows: tuple[np.ndarray | None, ...]
    feature_hash: str

    def __post_init__(self) -> None:
        answers = np.asarray(self.answers, dtype=np.float64)
        if answers.ndim != 2 or answers.shape[0] == 0 or not np.all(np.isfinite(answers)):
            raise ValueError("RHCG columns must be a non-empty finite matrix")
        if len(self.names) != len(answers) or len(self.atom_rows) != len(answers):
            raise ValueError("RHCG column metadata must match its answer matrix")
        if len(set(self.names)) != len(self.names):
            raise ValueError("RHCG column names must be unique")
        normalized_rows: list[np.ndarray | None] = []
        seen_rows: set[tuple[int, ...]] = set()
        for row in self.atom_rows:
            if row is None:
                normalized_rows.append(None)
                continue
            values = _readonly_int(np.asarray(row, dtype=np.int32).reshape(-1))
            key = tuple(int(value) for value in values)
            if key in seen_rows:
                raise ValueError("RHCG column set contains a duplicate legal-row atom")
            seen_rows.add(key)
            normalized_rows.append(values)
        object.__setattr__(self, "answers", _readonly_float(answers))
        object.__setattr__(self, "atom_rows", tuple(normalized_rows))

    @classmethod
    def create(
        cls,
        feature_map: ShadowFeatureMap,
        *,
        warm_names: Sequence[str] = (),
        warm_answers: np.ndarray | None = None,
        initial_rows: np.ndarray | None = None,
    ) -> RHCGColumnSet:
        names = [str(value) for value in warm_names]
        if warm_answers is None:
            answers = np.empty((0, feature_map.feature_dimension), dtype=np.float64)
        else:
            answers = np.asarray(warm_answers, dtype=np.float64)
            if answers.shape != (len(names), feature_map.feature_dimension):
                raise ValueError("Warm RHCG answers must match names and feature dimension")
        rows = (
            np.zeros((1, feature_map.domain.dimension), dtype=np.int32)
            if initial_rows is None and len(names) == 0
            else (
                np.empty((0, feature_map.domain.dimension), dtype=np.int32)
                if initial_rows is None
                else feature_map.domain.validate_rows(initial_rows)
            )
        )
        atom_rows: list[np.ndarray | None] = [None] * len(names)
        answer_parts = [answers] if len(answers) else []
        for row in rows:
            names.append("row:" + ":".join(str(int(value)) for value in row))
            atom_rows.append(np.asarray(row, dtype=np.int32))
        if len(rows):
            answer_parts.append(feature_map.row_atom_answers(rows))
        combined = (
            np.concatenate(answer_parts, axis=0)
            if answer_parts
            else np.empty((0, feature_map.feature_dimension), dtype=np.float64)
        )
        return cls(
            names=tuple(names),
            answers=combined,
            atom_rows=tuple(atom_rows),
            feature_hash=feature_map.feature_hash,
        )

    @property
    def column_count(self) -> int:
        return len(self.names)

    @property
    def atom_count(self) -> int:
        return sum(row is not None for row in self.atom_rows)

    @property
    def dictionary_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.feature_hash.encode("ascii"))
        for name, answer, row in zip(
            self.names, self.answers, self.atom_rows, strict=True
        ):
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(answer, dtype="<f8").tobytes())
            if row is not None:
                digest.update(np.asarray(row, dtype="<i4").tobytes())
        return digest.hexdigest()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": "released_history_legal_row_column_set_v1",
            "feature_hash": self.feature_hash,
            "dictionary_hash": self.dictionary_hash,
            "column_count": self.column_count,
            "atom_count": self.atom_count,
            "columns": [
                {
                    "name": name,
                    "legal_row": row.tolist() if row is not None else None,
                    "answer_sha256": hashlib.sha256(
                        np.asarray(answer, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
                for name, answer, row in zip(
                    self.names,
                    self.answers,
                    self.atom_rows,
                    strict=True,
                )
            ],
        }

    def contains_row(self, row: np.ndarray) -> bool:
        key = tuple(int(value) for value in np.asarray(row, dtype=np.int32).reshape(-1))
        return any(
            candidate is not None
            and tuple(int(value) for value in candidate) == key
            for candidate in self.atom_rows
        )

    def add_row(
        self,
        feature_map: ShadowFeatureMap,
        row: np.ndarray,
        *,
        maximum_columns: int,
    ) -> RHCGColumnSet:
        values = feature_map.domain.validate_rows(
            np.asarray(row, dtype=np.int32).reshape(1, -1)
        )[0]
        if self.contains_row(values):
            raise ValueError("RHCG pricing returned a duplicate legal row")
        if self.column_count >= int(maximum_columns):
            raise RuntimeError("RHCG column cap exhausted")
        name = "row:" + ":".join(str(int(value)) for value in values)
        return RHCGColumnSet(
            names=(*self.names, name),
            answers=np.concatenate(
                (self.answers, feature_map.row_atom_answers(values.reshape(1, -1))),
                axis=0,
            ),
            atom_rows=(*self.atom_rows, values),
            feature_hash=self.feature_hash,
        )

    def subset(self, indices: np.ndarray) -> RHCGColumnSet:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(selected) == 0 or len(np.unique(selected)) != len(selected):
            raise ValueError("RHCG column subset must be non-empty and distinct")
        if np.any(selected < 0) or np.any(selected >= self.column_count):
            raise ValueError("RHCG column subset index is out of range")
        return RHCGColumnSet(
            names=tuple(self.names[int(index)] for index in selected),
            answers=self.answers[selected],
            atom_rows=tuple(self.atom_rows[int(index)] for index in selected),
            feature_hash=self.feature_hash,
        )


@dataclass(frozen=True)
class MinimumNormGlobalDual:
    multipliers: np.ndarray
    active_constraints: tuple[int, ...]
    stationarity_residual: float
    complementarity_residual: float
    dual_negativity: float
    restricted_reduced_cost_violation: float
    certified: bool
    solver: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "multipliers", _readonly_float(self.multipliers))

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "multipliers": self.multipliers.tolist(),
            "active_constraints": list(self.active_constraints),
            "stationarity_residual": self.stationarity_residual,
            "complementarity_residual": self.complementarity_residual,
            "dual_negativity": self.dual_negativity,
            "restricted_reduced_cost_violation": self.restricted_reduced_cost_violation,
            "certified": self.certified,
            "solver": dict(self.solver),
        }


@dataclass(frozen=True)
class RHCGPhaseResult:
    phase: str
    objective: float
    inflation: float
    weights: np.ndarray
    answer: np.ndarray
    constraint_values: np.ndarray
    columns: RHCGColumnSet
    dual: MinimumNormGlobalDual
    pricing: RowPricingResult
    global_gap: float
    certified: bool
    iterations: int
    atoms_added: int
    master: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _readonly_float(self.weights))
        object.__setattr__(self, "answer", _readonly_float(self.answer))
        object.__setattr__(
            self, "constraint_values", _readonly_float(self.constraint_values)
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "objective": self.objective,
            "inflation": self.inflation,
            "weights": {
                name: float(weight)
                for name, weight in zip(self.columns.names, self.weights, strict=True)
                if weight > 1.0e-12
            },
            "constraint_values": self.constraint_values.tolist(),
            "dictionary_hash": self.columns.dictionary_hash,
            "column_count": self.columns.column_count,
            "atom_count": self.columns.atom_count,
            "dual": self.dual.to_public_dict(),
            "pricing": self.pricing.to_public_dict(),
            "global_gap": self.global_gap,
            "certified": self.certified,
            "iterations": self.iterations,
            "atoms_added": self.atoms_added,
            "master": dict(self.master),
        }


def _solver_record(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "objective": float(result.fun),
    }


def _caratheodory_reduce(
    columns: RHCGColumnSet,
    weights: np.ndarray,
    *,
    maximum_columns: int,
) -> tuple[RHCGColumnSet, np.ndarray, dict[str, Any]]:
    limit = int(maximum_columns)
    current = np.asarray(weights, dtype=np.float64).copy()
    if current.shape != (columns.column_count,):
        raise ValueError("Caratheodory weights must match the RHCG columns")
    if columns.column_count <= limit:
        return columns, current, {
            "applied": False,
            "columns_before": columns.column_count,
            "columns_after": columns.column_count,
            "answer_residual": 0.0,
        }
    original_answer = current @ columns.answers
    original_sum = float(np.sum(current))
    indices = np.arange(columns.column_count, dtype=np.int64)
    reductions = 0
    while len(indices) > limit:
        local_answers = columns.answers[indices]
        augmented = np.concatenate(
            (local_answers.T, np.ones((1, len(indices)), dtype=np.float64)),
            axis=0,
        )
        _, singular, right = np.linalg.svd(augmented, full_matrices=True)
        leading = float(singular[0]) if len(singular) else 0.0
        threshold = (
            np.finfo(np.float64).eps
            * max(augmented.shape)
            * max(1.0, leading)
            * 100.0
        )
        rank = int(np.sum(singular > threshold))
        if rank >= len(indices):
            raise RuntimeError(
                "RHCG column cap exhausted without an affine dependence"
            )
        direction = np.asarray(right[rank], dtype=np.float64)
        nonzero = np.flatnonzero(np.abs(direction) > threshold)
        if len(nonzero) == 0:
            raise RuntimeError("Caratheodory reduction found a zero null direction")
        if direction[int(nonzero[0])] < 0.0:
            direction = -direction
        positive = np.flatnonzero(direction > threshold)
        if len(positive) == 0:
            raise RuntimeError("Caratheodory null direction has no positive coordinate")
        local_weights = current[indices]
        step = float(np.min(local_weights[positive] / direction[positive]))
        local_weights = local_weights - step * direction
        local_weights[np.abs(local_weights) <= 1.0e-12] = 0.0
        if float(np.min(local_weights)) < -1.0e-10:
            raise RuntimeError("Caratheodory reduction produced a negative weight")
        current[indices] = np.maximum(local_weights, 0.0)
        removable = np.flatnonzero(current[indices] == 0.0)
        if len(removable) == 0:
            removable = np.asarray([int(np.argmin(current[indices]))])
        keep_mask = np.ones(len(indices), dtype=bool)
        keep_mask[int(removable[0])] = False
        indices = indices[keep_mask]
        reductions += 1
    reduced_weights = current[indices]
    reduced_columns = columns.subset(indices)
    answer_residual = float(
        np.max(
            np.abs(reduced_weights @ reduced_columns.answers - original_answer),
            initial=0.0,
        )
    )
    simplex_residual = abs(float(np.sum(reduced_weights)) - original_sum)
    tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(original_answer))))
    if answer_residual > tolerance or simplex_residual > 1.0e-10:
        raise RuntimeError("Caratheodory reduction did not preserve the master answer")
    return reduced_columns, reduced_weights, {
        "applied": True,
        "columns_before": columns.column_count,
        "columns_after": reduced_columns.column_count,
        "reductions": reductions,
        "answer_residual": answer_residual,
        "simplex_residual": simplex_residual,
    }


def _normalize_simplex(weights: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        values = np.zeros_like(values)
        values[0] = 1.0
        return values
    return values / total


def _compress_equalities(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    coefficients = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if coefficients.ndim != 2 or coefficients.shape[0] != len(targets):
        raise ValueError("Equality system has incompatible dimensions")
    if coefficients.shape[0] == 0:
        return coefficients, targets, 0.0
    left, singular, right = np.linalg.svd(coefficients, full_matrices=False)
    if len(singular) == 0 or singular[0] <= 0.0:
        inconsistency = float(np.max(np.abs(targets), initial=0.0))
        return np.empty((0, coefficients.shape[1])), np.empty(0), inconsistency
    threshold = (
        np.finfo(np.float64).eps
        * max(coefficients.shape)
        * float(singular[0])
        * 100.0
    )
    rank = int(np.sum(singular > threshold))
    projected = left[:, :rank] @ (left[:, :rank].T @ targets)
    inconsistency = float(np.max(np.abs(targets - projected), initial=0.0))
    reduced_matrix = right[:rank]
    reduced_rhs = (left[:, :rank].T @ targets) / singular[:rank]
    return reduced_matrix, reduced_rhs, inconsistency


def _phase_two_kkt_polish(
    *,
    weights: np.ndarray,
    columns: RHCGColumnSet,
    geometry: CanonicalConfidenceGeometry,
    prior_moments: np.ndarray,
    inflation: float,
    precision: np.ndarray,
    initial_constraint_multipliers: np.ndarray | None,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = columns.answers
    current = _normalize_simplex(weights)
    current_answer = current @ matrix
    current_values = geometry.constraint_values(current_answer)
    support = np.flatnonzero(current > 1.0e-8)
    active = np.flatnonzero(float(inflation) - current_values <= 1.0e-7)
    if len(support) == 0:
        return current, {
            "attempted": False,
            "reason": "empty_support",
        }

    multipliers = np.zeros(len(active), dtype=np.float64)
    if initial_constraint_multipliers is not None:
        candidate = np.asarray(initial_constraint_multipliers, dtype=np.float64)
        if candidate.shape == (geometry.constraint_count,):
            multipliers = np.maximum(candidate[active], 0.0)
    base_gradient = (current_answer - prior_moments) * precision
    active_energy = np.asarray(
        geometry.constraint_gradients(current_answer)[active] @ matrix.T,
        dtype=np.float64,
    ).T
    stationarity = matrix @ base_gradient + active_energy @ multipliers
    equality_multiplier = -float(np.mean(stationarity[support]))
    start = np.concatenate(
        (current[support], multipliers, [equality_multiplier])
    )

    def residual(local: np.ndarray) -> np.ndarray:
        local_weights = local[: len(support)]
        local_multipliers = local[
            len(support) : len(support) + len(active)
        ]
        local_equality = float(local[-1])
        full_weights = np.zeros_like(current)
        full_weights[support] = local_weights
        answer = full_weights @ matrix
        objective_gradient = (answer - prior_moments) * precision
        constraint_gradients = geometry.constraint_gradients(answer)[active]
        energy = (
            matrix @ objective_gradient
            + np.asarray(
                constraint_gradients @ matrix.T,
                dtype=np.float64,
            ).T
            @ local_multipliers
            + local_equality
        )
        return np.concatenate(
            (
                [float(np.sum(full_weights)) - 1.0],
                geometry.constraint_values(answer)[active] - float(inflation),
                energy[support],
            )
        )

    lower = np.concatenate(
        (
            np.zeros(len(support) + len(active), dtype=np.float64),
            [-np.inf],
        )
    )
    upper = np.full(len(start), np.inf, dtype=np.float64)
    polished = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        xtol=1.0e-15,
        ftol=1.0e-15,
        gtol=1.0e-15,
        x_scale="jac",
        max_nfev=max(10_000, int(max_iterations)),
    )
    polished_weights = np.zeros_like(current)
    polished_weights[support] = polished.x[: len(support)]
    polished_values = geometry.constraint_values(polished_weights @ matrix)
    maximum_residual = float(np.max(np.abs(residual(polished.x)), initial=0.0))
    accepted = bool(
        abs(float(np.sum(polished_weights)) - 1.0) <= 1.0e-10
        and np.all(polished_weights >= -1.0e-12)
        and float(np.max(polished_values - float(inflation), initial=0.0))
        <= 1.0e-8
        and maximum_residual <= 1.0e-7
    )
    record = {
        "attempted": True,
        "accepted": accepted,
        "success": bool(polished.success),
        "status": int(polished.status),
        "message": str(polished.message),
        "function_evaluations": int(polished.nfev),
        "optimality": float(polished.optimality),
        "cost": float(polished.cost),
        "maximum_kkt_residual": maximum_residual,
        "support_size": int(len(support)),
        "active_constraint_indices": active.tolist(),
    }
    return (polished_weights if accepted else current), record


def _phase_one_kkt_polish(
    *,
    weights: np.ndarray,
    columns: RHCGColumnSet,
    geometry: CanonicalConfidenceGeometry,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = columns.answers
    current = _normalize_simplex(weights)
    current_answer = current @ matrix
    current_values = geometry.constraint_values(current_answer)
    face = float(np.max(current_values))
    support = np.flatnonzero(current > 1.0e-8)
    active = np.flatnonzero(face - current_values <= 1.0e-7)
    if len(support) == 0 or len(active) == 0:
        return current, {
            "attempted": False,
            "reason": "empty_support_or_active_set",
        }
    multipliers = np.full(len(active), 1.0 / len(active), dtype=np.float64)
    active_energy = np.asarray(
        geometry.constraint_gradients(current_answer)[active] @ matrix.T,
        dtype=np.float64,
    ).T
    equality_multiplier = -float(
        np.mean((active_energy @ multipliers)[support])
    )
    start = np.concatenate(
        (
            current[support],
            multipliers,
            [equality_multiplier, face],
        )
    )

    def residual(local: np.ndarray) -> np.ndarray:
        local_weights = local[: len(support)]
        local_multipliers = local[
            len(support) : len(support) + len(active)
        ]
        local_equality = float(local[-2])
        local_face = float(local[-1])
        full_weights = np.zeros_like(current)
        full_weights[support] = local_weights
        answer = full_weights @ matrix
        constraint_gradients = geometry.constraint_gradients(answer)[active]
        energy = (
            np.asarray(
                constraint_gradients @ matrix.T,
                dtype=np.float64,
            ).T
            @ local_multipliers
            + local_equality
        )
        return np.concatenate(
            (
                [float(np.sum(full_weights)) - 1.0],
                [float(np.sum(local_multipliers)) - 1.0],
                geometry.constraint_values(answer)[active] - local_face,
                energy[support],
            )
        )

    lower = np.concatenate(
        (
            np.zeros(len(support) + len(active), dtype=np.float64),
            [-np.inf, -np.inf],
        )
    )
    upper = np.full(len(start), np.inf, dtype=np.float64)
    polished = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        xtol=1.0e-15,
        ftol=1.0e-15,
        gtol=1.0e-15,
        x_scale="jac",
        max_nfev=max(10_000, int(max_iterations)),
    )
    polished_weights = np.zeros_like(current)
    polished_weights[support] = polished.x[: len(support)]
    polished_values = geometry.constraint_values(polished_weights @ matrix)
    polished_face = float(np.max(polished_values))
    declared_face = float(polished.x[-1])
    maximum_residual = float(np.max(np.abs(residual(polished.x)), initial=0.0))
    inactive_violation = float(
        np.max(polished_values - declared_face, initial=0.0)
    )
    accepted = bool(
        abs(float(np.sum(polished_weights)) - 1.0) <= 1.0e-10
        and np.all(polished_weights >= -1.0e-12)
        and inactive_violation <= 1.0e-10
        and maximum_residual <= 1.0e-7
    )
    record = {
        "attempted": True,
        "accepted": accepted,
        "success": bool(polished.success),
        "status": int(polished.status),
        "message": str(polished.message),
        "function_evaluations": int(polished.nfev),
        "optimality": float(polished.optimality),
        "cost": float(polished.cost),
        "maximum_kkt_residual": maximum_residual,
        "support_size": int(len(support)),
        "active_constraint_indices": active.tolist(),
        "recomputed_face": polished_face,
        "declared_face": declared_face,
        "inactive_constraint_violation": max(0.0, inactive_violation),
    }
    return (polished_weights if accepted else current), record


def _phase_one_master(
    columns: RHCGColumnSet,
    geometry: CanonicalConfidenceGeometry,
    *,
    initial_weights: np.ndarray | None = None,
    max_iterations: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict[str, Any]]:
    matrix = columns.answers
    count = len(matrix)
    if initial_weights is None:
        column_objectives = np.asarray(
            [
                float(np.max(geometry.constraint_values(answer)))
                for answer in matrix
            ],
            dtype=np.float64,
        )
        weights0 = np.zeros(count, dtype=np.float64)
        weights0[int(np.argmin(column_objectives))] = 1.0
    else:
        weights0 = _normalize_simplex(initial_weights)
    if weights0.shape != (count,):
        raise ValueError("Phase-I initial weights must match the column count")
    answer0 = weights0 @ matrix
    slack0 = float(np.max(geometry.constraint_values(answer0)))
    start = np.concatenate((weights0, [slack0 + 1.0e-10]))

    def objective(values: np.ndarray) -> float:
        return float(values[-1])

    def objective_jacobian(values: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(values)
        gradient[-1] = 1.0
        return gradient

    def inequalities(values: np.ndarray) -> np.ndarray:
        answer = values[:-1] @ matrix
        return values[-1] - geometry.constraint_values(answer)

    def inequalities_jacobian(values: np.ndarray) -> np.ndarray:
        answer = values[:-1] @ matrix
        gradients = geometry.constraint_gradients(answer)
        jacobian = np.empty((geometry.constraint_count, count + 1), dtype=np.float64)
        jacobian[:, :count] = -np.asarray(
            gradients @ matrix.T,
            dtype=np.float64,
        )
        jacobian[:, -1] = 1.0
        return jacobian

    result = minimize(
        objective,
        start,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=Bounds(
            np.concatenate((np.zeros(count), [-np.inf])),
            np.concatenate((np.ones(count), [np.inf])),
        ),
        constraints=(
            {
                "type": "eq",
                "fun": lambda values: np.asarray([np.sum(values[:-1]) - 1.0]),
                "jac": lambda values: np.asarray([[*np.ones(count), 0.0]]),
            },
            {
                "type": "ineq",
                "fun": inequalities,
                "jac": inequalities_jacobian,
            },
        ),
        options={"maxiter": int(max_iterations), "ftol": 1.0e-13, "disp": False},
    )
    weights = _normalize_simplex(result.x[:-1])
    answer = weights @ matrix
    constraints = geometry.constraint_values(answer)
    active = np.flatnonzero(
        constraints >= float(np.max(constraints)) - 1.0e-8
    )
    polish_record: dict[str, Any] | None = None
    if len(active) == 1:
        active_index = int(active[0])

        def active_objective(local_weights: np.ndarray) -> float:
            return float(
                geometry.constraint_values(local_weights @ matrix)[active_index]
            )

        def active_jacobian(local_weights: np.ndarray) -> np.ndarray:
            local_answer = local_weights @ matrix
            return np.asarray(
                geometry.constraint_gradients(local_answer)[active_index] @ matrix.T,
                dtype=np.float64,
            ).reshape(-1)

        polish = minimize(
            active_objective,
            weights,
            jac=active_jacobian,
            method="SLSQP",
            bounds=Bounds(np.zeros(count), np.ones(count)),
            constraints=(
                {
                    "type": "eq",
                    "fun": lambda local_weights: np.asarray(
                        [np.sum(local_weights) - 1.0]
                    ),
                    "jac": lambda local_weights: np.ones(
                        (1, count), dtype=np.float64
                    ),
                },
            ),
            options={"maxiter": int(max_iterations), "ftol": 1.0e-14, "disp": False},
        )
        polished_weights = _normalize_simplex(polish.x)
        polished_answer = polished_weights @ matrix
        polished_constraints = geometry.constraint_values(polished_answer)
        if float(np.max(polished_constraints)) <= float(np.max(constraints)) + 1.0e-10:
            weights = polished_weights
            answer = polished_answer
            constraints = polished_constraints
        polish_record = _solver_record(polish)
    weights, kkt_polish = _phase_one_kkt_polish(
        weights=weights,
        columns=columns,
        geometry=geometry,
        max_iterations=int(max_iterations),
    )
    answer = weights @ matrix
    constraints = geometry.constraint_values(answer)
    objective_value = float(np.max(constraints))
    record = _solver_record(result)
    record.update(
        {
            "simplex_violation": abs(float(np.sum(weights)) - 1.0),
            "solver_reported_epigraph_violation": max(
                0.0, float(np.max(constraints - result.x[-1], initial=0.0))
            ),
            "epigraph_violation": max(
                0.0, float(np.max(constraints - objective_value, initial=0.0))
            ),
            "reported_epigraph": float(result.x[-1]),
            "recomputed_objective": objective_value,
            "single_active_polish": polish_record,
            "active_set_kkt_polish": kkt_polish,
            "certified_primal": bool(
                abs(float(np.sum(weights)) - 1.0) <= 1.0e-10
                and np.all(weights >= -1.0e-12)
            ),
        }
    )
    return weights, answer, objective_value, constraints, record


def _phase_two_master(
    columns: RHCGColumnSet,
    geometry: CanonicalConfidenceGeometry,
    prior_moments: np.ndarray,
    inflation: float,
    *,
    initial_weights: np.ndarray,
    max_iterations: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = columns.answers
    count = len(matrix)
    weights0 = _normalize_simplex(initial_weights)
    if weights0.shape != (count,):
        raise ValueError("Phase-II initial weights must match the column count")
    moments = np.asarray(prior_moments, dtype=np.float64)
    if moments.shape != (geometry.feature_map.feature_dimension,):
        raise ValueError("CCF prior moments must match the shadow feature map")
    unit_covariance = geometry.feature_map.unit_rho_covariance_diagonal()
    precision = np.divide(
        1.0,
        unit_covariance,
        out=np.zeros_like(unit_covariance),
        where=unit_covariance > 0.0,
    )

    initial_residual = weights0 @ matrix - moments
    objective_scale = max(
        1.0,
        float(0.5 * np.sum(initial_residual * initial_residual * precision)),
    )

    def objective(weights: np.ndarray) -> float:
        residual = weights @ matrix - moments
        return float(0.5 * np.sum(residual * residual * precision) / objective_scale)

    def objective_jacobian(weights: np.ndarray) -> np.ndarray:
        residual = weights @ matrix - moments
        return matrix @ (residual * precision) / objective_scale

    def inequalities(weights: np.ndarray) -> np.ndarray:
        return float(inflation) - geometry.constraint_values(weights @ matrix)

    def inequalities_jacobian(weights: np.ndarray) -> np.ndarray:
        gradients = geometry.constraint_gradients(weights @ matrix)
        return -np.asarray(gradients @ matrix.T, dtype=np.float64)

    initial_violation = float(
        np.max(geometry.constraint_values(weights0 @ matrix) - float(inflation))
    )
    if initial_violation > 1.0e-7:
        raise ValueError(
            f"Phase-II initial point violates inflated confidence by {initial_violation}"
        )
    result = minimize(
        objective,
        weights0,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=Bounds(np.zeros(count), np.ones(count)),
        constraints=(
            {
                "type": "eq",
                "fun": lambda weights: np.asarray([np.sum(weights) - 1.0]),
                "jac": lambda weights: np.ones((1, count), dtype=np.float64),
            },
            {
                "type": "ineq",
                "fun": inequalities,
                "jac": inequalities_jacobian,
            },
        ),
        options={"maxiter": int(max_iterations), "ftol": 1.0e-13, "disp": False},
    )
    weights = _normalize_simplex(result.x)
    raw_constraint_multipliers: np.ndarray | None = None
    reported_multipliers = getattr(result, "multipliers", None)
    if reported_multipliers is not None:
        values = np.asarray(reported_multipliers, dtype=np.float64).reshape(-1)
        if len(values) == 1 + geometry.constraint_count:
            raw_constraint_multipliers = (
                np.maximum(values[1:], 0.0) * objective_scale
            )
    weights, kkt_polish = _phase_two_kkt_polish(
        weights=weights,
        columns=columns,
        geometry=geometry,
        prior_moments=moments,
        inflation=float(inflation),
        precision=precision,
        initial_constraint_multipliers=raw_constraint_multipliers,
        max_iterations=int(max_iterations),
    )
    answer = weights @ matrix
    residual = answer - moments
    objective_value = float(0.5 * np.sum(residual * residual * precision))
    objective_gradient = residual * precision
    constraints = geometry.constraint_values(answer)
    record = _solver_record(result)
    record.update(
        {
            "simplex_violation": abs(float(np.sum(weights)) - 1.0),
            "confidence_violation": max(
                0.0, float(np.max(constraints - float(inflation), initial=0.0))
            ),
            "recomputed_objective": objective_value,
            "objective_scale": objective_scale,
            "active_set_kkt_polish": kkt_polish,
            "certified_primal": bool(
                abs(float(np.sum(weights)) - 1.0) <= 1.0e-10
                and np.all(weights >= -1.0e-12)
                and float(np.max(constraints - float(inflation), initial=0.0))
                <= 1.0e-8
            ),
        }
    )
    return (
        weights,
        answer,
        objective_value,
        constraints,
        objective_gradient,
        record,
    )


def _minimum_norm_dual(
    *,
    columns: RHCGColumnSet,
    weights: np.ndarray,
    constraint_values: np.ndarray,
    constraint_gradients: np.ndarray,
    face_value: float,
    base_gradient: np.ndarray,
    phase_one: bool,
) -> MinimumNormGlobalDual:
    matrix = columns.answers
    values = np.asarray(constraint_values, dtype=np.float64)
    gradients = (
        constraint_gradients
        if issparse(constraint_gradients)
        else np.asarray(constraint_gradients, dtype=np.float64)
    )
    base = np.asarray(base_gradient, dtype=np.float64)
    if gradients.shape != (len(values), matrix.shape[1]) or base.shape != (
        matrix.shape[1],
    ):
        raise ValueError("Minimum-norm dual inputs have incompatible dimensions")
    slack = float(face_value) - values
    active = np.flatnonzero(slack <= 2.0e-8)
    support = np.flatnonzero(np.asarray(weights) > 1.0e-7)
    if len(support) == 0:
        support = np.asarray([int(np.argmax(weights))], dtype=np.int64)
    pivot = int(support[0])
    base_energy = matrix @ base
    if len(active) == 0:
        total_energy = base_energy
        support_residual = float(
            np.max(np.abs(total_energy[support] - total_energy[pivot]), initial=0.0)
        )
        reduced_violation = max(
            0.0, float(total_energy[pivot] - np.min(total_energy, initial=math.inf))
        )
        certified = (
            not phase_one
            and support_residual <= 1.0e-7
            and reduced_violation <= 1.0e-7
        )
        return MinimumNormGlobalDual(
            multipliers=np.zeros(len(values), dtype=np.float64),
            active_constraints=(),
            stationarity_residual=support_residual,
            complementarity_residual=0.0,
            dual_negativity=0.0,
            restricted_reduced_cost_violation=reduced_violation,
            certified=certified,
            solver={"success": certified, "message": "zero_active_dual"},
        )
    active_energies = np.asarray(
        gradients[active] @ matrix.T,
        dtype=np.float64,
    )
    variable_count = len(active)

    if phase_one and variable_count == 1:
        multipliers = np.zeros(len(values), dtype=np.float64)
        multipliers[active[0]] = 1.0
        total_energy = base_energy + active_energies[0]
        support_residual = float(
            np.max(
                np.abs(total_energy[support] - total_energy[pivot]),
                initial=0.0,
            )
        )
        reduced_violation = max(
            0.0,
            float(total_energy[pivot] - np.min(total_energy, initial=math.inf)),
        )
        complementarity = float(
            np.max(np.abs(multipliers * slack), initial=0.0)
        )
        certified = bool(
            support_residual <= 1.0e-7
            and reduced_violation <= 1.0e-7
            and complementarity <= 1.0e-7
        )
        return MinimumNormGlobalDual(
            multipliers=multipliers,
            active_constraints=(int(active[0]),),
            stationarity_residual=support_residual,
            complementarity_residual=complementarity,
            dual_negativity=0.0,
            restricted_reduced_cost_violation=reduced_violation,
            certified=certified,
            solver={
                "success": certified,
                "message": "unique_active_phase_one_multiplier",
            },
        )

    def objective(local: np.ndarray) -> float:
        return float(0.5 * np.sum(local * local))

    def jacobian(local: np.ndarray) -> np.ndarray:
        return np.asarray(local, dtype=np.float64)

    constraints: list[Any] = []
    equality_rows: list[np.ndarray] = []
    equality_targets: list[np.ndarray] = []
    if phase_one:
        equality_rows.append(np.ones((1, variable_count), dtype=np.float64))
        equality_targets.append(np.ones(1, dtype=np.float64))
    if len(support) > 1:
        stationarity_rows = (
            active_energies[:, support[1:]].T
            - active_energies[:, pivot].reshape(1, -1)
        )
        stationarity_rhs = -(
            base_energy[support[1:]] - base_energy[pivot]
        )
        equality_rows.append(stationarity_rows)
        equality_targets.append(stationarity_rhs)
    equality_inconsistency = 0.0
    if equality_rows:
        compressed_rows, compressed_rhs, equality_inconsistency = (
            _compress_equalities(
                np.concatenate(equality_rows, axis=0),
                np.concatenate(equality_targets),
            )
        )
        if len(compressed_rows):
            constraints.append(
                LinearConstraint(
                    compressed_rows,
                    compressed_rhs,
                    compressed_rhs,
                )
            )
    reduced_rows = (
        active_energies.T - active_energies[:, pivot].reshape(1, -1)
    )
    reduced_rhs = -(base_energy - base_energy[pivot])
    constraints.append(LinearConstraint(reduced_rows, reduced_rhs, np.inf))
    initial = (
        np.full(variable_count, 1.0 / variable_count, dtype=np.float64)
        if phase_one
        else np.zeros(variable_count, dtype=np.float64)
    )
    result = minimize(
        objective,
        initial,
        jac=jacobian,
        method="SLSQP",
        bounds=Bounds(np.zeros(variable_count), np.full(variable_count, np.inf)),
        constraints=tuple(constraints),
        options={"maxiter": 5_000, "ftol": 1.0e-14, "disp": False},
    )
    local = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    if phase_one and float(np.sum(local)) > 0.0:
        local /= float(np.sum(local))
    multipliers = np.zeros(len(values), dtype=np.float64)
    multipliers[active] = local
    total_energy = base_energy + active_energies.T @ local
    support_residual = float(
        np.max(np.abs(total_energy[support] - total_energy[pivot]), initial=0.0)
    )
    reduced_violation = max(
        0.0, float(total_energy[pivot] - np.min(total_energy, initial=math.inf))
    )
    complementarity = float(
        np.max(np.abs(multipliers * slack), initial=0.0)
    )
    negativity = max(0.0, -float(np.min(multipliers, initial=0.0)))
    certified = bool(
        support_residual <= 1.0e-7
        and reduced_violation <= 1.0e-7
        and complementarity <= 1.0e-7
        and negativity <= 1.0e-10
        and equality_inconsistency <= 1.0e-7
        and (not phase_one or abs(float(np.sum(multipliers)) - 1.0) <= 1.0e-8)
    )
    solver_record = _solver_record(result)
    solver_record["equality_inconsistency"] = equality_inconsistency
    solver_record["compressed_equality_count"] = int(
        len(compressed_rows) if equality_rows else 0
    )
    return MinimumNormGlobalDual(
        multipliers=multipliers,
        active_constraints=tuple(int(value) for value in active),
        stationarity_residual=support_residual,
        complementarity_residual=complementarity,
        dual_negativity=negativity,
        restricted_reduced_cost_violation=reduced_violation,
        certified=certified,
        solver=solver_record,
    )


def _price_global_gradient(
    *,
    feature_map: ShadowFeatureMap,
    current_answer: np.ndarray,
    gradient: np.ndarray,
    solver: Literal["auto", "enumerate", "milp"],
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[RowPricingResult, float]:
    atom_scale = float(feature_map.domain.public_n)
    pricing = price_legal_row(
        feature_map,
        atom_scale * np.asarray(gradient, dtype=np.float64),
        solver=solver,
        progress_callback=progress_callback,
    )
    minimum_vertex_energy = pricing.objective
    current_energy = float(np.asarray(current_answer, dtype=np.float64) @ gradient)
    raw_gap = max(0.0, current_energy - minimum_vertex_energy)
    certified_gap = raw_gap + pricing.absolute_gap
    return pricing, certified_gap


def _weighted_constraint_gradient(
    geometry: CanonicalConfidenceGeometry,
    answer: np.ndarray,
    multipliers: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        geometry.constraint_gradients(answer).T
        @ np.asarray(multipliers, dtype=np.float64),
        dtype=np.float64,
    ).reshape(-1)


def solve_phase_one_rhcg(
    geometry: CanonicalConfidenceGeometry,
    columns: RHCGColumnSet,
    *,
    pricing_solver: Literal["auto", "enumerate", "milp"] = "auto",
    global_gap_tolerance: float = 1.0e-8,
    maximum_columns: int | None = None,
    max_iterations: int = 5_000,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> RHCGPhaseResult:
    if columns.feature_hash != geometry.feature_map.feature_hash:
        raise ValueError("Phase-I columns do not match the feature map")
    cap = int(maximum_columns or geometry.feature_map.atom_cap)
    theorem_qualified_cap = bool(
        cap == geometry.feature_map.affine_rank + 1 and cap <= 4096
    )
    transient_cap = cap + 1 if theorem_qualified_cap else cap
    current = columns
    initial_weights: np.ndarray | None = None
    initial_atom_count = current.atom_count
    last: tuple[Any, ...] | None = None
    for iteration in range(max(100, 4 * cap + 10)):
        if progress_callback is not None:
            progress_callback(
                "phase_one_master_started",
                {
                    "iteration": iteration,
                    "column_count": current.column_count,
                    "atom_count": current.atom_count,
                },
            )
        master_started = time.perf_counter()
        weights, answer, objective, values, master = _phase_one_master(
            current,
            geometry,
            initial_weights=initial_weights,
            max_iterations=int(max_iterations),
        )
        master_seconds = time.perf_counter() - master_started
        current, weights, reduction = _caratheodory_reduce(
            current,
            weights,
            maximum_columns=cap,
        )
        if reduction["applied"]:
            answer = weights @ current.answers
            values = geometry.constraint_values(answer)
            objective = float(np.max(values))
            master = {**master, "caratheodory_reduction": reduction}
        dual = _minimum_norm_dual(
            columns=current,
            weights=weights,
            constraint_values=values,
            constraint_gradients=geometry.constraint_gradients(answer),
            face_value=objective,
            base_gradient=np.zeros(geometry.feature_map.feature_dimension),
            phase_one=True,
        )
        gradient = _weighted_constraint_gradient(geometry, answer, dual.multipliers)
        if progress_callback is not None:
            progress_callback(
                "phase_one_pricing_started",
                {
                    "iteration": iteration,
                    "column_count": current.column_count,
                    "atom_count": current.atom_count,
                    "objective": objective,
                    "inflation": max(0.0, objective),
                    "dual_certified": dual.certified,
                    "active_constraint_count": len(dual.active_constraints),
                    "active_constraint_indices": list(dual.active_constraints),
                    "master_seconds": master_seconds,
                },
            )
        pricing_started = time.perf_counter()
        pricing, gap = _price_global_gradient(
            feature_map=geometry.feature_map,
            current_answer=answer,
            gradient=gradient,
            solver=pricing_solver,
            progress_callback=(
                None
                if progress_callback is None
                else lambda stage, values: progress_callback(
                    f"phase_one_pricing_{stage}",
                    {"iteration": iteration, **values},
                )
            ),
        )
        pricing_seconds = time.perf_counter() - pricing_started
        if progress_callback is not None:
            progress_callback(
                "phase_one_iteration",
                {
                    "iteration": iteration,
                    "column_count": current.column_count,
                    "atom_count": current.atom_count,
                    "objective": objective,
                    "inflation": max(0.0, objective),
                    "global_gap": gap,
                    "dual_certified": dual.certified,
                    "pricing_certified": pricing.certified,
                    "priced_row": pricing.row.tolist(),
                    "master_seconds": master_seconds,
                    "pricing_seconds": pricing_seconds,
                    "caratheodory_reduction": reduction,
                },
            )
        last = (weights, answer, objective, values, master, dual, pricing, gap, iteration)
        if dual.certified and pricing.certified and gap <= float(global_gap_tolerance):
            return RHCGPhaseResult(
                phase=PHASE_ONE_METHOD,
                objective=objective,
                inflation=max(0.0, objective),
                weights=weights,
                answer=answer,
                constraint_values=values,
                columns=current,
                dual=dual,
                pricing=pricing,
                global_gap=gap,
                certified=bool(
                    master["certified_primal"]
                    and master["epigraph_violation"] <= 1.0e-8
                ),
                iterations=iteration + 1,
                atoms_added=current.atom_count - initial_atom_count,
                master=master,
            )
        if gap <= float(global_gap_tolerance) and not dual.certified:
            raise RuntimeError("Phase-I master reached a small price gap without a dual certificate")
        if current.contains_row(pricing.row):
            raise RuntimeError("Phase-I pricing returned a duplicate row with positive global gap")
        previous_count = current.column_count
        current = current.add_row(
            geometry.feature_map,
            pricing.row,
            maximum_columns=transient_cap,
        )
        initial_weights = np.concatenate((weights, np.zeros(current.column_count - previous_count)))
    if last is None:  # pragma: no cover
        raise RuntimeError("Phase-I RHCG did not execute")
    raise RuntimeError("Phase-I RHCG exhausted its public column cap")


def solve_phase_two_rhcg(
    geometry: CanonicalConfidenceGeometry,
    phase_one: RHCGPhaseResult,
    prior_moments: np.ndarray,
    *,
    pricing_solver: Literal["auto", "enumerate", "milp"] = "auto",
    global_gap_tolerance: float = 1.0e-8,
    maximum_columns: int | None = None,
    max_iterations: int = 5_000,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> RHCGPhaseResult:
    if not phase_one.certified:
        raise ValueError("Phase II requires a certified Phase-I result")
    cap = int(maximum_columns or geometry.feature_map.atom_cap)
    theorem_qualified_cap = bool(
        cap == geometry.feature_map.affine_rank + 1 and cap <= 4096
    )
    transient_cap = cap + 1 if theorem_qualified_cap else cap
    current = phase_one.columns
    initial_weights = np.asarray(phase_one.weights, dtype=np.float64)
    initial_atom_count = current.atom_count
    for iteration in range(max(100, 4 * cap + 10)):
        if progress_callback is not None:
            progress_callback(
                "phase_two_master_started",
                {
                    "iteration": iteration,
                    "column_count": current.column_count,
                    "atom_count": current.atom_count,
                },
            )
        master_started = time.perf_counter()
        (
            weights,
            answer,
            objective,
            values,
            objective_gradient,
            master,
        ) = _phase_two_master(
            current,
            geometry,
            prior_moments,
            phase_one.inflation,
            initial_weights=initial_weights,
            max_iterations=int(max_iterations),
        )
        master_seconds = time.perf_counter() - master_started
        current, weights, reduction = _caratheodory_reduce(
            current,
            weights,
            maximum_columns=cap,
        )
        if reduction["applied"]:
            answer = weights @ current.answers
            values = geometry.constraint_values(answer)
            unit_covariance = geometry.feature_map.unit_rho_covariance_diagonal()
            precision = np.divide(
                1.0,
                unit_covariance,
                out=np.zeros_like(unit_covariance),
                where=unit_covariance > 0.0,
            )
            residual = answer - np.asarray(prior_moments, dtype=np.float64)
            objective = float(0.5 * np.sum(residual * residual * precision))
            objective_gradient = residual * precision
            master = {**master, "caratheodory_reduction": reduction}
        dual = _minimum_norm_dual(
            columns=current,
            weights=weights,
            constraint_values=values,
            constraint_gradients=geometry.constraint_gradients(answer),
            face_value=phase_one.inflation,
            base_gradient=objective_gradient,
            phase_one=False,
        )
        gradient = objective_gradient + _weighted_constraint_gradient(
            geometry,
            answer,
            dual.multipliers,
        )
        if progress_callback is not None:
            progress_callback(
                "phase_two_pricing_started",
                {
                    "iteration": iteration,
                    "column_count": current.column_count,
                    "atom_count": current.atom_count,
                    "objective": objective,
                    "dual_certified": dual.certified,
                    "active_constraint_count": len(dual.active_constraints),
                    "active_constraint_indices": list(dual.active_constraints),
                    "master_seconds": master_seconds,
                },
            )
        pricing_started = time.perf_counter()
        pricing, gap = _price_global_gradient(
            feature_map=geometry.feature_map,
            current_answer=answer,
            gradient=gradient,
            solver=pricing_solver,
            progress_callback=(
                None
                if progress_callback is None
                else lambda stage, values: progress_callback(
                    f"phase_two_pricing_{stage}",
                    {"iteration": iteration, **values},
                )
            ),
        )
        pricing_seconds = time.perf_counter() - pricing_started
        if progress_callback is not None:
            progress_callback(
                "phase_two_iteration",
                {
                    "iteration": iteration,
                    "column_count": current.column_count,
                    "atom_count": current.atom_count,
                    "objective": objective,
                    "inflation": phase_one.inflation,
                    "global_gap": gap,
                    "dual_certified": dual.certified,
                    "pricing_certified": pricing.certified,
                    "priced_row": pricing.row.tolist(),
                    "master_seconds": master_seconds,
                    "pricing_seconds": pricing_seconds,
                    "caratheodory_reduction": reduction,
                },
            )
        if dual.certified and pricing.certified and gap <= float(global_gap_tolerance):
            return RHCGPhaseResult(
                phase=PHASE_TWO_METHOD,
                objective=objective,
                inflation=phase_one.inflation,
                weights=weights,
                answer=answer,
                constraint_values=values,
                columns=current,
                dual=dual,
                pricing=pricing,
                global_gap=gap,
                certified=bool(
                    master["certified_primal"]
                    and master["confidence_violation"] <= 1.0e-8
                ),
                iterations=iteration + 1,
                atoms_added=current.atom_count - initial_atom_count,
                master=master,
            )
        if gap <= float(global_gap_tolerance) and not dual.certified:
            raise RuntimeError("Phase-II master reached a small price gap without a dual certificate")
        if current.contains_row(pricing.row):
            raise RuntimeError("Phase-II pricing returned a duplicate row with positive global gap")
        previous_count = current.column_count
        current = current.add_row(
            geometry.feature_map,
            pricing.row,
            maximum_columns=transient_cap,
        )
        initial_weights = np.concatenate((weights, np.zeros(current.column_count - previous_count)))
    raise RuntimeError("Phase-II RHCG exhausted its public column cap")


@dataclass(frozen=True)
class RHCGCCMPResult:
    phase_one: RHCGPhaseResult
    phase_two: RHCGPhaseResult
    pressure_source: str
    pressure_by_block: dict[str, float]
    certified: bool
    fallback_reasons: tuple[str, ...]
    method: str = RHCG_CCMP_METHOD

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "certified": self.certified,
            "fallback_reasons": list(self.fallback_reasons),
            "pressure_source": self.pressure_source,
            "pressure_by_block": dict(self.pressure_by_block),
            "phase_one": self.phase_one.to_public_dict(),
            "phase_two": self.phase_two.to_public_dict(),
            "positive_inflation_is_fallback": False,
        }


def solve_rhcg_ccmp(
    geometry: CanonicalConfidenceGeometry,
    columns: RHCGColumnSet,
    prior_moments: np.ndarray,
    *,
    pricing_solver: Literal["auto", "enumerate", "milp"] = "auto",
    global_gap_tolerance: float = 1.0e-8,
    maximum_columns: int | None = None,
    max_iterations: int = 5_000,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> RHCGCCMPResult:
    phase_one = solve_phase_one_rhcg(
        geometry,
        columns,
        pricing_solver=pricing_solver,
        global_gap_tolerance=global_gap_tolerance,
        maximum_columns=maximum_columns,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
    )
    phase_two = solve_phase_two_rhcg(
        geometry,
        phase_one,
        prior_moments,
        pricing_solver=pricing_solver,
        global_gap_tolerance=global_gap_tolerance,
        maximum_columns=maximum_columns,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
    )
    if phase_one.objective > 0.0:
        source = "phase_one_global_margin_dual"
        selected = phase_one
    else:
        source = "phase_two_ccf_moment_projection_dual"
        selected = phase_two
    pressure = geometry.block_pressures(selected.answer, selected.dual.multipliers)
    reasons: list[str] = []
    if not phase_one.certified:
        reasons.append("phase_one_not_certified")
    if not phase_two.certified:
        reasons.append("phase_two_not_certified")
    if not all(math.isfinite(value) and value >= 0.0 for value in pressure.values()):
        reasons.append("invalid_block_pressure")
    return RHCGCCMPResult(
        phase_one=phase_one,
        phase_two=phase_two,
        pressure_source=source,
        pressure_by_block=pressure,
        certified=not reasons,
        fallback_reasons=tuple(reasons),
    )


def solve_enumerated_phase_one(
    geometry: CanonicalConfidenceGeometry,
    *,
    maximum_rows: int = 100_000,
) -> RHCGPhaseResult:
    rows = geometry.feature_map.domain.enumerate_rows(maximum_rows=maximum_rows)
    columns = RHCGColumnSet.create(geometry.feature_map, initial_rows=rows)
    weights, answer, objective, values, master = _phase_one_master(columns, geometry)
    dual = _minimum_norm_dual(
        columns=columns,
        weights=weights,
        constraint_values=values,
        constraint_gradients=geometry.constraint_gradients(answer),
        face_value=objective,
        base_gradient=np.zeros(geometry.feature_map.feature_dimension),
        phase_one=True,
    )
    gradient = _weighted_constraint_gradient(geometry, answer, dual.multipliers)
    pricing, gap = _price_global_gradient(
        feature_map=geometry.feature_map,
        current_answer=answer,
        gradient=gradient,
        solver="enumerate",
    )
    return RHCGPhaseResult(
        phase=PHASE_ONE_METHOD,
        objective=objective,
        inflation=max(0.0, objective),
        weights=weights,
        answer=answer,
        constraint_values=values,
        columns=columns,
        dual=dual,
        pricing=pricing,
        global_gap=gap,
        certified=bool(
            master["certified_primal"]
            and master["epigraph_violation"] <= 1.0e-8
            and dual.certified
            and pricing.certified
            and gap <= 1.0e-8
        ),
        iterations=1,
        atoms_added=0,
        master=master,
    )


def solve_enumerated_phase_two(
    geometry: CanonicalConfidenceGeometry,
    phase_one: RHCGPhaseResult,
    prior_moments: np.ndarray,
) -> RHCGPhaseResult:
    (
        weights,
        answer,
        objective,
        values,
        objective_gradient,
        master,
    ) = _phase_two_master(
        phase_one.columns,
        geometry,
        prior_moments,
        phase_one.inflation,
        initial_weights=phase_one.weights,
    )
    dual = _minimum_norm_dual(
        columns=phase_one.columns,
        weights=weights,
        constraint_values=values,
        constraint_gradients=geometry.constraint_gradients(answer),
        face_value=phase_one.inflation,
        base_gradient=objective_gradient,
        phase_one=False,
    )
    gradient = objective_gradient + _weighted_constraint_gradient(
        geometry,
        answer,
        dual.multipliers,
    )
    pricing, gap = _price_global_gradient(
        feature_map=geometry.feature_map,
        current_answer=answer,
        gradient=gradient,
        solver="enumerate",
    )
    return RHCGPhaseResult(
        phase=PHASE_TWO_METHOD,
        objective=objective,
        inflation=phase_one.inflation,
        weights=weights,
        answer=answer,
        constraint_values=values,
        columns=phase_one.columns,
        dual=dual,
        pricing=pricing,
        global_gap=gap,
        certified=bool(
            master["certified_primal"]
            and master["confidence_violation"] <= 1.0e-8
            and dual.certified
            and pricing.certified
            and gap <= 1.0e-8
        ),
        iterations=1,
        atoms_added=0,
        master=master,
    )


__all__ = [
    "PHASE_ONE_METHOD",
    "PHASE_TWO_METHOD",
    "RHCG_CCMP_METHOD",
    "CanonicalConfidenceGeometry",
    "MinimumNormGlobalDual",
    "RHCGCCMPResult",
    "RHCGColumnSet",
    "RHCGPhaseResult",
    "solve_enumerated_phase_one",
    "solve_enumerated_phase_two",
    "solve_phase_one_rhcg",
    "solve_phase_two_rhcg",
    "solve_rhcg_ccmp",
]
