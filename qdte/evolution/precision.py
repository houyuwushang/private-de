from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.measurement.covariance import (
    ActiveSetCoefficientCovariance,
    CoefficientBootstrapDiagonal,
)
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.shrinkage import InteractionShrinkageResult
from qdte.queries.orthogonal import helmert_contrast
from qdte.queries.types import OP_EQ, QueryCatalogue
from qdte.queries.workload import WorkloadGroup


class PrecisionOperator(ABC):
    """Positive-semidefinite precision used by a quadratic QDTE objective."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def matvec(self, values: np.ndarray) -> np.ndarray:
        """Return ``W @ values`` in float64."""
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]:
        raise NotImplementedError

    def _vector(self, values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.dimension,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite vector with shape ({self.dimension},)")
        return array

    def _matrix(self, values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.dimension or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must be a finite matrix with shape (n, {self.dimension})"
            )
        return array

    def loss(self, residual: np.ndarray) -> float:
        r = self._vector(residual, name="residual")
        return 0.5 * float(r @ self.matvec(r))

    def quad(self, delta: np.ndarray) -> float:
        d = self._vector(delta, name="delta")
        value = float(d @ self.matvec(d))
        tolerance = 1.0e-10 * max(1.0, float(d @ d))
        if value < -tolerance:
            raise RuntimeError(f"Precision operator produced a negative quadratic form: {value}")
        return max(0.0, value)

    def matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        return np.stack([self.matvec(row) for row in matrix], axis=0)

    def quad_many(self, deltas: np.ndarray) -> np.ndarray:
        matrix = self._matrix(deltas, name="deltas")
        transformed = self.matvec_many(matrix)
        values = np.einsum("ij,ij->i", matrix, transformed, optimize=True)
        tolerance = 1.0e-10 * np.maximum(1.0, np.einsum("ij,ij->i", matrix, matrix))
        if np.any(values < -tolerance):
            raise RuntimeError("Precision operator produced a negative batch quadratic form")
        return np.maximum(values, 0.0)

    def advantages(
        self,
        residual: np.ndarray,
        deltas: np.ndarray,
        edit_cost: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        r = self._vector(residual, name="residual")
        d = self._matrix(deltas, name="deltas")
        costs = np.asarray(edit_cost, dtype=np.float64)
        if costs.shape != (d.shape[0],) or not np.all(np.isfinite(costs)):
            raise ValueError(f"edit_cost must be finite with shape ({d.shape[0]},)")
        if not np.isfinite(lambda_cost) or float(lambda_cost) < 0.0:
            raise ValueError("lambda_cost must be finite and nonnegative")
        linear = d @ self.matvec(r)
        return linear - 0.5 * self.quad_many(d) - float(lambda_cost) * costs

    def advantage(
        self,
        residual: np.ndarray,
        delta: np.ndarray,
        *,
        edit_cost: float = 0.0,
        lambda_cost: float = 0.0,
    ) -> float:
        d = self._vector(delta, name="delta")
        value = self.advantages(
            residual,
            d.reshape(1, -1),
            np.asarray([edit_cost], dtype=np.float64),
            lambda_cost,
        )
        return float(value[0])


class DiagonalPrecision(PrecisionOperator):
    """The frozen QDTE Base precision ``diag(inv_variance)``."""

    def __init__(self, inv_variance: np.ndarray):
        weights = np.asarray(inv_variance, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("inv_variance must be a non-empty vector")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("inv_variance must be finite and nonnegative")
        self._weights = weights.copy()
        self._weights.setflags(write=False)

    @property
    def dimension(self) -> int:
        return int(self._weights.size)

    @property
    def inv_variance(self) -> np.ndarray:
        return self._weights

    def matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        return vector * self._weights

    def matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        return matrix * self._weights.reshape(1, -1)

    def quad_many(self, deltas: np.ndarray) -> np.ndarray:
        matrix = self._matrix(deltas, name="deltas")
        return (matrix * matrix) @ self._weights

    def diagnostics(self) -> dict[str, Any]:
        positive = self._weights > 0.0
        return {
            "kind": "diagonal",
            "dimension": self.dimension,
            "effective_rank": int(np.sum(positive)),
            "nullity": int(np.sum(~positive)),
            "min_positive_precision": (
                float(np.min(self._weights[positive])) if np.any(positive) else 0.0
            ),
            "max_precision": float(np.max(self._weights)),
        }


@dataclass(frozen=True)
class _OrthogonalBlock:
    name: str
    kind: str
    scope: tuple[int, ...]
    query_indices: np.ndarray
    shape: tuple[int, ...]
    variance: float
    left_contrast: np.ndarray
    right_contrast: np.ndarray | None = None

    @property
    def rank(self) -> int:
        if self.kind == "oneway_contrast":
            return int(self.shape[0] - 1)
        return int((self.shape[0] - 1) * (self.shape[1] - 1))


def _parse_complete_partition_scope(group: WorkloadGroup) -> tuple[int, ...]:
    parts = str(group.name).split(":")
    if group.family == "oneway" and len(parts) == 2:
        return (int(parts[1]),)
    if group.family == "twoway" and len(parts) == 3:
        left, right = int(parts[1]), int(parts[2])
        if left >= right:
            raise ValueError(f"Pair group must use canonical attribute order: {group.name!r}")
        return (left, right)
    raise ValueError(
        "Orthogonal interaction precision requires groups named "
        "oneway:<attr> or twoway:<left>:<right>"
    )


def _validate_complete_partition_order(
    qcat: QueryCatalogue,
    group: WorkloadGroup,
    scope: tuple[int, ...],
    cardinalities: tuple[int, ...],
) -> tuple[int, ...]:
    indices = np.asarray(group.query_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"Group {group.name!r} must contain a non-empty query vector")
    if np.any(indices < 0) or np.any(indices >= qcat.m):
        raise ValueError(f"Group {group.name!r} contains out-of-range query indices")
    shape = tuple(int(cardinalities[attr]) for attr in scope)
    if indices.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(f"Group {group.name!r} does not contain its complete partition")
    expected_values = np.stack(
        np.unravel_index(np.arange(indices.size, dtype=np.int64), shape),
        axis=1,
    )
    for row, qid in enumerate(indices.tolist()):
        if int(qcat.num_terms[qid]) != len(scope) or int(qcat.linear_num_terms[qid]) != 0:
            raise ValueError(f"Query {qid} in {group.name!r} is not a pure equality cell")
        if tuple(int(value) for value in qcat.attrs[qid, : len(scope)]) != scope:
            raise ValueError(f"Query {qid} in {group.name!r} has a mismatched scope")
        if np.any(qcat.ops[qid, : len(scope)] != OP_EQ):
            raise ValueError(f"Query {qid} in {group.name!r} is not an equality cell")
        if not np.array_equal(qcat.values[qid, : len(scope)], expected_values[row]):
            raise ValueError(f"Query ordering in {group.name!r} is not canonical row-major order")
    return shape


class OrthogonalInteractionPrecision(PrecisionOperator):
    """Exact raw-coefficient likelihood represented in complete-cell coordinates."""

    def __init__(
        self,
        qcat: QueryCatalogue,
        workload_groups: list[WorkloadGroup],
        transcript: HierarchicalInteractionTranscript,
    ):
        qcat.validate(np.asarray(transcript.strategy.cardinalities, dtype=np.int64))
        cardinalities = tuple(int(value) for value in transcript.strategy.cardinalities)
        coverage = np.zeros(qcat.m, dtype=np.int32)
        blocks: list[_OrthogonalBlock] = []
        expected_names = {block.name for block in transcript.strategy.blocks}

        for group in workload_groups:
            if not group.is_partition:
                raise ValueError("Orthogonal interaction precision supports complete partitions only")
            scope = _parse_complete_partition_scope(group)
            shape = _validate_complete_partition_order(qcat, group, scope, cardinalities)
            indices = np.asarray(group.query_indices, dtype=np.int32).copy()
            coverage[indices] += 1
            if len(scope) == 1:
                name = f"oneway_contrast:{scope[0]}"
                kind = "oneway_contrast"
                left = helmert_contrast(shape[0])
                right = None
            else:
                name = f"pair_interaction:{scope[0]}:{scope[1]}"
                kind = "pair_interaction"
                left = helmert_contrast(shape[0])
                right = helmert_contrast(shape[1])
            if name not in expected_names or name not in transcript.component_variances:
                raise ValueError(f"Transcript is missing precision block {name!r}")
            variance = float(transcript.component_variances[name])
            if not np.isfinite(variance) or variance <= 0.0:
                raise ValueError(f"Transcript block {name!r} must have positive finite variance")
            indices.setflags(write=False)
            left.setflags(write=False)
            if right is not None:
                right.setflags(write=False)
            blocks.append(
                _OrthogonalBlock(
                    name=name,
                    kind=kind,
                    scope=scope,
                    query_indices=indices,
                    shape=shape,
                    variance=variance,
                    left_contrast=left,
                    right_contrast=right,
                )
            )

        if not np.all(coverage == 1):
            raise ValueError("Orthogonal precision groups must cover every query exactly once")
        actual_names = {block.name for block in blocks}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(f"Workload/transcript block mismatch: missing={missing}, extra={extra}")
        self._dimension = int(qcat.m)
        self._blocks = tuple(blocks)
        self._cardinalities = cardinalities
        self._coefficient_variances = np.concatenate(
            [np.full(block.rank, block.variance, dtype=np.float64) for block in self._blocks]
        )
        self._coefficient_variances.setflags(write=False)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def blocks(self) -> tuple[_OrthogonalBlock, ...]:
        return self._blocks

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return self._cardinalities

    @property
    def coefficient_dimension(self) -> int:
        return int(self._coefficient_variances.size)

    @property
    def coefficient_variances(self) -> np.ndarray:
        return self._coefficient_variances

    def coefficient_coordinates(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        coordinates: list[np.ndarray] = []
        for block in self._blocks:
            local = vector[block.query_indices].reshape(block.shape)
            left = block.left_contrast
            if block.kind == "oneway_contrast":
                transformed = left.T @ local
            else:
                right = block.right_contrast
                if right is None:
                    raise RuntimeError("Pair precision block is missing its right contrast")
                transformed = left.T @ local @ right
            coordinates.append(np.asarray(transformed, dtype=np.float64).reshape(-1))
        return np.concatenate(coordinates)

    def coefficient_coordinates_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        coordinates: list[np.ndarray] = []
        for block in self._blocks:
            local = matrix[:, block.query_indices].reshape((matrix.shape[0], *block.shape))
            if block.kind == "oneway_contrast":
                transformed = np.einsum(
                    "dc,nd->nc",
                    block.left_contrast,
                    local,
                    optimize=True,
                )
            else:
                right = block.right_contrast
                if right is None:
                    raise RuntimeError("Pair precision block is missing its right contrast")
                transformed = np.einsum(
                    "ia,nij,jb->nab",
                    block.left_contrast,
                    local,
                    right,
                    optimize=True,
                )
            coordinates.append(np.asarray(transformed, dtype=np.float64).reshape(matrix.shape[0], -1))
        return np.concatenate(coordinates, axis=1)

    def coefficient_adjoint(self, coefficient_values: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficient_values, dtype=np.float64)
        if (
            coefficients.shape != (self.coefficient_dimension,)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError(
                f"coefficient_values must be finite with shape ({self.coefficient_dimension},)"
            )
        output = np.zeros(self.dimension, dtype=np.float64)
        offset = 0
        for block in self._blocks:
            end = offset + block.rank
            local = coefficients[offset:end]
            if block.kind == "oneway_contrast":
                transformed = block.left_contrast @ local
            else:
                right = block.right_contrast
                if right is None:
                    raise RuntimeError("Pair precision block is missing its right contrast")
                transformed = (
                    block.left_contrast
                    @ local.reshape(block.left_contrast.shape[1], right.shape[1])
                    @ right.T
                )
            output[block.query_indices] = np.asarray(transformed).reshape(-1)
            offset = end
        return output

    def weighted_coefficient_residual(self, residual: np.ndarray) -> np.ndarray:
        return self.coefficient_coordinates(residual) / self._coefficient_variances

    def interaction_cell_potentials(
        self,
        residual: np.ndarray,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Return exact ``W @ residual`` pair potentials in cell coordinates."""
        weighted = self.matvec(residual)
        result: dict[tuple[int, int], np.ndarray] = {}
        for block in self._blocks:
            if block.kind != "pair_interaction":
                continue
            if len(block.scope) != 2:
                raise RuntimeError("Pair interaction block has an invalid scope")
            potential = np.asarray(
                weighted[block.query_indices].reshape(block.shape),
                dtype=np.float64,
            ).copy()
            potential.setflags(write=False)
            result[(int(block.scope[0]), int(block.scope[1]))] = potential
        return result

    def block_precision_parameters(self, name: str) -> tuple[float, float, np.ndarray]:
        for block in self._blocks:
            if block.name == name:
                return (
                    1.0 / block.variance,
                    0.0,
                    np.zeros(block.rank, dtype=np.float64),
                )
        raise KeyError(name)

    def row_features(self, rows: np.ndarray) -> np.ndarray:
        records = np.asarray(rows, dtype=np.int32)
        if records.ndim != 2 or records.shape[1] != len(self._cardinalities):
            raise ValueError(
                f"rows must have shape (n, {len(self._cardinalities)})"
            )
        for attr, cardinality in enumerate(self._cardinalities):
            if np.any(records[:, attr] < 0) or np.any(records[:, attr] >= cardinality):
                raise ValueError(f"rows contain out-of-range values for attribute {attr}")
        features = np.empty(
            (records.shape[0], self.coefficient_dimension),
            dtype=np.float64,
        )
        offset = 0
        for block in self._blocks:
            left_values = block.left_contrast[records[:, block.scope[0]]]
            if block.kind == "oneway_contrast":
                local = left_values
            else:
                right = block.right_contrast
                if right is None:
                    raise RuntimeError("Pair precision block is missing its right contrast")
                right_values = right[records[:, block.scope[1]]]
                local = np.einsum(
                    "ni,nj->nij",
                    left_values,
                    right_values,
                    optimize=True,
                ).reshape(records.shape[0], -1)
            end = offset + block.rank
            features[:, offset:end] = local
            offset = end
        return features

    def row_feature_deltas(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        if old.shape != new.shape:
            raise ValueError("old_rows and new_rows must have matching shapes")
        return self.row_features(new) - self.row_features(old)

    def feature_advantages(
        self,
        residual: np.ndarray,
        feature_deltas: np.ndarray,
        edit_cost: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        costs = np.asarray(edit_cost, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("feature_deltas have an invalid shape")
        if costs.shape != (deltas.shape[0],):
            raise ValueError("edit_cost has an invalid shape")
        weighted_residual = self.weighted_coefficient_residual(residual)
        linear = deltas @ weighted_residual
        quad = (deltas * deltas) @ (1.0 / self._coefficient_variances)
        return linear - 0.5 * quad - float(lambda_cost) * costs

    def feature_quadratic_many(self, feature_deltas: np.ndarray) -> np.ndarray:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("feature_deltas have an invalid shape")
        return (deltas * deltas) @ (1.0 / self._coefficient_variances)

    def feature_advantages_with_quadratic(
        self,
        residual: np.ndarray,
        feature_deltas: np.ndarray,
        edit_cost: np.ndarray,
        lambda_cost: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        costs = np.asarray(edit_cost, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("feature_deltas have an invalid shape")
        if costs.shape != (deltas.shape[0],) or not np.all(np.isfinite(costs)):
            raise ValueError("edit_cost has an invalid shape")
        quadratic = self.feature_quadratic_many(deltas)
        advantages = (
            deltas @ self.weighted_coefficient_residual(residual)
            - 0.5 * quadratic
            - float(lambda_cost) * costs
        )
        return advantages, quadratic

    def matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        output = np.zeros(self.dimension, dtype=np.float64)
        for block in self._blocks:
            local = vector[block.query_indices].reshape(block.shape)
            left = block.left_contrast
            if block.kind == "oneway_contrast":
                transformed = left @ (left.T @ local)
            else:
                right = block.right_contrast
                if right is None:
                    raise RuntimeError("Pair precision block is missing its right contrast")
                transformed = left @ (left.T @ local @ right) @ right.T
            output[block.query_indices] = transformed.reshape(-1) / block.variance
        return output

    def diagnostics(self) -> dict[str, Any]:
        rank = int(sum(block.rank for block in self._blocks))
        variances = np.asarray([block.variance for block in self._blocks], dtype=np.float64)
        return {
            "kind": "orthogonal_interaction",
            "dimension": self.dimension,
            "num_blocks": len(self._blocks),
            "num_oneway_blocks": sum(block.kind == "oneway_contrast" for block in self._blocks),
            "num_pair_blocks": sum(block.kind == "pair_interaction" for block in self._blocks),
            "effective_rank": rank,
            "nullity": int(self.dimension - rank),
            "min_component_variance": float(np.min(variances)),
            "max_component_variance": float(np.max(variances)),
        }


class ShrinkageAnalyticOrthogonalInteractionPrecision(PrecisionOperator):
    """Delta-method block precision for positive-part interaction shrinkage."""

    def __init__(
        self,
        qcat: QueryCatalogue,
        workload_groups: list[WorkloadGroup],
        transcript: HierarchicalInteractionTranscript,
        shrinkage: InteractionShrinkageResult,
    ):
        raw = OrthogonalInteractionPrecision(qcat, workload_groups, transcript)
        expected_names = tuple(block.name for block in transcript.strategy.blocks)
        actual_names = tuple(block.name for block in shrinkage.blocks)
        if actual_names != expected_names:
            raise ValueError("Shrinkage block order does not match the interaction transcript")
        if shrinkage.coefficient_dimension != raw.coefficient_dimension:
            raise ValueError("Shrinkage coefficient dimension mismatch")
        for raw_block, shrinkage_block in zip(raw.blocks, shrinkage.blocks, strict=True):
            if raw_block.name != shrinkage_block.name or raw_block.rank != shrinkage_block.dimension:
                raise ValueError(f"Shrinkage precision block mismatch for {raw_block.name!r}")
            if not np.isclose(
                raw_block.variance,
                shrinkage_block.variance,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError(f"Shrinkage variance mismatch for {raw_block.name!r}")
        self._raw = raw
        self._shrinkage = shrinkage

    @property
    def dimension(self) -> int:
        return self._raw.dimension

    @property
    def coefficient_dimension(self) -> int:
        return self._raw.coefficient_dimension

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return self._raw.cardinalities

    @property
    def blocks(self) -> tuple[_OrthogonalBlock, ...]:
        return self._raw.blocks

    @property
    def shrinkage(self) -> InteractionShrinkageResult:
        return self._shrinkage

    def coefficient_coordinates(self, values: np.ndarray) -> np.ndarray:
        return self._raw.coefficient_coordinates(values)

    def coefficient_coordinates_many(self, values: np.ndarray) -> np.ndarray:
        return self._raw.coefficient_coordinates_many(values)

    def coefficient_adjoint(self, values: np.ndarray) -> np.ndarray:
        return self._raw.coefficient_adjoint(values)

    def block_precision_parameters(self, name: str) -> tuple[float, float, np.ndarray]:
        return self._shrinkage.block(name).precision_parameters()

    def weighted_coefficient_residual(self, residual: np.ndarray) -> np.ndarray:
        coefficients = self.coefficient_coordinates(residual)
        return self._shrinkage.precision_matvec(coefficients)

    def row_feature_deltas(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        return self._raw.row_feature_deltas(old_rows, new_rows)

    def feature_quadratic_many(self, feature_deltas: np.ndarray) -> np.ndarray:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        if (
            deltas.ndim != 2
            or deltas.shape[1] != self.coefficient_dimension
            or not np.all(np.isfinite(deltas))
        ):
            raise ValueError(
                f"feature_deltas must be finite with shape (n, {self.coefficient_dimension})"
            )
        weighted = self._shrinkage.precision_matvec_many(deltas)
        values = np.einsum("ij,ij->i", deltas, weighted, optimize=True)
        tolerance = 1.0e-9 * np.maximum(
            1.0,
            np.einsum("ij,ij->i", deltas, deltas, optimize=True),
        )
        if np.any(values < -tolerance):
            raise RuntimeError("Shrinkage analytic precision produced a negative quadratic")
        return np.maximum(values, 0.0)

    def feature_advantages_with_quadratic(
        self,
        residual: np.ndarray,
        feature_deltas: np.ndarray,
        edit_cost: np.ndarray,
        lambda_cost: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        costs = np.asarray(edit_cost, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("feature_deltas have an invalid shape")
        if costs.shape != (deltas.shape[0],) or not np.all(np.isfinite(costs)):
            raise ValueError("edit_cost has an invalid shape")
        quadratic = self.feature_quadratic_many(deltas)
        advantages = (
            deltas @ self.weighted_coefficient_residual(residual)
            - 0.5 * quadratic
            - float(lambda_cost) * costs
        )
        return advantages, quadratic

    def matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        weighted = self._shrinkage.precision_matvec(
            self.coefficient_coordinates(vector)
        )
        return self.coefficient_adjoint(weighted)

    def matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        coefficients = self.coefficient_coordinates_many(matrix)
        weighted = self._shrinkage.precision_matvec_many(coefficients)
        return np.stack([self.coefficient_adjoint(row) for row in weighted], axis=0)

    def quad_many(self, deltas: np.ndarray) -> np.ndarray:
        matrix = self._matrix(deltas, name="deltas")
        return self.feature_quadratic_many(self.coefficient_coordinates_many(matrix))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "kind": "orthogonal_interaction_shrinkage_analytic",
            "dimension": self.dimension,
            "coefficient_dimension": self.coefficient_dimension,
            "effective_rank": self._shrinkage.effective_rank,
            "nullity": int(self.dimension - self._shrinkage.effective_rank),
            "shrinkage": self._shrinkage.diagnostics(),
        }


class BootstrapDiagonalOrthogonalInteractionPrecision(PrecisionOperator):
    """Projection-aware diagonal precision over independent orthogonal coefficients."""

    def __init__(
        self,
        qcat: QueryCatalogue,
        workload_groups: list[WorkloadGroup],
        transcript: HierarchicalInteractionTranscript,
        bootstrap: CoefficientBootstrapDiagonal,
    ):
        raw = OrthogonalInteractionPrecision(qcat, workload_groups, transcript)
        variances = np.asarray(bootstrap.coefficient_variances, dtype=np.float64)
        if variances.shape != (raw.coefficient_dimension,):
            raise ValueError("Coefficient BootDiag variance dimension mismatch")
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError("Coefficient BootDiag variances must be finite and positive")
        self._raw = raw
        self._variances = variances.copy()
        self._variances.setflags(write=False)
        self._bootstrap_diagnostics = dict(bootstrap.diagnostics)

    @property
    def dimension(self) -> int:
        return self._raw.dimension

    @property
    def coefficient_dimension(self) -> int:
        return self._raw.coefficient_dimension

    @property
    def coefficient_variances(self) -> np.ndarray:
        return self._variances

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return self._raw.cardinalities

    def coefficient_coordinates(self, values: np.ndarray) -> np.ndarray:
        return self._raw.coefficient_coordinates(values)

    def coefficient_coordinates_many(self, values: np.ndarray) -> np.ndarray:
        return self._raw.coefficient_coordinates_many(values)

    def coefficient_adjoint(self, values: np.ndarray) -> np.ndarray:
        return self._raw.coefficient_adjoint(values)

    def weighted_coefficient_residual(self, residual: np.ndarray) -> np.ndarray:
        return self.coefficient_coordinates(residual) / self._variances

    def row_feature_deltas(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        return self._raw.row_feature_deltas(old_rows, new_rows)

    def feature_quadratic_many(self, feature_deltas: np.ndarray) -> np.ndarray:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("feature_deltas have an invalid shape")
        return (deltas * deltas) @ (1.0 / self._variances)

    def feature_advantages_with_quadratic(
        self,
        residual: np.ndarray,
        feature_deltas: np.ndarray,
        edit_cost: np.ndarray,
        lambda_cost: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        costs = np.asarray(edit_cost, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("feature_deltas have an invalid shape")
        if costs.shape != (deltas.shape[0],) or not np.all(np.isfinite(costs)):
            raise ValueError("edit_cost has an invalid shape")
        quadratic = self.feature_quadratic_many(deltas)
        advantages = (
            deltas @ self.weighted_coefficient_residual(residual)
            - 0.5 * quadratic
            - float(lambda_cost) * costs
        )
        return advantages, quadratic

    def matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        weighted = self.coefficient_coordinates(vector) / self._variances
        return self.coefficient_adjoint(weighted)

    def matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        weighted = self.coefficient_coordinates_many(matrix) / self._variances.reshape(1, -1)
        return np.stack([self.coefficient_adjoint(row) for row in weighted], axis=0)

    def quad_many(self, deltas: np.ndarray) -> np.ndarray:
        matrix = self._matrix(deltas, name="deltas")
        return self.feature_quadratic_many(self.coefficient_coordinates_many(matrix))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "kind": "orthogonal_interaction_p3_bootstrap_diagonal",
            "dimension": self.dimension,
            "coefficient_dimension": self.coefficient_dimension,
            "effective_rank": self.coefficient_dimension,
            "nullity": int(self.dimension - self.coefficient_dimension),
            "min_component_variance": float(np.min(self._variances)),
            "max_component_variance": float(np.max(self._variances)),
            "bootstrap": dict(self._bootstrap_diagnostics),
        }


class ActiveSetOrthogonalInteractionPrecision(PrecisionOperator):
    """P3 fixed-face tangent precision represented in complete-cell coordinates."""

    def __init__(
        self,
        qcat: QueryCatalogue,
        workload_groups: list[WorkloadGroup],
        transcript: HierarchicalInteractionTranscript,
        covariance: ActiveSetCoefficientCovariance,
    ):
        raw = OrthogonalInteractionPrecision(qcat, workload_groups, transcript)
        if covariance.layout.strategy.cardinalities != transcript.strategy.cardinalities:
            raise ValueError("Active-set covariance cardinalities do not match the transcript")
        if covariance.layout.strategy.pairs != transcript.strategy.pairs:
            raise ValueError("Active-set covariance pair support does not match the transcript")
        if covariance.coefficient_dimension != raw.coefficient_dimension:
            raise ValueError("Active-set covariance coefficient dimension mismatch")
        expected_standard_deviation = np.sqrt(raw.coefficient_variances)
        if not np.allclose(
            covariance.standard_deviation,
            expected_standard_deviation,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("Active-set covariance raw variances do not match the transcript")
        self._raw = raw
        self._covariance = covariance

    @property
    def dimension(self) -> int:
        return self._raw.dimension

    @property
    def coefficient_dimension(self) -> int:
        return self._raw.coefficient_dimension

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return self._raw.cardinalities

    @property
    def covariance(self) -> ActiveSetCoefficientCovariance:
        return self._covariance

    def coefficient_coordinates(self, values: np.ndarray) -> np.ndarray:
        return self._raw.coefficient_coordinates(values)

    def coefficient_coordinates_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        coordinates: list[np.ndarray] = []
        for block in self._raw.blocks:
            local = matrix[:, block.query_indices].reshape((matrix.shape[0], *block.shape))
            left = block.left_contrast
            if block.kind == "oneway_contrast":
                transformed = np.einsum("dc,nd->nc", left, local, optimize=True)
            else:
                right = block.right_contrast
                if right is None:
                    raise RuntimeError("Pair precision block is missing its right contrast")
                transformed = np.einsum(
                    "ia,nij,jb->nab",
                    left,
                    local,
                    right,
                    optimize=True,
                )
            coordinates.append(np.asarray(transformed, dtype=np.float64).reshape(matrix.shape[0], -1))
        return np.concatenate(coordinates, axis=1)

    def coefficient_adjoint(self, coefficient_values: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficient_values, dtype=np.float64)
        if (
            coefficients.shape != (self.coefficient_dimension,)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError(
                f"coefficient_values must be finite with shape ({self.coefficient_dimension},)"
            )
        output = np.zeros(self.dimension, dtype=np.float64)
        offset = 0
        for block in self._raw.blocks:
            end = offset + block.rank
            local = coefficients[offset:end]
            if block.kind == "oneway_contrast":
                transformed = block.left_contrast @ local
            else:
                right = block.right_contrast
                if right is None:
                    raise RuntimeError("Pair precision block is missing its right contrast")
                transformed = (
                    block.left_contrast
                    @ local.reshape(block.left_contrast.shape[1], right.shape[1])
                    @ right.T
                )
            output[block.query_indices] = np.asarray(transformed).reshape(-1)
            offset = end
        return output

    def weighted_coefficient_residual(self, residual: np.ndarray) -> np.ndarray:
        return self._covariance.coefficient_tangent_precision_matvec(
            self.coefficient_coordinates(residual)
        )

    def row_feature_deltas(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        return self._raw.row_feature_deltas(old_rows, new_rows)

    def feature_quadratic_many(self, feature_deltas: np.ndarray) -> np.ndarray:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        if (
            deltas.ndim != 2
            or deltas.shape[1] != self.coefficient_dimension
            or not np.all(np.isfinite(deltas))
        ):
            raise ValueError(
                f"feature_deltas must be finite with shape (n, {self.coefficient_dimension})"
            )
        weighted = self._covariance.coefficient_tangent_precision_matvec_many(deltas)
        values = np.einsum("ij,ij->i", deltas, weighted, optimize=True)
        tolerance = 1.0e-9 * np.maximum(
            1.0,
            np.einsum("ij,ij->i", deltas, deltas, optimize=True),
        )
        if np.any(values < -tolerance):
            raise RuntimeError("Active-set precision produced a negative feature quadratic")
        return np.maximum(values, 0.0)

    def feature_advantages_with_quadratic(
        self,
        residual: np.ndarray,
        feature_deltas: np.ndarray,
        edit_cost: np.ndarray,
        lambda_cost: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        costs = np.asarray(edit_cost, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("feature_deltas have an invalid shape")
        if costs.shape != (deltas.shape[0],) or not np.all(np.isfinite(costs)):
            raise ValueError("edit_cost has an invalid shape")
        weighted_residual = self.weighted_coefficient_residual(residual)
        quadratic = self.feature_quadratic_many(deltas)
        advantages = (
            deltas @ weighted_residual
            - 0.5 * quadratic
            - float(lambda_cost) * costs
        )
        return advantages, quadratic

    def matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        weighted = self._covariance.coefficient_tangent_precision_matvec(
            self.coefficient_coordinates(vector)
        )
        return self.coefficient_adjoint(weighted)

    def matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        coefficients = self.coefficient_coordinates_many(matrix)
        weighted = self._covariance.coefficient_tangent_precision_matvec_many(
            coefficients
        )
        output = np.empty_like(matrix, dtype=np.float64)
        for row in range(matrix.shape[0]):
            output[row] = self.coefficient_adjoint(weighted[row])
        return output

    def quad_many(self, deltas: np.ndarray) -> np.ndarray:
        matrix = self._matrix(deltas, name="deltas")
        coefficients = self.coefficient_coordinates_many(matrix)
        return self.feature_quadratic_many(coefficients)

    def diagnostics(self) -> dict[str, Any]:
        covariance = self._covariance.diagnostics()
        effective_rank = covariance["effective_rank"]
        return {
            "kind": "orthogonal_interaction_p3_active_set",
            "dimension": self.dimension,
            "coefficient_dimension": self.coefficient_dimension,
            "effective_rank": effective_rank,
            "nullity": (
                None if effective_rank is None else int(self.dimension - int(effective_rank))
            ),
            "covariance": covariance,
        }
