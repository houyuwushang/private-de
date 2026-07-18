from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.evolution.precision import PrecisionOperator
from qdte.measurement.direct_coarse import CoarseInteractionTranscript
from qdte.queries.orthogonal import helmert_contrast
from qdte.queries.types import OP_EQ, QueryCatalogue
from qdte.queries.workload import WorkloadGroup


def _readonly(values: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _CoarsePrecisionBlock:
    name: str
    kind: str
    scope: tuple[int, ...]
    query_indices: np.ndarray
    original_shape: tuple[int, ...]
    left_features: np.ndarray
    right_features: np.ndarray | None
    covariance: np.ndarray
    precision: np.ndarray

    @property
    def rank(self) -> int:
        if self.right_features is None:
            return int(self.left_features.shape[1])
        return int(self.left_features.shape[1] * self.right_features.shape[1])


def _scope(group: WorkloadGroup) -> tuple[int, ...]:
    parts = str(group.name).split(":")
    if group.family == "oneway" and len(parts) == 2:
        return (int(parts[1]),)
    if group.family == "twoway" and len(parts) == 3:
        pair = (int(parts[1]), int(parts[2]))
        if pair[0] >= pair[1]:
            raise ValueError("C2 pair workload must use canonical attribute order")
        return pair
    raise ValueError("C2 precision requires one-way and two-way complete partitions")


def _validate_group(
    qcat: QueryCatalogue,
    group: WorkloadGroup,
    scope: tuple[int, ...],
    cardinalities: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, ...]]:
    indices = np.asarray(group.query_indices, dtype=np.int64)
    shape = tuple(cardinalities[attribute] for attribute in scope)
    if not group.is_partition or indices.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("C2 precision requires complete partition groups")
    expected = np.stack(
        np.unravel_index(np.arange(indices.size, dtype=np.int64), shape),
        axis=1,
    )
    for row, qid in enumerate(indices.tolist()):
        if int(qcat.num_terms[qid]) != len(scope):
            raise ValueError("C2 partition query has the wrong number of terms")
        if tuple(int(value) for value in qcat.attrs[qid, : len(scope)]) != scope:
            raise ValueError("C2 partition query has the wrong scope")
        if np.any(qcat.ops[qid, : len(scope)] != OP_EQ):
            raise ValueError("C2 partition query must contain equality terms")
        if not np.array_equal(qcat.values[qid, : len(scope)], expected[row]):
            raise ValueError("C2 partition query ordering must be canonical row-major")
    return _readonly(indices, dtype=np.int32), shape


class CoarsenedInteractionPrecision(PrecisionOperator):
    """Exact C2 likelihood in original query coordinates.

    One-way features remain in the original public domain. Pair features first
    map original categories through the released support partition and then use
    the coarse Helmert interaction basis. Each pair keeps its exact full
    covariance; no diagonal approximation is used.
    """

    def __init__(
        self,
        qcat: QueryCatalogue,
        workload_groups: list[WorkloadGroup],
        transcript: CoarseInteractionTranscript,
    ) -> None:
        cards = transcript.support.original_cardinalities
        qcat.validate(np.asarray(cards, dtype=np.int64))
        coverage = np.zeros(qcat.m, dtype=np.int32)
        blocks: list[_CoarsePrecisionBlock] = []
        released_centers: list[np.ndarray] = []
        seen_pairs: set[tuple[int, int]] = set()
        for group in workload_groups:
            scope = _scope(group)
            indices, shape = _validate_group(qcat, group, scope, cards)
            coverage[indices] += 1
            if len(scope) == 1:
                attribute = scope[0]
                left_features = helmert_contrast(cards[attribute])
                covariance = float(transcript.oneway_variances[attribute]) * np.eye(
                    cards[attribute] - 1,
                    dtype=np.float64,
                )
                right_features = None
                name = f"oneway_contrast:{attribute}"
                kind = "oneway_contrast"
                released_center = np.asarray(
                    transcript.oneway_components[attribute],
                    dtype=np.float64,
                ).reshape(-1)
            else:
                left, right = scope
                observation = transcript.pair(scope)
                left_map = transcript.support.attributes[left]
                right_map = transcript.support.attributes[right]
                left_features = (
                    left_map.aggregation_matrix().T
                    @ helmert_contrast(left_map.coarse_cardinality)
                )
                right_features = (
                    right_map.aggregation_matrix().T
                    @ helmert_contrast(right_map.coarse_cardinality)
                )
                covariance = np.asarray(
                    observation.interaction_covariance,
                    dtype=np.float64,
                )
                name = f"coarse_pair_interaction:{left}:{right}"
                kind = "coarse_pair_interaction"
                released_center = np.asarray(
                    observation.interaction_center,
                    dtype=np.float64,
                ).reshape(-1)
                seen_pairs.add(scope)
            symmetric = 0.5 * (covariance + covariance.T)
            eigenvalues = np.linalg.eigvalsh(symmetric)
            tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
            if np.min(eigenvalues) <= tolerance:
                raise ValueError(f"C2 covariance block {name!r} must be positive definite")
            precision = np.linalg.inv(symmetric)
            if released_center.shape != (int(symmetric.shape[0]),):
                raise ValueError(f"C2 released center {name!r} has the wrong shape")
            released_centers.append(released_center)
            blocks.append(
                _CoarsePrecisionBlock(
                    name=name,
                    kind=kind,
                    scope=scope,
                    query_indices=indices,
                    original_shape=shape,
                    left_features=_readonly(left_features),
                    right_features=(
                        _readonly(right_features)
                        if right_features is not None
                        else None
                    ),
                    covariance=_readonly(symmetric),
                    precision=_readonly(precision),
                )
            )
        if not np.all(coverage == 1):
            raise ValueError("C2 precision groups must cover every query exactly once")
        if seen_pairs != set(transcript.pairs):
            raise ValueError("C2 precision workload and pair transcript do not match")
        self._dimension = int(qcat.m)
        self._cardinalities = tuple(int(value) for value in cards)
        self._blocks = tuple(blocks)
        self._coefficient_dimension = int(sum(block.rank for block in blocks))
        covariance = np.zeros(
            (self._coefficient_dimension, self._coefficient_dimension),
            dtype=np.float64,
        )
        precision = np.zeros_like(covariance)
        offset = 0
        for block in blocks:
            end = offset + block.rank
            covariance[offset:end, offset:end] = block.covariance
            precision[offset:end, offset:end] = block.precision
            offset = end
        self._coefficient_covariance = _readonly(covariance)
        self._coefficient_precision = _readonly(precision)
        self._coefficient_variances = _readonly(np.diag(covariance))
        self._released_coefficient_center = _readonly(
            np.concatenate(released_centers)
        )
        proposal_precision_diagonal = np.zeros(self._dimension, dtype=np.float64)
        for block in blocks:
            if block.right_features is None:
                design = block.left_features
            else:
                design = np.einsum(
                    "ia,jb->ijab",
                    block.left_features,
                    block.right_features,
                    optimize=True,
                ).reshape(len(block.query_indices), block.rank)
            proposal_precision_diagonal[block.query_indices] = np.einsum(
                "ij,jk,ik->i",
                design,
                block.precision,
                design,
                optimize=True,
            )
        if np.any(proposal_precision_diagonal < -1.0e-10):
            raise RuntimeError("C2 query precision has a negative diagonal")
        self._proposal_precision_diagonal = _readonly(
            np.maximum(proposal_precision_diagonal, 0.0)
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return self._cardinalities

    @property
    def blocks(self) -> tuple[_CoarsePrecisionBlock, ...]:
        return self._blocks

    @property
    def coefficient_dimension(self) -> int:
        return self._coefficient_dimension

    @property
    def coefficient_covariance(self) -> np.ndarray:
        return self._coefficient_covariance

    @property
    def coefficient_variances(self) -> np.ndarray:
        return self._coefficient_variances

    @property
    def released_coefficient_center(self) -> np.ndarray:
        return self._released_coefficient_center

    def target_reconstruction_residual(self, target: np.ndarray) -> float:
        coordinates = self.coefficient_coordinates(target)
        return float(np.max(np.abs(coordinates - self.released_coefficient_center)))

    @property
    def proposal_precision_diagonal(self) -> np.ndarray:
        """Diagonal of the exact query-space precision operator.

        If the released coefficient noise has covariance ``Sigma``, the
        query-gradient noise ``F Sigma^-1 noise`` has this diagonal variance.
        It therefore supplies the proposal scheduler's scale without assigning
        weight to directions removed by support coarsening.
        """

        return self._proposal_precision_diagonal

    def proposal_signal(self, residual: np.ndarray) -> np.ndarray:
        """Return the nullspace-free linear signal used only for proposals."""

        return self.matvec(residual)

    def proposal_sigma(self) -> np.ndarray:
        return np.sqrt(self.proposal_precision_diagonal)

    def block_precision_matrix(self, name: str) -> np.ndarray:
        for block in self._blocks:
            if block.name == name:
                return block.precision
        raise KeyError(name)

    def coefficient_coordinates(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        coordinates: list[np.ndarray] = []
        for block in self._blocks:
            local = vector[block.query_indices].reshape(block.original_shape)
            if block.right_features is None:
                transformed = block.left_features.T @ local
            else:
                transformed = (
                    block.left_features.T @ local @ block.right_features
                )
            coordinates.append(np.asarray(transformed).reshape(-1))
        return np.concatenate(coordinates)

    def coefficient_coordinates_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        return np.stack(
            [self.coefficient_coordinates(row) for row in matrix],
            axis=0,
        )

    def coefficient_adjoint(self, coefficient_values: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficient_values, dtype=np.float64)
        if coefficients.shape != (self.coefficient_dimension,):
            raise ValueError("C2 coefficient_values have the wrong shape")
        output = np.zeros(self.dimension, dtype=np.float64)
        offset = 0
        for block in self._blocks:
            end = offset + block.rank
            local = coefficients[offset:end]
            if block.right_features is None:
                transformed = block.left_features @ local
            else:
                transformed = (
                    block.left_features
                    @ local.reshape(
                        block.left_features.shape[1],
                        block.right_features.shape[1],
                    )
                    @ block.right_features.T
                )
            output[block.query_indices] = transformed.reshape(-1)
            offset = end
        return output

    def weighted_coefficient_residual(self, residual: np.ndarray) -> np.ndarray:
        return self.coefficient_precision_matvec(
            self.coefficient_coordinates(residual)
        )

    def coefficient_precision_matvec(self, values: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(values, dtype=np.float64)
        if coefficients.shape != (self.coefficient_dimension,):
            raise ValueError("C2 coefficient values have the wrong shape")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("C2 coefficient values must be finite")
        output = np.empty_like(coefficients)
        offset = 0
        for block in self._blocks:
            end = offset + block.rank
            output[offset:end] = block.precision @ coefficients[offset:end]
            offset = end
        return output

    def coefficient_precision_matvec_many(self, values: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(values, dtype=np.float64)
        if (
            coefficients.ndim != 2
            or coefficients.shape[1] != self.coefficient_dimension
        ):
            raise ValueError("C2 coefficient values have the wrong matrix shape")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("C2 coefficient values must be finite")
        output = np.empty_like(coefficients)
        offset = 0
        for block in self._blocks:
            end = offset + block.rank
            output[:, offset:end] = (
                coefficients[:, offset:end] @ block.precision
            )
            offset = end
        return output

    def row_features(self, rows: np.ndarray) -> np.ndarray:
        records = np.asarray(rows, dtype=np.int32)
        if records.ndim != 2 or records.shape[1] != len(self.cardinalities):
            raise ValueError("C2 rows have the wrong shape")
        for attribute, cardinality in enumerate(self.cardinalities):
            if np.any(records[:, attribute] < 0) or np.any(
                records[:, attribute] >= cardinality
            ):
                raise ValueError("C2 rows contain out-of-domain values")
        features = np.empty(
            (records.shape[0], self.coefficient_dimension),
            dtype=np.float64,
        )
        offset = 0
        for block in self._blocks:
            left = block.left_features[records[:, block.scope[0]]]
            if block.right_features is None:
                local = left
            else:
                right = block.right_features[records[:, block.scope[1]]]
                local = np.einsum(
                    "ni,nj->nij",
                    left,
                    right,
                    optimize=True,
                ).reshape(records.shape[0], -1)
            end = offset + block.rank
            features[:, offset:end] = local
            offset = end
        return features

    def row_feature_deltas(
        self,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
    ) -> np.ndarray:
        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        if old.shape != new.shape:
            raise ValueError("C2 old_rows and new_rows must have matching shapes")
        return self.row_features(new) - self.row_features(old)

    def feature_quadratic_many(self, feature_deltas: np.ndarray) -> np.ndarray:
        deltas = np.asarray(feature_deltas, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != self.coefficient_dimension:
            raise ValueError("C2 feature_deltas have the wrong shape")
        weighted = self.coefficient_precision_matvec_many(deltas)
        values = np.einsum("ij,ij->i", deltas, weighted, optimize=True)
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
        if costs.shape != (deltas.shape[0],):
            raise ValueError("C2 edit_cost has the wrong shape")
        quadratic = self.feature_quadratic_many(deltas)
        advantages = (
            deltas @ self.weighted_coefficient_residual(residual)
            - 0.5 * quadratic
            - float(lambda_cost) * costs
        )
        return advantages, quadratic

    def matvec(self, values: np.ndarray) -> np.ndarray:
        coefficients = self.coefficient_coordinates(values)
        return self.coefficient_adjoint(
            self.coefficient_precision_matvec(coefficients)
        )

    def matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = self._matrix(values, name="values")
        coefficients = self.coefficient_coordinates_many(matrix)
        weighted = self.coefficient_precision_matvec_many(coefficients)
        return np.stack([self.coefficient_adjoint(row) for row in weighted], axis=0)

    def quad_many(self, deltas: np.ndarray) -> np.ndarray:
        matrix = self._matrix(deltas, name="deltas")
        return self.feature_quadratic_many(self.coefficient_coordinates_many(matrix))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "kind": "released_support_coarsened_interaction_full_covariance",
            "dimension": self.dimension,
            "coefficient_dimension": self.coefficient_dimension,
            "effective_rank": self.coefficient_dimension,
            "num_blocks": len(self.blocks),
            "num_pair_blocks": sum(
                block.kind == "coarse_pair_interaction" for block in self.blocks
            ),
            "minimum_coefficient_variance": float(
                np.min(self.coefficient_variances)
            ),
            "maximum_coefficient_variance": float(
                np.max(self.coefficient_variances)
            ),
            "proposal_signal": "query_precision_matvec",
            "proposal_sigma": "sqrt_query_precision_diagonal",
            "proposal_zero_precision_coordinates": int(
                np.sum(self.proposal_precision_diagonal <= 1.0e-15)
            ),
        }


__all__ = ["CoarsenedInteractionPrecision"]
