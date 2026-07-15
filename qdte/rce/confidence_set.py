from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.stats import chi2, norm


@dataclass(frozen=True)
class RCEConstraintEvaluation:
    squared_discrepancy: float
    ellipsoid_ratio: float
    max_standardized_coordinate: float
    tube_ratio: float
    slack: float
    inside: bool
    num_active_tube_coordinates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "squared_discrepancy": self.squared_discrepancy,
            "ellipsoid_ratio": self.ellipsoid_ratio,
            "max_standardized_coordinate": self.max_standardized_coordinate,
            "tube_ratio": self.tube_ratio,
            "slack": self.slack,
            "inside": self.inside,
            "num_active_tube_coordinates": self.num_active_tube_coordinates,
        }


class RCEConfidenceSet:
    """Mixed Gaussian ellipsoid and coordinate-tube confidence set.

    Residuals use the QDTE convention ``released_target - synthetic_answer``.
    The sign does not affect feasibility. A diagonal construction is used for
    the independent orthogonal transcript and remains O(m); the covariance
    construction supports small singular PSD systems for theorem tests and
    block implementations.
    """

    def __init__(
        self,
        *,
        marginal_variances: np.ndarray,
        effective_rank: int,
        precision_diagonal: np.ndarray | None,
        precision_matrix: np.ndarray | None,
        alpha_l2: float,
        alpha_linf: float,
        support_tolerance: float,
        kind: str,
    ) -> None:
        variances = np.asarray(marginal_variances, dtype=np.float64)
        if variances.ndim != 1 or variances.size == 0:
            raise ValueError("marginal_variances must be a non-empty vector")
        if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
            raise ValueError("marginal_variances must be finite and nonnegative")
        rank = int(effective_rank)
        if rank <= 0 or rank > variances.size:
            raise ValueError("effective_rank must be in [1, dimension]")
        alpha2 = float(alpha_l2)
        alpha_inf = float(alpha_linf)
        if not math.isfinite(alpha2) or not 0.0 < alpha2 < 1.0:
            raise ValueError("alpha_l2 must be finite and in (0, 1)")
        if not math.isfinite(alpha_inf) or not 0.0 < alpha_inf < 1.0:
            raise ValueError("alpha_linf must be finite and in (0, 1)")
        tolerance = float(support_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("support_tolerance must be finite and nonnegative")

        diagonal: np.ndarray | None = None
        matrix: np.ndarray | None = None
        if precision_diagonal is not None:
            diagonal = np.asarray(precision_diagonal, dtype=np.float64)
            if diagonal.shape != variances.shape:
                raise ValueError("precision_diagonal must match marginal_variances")
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal < 0.0):
                raise ValueError("precision_diagonal must be finite and nonnegative")
            diagonal = diagonal.copy()
            diagonal.setflags(write=False)
        if precision_matrix is not None:
            matrix = np.asarray(precision_matrix, dtype=np.float64)
            if matrix.shape != (variances.size, variances.size):
                raise ValueError("precision_matrix must be square with confidence dimension")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("precision_matrix must be finite")
            if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("precision_matrix must be symmetric")
            matrix = 0.5 * (matrix + matrix.T)
            matrix.setflags(write=False)
        if (diagonal is None) == (matrix is None):
            raise ValueError("Specify exactly one precision representation")

        tube_mask = variances > tolerance
        tube_dimension = int(np.sum(tube_mask))
        if tube_dimension <= 0:
            raise ValueError("At least one coordinate must have positive marginal variance")
        c2 = float(chi2.ppf(1.0 - alpha2, rank))
        c_inf = float(norm.ppf(1.0 - alpha_inf / (2.0 * tube_dimension)))
        if not math.isfinite(c2) or c2 <= 0.0:
            raise ValueError("chi-square threshold must be finite and positive")
        if not math.isfinite(c_inf) or c_inf <= 0.0:
            raise ValueError("coordinate threshold must be finite and positive")

        variances = variances.copy()
        variances.setflags(write=False)
        tube_mask.setflags(write=False)
        bounds = np.full(variances.shape, np.inf, dtype=np.float64)
        bounds[tube_mask] = c_inf * np.sqrt(variances[tube_mask])
        bounds.setflags(write=False)

        self._variances = variances
        self._precision_diagonal = diagonal
        self._precision_matrix = matrix
        self._tube_mask = tube_mask
        self._bounds = bounds
        self._rank = rank
        self._tube_dimension = tube_dimension
        self._alpha_l2 = alpha2
        self._alpha_linf = alpha_inf
        self._c2 = c2
        self._c_inf = c_inf
        self._support_tolerance = tolerance
        self._kind = str(kind)

    @classmethod
    def from_diagonal_variances(
        cls,
        variances: np.ndarray,
        *,
        alpha_l2: float = 0.025,
        alpha_linf: float = 0.025,
        support_tolerance: float = 0.0,
    ) -> RCEConfidenceSet:
        values = np.asarray(variances, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("variances must be one-dimensional")
        positive = values > float(support_tolerance)
        precision = np.zeros(values.shape, dtype=np.float64)
        precision[positive] = 1.0 / values[positive]
        return cls(
            marginal_variances=values,
            effective_rank=int(np.sum(positive)),
            precision_diagonal=precision,
            precision_matrix=None,
            alpha_l2=alpha_l2,
            alpha_linf=alpha_linf,
            support_tolerance=support_tolerance,
            kind="diagonal",
        )

    @classmethod
    def from_covariance(
        cls,
        covariance: np.ndarray,
        *,
        alpha_l2: float = 0.025,
        alpha_linf: float = 0.025,
        relative_rank_tolerance: float = 1.0e-12,
    ) -> RCEConfidenceSet:
        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("covariance must be a non-empty square matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("covariance must be finite")
        symmetric = 0.5 * (matrix + matrix.T)
        if not np.allclose(matrix, symmetric, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("covariance must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        tolerance = float(relative_rank_tolerance) * scale
        if np.min(eigenvalues) < -10.0 * tolerance:
            raise ValueError("covariance must be positive semidefinite")
        support = eigenvalues > tolerance
        if not np.any(support):
            raise ValueError("covariance must have positive rank")
        precision = (
            eigenvectors[:, support]
            * (1.0 / eigenvalues[support]).reshape(1, -1)
        ) @ eigenvectors[:, support].T
        return cls(
            marginal_variances=np.maximum(np.diag(symmetric), 0.0),
            effective_rank=int(np.sum(support)),
            precision_diagonal=None,
            precision_matrix=precision,
            alpha_l2=alpha_l2,
            alpha_linf=alpha_linf,
            support_tolerance=tolerance,
            kind="covariance_pseudoinverse",
        )

    @property
    def dimension(self) -> int:
        return int(self._variances.size)

    @property
    def effective_rank(self) -> int:
        return self._rank

    @property
    def tube_dimension(self) -> int:
        return self._tube_dimension

    @property
    def marginal_variances(self) -> np.ndarray:
        return self._variances

    @property
    def tube_mask(self) -> np.ndarray:
        return self._tube_mask

    @property
    def coordinate_bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def alpha_l2(self) -> float:
        return self._alpha_l2

    @property
    def alpha_linf(self) -> float:
        return self._alpha_linf

    @property
    def squared_discrepancy_threshold(self) -> float:
        return self._c2

    @property
    def coordinate_standardized_threshold(self) -> float:
        return self._c_inf

    def _vector(self, values: np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be finite with shape ({self.dimension},)")
        return vector

    def precision_matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        if self._precision_diagonal is not None:
            return vector * self._precision_diagonal
        if self._precision_matrix is None:
            raise RuntimeError("Confidence precision representation is missing")
        return self._precision_matrix @ vector

    def precision_matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(f"values must have shape (n, {self.dimension})")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("values must be finite")
        if self._precision_diagonal is not None:
            return matrix * self._precision_diagonal.reshape(1, -1)
        if self._precision_matrix is None:
            raise RuntimeError("Confidence precision representation is missing")
        return matrix @ self._precision_matrix

    def squared_discrepancy(self, residual: np.ndarray) -> float:
        vector = self._vector(residual, name="residual")
        value = float(vector @ self.precision_matvec(vector))
        tolerance = 1.0e-10 * max(1.0, float(vector @ vector))
        if value < -tolerance:
            raise RuntimeError("Confidence precision produced a negative quadratic form")
        return max(0.0, value)

    def standardized_coordinates(self, residual: np.ndarray) -> np.ndarray:
        vector = self._vector(residual, name="residual")
        result = np.zeros(vector.shape, dtype=np.float64)
        result[self._tube_mask] = (
            vector[self._tube_mask] / np.sqrt(self._variances[self._tube_mask])
        )
        return result

    def evaluate(self, residual: np.ndarray, *, tolerance: float = 1.0e-12) -> RCEConstraintEvaluation:
        vector = self._vector(residual, name="residual")
        squared = self.squared_discrepancy(vector)
        standardized = np.abs(self.standardized_coordinates(vector)[self._tube_mask])
        maximum = float(np.max(standardized))
        ellipsoid_ratio = squared / self._c2
        tube_ratio = maximum / self._c_inf
        slack = max(0.0, ellipsoid_ratio - 1.0, tube_ratio - 1.0)
        scale_tolerance = float(tolerance) * max(1.0, ellipsoid_ratio, tube_ratio)
        active = int(np.sum(np.abs(standardized / self._c_inf - 1.0) <= 1.0e-6))
        return RCEConstraintEvaluation(
            squared_discrepancy=squared,
            ellipsoid_ratio=ellipsoid_ratio,
            max_standardized_coordinate=maximum,
            tube_ratio=tube_ratio,
            slack=slack,
            inside=slack <= scale_tolerance,
            num_active_tube_coordinates=active,
        )

    def contains(self, residual: np.ndarray, *, tolerance: float = 1.0e-12) -> bool:
        return self.evaluate(residual, tolerance=tolerance).inside

    def diagnostics(self) -> dict[str, Any]:
        positive = self._variances[self._tube_mask]
        return {
            "kind": self._kind,
            "dimension": self.dimension,
            "effective_rank": self.effective_rank,
            "tube_dimension": self.tube_dimension,
            "nullity": self.dimension - self.effective_rank,
            "alpha_l2": self.alpha_l2,
            "alpha_linf": self.alpha_linf,
            "confidence_level_union_bound": 1.0 - self.alpha_l2 - self.alpha_linf,
            "squared_discrepancy_threshold": self.squared_discrepancy_threshold,
            "coordinate_standardized_threshold": self.coordinate_standardized_threshold,
            "minimum_positive_marginal_variance": float(np.min(positive)),
            "maximum_marginal_variance": float(np.max(positive)),
            "support_tolerance": self._support_tolerance,
        }
