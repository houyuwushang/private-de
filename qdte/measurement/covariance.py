from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.local_polytope import (
    LocalPolytopeLayout,
    LocalPolytopeProjectionResult,
    project_hierarchical_local_polytope,
)


@dataclass(frozen=True)
class CoefficientBootstrapDiagonal:
    component_variances: dict[str, np.ndarray]
    coefficient_variances: np.ndarray
    mean_projected_components: dict[str, np.ndarray]
    diagnostics: dict[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": dict(self.diagnostics),
            "blocks": [
                {
                    "name": name,
                    "shape": list(values.shape),
                    "effective_variances": values.tolist(),
                    "mean_projected_coefficients": self.mean_projected_components[name].tolist(),
                }
                for name, values in self.component_variances.items()
            ],
        }

    @classmethod
    def from_public_dict(
        cls,
        transcript: HierarchicalInteractionTranscript,
        data: dict[str, Any],
    ) -> CoefficientBootstrapDiagonal:
        if not isinstance(data, dict):
            raise ValueError("Coefficient BootDiag artifact must be a mapping")
        diagnostics = data.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise ValueError("Coefficient BootDiag diagnostics must be a mapping")
        raw_blocks = data.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise ValueError("Coefficient BootDiag blocks must be a list")
        by_name: dict[str, dict[str, Any]] = {}
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                raise ValueError("Coefficient BootDiag block must be a mapping")
            name = str(raw.get("name", ""))
            if not name or name in by_name:
                raise ValueError(f"Duplicate or empty coefficient BootDiag block {name!r}")
            by_name[name] = raw
        expected = {block.name for block in transcript.strategy.blocks}
        if set(by_name) != expected:
            raise ValueError("Coefficient BootDiag block set does not match the transcript")

        component_variances: dict[str, np.ndarray] = {}
        means: dict[str, np.ndarray] = {}
        for block in transcript.strategy.blocks:
            raw = by_name[block.name]
            if tuple(int(value) for value in raw.get("shape", [])) != block.coefficient_shape:
                raise ValueError(f"Coefficient BootDiag shape mismatch for {block.name!r}")
            variance = np.asarray(raw.get("effective_variances", []), dtype=np.float64)
            mean = np.asarray(raw.get("mean_projected_coefficients", []), dtype=np.float64)
            if (
                variance.shape != block.coefficient_shape
                or not np.all(np.isfinite(variance))
                or np.any(variance <= 0.0)
            ):
                raise ValueError(
                    f"Coefficient BootDiag variances for {block.name!r} must be finite and positive"
                )
            if mean.shape != block.coefficient_shape or not np.all(np.isfinite(mean)):
                raise ValueError(
                    f"Coefficient BootDiag means for {block.name!r} have an invalid shape"
                )
            component_variances[block.name] = variance
            means[block.name] = mean
        layout = LocalPolytopeLayout(transcript.strategy)
        vector = layout.flatten_components(component_variances)
        return cls(
            component_variances=component_variances,
            coefficient_variances=vector,
            mean_projected_components=means,
            diagnostics=dict(diagnostics),
        )


def bootstrap_local_polytope_coefficient_diagonal(
    transcript: HierarchicalInteractionTranscript,
    projection: LocalPolytopeProjectionResult,
    *,
    rng: np.random.Generator,
    num_samples: int = 16,
    min_variance: float = 1.0e-6,
    min_raw_variance_fraction: float = 0.02,
) -> CoefficientBootstrapDiagonal:
    """Parametric BootDiag in independent raw coefficient coordinates."""
    if int(num_samples) <= 1:
        raise ValueError("num_samples must be greater than one")
    if not np.isfinite(min_variance) or float(min_variance) <= 0.0:
        raise ValueError("min_variance must be finite and positive")
    if (
        not np.isfinite(min_raw_variance_fraction)
        or float(min_raw_variance_fraction) < 0.0
    ):
        raise ValueError("min_raw_variance_fraction must be finite and nonnegative")
    if not bool(projection.diagnostics.get("certificate_passed", False)):
        raise ValueError("Coefficient BootDiag requires a certified center projection")

    layout = LocalPolytopeLayout(transcript.strategy)
    center = layout.flatten_components(projection.projected_components)
    raw_standard_deviation = layout.coefficient_standard_deviations(
        transcript.component_variances
    )
    mean = np.zeros(layout.coefficient_dimension, dtype=np.float64)
    m2 = np.zeros(layout.coefficient_dimension, dtype=np.float64)
    center_active = np.asarray(projection.active_cell_mask, dtype=bool)
    active_counts: list[int] = []
    active_switches = 0

    for sample_index in range(1, int(num_samples) + 1):
        sampled = center + rng.normal(
            loc=0.0,
            scale=raw_standard_deviation,
            size=layout.coefficient_dimension,
        )
        bootstrap_transcript = HierarchicalInteractionTranscript(
            strategy=transcript.strategy,
            public_total=transcript.public_total,
            noisy_components=layout.unflatten_components(sampled),
            component_variances={
                name: float(value) for name, value in transcript.component_variances.items()
            },
            rho_by_block={name: float(value) for name, value in transcript.rho_by_block.items()},
            rho_total=float(transcript.rho_total),
            rho_spent=float(transcript.rho_spent),
            allocation_mode=transcript.allocation_mode,
            adjacency=transcript.adjacency,
        )
        bootstrap_projection = project_hierarchical_local_polytope(bootstrap_transcript)
        projected = layout.flatten_components(bootstrap_projection.projected_components)
        delta = projected - mean
        mean += delta / float(sample_index)
        m2 += delta * (projected - mean)
        active = np.asarray(bootstrap_projection.active_cell_mask, dtype=bool)
        active_counts.append(int(np.sum(active)))
        active_switches += int(not np.array_equal(active, center_active))

    sample_variance = m2 / float(int(num_samples) - 1)
    raw_variance = raw_standard_deviation * raw_standard_deviation
    effective_variance = np.maximum(sample_variance, float(min_variance))
    effective_variance = np.maximum(
        effective_variance,
        float(min_raw_variance_fraction) * raw_variance,
    )
    component_variances = layout.unflatten_components(effective_variance)
    mean_components = layout.unflatten_components(mean)
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "method": "coefficient_space_local_polytope_bootstrap_diagonal",
        "num_samples": int(num_samples),
        "center": "projected",
        "debias_target": False,
        "min_variance": float(min_variance),
        "min_raw_variance_fraction": float(min_raw_variance_fraction),
        "coefficient_dimension": layout.coefficient_dimension,
        "raw_variance_mean": float(np.mean(raw_variance)),
        "sample_variance_mean": float(np.mean(sample_variance)),
        "effective_variance_mean": float(np.mean(effective_variance)),
        "effective_variance_min": float(np.min(effective_variance)),
        "effective_variance_max": float(np.max(effective_variance)),
        "effective_to_raw_variance_mean": float(np.mean(effective_variance / raw_variance)),
        "mean_projected_l2_from_center": float(np.linalg.norm(mean - center)),
        "center_active_cell_count": int(np.sum(center_active)),
        "bootstrap_active_cell_count_min": int(min(active_counts)),
        "bootstrap_active_cell_count_max": int(max(active_counts)),
        "bootstrap_active_cell_count_mean": float(np.mean(active_counts)),
        "active_set_switch_count": int(active_switches),
        "active_set_switch_fraction": float(active_switches) / float(num_samples),
        "uncertainty_status": "projection_aware_diagonal_coefficient_covariance",
    }
    return CoefficientBootstrapDiagonal(
        component_variances=component_variances,
        coefficient_variances=effective_variance,
        mean_projected_components=mean_components,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class ActiveSetCovarianceCertificate:
    passed: bool
    nullspace_max_residual: float
    symmetry_max_residual: float
    idempotence_max_residual: float
    covariance_min_quadratic: float
    tolerance: float
    num_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_passed": self.passed,
            "certificate_nullspace_max_residual": self.nullspace_max_residual,
            "certificate_symmetry_max_residual": self.symmetry_max_residual,
            "certificate_idempotence_max_residual": self.idempotence_max_residual,
            "certificate_covariance_min_quadratic": self.covariance_min_quadratic,
            "certificate_tolerance": self.tolerance,
            "certificate_num_probes": self.num_probes,
        }


class ActiveSetCoefficientCovariance:
    """Fixed-active-set Jacobian and covariance in orthogonal coefficient space.

    Let ``S`` be the raw coefficient standard deviation and ``B = A S`` the
    whitened active-constraint matrix. The local projection derivative is
    ``P = I - B.T (B B.T)^+ B`` in whitened coordinates. This class applies
    ``P``, ``S P S^-1``, ``S P S``, and the tangent generalized precision
    ``S^-1 P S^-1`` without forming a global dense covariance.
    """

    def __init__(
        self,
        transcript: HierarchicalInteractionTranscript,
        projection: LocalPolytopeProjectionResult,
        *,
        active_set_tolerance: float = 1.0e-7,
        multiplier_tolerance: float = 1.0e-10,
        dense_reference_cells: int = 2_000_000,
        linear_tolerance: float = 1.0e-11,
        linear_max_iterations: int = 50_000,
    ):
        values = (
            active_set_tolerance,
            multiplier_tolerance,
            linear_tolerance,
        )
        if any(not np.isfinite(value) or float(value) < 0.0 for value in values):
            raise ValueError("Active-set covariance tolerances must be finite and nonnegative")
        if int(dense_reference_cells) < 0:
            raise ValueError("dense_reference_cells must be nonnegative")
        if int(linear_max_iterations) <= 0:
            raise ValueError("linear_max_iterations must be positive")
        if not bool(projection.diagnostics.get("certificate_passed", False)):
            raise ValueError("Active-set covariance requires a certified P3 projection")

        self.layout = LocalPolytopeLayout(transcript.strategy)
        self.standard_deviation = self.layout.coefficient_standard_deviations(
            transcript.component_variances
        )
        projected_cells = self.layout.reconstruct_vector(
            projection.projected_components,
            transcript.public_total,
        )
        boundary = np.asarray(
            projected_cells <= float(active_set_tolerance),
            dtype=bool,
        )
        if boundary.shape != (self.layout.cell_dimension,):
            raise RuntimeError("P3 boundary mask has an invalid shape")
        strong = np.asarray(
            projection.dual_multipliers > float(multiplier_tolerance),
            dtype=bool,
        )
        self.active_cell_indices = np.flatnonzero(boundary).astype(np.int64)
        self.strong_active_cell_indices = np.flatnonzero(boundary & strong).astype(np.int64)
        self.weak_active_cell_indices = np.flatnonzero(boundary & ~strong).astype(np.int64)
        self._active_rows = self.layout.coefficient_csr_rows_for_cells(
            self.active_cell_indices
        )
        self._whitened_active_rows = self._active_rows.multiply(
            self.standard_deviation.reshape(1, -1)
        ).tocsr()
        self._linear_tolerance = float(linear_tolerance)
        self._linear_max_iterations = int(linear_max_iterations)
        self._dense_row_basis: np.ndarray | None = None
        self._active_rank: int | None = None

        dense_cells = int(
            self._whitened_active_rows.shape[0]
            * self._whitened_active_rows.shape[1]
        )
        if dense_cells <= int(dense_reference_cells):
            dense = self._whitened_active_rows.toarray()
            if dense.shape[0] == 0:
                self._dense_row_basis = np.empty(
                    (0, self.layout.coefficient_dimension),
                    dtype=np.float64,
                )
                self._active_rank = 0
            else:
                _, singular_values, right = np.linalg.svd(
                    dense,
                    full_matrices=False,
                )
                threshold = max(dense.shape) * np.finfo(np.float64).eps * max(
                    1.0,
                    float(singular_values[0]),
                )
                rank = int(np.sum(singular_values > threshold))
                self._dense_row_basis = right[:rank].copy()
                self._active_rank = rank
            self._application_mode = "dense_svd_reference"
        else:
            self._application_mode = "sparse_lsmr_matrix_free"

        inactive = projected_cells[~boundary]
        positive_multiplier = projection.dual_multipliers[strong]
        self._min_inactive_cell = (
            float(np.min(inactive)) if inactive.size else float("inf")
        )
        self._min_positive_multiplier = (
            float(np.min(positive_multiplier)) if positive_multiplier.size else float("inf")
        )
        self._active_set_stable = bool(
            self.weak_active_cell_indices.size == 0
            and self._min_inactive_cell > 10.0 * float(active_set_tolerance)
            and self._min_positive_multiplier > 10.0 * float(multiplier_tolerance)
        )

    @property
    def coefficient_dimension(self) -> int:
        return self.layout.coefficient_dimension

    @property
    def active_rank(self) -> int | None:
        return self._active_rank

    @property
    def effective_rank(self) -> int | None:
        if self._active_rank is None:
            return None
        return self.coefficient_dimension - self._active_rank

    @property
    def whitened_active_rows(self) -> csr_matrix:
        return self._whitened_active_rows

    def _vector(self, values: np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.coefficient_dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} must be finite with shape ({self.coefficient_dimension},)"
            )
        return vector

    def _matrix(self, values: np.ndarray, *, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.coefficient_dimension
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError(
                f"{name} must be finite with shape (n, {self.coefficient_dimension})"
            )
        return matrix

    def whitened_jacobian_matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="whitened values")
        if self._whitened_active_rows.shape[0] == 0:
            return vector.copy()
        if self._dense_row_basis is not None:
            row_component = self._dense_row_basis.T @ (
                self._dense_row_basis @ vector
            )
        else:
            solve = lsmr(
                self._whitened_active_rows.T,
                vector,
                atol=self._linear_tolerance,
                btol=self._linear_tolerance,
                conlim=1.0e12,
                maxiter=self._linear_max_iterations,
            )
            row_component = np.asarray(
                self._whitened_active_rows.T @ solve[0],
                dtype=np.float64,
            )
        return vector - row_component

    def coefficient_jacobian_matvec(self, raw_coefficient_delta: np.ndarray) -> np.ndarray:
        delta = self._vector(raw_coefficient_delta, name="raw_coefficient_delta")
        whitened = delta / self.standard_deviation
        return self.standard_deviation * self.whitened_jacobian_matvec(whitened)

    def coefficient_covariance_matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="covariance values")
        return self.standard_deviation * self.whitened_jacobian_matvec(
            self.standard_deviation * vector
        )

    def coefficient_tangent_precision_matvec(self, values: np.ndarray) -> np.ndarray:
        """Apply ``S^-1 P S^-1``, the fixed-face tangent generalized precision."""
        vector = self._vector(values, name="precision values")
        return self.whitened_jacobian_matvec(
            vector / self.standard_deviation
        ) / self.standard_deviation

    def coefficient_tangent_precision_matvec_many(
        self,
        values: np.ndarray,
        *,
        max_sparse_rows: int = 64,
    ) -> np.ndarray:
        matrix = self._matrix(values, name="precision values")
        whitened = matrix / self.standard_deviation.reshape(1, -1)
        if self._dense_row_basis is not None:
            projected = whitened - (
                (whitened @ self._dense_row_basis.T) @ self._dense_row_basis
            )
        else:
            if matrix.shape[0] > int(max_sparse_rows):
                raise ValueError(
                    "Sparse active-set precision is a reference path and cannot apply "
                    f"{matrix.shape[0]} rows at once; max_sparse_rows={int(max_sparse_rows)}"
                )
            projected = np.stack(
                [self.whitened_jacobian_matvec(row) for row in whitened],
                axis=0,
            )
        return projected / self.standard_deviation.reshape(1, -1)

    def dense_whitened_jacobian(self, *, max_dense_cells: int = 2_000_000) -> np.ndarray:
        dense_cells = self.coefficient_dimension * self.coefficient_dimension
        if dense_cells > int(max_dense_cells):
            raise ValueError(
                f"Dense Jacobian has {dense_cells} cells, exceeding max_dense_cells="
                f"{int(max_dense_cells)}"
            )
        identity = np.eye(self.coefficient_dimension, dtype=np.float64)
        return np.column_stack(
            [self.whitened_jacobian_matvec(identity[:, column]) for column in range(identity.shape[1])]
        )

    def certificate(
        self,
        *,
        num_probes: int = 3,
        seed: int = 20260714,
        tolerance: float = 1.0e-7,
    ) -> ActiveSetCovarianceCertificate:
        if int(num_probes) <= 0:
            raise ValueError("num_probes must be positive")
        if not np.isfinite(tolerance) or float(tolerance) <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        rng = np.random.default_rng(int(seed))
        nullspace_residual = 0.0
        symmetry_residual = 0.0
        idempotence_residual = 0.0
        minimum_quadratic = float("inf")
        for _ in range(int(num_probes)):
            left = rng.normal(size=self.coefficient_dimension)
            right = rng.normal(size=self.coefficient_dimension)
            projected_left = self.whitened_jacobian_matvec(left)
            projected_right = self.whitened_jacobian_matvec(right)
            active_residual = np.asarray(
                self._whitened_active_rows @ projected_left,
                dtype=np.float64,
            )
            if active_residual.size:
                nullspace_residual = max(
                    nullspace_residual,
                    float(np.max(np.abs(active_residual))),
                )
            symmetry_residual = max(
                symmetry_residual,
                abs(float(left @ projected_right - projected_left @ right)),
            )
            second_projection = self.whitened_jacobian_matvec(projected_left)
            idempotence_residual = max(
                idempotence_residual,
                float(np.max(np.abs(second_projection - projected_left))),
            )
            covariance_value = float(left @ self.coefficient_covariance_matvec(left))
            minimum_quadratic = min(minimum_quadratic, covariance_value)

        scale = max(1.0, float(self.coefficient_dimension))
        passed = bool(
            nullspace_residual <= float(tolerance) * scale
            and symmetry_residual <= float(tolerance) * scale
            and idempotence_residual <= float(tolerance) * scale
            and minimum_quadratic >= -float(tolerance) * scale
        )
        return ActiveSetCovarianceCertificate(
            passed=passed,
            nullspace_max_residual=nullspace_residual,
            symmetry_max_residual=symmetry_residual,
            idempotence_max_residual=idempotence_residual,
            covariance_min_quadratic=minimum_quadratic,
            tolerance=float(tolerance),
            num_probes=int(num_probes),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": "fixed_active_set_orthogonal_coefficient_covariance",
            "application_mode": self._application_mode,
            "coefficient_dimension": self.coefficient_dimension,
            "active_constraint_count": int(self.active_cell_indices.size),
            "strong_active_constraint_count": int(self.strong_active_cell_indices.size),
            "weak_active_constraint_count": int(self.weak_active_cell_indices.size),
            "active_constraint_rank": self.active_rank,
            "effective_rank": self.effective_rank,
            "min_inactive_cell": self._min_inactive_cell,
            "min_positive_multiplier": self._min_positive_multiplier,
            "active_set_stable": self._active_set_stable,
            "bootdiag_fallback_recommended": not self._active_set_stable,
            "precision_semantics": "fixed_face_tangent_generalized_precision",
        }
