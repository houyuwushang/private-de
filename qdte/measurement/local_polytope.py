from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    HierarchicalPairStrategy,
)
from qdte.queries.orthogonal import helmert_contrast


@dataclass(frozen=True)
class _VectorBlock:
    name: str
    kind: str
    scope: tuple[int, ...]
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def vector_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class LocalPolytopeMarginals:
    """Nonnegative one-way and pair tables in a shared-marginal parameterization."""

    oneway: dict[int, np.ndarray]
    pairs: dict[tuple[int, int], np.ndarray]

    def min_cell(self) -> float:
        arrays = [*self.oneway.values(), *self.pairs.values()]
        return min(float(np.min(values)) for values in arrays)

    def max_consistency_violation(self) -> float:
        violation = 0.0
        for (left, right), table in self.pairs.items():
            violation = max(
                violation,
                float(np.max(np.abs(np.sum(table, axis=1) - self.oneway[left]))),
                float(np.max(np.abs(np.sum(table, axis=0) - self.oneway[right]))),
            )
        return violation

    def max_total_violation(self, total: float) -> float:
        expected = float(total)
        return max(
            [abs(float(np.sum(values)) - expected) for values in self.oneway.values()]
            + [abs(float(np.sum(values)) - expected) for values in self.pairs.values()]
        )


class LocalPolytopeLayout:
    """Matrix-free affine map from hierarchical coefficients to marginal cells.

    The hierarchy fixes every table total and shares each one-way contrast across
    all incident pair tables. Consequently, affine consistency is built into the
    parameterization and P3 only needs cell nonnegativity constraints.
    """

    def __init__(self, strategy: HierarchicalPairStrategy):
        self.strategy = strategy
        cards = strategy.cardinalities

        coefficient_blocks: list[_VectorBlock] = []
        coefficient_offset = 0
        for block in strategy.blocks:
            stop = coefficient_offset + block.dimension
            coefficient_blocks.append(
                _VectorBlock(
                    name=block.name,
                    kind=block.kind,
                    scope=block.scope,
                    shape=block.coefficient_shape,
                    start=coefficient_offset,
                    stop=stop,
                )
            )
            coefficient_offset = stop

        cell_blocks: list[_VectorBlock] = []
        cell_offset = 0
        for attr, cardinality in enumerate(cards):
            stop = cell_offset + int(cardinality)
            cell_blocks.append(
                _VectorBlock(
                    name=f"oneway:{attr}",
                    kind="oneway",
                    scope=(attr,),
                    shape=(int(cardinality),),
                    start=cell_offset,
                    stop=stop,
                )
            )
            cell_offset = stop
        for left, right in strategy.pairs:
            shape = (int(cards[left]), int(cards[right]))
            stop = cell_offset + int(np.prod(shape, dtype=np.int64))
            cell_blocks.append(
                _VectorBlock(
                    name=f"pair:{left}:{right}",
                    kind="pair",
                    scope=(left, right),
                    shape=shape,
                    start=cell_offset,
                    stop=stop,
                )
            )
            cell_offset = stop

        self._coefficient_blocks = tuple(coefficient_blocks)
        self._cell_blocks = tuple(cell_blocks)
        self._coefficient_by_name = {block.name: block for block in coefficient_blocks}
        self._cell_by_name = {block.name: block for block in cell_blocks}
        self._contrasts = {
            attr: helmert_contrast(cardinality)
            for attr, cardinality in enumerate(cards)
        }
        self._cell_stops = np.asarray([block.stop for block in cell_blocks], dtype=np.int64)
        self.coefficient_dimension = coefficient_offset
        self.cell_dimension = cell_offset

    def _coefficient_vector(self, values: np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.coefficient_dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} must be finite with shape ({self.coefficient_dimension},)"
            )
        return vector

    def _cell_vector(self, values: np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.cell_dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be finite with shape ({self.cell_dimension},)")
        return vector

    def flatten_components(self, components: dict[str, np.ndarray]) -> np.ndarray:
        expected = set(self._coefficient_by_name)
        if set(components) != expected:
            raise ValueError(
                "Component block set mismatch: "
                f"missing={sorted(expected - set(components))}, "
                f"extra={sorted(set(components) - expected)}"
            )
        vector = np.empty(self.coefficient_dimension, dtype=np.float64)
        for block in self._coefficient_blocks:
            values = np.asarray(components[block.name], dtype=np.float64)
            if values.shape != block.shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Component {block.name!r} must be finite with shape {block.shape}"
                )
            vector[block.vector_slice] = values.reshape(-1)
        return vector

    def unflatten_components(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        values = self._coefficient_vector(vector, name="coefficient vector")
        return {
            block.name: values[block.vector_slice].reshape(block.shape).copy()
            for block in self._coefficient_blocks
        }

    def coefficient_standard_deviations(
        self,
        component_variances: dict[str, float],
    ) -> np.ndarray:
        expected = set(self._coefficient_by_name)
        if set(component_variances) != expected:
            raise ValueError("Component variance block set does not match the strategy")
        standard_deviation = np.empty(self.coefficient_dimension, dtype=np.float64)
        for block in self._coefficient_blocks:
            variance = float(component_variances[block.name])
            if not np.isfinite(variance) or variance <= 0.0:
                raise ValueError(f"Variance for {block.name!r} must be finite and positive")
            standard_deviation[block.vector_slice] = np.sqrt(variance)
        return standard_deviation

    def coefficient_delta_to_cells(self, coefficient_delta: np.ndarray) -> np.ndarray:
        """Apply the linear coefficient-to-cell reconstruction map ``R``."""
        delta = self._coefficient_vector(coefficient_delta, name="coefficient_delta")
        cells = np.empty(self.cell_dimension, dtype=np.float64)

        for attr, cardinality in enumerate(self.strategy.cardinalities):
            coefficient = self._coefficient_by_name[f"oneway_contrast:{attr}"]
            cell = self._cell_by_name[f"oneway:{attr}"]
            cells[cell.vector_slice] = self._contrasts[attr] @ delta[coefficient.vector_slice]

        for left, right in self.strategy.pairs:
            left_coefficient = self._coefficient_by_name[f"oneway_contrast:{left}"]
            right_coefficient = self._coefficient_by_name[f"oneway_contrast:{right}"]
            interaction = self._coefficient_by_name[f"pair_interaction:{left}:{right}"]
            cell = self._cell_by_name[f"pair:{left}:{right}"]
            left_contrast = self._contrasts[left]
            right_contrast = self._contrasts[right]
            left_effect = left_contrast @ delta[left_coefficient.vector_slice]
            right_effect = right_contrast @ delta[right_coefficient.vector_slice]
            interaction_effect = (
                left_contrast
                @ delta[interaction.vector_slice].reshape(interaction.shape)
                @ right_contrast.T
            )
            table = (
                left_effect[:, None] / float(cell.shape[1])
                + right_effect[None, :] / float(cell.shape[0])
                + interaction_effect
            )
            cells[cell.vector_slice] = table.reshape(-1)
        return cells

    def cell_values_to_coefficient_adjoint(self, cell_values: np.ndarray) -> np.ndarray:
        """Apply ``R.T`` without materializing the global reconstruction matrix."""
        values = self._cell_vector(cell_values, name="cell_values")
        coefficients = np.zeros(self.coefficient_dimension, dtype=np.float64)

        for attr in range(len(self.strategy.cardinalities)):
            coefficient = self._coefficient_by_name[f"oneway_contrast:{attr}"]
            cell = self._cell_by_name[f"oneway:{attr}"]
            coefficients[coefficient.vector_slice] += (
                self._contrasts[attr].T @ values[cell.vector_slice]
            )

        for left, right in self.strategy.pairs:
            left_coefficient = self._coefficient_by_name[f"oneway_contrast:{left}"]
            right_coefficient = self._coefficient_by_name[f"oneway_contrast:{right}"]
            interaction = self._coefficient_by_name[f"pair_interaction:{left}:{right}"]
            cell = self._cell_by_name[f"pair:{left}:{right}"]
            multipliers = values[cell.vector_slice].reshape(cell.shape)
            left_contrast = self._contrasts[left]
            right_contrast = self._contrasts[right]
            coefficients[left_coefficient.vector_slice] += (
                left_contrast.T @ np.sum(multipliers, axis=1) / float(cell.shape[1])
            )
            coefficients[right_coefficient.vector_slice] += (
                right_contrast.T @ np.sum(multipliers, axis=0) / float(cell.shape[0])
            )
            coefficients[interaction.vector_slice] += (
                left_contrast.T @ multipliers @ right_contrast
            ).reshape(-1)
        return coefficients

    def coefficient_rows_for_cells(self, cell_indices: np.ndarray) -> np.ndarray:
        """Return selected rows of ``R`` without materializing the full matrix."""
        return self.coefficient_csr_rows_for_cells(cell_indices).toarray()

    def coefficient_csr_rows_for_cells(self, cell_indices: np.ndarray) -> csr_matrix:
        """Return selected rows of ``R`` as a sparse active-constraint matrix."""
        indices = np.asarray(cell_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.cell_dimension):
            raise ValueError("cell_indices must be a one-dimensional in-range index vector")
        data: list[float] = []
        columns: list[int] = []
        row_offsets = [0]
        for cell_index in indices.tolist():
            block_index = int(np.searchsorted(self._cell_stops, cell_index, side="right"))
            cell = self._cell_blocks[block_index]
            local_index = int(cell_index - cell.start)
            if cell.kind == "oneway":
                attr = cell.scope[0]
                coefficient = self._coefficient_by_name[f"oneway_contrast:{attr}"]
                values = self._contrasts[attr][local_index]
                data.extend(values.tolist())
                columns.extend(range(coefficient.start, coefficient.stop))
            else:
                left, right = cell.scope
                row, column = np.unravel_index(local_index, cell.shape)
                left_coefficient = self._coefficient_by_name[f"oneway_contrast:{left}"]
                right_coefficient = self._coefficient_by_name[f"oneway_contrast:{right}"]
                interaction = self._coefficient_by_name[f"pair_interaction:{left}:{right}"]
                left_values = self._contrasts[left][row] / float(cell.shape[1])
                right_values = self._contrasts[right][column] / float(cell.shape[0])
                interaction_values = np.outer(
                    self._contrasts[left][row],
                    self._contrasts[right][column],
                ).reshape(-1)
                data.extend(left_values.tolist())
                columns.extend(range(left_coefficient.start, left_coefficient.stop))
                data.extend(right_values.tolist())
                columns.extend(range(right_coefficient.start, right_coefficient.stop))
                data.extend(interaction_values.tolist())
                columns.extend(range(interaction.start, interaction.stop))
            row_offsets.append(len(data))
        return csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(columns, dtype=np.int64),
                np.asarray(row_offsets, dtype=np.int64),
            ),
            shape=(indices.size, self.coefficient_dimension),
        )

    def reconstruct_vector(self, components: dict[str, np.ndarray], total: float) -> np.ndarray:
        if not np.isfinite(total) or float(total) <= 0.0:
            raise ValueError("total must be finite and positive")
        baseline = np.empty(self.cell_dimension, dtype=np.float64)
        for block in self._cell_blocks:
            baseline[block.vector_slice] = float(total) / float(np.prod(block.shape))
        return baseline + self.coefficient_delta_to_cells(self.flatten_components(components))

    def unflatten_cells(self, cell_vector: np.ndarray) -> LocalPolytopeMarginals:
        values = self._cell_vector(cell_vector, name="cell_vector")
        oneway: dict[int, np.ndarray] = {}
        pairs: dict[tuple[int, int], np.ndarray] = {}
        for block in self._cell_blocks:
            table = values[block.vector_slice].reshape(block.shape).copy()
            if block.kind == "oneway":
                oneway[block.scope[0]] = table
            else:
                pairs[(block.scope[0], block.scope[1])] = table
        return LocalPolytopeMarginals(oneway=oneway, pairs=pairs)


@dataclass(frozen=True)
class LocalPolytopeProjectionResult:
    projected_components: dict[str, np.ndarray]
    marginals: LocalPolytopeMarginals
    standardized_correction: np.ndarray
    dual_multipliers: np.ndarray
    active_cell_mask: np.ndarray
    diagnostics: dict[str, Any]


def _polish_active_set(
    *,
    layout: LocalPolytopeLayout,
    raw_cells: np.ndarray,
    standard_deviation: np.ndarray,
    initial_multipliers: np.ndarray,
    feasibility_tolerance: float,
    stationarity_tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Polish a dual estimate through minimum-norm active-set projections."""
    multiplier_seed_tolerance = max(1.0e-12, float(stationarity_tolerance) * 1.0e-3)
    active = set(
        np.flatnonzero(initial_multipliers > multiplier_seed_tolerance).astype(int).tolist()
    )
    standardized = np.zeros(layout.coefficient_dimension, dtype=np.float64)
    multipliers = np.zeros(layout.cell_dimension, dtype=np.float64)
    projected_cells = raw_cells.copy()
    message = "active-set iteration limit reached"
    success = False
    active_rank: int | None = 0
    active_equality_residual = 0.0
    linear_solver = "not_needed"
    linear_solver_iterations = 0
    linear_solver_stop_code = 0

    for iteration in range(1, int(max_iterations) + 1):
        if not active:
            most_violated = int(np.argmin(projected_cells))
            if projected_cells[most_violated] >= -float(feasibility_tolerance):
                success = True
                message = "no active constraints required"
                break
            active.add(most_violated)

        active_indices = np.asarray(sorted(active), dtype=np.int64)
        active_rows = layout.coefficient_csr_rows_for_cells(active_indices)
        whitened_rows = active_rows.multiply(standard_deviation.reshape(1, -1)).tocsr()
        active_rhs = -raw_cells[active_indices]
        dense_cells = int(whitened_rows.shape[0] * whitened_rows.shape[1])
        if dense_cells <= 250_000:
            dense_rows = whitened_rows.toarray()
            standardized = np.linalg.lstsq(
                dense_rows,
                active_rhs,
                rcond=1.0e-12,
            )[0]
            active_lagrange = np.linalg.lstsq(
                dense_rows.T,
                standardized,
                rcond=1.0e-12,
            )[0]
            active_rank = int(np.linalg.matrix_rank(dense_rows, tol=1.0e-11))
            linear_solver = "dense_lstsq_small_active_set"
            linear_solver_iterations = 1
            linear_solver_stop_code = 0
        else:
            rhs_scale = 1.0 + float(np.max(np.abs(active_rhs)))
            linear_tolerance = max(
                1.0e-14,
                min(1.0e-10, float(feasibility_tolerance) / (100.0 * rhs_scale)),
            )
            linear_max_iterations = max(
                1_000,
                min(50_000, 20 * max(whitened_rows.shape)),
            )
            primal_solve = lsmr(
                whitened_rows,
                active_rhs,
                atol=linear_tolerance,
                btol=linear_tolerance,
                conlim=1.0e12,
                maxiter=linear_max_iterations,
            )
            standardized = np.asarray(primal_solve[0], dtype=np.float64)
            dual_solve = lsmr(
                whitened_rows.T,
                standardized,
                atol=linear_tolerance,
                btol=linear_tolerance,
                conlim=1.0e12,
                maxiter=linear_max_iterations,
            )
            active_lagrange = np.asarray(dual_solve[0], dtype=np.float64)
            active_rank = None
            linear_solver = "sparse_lsmr_active_set"
            linear_solver_iterations = int(primal_solve[2]) + int(dual_solve[2])
            linear_solver_stop_code = max(int(primal_solve[1]), int(dual_solve[1]))
        active_equality_residual = float(
            np.max(np.abs(whitened_rows @ standardized - active_rhs))
        )

        negative = np.flatnonzero(active_lagrange < -multiplier_seed_tolerance)
        if negative.size:
            remove_local = int(negative[np.argmin(active_lagrange[negative])])
            active.remove(int(active_indices[remove_local]))
            continue
        if active_equality_residual > float(feasibility_tolerance):
            remove_local = int(np.argmin(np.abs(active_lagrange)))
            active.remove(int(active_indices[remove_local]))
            continue

        coefficient_correction = standard_deviation * standardized
        projected_cells = raw_cells + layout.coefficient_delta_to_cells(
            coefficient_correction
        )
        inactive_mask = np.ones(layout.cell_dimension, dtype=bool)
        inactive_mask[active_indices] = False
        inactive_indices = np.flatnonzero(inactive_mask)
        if inactive_indices.size:
            most_violated = int(
                inactive_indices[np.argmin(projected_cells[inactive_indices])]
            )
            if projected_cells[most_violated] < -float(feasibility_tolerance):
                active.add(most_violated)
                continue

        multipliers.fill(0.0)
        multipliers[active_indices] = np.maximum(active_lagrange, 0.0)
        success = True
        message = "active set polished"
        break
    else:
        iteration = int(max_iterations)

    return standardized, multipliers, projected_cells, {
        "active_set_polish_success": success,
        "active_set_polish_message": message,
        "active_set_polish_iterations": int(iteration),
        "active_set_polish_active_count": len(active),
        "active_set_polish_rank": active_rank,
        "active_set_polish_equality_max_residual": active_equality_residual,
        "active_set_polish_linear_solver": linear_solver,
        "active_set_polish_linear_solver_iterations": linear_solver_iterations,
        "active_set_polish_linear_solver_stop_code": linear_solver_stop_code,
    }


def project_hierarchical_local_polytope(
    transcript: HierarchicalInteractionTranscript,
    *,
    feasibility_tolerance: float = 1.0e-7,
    stationarity_tolerance: float = 1.0e-7,
    complementarity_tolerance: float = 1.0e-6,
    gap_absolute_tolerance: float = 1.0e-6,
    gap_relative_tolerance: float = 1.0e-8,
    active_set_tolerance: float = 1.0e-7,
    solver_ftol: float = 1.0e-13,
    solver_gtol: float = 1.0e-9,
    solver_max_iterations: int = 5_000,
    require_certificate: bool = True,
) -> LocalPolytopeProjectionResult:
    """Project released orthogonal coefficients onto the nonnegative local polytope.

    This function reads only the released transcript and public schema/row count.
    The objective is the exact raw coefficient-space Gaussian likelihood.
    """
    tolerances = (
        feasibility_tolerance,
        stationarity_tolerance,
        complementarity_tolerance,
        gap_absolute_tolerance,
        gap_relative_tolerance,
        active_set_tolerance,
        solver_ftol,
        solver_gtol,
    )
    if any(not np.isfinite(value) or float(value) < 0.0 for value in tolerances):
        raise ValueError("Projection tolerances must be finite and nonnegative")
    if int(solver_max_iterations) <= 0:
        raise ValueError("solver_max_iterations must be positive")
    if transcript.adjacency != "add_remove":
        raise ValueError("ICE local-polytope P3 currently requires add_remove adjacency")

    layout = LocalPolytopeLayout(transcript.strategy)
    noisy = layout.flatten_components(transcript.noisy_components)
    standard_deviation = layout.coefficient_standard_deviations(
        transcript.component_variances
    )
    raw_cells = layout.reconstruct_vector(transcript.noisy_components, transcript.public_total)
    rhs = -raw_cells

    def dual_terms(multipliers: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        lagrange = np.asarray(multipliers, dtype=np.float64)
        coefficient_adjoint = layout.cell_values_to_coefficient_adjoint(lagrange)
        standardized = standard_deviation * coefficient_adjoint
        coefficient_correction = standard_deviation * standardized
        cell_correction = layout.coefficient_delta_to_cells(coefficient_correction)
        value = 0.5 * float(standardized @ standardized) - float(rhs @ lagrange)
        gradient = cell_correction - rhs
        return value, gradient, standardized

    if float(np.min(raw_cells)) >= -float(feasibility_tolerance):
        solver_success = True
        solver_status = 0
        solver_message = "raw equality-consistent target is already nonnegative"
        solver_iterations = 0
        solver_function_evaluations = 1
        multipliers = np.zeros(layout.cell_dimension, dtype=np.float64)
        _, dual_gradient, standardized = dual_terms(multipliers)
        polish_diagnostics = {
            "active_set_polish_success": True,
            "active_set_polish_message": "raw target already feasible",
            "active_set_polish_iterations": 0,
            "active_set_polish_active_count": 0,
            "active_set_polish_rank": 0,
            "active_set_polish_equality_max_residual": 0.0,
            "active_set_polish_linear_solver": "not_needed",
            "active_set_polish_linear_solver_iterations": 0,
            "active_set_polish_linear_solver_stop_code": 0,
        }
    else:
        result = minimize(
            lambda multipliers: dual_terms(multipliers)[0],
            np.zeros(layout.cell_dimension, dtype=np.float64),
            jac=lambda multipliers: dual_terms(multipliers)[1],
            bounds=[(0.0, None)] * layout.cell_dimension,
            method="L-BFGS-B",
            options={
                "ftol": float(solver_ftol),
                "gtol": float(solver_gtol),
                "maxiter": int(solver_max_iterations),
                "maxls": 100,
                "maxcor": 50,
            },
        )
        solver_success = bool(result.success)
        solver_status = int(result.status)
        solver_message = str(result.message)
        solver_iterations = int(result.nit)
        solver_function_evaluations = int(result.nfev)
        multipliers = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        _, dual_gradient, standardized = dual_terms(multipliers)
        polished_standardized, polished_multipliers, _, polish_diagnostics = (
            _polish_active_set(
                layout=layout,
                raw_cells=raw_cells,
                standard_deviation=standard_deviation,
                initial_multipliers=multipliers,
                feasibility_tolerance=float(feasibility_tolerance),
                stationarity_tolerance=float(stationarity_tolerance),
                max_iterations=max(10, min(layout.cell_dimension * 2, 5_000)),
            )
        )
        if bool(polish_diagnostics["active_set_polish_success"]):
            multipliers = polished_multipliers
            _, dual_gradient, dual_standardized = dual_terms(multipliers)
            polish_mismatch = float(
                np.max(np.abs(polished_standardized - dual_standardized))
            )
            polish_diagnostics["active_set_polish_stationarity_mismatch"] = polish_mismatch
            standardized = dual_standardized
        else:
            polish_diagnostics["active_set_polish_stationarity_mismatch"] = None

    coefficient_correction = standard_deviation * standardized
    projected = noisy + coefficient_correction
    projected_components = layout.unflatten_components(projected)
    projected_cells = layout.reconstruct_vector(projected_components, transcript.public_total)
    marginals = layout.unflatten_cells(projected_cells)

    coefficient_adjoint = layout.cell_values_to_coefficient_adjoint(multipliers)
    stationarity = standardized - standard_deviation * coefficient_adjoint
    primal_objective = 0.5 * float(standardized @ standardized)
    dual_objective = float(rhs @ multipliers) - 0.5 * float(
        (standard_deviation * coefficient_adjoint)
        @ (standard_deviation * coefficient_adjoint)
    )
    raw_duality_gap = primal_objective - dual_objective
    gap_tolerance = float(gap_absolute_tolerance) + float(gap_relative_tolerance) * max(
        1.0,
        abs(primal_objective),
        abs(dual_objective),
    )
    lower_bound_violation = max(0.0, -float(np.min(projected_cells)))
    equality_violation = max(
        marginals.max_consistency_violation(),
        marginals.max_total_violation(transcript.public_total),
    )
    dual_feasibility_violation = max(0.0, -float(np.min(multipliers)))
    stationarity_residual = float(np.max(np.abs(stationarity)))
    complementarity = multipliers * projected_cells
    complementarity_residual = float(np.max(np.abs(complementarity)))
    positive_multiplier = multipliers > float(active_set_tolerance)
    projected_dual_gradient = np.where(
        positive_multiplier,
        dual_gradient,
        np.minimum(dual_gradient, 0.0),
    )
    projected_gradient_residual = float(np.max(np.abs(projected_dual_gradient)))
    active_cell_mask = projected_cells <= float(active_set_tolerance)

    gap_passed = bool(
        np.isfinite(raw_duality_gap)
        and raw_duality_gap >= -gap_tolerance
        and max(0.0, raw_duality_gap) <= gap_tolerance
    )
    certificate_passed = bool(
        equality_violation <= float(feasibility_tolerance)
        and lower_bound_violation <= float(feasibility_tolerance)
        and dual_feasibility_violation <= float(feasibility_tolerance)
        and stationarity_residual <= float(stationarity_tolerance)
        and complementarity_residual <= float(complementarity_tolerance)
        and gap_passed
    )
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "method": "hierarchical_orthogonal_local_polytope_p3",
        "parameterization": "shared_hierarchical_orthogonal_coefficients",
        "partial_transcript_semantics": "measured_blocks_only_no_fake_variance",
        "solver": "lbfgsb_matrix_free_inequality_dual",
        "solver_success": solver_success,
        "solver_status": solver_status,
        "solver_message": solver_message,
        "solver_iterations": solver_iterations,
        "solver_function_evaluations": solver_function_evaluations,
        "coefficient_dimension": layout.coefficient_dimension,
        "cell_constraint_count": layout.cell_dimension,
        "dense_constraint_cells_avoided": layout.coefficient_dimension * layout.cell_dimension,
        "initial_min_cell": float(np.min(raw_cells)),
        "final_min_cell": float(np.min(projected_cells)),
        "active_cell_count": int(np.sum(active_cell_mask)),
        "positive_multiplier_count": int(np.sum(positive_multiplier)),
        "weighted_objective": primal_objective,
        "standardized_correction_l2": float(np.linalg.norm(standardized)),
        "coefficient_correction_l2": float(np.linalg.norm(coefficient_correction)),
        "certificate_passed": certificate_passed,
        "certificate_equality_max_violation": equality_violation,
        "certificate_lower_bound_violation": lower_bound_violation,
        "certificate_dual_feasibility_violation": dual_feasibility_violation,
        "certificate_stationarity_max_residual": stationarity_residual,
        "certificate_complementarity_max_residual": complementarity_residual,
        "certificate_projected_dual_gradient_max_residual": projected_gradient_residual,
        "certificate_primal_objective": primal_objective,
        "certificate_dual_lower_bound": dual_objective,
        "certificate_raw_duality_gap": raw_duality_gap,
        "certificate_gap_tolerance": gap_tolerance,
        "certificate_gap_passed": gap_passed,
        "certificate_feasibility_tolerance": float(feasibility_tolerance),
        "certificate_stationarity_tolerance": float(stationarity_tolerance),
        "certificate_complementarity_tolerance": float(complementarity_tolerance),
        "active_set_tolerance": float(active_set_tolerance),
        "uncertainty_status": "raw_covariance_not_yet_propagated",
    }
    diagnostics.update(polish_diagnostics)
    projection_result = LocalPolytopeProjectionResult(
        projected_components=projected_components,
        marginals=marginals,
        standardized_correction=standardized.copy(),
        dual_multipliers=multipliers.copy(),
        active_cell_mask=active_cell_mask.copy(),
        diagnostics=diagnostics,
    )
    if bool(require_certificate) and not certificate_passed:
        raise RuntimeError(
            "ICE local-polytope P3 failed its KKT certificate: "
            f"feasibility={max(equality_violation, lower_bound_violation):.3e}, "
            f"complementarity={complementarity_residual:.3e}, "
            f"gap={raw_duality_gap:.3e}"
        )
    return projection_result


def offline_local_polytope_dominance_diagnostics(
    *,
    transcript: HierarchicalInteractionTranscript,
    projection: LocalPolytopeProjectionResult,
    truth_components: dict[str, np.ndarray],
    tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Evaluate P3 target dominance with truth; never call from a DP generation path."""
    if not np.isfinite(tolerance) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    layout = LocalPolytopeLayout(transcript.strategy)
    noisy = layout.flatten_components(transcript.noisy_components)
    projected = layout.flatten_components(projection.projected_components)
    truth = layout.flatten_components(truth_components)
    standard_deviation = layout.coefficient_standard_deviations(
        transcript.component_variances
    )
    raw_to_truth = float(np.sum(((noisy - truth) / standard_deviation) ** 2))
    projected_to_truth = float(np.sum(((projected - truth) / standard_deviation) ** 2))
    projection_shift = float(np.sum(((projected - noisy) / standard_deviation) ** 2))
    dominance_slack = raw_to_truth - projection_shift - projected_to_truth
    scaled_tolerance = float(tolerance) * max(
        1.0,
        raw_to_truth,
        projected_to_truth,
        projection_shift,
    )
    return {
        "offline_truth_only": True,
        "norm": "raw_coefficient_gaussian_precision",
        "raw_to_truth_omega_squared": raw_to_truth,
        "projected_to_truth_omega_squared": projected_to_truth,
        "raw_to_projected_omega_squared": projection_shift,
        "dominance_slack": dominance_slack,
        "dominance_tolerance": scaled_tolerance,
        "dominance_passed": bool(dominance_slack >= -scaled_tolerance),
    }
