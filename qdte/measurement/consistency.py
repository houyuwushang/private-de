from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from qdte.measurement.projection import project_simplex
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue


@dataclass
class ConsistencyProjectionResult:
    projected: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class QuerySpaceProjectionResult:
    projected: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class _ScopeModel:
    attrs: tuple[int, ...]
    shape: tuple[int, ...]
    qids: list[int]
    masks: list[np.ndarray]
    table: np.ndarray
    precision: float

    @property
    def num_cells(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


@dataclass
class _CompleteCellPartition:
    scope: tuple[int, ...]
    assignments: dict[tuple[int, ...], int]

    @property
    def qids(self) -> list[int]:
        return list(self.assignments.values())


def _query_scope(qcat: QueryCatalogue, qid: int) -> tuple[int, ...]:
    attrs = {term[0] for term in qcat.query_terms(qid)}
    attrs.update(attr for attr, _ in qcat.linear_terms(qid))
    return tuple(sorted(attrs))


def _cell_assignment(qcat: QueryCatalogue, qid: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if int(qcat.linear_num_terms[qid]) > 0:
        return None
    terms = qcat.query_terms(qid)
    if not terms:
        return None
    if any(op != OP_EQ for _, op, _, _, _ in terms):
        return None
    ordered = sorted((int(attr), int(value)) for attr, _, value, _, _ in terms)
    scope = tuple(attr for attr, _ in ordered)
    assignment = tuple(value for _, value in ordered)
    if len(set(scope)) != len(scope):
        return None
    return scope, assignment


def _complete_cell_partitions(qcat: QueryCatalogue, cardinalities: np.ndarray) -> list[_CompleteCellPartition]:
    by_scope: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {}
    for qid in range(qcat.m):
        cell = _cell_assignment(qcat, qid)
        if cell is None:
            continue
        scope, assignment = cell
        if any(value < 0 or value >= int(cardinalities[attr]) for attr, value in zip(scope, assignment, strict=True)):
            continue
        by_scope.setdefault(scope, {})[assignment] = qid

    partitions: list[_CompleteCellPartition] = []
    for scope, assignments in sorted(by_scope.items()):
        shape = tuple(int(cardinalities[attr]) for attr in scope)
        expected = int(np.prod(shape, dtype=np.int64))
        if expected <= 0 or len(assignments) != expected:
            continue
        all_assignments = product(*(range(size) for size in shape))
        if all(tuple(values) in assignments for values in all_assignments):
            partitions.append(_CompleteCellPartition(scope=scope, assignments=assignments))
    return partitions


def _assignment_satisfies_query(
    qcat: QueryCatalogue,
    qid: int,
    partition_scope: tuple[int, ...],
    assignment: tuple[int, ...],
) -> bool:
    values_by_attr = {attr: value for attr, value in zip(partition_scope, assignment, strict=True)}
    for attr, op, value, lo, hi in qcat.query_terms(qid):
        if attr not in values_by_attr:
            return False
        observed = int(values_by_attr[attr])
        if op == OP_EQ:
            if observed != int(value):
                return False
        elif op == OP_LE:
            if observed > int(value):
                return False
        elif op == OP_GE:
            if observed < int(value):
                return False
        elif op == OP_RANGE:
            if observed < int(lo) or observed > int(hi):
                return False
        else:
            raise NotImplementedError(f"Unsupported query op {op}; query-space projection supports EQ/LE/GE/RANGE")
    if int(qcat.linear_num_terms[qid]) > 0:
        score = 0.0
        for attr, weight in qcat.linear_terms(qid):
            if attr not in values_by_attr:
                return False
            score += float(weight) * float(values_by_attr[attr])
        if score > float(qcat.linear_thresholds[qid]):
            return False
    return True


class _ConstraintBuilder:
    def __init__(self, num_queries: int):
        self.num_queries = int(num_queries)
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.rhs: list[float] = []
        self._seen: set[tuple[tuple[tuple[int, float], ...], float]] = set()
        self.num_duplicate_constraints = 0
        self.num_zero_constraints = 0

    def add(self, coeffs: dict[int, float], rhs: float) -> bool:
        cleaned: dict[int, float] = {}
        for qid, coef in coeffs.items():
            q = int(qid)
            c = float(coef)
            if q < 0 or q >= self.num_queries:
                raise IndexError(f"Query id {q} is outside [0, {self.num_queries})")
            if abs(c) <= 1.0e-12:
                continue
            cleaned[q] = cleaned.get(q, 0.0) + c
        cleaned = {qid: coef for qid, coef in cleaned.items() if abs(coef) > 1.0e-12}
        if not cleaned:
            self.num_zero_constraints += 1
            return False
        key = (tuple(sorted((qid, round(coef, 12)) for qid, coef in cleaned.items())), round(float(rhs), 12))
        if key in self._seen:
            self.num_duplicate_constraints += 1
            return False
        self._seen.add(key)
        row = len(self.rhs)
        for qid, coef in sorted(cleaned.items()):
            self.rows.append(row)
            self.cols.append(qid)
            self.data.append(coef)
        self.rhs.append(float(rhs))
        return True

    def build(self) -> tuple[csr_matrix, np.ndarray]:
        matrix = csr_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.rhs), self.num_queries),
            dtype=np.float64,
        )
        return matrix, np.asarray(self.rhs, dtype=np.float64)


def _build_query_space_constraints(
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    total: int,
    max_constraints: int,
) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    builder = _ConstraintBuilder(qcat.m)
    partitions = _complete_cell_partitions(qcat, cardinalities)
    partition_by_scope = {partition.scope: partition for partition in partitions}

    num_partition_sum_constraints = 0
    num_query_from_partition_constraints = 0
    num_skipped_by_constraint_cap = 0
    for partition in partitions:
        if len(builder.rhs) >= max_constraints:
            num_skipped_by_constraint_cap += 1
            continue
        if builder.add({qid: 1.0 for qid in partition.qids}, float(total)):
            num_partition_sum_constraints += 1

    for partition in partitions:
        cell_items = list(partition.assignments.items())
        partition_scope_set = set(partition.scope)
        for qid in range(qcat.m):
            if len(builder.rhs) >= max_constraints:
                num_skipped_by_constraint_cap += 1
                continue
            q_scope = _query_scope(qcat, qid)
            if not q_scope or not set(q_scope).issubset(partition_scope_set):
                continue
            matches = [
                cell_qid
                for assignment, cell_qid in cell_items
                if _assignment_satisfies_query(qcat, qid, partition.scope, assignment)
            ]
            coeffs = {qid: 1.0}
            for cell_qid in matches:
                coeffs[cell_qid] = coeffs.get(cell_qid, 0.0) - 1.0
            if builder.add(coeffs, 0.0):
                num_query_from_partition_constraints += 1

    matrix, rhs = builder.build()
    diagnostics = {
        "num_complete_cell_partitions": int(len(partitions)),
        "complete_cell_partition_scopes": [list(scope) for scope in sorted(partition_by_scope)],
        "num_constraints": int(matrix.shape[0]),
        "num_constraint_nonzeros": int(matrix.nnz),
        "num_partition_sum_constraints": int(num_partition_sum_constraints),
        "num_query_from_partition_constraints": int(num_query_from_partition_constraints),
        "num_duplicate_constraints": int(builder.num_duplicate_constraints),
        "num_zero_constraints": int(builder.num_zero_constraints),
        "num_skipped_by_constraint_cap": int(num_skipped_by_constraint_cap),
        "max_constraints": int(max_constraints),
    }
    return matrix, rhs, diagnostics


def project_query_space_lsq(
    noisy: np.ndarray,
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    total: int,
    variances: np.ndarray | None = None,
    max_constraints: int = 200_000,
    solver_atol: float = 1.0e-10,
    solver_btol: float = 1.0e-10,
    solver_max_iterations: int = 10_000,
) -> QuerySpaceProjectionResult:
    """Project noisy query answers onto exact linear consistency constraints.

    This is the equality-only weighted projection

        min_z 0.5 * (z - y)^T W (z - y),  subject to A z = b.

    No nonnegativity or clipping is applied, so the estimator remains affine in
    the noisy answers when the constraint matrix is fixed.
    """
    y = np.asarray(noisy, dtype=np.float64)
    if y.shape != (qcat.m,):
        raise ValueError(f"noisy must have shape ({qcat.m},), got {y.shape}")
    cards = np.asarray(cardinalities, dtype=np.int32)
    if np.any(cards <= 0):
        raise ValueError("cardinalities must be positive")
    if variances is None:
        winv = np.ones(qcat.m, dtype=np.float64)
    else:
        var = np.asarray(variances, dtype=np.float64)
        if var.shape != (qcat.m,):
            raise ValueError(f"variances must have shape ({qcat.m},), got {var.shape}")
        winv = np.maximum(var, 1.0e-12)

    constraints, rhs, diagnostics = _build_query_space_constraints(
        qcat,
        cards,
        int(total),
        max_constraints=int(max_constraints),
    )
    if constraints.shape[0] == 0:
        diagnostics.update(
            {
                "enabled": True,
                "method": "query_space_lsq",
                "solver": "none",
                "known_total_count": int(total),
                "initial_max_constraint_violation": 0.0,
                "final_max_constraint_violation": 0.0,
                "l2_correction": 0.0,
                "weighted_objective": 0.0,
            }
        )
        return QuerySpaceProjectionResult(projected=y.astype(np.float32), diagnostics=diagnostics)

    initial_violation = constraints @ y - rhs
    weighted_constraints = constraints.multiply(winv.reshape(1, -1))
    gram = weighted_constraints @ constraints.T
    solve = lsmr(
        gram,
        initial_violation,
        atol=float(solver_atol),
        btol=float(solver_btol),
        maxiter=int(solver_max_iterations),
    )
    lagrange = np.asarray(solve[0], dtype=np.float64)
    correction = winv * np.asarray(constraints.T @ lagrange, dtype=np.float64)
    projected = y - correction
    final_violation = constraints @ projected - rhs
    objective = 0.5 * float(np.sum((projected - y) * (projected - y) / winv))
    diagnostics.update(
        {
            "enabled": True,
            "method": "query_space_lsq",
            "solver": "lsmr_normal_equations",
            "known_total_count": int(total),
            "initial_max_constraint_violation": float(np.max(np.abs(initial_violation))),
            "initial_rms_constraint_violation": float(np.sqrt(np.mean(initial_violation * initial_violation))),
            "final_max_constraint_violation": float(np.max(np.abs(final_violation))),
            "final_rms_constraint_violation": float(np.sqrt(np.mean(final_violation * final_violation))),
            "l2_correction": float(np.linalg.norm(projected - y)),
            "weighted_objective": objective,
            "lsmr_iterations": int(solve[2]),
            "lsmr_stop_code": int(solve[1]),
            "lsmr_residual_norm": float(solve[3]),
            "min_projected_answer": float(np.min(projected)),
            "max_projected_answer": float(np.max(projected)),
        }
    )
    return QuerySpaceProjectionResult(projected=projected.astype(np.float32), diagnostics=diagnostics)


def _independent_constraint_rows(matrix: csr_matrix, tolerance: float | None = None) -> tuple[np.ndarray, int, float]:
    dense = matrix.toarray()
    if dense.shape[0] == 0:
        return np.empty(0, dtype=np.int32), 0, 0.0
    _, r, pivots = qr(dense.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if tolerance is None:
        max_diag = float(np.max(diagonal)) if len(diagonal) else 0.0
        tolerance = max(dense.shape) * np.finfo(np.float64).eps * max_diag
    rank = int(np.sum(diagonal > float(tolerance)))
    keep = np.sort(pivots[:rank]).astype(np.int32, copy=False)
    return keep, rank, float(tolerance)


def project_query_space_feasible_lsq(
    noisy: np.ndarray,
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    total: int,
    variances: np.ndarray | None = None,
    max_constraints: int = 200_000,
    solver_ftol: float = 1.0e-9,
    solver_max_iterations: int = 1_000,
    rank_tolerance: float | None = None,
    max_dense_constraint_cells: int = 20_000_000,
) -> QuerySpaceProjectionResult:
    """Project noisy query answers onto linear consistency and box constraints.

    This biased feasible projection solves

        min_z 0.5 * (z - y)^T W (z - y)
        s.t.  A z = b and 0 <= z <= total.

    Unlike :func:`project_query_space_lsq`, this estimator is nonlinear because
    of the bounds. It is intended as an optimization-feasible post-processing
    baseline, not as an unbiased estimator.
    """
    y = np.asarray(noisy, dtype=np.float64)
    if y.shape != (qcat.m,):
        raise ValueError(f"noisy must have shape ({qcat.m},), got {y.shape}")
    cards = np.asarray(cardinalities, dtype=np.int32)
    if np.any(cards <= 0):
        raise ValueError("cardinalities must be positive")
    if variances is None:
        var = np.ones(qcat.m, dtype=np.float64)
    else:
        var = np.asarray(variances, dtype=np.float64)
        if var.shape != (qcat.m,):
            raise ValueError(f"variances must have shape ({qcat.m},), got {var.shape}")
        var = np.maximum(var, 1.0e-12)
    inv = 1.0 / var

    constraints, rhs, diagnostics = _build_query_space_constraints(
        qcat,
        cards,
        int(total),
        max_constraints=int(max_constraints),
    )
    dense_cells = int(constraints.shape[0] * constraints.shape[1])
    if dense_cells > int(max_dense_constraint_cells):
        raise ValueError(
            "query_space_feasible_lsq currently uses a dense rank-revealing QR step; "
            f"constraint matrix has {dense_cells} dense cells, exceeding "
            f"max_dense_constraint_cells={max_dense_constraint_cells}"
        )

    if constraints.shape[0] == 0:
        projected = np.clip(y, 0.0, float(total))
        diagnostics.update(
            {
                "enabled": True,
                "method": "query_space_feasible_lsq",
                "solver": "box_clip",
                "known_total_count": int(total),
                "num_independent_constraints": 0,
                "num_redundant_constraints": 0,
                "initial_max_constraint_violation": 0.0,
                "final_max_constraint_violation": 0.0,
                "l2_correction": float(np.linalg.norm(projected - y)),
                "weighted_objective": 0.5 * float(np.sum((projected - y) * (projected - y) * inv)),
                "min_projected_answer": float(np.min(projected)),
                "max_projected_answer": float(np.max(projected)),
                "num_at_lower_bound": int(np.sum(projected <= 1.0e-8)),
                "num_at_upper_bound": int(np.sum(projected >= float(total) - 1.0e-8)),
            }
        )
        return QuerySpaceProjectionResult(projected=projected.astype(np.float32), diagnostics=diagnostics)

    independent_rows, rank, used_rank_tolerance = _independent_constraint_rows(
        constraints,
        tolerance=rank_tolerance,
    )
    dense_constraints = constraints.toarray()
    independent_constraints = dense_constraints[independent_rows]
    independent_rhs = rhs[independent_rows]

    equality_start = project_query_space_lsq(
        y,
        qcat,
        cards,
        int(total),
        variances=var,
        max_constraints=max_constraints,
    ).projected.astype(np.float64)
    x0 = np.clip(equality_start, 0.0, float(total))

    def objective(x: np.ndarray) -> float:
        diff = x - y
        return 0.5 * float(np.sum(diff * diff * inv))

    def gradient(x: np.ndarray) -> np.ndarray:
        return (x - y) * inv

    initial_violation = dense_constraints @ y - rhs
    result = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=Bounds(np.zeros(qcat.m, dtype=np.float64), np.full(qcat.m, float(total), dtype=np.float64)),
        constraints=[LinearConstraint(independent_constraints, independent_rhs, independent_rhs)],
        method="SLSQP",
        options={"ftol": float(solver_ftol), "maxiter": int(solver_max_iterations), "disp": False},
    )
    projected = np.asarray(result.x, dtype=np.float64)
    final_violation = dense_constraints @ projected - rhs
    diagnostics.update(
        {
            "enabled": True,
            "method": "query_space_feasible_lsq",
            "solver": "slsqp_independent_equalities",
            "known_total_count": int(total),
            "num_independent_constraints": int(rank),
            "num_redundant_constraints": int(constraints.shape[0] - rank),
            "rank_tolerance": float(used_rank_tolerance),
            "max_dense_constraint_cells": int(max_dense_constraint_cells),
            "initial_max_constraint_violation": float(np.max(np.abs(initial_violation))),
            "initial_rms_constraint_violation": float(np.sqrt(np.mean(initial_violation * initial_violation))),
            "final_max_constraint_violation": float(np.max(np.abs(final_violation))),
            "final_rms_constraint_violation": float(np.sqrt(np.mean(final_violation * final_violation))),
            "l2_correction": float(np.linalg.norm(projected - y)),
            "weighted_objective": objective(projected),
            "solver_success": bool(result.success),
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "solver_iterations": int(result.nit),
            "min_projected_answer": float(np.min(projected)),
            "max_projected_answer": float(np.max(projected)),
            "num_at_lower_bound": int(np.sum(projected <= 1.0e-8)),
            "num_at_upper_bound": int(np.sum(projected >= float(total) - 1.0e-8)),
        }
    )
    if not bool(result.success):
        raise RuntimeError(f"query_space_feasible_lsq failed: {result.message}")
    return QuerySpaceProjectionResult(projected=projected.astype(np.float32), diagnostics=diagnostics)


def _allowed_values_for_query(
    qcat: QueryCatalogue,
    qid: int,
    scope: tuple[int, ...],
    cardinalities: np.ndarray,
) -> list[np.ndarray]:
    allowed: dict[int, np.ndarray] = {
        int(attr): np.arange(int(cardinalities[int(attr)]), dtype=np.int32) for attr in scope
    }
    for attr, op, value, lo, hi in qcat.query_terms(qid):
        card = int(cardinalities[attr])
        if card <= 0:
            raise ValueError(f"Attribute {attr} has non-positive cardinality {card}")
        if op == OP_EQ:
            term_allowed = np.asarray([value], dtype=np.int32) if 0 <= value < card else np.empty(0, dtype=np.int32)
        elif op == OP_LE:
            upper = min(card - 1, value)
            term_allowed = np.arange(0, upper + 1, dtype=np.int32) if upper >= 0 else np.empty(0, dtype=np.int32)
        elif op == OP_GE:
            lower = max(0, value)
            term_allowed = np.arange(lower, card, dtype=np.int32) if lower < card else np.empty(0, dtype=np.int32)
        elif op == OP_RANGE:
            lower = max(0, lo)
            upper = min(card - 1, hi)
            term_allowed = (
                np.arange(lower, upper + 1, dtype=np.int32) if lower <= upper else np.empty(0, dtype=np.int32)
            )
        else:
            raise NotImplementedError(f"Unsupported query op {op}; consistency projection supports EQ/LE/GE/RANGE")
        allowed[attr] = np.intersect1d(allowed[attr], term_allowed, assume_unique=True)
    return [allowed[int(attr)] for attr in scope]


def _mask_indices(
    qcat: QueryCatalogue,
    qid: int,
    scope: tuple[int, ...],
    shape: tuple[int, ...],
    cardinalities: np.ndarray,
) -> np.ndarray:
    allowed = _allowed_values_for_query(qcat, qid, scope, cardinalities)
    if any(len(vals) == 0 for vals in allowed):
        return np.empty(0, dtype=np.int32)
    mesh = np.meshgrid(*allowed, indexing="ij")
    coords = [axis.ravel() for axis in mesh]
    if int(qcat.linear_num_terms[qid]) > 0:
        attr_to_axis = {attr: axis for axis, attr in enumerate(scope)}
        score = np.zeros_like(coords[0], dtype=np.float64)
        for attr, weight in qcat.linear_terms(qid):
            score += float(weight) * coords[attr_to_axis[int(attr)]].astype(np.float64)
        keep = score <= float(qcat.linear_thresholds[qid])
        if not np.any(keep):
            return np.empty(0, dtype=np.int32)
        coords = [axis[keep] for axis in coords]
    return np.ravel_multi_index(coords, shape).astype(np.int32, copy=False)


def _marginal_groups_for_scope(
    scope: tuple[int, ...],
    shape: tuple[int, ...],
    attrs: tuple[int, ...],
) -> list[np.ndarray]:
    positions = [scope.index(attr) for attr in attrs]
    marginal_shape = tuple(shape[pos] for pos in positions)
    groups: list[list[int]] = [[] for _ in range(int(np.prod(marginal_shape, dtype=np.int64)))]
    for flat in range(int(np.prod(shape, dtype=np.int64))):
        coords = np.unravel_index(flat, shape)
        marginal_coords = tuple(int(coords[pos]) for pos in positions)
        marginal_flat = int(np.ravel_multi_index(marginal_coords, marginal_shape))
        groups[marginal_flat].append(flat)
    return [np.asarray(group, dtype=np.int32) for group in groups]


def _build_latent_scope_models(
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    total: float,
    max_scope_cells: int,
) -> list[_ScopeModel]:
    by_scope: dict[tuple[int, ...], list[int]] = {}
    for qid in range(qcat.m):
        scope = _query_scope(qcat, qid)
        if not scope:
            raise NotImplementedError("Empty-scope total-count queries are not represented in QueryCatalogue")
        by_scope.setdefault(scope, []).append(qid)

    models: list[_ScopeModel] = []
    for scope, qids in sorted(by_scope.items()):
        shape = tuple(int(cardinalities[attr]) for attr in scope)
        num_cells = int(np.prod(shape, dtype=np.int64))
        if num_cells > int(max_scope_cells):
            raise ValueError(
                f"Local-table feasible projection scope {scope} has {num_cells} cells, "
                f"exceeding max_scope_cells={max_scope_cells}"
            )
        masks = [_mask_indices(qcat, qid, scope, shape, cardinalities) for qid in qids]
        table = np.full(num_cells, float(total) / max(1, num_cells), dtype=np.float64)
        models.append(
            _ScopeModel(
                attrs=scope,
                shape=shape,
                qids=qids,
                masks=masks,
                table=table,
                precision=1.0,
            )
        )
    return models


def _scope_offsets(models: list[_ScopeModel]) -> np.ndarray:
    offsets = np.zeros(len(models) + 1, dtype=np.int32)
    for idx, model in enumerate(models):
        offsets[idx + 1] = offsets[idx] + model.num_cells
    return offsets


def _build_local_table_measurement_matrix(
    qcat: QueryCatalogue,
    models: list[_ScopeModel],
    offsets: np.ndarray,
) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for model_idx, model in enumerate(models):
        offset = int(offsets[model_idx])
        for qid, mask in zip(model.qids, model.masks, strict=True):
            rows.extend([int(qid)] * len(mask))
            cols.extend((offset + mask.astype(np.int32)).astype(int).tolist())
            data.extend([1.0] * len(mask))
    return csr_matrix((data, (rows, cols)), shape=(qcat.m, int(offsets[-1])), dtype=np.float64)


def _build_local_table_equality_constraints(
    models: list[_ScopeModel],
    offsets: np.ndarray,
    total: float,
) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []

    def add_row(coeffs: dict[int, float], value: float) -> None:
        row = len(rhs)
        cleaned = {col: coef for col, coef in coeffs.items() if abs(coef) > 1.0e-12}
        if not cleaned:
            return
        for col, coef in sorted(cleaned.items()):
            rows.append(row)
            cols.append(int(col))
            data.append(float(coef))
        rhs.append(float(value))

    for model_idx, model in enumerate(models):
        offset = int(offsets[model_idx])
        add_row({offset + local_idx: 1.0 for local_idx in range(model.num_cells)}, float(total))

    intersections = _intersection_constraints(models)
    num_shared_marginal_constraints = 0
    num_shared_marginal_cells = 0
    for attrs in intersections:
        participants = [idx for idx, model in enumerate(models) if set(attrs).issubset(model.attrs)]
        if len(participants) < 2:
            continue
        reference_idx = participants[0]
        reference = models[reference_idx]
        reference_groups = _marginal_groups_for_scope(reference.attrs, reference.shape, attrs)
        for participant_idx in participants[1:]:
            participant = models[participant_idx]
            participant_groups = _marginal_groups_for_scope(participant.attrs, participant.shape, attrs)
            if len(participant_groups) != len(reference_groups):
                raise RuntimeError("Internal error: mismatched marginal group sizes")
            for ref_group, participant_group in zip(reference_groups, participant_groups, strict=True):
                coeffs: dict[int, float] = {}
                ref_offset = int(offsets[reference_idx])
                part_offset = int(offsets[participant_idx])
                for local_idx in participant_group:
                    coeffs[part_offset + int(local_idx)] = coeffs.get(part_offset + int(local_idx), 0.0) + 1.0
                for local_idx in ref_group:
                    coeffs[ref_offset + int(local_idx)] = coeffs.get(ref_offset + int(local_idx), 0.0) - 1.0
                add_row(coeffs, 0.0)
                num_shared_marginal_constraints += 1
            num_shared_marginal_cells += len(reference_groups)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(rhs), int(offsets[-1])), dtype=np.float64)
    diagnostics = {
        "num_total_constraints": int(len(models)),
        "num_intersection_constraints": int(len(intersections)),
        "intersection_constraint_scopes": [list(attrs) for attrs in intersections],
        "num_shared_marginal_constraints": int(num_shared_marginal_constraints),
        "num_shared_marginal_cells": int(num_shared_marginal_cells),
        "num_constraints": int(matrix.shape[0]),
        "num_constraint_nonzeros": int(matrix.nnz),
    }
    return matrix, np.asarray(rhs, dtype=np.float64), diagnostics


def project_local_table_feasible_lsq(
    noisy: np.ndarray,
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    total: int,
    variances: np.ndarray | None = None,
    max_scope_cells: int = 200_000,
    solver_ftol: float = 1.0e-9,
    solver_max_iterations: int = 1_000,
    rank_tolerance: float | None = None,
    max_dense_constraint_cells: int = 20_000_000,
) -> ConsistencyProjectionResult:
    """Fit nonnegative local scope tables with a weighted measurement objective.

    This biased feasible projection solves a local-table relaxation:

        min_T 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q
        s.t.  T_s >= 0, sum(T_s) = total,
              shared marginals agree across overlapping scopes.

    It keeps the original noisy measurements in the objective, unlike the old
    IPF-style reconciliation that averages derived marginals after local fits.
    """
    y = np.asarray(noisy, dtype=np.float64)
    if y.shape != (qcat.m,):
        raise ValueError(f"noisy must have shape ({qcat.m},), got {y.shape}")
    cards = np.asarray(cardinalities, dtype=np.int32)
    if np.any(cards <= 0):
        raise ValueError("cardinalities must be positive")
    if variances is None:
        var = np.ones(qcat.m, dtype=np.float64)
    else:
        var = np.asarray(variances, dtype=np.float64)
        if var.shape != (qcat.m,):
            raise ValueError(f"variances must have shape ({qcat.m},), got {var.shape}")
        var = np.maximum(var, 1.0e-12)
    inv = 1.0 / var

    models = _build_latent_scope_models(
        qcat,
        cards,
        float(total),
        max_scope_cells=int(max_scope_cells),
    )
    offsets = _scope_offsets(models)
    measurement_matrix = _build_local_table_measurement_matrix(qcat, models, offsets)
    constraints, rhs, constraint_diagnostics = _build_local_table_equality_constraints(models, offsets, float(total))
    dense_cells = int(constraints.shape[0] * constraints.shape[1])
    if dense_cells > int(max_dense_constraint_cells):
        raise ValueError(
            "local_table_feasible_lsq currently uses a dense rank-revealing QR step; "
            f"constraint matrix has {dense_cells} dense cells, exceeding "
            f"max_dense_constraint_cells={max_dense_constraint_cells}"
        )

    independent_rows, rank, used_rank_tolerance = _independent_constraint_rows(
        constraints,
        tolerance=rank_tolerance,
    )
    dense_constraints = constraints.toarray()
    independent_constraints = dense_constraints[independent_rows]
    independent_rhs = rhs[independent_rows]
    x0 = np.concatenate([model.table for model in models]).astype(np.float64)

    def objective(x: np.ndarray) -> float:
        residual = measurement_matrix @ x - y
        return 0.5 * float(np.sum(residual * residual * inv))

    def gradient(x: np.ndarray) -> np.ndarray:
        residual = np.asarray(measurement_matrix @ x - y, dtype=np.float64)
        return np.asarray(measurement_matrix.T @ (residual * inv), dtype=np.float64)

    initial_violation = dense_constraints @ x0 - rhs
    result = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=Bounds(np.zeros(int(offsets[-1]), dtype=np.float64), np.full(int(offsets[-1]), float(total))),
        constraints=[LinearConstraint(independent_constraints, independent_rhs, independent_rhs)],
        method="SLSQP",
        options={"ftol": float(solver_ftol), "maxiter": int(solver_max_iterations), "disp": False},
    )
    if not bool(result.success):
        raise RuntimeError(f"local_table_feasible_lsq failed: {result.message}")

    fitted = np.asarray(result.x, dtype=np.float64)
    projected = np.asarray(measurement_matrix @ fitted, dtype=np.float64)
    final_violation = dense_constraints @ fitted - rhs
    diagnostics: dict[str, Any] = {
        **constraint_diagnostics,
        "enabled": True,
        "method": "local_table_feasible_lsq",
        "solver": "slsqp_local_tables",
        "known_total_count": int(total),
        "num_scopes": int(len(models)),
        "scope_attrs": [list(model.attrs) for model in models],
        "num_scope_cells": int(offsets[-1]),
        "max_scope_cells_observed": int(max((model.num_cells for model in models), default=0)),
        "num_measurement_nonzeros": int(measurement_matrix.nnz),
        "num_independent_constraints": int(rank),
        "num_redundant_constraints": int(constraints.shape[0] - rank),
        "rank_tolerance": float(used_rank_tolerance),
        "max_dense_constraint_cells": int(max_dense_constraint_cells),
        "initial_max_constraint_violation": float(np.max(np.abs(initial_violation))),
        "initial_rms_constraint_violation": float(np.sqrt(np.mean(initial_violation * initial_violation))),
        "final_max_constraint_violation": float(np.max(np.abs(final_violation))),
        "final_rms_constraint_violation": float(np.sqrt(np.mean(final_violation * final_violation))),
        "weighted_objective": objective(fitted),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "min_table_cell": float(np.min(fitted)),
        "max_table_cell": float(np.max(fitted)),
        "num_table_cells_at_lower_bound": int(np.sum(fitted <= 1.0e-8)),
        "num_table_cells_at_upper_bound": int(np.sum(fitted >= float(total) - 1.0e-8)),
        "min_projected_answer": float(np.min(projected)),
        "max_projected_answer": float(np.max(projected)),
    }
    return ConsistencyProjectionResult(projected=projected.astype(np.float32), diagnostics=diagnostics)


def project_local_table_feasible_jax(
    noisy: np.ndarray,
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    total: int,
    variances: np.ndarray | None = None,
    max_scope_cells: int = 200_000,
    jax_iterations: int = 1_000,
    jax_active_set_tolerance: float = 1.0e-8,
    jax_kkt_ridge: float = 1.0e-10,
    max_dense_constraint_cells: int = 20_000_000,
) -> ConsistencyProjectionResult:
    """Fit local scope tables with a JAX dense active-set QP solver.

    This solves the same convex quadratic program as the CPU SLSQP reference:

        min_T 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q
        s.t.  T_s >= 0, sum(T_s) = total,
              shared marginals agree across overlapping scopes.

    The implementation uses JAX for dense KKT least-squares solves and a small
    Python active-set loop around those solves. It is intended for small and
    medium local-table projections where dense matrices are acceptable.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    y = np.asarray(noisy, dtype=np.float64)
    if y.shape != (qcat.m,):
        raise ValueError(f"noisy must have shape ({qcat.m},), got {y.shape}")
    cards = np.asarray(cardinalities, dtype=np.int32)
    if np.any(cards <= 0):
        raise ValueError("cardinalities must be positive")
    if variances is None:
        var = np.ones(qcat.m, dtype=np.float64)
    else:
        var = np.asarray(variances, dtype=np.float64)
        if var.shape != (qcat.m,):
            raise ValueError(f"variances must have shape ({qcat.m},), got {var.shape}")
        var = np.maximum(var, 1.0e-12)
    inv = 1.0 / var

    models = _build_latent_scope_models(
        qcat,
        cards,
        float(total),
        max_scope_cells=int(max_scope_cells),
    )
    offsets = _scope_offsets(models)
    measurement_matrix = _build_local_table_measurement_matrix(qcat, models, offsets)
    constraints, rhs, constraint_diagnostics = _build_local_table_equality_constraints(models, offsets, float(total))
    dense_cells = int(constraints.shape[0] * constraints.shape[1])
    if dense_cells > int(max_dense_constraint_cells):
        raise ValueError(
            "local_table_feasible_jax uses dense JAX matrices; "
            f"constraint matrix has {dense_cells} dense cells, exceeding "
            f"max_dense_constraint_cells={max_dense_constraint_cells}"
        )

    if int(jax_iterations) <= 0:
        raise ValueError("jax_iterations must be positive")
    active_tolerance = float(jax_active_set_tolerance)
    if active_tolerance < 0.0:
        raise ValueError("jax_active_set_tolerance must be non-negative")
    kkt_ridge = float(jax_kkt_ridge)
    if kkt_ridge < 0.0:
        raise ValueError("jax_kkt_ridge must be non-negative")

    dense_h = measurement_matrix.toarray().astype(np.float64, copy=False)
    dense_constraints = constraints.toarray().astype(np.float64, copy=False)
    independent_rows, rank, used_rank_tolerance = _independent_constraint_rows(constraints)
    independent_constraints = dense_constraints[independent_rows]
    independent_rhs = rhs[independent_rows].astype(np.float64, copy=False)
    y_np = y.astype(np.float64, copy=False)
    inv_np = inv.astype(np.float64, copy=False)
    x0_np = np.concatenate([model.table for model in models]).astype(np.float64)

    weighted_h = dense_h * inv_np.reshape(-1, 1)
    qmat_np = dense_h.T @ weighted_h
    linear_np = -dense_h.T @ (inv_np * y_np)

    qmat_j = jnp.asarray(qmat_np)
    linear_j = jnp.asarray(linear_np)
    eq_j = jnp.asarray(independent_constraints)
    rhs_j = jnp.asarray(independent_rhs)

    def solve_with_active(active_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        free = np.flatnonzero(~active_mask).astype(np.int32, copy=False)
        if len(free) == 0:
            return np.zeros(qmat_np.shape[0], dtype=np.float64), np.zeros(independent_constraints.shape[0]), free
        qff = qmat_j[np.ix_(free, free)]
        if kkt_ridge > 0.0:
            qff = qff + jnp.eye(len(free), dtype=qff.dtype) * kkt_ridge
        efree = eq_j[:, free]
        top = jnp.concatenate([qff, efree.T], axis=1)
        bottom = jnp.concatenate(
            [efree, jnp.zeros((efree.shape[0], efree.shape[0]), dtype=efree.dtype)],
            axis=1,
        )
        kkt = jnp.concatenate([top, bottom], axis=0)
        rhs_kkt = jnp.concatenate([-linear_j[free], rhs_j], axis=0)
        solution = jnp.linalg.lstsq(kkt, rhs_kkt, rcond=1.0e-12)[0]
        free_solution = np.asarray(jax.device_get(solution[: len(free)]), dtype=np.float64)
        multipliers = np.asarray(jax.device_get(solution[len(free) :]), dtype=np.float64)
        x_solution = np.zeros(qmat_np.shape[0], dtype=np.float64)
        x_solution[free] = free_solution
        return x_solution, multipliers, free

    initial_violation = dense_constraints @ x0_np - rhs
    x = x0_np.copy()
    active = x <= active_tolerance * 0.1
    active[:] = False
    multipliers = np.zeros(independent_constraints.shape[0], dtype=np.float64)
    solver_message = "maximum iterations reached"
    solver_success = False
    completed_iterations = 0
    for iteration in range(1, int(jax_iterations) + 1):
        completed_iterations = iteration
        candidate, multipliers, free = solve_with_active(active)
        if len(free) == 0:
            solver_message = "no free variables"
            break
        free_values = candidate[free]
        min_free = float(np.min(free_values)) if len(free_values) else 0.0
        if min_free < -active_tolerance:
            direction = candidate - x
            decreasing = direction < -active_tolerance
            if not np.any(decreasing):
                blocking = int(free[int(np.argmin(free_values))])
                x[blocking] = 0.0
                active[blocking] = True
                continue
            alpha_values = np.divide(
                x[decreasing],
                x[decreasing] - candidate[decreasing],
                out=np.ones(int(np.sum(decreasing)), dtype=np.float64),
                where=(x[decreasing] - candidate[decreasing]) > 0.0,
            )
            alpha = float(np.clip(np.min(alpha_values), 0.0, 1.0))
            x = x + alpha * direction
            x[np.abs(x) <= active_tolerance] = 0.0
            blocking_candidates = np.flatnonzero((np.abs(x) <= active_tolerance) & (~active))
            if len(blocking_candidates) == 0:
                blocking = int(free[int(np.argmin(free_values))])
            else:
                blocking = int(blocking_candidates[0])
            active[blocking] = True
            continue

        x = candidate
        x[(x < 0.0) & (x >= -active_tolerance)] = 0.0
        gradient = qmat_np @ x + linear_np
        reduced_gradient = gradient + independent_constraints.T @ multipliers
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            solver_success = True
            solver_message = "optimal with no active lower bounds"
            break
        active_multipliers = reduced_gradient[active_indices]
        min_multiplier = float(np.min(active_multipliers))
        if min_multiplier >= -active_tolerance:
            solver_success = True
            solver_message = "optimal active set found"
            break
        release = int(active_indices[int(np.argmin(active_multipliers))])
        active[release] = False

    fitted = np.maximum(x, 0.0)
    projected = np.asarray(measurement_matrix @ fitted, dtype=np.float64)
    final_violation = dense_constraints @ fitted - rhs
    residual_final = projected - y
    weighted_objective = 0.5 * float(np.sum(residual_final * residual_final * inv))
    gradient_final = qmat_np @ fitted + linear_np
    kkt_residual = gradient_final + independent_constraints.T @ multipliers
    free_mask = fitted > max(active_tolerance, 1.0e-10)
    free_kkt_residual = kkt_residual[free_mask]
    active_kkt_residual = kkt_residual[~free_mask]
    diagnostics: dict[str, Any] = {
        **constraint_diagnostics,
        "enabled": True,
        "method": "local_table_feasible_jax",
        "solver": "jax_dense_active_set_qp",
        "known_total_count": int(total),
        "num_scopes": int(len(models)),
        "scope_attrs": [list(model.attrs) for model in models],
        "num_scope_cells": int(offsets[-1]),
        "max_scope_cells_observed": int(max((model.num_cells for model in models), default=0)),
        "num_measurement_nonzeros": int(measurement_matrix.nnz),
        "num_independent_constraints": int(rank),
        "num_redundant_constraints": int(constraints.shape[0] - rank),
        "rank_tolerance": float(used_rank_tolerance),
        "max_dense_constraint_cells": int(max_dense_constraint_cells),
        "initial_max_constraint_violation": float(np.max(np.abs(initial_violation))),
        "initial_rms_constraint_violation": float(np.sqrt(np.mean(initial_violation * initial_violation))),
        "final_max_constraint_violation": float(np.max(np.abs(final_violation))),
        "final_rms_constraint_violation": float(np.sqrt(np.mean(final_violation * final_violation))),
        "weighted_objective": weighted_objective,
        "solver_success": bool(solver_success),
        "solver_message": solver_message,
        "solver_iterations": int(completed_iterations),
        "jax_max_active_set_iterations": int(jax_iterations),
        "jax_active_set_tolerance": float(active_tolerance),
        "jax_kkt_ridge": float(kkt_ridge),
        "num_active_lower_bounds": int(np.sum(~free_mask)),
        "free_kkt_residual_inf": float(np.max(np.abs(free_kkt_residual))) if len(free_kkt_residual) else 0.0,
        "active_lower_multiplier_min": float(np.min(active_kkt_residual)) if len(active_kkt_residual) else 0.0,
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(device) for device in jax.devices()],
        "min_table_cell": float(np.min(fitted)),
        "max_table_cell": float(np.max(fitted)),
        "num_table_cells_at_lower_bound": int(np.sum(fitted <= 1.0e-8)),
        "num_table_cells_at_upper_bound": int(np.sum(fitted >= float(total) - 1.0e-8)),
        "min_projected_answer": float(np.min(projected)),
        "max_projected_answer": float(np.max(projected)),
    }
    return ConsistencyProjectionResult(projected=projected.astype(np.float32), diagnostics=diagnostics)


def _fit_scope_table(
    scope: tuple[int, ...],
    shape: tuple[int, ...],
    qids: list[int],
    masks: list[np.ndarray],
    y: np.ndarray,
    weights: np.ndarray,
    total: float,
    max_lsq_iterations: int,
) -> np.ndarray:
    num_cells = int(np.prod(shape, dtype=np.int64))
    if num_cells <= 0:
        raise ValueError(f"Scope {scope} has no cells")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    target: list[float] = []
    row_id = 0
    scope_precision = 0.0
    for qid, mask in zip(qids, masks, strict=True):
        if len(mask) == 0:
            continue
        sqrt_w = float(np.sqrt(max(float(weights[qid]), 1.0e-12)))
        rows.extend([row_id] * len(mask))
        cols.extend(mask.astype(int).tolist())
        data.extend([sqrt_w] * len(mask))
        target.append(float(y[qid]) * sqrt_w)
        scope_precision += float(weights[qid])
        row_id += 1

    total_weight = max(scope_precision, 1.0)
    sqrt_total_weight = float(np.sqrt(total_weight))
    rows.extend([row_id] * num_cells)
    cols.extend(range(num_cells))
    data.extend([sqrt_total_weight] * num_cells)
    target.append(float(total) * sqrt_total_weight)
    row_id += 1

    if row_id == 1:
        return np.full(num_cells, float(total) / num_cells, dtype=np.float64)

    matrix = csr_matrix((data, (rows, cols)), shape=(row_id, num_cells), dtype=np.float64)
    result = lsq_linear(
        matrix,
        np.asarray(target, dtype=np.float64),
        bounds=(0.0, float(total)),
        max_iter=max_lsq_iterations,
        lsmr_tol="auto",
    )
    table = np.asarray(result.x, dtype=np.float64)
    table = project_simplex(table, float(total)).astype(np.float64)
    floor = max(float(total), 1.0) * 1.0e-12 / max(1, num_cells)
    table = np.maximum(table, floor)
    table = project_simplex(table, float(total)).astype(np.float64)
    return table


def _build_scope_models(
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    total: float,
    max_scope_cells: int,
    max_lsq_iterations: int,
) -> list[_ScopeModel]:
    by_scope: dict[tuple[int, ...], list[int]] = {}
    for qid in range(qcat.m):
        scope = _query_scope(qcat, qid)
        if not scope:
            raise NotImplementedError("Empty-scope total-count queries are not represented in QueryCatalogue")
        by_scope.setdefault(scope, []).append(qid)

    models: list[_ScopeModel] = []
    for scope, qids in sorted(by_scope.items()):
        shape = tuple(int(cardinalities[attr]) for attr in scope)
        num_cells = int(np.prod(shape, dtype=np.int64))
        if num_cells > int(max_scope_cells):
            raise ValueError(
                f"Consistency projection scope {scope} has {num_cells} cells, "
                f"exceeding max_scope_cells={max_scope_cells}"
            )
        masks = [_mask_indices(qcat, qid, scope, shape, cardinalities) for qid in qids]
        table = _fit_scope_table(scope, shape, qids, masks, y, weights, total, max_lsq_iterations)
        precision = float(sum(float(weights[qid]) for qid in qids))
        models.append(
            _ScopeModel(
                attrs=scope,
                shape=shape,
                qids=qids,
                masks=masks,
                table=table,
                precision=max(precision, 1.0e-12),
            )
        )
    return models


def _marginal(model: _ScopeModel, attrs: tuple[int, ...]) -> np.ndarray:
    table = model.table.reshape(model.shape)
    axes_to_sum = tuple(idx for idx, attr in enumerate(model.attrs) if attr not in attrs)
    marginal = table.sum(axis=axes_to_sum) if axes_to_sum else table
    if attrs != tuple(attr for attr in model.attrs if attr in attrs):
        current_order = [attr for attr in model.attrs if attr in attrs]
        transpose_order = [current_order.index(attr) for attr in attrs]
        marginal = np.transpose(marginal, axes=transpose_order)
    return np.asarray(marginal, dtype=np.float64).reshape(-1)


def _scale_to_marginal(model: _ScopeModel, attrs: tuple[int, ...], target: np.ndarray) -> None:
    table = model.table.reshape(model.shape)
    current = _marginal(model, attrs).reshape(tuple(model.shape[model.attrs.index(attr)] for attr in attrs))
    target_arr = np.asarray(target, dtype=np.float64).reshape(current.shape)
    ratio = np.divide(target_arr, current, out=np.ones_like(target_arr), where=current > 1.0e-12)
    for axis, attr in enumerate(model.attrs):
        if attr not in attrs:
            ratio = np.expand_dims(ratio, axis=axis)
    table *= ratio
    model.table = project_simplex(table.reshape(-1), float(target_arr.sum())).astype(np.float64)


def _intersection_constraints(models: list[_ScopeModel]) -> list[tuple[int, ...]]:
    intersections: set[tuple[int, ...]] = set()
    for left, right in combinations(models, 2):
        overlap = tuple(attr for attr in left.attrs if attr in set(right.attrs))
        if overlap:
            intersections.add(overlap)
    return sorted(intersections, key=lambda x: (len(x), x), reverse=True)


def _enforce_scope_consistency(
    models: list[_ScopeModel],
    total: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[int, float]:
    constraints = _intersection_constraints(models)
    if not constraints:
        return 0, 0.0

    max_error = float("inf")
    completed = 0
    for iteration in range(1, max_iterations + 1):
        max_error = 0.0
        for attrs in constraints:
            participants = [model for model in models if set(attrs).issubset(model.attrs)]
            if len(participants) < 2:
                continue
            marginals = [_marginal(model, attrs) for model in participants]
            weights = np.asarray([model.precision for model in participants], dtype=np.float64)
            target = np.average(np.stack(marginals, axis=0), axis=0, weights=weights)
            target = project_simplex(target, float(total)).astype(np.float64)
            for marginal in marginals:
                max_error = max(max_error, float(np.max(np.abs(marginal - target))))
            for model in participants:
                _scale_to_marginal(model, attrs, target)
        completed = iteration
        if max_error <= tolerance:
            break
    return completed, max_error


def _answers_from_models(qcat: QueryCatalogue, models: list[_ScopeModel]) -> np.ndarray:
    projected = np.zeros(qcat.m, dtype=np.float64)
    for model in models:
        for qid, mask in zip(model.qids, model.masks, strict=True):
            projected[qid] = float(model.table[mask].sum()) if len(mask) else 0.0
    return projected.astype(np.float32)


def project_consistent_targets(
    noisy: np.ndarray,
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    total: int,
    variances: np.ndarray | None = None,
    max_scope_cells: int = 200_000,
    max_iterations: int = 100,
    tolerance: float = 1.0e-2,
    max_lsq_iterations: int = 100,
) -> ConsistencyProjectionResult:
    """Project noisy query answers onto locally consistent marginal tables.

    Each query is represented as a linear sum over the full cell table of its
    attribute scope. Tables are constrained to be non-negative and to sum to the
    known row count. Overlapping scopes are iteratively reconciled on their
    shared marginals, so adding supported EQ/LE/GE/RANGE conjunctions introduces
    consistent constraints instead of independent noisy targets.
    """
    y = np.asarray(noisy, dtype=np.float64)
    if y.shape != (qcat.m,):
        raise ValueError(f"noisy must have shape ({qcat.m},), got {y.shape}")
    cards = np.asarray(cardinalities, dtype=np.int32)
    if np.any(cards <= 0):
        raise ValueError("cardinalities must be positive")
    if variances is None:
        weights = np.ones(qcat.m, dtype=np.float64)
    else:
        var = np.asarray(variances, dtype=np.float64)
        if var.shape != (qcat.m,):
            raise ValueError(f"variances must have shape ({qcat.m},), got {var.shape}")
        weights = 1.0 / np.maximum(var, 1.0e-12)

    models = _build_scope_models(
        qcat,
        cards,
        y,
        weights,
        float(total),
        max_scope_cells=int(max_scope_cells),
        max_lsq_iterations=int(max_lsq_iterations),
    )
    iterations, max_marginal_error = _enforce_scope_consistency(
        models,
        float(total),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
    )
    projected = _answers_from_models(qcat, models)
    diagnostics = {
        "enabled": True,
        "method": "local_marginal_ipf",
        "num_scopes": int(len(models)),
        "num_scope_cells": int(sum(model.num_cells for model in models)),
        "max_scope_cells_observed": int(max((model.num_cells for model in models), default=0)),
        "num_intersection_constraints": int(len(_intersection_constraints(models))),
        "iterations": int(iterations),
        "max_marginal_error": float(max_marginal_error),
        "tolerance": float(tolerance),
        "converged": bool(max_marginal_error <= float(tolerance)),
        "known_total_count": int(total),
    }
    return ConsistencyProjectionResult(projected=projected, diagnostics=diagnostics)
