from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
import warnings
from typing import Any, Callable, Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

from qdte.measurement.factorization import HierarchicalPairStrategy
from qdte.queries.orthogonal import helmert_contrast, interaction_features
from qdte.rce.forest_prior import ReleasedConfidenceForestPrior
from qdte.rce.public_domain import PublicLegalRowDomain


ROW_PRICING_METHOD = "exact_categorical_unary_pairwise_map_v1"
SHADOW_FEATURE_METHOD = "canonical_hierarchical_unary_pairwise_features_v1"


@dataclass(frozen=True)
class ShadowFeatureMap:
    strategy: HierarchicalPairStrategy
    domain: PublicLegalRowDomain
    method: str = SHADOW_FEATURE_METHOD

    def __post_init__(self) -> None:
        if self.method != SHADOW_FEATURE_METHOD:
            raise ValueError(f"Unsupported shadow feature method {self.method!r}")
        if tuple(self.strategy.cardinalities) != self.domain.cardinalities:
            raise ValueError("Shadow strategy and public row domain must share a schema")
        for block in self.strategy.blocks:
            if block.kind not in {"oneway_contrast", "pair_interaction"}:
                raise ValueError(
                    f"RHCG-CCMP-v1 rejects factor kind {block.kind!r}; order must be <= 2"
                )
            if len(block.scope) not in {1, 2}:
                raise ValueError("RHCG-CCMP-v1 only supports unary/pairwise factors")

    @property
    def block_slices(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        offset = 0
        for block in self.strategy.blocks:
            result[block.name] = slice(offset, offset + block.dimension)
            offset += block.dimension
        return result

    @property
    def feature_dimension(self) -> int:
        return sum(block.dimension for block in self.strategy.blocks)

    @property
    def affine_rank(self) -> int:
        return sum(cardinality - 1 for cardinality in self.domain.cardinalities) + sum(
            (self.domain.cardinalities[left] - 1)
            * (self.domain.cardinalities[right] - 1)
            for left, right in self.strategy.pairs
        )

    @property
    def atom_cap(self) -> int:
        return min(self.affine_rank + 1, 4096)

    @property
    def feature_hash(self) -> str:
        payload = {
            "method": self.method,
            "domain_manifest_hash": self.domain.manifest_hash,
            "blocks": [block.to_dict() for block in self.strategy.blocks],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()

    def row_features(self, rows: np.ndarray) -> np.ndarray:
        values = self.domain.validate_rows(rows)
        features: list[np.ndarray] = []
        for block in self.strategy.blocks:
            local = interaction_features(
                values,
                block.scope,
                self.strategy.cardinalities,
            )
            features.append(local.reshape(len(values), block.dimension))
        result = np.concatenate(features, axis=1)
        if result.shape != (len(values), self.feature_dimension):
            raise RuntimeError("Canonical shadow feature map produced an invalid shape")
        return result

    def row_atom_answers(self, rows: np.ndarray) -> np.ndarray:
        return float(self.domain.public_n) * self.row_features(rows)

    def table_answer(self, rows: np.ndarray) -> np.ndarray:
        values = self.domain.validate_rows(rows)
        if len(values) != self.domain.public_n:
            raise ValueError("Aggregate warm-start tables must have the public row count")
        return np.sum(self.row_features(values), axis=0, dtype=np.float64)

    def unit_rho_covariance_diagonal(self) -> np.ndarray:
        diagonal = np.empty(self.feature_dimension, dtype=np.float64)
        for block in self.strategy.blocks:
            diagonal[self.block_slices[block.name]] = 0.5 * block.sensitivity_l2**2
        return diagonal

    def energy_tables(
        self,
        coefficients: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[tuple[int, int], np.ndarray]]:
        theta = np.asarray(coefficients, dtype=np.float64)
        if theta.shape != (self.feature_dimension,) or not np.all(np.isfinite(theta)):
            raise ValueError("Pricing coefficients must match the shadow feature dimension")
        unary = [np.zeros(cardinality, dtype=np.float64) for cardinality in self.domain.cardinalities]
        pairwise: dict[tuple[int, int], np.ndarray] = {}
        slices = self.block_slices
        for block in self.strategy.blocks:
            local = theta[slices[block.name]].reshape(block.coefficient_shape)
            if block.kind == "oneway_contrast":
                attribute = block.scope[0]
                unary[attribute] += helmert_contrast(
                    self.domain.cardinalities[attribute]
                ) @ local
            elif block.kind == "pair_interaction":
                left, right = block.scope
                pairwise[(left, right)] = (
                    helmert_contrast(self.domain.cardinalities[left])
                    @ local
                    @ helmert_contrast(self.domain.cardinalities[right]).T
                )
            else:  # pragma: no cover - constructor rejects this path.
                raise AssertionError(block.kind)
        return tuple(unary), pairwise

    def row_energy(self, row: np.ndarray, coefficients: np.ndarray) -> float:
        values = np.asarray(row, dtype=np.int32).reshape(1, -1)
        return float(self.row_features(values)[0] @ np.asarray(coefficients, dtype=np.float64))


def _forest_pair_marginal(
    prior: ReleasedConfidenceForestPrior,
    left: int,
    right: int,
) -> np.ndarray:
    if left == right:
        raise ValueError("A pair marginal requires distinct attributes")
    edge_by_pair = {edge.pair: edge for edge in prior.edges}
    adjacency: list[list[int]] = [[] for _ in range(prior.dimension)]
    for first, second in edge_by_pair:
        adjacency[first].append(second)
        adjacency[second].append(first)
    parent = {int(left): -1}
    queue = [int(left)]
    while queue and int(right) not in parent:
        current = queue.pop(0)
        for neighbor in sorted(adjacency[current]):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if int(right) not in parent:
        return np.outer(prior.probabilities[left], prior.probabilities[right])
    path = [int(right)]
    while path[-1] != int(left):
        path.append(parent[path[-1]])
    path.reverse()
    transition = np.eye(prior.cardinalities[left], dtype=np.float64)
    for source, destination in zip(path[:-1], path[1:], strict=True):
        pair = (min(source, destination), max(source, destination))
        table = edge_by_pair[pair].table
        oriented = table if source < destination else table.T
        conditional = np.divide(
            oriented,
            prior.probabilities[source].reshape(-1, 1),
            out=np.zeros_like(oriented),
            where=prior.probabilities[source].reshape(-1, 1) > 0.0,
        )
        transition = transition @ conditional
    marginal = prior.probabilities[left].reshape(-1, 1) * transition
    marginal = np.maximum(marginal, 0.0)
    marginal /= float(np.sum(marginal))
    if np.max(np.abs(np.sum(marginal, axis=1) - prior.probabilities[left])) > 1.0e-9:
        raise RuntimeError("CCF path marginal does not preserve its left anchor")
    if np.max(np.abs(np.sum(marginal, axis=0) - prior.probabilities[right])) > 1.0e-9:
        raise RuntimeError("CCF path marginal does not preserve its right anchor")
    return marginal


def ccf_moment_answer(
    feature_map: ShadowFeatureMap,
    prior: ReleasedConfidenceForestPrior,
) -> np.ndarray:
    if prior.cardinalities != feature_map.domain.cardinalities:
        raise ValueError("CCF prior and shadow feature map use different schemas")
    moments = np.empty(feature_map.feature_dimension, dtype=np.float64)
    slices = feature_map.block_slices
    for block in feature_map.strategy.blocks:
        if block.kind == "oneway_contrast":
            attribute = block.scope[0]
            local = (
                prior.probabilities[attribute]
                @ helmert_contrast(prior.cardinalities[attribute])
            )
        else:
            left, right = block.scope
            pair = _forest_pair_marginal(prior, left, right)
            local = (
                helmert_contrast(prior.cardinalities[left]).T
                @ pair
                @ helmert_contrast(prior.cardinalities[right])
            )
        moments[slices[block.name]] = np.asarray(local, dtype=np.float64).reshape(-1)
    return float(feature_map.domain.public_n) * moments


@dataclass(frozen=True)
class RowPricingResult:
    row: np.ndarray
    objective: float
    dual_bound: float
    absolute_gap: float
    relative_gap: float
    certified: bool
    solver: str
    node_count: int
    lexicographic_tie_break: bool
    diagnostics: dict[str, Any]
    method: str = ROW_PRICING_METHOD

    def __post_init__(self) -> None:
        row = np.asarray(self.row, dtype=np.int32).copy()
        if row.ndim != 1 or not np.all(np.isfinite(row)):
            raise ValueError("Priced row must be a finite one-dimensional vector")
        row.setflags(write=False)
        object.__setattr__(self, "row", row)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "row": self.row.tolist(),
            "objective": self.objective,
            "dual_bound": self.dual_bound,
            "absolute_gap": self.absolute_gap,
            "relative_gap": self.relative_gap,
            "certified": self.certified,
            "solver": self.solver,
            "node_count": self.node_count,
            "lexicographic_tie_break": self.lexicographic_tie_break,
            "diagnostics": dict(self.diagnostics),
        }


def _enumeration_price(
    feature_map: ShadowFeatureMap,
    coefficients: np.ndarray,
) -> RowPricingResult:
    rows = feature_map.domain.enumerate_rows()
    energies = feature_map.row_features(rows) @ np.asarray(coefficients, dtype=np.float64)
    minimum = float(np.min(energies))
    tolerance = 1.0e-12 * max(1.0, abs(minimum))
    candidates = np.flatnonzero(energies <= minimum + tolerance)
    row_index = int(candidates[0])
    row = rows[row_index]
    objective = float(energies[row_index])
    return RowPricingResult(
        row=row,
        objective=objective,
        dual_bound=objective,
        absolute_gap=0.0,
        relative_gap=0.0,
        certified=True,
        solver="exhaustive_public_row_enumeration",
        node_count=len(rows),
        lexicographic_tie_break=True,
        diagnostics={"domain_size": len(rows)},
    )


@dataclass(frozen=True)
class _MILPLayout:
    unary_slices: tuple[slice, ...]
    pair_slices: dict[tuple[int, int], slice]
    variable_count: int
    equality: LinearConstraint
    bounds: Bounds
    integrality: np.ndarray


def _milp_layout(feature_map: ShadowFeatureMap) -> _MILPLayout:
    cards = feature_map.domain.cardinalities
    unary_slices: list[slice] = []
    offset = 0
    for cardinality in cards:
        unary_slices.append(slice(offset, offset + cardinality))
        offset += cardinality
    pair_slices: dict[tuple[int, int], slice] = {}
    for left, right in feature_map.strategy.pairs:
        width = cards[left] * cards[right]
        pair_slices[(left, right)] = slice(offset, offset + width)
        offset += width
    num_equalities = len(cards) + sum(
        cards[left] + cards[right] for left, right in feature_map.strategy.pairs
    )
    matrix = lil_matrix((num_equalities, offset), dtype=np.float64)
    row = 0
    for attribute, local_slice in enumerate(unary_slices):
        matrix[row, local_slice] = 1.0
        row += 1
    for left, right in feature_map.strategy.pairs:
        pair_slice = pair_slices[(left, right)]
        pair_indices = np.arange(pair_slice.start, pair_slice.stop).reshape(
            cards[left], cards[right]
        )
        for category in range(cards[left]):
            matrix[row, pair_indices[category, :]] = 1.0
            matrix[row, unary_slices[left].start + category] = -1.0
            row += 1
        for category in range(cards[right]):
            matrix[row, pair_indices[:, category]] = 1.0
            matrix[row, unary_slices[right].start + category] = -1.0
            row += 1
    if row != num_equalities:
        raise AssertionError("MILP equality layout was not filled")
    integrality = np.zeros(offset, dtype=np.int32)
    for local_slice in unary_slices:
        integrality[local_slice] = 1
    equality_rhs = np.zeros(num_equalities, dtype=np.float64)
    equality_rhs[: len(cards)] = 1.0
    return _MILPLayout(
        unary_slices=tuple(unary_slices),
        pair_slices=pair_slices,
        variable_count=offset,
        equality=LinearConstraint(csr_matrix(matrix), equality_rhs, equality_rhs),
        bounds=Bounds(np.zeros(offset), np.ones(offset)),
        integrality=integrality,
    )


def _milp_objective(
    feature_map: ShadowFeatureMap,
    layout: _MILPLayout,
    coefficients: np.ndarray,
) -> np.ndarray:
    unary, pairwise = feature_map.energy_tables(coefficients)
    objective = np.zeros(layout.variable_count, dtype=np.float64)
    for local_slice, values in zip(layout.unary_slices, unary, strict=True):
        objective[local_slice] = values
    for pair, local_slice in layout.pair_slices.items():
        objective[local_slice] = pairwise[pair].reshape(-1)
    return objective


def _solve_milp(
    objective: np.ndarray,
    layout: _MILPLayout,
    extra_constraints: tuple[LinearConstraint, ...] = (),
) -> Any:
    options = {
        "presolve": True,
        "node_limit": 2_000_000,
        "mip_rel_gap": 1.0e-9,
        "mip_abs_gap": 1.0e-9,
        "primal_feasibility_tolerance": 1.0e-10,
        "dual_feasibility_tolerance": 1.0e-10,
        "mip_feasibility_tolerance": 1.0e-10,
        "threads": 1,
        "random_seed": 0,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        return milp(
            objective,
            integrality=layout.integrality,
            bounds=layout.bounds,
            constraints=(layout.equality, *extra_constraints),
            options=options,
        )


def _row_from_milp_solution(
    feature_map: ShadowFeatureMap,
    layout: _MILPLayout,
    solution: np.ndarray,
) -> np.ndarray:
    row = np.asarray(
        [int(np.argmax(solution[local_slice])) for local_slice in layout.unary_slices],
        dtype=np.int32,
    )
    feature_map.domain.validate_rows(row.reshape(1, -1))
    return row


def _mip_node_count(result: Any) -> int:
    value = getattr(result, "mip_node_count", None)
    return -1 if value is None else int(value)


def _certified_milp_price(
    feature_map: ShadowFeatureMap,
    coefficients: np.ndarray,
    *,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> RowPricingResult:
    def progress(stage: str, **values: Any) -> None:
        if progress_callback is not None:
            progress_callback(stage, values)

    layout = _milp_layout(feature_map)
    primary_objective = _milp_objective(feature_map, layout, coefficients)
    primary_started = time.perf_counter()
    progress("primary_started", variable_count=layout.variable_count)
    primary = _solve_milp(primary_objective, layout)
    if not bool(primary.success) or primary.x is None or not np.isfinite(primary.fun):
        raise RuntimeError(f"Categorical pricing MILP failed: {primary.message}")
    primary_row = _row_from_milp_solution(feature_map, layout, primary.x)
    incumbent = feature_map.row_energy(primary_row, coefficients)
    dual_bound = float(getattr(primary, "mip_dual_bound", float(primary.fun)))
    gap_limit = 1.0e-9 * max(1.0, abs(incumbent))
    if dual_bound > incumbent + gap_limit:
        raise RuntimeError(
            "Categorical pricing MILP returned a dual bound above its canonical "
            f"legal-row incumbent: incumbent={incumbent:.17g}, "
            f"dual_bound={dual_bound:.17g}, tolerance={gap_limit:.17g}"
        )
    absolute_gap = max(0.0, incumbent - dual_bound)
    if absolute_gap > gap_limit:
        raise RuntimeError(
            f"Categorical pricing MILP gap {absolute_gap} exceeds {gap_limit}"
        )
    primary_tolerance = max(1.0e-10, gap_limit)
    primary_constraint = LinearConstraint(
        csr_matrix(primary_objective.reshape(1, -1)),
        -np.inf,
        incumbent + primary_tolerance,
    )
    progress(
        "primary_complete",
        elapsed_seconds=time.perf_counter() - primary_started,
        incumbent=incumbent,
        dual_bound=dual_bound,
        absolute_gap=absolute_gap,
        node_count=_mip_node_count(primary),
    )
    primary_selector = np.zeros(layout.variable_count, dtype=np.float64)
    for attribute, category in enumerate(primary_row):
        primary_selector[layout.unary_slices[attribute].start + int(category)] = 1.0
    no_good_constraint = LinearConstraint(
        csr_matrix(primary_selector.reshape(1, -1)),
        -np.inf,
        float(len(layout.unary_slices) - 1),
    )
    uniqueness_started = time.perf_counter()
    progress("uniqueness_started")
    alternative = _solve_milp(
        np.zeros(layout.variable_count, dtype=np.float64),
        layout,
        (primary_constraint, no_good_constraint),
    )
    progress(
        "uniqueness_complete",
        elapsed_seconds=time.perf_counter() - uniqueness_started,
        status=int(alternative.status),
        success=bool(alternative.success),
        node_count=_mip_node_count(alternative),
    )
    if int(alternative.status) == 2:
        return RowPricingResult(
            row=primary_row,
            objective=float(incumbent),
            dual_bound=dual_bound,
            absolute_gap=absolute_gap,
            relative_gap=absolute_gap / max(1.0, abs(float(incumbent))),
            certified=True,
            solver="scipy_highs_milp_unary_pairwise",
            node_count=_mip_node_count(primary),
            lexicographic_tie_break=True,
            diagnostics={
                "primary_status": int(primary.status),
                "primary_message": str(primary.message),
                "primary_solver_objective": float(primary.fun),
                "primary_canonical_objective": float(incumbent),
                "primary_mip_gap": float(getattr(primary, "mip_gap", math.inf)),
                "primary_tolerance": primary_tolerance,
                "primary_face_unique": True,
                "uniqueness_status": int(alternative.status),
                "uniqueness_message": str(alternative.message),
                "node_cap": 2_000_000,
                "threads": 1,
                "solver_seed": 0,
                "lexicographic_records": [],
            },
        )
    if not bool(alternative.success) or alternative.x is None:
        raise RuntimeError(
            "Categorical pricing uniqueness certification failed: "
            f"{alternative.message}"
        )
    fixed_constraints: list[LinearConstraint] = []
    chosen: list[int] = []
    lex_records: list[dict[str, Any]] = []
    for attribute, local_slice in enumerate(layout.unary_slices):
        lex_objective = np.zeros(layout.variable_count, dtype=np.float64)
        lex_objective[local_slice] = np.arange(
            feature_map.domain.cardinalities[attribute], dtype=np.float64
        )
        lex_started = time.perf_counter()
        progress("lex_attribute_started", attribute=attribute)
        result = _solve_milp(
            lex_objective,
            layout,
            (primary_constraint, *fixed_constraints),
        )
        if not bool(result.success) or result.x is None:
            raise RuntimeError(
                f"Lexicographic pricing tie-break failed at attribute {attribute}: "
                f"{result.message}"
            )
        category = int(np.argmax(result.x[local_slice]))
        progress(
            "lex_attribute_complete",
            attribute=attribute,
            category=category,
            elapsed_seconds=time.perf_counter() - lex_started,
            node_count=_mip_node_count(result),
        )
        chosen.append(category)
        selector = np.zeros(layout.variable_count, dtype=np.float64)
        selector[local_slice.start + category] = 1.0
        fixed_constraints.append(
            LinearConstraint(csr_matrix(selector.reshape(1, -1)), 1.0, 1.0)
        )
        lex_records.append(
            {
                "attribute": attribute,
                "category": category,
                "node_count": _mip_node_count(result),
                "mip_gap": float(getattr(result, "mip_gap", math.inf)),
            }
        )
    row = np.asarray(chosen, dtype=np.int32)
    feature_map.domain.validate_rows(row.reshape(1, -1))
    row_objective = feature_map.row_energy(row, coefficients)
    if row_objective > incumbent + primary_tolerance + 1.0e-10:
        raise RuntimeError(
            "Lexicographic priced row violates the primary optimum face: "
            f"row_objective={row_objective:.17g}, incumbent={incumbent:.17g}, "
            f"dual_bound={dual_bound:.17g}, tolerance={primary_tolerance:.17g}, "
            f"violation={row_objective - incumbent:.17g}"
        )
    return RowPricingResult(
        row=row,
        objective=float(row_objective),
        dual_bound=dual_bound,
        absolute_gap=max(0.0, float(row_objective) - dual_bound),
        relative_gap=max(0.0, float(row_objective) - dual_bound)
        / max(1.0, abs(float(row_objective))),
        certified=True,
        solver="scipy_highs_milp_unary_pairwise",
        node_count=_mip_node_count(primary),
        lexicographic_tie_break=True,
        diagnostics={
            "primary_status": int(primary.status),
            "primary_message": str(primary.message),
            "primary_solver_objective": float(primary.fun),
            "primary_canonical_objective": float(incumbent),
            "primary_mip_gap": float(getattr(primary, "mip_gap", math.inf)),
            "primary_tolerance": primary_tolerance,
            "primary_face_unique": False,
            "uniqueness_status": int(alternative.status),
            "uniqueness_message": str(alternative.message),
            "node_cap": 2_000_000,
            "threads": 1,
            "solver_seed": 0,
            "lexicographic_records": lex_records,
        },
    )


def price_legal_row(
    feature_map: ShadowFeatureMap,
    coefficients: np.ndarray,
    *,
    solver: Literal["auto", "enumerate", "milp"] = "auto",
    enumeration_cap: int = 100_000,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> RowPricingResult:
    mode = str(solver)
    if mode not in {"auto", "enumerate", "milp"}:
        raise ValueError("Pricing solver must be auto, enumerate, or milp")
    if mode == "enumerate" or (
        mode == "auto" and feature_map.domain.domain_size <= int(enumeration_cap)
    ):
        return _enumeration_price(feature_map, coefficients)
    return _certified_milp_price(
        feature_map,
        coefficients,
        progress_callback=progress_callback,
    )


__all__ = [
    "ROW_PRICING_METHOD",
    "SHADOW_FEATURE_METHOD",
    "RowPricingResult",
    "ShadowFeatureMap",
    "ccf_moment_answer",
    "price_legal_row",
]
