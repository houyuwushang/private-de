from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from qdte.measurement.factorization import HierarchicalPairStrategy
from qdte.queries.orthogonal import helmert_contrast, pair_reconstruction_maps
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue


@dataclass(frozen=True)
class FactoredPublicQuery:
    qid: int
    scope: tuple[int, ...]
    constant_per_row: float
    coefficients: dict[str, np.ndarray]


@dataclass(frozen=True)
class WorkloadFactorization:
    importance_by_block: dict[str, float]
    supported_query_ids: tuple[int, ...]
    unsupported_query_ids: tuple[int, ...]
    unsupported_scope_orders: dict[int, int]
    supported_weight: float
    unsupported_weight: float


def query_public_scope(qcat: QueryCatalogue, qid: int) -> tuple[int, ...]:
    index = int(qid)
    if not 0 <= index < int(qcat.m):
        raise IndexError(index)
    ordinary = {
        int(qcat.attrs[index, term]) for term in range(int(qcat.num_terms[index]))
    }
    linear = {
        int(qcat.linear_attrs[index, term])
        for term in range(int(qcat.linear_num_terms[index]))
    }
    return tuple(sorted(ordinary | linear))


def _ordinary_mask(
    values: np.ndarray,
    *,
    op: int,
    value: int,
    low: int,
    high: int,
) -> np.ndarray:
    if int(op) == OP_EQ:
        return values == int(value)
    if int(op) == OP_LE:
        return values <= int(value)
    if int(op) == OP_GE:
        return values >= int(value)
    if int(op) == OP_RANGE:
        return (values >= int(low)) & (values <= int(high))
    raise ValueError(f"unknown query operator {op}")


def public_query_indicator(
    qcat: QueryCatalogue,
    qid: int,
    cardinalities: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray]:
    cards = np.asarray(cardinalities, dtype=np.int64)
    if cards.ndim != 1 or cards.size == 0 or np.any(cards < 2):
        raise ValueError("cardinalities must be a non-empty vector with values >= 2")
    scope = query_public_scope(qcat, qid)
    if not scope or len(scope) > 2:
        raise ValueError("query must have public scope order one or two")
    if scope[-1] >= len(cards):
        raise ValueError("query scope exceeds the supplied cardinalities")

    shape = tuple(int(cards[attr]) for attr in scope)
    grid = np.indices(shape, dtype=np.int64)
    indicator = np.ones(shape, dtype=bool)

    ordinary_terms = int(qcat.num_terms[int(qid)])
    for term in range(ordinary_terms):
        attr = int(qcat.attrs[int(qid), term])
        axis = scope.index(attr)
        indicator &= _ordinary_mask(
            grid[axis],
            op=int(qcat.ops[int(qid), term]),
            value=int(qcat.values[int(qid), term]),
            low=int(qcat.lows[int(qid), term]),
            high=int(qcat.highs[int(qid), term]),
        )

    linear_terms = int(qcat.linear_num_terms[int(qid)])
    if linear_terms:
        score = np.zeros(shape, dtype=np.float64)
        for term in range(linear_terms):
            attr = int(qcat.linear_attrs[int(qid), term])
            axis = scope.index(attr)
            score += float(qcat.linear_weights[int(qid), term]) * grid[axis]
        indicator &= score <= float(qcat.linear_thresholds[int(qid)])

    return scope, indicator.astype(np.float64)


def factorize_public_query(
    strategy: HierarchicalPairStrategy,
    qcat: QueryCatalogue,
    qid: int,
) -> FactoredPublicQuery | None:
    scope = query_public_scope(qcat, qid)
    if not scope or len(scope) > 2:
        return None
    scope, indicator = public_query_indicator(
        qcat,
        qid,
        strategy.cardinalities,
    )
    if len(scope) == 1:
        attr = scope[0]
        coefficients = {
            f"oneway_contrast:{attr}": helmert_contrast(
                strategy.cardinalities[attr]
            ).T
            @ indicator.reshape(-1)
        }
        return FactoredPublicQuery(
            qid=int(qid),
            scope=scope,
            constant_per_row=float(np.mean(indicator)),
            coefficients=coefficients,
        )

    left, right = scope
    if (left, right) not in strategy.pairs:
        return None
    left_map, right_map, interaction_map = pair_reconstruction_maps(
        strategy.cardinalities[left],
        strategy.cardinalities[right],
    )
    flat = indicator.reshape(-1)
    coefficients = {
        f"oneway_contrast:{left}": left_map.T @ flat,
        f"oneway_contrast:{right}": right_map.T @ flat,
        f"pair_interaction:{left}:{right}": interaction_map.T @ flat,
    }
    return FactoredPublicQuery(
        qid=int(qid),
        scope=scope,
        constant_per_row=float(np.mean(indicator)),
        coefficients=coefficients,
    )


def factorize_public_workload(
    strategy: HierarchicalPairStrategy,
    qcat: QueryCatalogue,
    *,
    weights: np.ndarray | list[float] | None = None,
    include: np.ndarray | list[bool] | None = None,
) -> WorkloadFactorization:
    qcat.validate(np.asarray(strategy.cardinalities, dtype=np.int64))
    query_weights = (
        np.ones(int(qcat.m), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if query_weights.shape != (int(qcat.m),):
        raise ValueError("weights must have one entry per query")
    if not np.all(np.isfinite(query_weights)) or np.any(query_weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    query_include = (
        np.ones(int(qcat.m), dtype=bool)
        if include is None
        else np.asarray(include, dtype=bool)
    )
    if query_include.shape != (int(qcat.m),):
        raise ValueError("include must have one entry per query")

    importance = {block.name: 0.0 for block in strategy.blocks}
    supported: list[int] = []
    unsupported: list[int] = []
    unsupported_orders: Counter[int] = Counter()
    supported_weight = 0.0
    unsupported_weight = 0.0
    for qid in np.flatnonzero(query_include).tolist():
        weight = float(query_weights[qid])
        factored = factorize_public_query(strategy, qcat, qid)
        if factored is None:
            unsupported.append(int(qid))
            unsupported_orders[len(query_public_scope(qcat, qid))] += 1
            unsupported_weight += weight
            continue
        supported.append(int(qid))
        supported_weight += weight
        for name, coefficients in factored.coefficients.items():
            importance[name] += weight * float(
                np.dot(coefficients.reshape(-1), coefficients.reshape(-1))
            )

    return WorkloadFactorization(
        importance_by_block=importance,
        supported_query_ids=tuple(supported),
        unsupported_query_ids=tuple(unsupported),
        unsupported_scope_orders=dict(sorted(unsupported_orders.items())),
        supported_weight=float(supported_weight),
        unsupported_weight=float(unsupported_weight),
    )


def allocation_risk(
    strategy: HierarchicalPairStrategy,
    importance_by_block: Mapping[str, float],
    rho_by_block: Mapping[str, float],
) -> float:
    expected = {block.name for block in strategy.blocks}
    if set(importance_by_block) != expected or set(rho_by_block) != expected:
        raise ValueError("importance and rho mappings must cover every strategy block")
    total = 0.0
    for block in strategy.blocks:
        importance = float(importance_by_block[block.name])
        rho = float(rho_by_block[block.name])
        if not np.isfinite(importance) or importance < 0.0:
            raise ValueError("importance must be finite and nonnegative")
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("rho must be finite and positive")
        total += importance * block.sensitivity_l2**2 / (2.0 * rho)
    return float(total)


def optimal_rho_for_importance(
    strategy: HierarchicalPairStrategy,
    importance_by_block: Mapping[str, float],
    rho_total: float,
) -> dict[str, float]:
    expected = {block.name for block in strategy.blocks}
    if set(importance_by_block) != expected:
        raise ValueError("importance mapping must cover every strategy block")
    total = float(rho_total)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("rho_total must be finite and positive")
    weights = np.asarray(
        [
            block.sensitivity_l2
            * np.sqrt(float(importance_by_block[block.name]))
            for block in strategy.blocks
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        raise ValueError("at least one block must have positive finite importance")
    allocation = total * weights / float(np.sum(weights))
    allocation[-1] += total - float(np.sum(allocation))
    return {
        block.name: float(rho)
        for block, rho in zip(strategy.blocks, allocation, strict=True)
    }
