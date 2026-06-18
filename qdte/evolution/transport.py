from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from qdte.evolution.candidates import CandidateBatch
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.eval_jax import eval_records_queries_arrays
from qdte.queries.types import QueryCatalogue


@dataclass
class TransportResult:
    accepted_indices: np.ndarray
    delta_sum: np.ndarray
    batch_advantage: float
    mean_advantage: float
    diagnostics: dict[str, float | int] = field(default_factory=dict)


@dataclass
class _ConstructiveUnit:
    advantage: float
    candidate_indices: tuple[int, ...]
    local_indices: tuple[int, ...]
    delta: np.ndarray
    cost: float
    unit_type: str


def select_top_nonconflicting(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    max_accept: int,
    min_advantage: float,
) -> np.ndarray:
    if max_accept <= 0:
        return np.empty(0, dtype=np.int32)
    valid = np.flatnonzero(np.isfinite(advantages) & (advantages > float(min_advantage)))
    if len(valid) == 0:
        return np.empty(0, dtype=np.int32)
    valid_scores = advantages[valid]
    pool_size = min(len(valid), max(4 * max_accept, max_accept))
    while True:
        if pool_size < len(valid):
            pool = np.argpartition(-valid_scores, pool_size - 1)[:pool_size]
        else:
            pool = np.arange(len(valid))
        order = pool[np.argsort(-valid_scores[pool])]
        seen_rows: set[int] = set()
        selected: list[int] = []
        for local_idx in order.tolist():
            idx = int(valid[local_idx])
            row_id = int(candidates.row_ids[idx])
            if row_id in seen_rows:
                continue
            seen_rows.add(row_id)
            selected.append(idx)
            if len(selected) >= max_accept:
                break
        if len(selected) >= max_accept or pool_size >= len(valid):
            return np.asarray(selected, dtype=np.int32)
        pool_size = min(len(valid), pool_size * 2)


def select_nonconflicting_in_order(
    candidates: CandidateBatch,
    order: np.ndarray,
    max_accept: int,
) -> np.ndarray:
    if max_accept <= 0:
        return np.empty(0, dtype=np.int32)
    seen_rows: set[int] = set()
    selected: list[int] = []
    for idx_raw in order.tolist():
        idx = int(idx_raw)
        if idx < 0 or idx >= candidates.size:
            continue
        row_id = int(candidates.row_ids[idx])
        if row_id in seen_rows:
            continue
        seen_rows.add(row_id)
        selected.append(idx)
        if len(selected) >= max_accept:
            break
    return np.asarray(selected, dtype=np.int32)


def batch_advantage(
    residual: np.ndarray,
    inv_variance: np.ndarray,
    delta_sum: np.ndarray,
    cost_sum: float,
    lambda_cost: float,
) -> float:
    d = delta_sum.astype(np.float32)
    w = residual.astype(np.float32) * inv_variance.astype(np.float32)
    linear = float(d @ w)
    quad = float((d * d) @ inv_variance.astype(np.float32))
    return linear - 0.5 * quad - float(lambda_cost) * float(cost_sum)


def _finite_candidate_pool(
    advantages: np.ndarray,
    max_accept: int,
    pool_multiplier: int,
    max_pool: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(advantages))
    if len(valid) == 0 or max_accept <= 0:
        return np.empty(0, dtype=np.int32)
    pool_size = len(valid)
    if pool_multiplier > 0:
        pool_size = min(pool_size, max(max_accept, int(pool_multiplier) * max_accept))
    if max_pool > 0:
        pool_size = min(pool_size, int(max_pool))
    if pool_size < len(valid):
        valid = rng.choice(valid, size=pool_size, replace=False)
    return np.asarray(valid, dtype=np.int32)


def _ranked_finite_candidate_pool(
    advantages: np.ndarray,
    max_accept: int,
    pool_multiplier: int,
    max_pool: int,
) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(advantages))
    if len(valid) == 0 or max_accept <= 0:
        return np.empty(0, dtype=np.int32)
    if max_pool > 0:
        pool_size = min(len(valid), int(max_pool))
    else:
        pool_size = len(valid)
    if pool_multiplier > 0:
        pool_size = min(pool_size, max(max_accept, int(pool_multiplier) * max_accept))
    valid_scores = advantages[valid]
    if pool_size < len(valid):
        pool = np.argpartition(-valid_scores, pool_size - 1)[:pool_size]
    else:
        pool = np.arange(len(valid))
    order = pool[np.argsort(-valid_scores[pool])]
    return valid[order].astype(np.int32, copy=False)


def _random_nonconflicting_group(
    pool_indices: np.ndarray,
    row_ids: np.ndarray,
    group_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if group_size <= 0 or len(pool_indices) == 0:
        return np.empty(0, dtype=np.int32)
    order = rng.permutation(len(pool_indices))
    used_rows: set[int] = set()
    selected: list[int] = []
    for local_pos in order.tolist():
        candidate_idx = int(pool_indices[local_pos])
        row_id = int(row_ids[candidate_idx])
        if row_id in used_rows:
            continue
        used_rows.add(row_id)
        selected.append(candidate_idx)
        if len(selected) >= group_size:
            break
    return np.asarray(selected, dtype=np.int32)


def choose_random_group_transport(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    max_accept: int,
    min_advantage: float,
    rng: np.random.Generator,
    group_count: int = 64,
    min_group_size: int = 2,
    max_group_size: int = 0,
    pool_multiplier: int = 0,
    max_pool: int = 0,
    delta_index: QueryDeltaIndex | None = None,
) -> TransportResult:
    """Sample random nonconflicting edit groups and accept the best aggregate advantage.

    This is an exploratory transport for separating two effects:
    random individual edit proposals versus group-level QDTE advantage scoring.
    It does not use true answers and does not use individual positive advantage
    as a filter.
    """
    diagnostics: dict[str, float | int] = {
        "random_group_mode": 1,
        "random_group_pool_candidates": 0,
        "random_group_groups_evaluated": 0,
        "random_group_positive_groups": 0,
        "random_group_best_group_size": 0,
        "random_group_accepted_candidates": 0,
        "random_group_groups_with_negative_member": 0,
        "random_group_sparse_delta": int(delta_index is not None),
    }
    empty = TransportResult(
        accepted_indices=np.empty(0, dtype=np.int32),
        delta_sum=np.zeros_like(residual, dtype=np.float32),
        batch_advantage=0.0,
        mean_advantage=0.0,
        diagnostics=diagnostics,
    )
    if max_accept <= 0 or candidates.size == 0 or group_count <= 0:
        return empty

    pool_indices = _finite_candidate_pool(advantages, max_accept, pool_multiplier, max_pool, rng)
    diagnostics["random_group_pool_candidates"] = int(len(pool_indices))
    if len(pool_indices) == 0:
        return empty

    deltas = _candidate_deltas(candidates, pool_indices, qcat, delta_index=delta_index).astype(np.float32)
    local_for_candidate = {int(candidate_idx): local_idx for local_idx, candidate_idx in enumerate(pool_indices.tolist())}
    max_size = int(max_group_size) if int(max_group_size) > 0 else int(max_accept)
    max_size = max(1, min(max_size, int(max_accept), len(pool_indices)))
    min_size = max(1, min(int(min_group_size), max_size))

    best_adv = -np.inf
    best_group = np.empty(0, dtype=np.int32)
    best_delta = np.zeros_like(residual, dtype=np.float32)
    positive_groups = 0
    negative_member_groups = 0
    evaluated = 0
    for _ in range(int(group_count)):
        if min_size == max_size:
            group_size = max_size
        else:
            group_size = int(rng.integers(min_size, max_size + 1))
        group = _random_nonconflicting_group(pool_indices, candidates.row_ids, group_size, rng)
        if len(group) < min_size:
            continue
        local = np.asarray([local_for_candidate[int(idx)] for idx in group.tolist()], dtype=np.int32)
        delta_sum = deltas[local].sum(axis=0, dtype=np.float32)
        cost_sum = float(candidates.edit_cost[group].sum())
        adv = batch_advantage(residual, inv_variance, delta_sum, cost_sum, lambda_cost)
        evaluated += 1
        if adv > 0.0:
            positive_groups += 1
        if np.any(advantages[group] <= float(min_advantage)):
            negative_member_groups += 1
        if adv > best_adv:
            best_adv = float(adv)
            best_group = group
            best_delta = delta_sum.astype(np.float32, copy=True)

    diagnostics["random_group_groups_evaluated"] = int(evaluated)
    diagnostics["random_group_positive_groups"] = int(positive_groups)
    diagnostics["random_group_groups_with_negative_member"] = int(negative_member_groups)
    diagnostics["random_group_best_group_size"] = int(len(best_group))
    if evaluated == 0 or best_adv <= float(min_advantage):
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(best_adv if np.isfinite(best_adv) else 0.0),
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    diagnostics["random_group_accepted_candidates"] = int(len(best_group))
    return TransportResult(
        accepted_indices=best_group.astype(np.int32, copy=False),
        delta_sum=best_delta.astype(np.float32, copy=False),
        batch_advantage=float(best_adv),
        mean_advantage=float(advantages[best_group].mean()),
        diagnostics=diagnostics,
    )


def choose_directed_group_transport(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    max_accept: int,
    min_advantage: float,
    seed_count: int = 32,
    min_group_size: int = 1,
    max_group_size: int = 0,
    pool_multiplier: int = 0,
    max_pool: int = 0,
    allow_negative_steps: bool = False,
    delta_index: QueryDeltaIndex | None = None,
) -> TransportResult:
    """Build edit groups by greedy aggregate-advantage lookahead.

    Candidate edits may be random or directed; the group construction is
    residual/delta-directed. Starting from ranked seed edits, each expansion
    scores candidate marginals against the virtual residual induced by the
    current partial group.
    """
    diagnostics: dict[str, float | int] = {
        "directed_group_mode": 1,
        "directed_group_pool_candidates": 0,
        "directed_group_seed_candidates": 0,
        "directed_group_groups_evaluated": 0,
        "directed_group_positive_groups": 0,
        "directed_group_expansion_steps": 0,
        "directed_group_best_group_size": 0,
        "directed_group_accepted_candidates": 0,
        "directed_group_groups_with_negative_member": 0,
        "directed_group_sparse_delta": int(delta_index is not None),
    }
    empty = TransportResult(
        accepted_indices=np.empty(0, dtype=np.int32),
        delta_sum=np.zeros_like(residual, dtype=np.float32),
        batch_advantage=0.0,
        mean_advantage=0.0,
        diagnostics=diagnostics,
    )
    if max_accept <= 0 or candidates.size == 0:
        return empty

    pool_indices = _ranked_finite_candidate_pool(advantages, max_accept, pool_multiplier, max_pool)
    diagnostics["directed_group_pool_candidates"] = int(len(pool_indices))
    if len(pool_indices) == 0:
        return empty

    deltas = _candidate_deltas(candidates, pool_indices, qcat, delta_index=delta_index).astype(np.float32)
    inv = inv_variance.astype(np.float32, copy=False)
    residual_f = residual.astype(np.float32, copy=False)
    edit_cost = candidates.edit_cost[pool_indices].astype(np.float32, copy=False)
    local_advantages = advantages[pool_indices].astype(np.float32, copy=False)
    quad = (deltas * deltas) @ inv

    max_size = int(max_group_size) if int(max_group_size) > 0 else int(max_accept)
    max_size = max(1, min(max_size, int(max_accept), len(pool_indices)))
    min_size = max(1, min(int(min_group_size), max_size))
    if seed_count <= 0:
        num_seeds = len(pool_indices)
    else:
        num_seeds = min(int(seed_count), len(pool_indices))
    diagnostics["directed_group_seed_candidates"] = int(num_seeds)

    best_adv = -np.inf
    best_local_group: list[int] = []
    best_delta = np.zeros_like(residual_f, dtype=np.float32)
    positive_groups = 0
    negative_member_groups = 0
    expansion_steps = 0

    for seed_local in range(num_seeds):
        used_rows = {int(candidates.row_ids[pool_indices[seed_local]])}
        used_local = {int(seed_local)}
        group_local = [int(seed_local)]
        delta_sum = deltas[seed_local].astype(np.float32, copy=True)
        cost_sum = float(edit_cost[seed_local])
        seed_best_adv = -np.inf
        seed_best_has_negative_member = False

        def consider_current_group() -> None:
            nonlocal best_adv, best_local_group, best_delta, seed_best_adv, seed_best_has_negative_member
            if len(group_local) < min_size:
                return
            adv = batch_advantage(residual_f, inv, delta_sum, cost_sum, lambda_cost)
            group_advantages = local_advantages[np.asarray(group_local, dtype=np.int32)]
            has_negative_member = bool(np.any(group_advantages <= float(min_advantage)))
            if adv > seed_best_adv:
                seed_best_adv = float(adv)
                seed_best_has_negative_member = has_negative_member
            if adv > best_adv:
                best_adv = float(adv)
                best_local_group = list(group_local)
                best_delta = delta_sum.astype(np.float32, copy=True)

        consider_current_group()
        while len(group_local) < max_size:
            current_weight = (residual_f - delta_sum).astype(np.float32, copy=False) * inv
            marginals = deltas @ current_weight - 0.5 * quad - float(lambda_cost) * edit_cost
            if used_local:
                marginals[np.asarray(list(used_local), dtype=np.int32)] = -np.inf
            for local_idx, candidate_idx in enumerate(pool_indices.tolist()):
                if int(candidates.row_ids[candidate_idx]) in used_rows:
                    marginals[local_idx] = -np.inf
            next_local = int(np.argmax(marginals))
            next_margin = float(marginals[next_local])
            if not np.isfinite(next_margin):
                break
            if not allow_negative_steps and next_margin <= float(min_advantage):
                break
            group_local.append(next_local)
            used_local.add(next_local)
            used_rows.add(int(candidates.row_ids[pool_indices[next_local]]))
            delta_sum += deltas[next_local]
            cost_sum += float(edit_cost[next_local])
            expansion_steps += 1
            consider_current_group()
        if seed_best_adv > 0.0:
            positive_groups += 1
        if seed_best_has_negative_member:
            negative_member_groups += 1

    diagnostics["directed_group_groups_evaluated"] = int(num_seeds)
    diagnostics["directed_group_positive_groups"] = int(positive_groups)
    diagnostics["directed_group_expansion_steps"] = int(expansion_steps)
    diagnostics["directed_group_groups_with_negative_member"] = int(negative_member_groups)
    diagnostics["directed_group_best_group_size"] = int(len(best_local_group))
    if not best_local_group or best_adv <= float(min_advantage):
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(best_adv if np.isfinite(best_adv) else 0.0),
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    accepted_indices = pool_indices[np.asarray(best_local_group, dtype=np.int32)]
    diagnostics["directed_group_accepted_candidates"] = int(len(accepted_indices))
    return TransportResult(
        accepted_indices=accepted_indices.astype(np.int32, copy=False),
        delta_sum=best_delta.astype(np.float32, copy=False),
        batch_advantage=float(best_adv),
        mean_advantage=float(advantages[accepted_indices].mean()),
        diagnostics=diagnostics,
    )


@partial(jax.jit, static_argnames=("min_group_size", "max_group_size", "allow_negative_steps"))
def _directed_group_search_jit(
    deltas: jax.Array,
    row_ids: jax.Array,
    edit_cost: jax.Array,
    local_advantages: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    seed_locals: jax.Array,
    lambda_cost: jax.Array,
    min_advantage: jax.Array,
    *,
    min_group_size: int,
    max_group_size: int,
    allow_negative_steps: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    d = deltas.astype(jnp.float32)
    rows = row_ids.astype(jnp.int32)
    costs = edit_cost.astype(jnp.float32)
    adv = local_advantages.astype(jnp.float32)
    residual_f = residual.astype(jnp.float32)
    inv = inv_variance.astype(jnp.float32)
    weights = residual_f * inv
    quad = (d * d) @ inv
    p = d.shape[0]
    group_slots = jnp.arange(p, dtype=jnp.int32)
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float32)
    min_adv = min_advantage.astype(jnp.float32)

    def group_advantage(delta_sum: jax.Array, cost_sum: jax.Array) -> jax.Array:
        return (
            delta_sum @ weights
            - 0.5 * ((delta_sum * delta_sum) @ inv)
            - lambda_cost.astype(jnp.float32) * cost_sum
        ).astype(jnp.float32)

    def run_seed(seed_local: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        seed_local = seed_local.astype(jnp.int32)
        seed_row = rows[seed_local]
        group_mask = group_slots == seed_local
        used_rows = rows == seed_row
        delta_sum = d[seed_local]
        cost_sum = costs[seed_local]
        count = jnp.asarray(1, dtype=jnp.int32)
        seed_adv = group_advantage(delta_sum, cost_sum)
        valid_seed_group = count >= jnp.asarray(min_group_size, dtype=jnp.int32)
        has_negative_member = adv[seed_local] <= min_adv
        best_adv = jnp.where(valid_seed_group, seed_adv, neg_inf)
        best_mask = jnp.where(valid_seed_group, group_mask, jnp.zeros_like(group_mask))
        best_count = jnp.where(valid_seed_group, count, jnp.asarray(0, dtype=jnp.int32))
        best_has_negative = jnp.where(valid_seed_group, has_negative_member, False)
        expansion_steps = jnp.asarray(0, dtype=jnp.int32)

        def body(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            _: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
            (
                group_mask,
                used_rows,
                delta_sum,
                cost_sum,
                count,
                best_adv,
                best_mask,
                best_count,
                best_has_negative,
            ) = carry
            current_weight = (residual_f - delta_sum) * inv
            marginals = d @ current_weight - 0.5 * quad - lambda_cost.astype(jnp.float32) * costs
            marginals = jnp.where(used_rows, neg_inf, marginals)
            next_local = jnp.argmax(marginals).astype(jnp.int32)
            next_margin = marginals[next_local]
            can_add = jnp.isfinite(next_margin)
            if not allow_negative_steps:
                can_add = can_add & (next_margin > min_adv)
            next_row = rows[next_local]
            next_one_hot = group_slots == next_local
            next_row_mask = rows == next_row
            new_group_mask = jnp.where(can_add, group_mask | next_one_hot, group_mask)
            new_used_rows = jnp.where(can_add, used_rows | next_row_mask, used_rows)
            new_delta_sum = jnp.where(can_add, delta_sum + d[next_local], delta_sum)
            new_cost_sum = jnp.where(can_add, cost_sum + costs[next_local], cost_sum)
            new_count = jnp.where(can_add, count + jnp.asarray(1, dtype=jnp.int32), count)
            new_has_negative = jnp.any(jnp.where(new_group_mask, adv <= min_adv, False))
            candidate_adv = group_advantage(new_delta_sum, new_cost_sum)
            valid_group = new_count >= jnp.asarray(min_group_size, dtype=jnp.int32)
            better = valid_group & (candidate_adv > best_adv)
            out_best_adv = jnp.where(better, candidate_adv, best_adv)
            out_best_mask = jnp.where(better, new_group_mask, best_mask)
            out_best_count = jnp.where(better, new_count, best_count)
            out_best_has_negative = jnp.where(better, new_has_negative, best_has_negative)
            return (
                new_group_mask,
                new_used_rows,
                new_delta_sum,
                new_cost_sum,
                new_count,
                out_best_adv,
                out_best_mask,
                out_best_count,
                out_best_has_negative,
            ), None

        initial = (
            group_mask,
            used_rows,
            delta_sum,
            cost_sum,
            count,
            best_adv,
            best_mask,
            best_count,
            best_has_negative,
        )
        scanned, _ = jax.lax.scan(body, initial, jnp.arange(max(0, int(max_group_size) - 1), dtype=jnp.int32))
        _, _, _, _, final_count, best_adv, best_mask, best_count, best_has_negative = scanned
        expansion_steps = jnp.maximum(final_count - jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        return best_adv, best_mask, best_count, best_has_negative, expansion_steps

    seed_best_adv, seed_best_masks, seed_best_counts, seed_has_negative, seed_expansions = jax.vmap(run_seed)(seed_locals)
    best_seed = jnp.argmax(seed_best_adv).astype(jnp.int32)
    best_adv = seed_best_adv[best_seed]
    best_mask = seed_best_masks[best_seed]
    best_count = seed_best_counts[best_seed]
    best_delta = jnp.sum(jnp.where(best_mask[:, None], d, 0.0), axis=0).astype(jnp.float32)
    positive_groups = jnp.sum(seed_best_adv > 0.0).astype(jnp.int32)
    negative_member_groups = jnp.sum(seed_has_negative).astype(jnp.int32)
    expansion_steps = jnp.sum(seed_expansions).astype(jnp.int32)
    return (
        best_mask,
        best_delta,
        best_adv.astype(jnp.float32),
        best_count.astype(jnp.int32),
        positive_groups,
        negative_member_groups,
        expansion_steps,
    )


def choose_directed_group_transport_jax(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    max_accept: int,
    min_advantage: float,
    seed_count: int = 32,
    min_group_size: int = 1,
    max_group_size: int = 0,
    pool_multiplier: int = 0,
    max_pool: int = 0,
    allow_negative_steps: bool = False,
) -> TransportResult:
    diagnostics: dict[str, float | int] = {
        "directed_group_mode": 1,
        "directed_group_jax_mode": 1,
        "directed_group_pool_candidates": 0,
        "directed_group_seed_candidates": 0,
        "directed_group_groups_evaluated": 0,
        "directed_group_positive_groups": 0,
        "directed_group_expansion_steps": 0,
        "directed_group_best_group_size": 0,
        "directed_group_accepted_candidates": 0,
        "directed_group_groups_with_negative_member": 0,
        "directed_group_sparse_delta": 0,
    }
    empty = TransportResult(
        accepted_indices=np.empty(0, dtype=np.int32),
        delta_sum=np.zeros_like(residual, dtype=np.float32),
        batch_advantage=0.0,
        mean_advantage=0.0,
        diagnostics=diagnostics,
    )
    if max_accept <= 0 or candidates.size == 0:
        return empty

    pool_indices = _ranked_finite_candidate_pool(advantages, max_accept, pool_multiplier, max_pool)
    diagnostics["directed_group_pool_candidates"] = int(len(pool_indices))
    if len(pool_indices) == 0:
        return empty

    max_size = int(max_group_size) if int(max_group_size) > 0 else int(max_accept)
    max_size = max(1, min(max_size, int(max_accept), len(pool_indices)))
    min_size = max(1, min(int(min_group_size), max_size))
    if int(seed_count) <= 0:
        num_seeds = len(pool_indices)
    else:
        num_seeds = min(int(seed_count), len(pool_indices))
    diagnostics["directed_group_seed_candidates"] = int(num_seeds)
    diagnostics["directed_group_groups_evaluated"] = int(num_seeds)
    if num_seeds <= 0:
        return empty

    deltas = _candidate_deltas_jax(candidates, pool_indices, qcat).astype(jnp.float32)
    seed_locals = jnp.arange(num_seeds, dtype=jnp.int32)
    (
        best_mask,
        best_delta,
        best_adv,
        best_count,
        positive_groups,
        negative_member_groups,
        expansion_steps,
    ) = _directed_group_search_jit(
        deltas,
        jnp.asarray(candidates.row_ids[pool_indices], dtype=jnp.int32),
        jnp.asarray(candidates.edit_cost[pool_indices], dtype=jnp.float32),
        jnp.asarray(advantages[pool_indices], dtype=jnp.float32),
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(inv_variance, dtype=jnp.float32),
        seed_locals,
        jnp.asarray(lambda_cost, dtype=jnp.float32),
        jnp.asarray(min_advantage, dtype=jnp.float32),
        min_group_size=min_size,
        max_group_size=max_size,
        allow_negative_steps=bool(allow_negative_steps),
    )
    best_adv_f = float(np.asarray(best_adv))
    best_count_i = int(np.asarray(best_count))
    diagnostics["directed_group_positive_groups"] = int(np.asarray(positive_groups))
    diagnostics["directed_group_expansion_steps"] = int(np.asarray(expansion_steps))
    diagnostics["directed_group_groups_with_negative_member"] = int(np.asarray(negative_member_groups))
    diagnostics["directed_group_best_group_size"] = int(best_count_i)
    if best_count_i <= 0 or best_adv_f <= float(min_advantage):
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=best_adv_f if np.isfinite(best_adv_f) else 0.0,
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    mask_np = np.asarray(best_mask, dtype=bool)
    accepted_indices = pool_indices[np.flatnonzero(mask_np)].astype(np.int32, copy=False)
    diagnostics["directed_group_accepted_candidates"] = int(len(accepted_indices))
    return TransportResult(
        accepted_indices=accepted_indices,
        delta_sum=np.asarray(best_delta, dtype=np.float32),
        batch_advantage=best_adv_f,
        mean_advantage=float(advantages[accepted_indices].mean()) if len(accepted_indices) else 0.0,
        diagnostics=diagnostics,
    )


def choose_blind_transport(
    candidates: CandidateBatch,
    deltas: np.ndarray,
    selected: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
) -> TransportResult:
    if len(selected) == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
            diagnostics={"blind_accept_mode": 1, "blind_batch_advantage_diagnostic": 0.0},
        )
    delta_sum = deltas.astype(np.float32).sum(axis=0)
    cost_sum = float(candidates.edit_cost[selected].sum())
    diagnostic_advantage = batch_advantage(residual, inv_variance, delta_sum, cost_sum, lambda_cost)
    return TransportResult(
        accepted_indices=selected.astype(np.int32, copy=False),
        delta_sum=delta_sum.astype(np.float32),
        batch_advantage=float(diagnostic_advantage),
        mean_advantage=0.0,
        diagnostics={
            "blind_accept_mode": 1,
            "blind_accepted_candidates": int(len(selected)),
            "blind_batch_advantage_diagnostic": float(diagnostic_advantage),
        },
    )


def _candidate_deltas(
    candidates: CandidateBatch,
    indices: np.ndarray,
    qcat: QueryCatalogue,
    delta_index: QueryDeltaIndex | None = None,
) -> np.ndarray:
    if delta_index is not None:
        return delta_index.dense_candidate_deltas(candidates.old_rows[indices], candidates.new_rows[indices])
    return np.asarray(_candidate_deltas_jax(candidates, indices, qcat), dtype=np.int8)


def _candidate_deltas_jax(
    candidates: CandidateBatch,
    indices: np.ndarray,
    qcat: QueryCatalogue,
) -> jax.Array:
    if len(indices) == 0:
        return jnp.empty((0, qcat.m), dtype=jnp.int8)
    qarrays = qcat.eval_arrays()
    phi_old = eval_records_queries_arrays(
        jnp.asarray(candidates.old_rows[indices], dtype=jnp.int32),
        *(
            jnp.asarray(arr, dtype=jnp.float32)
            if idx in {6, 7}
            else jnp.asarray(arr, dtype=jnp.int32)
            for idx, arr in enumerate(qarrays)
        ),
    )
    phi_new = eval_records_queries_arrays(
        jnp.asarray(candidates.new_rows[indices], dtype=jnp.int32),
        *(
            jnp.asarray(arr, dtype=jnp.float32)
            if idx in {6, 7}
            else jnp.asarray(arr, dtype=jnp.int32)
            for idx, arr in enumerate(qarrays)
        ),
    )
    return phi_new.astype(jnp.int8) - phi_old.astype(jnp.int8)


@jax.jit
def _delta_quad_jit(
    deltas: jax.Array,
    inv_variance: jax.Array,
) -> jax.Array:
    d = deltas.astype(jnp.float32)
    inv = inv_variance.astype(jnp.float32)
    quad = (d * d) @ inv
    return quad.astype(jnp.float32)


@dataclass
class _AtomFlowEdge:
    source_atom: tuple[int, ...]
    target_atom: tuple[int, ...]
    candidate_indices: list[int]
    row_ids: list[int]
    qids: np.ndarray
    signs: np.ndarray
    cost: float
    cursor: int = 0
    active: bool = True
    marginal: float = 0.0
    version: int = 0

    def next_available_index(self, used_rows: set[int]) -> int | None:
        while self.cursor < len(self.candidate_indices) and self.row_ids[self.cursor] in used_rows:
            self.cursor += 1
        if self.cursor >= len(self.candidate_indices):
            self.active = False
            return None
        return self.candidate_indices[self.cursor]


def _candidate_pool(
    advantages: np.ndarray,
    max_accept: int,
    min_advantage: float,
    pool_multiplier: int,
    max_pool: int,
) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(advantages) & (advantages > float(min_advantage)))
    if len(valid) == 0 or max_accept <= 0:
        return np.empty(0, dtype=np.int32)
    if max_pool > 0:
        pool_size = min(len(valid), int(max_pool))
    else:
        pool_size = len(valid)
    if pool_multiplier > 0:
        pool_size = min(pool_size, max(max_accept, int(pool_multiplier) * max_accept))
    valid_scores = advantages[valid]
    if pool_size < len(valid):
        pool = np.argpartition(-valid_scores, pool_size - 1)[:pool_size]
    else:
        pool = np.arange(len(valid))
    order = pool[np.argsort(-valid_scores[pool])]
    return valid[order].astype(np.int32, copy=False)


def _build_atom_flow_edges(
    candidates: CandidateBatch,
    pool_indices: np.ndarray,
    deltas: np.ndarray,
    advantages: np.ndarray,
    num_queries: int,
) -> tuple[list[_AtomFlowEdge], list[list[tuple[int, int]]]]:
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[float, int, int, int]]] = {}
    first_delta: dict[tuple[tuple[int, ...], tuple[int, ...]], np.ndarray] = {}
    first_cost: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for local_idx, candidate_idx in enumerate(pool_indices.tolist()):
        source_atom = tuple(int(x) for x in candidates.old_rows[candidate_idx].tolist())
        target_atom = tuple(int(x) for x in candidates.new_rows[candidate_idx].tolist())
        if source_atom == target_atom:
            continue
        key = (source_atom, target_atom)
        grouped.setdefault(key, []).append(
            (
                -float(advantages[candidate_idx]),
                int(candidate_idx),
                int(candidates.row_ids[candidate_idx]),
                int(local_idx),
            )
        )
        if key not in first_delta:
            first_delta[key] = deltas[local_idx].copy()
            first_cost[key] = float(candidates.edit_cost[candidate_idx])

    inverse_index: list[list[tuple[int, int]]] = [[] for _ in range(int(num_queries))]
    edges: list[_AtomFlowEdge] = []
    for key, rows in grouped.items():
        rows.sort()
        best_by_row: dict[int, tuple[float, int, int]] = {}
        for neg_adv, candidate_idx, row_id, local_idx in rows:
            if row_id not in best_by_row:
                best_by_row[row_id] = (neg_adv, candidate_idx, local_idx)
        ordered = sorted(
            (neg_adv, candidate_idx, row_id, local_idx)
            for row_id, (neg_adv, candidate_idx, local_idx) in best_by_row.items()
        )
        candidate_indices = [int(candidate_idx) for _, candidate_idx, _, _ in ordered]
        row_ids = [int(row_id) for _, _, row_id, _ in ordered]
        delta = first_delta[key]
        qids = np.flatnonzero(delta).astype(np.int32)
        signs = delta[qids].astype(np.int8)
        if len(qids) == 0:
            continue
        edge_id = len(edges)
        edge = _AtomFlowEdge(
            source_atom=key[0],
            target_atom=key[1],
            candidate_indices=candidate_indices,
            row_ids=row_ids,
            qids=qids,
            signs=signs,
            cost=first_cost[key],
        )
        edges.append(edge)
        for qid, sign in zip(qids.tolist(), signs.tolist(), strict=True):
            inverse_index[int(qid)].append((edge_id, int(sign)))
    return edges, inverse_index


def _edge_marginal(edge: _AtomFlowEdge, current_weight: np.ndarray, inv_variance: np.ndarray, lambda_cost: float) -> float:
    signs = edge.signs.astype(np.float32)
    qids = edge.qids
    linear = float(signs @ current_weight[qids].astype(np.float32))
    quad = float(inv_variance[qids].astype(np.float32).sum())
    return linear - 0.5 * quad - float(lambda_cost) * float(edge.cost)


@jax.jit
def _choose_delta_prefix_jit(
    deltas: jax.Array,
    edit_cost: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    lambda_cost: jax.Array,
    strategy_code: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    d = deltas.astype(jnp.float32)
    delta_prefix = jnp.cumsum(d, axis=0)
    cost_prefix = jnp.cumsum(edit_cost.astype(jnp.float32))
    inv = inv_variance.astype(jnp.float32)
    weights = residual.astype(jnp.float32) * inv
    prefix_advantages = (
        delta_prefix @ weights
        - 0.5 * ((delta_prefix * delta_prefix) @ inv)
        - lambda_cost.astype(jnp.float32) * cost_prefix
    )
    best_idx = jnp.argmax(prefix_advantages)
    positive = prefix_advantages > 0.0
    last_positive_idx = jnp.max(jnp.where(positive, jnp.arange(prefix_advantages.shape[0], dtype=jnp.int32), -1))
    use_best = strategy_code == jnp.int32(1)
    chosen_idx = jnp.where(use_best, best_idx.astype(jnp.int32), last_positive_idx)
    chosen_adv = jnp.where(chosen_idx >= 0, prefix_advantages[chosen_idx], jnp.max(prefix_advantages))
    chosen_delta = jnp.where(
        chosen_idx >= 0,
        delta_prefix[chosen_idx],
        jnp.zeros((residual.shape[0],), dtype=jnp.float32),
    )
    accepted_count = jnp.maximum(chosen_idx + 1, 0)
    return accepted_count.astype(jnp.int32), chosen_delta.astype(jnp.float32), chosen_adv.astype(jnp.float32)


def _select_batch_atom_flow_units(
    candidates: CandidateBatch,
    pool_indices: np.ndarray,
    pool_advantages: np.ndarray,
    pool_quad: np.ndarray,
    max_accept: int,
    min_advantage: float,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[float, int, int, int]]] = {}
    for local_idx, candidate_idx in enumerate(pool_indices.tolist()):
        advantage = float(pool_advantages[local_idx])
        if not np.isfinite(advantage) or advantage <= float(min_advantage):
            continue
        source_atom = tuple(int(x) for x in candidates.old_rows[candidate_idx].tolist())
        target_atom = tuple(int(x) for x in candidates.new_rows[candidate_idx].tolist())
        if source_atom == target_atom:
            continue
        grouped.setdefault((source_atom, target_atom), []).append(
            (
                -advantage,
                int(local_idx),
                int(candidate_idx),
                int(candidates.row_ids[candidate_idx]),
            )
        )
    source_count = len({key[0] for key in grouped})
    target_count = len({key[1] for key in grouped})

    units: list[tuple[float, int, int, int]] = []
    for rows in grouped.values():
        rows.sort()
        best_by_row: dict[int, tuple[float, int, int]] = {}
        for neg_advantage, local_idx, candidate_idx, row_id in rows:
            if row_id not in best_by_row:
                best_by_row[row_id] = (neg_advantage, local_idx, candidate_idx)
        ordered = sorted(
            (neg_advantage, local_idx, candidate_idx, row_id)
            for row_id, (neg_advantage, local_idx, candidate_idx) in best_by_row.items()
        )
        if not ordered:
            continue
        first_local = ordered[0][1]
        edge_quad = float(pool_quad[first_local])
        if not np.isfinite(edge_quad) or edge_quad <= 0.0:
            continue
        base_marginal = -float(ordered[0][0])
        for flow_rank, (_, local_idx, candidate_idx, row_id) in enumerate(ordered):
            marginal = base_marginal - float(flow_rank) * edge_quad
            if marginal <= float(min_advantage):
                break
            units.append((-marginal, int(local_idx), int(candidate_idx), int(row_id)))

    if not units or max_accept <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), len(grouped), source_count, target_count

    units.sort()
    used_rows: set[int] = set()
    selected_local: list[int] = []
    selected_indices: list[int] = []
    for _, local_idx, candidate_idx, row_id in units:
        if row_id in used_rows:
            continue
        used_rows.add(row_id)
        selected_local.append(local_idx)
        selected_indices.append(candidate_idx)
        if len(selected_indices) >= max_accept:
            break
    return (
        np.asarray(selected_indices, dtype=np.int32),
        np.asarray(selected_local, dtype=np.int32),
        len(grouped),
        source_count,
        target_count,
    )


def choose_atom_flow_batch_transport(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    max_accept: int,
    min_advantage: float,
    pool_multiplier: int = 16,
    max_pool: int = 0,
    prefix_strategy: str = "best_advantage",
    delta_index: QueryDeltaIndex | None = None,
) -> TransportResult:
    """Choose an atom-flow batch with one residual update per accepted prefix.

    This is an approximate batched atom-flow selector: candidate deltas and
    initial edge marginals are computed on JAX, identical atom edges get a
    same-edge diminishing-return capacity, row capacity is enforced once, and
    the accepted prefix is checked with the exact QDTE batch objective before
    the caller updates residuals.
    """
    empty_diagnostics: dict[str, float | int] = {
        "atom_flow_pool_candidates": 0,
        "atom_flow_edges": 0,
        "atom_flow_source_atoms": 0,
        "atom_flow_target_atoms": 0,
        "atom_flow_augments": 0,
        "atom_flow_batch_mode": 1,
        "atom_flow_exact_mode": 0,
        "atom_flow_selected_candidates": 0,
        "atom_flow_prefix_candidates": 0,
        "atom_flow_sparse_delta": int(delta_index is not None),
    }
    empty = TransportResult(
        accepted_indices=np.empty(0, dtype=np.int32),
        delta_sum=np.zeros_like(residual, dtype=np.float32),
        batch_advantage=0.0,
        mean_advantage=0.0,
        diagnostics=empty_diagnostics,
    )
    if max_accept <= 0 or candidates.size == 0:
        return empty

    pool_indices = _candidate_pool(advantages, max_accept, min_advantage, pool_multiplier, max_pool)
    if len(pool_indices) == 0:
        return empty

    diagnostics: dict[str, float | int] = {
        "atom_flow_pool_candidates": int(len(pool_indices)),
        "atom_flow_edges": 0,
        "atom_flow_source_atoms": 0,
        "atom_flow_target_atoms": 0,
        "atom_flow_augments": 0,
        "atom_flow_batch_mode": 1,
        "atom_flow_exact_mode": 0,
        "atom_flow_selected_candidates": 0,
        "atom_flow_prefix_candidates": 0,
        "atom_flow_sparse_delta": int(delta_index is not None),
    }

    if delta_index is None:
        deltas_jax = _candidate_deltas_jax(candidates, pool_indices, qcat)
        delta_backend = 0
    else:
        deltas_jax = jnp.asarray(
            delta_index.dense_candidate_deltas(candidates.old_rows[pool_indices], candidates.new_rows[pool_indices]),
            dtype=jnp.int8,
        )
        delta_backend = 1
    pool_quad_jax = _delta_quad_jit(
        deltas_jax,
        jnp.asarray(inv_variance, dtype=jnp.float32),
    )
    pool_advantages = advantages[pool_indices].astype(np.float32, copy=False)
    pool_quad = np.asarray(pool_quad_jax, dtype=np.float32)
    selected_indices, selected_local, grouped_edges, source_count, target_count = _select_batch_atom_flow_units(
        candidates,
        pool_indices,
        pool_advantages,
        pool_quad,
        max_accept=max_accept,
        min_advantage=min_advantage,
    )
    diagnostics["atom_flow_edges"] = int(grouped_edges)
    diagnostics["atom_flow_source_atoms"] = int(source_count)
    diagnostics["atom_flow_target_atoms"] = int(target_count)
    diagnostics["atom_flow_selected_candidates"] = int(len(selected_indices))
    diagnostics["atom_flow_sparse_delta"] = int(delta_backend)
    if len(selected_indices) == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    strategy_code = 1 if prefix_strategy == "best_advantage" else 0
    accepted_count, delta_sum, batch_adv = _choose_delta_prefix_jit(
        deltas_jax[jnp.asarray(selected_local, dtype=jnp.int32)],
        jnp.asarray(candidates.edit_cost[selected_indices], dtype=jnp.float32),
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(inv_variance, dtype=jnp.float32),
        jnp.asarray(lambda_cost, dtype=jnp.float32),
        jnp.asarray(strategy_code, dtype=jnp.int32),
    )
    count = int(np.asarray(accepted_count))
    batch_adv_float = float(np.asarray(batch_adv))
    diagnostics["atom_flow_batch_advantage_check"] = batch_adv_float
    diagnostics["atom_flow_prefix_candidates"] = int(count)
    diagnostics["atom_flow_augments"] = int(count)
    if count <= 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=batch_adv_float,
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    idx = selected_indices[:count]
    return TransportResult(
        accepted_indices=idx.astype(np.int32),
        delta_sum=np.asarray(delta_sum, dtype=np.float32),
        batch_advantage=batch_adv_float,
        mean_advantage=float(advantages[idx].mean()),
        diagnostics=diagnostics,
    )


def choose_atom_flow_transport(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    max_accept: int,
    min_advantage: float,
    pool_multiplier: int = 16,
    max_pool: int = 0,
    delta_index: QueryDeltaIndex | None = None,
) -> TransportResult:
    """Choose edits with exact successive atom-flow marginal updates.

    Candidate edits are grouped as atom-flow edges from full encoded old-row
    atoms to full encoded new-row atoms. Each accepted unit updates the current
    weighted residual and all affected edge marginals before the next unit is
    selected. Row ids remain capacity-one, so a synthetic row cannot flow twice
    in the same transport batch.
    """
    empty = TransportResult(
        accepted_indices=np.empty(0, dtype=np.int32),
        delta_sum=np.zeros_like(residual, dtype=np.float32),
        batch_advantage=0.0,
        mean_advantage=0.0,
        diagnostics={
            "atom_flow_pool_candidates": 0,
            "atom_flow_edges": 0,
            "atom_flow_source_atoms": 0,
            "atom_flow_target_atoms": 0,
            "atom_flow_augments": 0,
            "atom_flow_batch_mode": 0,
            "atom_flow_exact_mode": 1,
            "atom_flow_selected_candidates": 0,
            "atom_flow_prefix_candidates": 0,
            "atom_flow_sparse_delta": int(delta_index is not None),
        },
    )
    if max_accept <= 0 or candidates.size == 0:
        return empty

    pool_indices = _candidate_pool(advantages, max_accept, min_advantage, pool_multiplier, max_pool)
    if len(pool_indices) == 0:
        return empty
    deltas = _candidate_deltas(candidates, pool_indices, qcat, delta_index=delta_index)
    edges, inverse_index = _build_atom_flow_edges(
        candidates,
        pool_indices,
        deltas,
        advantages,
        qcat.m,
    )
    diagnostics: dict[str, float | int] = {
        "atom_flow_pool_candidates": int(len(pool_indices)),
        "atom_flow_edges": int(len(edges)),
        "atom_flow_source_atoms": int(len({edge.source_atom for edge in edges})),
        "atom_flow_target_atoms": int(len({edge.target_atom for edge in edges})),
        "atom_flow_augments": 0,
        "atom_flow_batch_mode": 0,
        "atom_flow_exact_mode": 1,
        "atom_flow_selected_candidates": 0,
        "atom_flow_prefix_candidates": 0,
        "atom_flow_sparse_delta": int(delta_index is not None),
    }
    if not edges:
        empty.diagnostics = diagnostics
        return empty

    current_weight = residual.astype(np.float32) * inv_variance.astype(np.float32)
    heap: list[tuple[float, int, int]] = []
    used_rows: set[int] = set()
    for edge_id, edge in enumerate(edges):
        if edge.next_available_index(used_rows) is None:
            continue
        edge.marginal = _edge_marginal(edge, current_weight, inv_variance, lambda_cost)
        heapq.heappush(heap, (-edge.marginal, edge.version, edge_id))

    accepted: list[int] = []
    delta_sum = np.zeros_like(residual, dtype=np.float32)
    cost_sum = 0.0
    while heap and len(accepted) < max_accept:
        neg_margin, version, edge_id = heapq.heappop(heap)
        edge = edges[edge_id]
        if not edge.active or version != edge.version:
            continue
        candidate_idx = edge.next_available_index(used_rows)
        if candidate_idx is None:
            continue
        margin = -float(neg_margin)
        if margin <= float(min_advantage):
            break

        accepted.append(int(candidate_idx))
        used_rows.add(int(candidates.row_ids[candidate_idx]))
        edge.cursor += 1
        diagnostics["atom_flow_augments"] = int(diagnostics["atom_flow_augments"]) + 1
        diagnostics["atom_flow_selected_candidates"] = int(diagnostics["atom_flow_augments"])
        diagnostics["atom_flow_prefix_candidates"] = int(diagnostics["atom_flow_augments"])
        cost_sum += edge.cost

        qids = edge.qids
        signs = edge.signs.astype(np.float32)
        delta_sum[qids] += signs
        delta_weight = -signs * inv_variance[qids].astype(np.float32)
        current_weight[qids] += delta_weight
        edge_updates: dict[int, float] = {}
        for qid, d_weight in zip(qids.tolist(), delta_weight.tolist(), strict=True):
            for affected_edge_id, affected_sign in inverse_index[int(qid)]:
                edge_updates[affected_edge_id] = edge_updates.get(affected_edge_id, 0.0) + float(
                    affected_sign
                ) * float(d_weight)
        for affected_edge_id, marginal_update in edge_updates.items():
            affected = edges[affected_edge_id]
            if not affected.active:
                continue
            affected.marginal += marginal_update
            affected.version += 1
            if affected.next_available_index(used_rows) is not None:
                heapq.heappush(heap, (-affected.marginal, affected.version, affected_edge_id))

    if not accepted:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    idx = np.asarray(accepted, dtype=np.int32)
    batch_adv = batch_advantage(residual, inv_variance, delta_sum, cost_sum, lambda_cost)
    diagnostics["atom_flow_batch_advantage_check"] = float(batch_adv)
    if batch_adv <= 0.0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(batch_adv),
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )
    return TransportResult(
        accepted_indices=idx,
        delta_sum=delta_sum.astype(np.float32),
        batch_advantage=float(batch_adv),
        mean_advantage=float(advantages[idx].mean()),
        diagnostics=diagnostics,
    )


def _empty_constructive_pair_result(
    residual: np.ndarray,
    delta_index: QueryDeltaIndex | None,
    **extra: float | int,
) -> TransportResult:
    diagnostics: dict[str, float | int] = {
        "constructive_pair_mode": 1,
        "constructive_pair_pool_candidates": 0,
        "constructive_pair_seed_candidates": 0,
        "constructive_pair_pairs_evaluated": 0,
        "constructive_pair_positive_pairs": 0,
        "constructive_pair_explicit_pairs_evaluated": 0,
        "constructive_pair_explicit_positive_pairs": 0,
        "constructive_pair_units": 0,
        "constructive_pair_single_units": 0,
        "constructive_pair_pair_units": 0,
        "constructive_pair_explicit_pair_units": 0,
        "constructive_pair_selected_units": 0,
        "constructive_pair_prefix_units": 0,
        "constructive_pair_selected_candidates": 0,
        "constructive_pair_accepted_candidates": 0,
        "constructive_pair_sparse_delta": int(delta_index is not None),
    }
    diagnostics.update(extra)
    return TransportResult(
        accepted_indices=np.empty(0, dtype=np.int32),
        delta_sum=np.zeros_like(residual, dtype=np.float32),
        batch_advantage=0.0,
        mean_advantage=0.0,
        diagnostics=diagnostics,
    )


def _constructive_pair_advantage(
    delta: np.ndarray,
    cost: float,
    weights: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
) -> float:
    d = delta.astype(np.float32, copy=False)
    linear = float(d @ weights.astype(np.float32, copy=False))
    quad = float((d * d) @ inv_variance.astype(np.float32, copy=False))
    return linear - 0.5 * quad - float(lambda_cost) * float(cost)


def choose_constructive_pair_transport(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    max_accept: int,
    min_advantage: float,
    pool_multiplier: int = 16,
    max_pool: int = 0,
    partner_limit: int = 16,
    harm_query_limit: int = 16,
    max_units: int = 0,
    min_target_component: float = 0.0,
    prefix_strategy: str = "best_advantage",
    delta_index: QueryDeltaIndex | None = None,
) -> TransportResult:
    """Accept individual edits or constructed compensating edit pairs.

    The generator still proposes individual row edits. This transport layer
    computes their query-delta vectors, builds an inverted index keyed by
    affected query and delta sign, and constructs pairs where one edit's
    harmful residual-direction support is offset by an opposite-signed partner.
    Each pair is scored by the original QDTE batch objective on delta_a+delta_b.
    """
    if max_accept <= 0 or candidates.size == 0:
        return _empty_constructive_pair_result(residual, delta_index)

    finite = np.flatnonzero(np.isfinite(advantages))
    if len(finite) == 0:
        return _empty_constructive_pair_result(residual, delta_index)

    deltas_all = _candidate_deltas(candidates, finite, qcat, delta_index=delta_index).astype(np.int8, copy=False)
    finite_local_by_candidate = {int(candidate_idx): int(local_idx) for local_idx, candidate_idx in enumerate(finite.tolist())}
    weights = residual.astype(np.float32, copy=False) * inv_variance.astype(np.float32, copy=False)
    d_float = deltas_all.astype(np.float32, copy=False)
    full_linear = d_float @ weights

    target_component = np.full(len(finite), -np.inf, dtype=np.float32)
    valid_target = (candidates.target_query_ids[finite] >= 0) & (candidates.target_query_ids[finite] < qcat.m)
    valid_local = np.flatnonzero(valid_target)
    if len(valid_local) > 0:
        qids = candidates.target_query_ids[finite[valid_local]].astype(np.int32, copy=False)
        target_delta = d_float[valid_local, qids]
        target_component[valid_local] = (
            target_delta * weights[qids] - 0.5 * (target_delta * target_delta) * inv_variance[qids]
        ).astype(np.float32, copy=False)

    seed_score = (
        np.maximum(advantages[finite].astype(np.float32, copy=False), 0.0)
        + np.maximum(target_component, 0.0)
        + 0.05 * np.maximum(full_linear.astype(np.float32, copy=False), 0.0)
    )
    if max_pool > 0:
        pool_size = min(len(finite), int(max_pool))
    elif pool_multiplier > 0:
        pool_size = min(len(finite), max(int(max_accept), int(pool_multiplier) * int(max_accept)))
    else:
        pool_size = len(finite)
    if pool_size < len(finite):
        pool_local = np.argpartition(-seed_score, pool_size - 1)[:pool_size]
        pool_local = pool_local[np.argsort(-seed_score[pool_local])]
    else:
        pool_local = np.arange(len(finite), dtype=np.int32)

    pool_indices = finite[pool_local].astype(np.int32, copy=False)
    deltas = deltas_all[pool_local]
    pool_advantages = advantages[pool_indices].astype(np.float32, copy=False)
    pool_target_component = target_component[pool_local]
    pool_full_linear = full_linear[pool_local].astype(np.float32, copy=False)
    pool_size = int(len(pool_indices))
    if pool_size == 0:
        return _empty_constructive_pair_result(residual, delta_index)

    sign_index: dict[tuple[int, int], list[int]] = {}
    for local_idx in range(pool_size):
        qids = np.flatnonzero(deltas[local_idx])
        for qid in qids.tolist():
            sign = int(deltas[local_idx, int(qid)])
            sign_index.setdefault((int(qid), sign), []).append(local_idx)

    units: list[_ConstructiveUnit] = []
    for local_idx, candidate_idx in enumerate(pool_indices.tolist()):
        advantage = float(pool_advantages[local_idx])
        if advantage > float(min_advantage) and int(candidates.repair_type[candidate_idx]) != 16:
            units.append(
                _ConstructiveUnit(
                    advantage=advantage,
                    candidate_indices=(int(candidate_idx),),
                    local_indices=(int(local_idx),),
                    delta=deltas[local_idx].astype(np.float32, copy=True),
                    cost=float(candidates.edit_cost[candidate_idx]),
                    unit_type="single",
                )
            )

    seen_pairs: set[tuple[int, int]] = set()
    explicit_pairs_evaluated = 0
    explicit_positive_pairs = 0
    attached_pair_indices = getattr(candidates, "attached_pair_indices", None)
    if attached_pair_indices is not None:
        attached = np.asarray(attached_pair_indices, dtype=np.int32).reshape(-1, 2)
        for seed_idx_raw, partner_idx_raw in attached.tolist():
            seed_idx = int(seed_idx_raw)
            partner_idx = int(partner_idx_raw)
            if seed_idx == partner_idx:
                continue
            if seed_idx not in finite_local_by_candidate or partner_idx not in finite_local_by_candidate:
                continue
            if int(candidates.row_ids[seed_idx]) == int(candidates.row_ids[partner_idx]):
                continue
            pair_key = tuple(sorted((seed_idx, partner_idx)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            seed_local = finite_local_by_candidate[seed_idx]
            partner_local = finite_local_by_candidate[partner_idx]
            pair_delta = (
                deltas_all[seed_local].astype(np.float32, copy=False)
                + deltas_all[partner_local].astype(np.float32, copy=False)
            )
            pair_cost = float(candidates.edit_cost[seed_idx]) + float(candidates.edit_cost[partner_idx])
            pair_adv = _constructive_pair_advantage(pair_delta, pair_cost, weights, inv_variance, lambda_cost)
            explicit_pairs_evaluated += 1
            if pair_adv <= float(min_advantage):
                continue
            explicit_positive_pairs += 1
            units.append(
                _ConstructiveUnit(
                    advantage=float(pair_adv),
                    candidate_indices=(seed_idx, partner_idx),
                    local_indices=(seed_local, partner_local),
                    delta=pair_delta.astype(np.float32, copy=True),
                    cost=pair_cost,
                    unit_type="explicit_pair",
                )
            )

    contribution = deltas.astype(np.float32, copy=False) * weights.reshape(1, -1)
    seed_mask = (
        ((pool_target_component > float(min_target_component)) | (pool_full_linear > 0.0))
        & np.any(contribution < 0.0, axis=1)
        & (candidates.repair_type[pool_indices] != 16)
    )
    seed_locals = np.flatnonzero(seed_mask)
    pairs_evaluated = 0
    positive_pairs = 0

    for seed_local in seed_locals.tolist():
        seed_delta = deltas[int(seed_local)]
        harmed_qids = np.flatnonzero(contribution[int(seed_local)] < 0.0)
        if len(harmed_qids) == 0:
            continue
        harm_scores = np.abs(contribution[int(seed_local), harmed_qids])
        order = np.argsort(-harm_scores)
        if harm_query_limit > 0:
            order = order[: int(harm_query_limit)]
        partner_votes: dict[int, float] = {}
        for pos in order.tolist():
            qid = int(harmed_qids[int(pos)])
            sign = int(seed_delta[qid])
            if sign == 0:
                continue
            for partner_local in sign_index.get((qid, -sign), []):
                if int(partner_local) == int(seed_local):
                    continue
                partner_votes[int(partner_local)] = partner_votes.get(int(partner_local), 0.0) + float(
                    harm_scores[int(pos)]
                )
        if not partner_votes:
            continue
        partners = sorted(partner_votes.items(), key=lambda item: -item[1])
        if partner_limit > 0:
            partners = partners[: int(partner_limit)]
        for partner_local, _vote in partners:
            a, b = sorted((int(seed_local), int(partner_local)))
            idx_a = int(pool_indices[a])
            idx_b = int(pool_indices[b])
            pair_key = tuple(sorted((idx_a, idx_b)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if int(candidates.row_ids[idx_a]) == int(candidates.row_ids[idx_b]):
                continue
            pair_delta = deltas[a].astype(np.float32, copy=False) + deltas[b].astype(np.float32, copy=False)
            pair_cost = float(candidates.edit_cost[idx_a]) + float(candidates.edit_cost[idx_b])
            pair_adv = _constructive_pair_advantage(pair_delta, pair_cost, weights, inv_variance, lambda_cost)
            pairs_evaluated += 1
            if pair_adv <= float(min_advantage):
                continue
            positive_pairs += 1
            units.append(
                _ConstructiveUnit(
                    advantage=float(pair_adv),
                    candidate_indices=(idx_a, idx_b),
                    local_indices=(a, b),
                    delta=pair_delta.astype(np.float32, copy=True),
                    cost=pair_cost,
                    unit_type="pair",
                )
            )

    single_units = sum(1 for unit in units if unit.unit_type == "single")
    pair_units = sum(1 for unit in units if unit.unit_type in {"pair", "explicit_pair"})
    explicit_pair_units = sum(1 for unit in units if unit.unit_type == "explicit_pair")
    diagnostics: dict[str, float | int] = {
        "constructive_pair_mode": 1,
        "constructive_pair_pool_candidates": int(pool_size),
        "constructive_pair_seed_candidates": int(len(seed_locals)),
        "constructive_pair_pairs_evaluated": int(pairs_evaluated),
        "constructive_pair_positive_pairs": int(positive_pairs),
        "constructive_pair_explicit_pairs_evaluated": int(explicit_pairs_evaluated),
        "constructive_pair_explicit_positive_pairs": int(explicit_positive_pairs),
        "constructive_pair_units": int(len(units)),
        "constructive_pair_single_units": int(single_units),
        "constructive_pair_pair_units": int(pair_units),
        "constructive_pair_explicit_pair_units": int(explicit_pair_units),
        "constructive_pair_selected_units": 0,
        "constructive_pair_prefix_units": 0,
        "constructive_pair_selected_candidates": 0,
        "constructive_pair_accepted_candidates": 0,
        "constructive_pair_sparse_delta": int(delta_index is not None),
    }
    if not units:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    units.sort(key=lambda unit: -float(unit.advantage))
    if max_units > 0:
        units = units[: int(max_units)]
        diagnostics["constructive_pair_units"] = int(len(units))
        diagnostics["constructive_pair_single_units"] = int(sum(1 for unit in units if unit.unit_type == "single"))
        diagnostics["constructive_pair_pair_units"] = int(
            sum(1 for unit in units if unit.unit_type in {"pair", "explicit_pair"})
        )
        diagnostics["constructive_pair_explicit_pair_units"] = int(
            sum(1 for unit in units if unit.unit_type == "explicit_pair")
        )

    used_rows: set[int] = set()
    selected_units: list[_ConstructiveUnit] = []
    selected_candidate_count = 0
    for unit in units:
        if selected_candidate_count + len(unit.candidate_indices) > int(max_accept):
            continue
        rows = [int(candidates.row_ids[idx]) for idx in unit.candidate_indices]
        if len(set(rows)) != len(rows):
            continue
        if any(row in used_rows for row in rows):
            continue
        selected_units.append(unit)
        selected_candidate_count += len(unit.candidate_indices)
        used_rows.update(rows)
        if selected_candidate_count >= int(max_accept):
            break

    diagnostics["constructive_pair_selected_units"] = int(len(selected_units))
    diagnostics["constructive_pair_selected_candidates"] = int(selected_candidate_count)
    if not selected_units:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    delta_prefix = np.cumsum(np.asarray([unit.delta for unit in selected_units], dtype=np.float32), axis=0)
    cost_prefix = np.cumsum(np.asarray([unit.cost for unit in selected_units], dtype=np.float32))
    batch_advantages = (
        delta_prefix @ weights
        - 0.5 * ((delta_prefix * delta_prefix) @ inv_variance.astype(np.float32, copy=False))
        - float(lambda_cost) * cost_prefix
    )
    positive = np.flatnonzero(batch_advantages > 0.0)
    if prefix_strategy == "best_advantage":
        best_idx = int(np.argmax(batch_advantages))
        best_adv = float(batch_advantages[best_idx])
        prefix_units = best_idx + 1 if best_adv > 0.0 else 0
    elif len(positive) > 0:
        prefix_units = int(positive[-1]) + 1
        best_adv = float(batch_advantages[prefix_units - 1])
    else:
        best_idx = int(np.argmax(batch_advantages))
        prefix_units = 0
        best_adv = float(batch_advantages[best_idx])
    diagnostics["constructive_pair_batch_advantage_check"] = float(best_adv)
    diagnostics["constructive_pair_prefix_units"] = int(prefix_units)
    if prefix_units <= 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(best_adv if np.isfinite(best_adv) else 0.0),
            mean_advantage=0.0,
            diagnostics=diagnostics,
        )

    accepted: list[int] = []
    for unit in selected_units[:prefix_units]:
        accepted.extend(int(idx) for idx in unit.candidate_indices)
    accepted_indices = np.asarray(accepted, dtype=np.int32)
    diagnostics["constructive_pair_accepted_candidates"] = int(len(accepted_indices))
    return TransportResult(
        accepted_indices=accepted_indices,
        delta_sum=delta_prefix[prefix_units - 1].astype(np.float32),
        batch_advantage=float(best_adv),
        mean_advantage=float(np.mean(advantages[accepted_indices])) if len(accepted_indices) else 0.0,
        diagnostics=diagnostics,
    )


def choose_transport_batch(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    deltas: np.ndarray,
    selected: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    prefix_strategy: str = "largest_positive",
) -> TransportResult:
    if len(selected) == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
        )
    order = np.argsort(-advantages[selected])
    selected = selected[order]
    deltas = deltas[order]
    full_delta_sum = deltas.astype(np.float32).sum(axis=0)
    full_cost_sum = float(candidates.edit_cost[selected].sum())
    full_adv = batch_advantage(residual, inv_variance, full_delta_sum, full_cost_sum, lambda_cost)
    if full_adv > 0.0 and prefix_strategy != "best_advantage":
        best_prefix = len(selected)
        best_adv = float(full_adv)
        delta_sum = full_delta_sum
    else:
        delta_prefix = np.cumsum(deltas.astype(np.float32), axis=0)
        cost_prefix = np.cumsum(candidates.edit_cost[selected].astype(np.float32))
        weights = residual.astype(np.float32) * inv_variance.astype(np.float32)
        batch_advantages = (
            delta_prefix @ weights
            - 0.5 * ((delta_prefix * delta_prefix) @ inv_variance.astype(np.float32))
            - float(lambda_cost) * cost_prefix
        )
        positive = np.flatnonzero(batch_advantages > 0.0)
        if prefix_strategy == "best_advantage":
            best_idx = int(np.argmax(batch_advantages))
            best_adv = float(batch_advantages[best_idx])
            best_prefix = best_idx + 1 if best_adv > 0.0 else 0
            delta_sum = delta_prefix[best_idx] if best_prefix > 0 else np.zeros_like(residual, dtype=np.float32)
        elif len(positive) > 0:
            best_prefix = int(positive[-1]) + 1
            best_adv = float(batch_advantages[best_prefix - 1])
            delta_sum = delta_prefix[best_prefix - 1]
        else:
            best_idx = int(np.argmax(batch_advantages))
            best_prefix = 0
            best_adv = float(batch_advantages[best_idx])
            delta_sum = np.zeros_like(residual, dtype=np.float32)
    if best_prefix == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(best_adv if np.isfinite(best_adv) else 0.0),
            mean_advantage=0.0,
        )
    idx = selected[:best_prefix]
    return TransportResult(
        accepted_indices=idx.astype(np.int32),
        delta_sum=delta_sum.astype(np.float32),
        batch_advantage=float(best_adv),
        mean_advantage=float(advantages[idx].mean()),
    )


@jax.jit
def _choose_transport_prefix_jit(
    old_rows: jax.Array,
    new_rows: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    edit_cost: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
    lambda_cost: jax.Array,
    strategy_code: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    phi_old = eval_records_queries_arrays(
        old_rows,
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    )
    phi_new = eval_records_queries_arrays(
        new_rows,
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    )
    delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    delta_prefix = jnp.cumsum(delta, axis=0)
    cost_prefix = jnp.cumsum(edit_cost.astype(jnp.float32))
    weights = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    prefix_advantages = (
        delta_prefix @ weights
        - 0.5 * ((delta_prefix * delta_prefix) @ inv_variance.astype(jnp.float32))
        - lambda_cost.astype(jnp.float32) * cost_prefix
    )
    best_idx = jnp.argmax(prefix_advantages)
    positive = prefix_advantages > 0.0
    last_positive_idx = jnp.max(jnp.where(positive, jnp.arange(prefix_advantages.shape[0], dtype=jnp.int32), -1))
    use_best = strategy_code == jnp.int32(1)
    chosen_idx = jnp.where(use_best, best_idx.astype(jnp.int32), last_positive_idx)
    chosen_adv = jnp.where(chosen_idx >= 0, prefix_advantages[chosen_idx], jnp.max(prefix_advantages))
    chosen_delta = jnp.where(
        chosen_idx >= 0,
        delta_prefix[chosen_idx],
        jnp.zeros((residual.shape[0],), dtype=jnp.float32),
    )
    accepted_count = jnp.maximum(chosen_idx + 1, 0)
    return accepted_count.astype(jnp.int32), chosen_delta.astype(jnp.float32), chosen_adv.astype(jnp.float32)


def choose_transport_batch_jax(
    candidates: CandidateBatch,
    advantages: np.ndarray,
    selected: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    lambda_cost: float,
    qcat: QueryCatalogue,
    prefix_strategy: str = "largest_positive",
) -> TransportResult:
    if len(selected) == 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=0.0,
            mean_advantage=0.0,
        )
    order = np.argsort(-advantages[selected])
    selected = selected[order]
    strategy_code = 1 if prefix_strategy == "best_advantage" else 0
    accepted_count, delta_sum, batch_adv = _choose_transport_prefix_jit(
        jnp.asarray(candidates.old_rows[selected], dtype=jnp.int32),
        jnp.asarray(candidates.new_rows[selected], dtype=jnp.int32),
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(inv_variance, dtype=jnp.float32),
        jnp.asarray(candidates.edit_cost[selected], dtype=jnp.float32),
        jnp.asarray(qcat.attrs, dtype=jnp.int32),
        jnp.asarray(qcat.ops, dtype=jnp.int32),
        jnp.asarray(qcat.values, dtype=jnp.int32),
        jnp.asarray(qcat.lows, dtype=jnp.int32),
        jnp.asarray(qcat.highs, dtype=jnp.int32),
        jnp.asarray(qcat.linear_attrs, dtype=jnp.int32),
        jnp.asarray(qcat.linear_weights, dtype=jnp.float32),
        jnp.asarray(qcat.linear_thresholds, dtype=jnp.float32),
        jnp.asarray(qcat.linear_num_terms, dtype=jnp.int32),
        jnp.asarray(lambda_cost, dtype=jnp.float32),
        jnp.asarray(strategy_code, dtype=jnp.int32),
    )
    count = int(np.asarray(accepted_count))
    if count <= 0:
        return TransportResult(
            accepted_indices=np.empty(0, dtype=np.int32),
            delta_sum=np.zeros_like(residual, dtype=np.float32),
            batch_advantage=float(np.asarray(batch_adv)),
            mean_advantage=0.0,
        )
    idx = selected[:count]
    return TransportResult(
        accepted_indices=idx.astype(np.int32),
        delta_sum=np.asarray(delta_sum, dtype=np.float32),
        batch_advantage=float(np.asarray(batch_adv)),
        mean_advantage=float(advantages[idx].mean()),
    )


def apply_edits(X_syn: np.ndarray, candidates: CandidateBatch, accepted_indices: np.ndarray) -> None:
    if len(accepted_indices) == 0:
        return
    X_syn[candidates.row_ids[accepted_indices]] = candidates.new_rows[accepted_indices]
