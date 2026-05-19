from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue
from qdte.schema import TableSchema


@dataclass
class CandidateBatch:
    row_ids: np.ndarray
    old_rows: np.ndarray
    new_rows: np.ndarray
    target_query_ids: np.ndarray
    edit_cost: np.ndarray
    repair_type: np.ndarray
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.row_ids.shape[0])


def _clip_record(row: np.ndarray, cardinalities: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(row, 0), cardinalities - 1).astype(np.int32)


def repair_enter(row: np.ndarray, qcat: QueryCatalogue, qid: int, cardinalities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new = row.copy()
    for attr, op, value, lo, hi in qcat.query_terms(qid):
        if op == OP_EQ:
            new[attr] = value
        elif op == OP_LE:
            new[attr] = int(rng.integers(0, max(1, min(value + 1, cardinalities[attr]))))
        elif op == OP_GE:
            new[attr] = int(rng.integers(max(0, value), cardinalities[attr]))
        elif op == OP_RANGE:
            upper = min(hi + 1, cardinalities[attr])
            new[attr] = int(rng.integers(lo, max(lo + 1, upper)))
        else:
            raise ValueError(f"Unknown op {op}")
    return _clip_record(new, cardinalities)


def repair_exit(row: np.ndarray, qcat: QueryCatalogue, qid: int, cardinalities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    terms = qcat.query_terms(qid)
    rng.shuffle(terms)
    for attr, op, value, lo, hi in terms:
        new = row.copy()
        card = int(cardinalities[attr])
        if card <= 1:
            continue
        if op == OP_EQ:
            choices = [v for v in range(card) if v != value]
            if choices:
                new[attr] = int(rng.choice(choices))
                return _clip_record(new, cardinalities)
        elif op == OP_LE:
            if value + 1 < card:
                new[attr] = int(rng.integers(value + 1, card))
                return _clip_record(new, cardinalities)
        elif op == OP_GE:
            if value > 0:
                new[attr] = int(rng.integers(0, value))
                return _clip_record(new, cardinalities)
        elif op == OP_RANGE:
            below = list(range(0, max(0, lo)))
            above = list(range(min(card, hi + 1), card))
            choices = below + above
            if choices:
                new[attr] = int(rng.choice(choices))
                return _clip_record(new, cardinalities)
    return random_mutation(row, cardinalities, rng)


def random_mutation(row: np.ndarray, cardinalities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new = row.copy()
    mutable = np.flatnonzero(cardinalities > 1)
    if len(mutable) == 0:
        return new
    attr = int(rng.choice(mutable))
    old = int(new[attr])
    card = int(cardinalities[attr])
    val = int(rng.integers(0, card - 1))
    if val >= old:
        val += 1
    new[attr] = val
    return _clip_record(new, cardinalities)


def _random_mutation_batch(rows: np.ndarray, cardinalities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new = rows.copy()
    if len(new) == 0:
        return new
    mutable = np.flatnonzero(cardinalities > 1)
    if len(mutable) == 0:
        return new
    attrs = rng.choice(mutable, size=len(new))
    for attr in np.unique(attrs):
        mask = attrs == attr
        card = int(cardinalities[int(attr)])
        old = new[mask, int(attr)].astype(np.int32)
        vals = rng.integers(0, card - 1, size=int(mask.sum()), dtype=np.int32)
        vals = vals + (vals >= old)
        new[mask, int(attr)] = vals
    return _clip_record(new, cardinalities)


def _repair_enter_batch(
    rows: np.ndarray,
    qcat: QueryCatalogue,
    qid: int,
    cardinalities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    new = rows.copy()
    if len(new) == 0:
        return new
    for attr, op, value, lo, hi in qcat.query_terms(qid):
        card = int(cardinalities[attr])
        if op == OP_EQ:
            new[:, attr] = value
        elif op == OP_LE:
            upper = max(1, min(value + 1, card))
            new[:, attr] = rng.integers(0, upper, size=len(new), dtype=np.int32)
        elif op == OP_GE:
            lower = max(0, min(value, card - 1))
            new[:, attr] = rng.integers(lower, card, size=len(new), dtype=np.int32)
        elif op == OP_RANGE:
            lower = max(0, min(lo, card - 1))
            upper = max(lower + 1, min(hi + 1, card))
            new[:, attr] = rng.integers(lower, upper, size=len(new), dtype=np.int32)
        else:
            raise ValueError(f"Unknown op {op}")
    return _clip_record(new, cardinalities)


def _repair_exit_batch(
    rows: np.ndarray,
    qcat: QueryCatalogue,
    qid: int,
    cardinalities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    new = rows.copy()
    if len(new) == 0:
        return new
    breakable: list[tuple[int, int, int, int, int]] = []
    for attr, op, value, lo, hi in qcat.query_terms(qid):
        card = int(cardinalities[attr])
        if card <= 1:
            continue
        if op == OP_EQ:
            breakable.append((attr, op, value, lo, hi))
        elif op == OP_LE and value + 1 < card:
            breakable.append((attr, op, value, lo, hi))
        elif op == OP_GE and value > 0:
            breakable.append((attr, op, value, lo, hi))
        elif op == OP_RANGE and (lo > 0 or hi + 1 < card):
            breakable.append((attr, op, value, lo, hi))
    if not breakable:
        return _random_mutation_batch(rows, cardinalities, rng)

    chosen = rng.integers(0, len(breakable), size=len(new), dtype=np.int32)
    for term_idx, (attr, op, value, lo, hi) in enumerate(breakable):
        mask = chosen == term_idx
        n = int(mask.sum())
        if n == 0:
            continue
        card = int(cardinalities[attr])
        if op == OP_EQ:
            vals = rng.integers(0, card - 1, size=n, dtype=np.int32)
            vals = vals + (vals >= int(value))
            new[mask, attr] = vals
        elif op == OP_LE:
            new[mask, attr] = rng.integers(value + 1, card, size=n, dtype=np.int32)
        elif op == OP_GE:
            new[mask, attr] = rng.integers(0, value, size=n, dtype=np.int32)
        elif op == OP_RANGE:
            below_count = max(0, int(lo))
            above_start = min(card, int(hi) + 1)
            above_count = max(0, card - above_start)
            vals = np.empty(n, dtype=np.int32)
            if below_count > 0 and above_count > 0:
                use_below = rng.random(n) < (below_count / float(below_count + above_count))
                vals[use_below] = rng.integers(0, lo, size=int(use_below.sum()), dtype=np.int32)
                vals[~use_below] = rng.integers(above_start, card, size=int((~use_below).sum()), dtype=np.int32)
            elif below_count > 0:
                vals[:] = rng.integers(0, lo, size=n, dtype=np.int32)
            else:
                vals[:] = rng.integers(above_start, card, size=n, dtype=np.int32)
            new[mask, attr] = vals
        else:
            raise ValueError(f"Unknown op {op}")
    return _clip_record(new, cardinalities)


def compute_edit_cost(old_rows: np.ndarray, new_rows: np.ndarray, schema: TableSchema, numerical_gamma: float) -> np.ndarray:
    changed = old_rows != new_rows
    hamming = changed.sum(axis=1).astype(np.float32)
    num_dist = np.zeros(old_rows.shape[0], dtype=np.float32)
    card = schema.cardinalities
    for attr in schema.numerical_indices:
        denom = max(1, int(card[attr]) - 1)
        num_dist += np.abs(old_rows[:, attr] - new_rows[:, attr]).astype(np.float32) / float(denom)
    return hamming + float(numerical_gamma) * num_dist


def generate_candidates(
    X_syn: np.ndarray,
    qcat: QueryCatalogue,
    schema: TableSchema,
    target_query_ids: np.ndarray,
    residual: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> CandidateBatch:
    cfg = config.get("qdte", {})
    candidates_per_target = int(cfg.get("candidates_per_target", 64))
    total_candidates = int(cfg.get("total_candidates_per_iter", max(1, len(target_query_ids) * candidates_per_target)))
    random_fraction = float(cfg.get("random_candidate_fraction", 0.05))
    over_sample = int(cfg.get("source_over_sample_factor", 8))
    numerical_gamma = float(cfg.get("numerical_distance_gamma", 0.1))
    cardinalities = schema.cardinalities
    random_target = int(round(total_candidates * min(1.0, max(0.0, random_fraction))))
    directed_budget = max(0, total_candidates - random_target)

    row_id_buf = np.empty(total_candidates, dtype=np.int32)
    old_buf = np.empty((total_candidates, schema.d), dtype=np.int32)
    new_buf = np.empty((total_candidates, schema.d), dtype=np.int32)
    target_buf = np.empty(total_candidates, dtype=np.int32)
    repair_buf = np.empty(total_candidates, dtype=np.int32)
    source_filter_attempts = 0
    source_filter_failures = 0
    source_filter_kept = 0
    produced = 0

    def produced_count() -> int:
        return produced

    def append_chunk(row_ids: np.ndarray, old_rows: np.ndarray, new_rows: np.ndarray, target_qid: int, repair_type: int) -> None:
        nonlocal produced
        if len(row_ids) == 0 or produced >= total_candidates:
            return
        changed = np.any(old_rows != new_rows, axis=1)
        if not np.any(changed):
            return
        keep = np.flatnonzero(changed)
        n = min(int(len(keep)), total_candidates - produced)
        if n <= 0:
            return
        keep = keep[:n]
        end = produced + n
        row_id_buf[produced:end] = row_ids[keep].astype(np.int32, copy=False)
        old_buf[produced:end] = old_rows[keep].astype(np.int32, copy=False)
        new_buf[produced:end] = new_rows[keep].astype(np.int32, copy=False)
        target_buf[produced:end] = int(target_qid)
        repair_buf[produced:end] = int(repair_type)
        produced = end

    def add_random_candidates(count: int) -> None:
        if count <= 0:
            return
        ids = rng.integers(0, X_syn.shape[0], size=count, dtype=np.int32)
        old = X_syn[ids].copy()
        new = _random_mutation_batch(old, cardinalities, rng)
        append_chunk(ids, old, new, -1, 0)

    def build_batch() -> CandidateBatch:
        if produced <= 0:
            empty_rows = np.empty((0, schema.d), dtype=np.int32)
            return CandidateBatch(
                row_ids=np.empty(0, dtype=np.int32),
                old_rows=empty_rows,
                new_rows=empty_rows,
                target_query_ids=np.empty(0, dtype=np.int32),
                edit_cost=np.empty(0, dtype=np.float32),
                repair_type=np.empty(0, dtype=np.int32),
                diagnostics={
                    "requested_candidates": float(total_candidates),
                    "produced_candidates": 0.0,
                    "directed_candidates": 0.0,
                    "random_candidates": 0.0,
                    "candidate_shortfall": float(total_candidates),
                    "source_filter_attempts": float(source_filter_attempts),
                    "source_filter_failures": float(source_filter_failures),
                    "source_filter_kept": float(source_filter_kept),
                },
            )
        row_id_arr = row_id_buf[:produced]
        old_arr = old_buf[:produced]
        new_arr = new_buf[:produced]
        target_arr = target_buf[:produced]
        repair_arr = repair_buf[:produced]
        cost = compute_edit_cost(old_arr, new_arr, schema, numerical_gamma)
        produced_float = float(len(row_id_arr))
        random_produced = float(np.sum(target_arr < 0))
        return CandidateBatch(
            row_ids=row_id_arr,
            old_rows=old_arr,
            new_rows=new_arr,
            target_query_ids=target_arr,
            edit_cost=cost.astype(np.float32),
            repair_type=repair_arr,
            diagnostics={
                "requested_candidates": float(total_candidates),
                "produced_candidates": produced_float,
                "directed_candidates": produced_float - random_produced,
                "random_candidates": random_produced,
                "candidate_shortfall": float(max(0, total_candidates - len(row_id_arr))),
                "source_filter_attempts": float(source_filter_attempts),
                "source_filter_failures": float(source_filter_failures),
                "source_filter_kept": float(source_filter_kept),
            },
        )

    random_only = random_fraction >= 1.0 or (len(target_query_ids) == 0 and int(cfg.get("num_active_targets", 1)) == 0)
    if random_only:
        add_random_candidates(total_candidates)
        return build_batch()

    if len(target_query_ids) == 0:
        target_query_ids = np.arange(qcat.m, dtype=np.int32)

    for qid_raw in target_query_ids.tolist():
        if produced_count() >= directed_budget:
            break
        qid = int(qid_raw)
        need_enter = residual[qid] > 0
        need_source_sat = not need_enter
        picked = 0
        attempts = 0
        while picked < candidates_per_target and attempts < 8 and produced_count() < directed_budget:
            attempts += 1
            source_filter_attempts += 1
            remaining_for_query = candidates_per_target - picked
            remaining_total = directed_budget - produced_count()
            sample_count = max(remaining_for_query * over_sample, candidates_per_target)
            ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
            rows = X_syn[ids]
            sat = qcat.eval_query_np(rows, qid)
            keep = np.flatnonzero(sat == need_source_sat)
            source_filter_kept += int(len(keep))
            if len(keep) == 0:
                source_filter_failures += 1
                continue
            rng.shuffle(keep)
            take = min(remaining_for_query, remaining_total, len(keep))
            selected_ids = ids[keep[:take]].astype(np.int32, copy=False)
            old = X_syn[selected_ids].copy()
            if need_enter:
                new = _repair_enter_batch(old, qcat, qid, cardinalities, rng)
                rtype = 1
            else:
                new = _repair_exit_batch(old, qcat, qid, cardinalities, rng)
                rtype = 2
            before = produced_count()
            append_chunk(selected_ids, old, new, qid, rtype)
            picked += produced_count() - before

    add_random_candidates(min(random_target, max(0, total_candidates - produced_count())))

    while produced_count() < total_candidates:
        before = produced_count()
        add_random_candidates(total_candidates - before)
        if produced_count() == before:
            break

    return build_batch()
