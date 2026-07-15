from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.eval_jax import eval_records_queries_arrays
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
    attached_pair_indices: np.ndarray | None = None

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
    if int(qcat.linear_num_terms[qid]) > 0:
        new = _repair_halfspace_batch(new.reshape(1, -1), qcat, qid, cardinalities, need_enter=True, rng=rng)[0]
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
                if int(qcat.linear_num_terms[qid]) > 0:
                    new = _repair_halfspace_batch(
                        new.reshape(1, -1), qcat, qid, cardinalities, need_enter=False, rng=rng
                    )[0]
                return _clip_record(new, cardinalities)
        elif op == OP_LE:
            if value + 1 < card:
                new[attr] = int(rng.integers(value + 1, card))
                if int(qcat.linear_num_terms[qid]) > 0:
                    new = _repair_halfspace_batch(
                        new.reshape(1, -1), qcat, qid, cardinalities, need_enter=False, rng=rng
                    )[0]
                return _clip_record(new, cardinalities)
        elif op == OP_GE:
            if value > 0:
                new[attr] = int(rng.integers(0, value))
                if int(qcat.linear_num_terms[qid]) > 0:
                    new = _repair_halfspace_batch(
                        new.reshape(1, -1), qcat, qid, cardinalities, need_enter=False, rng=rng
                    )[0]
                return _clip_record(new, cardinalities)
        elif op == OP_RANGE:
            below = list(range(0, max(0, lo)))
            above = list(range(min(card, hi + 1), card))
            choices = below + above
            if choices:
                new[attr] = int(rng.choice(choices))
                if int(qcat.linear_num_terms[qid]) > 0:
                    new = _repair_halfspace_batch(
                        new.reshape(1, -1), qcat, qid, cardinalities, need_enter=False, rng=rng
                    )[0]
                return _clip_record(new, cardinalities)
    if int(qcat.linear_num_terms[qid]) > 0:
        return _clip_record(
            _repair_halfspace_batch(row.reshape(1, -1), qcat, qid, cardinalities, need_enter=False, rng=rng)[0],
            cardinalities,
        )
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


def _halfspace_score(rows: np.ndarray, qcat: QueryCatalogue, qid: int) -> np.ndarray:
    score = np.zeros(len(rows), dtype=np.float32)
    for attr, weight in qcat.linear_terms(qid):
        score += float(weight) * rows[:, int(attr)].astype(np.float32)
    return score


def _repair_halfspace_batch(
    rows: np.ndarray,
    qcat: QueryCatalogue,
    qid: int,
    cardinalities: np.ndarray,
    need_enter: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    new = rows.copy()
    if len(new) == 0 or int(qcat.linear_num_terms[qid]) == 0:
        return new
    terms = qcat.linear_terms(qid)
    threshold = float(qcat.linear_thresholds[qid])
    score = _halfspace_score(new, qcat, qid)
    needs_fix = score > threshold if need_enter else score <= threshold
    for row_idx in np.flatnonzero(needs_fix).tolist():
        for _ in range(len(terms)):
            if bool(qcat.eval_query_np(new[row_idx : row_idx + 1], qid)[0]) == bool(need_enter):
                break
            cur_score = float(_halfspace_score(new[row_idx : row_idx + 1], qcat, qid)[0])
            best_attr = None
            best_value = None
            best_gain = 0.0
            for attr, weight in terms:
                attr = int(attr)
                weight = float(weight)
                card = int(cardinalities[attr])
                if card <= 1:
                    continue
                current = int(new[row_idx, attr])
                if need_enter:
                    target = 0 if weight > 0.0 else card - 1
                    gain = cur_score - (cur_score + weight * (target - current))
                else:
                    target = card - 1 if weight > 0.0 else 0
                    gain = (cur_score + weight * (target - current)) - cur_score
                if gain > best_gain and target != current:
                    best_attr = attr
                    best_value = target
                    best_gain = float(gain)
            if best_attr is None:
                break
            new[row_idx, best_attr] = int(best_value)
        if not need_enter and qcat.eval_query_np(new[row_idx : row_idx + 1], qid)[0]:
            new[row_idx] = random_mutation(new[row_idx], cardinalities, rng)
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
    if int(qcat.linear_num_terms[qid]) > 0:
        new = _repair_halfspace_batch(new, qcat, qid, cardinalities, need_enter=True, rng=rng)
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
        if int(qcat.linear_num_terms[qid]) > 0:
            return _repair_halfspace_batch(new, qcat, qid, cardinalities, need_enter=False, rng=rng)
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


def _eval_terms_np(rows: np.ndarray, terms: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    sat = np.ones(rows.shape[0], dtype=bool)
    for attr, op, value, lo, hi in terms:
        x = rows[:, int(attr)]
        if op == OP_EQ:
            cond = x == int(value)
        elif op == OP_LE:
            cond = x <= int(value)
        elif op == OP_GE:
            cond = x >= int(value)
        elif op == OP_RANGE:
            cond = (x >= int(lo)) & (x <= int(hi))
        else:
            raise ValueError(f"Unknown op {op}")
        sat &= cond
    return sat


def _dense_candidate_deltas_cpu(old_rows: np.ndarray, new_rows: np.ndarray, qcat: QueryCatalogue) -> np.ndarray:
    old_sat = np.empty((len(old_rows), qcat.m), dtype=bool)
    new_sat = np.empty((len(new_rows), qcat.m), dtype=bool)
    for qid in range(qcat.m):
        old_sat[:, qid] = qcat.eval_query_np(old_rows, qid)
        new_sat[:, qid] = qcat.eval_query_np(new_rows, qid)
    return new_sat.astype(np.int8) - old_sat.astype(np.int8)


def _dense_candidate_deltas_cpu_unique(old_rows: np.ndarray, new_rows: np.ndarray, qcat: QueryCatalogue) -> np.ndarray:
    old = np.asarray(old_rows, dtype=np.int32)
    new = np.asarray(new_rows, dtype=np.int32)
    if old.shape != new.shape:
        raise ValueError(f"old_rows and new_rows must have the same shape, got {old.shape} and {new.shape}")
    if len(old) == 0:
        return np.empty((0, qcat.m), dtype=np.int8)
    pairs = np.concatenate([old, new], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    if len(unique_pairs) == len(pairs):
        return _dense_candidate_deltas_cpu(old, new, qcat)
    width = old.shape[1]
    unique_delta = _dense_candidate_deltas_cpu(unique_pairs[:, :width], unique_pairs[:, width:], qcat)
    return unique_delta[inverse]


def _dense_candidate_deltas_jax(old_rows: np.ndarray, new_rows: np.ndarray, qcat: QueryCatalogue) -> np.ndarray:
    if len(old_rows) == 0:
        return np.empty((0, qcat.m), dtype=np.int8)
    qarrays = qcat.eval_arrays()
    phi_old = eval_records_queries_arrays(
        jnp.asarray(old_rows, dtype=jnp.int32),
        *(
            jnp.asarray(arr, dtype=jnp.float32)
            if idx in {6, 7}
            else jnp.asarray(arr, dtype=jnp.int32)
            for idx, arr in enumerate(qarrays)
        ),
    )
    phi_new = eval_records_queries_arrays(
        jnp.asarray(new_rows, dtype=jnp.int32),
        *(
            jnp.asarray(arr, dtype=jnp.float32)
            if idx in {6, 7}
            else jnp.asarray(arr, dtype=jnp.int32)
            for idx, arr in enumerate(qarrays)
        ),
    )
    return np.asarray(phi_new.astype(jnp.int8) - phi_old.astype(jnp.int8), dtype=np.int8)


def _dense_candidate_deltas_jax_fixed_batch(
    old_rows: np.ndarray,
    new_rows: np.ndarray,
    qcat: QueryCatalogue,
    batch_size: int,
) -> np.ndarray:
    old = np.asarray(old_rows, dtype=np.int32)
    new = np.asarray(new_rows, dtype=np.int32)
    if old.shape != new.shape:
        raise ValueError(f"old_rows and new_rows must have the same shape, got {old.shape} and {new.shape}")
    if len(old) == 0:
        return np.empty((0, qcat.m), dtype=np.int8)
    fixed = max(1, int(batch_size))
    out = np.empty((len(old), qcat.m), dtype=np.int8)
    for start in range(0, len(old), fixed):
        stop = min(len(old), start + fixed)
        count = stop - start
        chunk_old = old[start:stop]
        chunk_new = new[start:stop]
        if count == fixed:
            out[start:stop] = _dense_candidate_deltas_jax(chunk_old, chunk_new, qcat)
            continue
        padded_old = np.empty((fixed, old.shape[1]), dtype=np.int32)
        padded_new = np.empty((fixed, new.shape[1]), dtype=np.int32)
        padded_old[:count] = chunk_old
        padded_new[:count] = chunk_new
        padded_old[count:] = chunk_old[0]
        padded_new[count:] = chunk_new[0]
        out[start:stop] = _dense_candidate_deltas_jax(padded_old, padded_new, qcat)[:count]
    return out


def _repair_enter_terms_batch(
    rows: np.ndarray,
    terms: list[tuple[int, int, int, int, int]],
    cardinalities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    new = rows.copy()
    if len(new) == 0:
        return new
    for attr, op, value, lo, hi in terms:
        attr = int(attr)
        card = int(cardinalities[attr])
        if op == OP_EQ:
            new[:, attr] = int(value)
        elif op == OP_LE:
            upper = max(1, min(int(value) + 1, card))
            new[:, attr] = rng.integers(0, upper, size=len(new), dtype=np.int32)
        elif op == OP_GE:
            lower = max(0, min(int(value), card - 1))
            new[:, attr] = rng.integers(lower, card, size=len(new), dtype=np.int32)
        elif op == OP_RANGE:
            lower = max(0, min(int(lo), card - 1))
            upper = max(lower + 1, min(int(hi) + 1, card))
            new[:, attr] = rng.integers(lower, upper, size=len(new), dtype=np.int32)
        else:
            raise ValueError(f"Unknown op {op}")
    return _clip_record(new, cardinalities)


def _repair_exit_terms_batch(
    rows: np.ndarray,
    terms: list[tuple[int, int, int, int, int]],
    cardinalities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    new = rows.copy()
    if len(new) == 0:
        return new
    breakable: list[tuple[int, int, int, int, int]] = []
    for attr, op, value, lo, hi in terms:
        card = int(cardinalities[int(attr)])
        if card <= 1:
            continue
        if op == OP_EQ:
            breakable.append((attr, op, value, lo, hi))
        elif op == OP_LE and int(value) + 1 < card:
            breakable.append((attr, op, value, lo, hi))
        elif op == OP_GE and int(value) > 0:
            breakable.append((attr, op, value, lo, hi))
        elif op == OP_RANGE and (int(lo) > 0 or int(hi) + 1 < card):
            breakable.append((attr, op, value, lo, hi))
    if not breakable:
        return _random_mutation_batch(rows, cardinalities, rng)

    chosen = rng.integers(0, len(breakable), size=len(new), dtype=np.int32)
    for term_idx, (attr, op, value, lo, hi) in enumerate(breakable):
        attr = int(attr)
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
            new[mask, attr] = rng.integers(int(value) + 1, card, size=n, dtype=np.int32)
        elif op == OP_GE:
            new[mask, attr] = rng.integers(0, int(value), size=n, dtype=np.int32)
        elif op == OP_RANGE:
            below_count = max(0, int(lo))
            above_start = min(card, int(hi) + 1)
            above_count = max(0, card - above_start)
            vals = np.empty(n, dtype=np.int32)
            if below_count > 0 and above_count > 0:
                use_below = rng.random(n) < (below_count / float(below_count + above_count))
                vals[use_below] = rng.integers(0, int(lo), size=int(use_below.sum()), dtype=np.int32)
                vals[~use_below] = rng.integers(above_start, card, size=int((~use_below).sum()), dtype=np.int32)
            elif below_count > 0:
                vals[:] = rng.integers(0, int(lo), size=n, dtype=np.int32)
            else:
                vals[:] = rng.integers(above_start, card, size=n, dtype=np.int32)
            new[mask, attr] = vals
        else:
            raise ValueError(f"Unknown op {op}")
    return _clip_record(new, cardinalities)


def _term_value_mask(op: int, value: int, lo: int, hi: int, card: int) -> np.ndarray:
    mask = np.zeros(card, dtype=bool)
    if op == OP_EQ:
        if 0 <= int(value) < card:
            mask[int(value)] = True
    elif op == OP_LE:
        mask[: min(card, int(value) + 1)] = True
    elif op == OP_GE:
        mask[max(0, int(value)) :] = True
    elif op == OP_RANGE:
        lower = max(0, int(lo))
        upper = min(card, int(hi) + 1)
        if lower < upper:
            mask[lower:upper] = True
    else:
        raise ValueError(f"Unknown op {op}")
    return mask


def _residual_value_tables(
    qcat: QueryCatalogue,
    cardinalities: np.ndarray,
    active_qids: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray | None,
) -> tuple[list[np.ndarray], np.ndarray]:
    value_scores = [np.zeros(int(card), dtype=np.float32) for card in cardinalities.tolist()]
    active_qids = np.asarray(active_qids, dtype=np.int32)
    active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
    if len(active_qids) == 0:
        active_qids = np.arange(qcat.m, dtype=np.int32)
    weights = residual.astype(np.float32, copy=False)
    if inv_variance is not None:
        weights = weights * np.asarray(inv_variance, dtype=np.float32)
    for qid_raw in active_qids.tolist():
        qid = int(qid_raw)
        q_weight = float(weights[qid])
        if q_weight == 0.0:
            continue
        terms = qcat.query_terms(qid)
        term_scale = 1.0 / float(max(1, len(terms)))
        for attr, op, value, lo, hi in terms:
            attr = int(attr)
            card = int(cardinalities[attr])
            if card <= 1:
                continue
            mask = _term_value_mask(int(op), int(value), int(lo), int(hi), card)
            value_scores[attr][mask] += np.float32(q_weight * term_scale)
        linear_terms = qcat.linear_terms(qid)
        linear_scale = 1.0 / float(max(1, len(linear_terms)))
        for attr, weight in linear_terms:
            attr = int(attr)
            card = int(cardinalities[attr])
            if card <= 1:
                continue
            denom = float(max(1, card - 1))
            values = np.arange(card, dtype=np.float32) / denom
            value_scores[attr] += np.float32(-q_weight * float(weight) * linear_scale) * values
    attr_scores = np.zeros(len(cardinalities), dtype=np.float32)
    for attr, scores in enumerate(value_scores):
        if len(scores) <= 1:
            continue
        attr_scores[attr] = float(np.max(scores) - np.min(scores))
    return value_scores, attr_scores


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
    inv_variance: np.ndarray | None = None,
) -> CandidateBatch:
    cfg = config.get("qdte", {})
    candidates_per_target = int(cfg.get("candidates_per_target", 64))
    total_candidates = int(cfg.get("total_candidates_per_iter", max(1, len(target_query_ids) * candidates_per_target)))
    random_fraction = float(cfg.get("random_candidate_fraction", 0.05))
    over_sample = int(cfg.get("source_over_sample_factor", 8))
    numerical_gamma = float(cfg.get("numerical_distance_gamma", 0.1))
    candidate_compiler = str(cfg.get("candidate_compiler", "single_query"))
    qdte_mixture_enabled = candidate_compiler in {"qdte_mixture", "fair_mixture", "directed_mixture"}
    masked_single_enabled = candidate_compiler in {"masked_single_query", "masked_single"}
    relaxed_masked_single_enabled = candidate_compiler in {
        "relaxed_masked_single_query",
        "relaxed_masked_single",
    }
    residual_weighted_enabled = candidate_compiler in {"residual_weighted_mutation", "residual_weighted"}
    enumerated_local_enabled = candidate_compiler in {"enumerated_local", "local_enumeration"}
    soft_single_enabled = candidate_compiler in {"soft_single_query", "soft_source_single_query"}
    residual_value_enabled = candidate_compiler in {"residual_value_mutation", "residual_value"}
    proposal_mixture_enabled = candidate_compiler in {"proposal_mixture", "mixture_scheduler"}
    constructive_partner_enabled = candidate_compiler in {"constructive_partner", "synthesized_partner"}
    constructive_partner_attached_enabled = candidate_compiler in {
        "constructive_partner_b2",
        "attached_constructive_partner",
        "constructive_partner_attached",
    }
    bounded_best_partner_enabled = candidate_compiler in {
        "bounded_best_partner",
        "best_partner",
        "exact_best_partner",
    }
    protected_same_row_enabled = candidate_compiler in {
        "protected_same_row",
        "protected_repair",
        "constructive_protected",
    }
    directed_exit_only_enabled = candidate_compiler in {"directed_exit_only", "exit_only"}
    masked_exit_only_enabled = candidate_compiler in {"masked_exit_only", "masked_directed_exit_only"}
    random_source_directed_exit_enabled = candidate_compiler in {
        "random_source_directed_exit",
        "random_directed_exit",
    }
    masked_exit_enabled = candidate_compiler in {"masked_exit_query", "masked_exit_protect"}
    masked_paired_enabled = candidate_compiler in {"masked_paired_query", "masked_paired"}
    paired_enabled = candidate_compiler in {"paired_query", "paired"} or masked_paired_enabled or masked_exit_enabled
    masked_enabled = masked_single_enabled or masked_paired_enabled or masked_exit_enabled or masked_exit_only_enabled
    mask_sampling_enabled = masked_enabled or relaxed_masked_single_enabled or qdte_mixture_enabled
    paired_fraction = float(cfg.get("paired_candidate_fraction", 1.0 if paired_enabled else 0.0))
    if masked_paired_enabled:
        paired_fraction = float(cfg.get("masked_paired_candidate_fraction", paired_fraction))
    if masked_exit_enabled:
        paired_fraction = float(cfg.get("masked_exit_candidate_fraction", paired_fraction))
    paired_try_break_source = bool(cfg.get("paired_try_break_source", True))
    if masked_paired_enabled:
        paired_try_break_source = bool(cfg.get("masked_paired_try_break_source", paired_try_break_source))
    paired_over_sample = int(cfg.get("paired_source_over_sample_factor", over_sample))
    paired_candidates_per_pair = int(cfg.get("paired_candidates_per_pair", candidates_per_target))
    paired_pair_rounds = int(cfg.get("paired_pair_rounds", 8))
    mask_min_terms = max(1, int(cfg.get("mask_min_terms", cfg.get("masked_pair_min_terms", 1))))
    mask_max_terms = max(mask_min_terms, int(cfg.get("mask_max_terms", cfg.get("masked_pair_max_terms", 2))))
    residual_value_temperature = max(1.0e-6, float(cfg.get("residual_value_temperature", 1.0)))
    residual_value_uniform_mix = min(1.0, max(0.0, float(cfg.get("residual_value_uniform_mix", 0.15))))
    residual_attr_uniform_mix = min(1.0, max(0.0, float(cfg.get("residual_attr_uniform_mix", 0.15))))
    residual_score_clip = max(1.0, float(cfg.get("residual_score_clip", 20.0)))
    soft_source_match_weight = max(0.0, float(cfg.get("soft_source_match_weight", 4.0)))
    soft_source_mismatch_weight = max(0.0, float(cfg.get("soft_source_mismatch_weight", 1.0)))
    enumerated_values_per_attr = int(cfg.get("enumerated_values_per_attr", 0))
    constructive_partner_seed_fraction = min(
        1.0,
        max(0.0, float(cfg.get("constructive_partner_seed_fraction", 0.5))),
    )
    constructive_partner_harm_queries = max(1, int(cfg.get("constructive_partner_harm_queries", 4)))
    constructive_partner_partners_per_seed = max(1, int(cfg.get("constructive_partner_partners_per_seed", 1)))
    constructive_partner_source_over_sample = max(
        1,
        int(cfg.get("constructive_partner_source_over_sample_factor", over_sample)),
    )
    constructive_partner_delta_backend = str(cfg.get("constructive_partner_delta_backend", "jax"))
    constructive_partner_side_budget = max(
        0,
        int(cfg.get("constructive_partner_side_budget", int(round(total_candidates * constructive_partner_seed_fraction)))),
    )
    best_partner_seed_fraction = min(
        1.0,
        max(0.0, float(cfg.get("best_partner_seed_fraction", constructive_partner_seed_fraction))),
    )
    best_partner_harm_queries = max(
        1,
        int(cfg.get("best_partner_harm_queries", constructive_partner_harm_queries)),
    )
    best_partner_partners_per_seed = max(
        1,
        int(cfg.get("best_partner_partners_per_seed", constructive_partner_partners_per_seed)),
    )
    best_partner_source_samples = max(
        1,
        int(cfg.get("best_partner_source_samples", max(candidates_per_target, constructive_partner_source_over_sample))),
    )
    best_partner_repairs_per_source = max(1, int(cfg.get("best_partner_repairs_per_source", 2)))
    best_partner_side_budget = max(
        0,
        int(cfg.get("best_partner_side_budget", int(round(total_candidates * best_partner_seed_fraction)))),
    )
    best_partner_delta_backend = str(cfg.get("best_partner_delta_backend", constructive_partner_delta_backend))
    best_partner_jax_batch_size = max(1, int(cfg.get("best_partner_jax_batch_size", 16384)))
    best_partner_seed_batch_size = max(1, int(cfg.get("best_partner_seed_batch_size", 32)))
    best_partner_cached_verify_margin = max(0.0, float(cfg.get("best_partner_cached_verify_margin", 0.0)))
    best_partner_min_pair_advantage = float(cfg.get("best_partner_min_pair_advantage", cfg.get("min_advantage", 1.0e-6)))
    lambda_cost = float(cfg.get("lambda_cost", 0.01))
    protected_repair_seed_fraction = min(
        1.0,
        max(0.0, float(cfg.get("protected_repair_seed_fraction", 0.5))),
    )
    protected_repair_harm_queries = max(1, int(cfg.get("protected_repair_harm_queries", 4)))
    protected_repair_restarts_per_seed = max(1, int(cfg.get("protected_repair_restarts_per_seed", 2)))
    protected_repair_max_protection_passes = max(1, int(cfg.get("protected_repair_max_protection_passes", 2)))
    protected_repair_require_target_direction = bool(cfg.get("protected_repair_require_target_direction", True))
    protected_repair_delta_backend = str(cfg.get("protected_repair_delta_backend", constructive_partner_delta_backend))
    cardinalities = schema.cardinalities
    constructive_partner_delta_index = (
        QueryDeltaIndex.build(qcat, num_attrs=schema.d)
        if (constructive_partner_enabled or constructive_partner_attached_enabled)
        and constructive_partner_delta_backend == "sparse_cpu"
        else None
    )
    best_partner_delta_index = (
        QueryDeltaIndex.build(qcat, num_attrs=schema.d)
        if bounded_best_partner_enabled and best_partner_delta_backend == "sparse_cpu"
        else None
    )
    protected_repair_delta_index = (
        QueryDeltaIndex.build(qcat, num_attrs=schema.d)
        if protected_same_row_enabled and protected_repair_delta_backend == "sparse_cpu"
        else None
    )
    candidate_capacity = total_candidates
    if constructive_partner_attached_enabled:
        candidate_capacity += constructive_partner_side_budget
    if bounded_best_partner_enabled:
        candidate_capacity += best_partner_side_budget
    random_candidate_count_cfg = cfg.get("random_candidate_count")
    directed_candidate_count_cfg = cfg.get("directed_candidate_count")
    if random_candidate_count_cfg is not None:
        random_target = min(total_candidates, max(0, int(random_candidate_count_cfg)))
        if directed_candidate_count_cfg is not None:
            directed_budget = min(total_candidates - random_target, max(0, int(directed_candidate_count_cfg)))
        else:
            directed_budget = max(0, total_candidates - random_target)
    elif directed_candidate_count_cfg is not None:
        directed_budget = min(total_candidates, max(0, int(directed_candidate_count_cfg)))
        random_target = max(0, total_candidates - directed_budget)
    else:
        random_target = int(round(total_candidates * min(1.0, max(0.0, random_fraction))))
        directed_budget = max(0, total_candidates - random_target)
    shortfall_policy = str(cfg.get("candidate_shortfall_policy", "random"))
    paired_budget = int(round(directed_budget * min(1.0, max(0.0, paired_fraction)))) if paired_enabled else 0

    row_id_buf = np.empty(candidate_capacity, dtype=np.int32)
    old_buf = np.empty((candidate_capacity, schema.d), dtype=np.int32)
    new_buf = np.empty((candidate_capacity, schema.d), dtype=np.int32)
    target_buf = np.empty(candidate_capacity, dtype=np.int32)
    repair_buf = np.empty(candidate_capacity, dtype=np.int32)
    attached_pairs: list[tuple[int, int]] = []
    source_filter_attempts = 0
    source_filter_failures = 0
    source_filter_kept = 0
    paired_source_filter_attempts = 0
    paired_source_filter_failures = 0
    paired_source_filter_kept = 0
    random_source_exit_attempts = 0
    constructive_partner_seed_candidates = 0
    constructive_partner_source_attempts = 0
    constructive_partner_source_failures = 0
    best_partner_seed_candidates = 0
    best_partner_source_attempts = 0
    best_partner_source_failures = 0
    best_partner_pairs_evaluated = 0
    best_partner_positive_pairs = 0
    protected_repair_seed_candidates = 0
    protected_repair_attempts = 0
    protected_repair_target_failures = 0
    protected_repair_protection_successes = 0
    planned_random_candidates = 0
    fallback_random_candidates = 0
    mixture_random_candidates = 0
    produced = 0
    x_syn_query_satisfaction_cache: np.ndarray | None = None

    def produced_count() -> int:
        return produced

    def append_chunk(row_ids: np.ndarray, old_rows: np.ndarray, new_rows: np.ndarray, target_qid: int, repair_type: int) -> None:
        nonlocal produced
        if len(row_ids) == 0 or produced >= candidate_capacity:
            return
        changed = np.any(old_rows != new_rows, axis=1)
        if not np.any(changed):
            return
        keep = np.flatnonzero(changed)
        n = min(int(len(keep)), candidate_capacity - produced)
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

    def append_chunk_targets(
        row_ids: np.ndarray,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        target_qids: np.ndarray,
        repair_type: int,
    ) -> None:
        nonlocal produced
        if len(row_ids) == 0 or produced >= candidate_capacity:
            return
        changed = np.any(old_rows != new_rows, axis=1)
        if not np.any(changed):
            return
        keep = np.flatnonzero(changed)
        n = min(int(len(keep)), candidate_capacity - produced)
        if n <= 0:
            return
        keep = keep[:n]
        end = produced + n
        row_id_buf[produced:end] = row_ids[keep].astype(np.int32, copy=False)
        old_buf[produced:end] = old_rows[keep].astype(np.int32, copy=False)
        new_buf[produced:end] = new_rows[keep].astype(np.int32, copy=False)
        target_buf[produced:end] = np.asarray(target_qids, dtype=np.int32)[keep]
        repair_buf[produced:end] = int(repair_type)
        produced = end

    def add_random_candidates(count: int, source: str = "planned") -> None:
        nonlocal planned_random_candidates
        nonlocal fallback_random_candidates
        nonlocal mixture_random_candidates
        if count <= 0:
            return
        before = produced_count()
        ids = rng.integers(0, X_syn.shape[0], size=count, dtype=np.int32)
        old = X_syn[ids].copy()
        new = _random_mutation_batch(old, cardinalities, rng)
        append_chunk(ids, old, new, -1, 0)
        appended = produced_count() - before
        if source == "fallback":
            fallback_random_candidates += int(appended)
        elif source == "mixture":
            mixture_random_candidates += int(appended)
        else:
            planned_random_candidates += int(appended)

    def current_x_syn_query_satisfaction() -> np.ndarray:
        nonlocal x_syn_query_satisfaction_cache
        if x_syn_query_satisfaction_cache is None:
            sat = np.empty((X_syn.shape[0], qcat.m), dtype=bool)
            for qid in range(qcat.m):
                sat[:, qid] = qcat.eval_query_np(X_syn, qid)
            x_syn_query_satisfaction_cache = sat
        return x_syn_query_satisfaction_cache

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
                    "directed_candidate_budget": float(directed_budget),
                    "random_candidate_budget": float(random_target),
                    "candidate_shortfall_policy_random": 1.0 if shortfall_policy == "random" else 0.0,
                    "produced_candidates": 0.0,
                    "directed_candidates": 0.0,
                    "random_candidates": 0.0,
                    "planned_random_candidates": 0.0,
                    "fallback_random_candidates": 0.0,
                    "mixture_random_candidates": 0.0,
                    "directed_candidate_shortfall": float(directed_budget),
                    "candidate_shortfall": float(total_candidates),
                    "source_filter_attempts": float(source_filter_attempts),
                    "source_filter_failures": float(source_filter_failures),
                    "source_filter_kept": float(source_filter_kept),
                    "paired_candidates": 0.0,
                    "masked_paired_candidates": 0.0,
                    "masked_exit_candidates": 0.0,
                    "masked_single_query_candidates": 0.0,
                    "relaxed_masked_single_query_candidates": 0.0,
                    "directed_exit_only_candidates": 0.0,
                    "masked_exit_only_candidates": 0.0,
                    "random_source_directed_exit_candidates": 0.0,
                    "residual_weighted_mutation_candidates": 0.0,
                    "enumerated_local_candidates": 0.0,
                    "soft_single_query_candidates": 0.0,
                    "residual_value_mutation_candidates": 0.0,
                    "constructive_partner_candidates": 0.0,
                    "constructive_attached_partner_candidates": 0.0,
                    "constructive_attached_pair_units": 0.0,
                    "constructive_partner_seed_candidates": float(constructive_partner_seed_candidates),
                    "constructive_partner_source_attempts": float(constructive_partner_source_attempts),
                    "constructive_partner_source_failures": float(constructive_partner_source_failures),
                    "best_partner_candidates": 0.0,
                    "best_partner_pair_units": 0.0,
                    "best_partner_seed_candidates": float(best_partner_seed_candidates),
                    "best_partner_source_attempts": float(best_partner_source_attempts),
                    "best_partner_source_failures": float(best_partner_source_failures),
                    "best_partner_pairs_evaluated": float(best_partner_pairs_evaluated),
                    "best_partner_positive_pairs": float(best_partner_positive_pairs),
                    "protected_same_row_candidates": 0.0,
                    "protected_repair_seed_candidates": float(protected_repair_seed_candidates),
                    "protected_repair_attempts": float(protected_repair_attempts),
                    "protected_repair_target_failures": float(protected_repair_target_failures),
                    "protected_repair_protection_successes": float(protected_repair_protection_successes),
                    "proposal_mixture_candidates": 0.0,
                    "qdte_mixture_candidates": 0.0,
                    "single_directed_candidates": 0.0,
                    "paired_source_filter_attempts": float(paired_source_filter_attempts),
                    "paired_source_filter_failures": float(paired_source_filter_failures),
                    "paired_source_filter_kept": float(paired_source_filter_kept),
                    "random_source_exit_attempts": float(random_source_exit_attempts),
                },
                attached_pair_indices=np.empty((0, 2), dtype=np.int32),
            )
        row_id_arr = row_id_buf[:produced]
        old_arr = old_buf[:produced]
        new_arr = new_buf[:produced]
        target_arr = target_buf[:produced]
        repair_arr = repair_buf[:produced]
        cost = compute_edit_cost(old_arr, new_arr, schema, numerical_gamma)
        produced_float = float(len(row_id_arr))
        random_produced = float(np.sum(repair_arr == 0))
        paired_produced = float(np.sum((repair_arr == 3) | (repair_arr == 4) | (repair_arr == 5)))
        masked_paired_produced = float(np.sum(repair_arr == 4))
        masked_exit_produced = float(np.sum(repair_arr == 5))
        directed_exit_only_produced = float(np.sum(repair_arr == 6))
        random_source_directed_exit_produced = float(np.sum(repair_arr == 7))
        masked_exit_only_produced = float(np.sum(repair_arr == 8))
        masked_single_produced = float(np.sum(repair_arr == 9))
        residual_weighted_produced = float(np.sum(repair_arr == 10))
        enumerated_local_produced = float(np.sum(repair_arr == 11))
        soft_single_produced = float(np.sum(repair_arr == 12))
        residual_value_produced = float(np.sum(repair_arr == 13))
        relaxed_masked_single_produced = float(np.sum(repair_arr == 14))
        constructive_partner_produced = float(np.sum((repair_arr == 15) | (repair_arr == 16)))
        constructive_attached_partner_produced = float(np.sum(repair_arr == 16))
        protected_same_row_produced = float(np.sum(repair_arr == 17))
        best_partner_produced = float(np.sum(repair_arr == 18))
        proposal_mixture_produced = 0.0
        qdte_mixture_produced = 0.0
        if proposal_mixture_enabled:
            proposal_mixture_produced = (
                random_produced
                + residual_weighted_produced
                + enumerated_local_produced
                + soft_single_produced
                + residual_value_produced
            )
        if qdte_mixture_enabled:
            qdte_mixture_produced = (
                relaxed_masked_single_produced
                + masked_single_produced
                + enumerated_local_produced
                + np.sum((repair_arr == 1) | (repair_arr == 2))
            )
        directed_produced = produced_float - random_produced
        return CandidateBatch(
            row_ids=row_id_arr,
            old_rows=old_arr,
            new_rows=new_arr,
            target_query_ids=target_arr,
            edit_cost=cost.astype(np.float32),
            repair_type=repair_arr,
            diagnostics={
                "requested_candidates": float(total_candidates),
                "directed_candidate_budget": float(directed_budget),
                "random_candidate_budget": float(random_target),
                "candidate_shortfall_policy_random": 1.0 if shortfall_policy == "random" else 0.0,
                "produced_candidates": produced_float,
                "directed_candidates": directed_produced,
                "random_candidates": random_produced,
                "planned_random_candidates": float(planned_random_candidates),
                "fallback_random_candidates": float(fallback_random_candidates),
                "mixture_random_candidates": float(mixture_random_candidates),
                "directed_candidate_shortfall": float(max(0.0, float(directed_budget) - directed_produced)),
                "paired_candidates": paired_produced,
                "masked_paired_candidates": masked_paired_produced,
                "masked_exit_candidates": masked_exit_produced,
                "masked_single_query_candidates": masked_single_produced,
                "relaxed_masked_single_query_candidates": relaxed_masked_single_produced,
                "directed_exit_only_candidates": directed_exit_only_produced,
                "masked_exit_only_candidates": masked_exit_only_produced,
                "random_source_directed_exit_candidates": random_source_directed_exit_produced,
                "residual_weighted_mutation_candidates": residual_weighted_produced,
                "enumerated_local_candidates": enumerated_local_produced,
                "soft_single_query_candidates": soft_single_produced,
                "residual_value_mutation_candidates": residual_value_produced,
                "constructive_partner_candidates": constructive_partner_produced,
                "constructive_attached_partner_candidates": constructive_attached_partner_produced,
                "constructive_attached_pair_units": float(len(attached_pairs)),
                "constructive_partner_seed_candidates": float(constructive_partner_seed_candidates),
                "constructive_partner_source_attempts": float(constructive_partner_source_attempts),
                "constructive_partner_source_failures": float(constructive_partner_source_failures),
                "best_partner_candidates": best_partner_produced,
                "best_partner_pair_units": float(
                    sum(1 for seed_idx, partner_idx in attached_pairs if int(repair_arr[int(partner_idx)]) == 18)
                ),
                "best_partner_seed_candidates": float(best_partner_seed_candidates),
                "best_partner_source_attempts": float(best_partner_source_attempts),
                "best_partner_source_failures": float(best_partner_source_failures),
                "best_partner_pairs_evaluated": float(best_partner_pairs_evaluated),
                "best_partner_positive_pairs": float(best_partner_positive_pairs),
                "protected_same_row_candidates": protected_same_row_produced,
                "protected_repair_seed_candidates": float(protected_repair_seed_candidates),
                "protected_repair_attempts": float(protected_repair_attempts),
                "protected_repair_target_failures": float(protected_repair_target_failures),
                "protected_repair_protection_successes": float(protected_repair_protection_successes),
                "proposal_mixture_candidates": proposal_mixture_produced,
                "qdte_mixture_candidates": qdte_mixture_produced,
                "single_directed_candidates": directed_produced
                - paired_produced
                - directed_exit_only_produced
                - random_source_directed_exit_produced
                - masked_exit_only_produced
                - masked_single_produced
                - relaxed_masked_single_produced
                - residual_weighted_produced
                - enumerated_local_produced
                - soft_single_produced
                - residual_value_produced
                - constructive_partner_produced
                - protected_same_row_produced
                - best_partner_produced,
                "candidate_shortfall": float(max(0, total_candidates - len(row_id_arr))),
                "source_filter_attempts": float(source_filter_attempts),
                "source_filter_failures": float(source_filter_failures),
                "source_filter_kept": float(source_filter_kept),
                "paired_source_filter_attempts": float(paired_source_filter_attempts),
                "paired_source_filter_failures": float(paired_source_filter_failures),
                "paired_source_filter_kept": float(paired_source_filter_kept),
                "random_source_exit_attempts": float(random_source_exit_attempts),
            },
            attached_pair_indices=np.asarray(attached_pairs, dtype=np.int32).reshape(-1, 2)
            if attached_pairs
            else np.empty((0, 2), dtype=np.int32),
        )

    random_only = random_fraction >= 1.0 or (len(target_query_ids) == 0 and int(cfg.get("num_active_targets", 1)) == 0)
    if random_only:
        add_random_candidates(total_candidates)
        return build_batch()

    if len(target_query_ids) == 0:
        target_query_ids = np.arange(qcat.m, dtype=np.int32)

    value_scores, attr_scores = _residual_value_tables(qcat, cardinalities, target_query_ids, residual, inv_variance)

    def sample_target_ids(count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.int32)
        active_qids = np.asarray(target_query_ids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        if len(active_qids) == 0:
            return np.full(count, -1, dtype=np.int32)
        weights = np.abs(residual[active_qids].astype(np.float64, copy=False))
        if inv_variance is not None:
            weights = weights * np.asarray(inv_variance, dtype=np.float64)[active_qids]
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return rng.choice(active_qids, size=count, replace=True).astype(np.int32, copy=False)
        probs = np.asarray(weights / total, dtype=np.float64)
        probs /= float(np.sum(probs))
        return rng.choice(active_qids, size=count, replace=True, p=probs).astype(np.int32, copy=False)

    def sample_value_for_attr(attr: int, old_value: int) -> int:
        card = int(cardinalities[attr])
        if card <= 1:
            return int(old_value)
        scores = np.asarray(value_scores[attr], dtype=np.float32)
        if len(scores) != card or not np.any(np.isfinite(scores)):
            val = int(rng.integers(0, card - 1))
            return val + int(val >= old_value)
        logits = np.clip((scores - float(np.max(scores))) / residual_value_temperature, -residual_score_clip, 0.0)
        probs = np.exp(logits.astype(np.float64))
        if 0 <= old_value < card:
            probs[int(old_value)] = 0.0
        total = float(np.sum(probs))
        if not np.isfinite(total) or total <= 0.0:
            probs = np.ones(card, dtype=np.float64)
            if 0 <= old_value < card:
                probs[int(old_value)] = 0.0
            total = float(np.sum(probs))
        probs /= total
        if residual_value_uniform_mix > 0.0:
            uniform = np.ones(card, dtype=np.float64)
            if 0 <= old_value < card:
                uniform[int(old_value)] = 0.0
            uniform_total = float(np.sum(uniform))
            if uniform_total > 0.0:
                uniform /= uniform_total
                probs = (1.0 - residual_value_uniform_mix) * probs + residual_value_uniform_mix * uniform
                probs /= float(np.sum(probs))
        return int(rng.choice(card, p=probs))

    def residual_guided_mutation_batch(rows: np.ndarray, attr_policy: str) -> np.ndarray:
        new = rows.copy()
        if len(new) == 0:
            return new
        mutable = np.flatnonzero(cardinalities > 1)
        if len(mutable) == 0:
            return new
        if attr_policy == "weighted":
            weights = np.asarray(attr_scores[mutable], dtype=np.float64)
            if residual_attr_uniform_mix > 0.0 or float(np.sum(weights)) <= 0.0 or not np.all(np.isfinite(weights)):
                uniform = np.ones(len(mutable), dtype=np.float64) / float(len(mutable))
                if float(np.sum(weights)) > 0.0 and np.all(np.isfinite(weights)):
                    weights = weights / float(np.sum(weights))
                    probs = (1.0 - residual_attr_uniform_mix) * weights + residual_attr_uniform_mix * uniform
                else:
                    probs = uniform
            else:
                probs = weights / float(np.sum(weights))
            attrs = rng.choice(mutable, size=len(new), replace=True, p=probs)
        else:
            attrs = rng.choice(mutable, size=len(new), replace=True)
        for row_idx, attr_raw in enumerate(attrs.tolist()):
            attr = int(attr_raw)
            new[row_idx, attr] = sample_value_for_attr(attr, int(new[row_idx, attr]))
        return _clip_record(new, cardinalities)

    def sample_mask_terms(qid: int) -> list[tuple[int, int, int, int, int]]:
        terms = qcat.query_terms(qid)
        if not mask_sampling_enabled or len(terms) <= mask_min_terms:
            return terms
        max_terms = min(len(terms), mask_max_terms)
        k = int(rng.integers(mask_min_terms, max_terms + 1))
        positions = rng.choice(len(terms), size=k, replace=False)
        return [terms[int(pos)] for pos in positions.tolist()]

    def sample_mask_split(
        qid: int,
    ) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int, int]]]:
        terms = qcat.query_terms(qid)
        if len(terms) == 0:
            return [], []
        if len(terms) <= mask_min_terms:
            return terms, []
        max_terms = min(len(terms), mask_max_terms)
        k = int(rng.integers(mask_min_terms, max_terms + 1))
        positions = set(int(pos) for pos in rng.choice(len(terms), size=k, replace=False).tolist())
        masked_terms = [term for pos, term in enumerate(terms) if pos in positions]
        unmasked_terms = [term for pos, term in enumerate(terms) if pos not in positions]
        return masked_terms, unmasked_terms

    def eval_source(rows: np.ndarray, qid: int, terms: list[tuple[int, int, int, int, int]]) -> np.ndarray:
        if masked_enabled and len(terms):
            return _eval_terms_np(rows, terms)
        return qcat.eval_query_np(rows, qid)

    def eval_destination(rows: np.ndarray, qid: int, terms: list[tuple[int, int, int, int, int]]) -> np.ndarray:
        if masked_enabled and len(terms):
            return _eval_terms_np(rows, terms)
        return qcat.eval_query_np(rows, qid)

    def repair_destination(
        rows: np.ndarray,
        qid: int,
        terms: list[tuple[int, int, int, int, int]],
    ) -> np.ndarray:
        if masked_enabled and len(terms):
            return _repair_enter_terms_batch(rows, terms, cardinalities, rng)
        return _repair_enter_batch(rows, qcat, qid, cardinalities, rng)

    def repair_source_exit(
        rows: np.ndarray,
        qid: int,
        terms: list[tuple[int, int, int, int, int]],
    ) -> np.ndarray:
        if masked_enabled and len(terms):
            return _repair_exit_terms_batch(rows, terms, cardinalities, rng)
        return _repair_exit_batch(rows, qcat, qid, cardinalities, rng)

    def add_masked_single_candidates(active_qids: np.ndarray, budget: int) -> None:
        nonlocal source_filter_attempts
        nonlocal source_filter_failures
        nonlocal source_filter_kept
        if budget <= 0 or produced_count() >= directed_budget:
            return
        active_qids = np.asarray(active_qids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        if len(active_qids) == 0:
            return
        end_limit = min(directed_budget, produced_count() + int(budget))
        for qid_raw in active_qids.tolist():
            if produced_count() >= end_limit:
                break
            qid = int(qid_raw)
            need_enter = residual[qid] > 0.0
            picked = 0
            attempts = 0
            while picked < candidates_per_target and attempts < 8 and produced_count() < end_limit:
                attempts += 1
                source_filter_attempts += 1
                remaining_for_query = candidates_per_target - picked
                remaining_total = end_limit - produced_count()
                sample_count = max(remaining_for_query * over_sample, candidates_per_target)
                ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
                rows = X_syn[ids]
                full_sat = qcat.eval_query_np(rows, qid)
                masked_terms, unmasked_terms = sample_mask_split(qid)
                if need_enter:
                    if unmasked_terms:
                        unmasked_sat = _eval_terms_np(rows, unmasked_terms)
                    else:
                        unmasked_sat = np.ones(len(rows), dtype=bool)
                    keep = np.flatnonzero(unmasked_sat & ~full_sat)
                else:
                    keep = np.flatnonzero(full_sat)
                source_filter_kept += int(len(keep))
                if len(keep) == 0:
                    source_filter_failures += 1
                    continue
                rng.shuffle(keep)
                take = min(remaining_for_query, remaining_total, len(keep))
                selected_ids = ids[keep[:take]].astype(np.int32, copy=False)
                old = X_syn[selected_ids].copy()
                if need_enter:
                    if masked_terms:
                        new = _repair_enter_terms_batch(old, masked_terms, cardinalities, rng)
                    else:
                        new = _repair_enter_batch(old, qcat, qid, cardinalities, rng)
                    if int(qcat.linear_num_terms[qid]) > 0:
                        new = _repair_halfspace_batch(new, qcat, qid, cardinalities, need_enter=True, rng=rng)
                    keep_after = qcat.eval_query_np(new, qid)
                else:
                    if masked_terms:
                        new = _repair_exit_terms_batch(old, masked_terms, cardinalities, rng)
                    else:
                        new = _repair_exit_batch(old, qcat, qid, cardinalities, rng)
                    if int(qcat.linear_num_terms[qid]) > 0 and np.any(qcat.eval_query_np(new, qid)):
                        new = _repair_halfspace_batch(new, qcat, qid, cardinalities, need_enter=False, rng=rng)
                    keep_after = ~qcat.eval_query_np(new, qid)
                if not np.any(keep_after):
                    continue
                selected_ids = selected_ids[keep_after]
                old = old[keep_after]
                new = new[keep_after]
                before = produced_count()
                append_chunk(selected_ids, old, new, qid, 9)
                picked += produced_count() - before

    def add_relaxed_masked_single_candidates(active_qids: np.ndarray, budget: int) -> None:
        nonlocal source_filter_attempts
        nonlocal source_filter_failures
        nonlocal source_filter_kept
        if budget <= 0 or produced_count() >= directed_budget:
            return
        active_qids = np.asarray(active_qids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        if len(active_qids) == 0:
            return
        end_limit = min(directed_budget, produced_count() + int(budget))
        for qid_raw in active_qids.tolist():
            if produced_count() >= end_limit:
                break
            qid = int(qid_raw)
            need_enter = residual[qid] > 0.0
            picked = 0
            attempts = 0
            while picked < candidates_per_target and attempts < 8 and produced_count() < end_limit:
                attempts += 1
                source_filter_attempts += 1
                masked_terms = sample_mask_terms(qid)
                if not masked_terms:
                    source_filter_failures += 1
                    continue
                remaining_for_query = candidates_per_target - picked
                remaining_total = end_limit - produced_count()
                sample_count = max(remaining_for_query * over_sample, candidates_per_target)
                ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
                rows = X_syn[ids]
                masked_sat = _eval_terms_np(rows, masked_terms)
                if need_enter:
                    keep = np.flatnonzero(~masked_sat)
                else:
                    keep = np.flatnonzero(masked_sat)
                source_filter_kept += int(len(keep))
                if len(keep) == 0:
                    source_filter_failures += 1
                    continue
                rng.shuffle(keep)
                take = min(remaining_for_query, remaining_total, len(keep))
                selected_ids = ids[keep[:take]].astype(np.int32, copy=False)
                old = X_syn[selected_ids].copy()
                if need_enter:
                    new = _repair_enter_terms_batch(old, masked_terms, cardinalities, rng)
                    keep_after = _eval_terms_np(new, masked_terms)
                else:
                    new = _repair_exit_terms_batch(old, masked_terms, cardinalities, rng)
                    keep_after = ~_eval_terms_np(new, masked_terms)
                if not np.any(keep_after):
                    continue
                selected_ids = selected_ids[keep_after]
                old = old[keep_after]
                new = new[keep_after]
                before = produced_count()
                append_chunk(selected_ids, old, new, qid, 14)
                picked += produced_count() - before

    def add_paired_candidates(active_qids: np.ndarray, budget: int) -> None:
        nonlocal source_filter_attempts
        nonlocal source_filter_failures
        nonlocal source_filter_kept
        nonlocal paired_source_filter_attempts
        nonlocal paired_source_filter_failures
        nonlocal paired_source_filter_kept
        if budget <= 0 or produced_count() >= directed_budget:
            return
        active_qids = np.asarray(active_qids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        if len(active_qids) == 0:
            return
        weights = residual.astype(np.float32, copy=False)
        if inv_variance is not None:
            weights = weights * np.asarray(inv_variance, dtype=np.float32)
        positive = active_qids[residual[active_qids] > 0.0]
        negative = active_qids[residual[active_qids] < 0.0]
        if len(positive) == 0 or len(negative) == 0:
            return
        positive = positive[np.argsort(-weights[positive])]
        negative = negative[np.argsort(weights[negative])]
        end_limit = min(directed_budget, produced_count() + int(budget))
        pair_count = len(positive) * len(negative)
        max_pair_visits = max(1, pair_count * max(1, paired_pair_rounds))
        pair_visit = 0
        while produced_count() < end_limit and pair_visit < max_pair_visits:
            q_plus = int(positive[pair_visit % len(positive)])
            q_minus = int(negative[(pair_visit // len(positive)) % len(negative)])
            source_terms = sample_mask_terms(q_minus)
            dest_terms = sample_mask_terms(q_plus)
            pair_visit += 1
            picked = 0
            attempts = 0
            while picked < paired_candidates_per_pair and attempts < 8 and produced_count() < end_limit:
                attempts += 1
                source_filter_attempts += 1
                paired_source_filter_attempts += 1
                remaining_for_pair = paired_candidates_per_pair - picked
                remaining_total = end_limit - produced_count()
                sample_count = max(remaining_for_pair * paired_over_sample, paired_candidates_per_pair)
                ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
                rows = X_syn[ids]
                source_sat = eval_source(rows, q_minus, source_terms)
                dest_sat = eval_destination(rows, q_plus, dest_terms)
                if masked_exit_enabled:
                    keep = np.flatnonzero(source_sat & dest_sat)
                else:
                    keep = np.flatnonzero(source_sat & ~dest_sat)
                source_filter_kept += int(len(keep))
                paired_source_filter_kept += int(len(keep))
                if len(keep) == 0:
                    source_filter_failures += 1
                    paired_source_filter_failures += 1
                    continue
                rng.shuffle(keep)
                take = min(remaining_for_pair, remaining_total, len(keep))
                selected_ids = ids[keep[:take]].astype(np.int32, copy=False)
                old = X_syn[selected_ids].copy()
                if masked_exit_enabled:
                    new = repair_source_exit(old, q_minus, source_terms)
                    keep_after = eval_destination(new, q_plus, dest_terms) & ~eval_source(new, q_minus, source_terms)
                    if not np.any(keep_after):
                        continue
                    selected_ids = selected_ids[keep_after]
                    old = old[keep_after]
                    new = new[keep_after]
                else:
                    new = repair_destination(old, q_plus, dest_terms)
                if (not masked_exit_enabled) and paired_try_break_source and q_plus != q_minus:
                    still_source = eval_source(new, q_minus, source_terms)
                    if np.any(still_source):
                        broken = new.copy()
                        broken[still_source] = repair_source_exit(
                            broken[still_source],
                            q_minus,
                            source_terms,
                        )
                        broken_dest = eval_destination(broken, q_plus, dest_terms)
                        broken_source = eval_source(broken, q_minus, source_terms)
                        use_broken = still_source & broken_dest & ~broken_source
                        new[use_broken] = broken[use_broken]
                before = produced_count()
                if masked_exit_enabled:
                    append_chunk(selected_ids, old, new, q_minus, 5)
                else:
                    append_chunk(selected_ids, old, new, q_plus, 4 if masked_paired_enabled else 3)
                picked += produced_count() - before

    add_paired_candidates(target_query_ids, paired_budget)

    def add_directed_exit_only_candidates(active_qids: np.ndarray, budget: int) -> None:
        nonlocal source_filter_attempts
        nonlocal source_filter_failures
        nonlocal source_filter_kept
        if budget <= 0 or produced_count() >= directed_budget:
            return
        active_qids = np.asarray(active_qids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        negative = active_qids[residual[active_qids] < 0.0]
        if len(negative) == 0:
            return
        weights = residual.astype(np.float32, copy=False)
        if inv_variance is not None:
            weights = weights * np.asarray(inv_variance, dtype=np.float32)
        negative = negative[np.argsort(weights[negative])]
        end_limit = min(directed_budget, produced_count() + int(budget))
        query_visit = 0
        stale_visits = 0
        max_stale_visits = max(1, len(negative) * 4)
        while produced_count() < end_limit and stale_visits < max_stale_visits:
            qid = int(negative[query_visit % len(negative)])
            source_terms = sample_mask_terms(qid) if masked_exit_only_enabled else []
            query_visit += 1
            before_query = produced_count()
            picked = 0
            attempts = 0
            while picked < candidates_per_target and attempts < 8 and produced_count() < end_limit:
                attempts += 1
                source_filter_attempts += 1
                remaining_for_query = candidates_per_target - picked
                remaining_total = end_limit - produced_count()
                sample_count = max(remaining_for_query * over_sample, candidates_per_target)
                ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
                rows = X_syn[ids]
                sat = eval_source(rows, qid, source_terms) if masked_exit_only_enabled else qcat.eval_query_np(rows, qid)
                keep = np.flatnonzero(sat)
                source_filter_kept += int(len(keep))
                if len(keep) == 0:
                    source_filter_failures += 1
                    continue
                rng.shuffle(keep)
                take = min(remaining_for_query, remaining_total, len(keep))
                selected_ids = ids[keep[:take]].astype(np.int32, copy=False)
                old = X_syn[selected_ids].copy()
                if masked_exit_only_enabled:
                    new = repair_source_exit(old, qid, source_terms)
                    keep_after = ~eval_source(new, qid, source_terms)
                    if not np.any(keep_after):
                        continue
                    selected_ids = selected_ids[keep_after]
                    old = old[keep_after]
                    new = new[keep_after]
                    rtype = 8
                else:
                    new = _repair_exit_batch(old, qcat, qid, cardinalities, rng)
                    rtype = 6
                before = produced_count()
                append_chunk(selected_ids, old, new, qid, rtype)
                picked += produced_count() - before
            if produced_count() == before_query:
                stale_visits += 1
            else:
                stale_visits = 0

    def add_random_source_directed_exit_candidates(active_qids: np.ndarray, budget: int) -> None:
        nonlocal random_source_exit_attempts
        if budget <= 0 or produced_count() >= directed_budget:
            return
        active_qids = np.asarray(active_qids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        negative = active_qids[residual[active_qids] < 0.0]
        if len(negative) == 0:
            return
        weights = residual.astype(np.float32, copy=False)
        if inv_variance is not None:
            weights = weights * np.asarray(inv_variance, dtype=np.float32)
        negative = negative[np.argsort(weights[negative])]
        end_limit = min(directed_budget, produced_count() + int(budget))
        query_visit = 0
        stale_visits = 0
        max_stale_visits = max(1, len(negative) * 4)
        while produced_count() < end_limit and stale_visits < max_stale_visits:
            qid = int(negative[query_visit % len(negative)])
            query_visit += 1
            remaining = end_limit - produced_count()
            count = min(candidates_per_target, remaining)
            random_source_exit_attempts += 1
            ids = rng.integers(0, X_syn.shape[0], size=count, dtype=np.int32)
            old = X_syn[ids].copy()
            new = _repair_exit_batch(old, qcat, qid, cardinalities, rng)
            before = produced_count()
            append_chunk(ids, old, new, qid, 7)
            if produced_count() == before:
                stale_visits += 1
            else:
                stale_visits = 0

    def add_residual_guided_mutation_candidates(budget: int, attr_policy: str, repair_type: int) -> None:
        if budget <= 0 or produced_count() >= directed_budget:
            return
        end_limit = min(directed_budget, produced_count() + int(budget))
        stale_visits = 0
        while produced_count() < end_limit and stale_visits < 4:
            count = end_limit - produced_count()
            ids = rng.integers(0, X_syn.shape[0], size=count, dtype=np.int32)
            old = X_syn[ids].copy()
            new = residual_guided_mutation_batch(old, attr_policy)
            qids = sample_target_ids(len(ids))
            before = produced_count()
            append_chunk_targets(ids, old, new, qids, repair_type)
            if produced_count() == before:
                stale_visits += 1
            else:
                stale_visits = 0

    def add_enumerated_local_candidates(budget: int) -> None:
        if budget <= 0 or produced_count() >= directed_budget:
            return
        mutable = np.flatnonzero(cardinalities > 1)
        if len(mutable) == 0:
            return
        end_limit = min(directed_budget, produced_count() + int(budget))
        attempts = 0
        max_attempts = max(4, int(np.ceil(max(1, budget) / max(1, len(mutable)))) * 4)
        while produced_count() < end_limit and attempts < max_attempts:
            attempts += 1
            remaining = end_limit - produced_count()
            row_sample = max(1, min(X_syn.shape[0], max(remaining, candidates_per_target)))
            ids_sample = rng.integers(0, X_syn.shape[0], size=row_sample, dtype=np.int32)
            rows_out: list[np.ndarray] = []
            new_out: list[np.ndarray] = []
            ids_out: list[int] = []
            for row_id in ids_sample.tolist():
                old = X_syn[int(row_id)]
                attr_order = mutable.copy()
                if float(np.sum(attr_scores[attr_order])) > 0.0:
                    jitter = rng.random(len(attr_order)) * 1.0e-6
                    order = np.argsort(-(attr_scores[attr_order].astype(np.float64) + jitter))
                    attr_order = attr_order[order]
                else:
                    rng.shuffle(attr_order)
                for attr_raw in attr_order.tolist():
                    attr = int(attr_raw)
                    card = int(cardinalities[attr])
                    old_value = int(old[attr])
                    values = np.asarray([v for v in range(card) if v != old_value], dtype=np.int32)
                    if len(values) == 0:
                        continue
                    value_rank = value_scores[attr][values].astype(np.float64) + rng.random(len(values)) * 1.0e-6
                    values = values[np.argsort(-value_rank)]
                    if enumerated_values_per_attr > 0:
                        values = values[:enumerated_values_per_attr]
                    for value in values.tolist():
                        candidate = old.copy()
                        candidate[attr] = int(value)
                        ids_out.append(int(row_id))
                        rows_out.append(old.copy())
                        new_out.append(candidate)
                        if len(ids_out) >= remaining:
                            break
                    if len(ids_out) >= remaining:
                        break
                if len(ids_out) >= remaining:
                    break
            if not ids_out:
                continue
            row_ids = np.asarray(ids_out, dtype=np.int32)
            old_rows = np.asarray(rows_out, dtype=np.int32)
            new_rows = np.asarray(new_out, dtype=np.int32)
            qids = sample_target_ids(len(row_ids))
            append_chunk_targets(row_ids, old_rows, new_rows, qids, 11)

    def add_soft_single_candidates(active_qids: np.ndarray, budget: int) -> None:
        nonlocal source_filter_attempts
        nonlocal source_filter_failures
        nonlocal source_filter_kept
        if budget <= 0 or produced_count() >= directed_budget:
            return
        active_qids = np.asarray(active_qids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        if len(active_qids) == 0:
            return
        weights = np.abs(residual[active_qids].astype(np.float32, copy=False))
        if inv_variance is not None:
            weights = weights * np.asarray(inv_variance, dtype=np.float32)[active_qids]
        if float(np.sum(weights)) > 0.0:
            active_qids = active_qids[np.argsort(-weights)]
        end_limit = min(directed_budget, produced_count() + int(budget))
        query_visit = 0
        stale_visits = 0
        max_stale_visits = max(1, len(active_qids) * 4)
        while produced_count() < end_limit and stale_visits < max_stale_visits:
            qid = int(active_qids[query_visit % len(active_qids)])
            query_visit += 1
            need_enter = residual[qid] > 0.0
            need_source_sat = not need_enter
            before_query = produced_count()
            picked = 0
            attempts = 0
            while picked < candidates_per_target and attempts < 8 and produced_count() < end_limit:
                attempts += 1
                source_filter_attempts += 1
                remaining_for_query = candidates_per_target - picked
                remaining_total = end_limit - produced_count()
                sample_count = max(remaining_for_query * over_sample, candidates_per_target)
                ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
                rows = X_syn[ids]
                sat = qcat.eval_query_np(rows, qid)
                preferred = sat == need_source_sat
                source_filter_kept += int(np.sum(preferred))
                probs = np.where(preferred, soft_source_match_weight, soft_source_mismatch_weight).astype(np.float64)
                total = float(np.sum(probs))
                if total <= 0.0 or not np.isfinite(total):
                    source_filter_failures += 1
                    continue
                probs /= total
                take = min(remaining_for_query, remaining_total, len(ids))
                selected_pos = rng.choice(len(ids), size=take, replace=False, p=probs)
                selected_ids = ids[selected_pos].astype(np.int32, copy=False)
                old = X_syn[selected_ids].copy()
                if need_enter:
                    new = _repair_enter_batch(old, qcat, qid, cardinalities, rng)
                else:
                    new = _repair_exit_batch(old, qcat, qid, cardinalities, rng)
                before = produced_count()
                append_chunk(selected_ids, old, new, qid, 12)
                picked += produced_count() - before
            if produced_count() == before_query:
                stale_visits += 1
            else:
                stale_visits = 0

    def add_single_query_candidates(active_qids: np.ndarray, budget: int) -> None:
        nonlocal source_filter_attempts
        nonlocal source_filter_failures
        nonlocal source_filter_kept
        if budget <= 0 or produced_count() >= directed_budget:
            return
        active_qids = np.asarray(active_qids, dtype=np.int32)
        active_qids = active_qids[(active_qids >= 0) & (active_qids < qcat.m)]
        if len(active_qids) == 0:
            return
        end_limit = min(directed_budget, produced_count() + int(budget))
        query_visit = 0
        stale_visits = 0
        max_stale_visits = max(1, len(active_qids))
        max_query_visits = len(active_qids) if shortfall_policy == "random" else None
        while (
            produced_count() < end_limit
            and stale_visits < max_stale_visits
            and (max_query_visits is None or query_visit < max_query_visits)
        ):
            qid = int(active_qids[query_visit % len(active_qids)])
            query_visit += 1
            before_query = produced_count()
            need_enter = residual[qid] > 0
            need_source_sat = not need_enter
            picked = 0
            attempts = 0
            while picked < candidates_per_target and attempts < 8 and produced_count() < end_limit:
                attempts += 1
                source_filter_attempts += 1
                remaining_for_query = candidates_per_target - picked
                remaining_total = end_limit - produced_count()
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
            if produced_count() == before_query:
                stale_visits += 1
            else:
                stale_visits = 0

    def dense_candidate_deltas(
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        backend: str,
        delta_index: QueryDeltaIndex | None,
    ) -> np.ndarray:
        if backend == "sparse_cpu":
            if delta_index is None:
                delta_index = QueryDeltaIndex.build(qcat, num_attrs=schema.d)
            return delta_index.dense_candidate_deltas(old_rows, new_rows)
        if backend in {"dense_unique", "unique_dense", "cpu_unique"}:
            return _dense_candidate_deltas_cpu_unique(old_rows, new_rows, qcat)
        if backend in {"dense_cpu", "dense_cached", "cpu_cached", "cached_cpu", "source_cached", "dense_source_cached"}:
            return _dense_candidate_deltas_cpu(old_rows, new_rows, qcat)
        if backend in {"jax_batch", "batched_jax", "gpu_batch", "jax_fixed", "gpu_fixed"}:
            return _dense_candidate_deltas_jax_fixed_batch(
                old_rows,
                new_rows,
                qcat,
                best_partner_jax_batch_size,
            )
        return _dense_candidate_deltas_jax(old_rows, new_rows, qcat)

    def dense_candidate_deltas_for_source_rows(
        row_ids: np.ndarray,
        old_rows: np.ndarray,
        new_rows: np.ndarray,
        backend: str,
        delta_index: QueryDeltaIndex | None,
    ) -> np.ndarray:
        if backend in {"dense_cached", "cpu_cached", "cached_cpu"}:
            ids = np.asarray(row_ids, dtype=np.int32)
            if len(ids) == len(old_rows) and np.all((ids >= 0) & (ids < X_syn.shape[0])):
                old_sat = current_x_syn_query_satisfaction()[ids]
                new_sat = np.empty((len(new_rows), qcat.m), dtype=bool)
                for qid in range(qcat.m):
                    new_sat[:, qid] = qcat.eval_query_np(new_rows, qid)
                return new_sat.astype(np.int8) - old_sat.astype(np.int8)
        return dense_candidate_deltas(old_rows, new_rows, backend, delta_index)

    def source_satisfaction_for_ids(ids: np.ndarray, rows: np.ndarray, qid: int, backend: str) -> np.ndarray:
        if backend in {"dense_cached", "cpu_cached", "cached_cpu", "source_cached", "dense_source_cached"}:
            source_ids = np.asarray(ids, dtype=np.int32)
            if len(source_ids) == len(rows) and np.all((source_ids >= 0) & (source_ids < X_syn.shape[0])):
                return current_x_syn_query_satisfaction()[source_ids, int(qid)]
        return qcat.eval_query_np(rows, qid)

    def row_satisfies(row: np.ndarray, qid: int) -> bool:
        return bool(qcat.eval_query_np(row.reshape(1, -1), qid)[0])

    def target_direction_ok(old_row: np.ndarray, new_row: np.ndarray, qid: int) -> bool:
        if qid < 0 or qid >= qcat.m:
            return True
        if residual[qid] > 0.0:
            return (not row_satisfies(old_row, qid)) and row_satisfies(new_row, qid)
        if residual[qid] < 0.0:
            return row_satisfies(old_row, qid) and (not row_satisfies(new_row, qid))
        return True

    def repair_row_to_membership(row: np.ndarray, qid: int, should_satisfy: bool) -> np.ndarray:
        rows = row.reshape(1, -1)
        if should_satisfy:
            return _repair_enter_batch(rows, qcat, qid, cardinalities, rng)[0]
        return _repair_exit_batch(rows, qcat, qid, cardinalities, rng)[0]

    def add_protected_same_row_candidates(seed_start: int, seed_end: int, budget: int) -> None:
        nonlocal protected_repair_seed_candidates
        nonlocal protected_repair_attempts
        nonlocal protected_repair_target_failures
        nonlocal protected_repair_protection_successes
        if budget <= 0 or produced_count() >= directed_budget or seed_end <= seed_start:
            return
        seed_old = old_buf[seed_start:seed_end].copy()
        seed_new = new_buf[seed_start:seed_end].copy()
        seed_rows = row_id_buf[seed_start:seed_end].copy()
        seed_targets = target_buf[seed_start:seed_end].copy()
        if len(seed_old) == 0:
            return
        weights = residual.astype(np.float32, copy=False)
        if inv_variance is not None:
            weights = weights * np.asarray(inv_variance, dtype=np.float32)
        seed_delta = dense_candidate_deltas(
            seed_old,
            seed_new,
            protected_repair_delta_backend,
            protected_repair_delta_index,
        )
        contribution = seed_delta.astype(np.float32) * weights.reshape(1, -1)
        harm = contribution < 0.0
        damage = np.where(harm, -contribution, 0.0)
        seed_scores = damage.sum(axis=1)
        seed_order = np.argsort(-seed_scores)
        end_limit = min(directed_budget, produced_count() + int(budget))
        for seed_local_raw in seed_order.tolist():
            if produced_count() >= end_limit:
                break
            seed_local = int(seed_local_raw)
            if float(seed_scores[seed_local]) <= 0.0:
                break
            target_qid = int(seed_targets[seed_local])
            if target_qid < 0 or target_qid >= qcat.m:
                continue
            harmed_qids = np.flatnonzero(harm[seed_local])
            if len(harmed_qids) == 0:
                continue
            harmed_order = harmed_qids[np.argsort(-damage[seed_local, harmed_qids])]
            harmed_order = np.asarray([qid for qid in harmed_order.tolist() if int(qid) != target_qid], dtype=np.int32)
            if len(harmed_order) == 0:
                continue
            harmed_order = harmed_order[:protected_repair_harm_queries]
            protected_repair_seed_candidates += 1
            produced_for_seed = 0
            for restart in range(protected_repair_restarts_per_seed):
                if produced_count() >= end_limit:
                    break
                protected_repair_attempts += 1
                candidate = seed_new[seed_local].copy()
                protect_order = harmed_order.copy()
                if restart > 0 and len(protect_order) > 1:
                    rng.shuffle(protect_order)
                for _ in range(protected_repair_max_protection_passes):
                    if not target_direction_ok(seed_old[seed_local], candidate, target_qid):
                        candidate = repair_row_to_membership(candidate, target_qid, bool(residual[target_qid] > 0.0))
                    for harmed_qid_raw in protect_order.tolist():
                        harmed_qid = int(harmed_qid_raw)
                        old_sat = row_satisfies(seed_old[seed_local], harmed_qid)
                        if row_satisfies(candidate, harmed_qid) == old_sat:
                            continue
                        candidate = repair_row_to_membership(candidate, harmed_qid, old_sat)
                if protected_repair_require_target_direction and not target_direction_ok(
                    seed_old[seed_local],
                    candidate,
                    target_qid,
                ):
                    protected_repair_target_failures += 1
                    continue
                if np.array_equal(candidate, seed_new[seed_local]):
                    continue
                protected_count = 0
                for harmed_qid_raw in protect_order.tolist():
                    harmed_qid = int(harmed_qid_raw)
                    old_sat = row_satisfies(seed_old[seed_local], harmed_qid)
                    new_sat = row_satisfies(candidate, harmed_qid)
                    if old_sat == new_sat and int(seed_delta[seed_local, harmed_qid]) != 0:
                        protected_count += 1
                if protected_count <= 0:
                    continue
                before = produced_count()
                append_chunk(
                    np.asarray([int(seed_rows[seed_local])], dtype=np.int32),
                    seed_old[seed_local : seed_local + 1],
                    candidate.reshape(1, -1),
                    target_qid,
                    17,
                )
                appended = produced_count() - before
                if appended > 0:
                    protected_repair_protection_successes += int(protected_count)
                    produced_for_seed += int(appended)
                if produced_for_seed >= protected_repair_restarts_per_seed:
                    break

    def add_constructive_partner_candidates(
        seed_start: int,
        seed_end: int,
        budget: int,
        *,
        end_limit_total: int | None = None,
        attach_to_seed: bool = False,
        repair_type: int = 15,
    ) -> None:
        nonlocal constructive_partner_seed_candidates
        nonlocal constructive_partner_source_attempts
        nonlocal constructive_partner_source_failures
        limit_total = directed_budget if end_limit_total is None else int(end_limit_total)
        if budget <= 0 or produced_count() >= limit_total or seed_end <= seed_start:
            return
        seed_old = old_buf[seed_start:seed_end].copy()
        seed_new = new_buf[seed_start:seed_end].copy()
        seed_rows = row_id_buf[seed_start:seed_end].copy()
        if len(seed_old) == 0:
            return
        weights = residual.astype(np.float32, copy=False)
        if inv_variance is not None:
            weights = weights * np.asarray(inv_variance, dtype=np.float32)
        if constructive_partner_delta_backend == "sparse_cpu":
            delta_index = constructive_partner_delta_index
            if delta_index is None:
                delta_index = QueryDeltaIndex.build(qcat, num_attrs=schema.d)
            seed_delta = delta_index.dense_candidate_deltas(seed_old, seed_new)
        elif constructive_partner_delta_backend == "dense_cpu":
            seed_delta = _dense_candidate_deltas_cpu(seed_old, seed_new, qcat)
        else:
            seed_delta = _dense_candidate_deltas_jax(seed_old, seed_new, qcat)
        contribution = seed_delta.astype(np.float32) * weights.reshape(1, -1)
        harm = contribution < 0.0
        damage = np.where(harm, -contribution, 0.0)
        seed_scores = damage.sum(axis=1)
        seed_order = np.argsort(-seed_scores)
        end_limit = min(limit_total, produced_count() + int(budget))
        for seed_local in seed_order.tolist():
            if produced_count() >= end_limit:
                break
            if float(seed_scores[int(seed_local)]) <= 0.0:
                break
            constructive_partner_seed_candidates += 1
            harmed_qids = np.flatnonzero(harm[int(seed_local)])
            if len(harmed_qids) == 0:
                continue
            harmed_order = harmed_qids[np.argsort(-damage[int(seed_local), harmed_qids])]
            harmed_order = harmed_order[:constructive_partner_harm_queries]
            partner_count_for_seed = 0
            for qid_raw in harmed_order.tolist():
                if produced_count() >= end_limit or partner_count_for_seed >= constructive_partner_partners_per_seed:
                    break
                qid = int(qid_raw)
                sign = int(seed_delta[int(seed_local), qid])
                if sign == 0:
                    continue
                partner_need_enter = sign < 0
                attempts = 0
                while (
                    attempts < 4
                    and produced_count() < end_limit
                    and partner_count_for_seed < constructive_partner_partners_per_seed
                ):
                    attempts += 1
                    constructive_partner_source_attempts += 1
                    remaining = min(
                        end_limit - produced_count(),
                        constructive_partner_partners_per_seed - partner_count_for_seed,
                    )
                    sample_count = max(
                        remaining * constructive_partner_source_over_sample,
                        candidates_per_target,
                    )
                    ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
                    rows = X_syn[ids]
                    sat = qcat.eval_query_np(rows, qid)
                    keep_mask = ~sat if partner_need_enter else sat
                    keep_mask &= ids != int(seed_rows[int(seed_local)])
                    keep = np.flatnonzero(keep_mask)
                    if len(keep) == 0:
                        constructive_partner_source_failures += 1
                        continue
                    rng.shuffle(keep)
                    take = min(remaining, len(keep))
                    selected_ids = ids[keep[:take]].astype(np.int32, copy=False)
                    old = X_syn[selected_ids].copy()
                    if partner_need_enter:
                        new = _repair_enter_batch(old, qcat, qid, cardinalities, rng)
                        keep_after = qcat.eval_query_np(new, qid)
                    else:
                        new = _repair_exit_batch(old, qcat, qid, cardinalities, rng)
                        keep_after = ~qcat.eval_query_np(new, qid)
                    if not np.any(keep_after):
                        constructive_partner_source_failures += 1
                        continue
                    selected_ids = selected_ids[keep_after]
                    old = old[keep_after]
                    new = new[keep_after]
                    before = produced_count()
                    append_chunk(selected_ids, old, new, qid, repair_type)
                    appended = produced_count() - before
                    if attach_to_seed and appended > 0:
                        seed_candidate_idx = int(seed_start + int(seed_local))
                        for partner_idx in range(before, before + appended):
                            attached_pairs.append((seed_candidate_idx, int(partner_idx)))
                    partner_count_for_seed += int(appended)
                    if appended == 0:
                        constructive_partner_source_failures += 1

    def add_bounded_best_partner_candidates(seed_start: int, seed_end: int, budget: int) -> None:
        nonlocal best_partner_seed_candidates
        nonlocal best_partner_source_attempts
        nonlocal best_partner_source_failures
        nonlocal best_partner_pairs_evaluated
        nonlocal best_partner_positive_pairs
        if budget <= 0 or produced_count() >= candidate_capacity or seed_end <= seed_start:
            return
        seed_old = old_buf[seed_start:seed_end].copy()
        seed_new = new_buf[seed_start:seed_end].copy()
        seed_rows = row_id_buf[seed_start:seed_end].copy()
        if len(seed_old) == 0:
            return
        weights = residual.astype(np.float32, copy=False)
        inv = np.ones(qcat.m, dtype=np.float32)
        if inv_variance is not None:
            inv = np.asarray(inv_variance, dtype=np.float32)
            weights = weights * inv
        seed_delta = dense_candidate_deltas_for_source_rows(
            seed_rows,
            seed_old,
            seed_new,
            best_partner_delta_backend,
            best_partner_delta_index,
        )
        seed_delta_float = seed_delta.astype(np.float32, copy=False)
        seed_cost = compute_edit_cost(seed_old, seed_new, schema, numerical_gamma).astype(np.float32, copy=False)
        seed_advantage = (
            seed_delta_float @ weights
            - 0.5 * ((seed_delta_float * seed_delta_float) @ inv)
            - float(lambda_cost) * seed_cost
        )
        contribution = seed_delta_float * weights.reshape(1, -1)
        harm = contribution < 0.0
        damage = np.where(harm, -contribution, 0.0)
        seed_damage = damage.sum(axis=1)
        seed_score = seed_damage + np.maximum(seed_advantage, 0.0)
        seed_order = np.argsort(-seed_score)
        end_limit = min(candidate_capacity, produced_count() + int(budget))
        batched_best_partner = best_partner_delta_backend in {"jax_batch", "batched_jax", "gpu_batch"}
        if batched_best_partner:
            seed_order_list = [int(x) for x in seed_order.tolist()]
            stop_seed_scan = False
            for chunk_start in range(0, len(seed_order_list), best_partner_seed_batch_size):
                if produced_count() >= end_limit or stop_seed_scan:
                    break
                chunk_seed_order: list[int] = []
                partner_ids: list[np.ndarray] = []
                partner_old: list[np.ndarray] = []
                partner_new: list[np.ndarray] = []
                partner_targets: list[np.ndarray] = []
                partner_seed_locals: list[np.ndarray] = []
                chunk = seed_order_list[chunk_start : chunk_start + best_partner_seed_batch_size]
                for seed_local in chunk:
                    if float(seed_damage[seed_local]) <= 0.0:
                        stop_seed_scan = True
                        break
                    best_partner_seed_candidates += 1
                    harmed_qids = np.flatnonzero(harm[seed_local])
                    if len(harmed_qids) == 0:
                        continue
                    harmed_order = harmed_qids[np.argsort(-damage[seed_local, harmed_qids])]
                    harmed_order = harmed_order[:best_partner_harm_queries]
                    chunk_seed_order.append(seed_local)
                    seed_partner_parts = 0
                    for qid_raw in harmed_order.tolist():
                        qid = int(qid_raw)
                        sign = int(seed_delta[seed_local, qid])
                        if sign == 0:
                            continue
                        partner_need_enter = sign < 0
                        best_partner_source_attempts += 1
                        ids = rng.integers(0, X_syn.shape[0], size=best_partner_source_samples, dtype=np.int32)
                        rows = X_syn[ids]
                        sat = source_satisfaction_for_ids(ids, rows, qid, best_partner_delta_backend)
                        keep_mask = ~sat if partner_need_enter else sat
                        keep_mask &= ids != int(seed_rows[seed_local])
                        keep = np.flatnonzero(keep_mask)
                        if len(keep) == 0:
                            best_partner_source_failures += 1
                            continue
                        rng.shuffle(keep)
                        selected_ids = ids[keep].astype(np.int32, copy=False)
                        old_base = X_syn[selected_ids].copy()
                        for _ in range(best_partner_repairs_per_source):
                            if partner_need_enter:
                                new = _repair_enter_batch(old_base, qcat, qid, cardinalities, rng)
                                keep_after = qcat.eval_query_np(new, qid)
                            else:
                                new = _repair_exit_batch(old_base, qcat, qid, cardinalities, rng)
                                keep_after = ~qcat.eval_query_np(new, qid)
                            keep_after &= np.any(old_base != new, axis=1)
                            if not np.any(keep_after):
                                continue
                            count = int(np.sum(keep_after))
                            partner_ids.append(selected_ids[keep_after].astype(np.int32, copy=False))
                            partner_old.append(old_base[keep_after].copy())
                            partner_new.append(new[keep_after].copy())
                            partner_targets.append(np.full(count, qid, dtype=np.int32))
                            partner_seed_locals.append(np.full(count, seed_local, dtype=np.int32))
                            seed_partner_parts += count
                    if seed_partner_parts <= 0:
                        best_partner_source_failures += 1
                if not partner_old:
                    continue
                ids_all = np.concatenate(partner_ids).astype(np.int32, copy=False)
                old_all = np.concatenate(partner_old, axis=0).astype(np.int32, copy=False)
                new_all = np.concatenate(partner_new, axis=0).astype(np.int32, copy=False)
                targets_all = np.concatenate(partner_targets).astype(np.int32, copy=False)
                seed_local_all = np.concatenate(partner_seed_locals).astype(np.int32, copy=False)
                partner_delta = _dense_candidate_deltas_jax_fixed_batch(
                    old_all,
                    new_all,
                    qcat,
                    best_partner_jax_batch_size,
                )
                partner_delta_float = partner_delta.astype(np.float32, copy=False)
                partner_cost = compute_edit_cost(old_all, new_all, schema, numerical_gamma).astype(np.float32, copy=False)
                pair_delta = partner_delta_float + seed_delta_float[seed_local_all]
                pair_cost = partner_cost + seed_cost[seed_local_all]
                pair_advantage = (
                    pair_delta @ weights
                    - 0.5 * ((pair_delta * pair_delta) @ inv)
                    - float(lambda_cost) * pair_cost
                )
                best_partner_pairs_evaluated += int(len(pair_advantage))
                positive = np.flatnonzero(pair_advantage > float(best_partner_min_pair_advantage))
                best_partner_positive_pairs += int(len(positive))
                if len(positive) == 0:
                    continue
                for seed_local in chunk_seed_order:
                    if produced_count() >= end_limit:
                        break
                    seed_positive = positive[seed_local_all[positive] == int(seed_local)]
                    if len(seed_positive) == 0:
                        continue
                    order = seed_positive[np.argsort(-pair_advantage[seed_positive])]
                    take = min(best_partner_partners_per_seed, len(order), end_limit - produced_count())
                    if take <= 0:
                        break
                    for partner_local in order[:take].tolist():
                        before = produced_count()
                        append_chunk(
                            ids_all[int(partner_local) : int(partner_local) + 1],
                            old_all[int(partner_local) : int(partner_local) + 1],
                            new_all[int(partner_local) : int(partner_local) + 1],
                            int(targets_all[int(partner_local)]),
                            18,
                        )
                        appended = produced_count() - before
                        if appended > 0:
                            attached_pairs.append((int(seed_start + seed_local), int(before)))
            return
        for seed_local_raw in seed_order.tolist():
            if produced_count() >= end_limit:
                break
            seed_local = int(seed_local_raw)
            if float(seed_damage[seed_local]) <= 0.0:
                break
            best_partner_seed_candidates += 1
            harmed_qids = np.flatnonzero(harm[seed_local])
            if len(harmed_qids) == 0:
                continue
            harmed_order = harmed_qids[np.argsort(-damage[seed_local, harmed_qids])]
            harmed_order = harmed_order[:best_partner_harm_queries]
            partner_ids: list[np.ndarray] = []
            partner_old: list[np.ndarray] = []
            partner_new: list[np.ndarray] = []
            partner_targets: list[np.ndarray] = []
            for qid_raw in harmed_order.tolist():
                if produced_count() >= end_limit:
                    break
                qid = int(qid_raw)
                sign = int(seed_delta[seed_local, qid])
                if sign == 0:
                    continue
                partner_need_enter = sign < 0
                best_partner_source_attempts += 1
                ids = rng.integers(0, X_syn.shape[0], size=best_partner_source_samples, dtype=np.int32)
                rows = X_syn[ids]
                sat = source_satisfaction_for_ids(ids, rows, qid, best_partner_delta_backend)
                keep_mask = ~sat if partner_need_enter else sat
                keep_mask &= ids != int(seed_rows[seed_local])
                keep = np.flatnonzero(keep_mask)
                if len(keep) == 0:
                    best_partner_source_failures += 1
                    continue
                rng.shuffle(keep)
                selected_ids = ids[keep].astype(np.int32, copy=False)
                old_base = X_syn[selected_ids].copy()
                for _ in range(best_partner_repairs_per_source):
                    if partner_need_enter:
                        new = _repair_enter_batch(old_base, qcat, qid, cardinalities, rng)
                        keep_after = qcat.eval_query_np(new, qid)
                    else:
                        new = _repair_exit_batch(old_base, qcat, qid, cardinalities, rng)
                        keep_after = ~qcat.eval_query_np(new, qid)
                    keep_after &= np.any(old_base != new, axis=1)
                    if not np.any(keep_after):
                        continue
                    partner_ids.append(selected_ids[keep_after].astype(np.int32, copy=False))
                    partner_old.append(old_base[keep_after].copy())
                    partner_new.append(new[keep_after].copy())
                    partner_targets.append(np.full(int(np.sum(keep_after)), qid, dtype=np.int32))
            if not partner_old:
                best_partner_source_failures += 1
                continue
            ids_all = np.concatenate(partner_ids).astype(np.int32, copy=False)
            old_all = np.concatenate(partner_old, axis=0).astype(np.int32, copy=False)
            new_all = np.concatenate(partner_new, axis=0).astype(np.int32, copy=False)
            targets_all = np.concatenate(partner_targets).astype(np.int32, copy=False)
            partner_delta = dense_candidate_deltas_for_source_rows(
                ids_all,
                old_all,
                new_all,
                best_partner_delta_backend,
                best_partner_delta_index,
            )
            partner_delta_float = partner_delta.astype(np.float32, copy=False)
            partner_cost = compute_edit_cost(old_all, new_all, schema, numerical_gamma).astype(np.float32, copy=False)
            pair_delta = partner_delta_float + seed_delta_float[seed_local].reshape(1, -1)
            pair_cost = partner_cost + float(seed_cost[seed_local])
            pair_advantage = (
                pair_delta @ weights
                - 0.5 * ((pair_delta * pair_delta) @ inv)
                - float(lambda_cost) * pair_cost
            )
            best_partner_pairs_evaluated += int(len(pair_advantage))
            if (
                best_partner_delta_backend in {"dense_cached", "cpu_cached", "cached_cpu"}
                and len(pair_advantage) > 0
                and best_partner_cached_verify_margin > 0.0
            ):
                approx_positive = np.flatnonzero(pair_advantage > float(best_partner_min_pair_advantage))
                if len(approx_positive) > 0:
                    verify_cutoff = max(
                        float(best_partner_min_pair_advantage) - best_partner_cached_verify_margin,
                        float(np.max(pair_advantage[approx_positive])) - best_partner_cached_verify_margin,
                    )
                else:
                    verify_cutoff = float(best_partner_min_pair_advantage) - best_partner_cached_verify_margin
                verify = np.flatnonzero(pair_advantage >= verify_cutoff)
                if len(verify) > 0:
                    exact_delta = _dense_candidate_deltas_cpu(old_all[verify], new_all[verify], qcat).astype(
                        np.float32,
                        copy=False,
                    )
                    exact_cost = partner_cost[verify]
                    exact_pair_delta = exact_delta + seed_delta_float[seed_local].reshape(1, -1)
                    exact_pair_cost = exact_cost + float(seed_cost[seed_local])
                    pair_advantage[verify] = (
                        exact_pair_delta @ weights
                        - 0.5 * ((exact_pair_delta * exact_pair_delta) @ inv)
                        - float(lambda_cost) * exact_pair_cost
                    )
            positive = np.flatnonzero(pair_advantage > float(best_partner_min_pair_advantage))
            best_partner_positive_pairs += int(len(positive))
            if len(positive) == 0:
                continue
            order = positive[np.argsort(-pair_advantage[positive])]
            take = min(best_partner_partners_per_seed, len(order), end_limit - produced_count())
            if take <= 0:
                break
            for partner_local in order[:take].tolist():
                before = produced_count()
                append_chunk(
                    ids_all[int(partner_local) : int(partner_local) + 1],
                    old_all[int(partner_local) : int(partner_local) + 1],
                    new_all[int(partner_local) : int(partner_local) + 1],
                    int(targets_all[int(partner_local)]),
                    18,
                )
                appended = produced_count() - before
                if appended > 0:
                    attached_pairs.append((int(seed_start + seed_local), int(before)))

    def allocate_fractional_budgets(total: int, fractions: list[float]) -> np.ndarray:
        total = max(0, int(total))
        values = np.asarray([min(1.0, max(0.0, float(x))) for x in fractions], dtype=np.float64)
        fraction_sum = float(np.sum(values))
        if fraction_sum <= 0.0:
            values[:] = 0.0
            values[0] = 1.0
            fraction_sum = 1.0
        if fraction_sum > 1.0:
            values /= fraction_sum
        budgets = np.floor(values * total).astype(np.int32)
        remainder = int(total - int(np.sum(budgets)))
        for idx in range(remainder):
            budgets[idx % len(budgets)] += 1
        return budgets

    if qdte_mixture_enabled:
        mixture_budget = max(0, directed_budget - produced_count())
        budgets = allocate_fractional_budgets(
            mixture_budget,
            [
                float(cfg.get("qdte_mixture_single_fraction", 0.45)),
                float(cfg.get("qdte_mixture_masked_single_fraction", 0.20)),
                float(cfg.get("qdte_mixture_enumerated_fraction", 0.25)),
                float(cfg.get("qdte_mixture_relaxed_masked_fraction", 0.10)),
            ],
        )
        add_single_query_candidates(target_query_ids, int(budgets[0]))
        add_masked_single_candidates(target_query_ids, int(budgets[1]))
        add_enumerated_local_candidates(int(budgets[2]))
        add_relaxed_masked_single_candidates(target_query_ids, int(budgets[3]))
    elif proposal_mixture_enabled:
        mixture_budget = max(0, directed_budget - produced_count())
        mixture_random_fraction = min(1.0, max(0.0, float(cfg.get("mixture_random_fraction", 0.20))))
        mixture_residual_weighted_fraction = min(
            1.0, max(0.0, float(cfg.get("mixture_residual_weighted_fraction", 0.25)))
        )
        mixture_enumerated_fraction = min(1.0, max(0.0, float(cfg.get("mixture_enumerated_fraction", 0.20))))
        mixture_soft_single_fraction = min(1.0, max(0.0, float(cfg.get("mixture_soft_single_fraction", 0.20))))
        mixture_residual_value_fraction = min(1.0, max(0.0, float(cfg.get("mixture_residual_value_fraction", 0.15))))
        budgets = allocate_fractional_budgets(
            mixture_budget,
            [
                mixture_random_fraction,
                mixture_residual_weighted_fraction,
                mixture_enumerated_fraction,
                mixture_soft_single_fraction,
                mixture_residual_value_fraction,
            ],
        )
        add_random_candidates(int(budgets[0]), source="mixture")
        add_residual_guided_mutation_candidates(int(budgets[1]), "weighted", 10)
        add_enumerated_local_candidates(int(budgets[2]))
        add_soft_single_candidates(target_query_ids, int(budgets[3]))
        add_residual_guided_mutation_candidates(int(budgets[4]), "uniform", 13)
    elif bounded_best_partner_enabled:
        seed_start = produced_count()
        add_single_query_candidates(target_query_ids, directed_budget - produced_count())
        seed_end = produced_count()
        add_random_candidates(min(random_target, max(0, total_candidates - produced_count())), source="planned")
        add_bounded_best_partner_candidates(
            seed_start,
            seed_end,
            min(best_partner_side_budget, max(0, candidate_capacity - produced_count())),
        )
    elif constructive_partner_attached_enabled:
        seed_start = produced_count()
        add_single_query_candidates(target_query_ids, directed_budget - produced_count())
        seed_end = produced_count()
        add_random_candidates(min(random_target, max(0, total_candidates - produced_count())), source="planned")
        add_constructive_partner_candidates(
            seed_start,
            seed_end,
            min(constructive_partner_side_budget, max(0, candidate_capacity - produced_count())),
            end_limit_total=candidate_capacity,
            attach_to_seed=True,
            repair_type=16,
        )
    elif constructive_partner_enabled:
        constructive_budget = max(0, directed_budget - produced_count())
        seed_budget = int(round(constructive_budget * constructive_partner_seed_fraction))
        if constructive_budget > 1:
            seed_budget = min(max(1, seed_budget), constructive_budget - 1)
        else:
            seed_budget = constructive_budget
        seed_start = produced_count()
        add_single_query_candidates(target_query_ids, seed_budget)
        seed_end = produced_count()
        add_constructive_partner_candidates(seed_start, seed_end, directed_budget - produced_count())
        if produced_count() < directed_budget:
            add_single_query_candidates(target_query_ids, directed_budget - produced_count())
    elif protected_same_row_enabled:
        protected_budget = max(0, directed_budget - produced_count())
        seed_budget = int(round(protected_budget * protected_repair_seed_fraction))
        if protected_budget > 1:
            seed_budget = min(max(1, seed_budget), protected_budget - 1)
        else:
            seed_budget = protected_budget
        seed_start = produced_count()
        add_single_query_candidates(target_query_ids, seed_budget)
        seed_end = produced_count()
        add_protected_same_row_candidates(seed_start, seed_end, directed_budget - produced_count())
        if produced_count() < directed_budget:
            add_single_query_candidates(target_query_ids, directed_budget - produced_count())
    elif residual_weighted_enabled:
        add_residual_guided_mutation_candidates(directed_budget - produced_count(), "weighted", 10)
    elif enumerated_local_enabled:
        add_enumerated_local_candidates(directed_budget - produced_count())
    elif soft_single_enabled:
        add_soft_single_candidates(target_query_ids, directed_budget - produced_count())
    elif residual_value_enabled:
        add_residual_guided_mutation_candidates(directed_budget - produced_count(), "uniform", 13)
    elif relaxed_masked_single_enabled:
        add_relaxed_masked_single_candidates(target_query_ids, directed_budget - produced_count())
    elif masked_single_enabled:
        add_masked_single_candidates(target_query_ids, directed_budget - produced_count())
    elif directed_exit_only_enabled or masked_exit_only_enabled:
        add_directed_exit_only_candidates(target_query_ids, directed_budget - produced_count())
    elif random_source_directed_exit_enabled:
        add_random_source_directed_exit_candidates(target_query_ids, directed_budget - produced_count())
    else:
        add_single_query_candidates(target_query_ids, directed_budget - produced_count())

    add_random_candidates(min(random_target, max(0, total_candidates - produced_count())), source="planned")

    if shortfall_policy == "random":
        while produced_count() < total_candidates:
            before = produced_count()
            add_random_candidates(total_candidates - before, source="fallback")
            if produced_count() == before:
                break

    return build_batch()
