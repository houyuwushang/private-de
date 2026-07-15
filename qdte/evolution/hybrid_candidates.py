from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from qdte.evolution.candidates import compute_edit_cost, generate_candidates
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.queries.delta_index import QueryDeltaIndex
from qdte.queries.eval_jax import eval_records_queries, eval_records_queries_arrays
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema


OPERATOR_DIRECTED = 0
OPERATOR_MUTATION = 1
OPERATOR_SWAP = 2
OPERATOR_CROSSOVER = 3
OPERATOR_DIRECTED_SWAP = 4
OPERATOR_DIRECTED_CROSSOVER = 5
OPERATOR_GLOBAL_DIRECTED_SWAP = 6
OPERATOR_INTERACTION_CYCLE = 7

OPERATOR_NAMES = {
    OPERATOR_DIRECTED: "directed",
    OPERATOR_MUTATION: "mutation",
    OPERATOR_SWAP: "swap",
    OPERATOR_CROSSOVER: "crossover",
    OPERATOR_DIRECTED_SWAP: "directed_swap",
    OPERATOR_DIRECTED_CROSSOVER: "directed_crossover",
    OPERATOR_GLOBAL_DIRECTED_SWAP: "global_directed_swap",
    OPERATOR_INTERACTION_CYCLE: "interaction_cycle",
}


@dataclass(frozen=True)
class HybridCandidateUnitBatch:
    row_ids: np.ndarray
    old_rows: np.ndarray
    new_rows: np.ndarray
    row_mask: np.ndarray
    operator_ids: np.ndarray
    target_query_ids: np.ndarray
    edit_cost: np.ndarray

    @property
    def count(self) -> int:
        return int(self.row_ids.shape[0])

    @property
    def row_evaluations(self) -> int:
        return int(np.sum(self.row_mask, dtype=np.int64))

    @property
    def width(self) -> int:
        return int(self.old_rows.shape[2])

    def validate(self) -> None:
        count = self.count
        if self.row_ids.shape != (count, 2):
            raise ValueError(f"row_ids must have shape (C,2), got {self.row_ids.shape}")
        if self.old_rows.ndim != 3 or self.old_rows.shape[:2] != (count, 2):
            raise ValueError(f"old_rows must have shape (C,2,d), got {self.old_rows.shape}")
        if self.new_rows.shape != self.old_rows.shape:
            raise ValueError("new_rows must have the same shape as old_rows")
        if self.row_mask.shape != (count, 2):
            raise ValueError(f"row_mask must have shape (C,2), got {self.row_mask.shape}")
        for name, values in (
            ("operator_ids", self.operator_ids),
            ("target_query_ids", self.target_query_ids),
            ("edit_cost", self.edit_cost),
        ):
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape (C,), got {values.shape}")
        if np.any(np.sum(self.row_mask, axis=1) <= 0):
            raise ValueError("Every candidate unit must contain at least one active row")
        if np.any(self.row_ids[self.row_mask] < 0):
            raise ValueError("Active candidate row IDs must be non-negative")
        if np.any(self.row_ids[~self.row_mask] != -1):
            raise ValueError("Inactive candidate row IDs must be -1")
        both_active = self.row_mask[:, 0] & self.row_mask[:, 1]
        if np.any(self.row_ids[both_active, 0] == self.row_ids[both_active, 1]):
            raise ValueError("Rows within a candidate unit must be distinct")
        changed = np.any(self.old_rows != self.new_rows, axis=2) & self.row_mask
        if np.any(np.sum(changed, axis=1) <= 0):
            raise ValueError("No-op candidate units are not allowed")

    def take(self, indices: np.ndarray) -> "HybridCandidateUnitBatch":
        ids = np.asarray(indices, dtype=np.int64)
        return HybridCandidateUnitBatch(
            row_ids=self.row_ids[ids],
            old_rows=self.old_rows[ids],
            new_rows=self.new_rows[ids],
            row_mask=self.row_mask[ids],
            operator_ids=self.operator_ids[ids],
            target_query_ids=self.target_query_ids[ids],
            edit_cost=self.edit_cost[ids],
        )


@dataclass(frozen=True)
class HybridScoreContext:
    attrs: jax.Array
    ops: jax.Array
    values: jax.Array
    lows: jax.Array
    highs: jax.Array
    linear_attrs: jax.Array
    linear_weights: jax.Array
    linear_thresholds: jax.Array
    linear_num_terms: jax.Array


@dataclass(frozen=True)
class HybridSparseScoreContext:
    qarrays: tuple[jax.Array, ...]
    affected_qids: jax.Array
    real_query_count: int
    num_attrs: int


@dataclass
class AdaptiveSwapAttributePolicy:
    prior: np.ndarray
    reward_sum: np.ndarray
    exposure: np.ndarray
    exploration_floor: float = 0.20
    decay: float = 0.95
    prior_strength: float = 32.0

    @classmethod
    def create(
        cls,
        schema: TableSchema,
        *,
        exploration_floor: float = 0.20,
        decay: float = 0.95,
        prior_strength: float = 32.0,
    ) -> "AdaptiveSwapAttributePolicy":
        floor = float(exploration_floor)
        decay_value = float(decay)
        strength = float(prior_strength)
        if not 0.0 <= floor <= 1.0:
            raise ValueError("exploration_floor must be in [0, 1]")
        if not 0.0 <= decay_value < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("prior_strength must be finite and non-negative")
        prior = _attr_probabilities(schema.cardinalities)
        return cls(
            prior=prior,
            reward_sum=np.zeros(schema.d, dtype=np.float64),
            exposure=np.zeros(schema.d, dtype=np.float64),
            exploration_floor=floor,
            decay=decay_value,
            prior_strength=strength,
        )

    def probabilities(self) -> np.ndarray:
        total_exposure = float(np.sum(self.exposure))
        total_reward = float(np.sum(self.reward_sum))
        if total_exposure <= 0.0 or total_reward <= 0.0:
            return self.prior.copy()
        global_yield = total_reward / total_exposure
        smoothed_yield = (
            self.reward_sum + self.prior_strength * global_yield
        ) / np.maximum(self.exposure + self.prior_strength, 1.0e-12)
        utility = np.sqrt(np.maximum(smoothed_yield, 0.0))
        directed = self.prior * utility
        directed_total = float(np.sum(directed))
        if not np.isfinite(directed_total) or directed_total <= 0.0:
            return self.prior.copy()
        directed /= directed_total
        probabilities = self.exploration_floor * self.prior + (1.0 - self.exploration_floor) * directed
        return probabilities / float(np.sum(probabilities))

    def update(
        self,
        candidates: HybridCandidateUnitBatch,
        scores: np.ndarray,
        *,
        operator_id: int = OPERATOR_GLOBAL_DIRECTED_SWAP,
    ) -> None:
        score_arr = np.asarray(scores, dtype=np.float64)
        if score_arr.shape != (candidates.count,):
            raise ValueError("scores must contain one value per candidate unit")
        self.reward_sum *= self.decay
        self.exposure *= self.decay
        mask = candidates.operator_ids == int(operator_id)
        if not np.any(mask):
            return
        attrs, counts = _unit_changed_attrs(candidates)
        mask &= counts == 1
        attrs = attrs[mask]
        rewards = np.maximum(score_arr[mask], 0.0)
        finite = np.isfinite(rewards)
        attrs = attrs[finite]
        rewards = rewards[finite]
        np.add.at(self.exposure, attrs, 1.0)
        np.add.at(self.reward_sum, attrs, rewards)

    def diagnostics(self) -> dict[str, list[float]]:
        return {
            "probabilities": self.probabilities().astype(float).tolist(),
            "reward_sum": self.reward_sum.astype(float).tolist(),
            "exposure": self.exposure.astype(float).tolist(),
        }


def prepare_hybrid_score_context(qcat: QueryCatalogue) -> HybridScoreContext:
    arrays = qcat.eval_arrays()
    return HybridScoreContext(
        attrs=jnp.asarray(arrays[0], dtype=jnp.int32),
        ops=jnp.asarray(arrays[1], dtype=jnp.int32),
        values=jnp.asarray(arrays[2], dtype=jnp.int32),
        lows=jnp.asarray(arrays[3], dtype=jnp.int32),
        highs=jnp.asarray(arrays[4], dtype=jnp.int32),
        linear_attrs=jnp.asarray(arrays[5], dtype=jnp.int32),
        linear_weights=jnp.asarray(arrays[6], dtype=jnp.float32),
        linear_thresholds=jnp.asarray(arrays[7], dtype=jnp.float32),
        linear_num_terms=jnp.asarray(arrays[8], dtype=jnp.int32),
    )


def prepare_hybrid_sparse_score_context(
    qcat: QueryCatalogue,
    *,
    num_attrs: int,
) -> HybridSparseScoreContext:
    if int(num_attrs) <= 0:
        raise ValueError("num_attrs must be positive")
    delta_index = QueryDeltaIndex.build(qcat, num_attrs=int(num_attrs))
    affected_qids = delta_index.padded_attr_query_ids(num_attrs=int(num_attrs), pad_value=qcat.m)
    if affected_qids.shape[1] == 0:
        affected_qids = np.full((int(num_attrs), 1), qcat.m, dtype=np.int32)
    qarrays_np = (
        np.pad(qcat.attrs, ((0, 1), (0, 0)), constant_values=-1),
        np.pad(qcat.ops, ((0, 1), (0, 0)), constant_values=0),
        np.pad(qcat.values, ((0, 1), (0, 0)), constant_values=0),
        np.pad(qcat.lows, ((0, 1), (0, 0)), constant_values=0),
        np.pad(qcat.highs, ((0, 1), (0, 0)), constant_values=0),
        np.pad(qcat.linear_attrs, ((0, 1), (0, 0)), constant_values=-1),
        np.pad(qcat.linear_weights, ((0, 1), (0, 0)), constant_values=0.0),
        np.pad(qcat.linear_thresholds, (0, 1), constant_values=0.0),
        np.pad(qcat.linear_num_terms, (0, 1), constant_values=0),
    )
    qarrays = tuple(
        jax.device_put(jnp.asarray(values, dtype=jnp.float32 if idx in {6, 7} else jnp.int32))
        for idx, values in enumerate(qarrays_np)
    )
    return HybridSparseScoreContext(
        qarrays=qarrays,
        affected_qids=jax.device_put(jnp.asarray(affected_qids, dtype=jnp.int32)),
        real_query_count=int(qcat.m),
        num_attrs=int(num_attrs),
    )


def _empty_batch(width: int) -> HybridCandidateUnitBatch:
    return HybridCandidateUnitBatch(
        row_ids=np.empty((0, 2), dtype=np.int32),
        old_rows=np.empty((0, 2, width), dtype=np.int32),
        new_rows=np.empty((0, 2, width), dtype=np.int32),
        row_mask=np.empty((0, 2), dtype=bool),
        operator_ids=np.empty(0, dtype=np.int8),
        target_query_ids=np.empty(0, dtype=np.int32),
        edit_cost=np.empty(0, dtype=np.float32),
    )


def concatenate_unit_batches(
    batches: list[HybridCandidateUnitBatch],
    *,
    width: int,
) -> HybridCandidateUnitBatch:
    nonempty = [batch for batch in batches if batch.count > 0]
    if not nonempty:
        return _empty_batch(width)
    for batch in nonempty:
        if batch.width != width:
            raise ValueError(f"Candidate width {batch.width} does not match expected width {width}")
    result = HybridCandidateUnitBatch(
        row_ids=np.concatenate([batch.row_ids for batch in nonempty], axis=0),
        old_rows=np.concatenate([batch.old_rows for batch in nonempty], axis=0),
        new_rows=np.concatenate([batch.new_rows for batch in nonempty], axis=0),
        row_mask=np.concatenate([batch.row_mask for batch in nonempty], axis=0),
        operator_ids=np.concatenate([batch.operator_ids for batch in nonempty], axis=0),
        target_query_ids=np.concatenate([batch.target_query_ids for batch in nonempty], axis=0),
        edit_cost=np.concatenate([batch.edit_cost for batch in nonempty], axis=0),
    )
    result.validate()
    return result


def allocate_operator_counts(total: int, fractions: Mapping[int, float]) -> dict[int, int]:
    total = int(total)
    if total < 0:
        raise ValueError("total must be non-negative")
    operator_ids = sorted(int(operator_id) for operator_id in fractions)
    if not operator_ids:
        raise ValueError("At least one operator fraction is required")
    weights = np.asarray([max(0.0, float(fractions[operator_id])) for operator_id in operator_ids], dtype=np.float64)
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        raise ValueError("Operator fractions must contain a positive finite value")
    weights /= float(np.sum(weights))
    raw = weights * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(np.sum(counts))
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        for idx in order[:remainder].tolist():
            counts[int(idx)] += 1
    return {operator_id: int(count) for operator_id, count in zip(operator_ids, counts.tolist())}


def _attr_probabilities(cardinalities: np.ndarray) -> np.ndarray:
    cards = np.asarray(cardinalities, dtype=np.int64)
    mutable = cards > 1
    weights = np.where(mutable, np.maximum(np.log(np.maximum(cards.astype(np.float64), 2.0)), 1.0), 0.0)
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("Schema has no mutable attributes")
    return weights / float(np.sum(weights))


def _sample_replacement_values(
    old_values: np.ndarray,
    cardinalities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    old = np.asarray(old_values, dtype=np.int32)
    cards = np.asarray(cardinalities, dtype=np.int32)
    out = old.copy()
    for card_raw in np.unique(cards).tolist():
        card = int(card_raw)
        mask = cards == card
        if card <= 1 or not np.any(mask):
            continue
        values = rng.integers(0, card - 1, size=int(np.sum(mask)), dtype=np.int32)
        out[mask] = values + (values >= old[mask])
    return out


def _single_row_unit_batch(
    *,
    row_ids: np.ndarray,
    old_rows: np.ndarray,
    new_rows: np.ndarray,
    operator_id: int,
    target_query_ids: np.ndarray,
    edit_cost: np.ndarray,
) -> HybridCandidateUnitBatch:
    count, width = old_rows.shape
    unit_row_ids = np.full((count, 2), -1, dtype=np.int32)
    unit_row_ids[:, 0] = np.asarray(row_ids, dtype=np.int32)
    unit_old = np.zeros((count, 2, width), dtype=np.int32)
    unit_new = np.zeros((count, 2, width), dtype=np.int32)
    unit_old[:, 0] = np.asarray(old_rows, dtype=np.int32)
    unit_new[:, 0] = np.asarray(new_rows, dtype=np.int32)
    row_mask = np.zeros((count, 2), dtype=bool)
    row_mask[:, 0] = True
    result = HybridCandidateUnitBatch(
        row_ids=unit_row_ids,
        old_rows=unit_old,
        new_rows=unit_new,
        row_mask=row_mask,
        operator_ids=np.full(count, int(operator_id), dtype=np.int8),
        target_query_ids=np.asarray(target_query_ids, dtype=np.int32),
        edit_cost=np.asarray(edit_cost, dtype=np.float32),
    )
    result.validate()
    return result


def generate_random_mutation_units(
    X_syn: np.ndarray,
    schema: TableSchema,
    count: int,
    rng: np.random.Generator,
    *,
    numerical_gamma: float = 0.1,
) -> HybridCandidateUnitBatch:
    count = max(0, int(count))
    if count == 0:
        return _empty_batch(schema.d)
    probabilities = _attr_probabilities(schema.cardinalities)
    row_ids = rng.integers(0, X_syn.shape[0], size=count, dtype=np.int32)
    old_rows = np.asarray(X_syn[row_ids], dtype=np.int32).copy()
    new_rows = old_rows.copy()
    attrs = rng.choice(schema.d, size=count, replace=True, p=probabilities).astype(np.int32)
    row_positions = np.arange(count, dtype=np.int32)
    old_values = old_rows[row_positions, attrs]
    new_rows[row_positions, attrs] = _sample_replacement_values(
        old_values,
        schema.cardinalities[attrs],
        rng,
    )
    edit_cost = compute_edit_cost(old_rows, new_rows, schema, numerical_gamma)
    return _single_row_unit_batch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        operator_id=OPERATOR_MUTATION,
        target_query_ids=np.full(count, -1, dtype=np.int32),
        edit_cost=edit_cost,
    )


def generate_random_swap_units(
    X_syn: np.ndarray,
    schema: TableSchema,
    count: int,
    rng: np.random.Generator,
    *,
    numerical_gamma: float = 0.1,
    max_rounds: int = 8,
    attr_probabilities: np.ndarray | None = None,
    operator_id: int = OPERATOR_SWAP,
) -> HybridCandidateUnitBatch:
    requested = max(0, int(count))
    if requested == 0:
        return _empty_batch(schema.d)
    if X_syn.shape[0] < 2:
        return _empty_batch(schema.d)
    if attr_probabilities is None:
        probabilities = _attr_probabilities(schema.cardinalities)
    else:
        probabilities = np.asarray(attr_probabilities, dtype=np.float64)
        if probabilities.shape != (schema.d,):
            raise ValueError(f"attr_probabilities must have shape ({schema.d},)")
        probabilities = np.where(schema.cardinalities > 1, probabilities, 0.0)
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("attr_probabilities must be finite and non-negative")
        total_probability = float(np.sum(probabilities))
        if total_probability <= 0.0:
            raise ValueError("attr_probabilities must assign positive mass to a mutable attribute")
        probabilities = probabilities / total_probability
    row_id_parts: list[np.ndarray] = []
    old_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    produced = 0
    for _ in range(max(1, int(max_rounds))):
        if produced >= requested:
            break
        sample_count = max(32, 2 * (requested - produced))
        row_ids = rng.integers(0, X_syn.shape[0], size=(sample_count, 2), dtype=np.int32)
        same_row = row_ids[:, 0] == row_ids[:, 1]
        row_ids[same_row, 1] = (row_ids[same_row, 1] + 1) % X_syn.shape[0]
        attrs = rng.choice(schema.d, size=sample_count, replace=True, p=probabilities).astype(np.int32)
        old_rows = np.asarray(X_syn[row_ids], dtype=np.int32).copy()
        positions = np.arange(sample_count, dtype=np.int32)
        left = old_rows[positions, 0, attrs]
        right = old_rows[positions, 1, attrs]
        valid = left != right
        if not np.any(valid):
            continue
        row_ids = row_ids[valid]
        attrs = attrs[valid]
        old_rows = old_rows[valid]
        new_rows = old_rows.copy()
        positions = np.arange(len(row_ids), dtype=np.int32)
        left = old_rows[positions, 0, attrs].copy()
        right = old_rows[positions, 1, attrs].copy()
        new_rows[positions, 0, attrs] = right
        new_rows[positions, 1, attrs] = left
        take = min(requested - produced, len(row_ids))
        row_id_parts.append(row_ids[:take])
        old_parts.append(old_rows[:take])
        new_parts.append(new_rows[:take])
        produced += take
    if produced == 0:
        return _empty_batch(schema.d)
    row_ids = np.concatenate(row_id_parts, axis=0)
    old_rows = np.concatenate(old_parts, axis=0)
    new_rows = np.concatenate(new_parts, axis=0)
    flat_cost = compute_edit_cost(
        old_rows.reshape(-1, schema.d),
        new_rows.reshape(-1, schema.d),
        schema,
        numerical_gamma,
    ).reshape(len(row_ids), 2)
    result = HybridCandidateUnitBatch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        row_mask=np.ones((len(row_ids), 2), dtype=bool),
        operator_ids=np.full(len(row_ids), int(operator_id), dtype=np.int8),
        target_query_ids=np.full(len(row_ids), -1, dtype=np.int32),
        edit_cost=np.sum(flat_cost, axis=1, dtype=np.float32),
    )
    result.validate()
    return result


def global_residual_swap_attr_probabilities(
    qcat: QueryCatalogue,
    schema: TableSchema,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    *,
    exploration_floor: float = 0.10,
) -> np.ndarray:
    """Allocate swap attributes from all released weighted residuals.

    Atomic one-column swaps preserve every one-way count, so only queries with
    at least two active attributes contribute directed pressure.
    """
    residual_arr = np.asarray(residual, dtype=np.float64)
    inv_arr = np.asarray(inv_variance, dtype=np.float64)
    if residual_arr.shape != (qcat.m,) or inv_arr.shape != (qcat.m,):
        raise ValueError("residual and inv_variance must match the query catalogue")
    if not np.all(np.isfinite(residual_arr)) or not np.all(np.isfinite(inv_arr)):
        raise ValueError("residual and inv_variance must be finite")
    if np.any(inv_arr < 0.0):
        raise ValueError("inv_variance must be non-negative")
    floor = float(exploration_floor)
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("exploration_floor must be in [0, 1]")

    pressure = np.zeros(schema.d, dtype=np.float64)
    weighted_magnitude = np.abs(residual_arr * inv_arr)
    for qid in range(qcat.m):
        attrs = {
            int(attr)
            for attr, *_ in qcat.query_terms(qid)
            if 0 <= int(attr) < schema.d
        }
        attrs.update(
            int(attr)
            for attr, _ in qcat.linear_terms(qid)
            if 0 <= int(attr) < schema.d
        )
        if len(attrs) < 2:
            continue
        contribution = float(weighted_magnitude[qid]) / float(len(attrs))
        for attr in attrs:
            pressure[attr] += contribution

    prior = _attr_probabilities(schema.cardinalities)
    pressure = np.where(schema.cardinalities > 1, pressure, 0.0)
    pressure_total = float(np.sum(pressure))
    if pressure_total <= 0.0:
        return prior
    directed = pressure / pressure_total
    probabilities = floor * prior + (1.0 - floor) * directed
    probabilities = np.where(schema.cardinalities > 1, probabilities, 0.0)
    return probabilities / float(np.sum(probabilities))


def generate_global_directed_swap_units(
    X_syn: np.ndarray,
    qcat: QueryCatalogue,
    schema: TableSchema,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    exploration_floor: float = 0.10,
    attr_probabilities: np.ndarray | None = None,
    numerical_gamma: float = 0.1,
    max_rounds: int = 8,
) -> HybridCandidateUnitBatch:
    probabilities = (
        global_residual_swap_attr_probabilities(
            qcat,
            schema,
            residual,
            inv_variance,
            exploration_floor=exploration_floor,
        )
        if attr_probabilities is None
        else np.asarray(attr_probabilities, dtype=np.float64)
    )
    return generate_random_swap_units(
        X_syn,
        schema,
        count,
        rng,
        numerical_gamma=numerical_gamma,
        max_rounds=max_rounds,
        attr_probabilities=probabilities,
        operator_id=OPERATOR_GLOBAL_DIRECTED_SWAP,
    )


def generate_random_crossover_units(
    X_syn: np.ndarray,
    schema: TableSchema,
    count: int,
    rng: np.random.Generator,
    *,
    numerical_gamma: float = 0.1,
    max_rounds: int = 8,
) -> HybridCandidateUnitBatch:
    requested = max(0, int(count))
    if requested == 0:
        return _empty_batch(schema.d)
    categorical = np.asarray(schema.categorical_indices, dtype=np.int32)
    categorical = categorical[schema.cardinalities[categorical] > 1] if len(categorical) else categorical
    if len(categorical) == 0:
        categorical = np.flatnonzero(schema.cardinalities > 1).astype(np.int32)
    if len(categorical) == 0:
        return _empty_batch(schema.d)
    row_id_parts: list[np.ndarray] = []
    old_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    produced = 0
    for _ in range(max(1, int(max_rounds))):
        if produced >= requested:
            break
        sample_count = max(32, 2 * (requested - produced))
        recipient_ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
        donor_ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
        if X_syn.shape[0] > 1:
            same = recipient_ids == donor_ids
            donor_ids[same] = (donor_ids[same] + 1) % X_syn.shape[0]
        old_rows = np.asarray(X_syn[recipient_ids], dtype=np.int32).copy()
        donor_rows = np.asarray(X_syn[donor_ids], dtype=np.int32)
        new_rows = old_rows.copy()
        max_updates = min(2, len(categorical))
        update_counts = rng.integers(1, max_updates + 1, size=sample_count, dtype=np.int32)
        for idx, update_count in enumerate(update_counts.tolist()):
            attrs = rng.choice(categorical, size=int(update_count), replace=False)
            new_rows[idx, attrs] = donor_rows[idx, attrs]
        valid = np.any(old_rows != new_rows, axis=1)
        if not np.any(valid):
            continue
        recipient_ids = recipient_ids[valid]
        old_rows = old_rows[valid]
        new_rows = new_rows[valid]
        take = min(requested - produced, len(recipient_ids))
        row_id_parts.append(recipient_ids[:take])
        old_parts.append(old_rows[:take])
        new_parts.append(new_rows[:take])
        produced += take
    if produced == 0:
        return _empty_batch(schema.d)
    row_ids = np.concatenate(row_id_parts)
    old_rows = np.concatenate(old_parts, axis=0)
    new_rows = np.concatenate(new_parts, axis=0)
    edit_cost = compute_edit_cost(old_rows, new_rows, schema, numerical_gamma)
    return _single_row_unit_batch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        operator_id=OPERATOR_CROSSOVER,
        target_query_ids=np.full(len(row_ids), -1, dtype=np.int32),
        edit_cost=edit_cost,
    )


def generate_directed_single_units(
    X_syn: np.ndarray,
    qcat: QueryCatalogue,
    schema: TableSchema,
    target_query_ids: np.ndarray,
    residual: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    inv_variance: np.ndarray | None = None,
    numerical_gamma: float = 0.1,
    source_over_sample_factor: int = 32,
) -> HybridCandidateUnitBatch:
    requested = max(0, int(count))
    if requested == 0:
        return _empty_batch(schema.d)
    active = np.asarray(target_query_ids, dtype=np.int32)
    active = active[(active >= 0) & (active < qcat.m)]
    if len(active) == 0:
        return _empty_batch(schema.d)
    candidates_per_target = max(1, int(np.ceil(requested / len(active))))
    config = {
        "qdte": {
            "candidate_compiler": "single_query",
            "candidates_per_target": candidates_per_target,
            "total_candidates_per_iter": requested,
            "directed_candidate_count": requested,
            "random_candidate_count": 0,
            "candidate_shortfall_policy": "none",
            "source_over_sample_factor": int(source_over_sample_factor),
            "numerical_distance_gamma": float(numerical_gamma),
        }
    }
    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        active,
        residual,
        config,
        rng,
        inv_variance=inv_variance,
    )
    if len(candidates.row_ids) == 0:
        return _empty_batch(schema.d)
    return _single_row_unit_batch(
        row_ids=candidates.row_ids,
        old_rows=candidates.old_rows,
        new_rows=candidates.new_rows,
        operator_id=OPERATOR_DIRECTED,
        target_query_ids=candidates.target_query_ids,
        edit_cost=candidates.edit_cost,
    )


def _sample_residual_query_ids(
    target_query_ids: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray | None,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    active = np.asarray(target_query_ids, dtype=np.int32)
    active = active[(active >= 0) & (active < len(residual))]
    if len(active) == 0 or count <= 0:
        return np.empty(0, dtype=np.int32)
    weights = np.abs(np.asarray(residual, dtype=np.float64)[active])
    if inv_variance is not None:
        weights *= np.asarray(inv_variance, dtype=np.float64)[active]
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return rng.choice(active, size=count, replace=True).astype(np.int32, copy=False)
    probabilities = weights / total
    return rng.choice(active, size=count, replace=True, p=probabilities).astype(np.int32, copy=False)


def _target_query_unit_deltas(
    old_rows: np.ndarray,
    new_rows: np.ndarray,
    target_query_ids: np.ndarray,
    qcat: QueryCatalogue,
) -> np.ndarray:
    old = np.asarray(old_rows, dtype=np.int32)
    new = np.asarray(new_rows, dtype=np.int32)
    qids = np.asarray(target_query_ids, dtype=np.int32)
    if old.shape != new.shape or old.ndim != 3 or old.shape[0] != len(qids):
        raise ValueError("Target-query unit delta inputs have incompatible shapes")
    out = np.zeros(len(qids), dtype=np.int8)
    for qid_raw in np.unique(qids).tolist():
        qid = int(qid_raw)
        positions = np.flatnonzero(qids == qid)
        old_sat = qcat.eval_query_np(old[positions].reshape(-1, old.shape[2]), qid).reshape(len(positions), old.shape[1])
        new_sat = qcat.eval_query_np(new[positions].reshape(-1, new.shape[2]), qid).reshape(len(positions), new.shape[1])
        out[positions] = np.sum(new_sat.astype(np.int8) - old_sat.astype(np.int8), axis=1)
    return out


def _ordinary_mutable_attrs_by_query(
    qcat: QueryCatalogue,
    schema: TableSchema,
    target_query_ids: np.ndarray,
    *,
    categorical_only: bool,
) -> dict[int, np.ndarray]:
    categorical = set(schema.categorical_indices)
    result: dict[int, np.ndarray] = {}
    for qid_raw in np.asarray(target_query_ids, dtype=np.int32).tolist():
        qid = int(qid_raw)
        if qid < 0 or qid >= qcat.m:
            continue
        attrs = {
            int(attr)
            for attr, *_ in qcat.query_terms(qid)
            if int(schema.cardinalities[int(attr)]) > 1
            and (not categorical_only or int(attr) in categorical)
        }
        if attrs:
            result[qid] = np.asarray(sorted(attrs), dtype=np.int32)
    return result


def generate_directed_swap_units(
    X_syn: np.ndarray,
    qcat: QueryCatalogue,
    schema: TableSchema,
    target_query_ids: np.ndarray,
    residual: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    inv_variance: np.ndarray | None = None,
    numerical_gamma: float = 0.1,
    over_sample_factor: int = 32,
    max_rounds: int = 8,
) -> HybridCandidateUnitBatch:
    requested = max(0, int(count))
    if requested == 0 or X_syn.shape[0] < 2:
        return _empty_batch(schema.d)
    attr_map = _ordinary_mutable_attrs_by_query(
        qcat,
        schema,
        target_query_ids,
        categorical_only=False,
    )
    eligible_qids = np.asarray(
        [qid for qid in attr_map if float(residual[qid]) != 0.0],
        dtype=np.int32,
    )
    if len(eligible_qids) == 0:
        return _empty_batch(schema.d)

    row_id_parts: list[np.ndarray] = []
    old_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    qid_parts: list[np.ndarray] = []
    produced = 0
    for _ in range(max(1, int(max_rounds))):
        if produced >= requested:
            break
        sample_count = max(64, (requested - produced) * max(1, int(over_sample_factor)))
        qids = _sample_residual_query_ids(eligible_qids, residual, inv_variance, sample_count, rng)
        row_ids = rng.integers(0, X_syn.shape[0], size=(sample_count, 2), dtype=np.int32)
        same_row = row_ids[:, 0] == row_ids[:, 1]
        row_ids[same_row, 1] = (row_ids[same_row, 1] + 1) % X_syn.shape[0]
        attrs = np.asarray([rng.choice(attr_map[int(qid)]) for qid in qids.tolist()], dtype=np.int32)
        old_rows = np.asarray(X_syn[row_ids], dtype=np.int32).copy()
        positions = np.arange(sample_count, dtype=np.int32)
        left = old_rows[positions, 0, attrs]
        right = old_rows[positions, 1, attrs]
        changed = left != right
        if not np.any(changed):
            continue
        row_ids = row_ids[changed]
        qids = qids[changed]
        attrs = attrs[changed]
        old_rows = old_rows[changed]
        new_rows = old_rows.copy()
        positions = np.arange(len(row_ids), dtype=np.int32)
        left = old_rows[positions, 0, attrs].copy()
        right = old_rows[positions, 1, attrs].copy()
        new_rows[positions, 0, attrs] = right
        new_rows[positions, 1, attrs] = left
        target_delta = _target_query_unit_deltas(old_rows, new_rows, qids, qcat)
        direction_ok = np.asarray(residual, dtype=np.float64)[qids] * target_delta.astype(np.float64) > 0.0
        if not np.any(direction_ok):
            continue
        row_ids = row_ids[direction_ok]
        old_rows = old_rows[direction_ok]
        new_rows = new_rows[direction_ok]
        qids = qids[direction_ok]
        take = min(requested - produced, len(row_ids))
        row_id_parts.append(row_ids[:take])
        old_parts.append(old_rows[:take])
        new_parts.append(new_rows[:take])
        qid_parts.append(qids[:take])
        produced += take
    if produced == 0:
        return _empty_batch(schema.d)
    row_ids = np.concatenate(row_id_parts, axis=0)
    old_rows = np.concatenate(old_parts, axis=0)
    new_rows = np.concatenate(new_parts, axis=0)
    qids = np.concatenate(qid_parts, axis=0)
    flat_cost = compute_edit_cost(
        old_rows.reshape(-1, schema.d),
        new_rows.reshape(-1, schema.d),
        schema,
        numerical_gamma,
    ).reshape(len(row_ids), 2)
    result = HybridCandidateUnitBatch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        row_mask=np.ones((len(row_ids), 2), dtype=bool),
        operator_ids=np.full(len(row_ids), OPERATOR_DIRECTED_SWAP, dtype=np.int8),
        target_query_ids=qids,
        edit_cost=np.sum(flat_cost, axis=1, dtype=np.float32),
    )
    result.validate()
    return result


def generate_directed_crossover_units(
    X_syn: np.ndarray,
    qcat: QueryCatalogue,
    schema: TableSchema,
    target_query_ids: np.ndarray,
    residual: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    inv_variance: np.ndarray | None = None,
    numerical_gamma: float = 0.1,
    over_sample_factor: int = 32,
    max_rounds: int = 8,
) -> HybridCandidateUnitBatch:
    requested = max(0, int(count))
    if requested == 0 or X_syn.shape[0] < 2:
        return _empty_batch(schema.d)
    attr_map = _ordinary_mutable_attrs_by_query(
        qcat,
        schema,
        target_query_ids,
        categorical_only=True,
    )
    eligible_qids = np.asarray(
        [qid for qid in attr_map if float(residual[qid]) != 0.0],
        dtype=np.int32,
    )
    if len(eligible_qids) == 0:
        return _empty_batch(schema.d)

    row_id_parts: list[np.ndarray] = []
    old_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    qid_parts: list[np.ndarray] = []
    produced = 0
    for _ in range(max(1, int(max_rounds))):
        if produced >= requested:
            break
        sample_count = max(64, (requested - produced) * max(1, int(over_sample_factor)))
        qids = _sample_residual_query_ids(eligible_qids, residual, inv_variance, sample_count, rng)
        recipient_ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
        donor_ids = rng.integers(0, X_syn.shape[0], size=sample_count, dtype=np.int32)
        same_row = recipient_ids == donor_ids
        donor_ids[same_row] = (donor_ids[same_row] + 1) % X_syn.shape[0]
        old_rows = np.asarray(X_syn[recipient_ids], dtype=np.int32).copy()
        donor_rows = np.asarray(X_syn[donor_ids], dtype=np.int32)
        new_rows = old_rows.copy()
        for idx, qid_raw in enumerate(qids.tolist()):
            attrs = attr_map[int(qid_raw)]
            update_count = 1 if len(attrs) == 1 else int(rng.integers(1, min(2, len(attrs)) + 1))
            selected = rng.choice(attrs, size=update_count, replace=False)
            new_rows[idx, selected] = donor_rows[idx, selected]
        changed = np.any(old_rows != new_rows, axis=1)
        if not np.any(changed):
            continue
        recipient_ids = recipient_ids[changed]
        qids = qids[changed]
        old_rows = old_rows[changed]
        new_rows = new_rows[changed]
        target_delta = _target_query_unit_deltas(
            old_rows[:, None, :],
            new_rows[:, None, :],
            qids,
            qcat,
        )
        direction_ok = np.asarray(residual, dtype=np.float64)[qids] * target_delta.astype(np.float64) > 0.0
        if not np.any(direction_ok):
            continue
        recipient_ids = recipient_ids[direction_ok]
        qids = qids[direction_ok]
        old_rows = old_rows[direction_ok]
        new_rows = new_rows[direction_ok]
        take = min(requested - produced, len(recipient_ids))
        row_id_parts.append(recipient_ids[:take])
        old_parts.append(old_rows[:take])
        new_parts.append(new_rows[:take])
        qid_parts.append(qids[:take])
        produced += take
    if produced == 0:
        return _empty_batch(schema.d)
    row_ids = np.concatenate(row_id_parts, axis=0)
    old_rows = np.concatenate(old_parts, axis=0)
    new_rows = np.concatenate(new_parts, axis=0)
    qids = np.concatenate(qid_parts, axis=0)
    edit_cost = compute_edit_cost(old_rows, new_rows, schema, numerical_gamma)
    return _single_row_unit_batch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        operator_id=OPERATOR_DIRECTED_CROSSOVER,
        target_query_ids=qids,
        edit_cost=edit_cost,
    )


@jax.jit
def _score_candidate_units_impl(
    old_rows: jax.Array,
    new_rows: jax.Array,
    row_mask: jax.Array,
    edit_cost: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
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
) -> jax.Array:
    count, arity, width = old_rows.shape
    old_flat = old_rows.reshape(count * arity, width)
    new_flat = new_rows.reshape(count * arity, width)
    phi_old = eval_records_queries_arrays(
        old_flat,
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
        new_flat,
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
    row_delta = phi_new.astype(jnp.float32) - phi_old.astype(jnp.float32)
    row_delta = row_delta.reshape(count, arity, attrs.shape[0])
    delta = jnp.sum(row_delta * row_mask[:, :, None].astype(jnp.float32), axis=1)
    weighted_residual = residual.astype(jnp.float32) * inv_variance.astype(jnp.float32)
    linear = delta @ weighted_residual
    quadratic = 0.5 * ((delta * delta) @ inv_variance.astype(jnp.float32))
    scores = linear - quadratic - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)
    changed = jnp.any((old_rows != new_rows) & row_mask[:, :, None], axis=(1, 2))
    return jnp.where(changed, scores, jnp.asarray(-jnp.inf, dtype=jnp.float32))


@partial(jax.pmap, in_axes=(0, 0, 0, 0, None, None, None, None))
def _score_candidate_units_pmap(
    old_rows: jax.Array,
    new_rows: jax.Array,
    row_mask: jax.Array,
    edit_cost: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    qarrays: tuple[jax.Array, ...],
    lambda_cost: jax.Array,
) -> jax.Array:
    return _score_candidate_units_impl(
        old_rows,
        new_rows,
        row_mask,
        edit_cost,
        residual,
        inv_variance,
        *qarrays,
        lambda_cost,
    )


def _eval_rows_query_grid(
    rows: jax.Array,
    qids: jax.Array,
    qarrays: tuple[jax.Array, ...],
) -> jax.Array:
    (
        attrs,
        ops,
        values,
        lows,
        highs,
        linear_attrs,
        linear_weights,
        linear_thresholds,
        linear_num_terms,
    ) = qarrays
    row_idx = jnp.arange(rows.shape[0], dtype=jnp.int32)[:, None]
    q_attrs = attrs[qids]
    q_ops = ops[qids]
    q_values = values[qids]
    q_lows = lows[qids]
    q_highs = highs[qids]
    satisfied = jnp.ones(qids.shape, dtype=jnp.bool_)
    for term in range(attrs.shape[1]):
        attr = q_attrs[:, :, term]
        valid = attr >= 0
        xvals = rows[row_idx, jnp.maximum(attr, 0)]
        op = q_ops[:, :, term]
        value = q_values[:, :, term]
        lo = q_lows[:, :, term]
        hi = q_highs[:, :, term]
        cond = jnp.where(op == 0, xvals == value, xvals <= value)
        cond = jnp.where(op == 2, xvals >= value, cond)
        cond = jnp.where(op == 3, (xvals >= lo) & (xvals <= hi), cond)
        satisfied = satisfied & jnp.where(valid, cond, True)

    q_linear_attrs = linear_attrs[qids]
    q_linear_weights = linear_weights[qids]
    linear_scores = jnp.zeros(qids.shape, dtype=jnp.float32)
    for term in range(linear_attrs.shape[1]):
        attr = q_linear_attrs[:, :, term]
        valid = attr >= 0
        xvals = rows[row_idx, jnp.maximum(attr, 0)].astype(jnp.float32)
        linear_scores = linear_scores + jnp.where(valid, xvals * q_linear_weights[:, :, term], 0.0)
    linear_valid = linear_num_terms[qids] > 0
    linear_cond = linear_scores <= linear_thresholds[qids].astype(jnp.float32)
    return satisfied & jnp.where(linear_valid, linear_cond, True)


def _score_one_attr_candidate_units_sparse_impl(
    old_rows: jax.Array,
    new_rows: jax.Array,
    row_mask: jax.Array,
    edit_cost: jax.Array,
    changed_attrs: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    qarrays: tuple[jax.Array, ...],
    affected_qids: jax.Array,
    lambda_cost: jax.Array,
) -> jax.Array:
    count, arity, width = old_rows.shape
    qids = affected_qids[changed_attrs]
    row_qids = jnp.repeat(qids, repeats=arity, axis=0)
    old_sat = _eval_rows_query_grid(old_rows.reshape(count * arity, width), row_qids, qarrays)
    new_sat = _eval_rows_query_grid(new_rows.reshape(count * arity, width), row_qids, qarrays)
    row_delta = new_sat.astype(jnp.float32) - old_sat.astype(jnp.float32)
    row_delta = row_delta.reshape(count, arity, qids.shape[1])
    delta = jnp.sum(row_delta * row_mask[:, :, None].astype(jnp.float32), axis=1)
    weighted_residual = residual[qids].astype(jnp.float32) * inv_variance[qids].astype(jnp.float32)
    linear = jnp.sum(delta * weighted_residual, axis=1)
    quadratic = 0.5 * jnp.sum(delta * delta * inv_variance[qids].astype(jnp.float32), axis=1)
    scores = linear - quadratic - lambda_cost.astype(jnp.float32) * edit_cost.astype(jnp.float32)
    changed = jnp.any((old_rows != new_rows) & row_mask[:, :, None], axis=(1, 2))
    return jnp.where(changed, scores, jnp.asarray(-jnp.inf, dtype=jnp.float32))


_score_one_attr_candidate_units_sparse_jit = jax.jit(_score_one_attr_candidate_units_sparse_impl)


@partial(jax.pmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))
def _score_one_attr_candidate_units_sparse_pmap(
    old_rows: jax.Array,
    new_rows: jax.Array,
    row_mask: jax.Array,
    edit_cost: jax.Array,
    changed_attrs: jax.Array,
    residual: jax.Array,
    inv_variance: jax.Array,
    qarrays: tuple[jax.Array, ...],
    affected_qids: jax.Array,
    lambda_cost: jax.Array,
) -> jax.Array:
    return _score_one_attr_candidate_units_sparse_impl(
        old_rows,
        new_rows,
        row_mask,
        edit_cost,
        changed_attrs,
        residual,
        inv_variance,
        qarrays,
        affected_qids,
        lambda_cost,
    )


def _unit_changed_attrs(candidates: HybridCandidateUnitBatch) -> tuple[np.ndarray, np.ndarray]:
    changed = np.any(
        (candidates.old_rows != candidates.new_rows) & candidates.row_mask[:, :, None],
        axis=1,
    )
    counts = np.sum(changed, axis=1)
    attrs = np.argmax(changed, axis=1).astype(np.int32, copy=False)
    return attrs, counts


def score_one_attr_candidate_units_sparse(
    candidates: HybridCandidateUnitBatch,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    context: HybridSparseScoreContext,
    *,
    lambda_cost: float = 0.0,
    chunk_size: int = 4096,
    use_pmap: bool = True,
) -> np.ndarray:
    candidates.validate()
    if candidates.count == 0:
        return np.empty(0, dtype=np.float32)
    if candidates.width != context.num_attrs:
        raise ValueError("Candidate width does not match sparse score context")
    changed_attrs, changed_counts = _unit_changed_attrs(candidates)
    if np.any(changed_counts != 1):
        raise ValueError("Sparse one-attribute scoring requires exactly one changed attribute per unit")
    residual_arr = np.asarray(residual, dtype=np.float32)
    inv_arr = np.asarray(inv_variance, dtype=np.float32)
    if residual_arr.shape != (context.real_query_count,) or inv_arr.shape != (context.real_query_count,):
        raise ValueError("Objective vectors do not match sparse score context")
    residual_padded = jnp.asarray(np.pad(residual_arr, (0, 1)), dtype=jnp.float32)
    inv_padded = jnp.asarray(np.pad(inv_arr, (0, 1)), dtype=jnp.float32)
    lambda_j = jnp.asarray(lambda_cost, dtype=jnp.float32)
    chunk_size = max(1, int(chunk_size))
    ndev = max(1, jax.local_device_count())
    can_pmap = bool(use_pmap and ndev > 1)
    outputs: list[np.ndarray] = []
    for start in range(0, candidates.count, chunk_size):
        end = min(start + chunk_size, candidates.count)
        old = candidates.old_rows[start:end]
        new = candidates.new_rows[start:end]
        mask = candidates.row_mask[start:end]
        cost = candidates.edit_cost[start:end]
        attrs = changed_attrs[start:end]
        if can_pmap and len(old) >= ndev:
            pad = (-len(old)) % ndev
            if pad:
                old = np.concatenate([old, np.repeat(old[-1:], pad, axis=0)], axis=0)
                new = np.concatenate([new, np.repeat(new[-1:], pad, axis=0)], axis=0)
                mask = np.concatenate([mask, np.repeat(mask[-1:], pad, axis=0)], axis=0)
                cost = np.concatenate([cost, np.repeat(cost[-1:], pad, axis=0)], axis=0)
                attrs = np.concatenate([attrs, np.repeat(attrs[-1:], pad, axis=0)], axis=0)
            per_device = len(old) // ndev
            scores = _score_one_attr_candidate_units_sparse_pmap(
                jnp.asarray(old.reshape(ndev, per_device, 2, candidates.width), dtype=jnp.int32),
                jnp.asarray(new.reshape(ndev, per_device, 2, candidates.width), dtype=jnp.int32),
                jnp.asarray(mask.reshape(ndev, per_device, 2), dtype=jnp.bool_),
                jnp.asarray(cost.reshape(ndev, per_device), dtype=jnp.float32),
                jnp.asarray(attrs.reshape(ndev, per_device), dtype=jnp.int32),
                residual_padded,
                inv_padded,
                context.qarrays,
                context.affected_qids,
                lambda_j,
            )
            outputs.append(np.asarray(scores).reshape(-1)[: end - start].astype(np.float32, copy=False))
        else:
            scores = _score_one_attr_candidate_units_sparse_jit(
                jnp.asarray(old, dtype=jnp.int32),
                jnp.asarray(new, dtype=jnp.int32),
                jnp.asarray(mask, dtype=jnp.bool_),
                jnp.asarray(cost, dtype=jnp.float32),
                jnp.asarray(attrs, dtype=jnp.int32),
                residual_padded,
                inv_padded,
                context.qarrays,
                context.affected_qids,
                lambda_j,
            )
            outputs.append(np.asarray(scores, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def score_candidate_units(
    candidates: HybridCandidateUnitBatch,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    context: HybridScoreContext,
    *,
    lambda_cost: float = 0.0,
    chunk_size: int = 4096,
    use_pmap: bool = False,
) -> np.ndarray:
    candidates.validate()
    if candidates.count == 0:
        return np.empty(0, dtype=np.float32)
    qarrays = (
        context.attrs,
        context.ops,
        context.values,
        context.lows,
        context.highs,
        context.linear_attrs,
        context.linear_weights,
        context.linear_thresholds,
        context.linear_num_terms,
    )
    residual_j = jnp.asarray(residual, dtype=jnp.float32)
    inv_j = jnp.asarray(inv_variance, dtype=jnp.float32)
    lambda_j = jnp.asarray(lambda_cost, dtype=jnp.float32)
    ndev = max(1, jax.local_device_count())
    can_pmap = bool(use_pmap and ndev > 1)
    chunk_size = max(1, int(chunk_size))
    outputs: list[np.ndarray] = []
    for start in range(0, candidates.count, chunk_size):
        end = min(start + chunk_size, candidates.count)
        old = candidates.old_rows[start:end]
        new = candidates.new_rows[start:end]
        mask = candidates.row_mask[start:end]
        cost = candidates.edit_cost[start:end]
        if can_pmap and len(old) >= ndev:
            pad = (-len(old)) % ndev
            if pad:
                old = np.concatenate([old, np.repeat(old[-1:], pad, axis=0)], axis=0)
                new = np.concatenate([new, np.repeat(new[-1:], pad, axis=0)], axis=0)
                mask = np.concatenate([mask, np.repeat(mask[-1:], pad, axis=0)], axis=0)
                cost = np.concatenate([cost, np.repeat(cost[-1:], pad, axis=0)], axis=0)
            per_device = len(old) // ndev
            scores = _score_candidate_units_pmap(
                jnp.asarray(old.reshape(ndev, per_device, 2, candidates.width), dtype=jnp.int32),
                jnp.asarray(new.reshape(ndev, per_device, 2, candidates.width), dtype=jnp.int32),
                jnp.asarray(mask.reshape(ndev, per_device, 2), dtype=jnp.bool_),
                jnp.asarray(cost.reshape(ndev, per_device), dtype=jnp.float32),
                residual_j,
                inv_j,
                qarrays,
                lambda_j,
            )
            outputs.append(np.asarray(scores).reshape(-1)[: end - start].astype(np.float32, copy=False))
        else:
            scores = _score_candidate_units_impl(
                jnp.asarray(old, dtype=jnp.int32),
                jnp.asarray(new, dtype=jnp.int32),
                jnp.asarray(mask, dtype=jnp.bool_),
                jnp.asarray(cost, dtype=jnp.float32),
                residual_j,
                inv_j,
                *qarrays,
                lambda_j,
            )
            outputs.append(np.asarray(scores, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def score_candidate_units_optimized(
    candidates: HybridCandidateUnitBatch,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    dense_context: HybridScoreContext,
    sparse_context: HybridSparseScoreContext,
    *,
    lambda_cost: float = 0.0,
    chunk_size: int = 4096,
    use_pmap: bool = True,
) -> np.ndarray:
    """Score one-attribute units sparsely and fall back to dense exact scoring."""
    candidates.validate()
    if candidates.count == 0:
        return np.empty(0, dtype=np.float32)
    _, changed_counts = _unit_changed_attrs(candidates)
    sparse_mask = changed_counts == 1
    scores = np.full(candidates.count, -np.inf, dtype=np.float32)
    if np.any(sparse_mask):
        sparse_indices = np.flatnonzero(sparse_mask)
        scores[sparse_indices] = score_one_attr_candidate_units_sparse(
            candidates.take(sparse_indices),
            residual,
            inv_variance,
            sparse_context,
            lambda_cost=lambda_cost,
            chunk_size=chunk_size,
            use_pmap=use_pmap,
        )
    if np.any(~sparse_mask):
        dense_indices = np.flatnonzero(~sparse_mask)
        scores[dense_indices] = score_candidate_units(
            candidates.take(dense_indices),
            residual,
            inv_variance,
            dense_context,
            lambda_cost=lambda_cost,
            chunk_size=chunk_size,
            use_pmap=use_pmap,
        )
    return scores


def candidate_unit_delta(
    candidates: HybridCandidateUnitBatch,
    index: int,
    qcat: QueryCatalogue,
) -> np.ndarray:
    idx = int(index)
    if idx < 0 or idx >= candidates.count:
        raise IndexError(idx)
    mask = candidates.row_mask[idx]
    old_rows = candidates.old_rows[idx, mask]
    new_rows = candidates.new_rows[idx, mask]
    phi_old = np.asarray(eval_records_queries(old_rows, qcat), dtype=np.float32)
    phi_new = np.asarray(eval_records_queries(new_rows, qcat), dtype=np.float32)
    return np.sum(phi_new - phi_old, axis=0, dtype=np.float32)


@jax.jit
def _candidate_unit_deltas_impl(
    old_rows: jax.Array,
    new_rows: jax.Array,
    row_mask: jax.Array,
    attrs: jax.Array,
    ops: jax.Array,
    values: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    linear_attrs: jax.Array,
    linear_weights: jax.Array,
    linear_thresholds: jax.Array,
    linear_num_terms: jax.Array,
) -> jax.Array:
    count, arity, width = old_rows.shape
    old_sat = eval_records_queries_arrays(
        old_rows.reshape(count * arity, width),
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
    new_sat = eval_records_queries_arrays(
        new_rows.reshape(count * arity, width),
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
    row_delta = new_sat.astype(jnp.int8) - old_sat.astype(jnp.int8)
    row_delta = row_delta.reshape(count, arity, attrs.shape[0])
    return jnp.sum(row_delta * row_mask[:, :, None].astype(jnp.int8), axis=1).astype(jnp.int8)


def candidate_unit_deltas(
    candidates: HybridCandidateUnitBatch,
    context: HybridScoreContext,
) -> np.ndarray:
    candidates.validate()
    if candidates.count == 0:
        return np.empty((0, int(context.attrs.shape[0])), dtype=np.int8)
    deltas = _candidate_unit_deltas_impl(
        jnp.asarray(candidates.old_rows, dtype=jnp.int32),
        jnp.asarray(candidates.new_rows, dtype=jnp.int32),
        jnp.asarray(candidates.row_mask, dtype=jnp.bool_),
        context.attrs,
        context.ops,
        context.values,
        context.lows,
        context.highs,
        context.linear_attrs,
        context.linear_weights,
        context.linear_thresholds,
        context.linear_num_terms,
    )
    return np.asarray(deltas, dtype=np.int8)


def candidate_unit_feature_deltas(
    candidates: HybridCandidateUnitBatch,
    precision: OrthogonalInteractionPrecision,
) -> np.ndarray:
    """Return aggregate raw orthogonal feature deltas for candidate units."""
    candidates.validate()
    if candidates.width != len(precision.cardinalities):
        raise ValueError("Candidate width must match orthogonal precision cardinalities")
    if candidates.count == 0:
        return np.empty((0, precision.coefficient_dimension), dtype=np.float64)
    row_deltas = precision.row_feature_deltas(
        candidates.old_rows.reshape(-1, candidates.width),
        candidates.new_rows.reshape(-1, candidates.width),
    ).reshape(candidates.count, 2, precision.coefficient_dimension)
    return np.sum(
        row_deltas * candidates.row_mask[:, :, None].astype(np.float64),
        axis=1,
        dtype=np.float64,
    )


def score_candidate_units_precision(
    candidates: HybridCandidateUnitBatch,
    residual: np.ndarray,
    precision: OrthogonalInteractionPrecision,
    *,
    lambda_cost: float = 0.0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Score multi-row units by the exact raw orthogonal finite difference."""
    candidates.validate()
    if candidates.count == 0:
        return np.empty(0, dtype=np.float32)
    size = max(1, int(chunk_size))
    outputs: list[np.ndarray] = []
    for start in range(0, candidates.count, size):
        end = min(start + size, candidates.count)
        indices = np.arange(start, end, dtype=np.int64)
        chunk = candidates.take(indices)
        feature_deltas = candidate_unit_feature_deltas(chunk, precision)
        outputs.append(
            precision.feature_advantages(
                residual,
                feature_deltas,
                chunk.edit_cost,
                float(lambda_cost),
            ).astype(np.float32)
        )
    return np.concatenate(outputs, axis=0)


def score_candidate_units_l1(
    candidates: HybridCandidateUnitBatch,
    residual: np.ndarray,
    objective_weights: np.ndarray,
    context: HybridScoreContext,
    *,
    lambda_cost: float = 0.0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Score multi-row units by their exact weighted L1 finite difference."""
    candidates.validate()
    if candidates.count == 0:
        return np.empty(0, dtype=np.float32)
    residual_arr = np.asarray(residual, dtype=np.float32)
    weights = np.asarray(objective_weights, dtype=np.float32)
    if residual_arr.shape != weights.shape or residual_arr.shape != (int(context.attrs.shape[0]),):
        raise ValueError("L1 objective vectors must match the score context")
    size = max(1, int(chunk_size))
    outputs: list[np.ndarray] = []
    before = np.abs(residual_arr)
    for start in range(0, candidates.count, size):
        end = min(start + size, candidates.count)
        chunk = candidates.take(np.arange(start, end, dtype=np.int64))
        deltas = candidate_unit_deltas(chunk, context).astype(np.float32)
        gain = np.sum(
            (before[None, :] - np.abs(residual_arr[None, :] - deltas))
            * weights[None, :],
            axis=1,
            dtype=np.float32,
        )
        gain -= float(lambda_cost) * chunk.edit_cost.astype(np.float32)
        outputs.append(gain.astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def exact_unit_advantage(
    delta: np.ndarray,
    residual: np.ndarray,
    inv_variance: np.ndarray,
    *,
    lambda_cost: float = 0.0,
    edit_cost: float = 0.0,
) -> float:
    delta64 = np.asarray(delta, dtype=np.float64)
    residual64 = np.asarray(residual, dtype=np.float64)
    inv64 = np.asarray(inv_variance, dtype=np.float64)
    return float(
        delta64 @ (residual64 * inv64)
        - 0.5 * ((delta64 * delta64) @ inv64)
        - float(lambda_cost) * float(edit_cost)
    )


def apply_candidate_unit(
    X_syn: np.ndarray,
    candidates: HybridCandidateUnitBatch,
    index: int,
) -> None:
    idx = int(index)
    mask = candidates.row_mask[idx]
    row_ids = candidates.row_ids[idx, mask]
    X_syn[row_ids] = candidates.new_rows[idx, mask]
