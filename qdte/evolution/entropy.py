from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from qdte.evolution.confidence import ChiSquareDiscrepancyStop
from qdte.measurement.projection import project_simplex
from qdte.queries.types import OP_EQ, QueryCatalogue


PRODUCT_PRIOR_METHOD = "released_oneway_product_laplace_v1"
ENTROPY_CONTROLLER_METHOD = "confidence_constrained_product_kl_primal_dual_v1"


def _readonly_float64(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _validate_rows(rows: np.ndarray, cardinalities: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != len(cardinalities):
        raise ValueError(
            f"rows must have shape (n, {len(cardinalities)}), got {array.shape}"
        )
    for attr, cardinality in enumerate(cardinalities):
        if cardinality <= 0:
            raise ValueError("cardinalities must be positive")
        values = array[:, attr]
        if np.any(values < 0) or np.any(values >= cardinality):
            raise ValueError(f"rows contain out-of-range values for attribute {attr}")
    return array


def _mixed_radix_strides(cardinalities: tuple[int, ...]) -> tuple[np.ndarray, int]:
    limit = int(np.iinfo(np.uint64).max)
    strides: list[int] = []
    domain_size = 1
    for cardinality in cardinalities:
        value = int(cardinality)
        if value <= 0:
            raise ValueError("cardinalities must be positive")
        strides.append(domain_size)
        if domain_size > limit // value:
            raise ValueError(
                "The first entropy profile requires the public full-row domain "
                "to fit in an exact uint64 mixed-radix atom id"
            )
        domain_size *= value
    return np.asarray(strides, dtype=np.uint64), int(domain_size)


@dataclass(frozen=True)
class ReleasedProductPrior:
    cardinalities: tuple[int, ...]
    probabilities: tuple[np.ndarray, ...]
    public_total: int
    smoothing: float
    method: str = PRODUCT_PRIOR_METHOD
    _strides: np.ndarray = field(init=False, repr=False)
    _domain_size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cardinalities = tuple(int(value) for value in self.cardinalities)
        if not cardinalities:
            raise ValueError("cardinalities must be nonempty")
        if int(self.public_total) <= 0:
            raise ValueError("public_total must be positive")
        smoothing = float(self.smoothing)
        if not math.isfinite(smoothing) or smoothing <= 0.0:
            raise ValueError("smoothing must be finite and positive")
        if self.method != PRODUCT_PRIOR_METHOD:
            raise ValueError(f"Unsupported product-prior method {self.method!r}")
        if len(self.probabilities) != len(cardinalities):
            raise ValueError("probabilities must contain one vector per attribute")

        normalized: list[np.ndarray] = []
        for attr, (cardinality, raw) in enumerate(
            zip(cardinalities, self.probabilities, strict=True)
        ):
            probabilities = np.asarray(raw, dtype=np.float64)
            if probabilities.shape != (cardinality,):
                raise ValueError(
                    f"Attribute {attr} probabilities must have shape ({cardinality},)"
                )
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
                raise ValueError("Product-prior probabilities must be finite and positive")
            if not np.isclose(float(probabilities.sum()), 1.0, rtol=1.0e-12, atol=1.0e-12):
                raise ValueError("Each product-prior probability vector must sum to one")
            normalized.append(_readonly_float64(probabilities))

        strides, domain_size = _mixed_radix_strides(cardinalities)
        strides.setflags(write=False)
        object.__setattr__(self, "cardinalities", cardinalities)
        object.__setattr__(self, "probabilities", tuple(normalized))
        object.__setattr__(self, "public_total", int(self.public_total))
        object.__setattr__(self, "smoothing", smoothing)
        object.__setattr__(self, "_strides", strides)
        object.__setattr__(self, "_domain_size", domain_size)

    @classmethod
    def from_released_oneway(
        cls,
        qcat: QueryCatalogue,
        released_target: np.ndarray,
        cardinalities: np.ndarray | tuple[int, ...],
        *,
        public_total: int,
        smoothing: float = 1.0,
    ) -> ReleasedProductPrior:
        cards = tuple(int(value) for value in np.asarray(cardinalities, dtype=np.int64))
        target = np.asarray(released_target, dtype=np.float64)
        if target.shape != (qcat.m,) or not np.all(np.isfinite(target)):
            raise ValueError("released_target must be a finite vector matching the query catalogue")
        if int(public_total) <= 0:
            raise ValueError("public_total must be positive")
        if not math.isfinite(float(smoothing)) or float(smoothing) <= 0.0:
            raise ValueError("smoothing must be finite and positive")

        probabilities: list[np.ndarray] = []
        for attr, cardinality in enumerate(cards):
            qids_by_value: dict[int, int] = {}
            expected_group = f"oneway:{attr}"
            for qid, group in enumerate(qcat.groups):
                if group != expected_group or int(qcat.linear_num_terms[qid]) != 0:
                    continue
                terms = qcat.query_terms(qid)
                if len(terms) != 1:
                    continue
                query_attr, op, value, _, _ = terms[0]
                if int(query_attr) == attr and int(op) == OP_EQ:
                    if int(value) in qids_by_value:
                        raise ValueError(
                            f"Duplicate released one-way cell for attribute {attr}, value {value}"
                        )
                    qids_by_value[int(value)] = int(qid)
            if set(qids_by_value) != set(range(cardinality)):
                raise ValueError(
                    f"Released one-way product prior requires a complete partition for attribute {attr}"
                )
            qids = np.asarray([qids_by_value[value] for value in range(cardinality)], dtype=np.int32)
            counts = project_simplex(target[qids], float(public_total)).astype(np.float64)
            smoothed = counts + float(smoothing)
            probabilities.append(smoothed / float(smoothed.sum()))

        return cls(
            cardinalities=cards,
            probabilities=tuple(probabilities),
            public_total=int(public_total),
            smoothing=float(smoothing),
        )

    @property
    def dimension(self) -> int:
        return len(self.cardinalities)

    @property
    def domain_size(self) -> int:
        return self._domain_size

    def encode_rows(self, rows: np.ndarray) -> np.ndarray:
        array = _validate_rows(rows, self.cardinalities).astype(np.uint64, copy=False)
        return np.sum(array * self._strides.reshape(1, -1), axis=1, dtype=np.uint64)

    def log_probability_rows(self, rows: np.ndarray) -> np.ndarray:
        array = _validate_rows(rows, self.cardinalities)
        result = np.zeros(array.shape[0], dtype=np.float64)
        for attr, probabilities in enumerate(self.probabilities):
            result += np.log(probabilities[array[:, attr]])
        return result

    def diagnostics(self) -> dict[str, Any]:
        minimum = min(float(np.min(values)) for values in self.probabilities)
        maximum = max(float(np.max(values)) for values in self.probabilities)
        return {
            "method": self.method,
            "num_attributes": self.dimension,
            "public_total": self.public_total,
            "smoothing": self.smoothing,
            "domain_size": self.domain_size,
            "minimum_probability": minimum,
            "maximum_probability": maximum,
        }


def _regularizer_terms(
    counts: np.ndarray,
    log_probabilities: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    count_values = np.asarray(counts, dtype=np.float64)
    log_p = np.asarray(log_probabilities, dtype=np.float64)
    if count_values.shape != log_p.shape:
        raise ValueError("counts and log_probabilities must have matching shapes")
    if np.any(count_values < 0.0):
        raise ValueError("atom counts must be nonnegative")
    result = np.zeros(count_values.shape, dtype=np.float64)
    positive = count_values > 0.0
    result[positive] = count_values[positive] * (
        np.log(count_values[positive]) - math.log(float(n_rows)) - log_p[positive]
    )
    return result


@dataclass
class AtomEntropyState:
    prior: ReleasedProductPrior
    n_rows: int
    counts: dict[int, int]
    regularizer: float

    @classmethod
    def from_rows(
        cls,
        rows: np.ndarray,
        prior: ReleasedProductPrior,
    ) -> AtomEntropyState:
        array = _validate_rows(rows, prior.cardinalities)
        if len(array) <= 0:
            raise ValueError("rows must be nonempty")
        codes = prior.encode_rows(array)
        unique, first_indices, atom_counts = np.unique(
            codes,
            return_index=True,
            return_counts=True,
        )
        log_p = prior.log_probability_rows(array[first_indices])
        regularizer = float(
            np.sum(_regularizer_terms(atom_counts, log_p, len(array)), dtype=np.float64)
        )
        return cls(
            prior=prior,
            n_rows=int(len(array)),
            counts={int(code): int(count) for code, count in zip(unique, atom_counts, strict=True)},
            regularizer=regularizer,
        )

    def _lookup_counts(self, codes: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (self.counts.get(int(code), 0) for code in np.asarray(codes).tolist()),
            dtype=np.int64,
            count=len(codes),
        )

    def candidate_gains(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        old = _validate_rows(old_rows, self.prior.cardinalities)
        new = _validate_rows(new_rows, self.prior.cardinalities)
        if old.shape != new.shape:
            raise ValueError("old_rows and new_rows must have matching shapes")
        old_codes = self.prior.encode_rows(old)
        new_codes = self.prior.encode_rows(new)
        old_counts = self._lookup_counts(old_codes)
        new_counts = self._lookup_counts(new_codes)
        if np.any(old_counts <= 0):
            raise ValueError("Every candidate old atom must be present in the entropy state")
        old_log_p = self.prior.log_probability_rows(old)
        new_log_p = self.prior.log_probability_rows(new)
        gain = (
            _regularizer_terms(old_counts, old_log_p, self.n_rows)
            + _regularizer_terms(new_counts, new_log_p, self.n_rows)
            - _regularizer_terms(old_counts - 1, old_log_p, self.n_rows)
            - _regularizer_terms(new_counts + 1, new_log_p, self.n_rows)
        )
        gain[old_codes == new_codes] = 0.0
        return gain.astype(np.float64, copy=False)

    def prefix_gains(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        old = _validate_rows(old_rows, self.prior.cardinalities)
        new = _validate_rows(new_rows, self.prior.cardinalities)
        if old.shape != new.shape:
            raise ValueError("old_rows and new_rows must have matching shapes")
        old_codes = self.prior.encode_rows(old)
        new_codes = self.prior.encode_rows(new)
        old_log_p = self.prior.log_probability_rows(old)
        new_log_p = self.prior.log_probability_rows(new)
        local_counts: dict[int, int] = {}

        def count(code: int) -> int:
            return local_counts.get(code, self.counts.get(code, 0))

        cumulative = np.zeros(len(old), dtype=np.float64)
        total_gain = 0.0
        for index in range(len(old)):
            old_code = int(old_codes[index])
            new_code = int(new_codes[index])
            if old_code == new_code:
                cumulative[index] = total_gain
                continue
            old_count = count(old_code)
            new_count = count(new_code)
            if old_count <= 0:
                raise ValueError("A batch removes more rows from an atom than are available")
            incremental = float(
                _regularizer_terms(
                    np.asarray([old_count, new_count]),
                    np.asarray([old_log_p[index], new_log_p[index]]),
                    self.n_rows,
                ).sum()
                - _regularizer_terms(
                    np.asarray([old_count - 1, new_count + 1]),
                    np.asarray([old_log_p[index], new_log_p[index]]),
                    self.n_rows,
                ).sum()
            )
            local_counts[old_code] = old_count - 1
            local_counts[new_code] = new_count + 1
            total_gain += incremental
            cumulative[index] = total_gain
        return cumulative

    def batch_gain(self, old_rows: np.ndarray, new_rows: np.ndarray) -> float:
        gains = self.prefix_gains(old_rows, new_rows)
        return float(gains[-1]) if len(gains) else 0.0

    def apply_batch(self, old_rows: np.ndarray, new_rows: np.ndarray) -> float:
        old = _validate_rows(old_rows, self.prior.cardinalities)
        new = _validate_rows(new_rows, self.prior.cardinalities)
        if old.shape != new.shape:
            raise ValueError("old_rows and new_rows must have matching shapes")
        gain = self.batch_gain(old, new)
        old_codes = self.prior.encode_rows(old)
        new_codes = self.prior.encode_rows(new)
        updates: dict[int, int] = {}
        for code in old_codes.tolist():
            key = int(code)
            updates[key] = updates.get(key, 0) - 1
        for code in new_codes.tolist():
            key = int(code)
            updates[key] = updates.get(key, 0) + 1
        for code, delta in updates.items():
            updated = self.counts.get(code, 0) + int(delta)
            if updated < 0:
                raise ValueError("Entropy batch produced a negative atom count")
            if updated == 0:
                self.counts.pop(code, None)
            else:
                self.counts[code] = updated
        if sum(self.counts.values()) != self.n_rows:
            raise AssertionError("Entropy atom counts no longer sum to n_rows")
        self.regularizer = float(self.regularizer - gain)
        return gain

    def recompute_regularizer(self, rows: np.ndarray) -> float:
        reference = AtomEntropyState.from_rows(rows, self.prior)
        return reference.regularizer

    def assert_matches(self, rows: np.ndarray, *, tolerance: float = 1.0e-8) -> None:
        reference = AtomEntropyState.from_rows(rows, self.prior)
        if reference.counts != self.counts:
            raise AssertionError("Incremental entropy atom counts do not match the synthetic table")
        scale = max(1.0, abs(reference.regularizer))
        if abs(reference.regularizer - self.regularizer) > float(tolerance) * scale:
            raise AssertionError("Incremental entropy regularizer drift exceeds tolerance")

    def diagnostics(self) -> dict[str, Any]:
        counts = np.asarray(list(self.counts.values()), dtype=np.int64)
        return {
            "num_rows": self.n_rows,
            "num_active_atoms": int(len(self.counts)),
            "max_atom_count": int(np.max(counts)) if len(counts) else 0,
            "regularizer": float(self.regularizer),
            "kl_per_row": float(self.regularizer / self.n_rows),
        }


@dataclass
class EntropyDiscrepancyController:
    confidence: ChiSquareDiscrepancyStop
    max_iterations: int
    dual_weight: float = 1.0
    dual_min: float = 1.0e-8
    dual_max: float = 1.0e8
    method: str = ENTROPY_CONTROLLER_METHOD
    num_updates: int = 0
    num_feasibility_calibrations: int = 0

    def __post_init__(self) -> None:
        if self.method != ENTROPY_CONTROLLER_METHOD:
            raise ValueError(f"Unsupported entropy controller method {self.method!r}")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        for name, value in (
            ("dual_weight", self.dual_weight),
            ("dual_min", self.dual_min),
            ("dual_max", self.dual_max),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.dual_min > self.dual_max:
            raise ValueError("dual_min must not exceed dual_max")
        self.dual_weight = float(np.clip(self.dual_weight, self.dual_min, self.dual_max))

    @classmethod
    def create(
        cls,
        *,
        alpha: float,
        effective_rank: int,
        max_iterations: int,
        dual_initial: float = 1.0,
    ) -> EntropyDiscrepancyController:
        return cls(
            confidence=ChiSquareDiscrepancyStop.create(
                alpha=float(alpha),
                effective_rank=int(effective_rank),
            ),
            max_iterations=int(max_iterations),
            dual_weight=float(dual_initial),
        )

    @property
    def objective_threshold(self) -> float:
        return self.confidence.objective_threshold

    @property
    def dual_step_size(self) -> float:
        return 1.0 / math.sqrt(float(self.max_iterations))

    def inside(self, data_objective: float) -> bool:
        return self.confidence.reached(float(data_objective))

    def update_dual(self, data_objective: float) -> float:
        value = float(data_objective)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("data_objective must be finite and nonnegative")
        relative_violation = value / self.objective_threshold - 1.0
        log_step = self.dual_step_size * float(np.clip(relative_violation, -1.0, 1.0))
        updated = self.dual_weight * math.exp(log_step)
        self.dual_weight = float(np.clip(updated, self.dual_min, self.dual_max))
        self.num_updates += 1
        return self.dual_weight

    def calibrate_feasibility_floor(
        self,
        *,
        data_objective: float,
        data_gains: np.ndarray,
        entropy_gains: np.ndarray,
        edit_costs: np.ndarray,
        lambda_cost: float,
        min_advantage: float,
    ) -> float:
        if self.inside(data_objective):
            return self.dual_weight
        data = np.asarray(data_gains, dtype=np.float64)
        entropy = np.asarray(entropy_gains, dtype=np.float64)
        costs = np.asarray(edit_costs, dtype=np.float64)
        if data.shape != entropy.shape or data.shape != costs.shape:
            raise ValueError("candidate gain vectors must have matching shapes")
        eligible = np.isfinite(data) & np.isfinite(entropy) & np.isfinite(costs) & (data > 0.0)
        if not np.any(eligible):
            return self.dual_weight
        current = (
            self.dual_weight * data[eligible]
            + entropy[eligible]
            - float(lambda_cost) * costs[eligible]
        )
        if np.any(current > float(min_advantage)):
            return self.dual_weight
        required = (
            float(lambda_cost) * costs[eligible]
            + float(min_advantage)
            - entropy[eligible]
        ) / data[eligible]
        required_weight = max(0.0, float(np.min(required)))
        if required_weight >= self.dual_weight:
            margin = max(1.0e-12, 1.0e-9 * max(1.0, required_weight))
            self.dual_weight = float(
                np.clip(required_weight + margin, self.dual_min, self.dual_max)
            )
            self.num_feasibility_calibrations += 1
        return self.dual_weight

    def candidate_advantages(
        self,
        *,
        data_objective: float,
        data_gains: np.ndarray,
        entropy_gains: np.ndarray,
        edit_costs: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        data = np.asarray(data_gains, dtype=np.float64)
        entropy = np.asarray(entropy_gains, dtype=np.float64)
        costs = np.asarray(edit_costs, dtype=np.float64)
        if data.shape != entropy.shape or data.shape != costs.shape:
            raise ValueError("candidate gain vectors must have matching shapes")
        combined = self.dual_weight * data + entropy - float(lambda_cost) * costs
        if not self.inside(data_objective):
            combined = np.where(data > 0.0, combined, -np.inf)
        return combined

    def prefix_advantages(
        self,
        *,
        data_objective: float,
        data_gains: np.ndarray,
        entropy_gains: np.ndarray,
        edit_costs: np.ndarray,
        lambda_cost: float,
    ) -> np.ndarray:
        data = np.asarray(data_gains, dtype=np.float64)
        entropy = np.asarray(entropy_gains, dtype=np.float64)
        costs = np.asarray(edit_costs, dtype=np.float64)
        if data.shape != entropy.shape or data.shape != costs.shape:
            raise ValueError("prefix gain vectors must have matching shapes")
        combined = self.dual_weight * data + entropy - float(lambda_cost) * costs
        resulting_objective = float(data_objective) - data
        tolerance = 1.0e-12 * max(1.0, self.objective_threshold)
        if self.inside(data_objective):
            valid = resulting_objective <= self.objective_threshold + tolerance
        else:
            valid = data > 0.0
        return np.where(valid, combined, -np.inf)

    def regularized_objective(self, data_objective: float, regularizer: float) -> float:
        return float(self.dual_weight * float(data_objective) + float(regularizer))

    def diagnostics(self, data_objective: float) -> dict[str, Any]:
        result = self.confidence.diagnostics()
        result.update(
            {
                "method": self.method,
                "data_objective": float(data_objective),
                "inside_confidence_set": self.inside(data_objective),
                "dual_weight": float(self.dual_weight),
                "dual_step_size": float(self.dual_step_size),
                "dual_updates": int(self.num_updates),
                "feasibility_calibrations": int(self.num_feasibility_calibrations),
            }
        )
        return result
