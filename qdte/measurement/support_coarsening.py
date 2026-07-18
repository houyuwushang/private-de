from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.stats import norm

from qdte.measurement.factorization import HierarchicalInteractionTranscript


SUPPORT_BETA = 0.01
RELEASED_SUPPORT_METHOD = "released_oneway_simultaneous_lcb_v1"
ORACLE_SUPPORT_METHOD = "offline_true_topk_matching_released_size_v1"


def _readonly_int(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int32).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AttributeSupportMap:
    attribute: int
    original_cardinality: int
    retained_categories: tuple[int, ...]
    rare_categories: tuple[int, ...]
    original_to_coarse: np.ndarray

    def __post_init__(self) -> None:
        cardinality = int(self.original_cardinality)
        if cardinality < 2:
            raise ValueError("Support coarsening requires cardinality at least two")
        retained = tuple(int(value) for value in self.retained_categories)
        rare = tuple(int(value) for value in self.rare_categories)
        if not retained:
            raise ValueError("Every support map must retain at least one category")
        if set(retained) & set(rare) or set(retained) | set(rare) != set(range(cardinality)):
            raise ValueError("Retained and rare categories must partition the public domain")
        mapping = np.asarray(self.original_to_coarse, dtype=np.int32)
        if mapping.shape != (cardinality,):
            raise ValueError("Support mapping shape does not match cardinality")
        coarse_cardinality = cardinality if not rare else len(retained) + 1
        if np.any(mapping < 0) or np.any(mapping >= coarse_cardinality):
            raise ValueError("Support mapping contains an out-of-range coarse category")
        if set(mapping.tolist()) != set(range(coarse_cardinality)):
            raise ValueError("Support mapping must cover every coarse category")
        object.__setattr__(self, "retained_categories", retained)
        object.__setattr__(self, "rare_categories", rare)
        object.__setattr__(self, "original_to_coarse", _readonly_int(mapping))

    @property
    def coarse_cardinality(self) -> int:
        return (
            self.original_cardinality
            if not self.rare_categories
            else len(self.retained_categories) + 1
        )

    @property
    def is_identity(self) -> bool:
        return not self.rare_categories

    def aggregation_matrix(self) -> np.ndarray:
        matrix = np.zeros(
            (self.coarse_cardinality, self.original_cardinality),
            dtype=np.float64,
        )
        matrix[self.original_to_coarse, np.arange(self.original_cardinality)] = 1.0
        return matrix

    def aggregate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (self.original_cardinality,):
            raise ValueError("Probability vector shape does not match support map")
        return self.aggregation_matrix() @ values

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "original_cardinality": self.original_cardinality,
            "coarse_cardinality": self.coarse_cardinality,
            "retained_categories": list(self.retained_categories),
            "rare_categories": list(self.rare_categories),
            "original_to_coarse": self.original_to_coarse.tolist(),
            "identity": self.is_identity,
        }


@dataclass(frozen=True)
class SupportCoarsening:
    attributes: tuple[AttributeSupportMap, ...]
    method: str
    beta: float = SUPPORT_BETA
    kappa: float | None = None
    private_diagnostic: bool = False

    def __post_init__(self) -> None:
        if not self.attributes:
            raise ValueError("Support coarsening must contain every attribute")
        if tuple(item.attribute for item in self.attributes) != tuple(
            range(len(self.attributes))
        ):
            raise ValueError("Support maps must be ordered by attribute id")
        if self.method not in {RELEASED_SUPPORT_METHOD, ORACLE_SUPPORT_METHOD}:
            raise ValueError(f"Unsupported support method {self.method!r}")
        if self.method == ORACLE_SUPPORT_METHOD and not self.private_diagnostic:
            raise ValueError("Oracle support must be marked private_diagnostic")
        if self.method == RELEASED_SUPPORT_METHOD and self.private_diagnostic:
            raise ValueError("Released support must not be marked private_diagnostic")
        if not math.isclose(float(self.beta), SUPPORT_BETA, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("Support-v1 requires beta=0.01")

    @property
    def original_cardinalities(self) -> tuple[int, ...]:
        return tuple(item.original_cardinality for item in self.attributes)

    @property
    def coarse_cardinalities(self) -> tuple[int, ...]:
        return tuple(item.coarse_cardinality for item in self.attributes)

    @property
    def support_sizes(self) -> tuple[int, ...]:
        return tuple(len(item.retained_categories) for item in self.attributes)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "method": self.method,
            "beta": self.beta,
            "kappa": self.kappa,
            "private_diagnostic": self.private_diagnostic,
            "original_cardinalities": list(self.original_cardinalities),
            "coarse_cardinalities": list(self.coarse_cardinalities),
            "support_sizes": list(self.support_sizes),
            "attributes": [item.to_dict() for item in self.attributes],
        }
        payload["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupportCoarsening:
        if not isinstance(data, dict):
            raise ValueError("Support coarsening artifact must be a mapping")
        raw_attributes = data.get("attributes")
        if not isinstance(raw_attributes, list) or not raw_attributes:
            raise ValueError("Support coarsening artifact requires attribute maps")
        attributes: list[AttributeSupportMap] = []
        for raw in raw_attributes:
            if not isinstance(raw, dict):
                raise ValueError("Each support attribute map must be a mapping")
            attributes.append(
                AttributeSupportMap(
                    attribute=int(raw["attribute"]),
                    original_cardinality=int(raw["original_cardinality"]),
                    retained_categories=tuple(
                        int(value) for value in raw["retained_categories"]
                    ),
                    rare_categories=tuple(
                        int(value) for value in raw["rare_categories"]
                    ),
                    original_to_coarse=np.asarray(
                        raw["original_to_coarse"], dtype=np.int32
                    ),
                )
            )
        result = cls(
            attributes=tuple(attributes),
            method=str(data["method"]),
            beta=float(data.get("beta", SUPPORT_BETA)),
            kappa=(
                float(data["kappa"]) if data.get("kappa") is not None else None
            ),
            private_diagnostic=bool(data.get("private_diagnostic", False)),
        )
        expected_hash = data.get("sha256")
        if expected_hash is not None and str(expected_hash) != result.to_dict()["sha256"]:
            raise ValueError("Support coarsening artifact hash mismatch")
        return result


def _map_from_retained(
    attribute: int,
    cardinality: int,
    retained_categories: tuple[int, ...] | list[int],
) -> AttributeSupportMap:
    retained = tuple(sorted(int(value) for value in retained_categories))
    rare = tuple(value for value in range(int(cardinality)) if value not in set(retained))
    if not rare:
        mapping = np.arange(cardinality, dtype=np.int32)
        retained = tuple(range(cardinality))
    else:
        mapping = np.full(cardinality, len(retained), dtype=np.int32)
        for coarse, original in enumerate(retained):
            mapping[original] = coarse
    return AttributeSupportMap(
        attribute=int(attribute),
        original_cardinality=int(cardinality),
        retained_categories=retained,
        rare_categories=rare,
        original_to_coarse=mapping,
    )


def released_support_coarsening(
    transcript: HierarchicalInteractionTranscript,
) -> SupportCoarsening:
    cards = transcript.strategy.cardinalities
    total_categories = int(sum(cards))
    kappa = float(norm.ppf(1.0 - SUPPORT_BETA / (2.0 * total_categories)))
    reconstructed = transcript.reconstruct()
    maps: list[AttributeSupportMap] = []
    for attribute, cardinality in enumerate(cards):
        released_counts = reconstructed.oneway[attribute]
        standard_deviation = np.sqrt(reconstructed.oneway_variances[attribute])
        retained = np.flatnonzero(
            released_counts - kappa * standard_deviation > 0.0
        ).astype(np.int64)
        if retained.size == 0:
            maximum = float(np.max(released_counts))
            retained = np.asarray(
                [int(np.flatnonzero(released_counts == maximum)[0])],
                dtype=np.int64,
            )
        maps.append(
            _map_from_retained(attribute, cardinality, retained.tolist())
        )
    return SupportCoarsening(
        attributes=tuple(maps),
        method=RELEASED_SUPPORT_METHOD,
        beta=SUPPORT_BETA,
        kappa=kappa,
        private_diagnostic=False,
    )


def oracle_support_coarsening(
    released_support: SupportCoarsening,
    private_rows: np.ndarray,
) -> SupportCoarsening:
    if released_support.method != RELEASED_SUPPORT_METHOD:
        raise ValueError("Oracle support requires the paired released support rule")
    rows = np.asarray(private_rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != len(released_support.attributes):
        raise ValueError("Private diagnostic rows do not match support width")
    maps: list[AttributeSupportMap] = []
    for item in released_support.attributes:
        values = rows[:, item.attribute]
        if np.any(values < 0) or np.any(values >= item.original_cardinality):
            raise ValueError("Private diagnostic rows contain out-of-domain values")
        counts = np.bincount(values, minlength=item.original_cardinality)
        retain_count = max(1, len(item.retained_categories))
        category_ids = np.arange(item.original_cardinality, dtype=np.int64)
        ranking = np.lexsort((category_ids, -counts))
        retained = ranking[:retain_count].tolist()
        maps.append(
            _map_from_retained(
                item.attribute,
                item.original_cardinality,
                retained,
            )
        )
    return SupportCoarsening(
        attributes=tuple(maps),
        method=ORACLE_SUPPORT_METHOD,
        beta=SUPPORT_BETA,
        kappa=released_support.kappa,
        private_diagnostic=True,
    )


def lift_coarse_pair_distribution(
    coarse_table: np.ndarray,
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray,
    left_map: AttributeSupportMap,
    right_map: AttributeSupportMap,
) -> np.ndarray:
    q_left = np.asarray(left_probabilities, dtype=np.float64)
    q_right = np.asarray(right_probabilities, dtype=np.float64)
    coarse = np.asarray(coarse_table, dtype=np.float64)
    coarse_left = left_map.aggregate_probabilities(q_left)
    coarse_right = right_map.aggregate_probabilities(q_right)
    if coarse.shape != (left_map.coarse_cardinality, right_map.coarse_cardinality):
        raise ValueError("Coarse pair table shape does not match support maps")
    conditional_left = q_left / coarse_left[left_map.original_to_coarse]
    conditional_right = q_right / coarse_right[right_map.original_to_coarse]
    lifted = coarse[
        left_map.original_to_coarse[:, None],
        right_map.original_to_coarse[None, :],
    ]
    lifted = lifted * conditional_left[:, None] * conditional_right[None, :]
    return lifted


__all__ = [
    "AttributeSupportMap",
    "ORACLE_SUPPORT_METHOD",
    "RELEASED_SUPPORT_METHOD",
    "SUPPORT_BETA",
    "SupportCoarsening",
    "lift_coarse_pair_distribution",
    "oracle_support_coarsening",
    "released_support_coarsening",
]
