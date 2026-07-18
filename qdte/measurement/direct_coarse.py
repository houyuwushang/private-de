from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import numpy as np

from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.measure import MeasurementGroup, Measurements
from qdte.measurement.projection import project_simplex
from qdte.measurement.support_coarsening import (
    AttributeSupportMap,
    RELEASED_SUPPORT_METHOD,
    SupportCoarsening,
    lift_coarse_pair_distribution,
)
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.orthogonal import (
    helmert_contrast,
    oneway_contrast_from_counts,
    pair_reconstruction_maps,
    reconstruct_oneway,
    reconstruct_pair,
)
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup


C2_DIRECT_MODE = "direct_coarse_complete_partition"
C2_AGGREGATE_MODE = "full_complete_partition_then_aggregate"
C2_COMPLETE_PARTITION_SENSITIVITY_L2 = 1.0

CoarseMeasurementMode = Literal[
    "direct_coarse_complete_partition",
    "full_complete_partition_then_aggregate",
]


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoarsePairOperators:
    pair: tuple[int, int]
    full_shape: tuple[int, int]
    coarse_shape: tuple[int, int]
    aggregation: np.ndarray
    interaction: np.ndarray

    def __post_init__(self) -> None:
        full_dimension = int(np.prod(self.full_shape, dtype=np.int64))
        coarse_dimension = int(np.prod(self.coarse_shape, dtype=np.int64))
        interaction_dimension = int(
            (self.coarse_shape[0] - 1) * (self.coarse_shape[1] - 1)
        )
        aggregation = np.asarray(self.aggregation, dtype=np.float64)
        interaction = np.asarray(self.interaction, dtype=np.float64)
        if aggregation.shape != (coarse_dimension, full_dimension):
            raise ValueError("Coarse aggregation operator has the wrong shape")
        if interaction.shape != (interaction_dimension, coarse_dimension):
            raise ValueError("Coarse interaction operator has the wrong shape")
        if not np.all(np.isfinite(aggregation)) or not np.all(
            np.isfinite(interaction)
        ):
            raise ValueError("Coarse operators must be finite")
        object.__setattr__(self, "aggregation", _readonly(aggregation))
        object.__setattr__(self, "interaction", _readonly(interaction))

    @property
    def full_dimension(self) -> int:
        return int(np.prod(self.full_shape, dtype=np.int64))

    @property
    def coarse_dimension(self) -> int:
        return int(np.prod(self.coarse_shape, dtype=np.int64))

    @property
    def interaction_dimension(self) -> int:
        return int((self.coarse_shape[0] - 1) * (self.coarse_shape[1] - 1))


@dataclass(frozen=True)
class CoarsePairObservation:
    pair: tuple[int, int]
    mode: CoarseMeasurementMode
    rho: float
    noisy_coarse_table: np.ndarray
    coarse_covariance: np.ndarray
    interaction_center: np.ndarray
    interaction_covariance: np.ndarray
    coupled_standard_draw: np.ndarray | None

    def __post_init__(self) -> None:
        if self.mode not in {C2_DIRECT_MODE, C2_AGGREGATE_MODE}:
            raise ValueError(f"Unsupported C2 coarse measurement mode {self.mode!r}")
        rho = float(self.rho)
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("C2 coarse measurement rho must be finite and positive")
        table = np.asarray(self.noisy_coarse_table, dtype=np.float64)
        covariance = np.asarray(self.coarse_covariance, dtype=np.float64)
        center = np.asarray(self.interaction_center, dtype=np.float64)
        interaction_covariance = np.asarray(
            self.interaction_covariance,
            dtype=np.float64,
        )
        draw = (
            np.asarray(self.coupled_standard_draw, dtype=np.float64)
            if self.coupled_standard_draw is not None
            else None
        )
        if table.ndim != 2 or min(table.shape) < 2:
            raise ValueError("C2 noisy coarse table must be a two-dimensional partition")
        coarse_dimension = int(table.size)
        interaction_dimension = int((table.shape[0] - 1) * (table.shape[1] - 1))
        if covariance.shape != (coarse_dimension, coarse_dimension):
            raise ValueError("C2 coarse covariance has the wrong shape")
        if center.shape != (table.shape[0] - 1, table.shape[1] - 1):
            raise ValueError("C2 interaction center has the wrong shape")
        if interaction_covariance.shape != (
            interaction_dimension,
            interaction_dimension,
        ):
            raise ValueError("C2 interaction covariance has the wrong shape")
        if draw is not None and draw.shape != (coarse_dimension,):
            raise ValueError("C2 coupled standard draw has the wrong shape")
        for name, value in (
            ("noisy_coarse_table", table),
            ("coarse_covariance", covariance),
            ("interaction_center", center),
            ("interaction_covariance", interaction_covariance),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"C2 {name} must be finite")
        if draw is not None and not np.all(np.isfinite(draw)):
            raise ValueError("C2 coupled_standard_draw must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("C2 coarse covariance must be symmetric")
        if not np.allclose(
            interaction_covariance,
            interaction_covariance.T,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("C2 interaction covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1.0e-10:
            raise ValueError("C2 coarse covariance must be positive semidefinite")
        if np.min(np.linalg.eigvalsh(interaction_covariance)) < -1.0e-10:
            raise ValueError("C2 interaction covariance must be positive semidefinite")
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "noisy_coarse_table", _readonly(table))
        object.__setattr__(self, "coarse_covariance", _readonly(covariance))
        object.__setattr__(self, "interaction_center", _readonly(center))
        object.__setattr__(
            self,
            "interaction_covariance",
            _readonly(interaction_covariance),
        )
        if draw is not None:
            object.__setattr__(self, "coupled_standard_draw", _readonly(draw))
        else:
            object.__setattr__(self, "coupled_standard_draw", None)

    def diagnostics(self) -> dict[str, Any]:
        interaction_eigenvalues = np.linalg.eigvalsh(self.interaction_covariance)
        return {
            "pair": list(self.pair),
            "mode": self.mode,
            "mechanism": "gaussian_vector",
            "adjacency": "add_remove",
            "sensitivity_l2": C2_COMPLETE_PARTITION_SENSITIVITY_L2,
            "rho": self.rho,
            "coarse_shape": list(self.noisy_coarse_table.shape),
            "coarse_dimension": int(self.noisy_coarse_table.size),
            "interaction_dimension": int(self.interaction_center.size),
            "coarse_variance_trace": float(np.trace(self.coarse_covariance)),
            "interaction_variance_trace": float(
                np.trace(self.interaction_covariance)
            ),
            "maximum_interaction_variance": float(
                np.max(np.diag(self.interaction_covariance))
            ),
            "minimum_interaction_eigenvalue": float(
                np.min(interaction_eigenvalues)
            ),
            "maximum_interaction_eigenvalue": float(
                np.max(interaction_eigenvalues)
            ),
        }

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            **self.diagnostics(),
            "noisy_coarse_table": self.noisy_coarse_table.tolist(),
            "coarse_covariance": self.coarse_covariance.tolist(),
            "interaction_center": self.interaction_center.tolist(),
            "interaction_covariance": self.interaction_covariance.tolist(),
        }
        return payload

    @classmethod
    def from_public_dict(cls, data: dict[str, Any]) -> CoarsePairObservation:
        if not isinstance(data, dict):
            raise ValueError("C2 pair observation must be a mapping")
        return cls(
            pair=tuple(int(value) for value in data["pair"]),
            mode=str(data["mode"]),
            rho=float(data["rho"]),
            noisy_coarse_table=np.asarray(
                data["noisy_coarse_table"],
                dtype=np.float64,
            ),
            coarse_covariance=np.asarray(
                data["coarse_covariance"],
                dtype=np.float64,
            ),
            interaction_center=np.asarray(
                data["interaction_center"],
                dtype=np.float64,
            ),
            interaction_covariance=np.asarray(
                data["interaction_covariance"],
                dtype=np.float64,
            ),
            coupled_standard_draw=None,
        )


@dataclass(frozen=True)
class CoarseInteractionTranscript:
    support: SupportCoarsening
    public_total: int
    oneway_components: tuple[np.ndarray, ...]
    oneway_variances: tuple[float, ...]
    oneway_rho: tuple[float, ...]
    pair_observations: tuple[CoarsePairObservation, ...]
    rho_total: float
    rho_spent: float
    mode: CoarseMeasurementMode

    def __post_init__(self) -> None:
        if self.support.method != RELEASED_SUPPORT_METHOD:
            raise ValueError("C2 requires the released support partition")
        width = len(self.support.attributes)
        if int(self.public_total) <= 0:
            raise ValueError("C2 public_total must be positive")
        if not (
            len(self.oneway_components)
            == len(self.oneway_variances)
            == len(self.oneway_rho)
            == width
        ):
            raise ValueError("C2 one-way transcript must cover every attribute")
        components: list[np.ndarray] = []
        for attribute, (component, variance, rho) in enumerate(
            zip(
                self.oneway_components,
                self.oneway_variances,
                self.oneway_rho,
                strict=True,
            )
        ):
            values = np.asarray(component, dtype=np.float64)
            expected = self.support.attributes[attribute].original_cardinality - 1
            if values.shape != (expected,) or not np.all(np.isfinite(values)):
                raise ValueError("C2 one-way component has the wrong shape")
            if not np.isfinite(variance) or float(variance) <= 0.0:
                raise ValueError("C2 one-way variance must be finite and positive")
            if not np.isfinite(rho) or float(rho) <= 0.0:
                raise ValueError("C2 one-way rho must be finite and positive")
            cardinality = self.support.attributes[attribute].original_cardinality
            expected_variance = (1.0 - 1.0 / float(cardinality)) / (
                2.0 * float(rho)
            )
            if not np.isclose(
                float(variance),
                expected_variance,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("C2 one-way variance/rho metadata is inconsistent")
            components.append(_readonly(values))
        if self.mode not in {C2_DIRECT_MODE, C2_AGGREGATE_MODE}:
            raise ValueError(f"Unsupported C2 transcript mode {self.mode!r}")
        pairs = tuple(observation.pair for observation in self.pair_observations)
        if pairs != tuple(sorted(set(pairs))):
            raise ValueError("C2 pair observations must be unique and sorted")
        if any(observation.mode != self.mode for observation in self.pair_observations):
            raise ValueError("C2 pair observations must share the transcript mode")
        if any(
            min(pair) < 0 or max(pair) >= width or pair[0] >= pair[1]
            for pair in pairs
        ):
            raise ValueError("C2 pair observation has an invalid scope")
        for observation in self.pair_observations:
            left, right = observation.pair
            operators = coarse_pair_operators(
                self.support.attributes[left],
                self.support.attributes[right],
            )
            if observation.noisy_coarse_table.shape != operators.coarse_shape:
                raise ValueError("C2 pair table does not match the released support map")
            cell_variance = C2_COMPLETE_PARTITION_SENSITIVITY_L2**2 / (
                2.0 * observation.rho
            )
            if observation.mode == C2_DIRECT_MODE:
                expected_covariance = cell_variance * np.eye(
                    operators.coarse_dimension,
                    dtype=np.float64,
                )
            else:
                expected_covariance = cell_variance * (
                    operators.aggregation @ operators.aggregation.T
                )
            expected_covariance = 0.5 * (
                expected_covariance + expected_covariance.T
            )
            if not np.allclose(
                observation.coarse_covariance,
                expected_covariance,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("C2 pair covariance is inconsistent with mode and rho")
            expected_center = operators.interaction @ (
                observation.noisy_coarse_table.reshape(-1)
            )
            if not np.allclose(
                observation.interaction_center.reshape(-1),
                expected_center,
                rtol=1.0e-10,
                atol=1.0e-10,
            ):
                raise ValueError("C2 interaction center is inconsistent with the pair table")
            expected_interaction_covariance = (
                operators.interaction
                @ expected_covariance
                @ operators.interaction.T
            )
            expected_interaction_covariance = 0.5 * (
                expected_interaction_covariance
                + expected_interaction_covariance.T
            )
            if not np.allclose(
                observation.interaction_covariance,
                expected_interaction_covariance,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "C2 interaction covariance is inconsistent with the pair mechanism"
                )
        total = float(self.rho_total)
        spent = float(self.rho_spent)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("C2 rho_total must be finite and positive")
        if not np.isfinite(spent) or spent <= 0.0:
            raise ValueError("C2 rho_spent must be finite and positive")
        expected_spent = float(sum(self.oneway_rho)) + float(
            sum(observation.rho for observation in self.pair_observations)
        )
        tolerance = 1.0e-11 * max(1.0, total)
        if not np.isclose(spent, expected_spent, rtol=1.0e-11, atol=tolerance):
            raise ValueError("C2 rho_spent does not match the block ledger")
        if spent > total + tolerance:
            raise ValueError("C2 transcript exceeds rho_total")
        object.__setattr__(self, "oneway_components", tuple(components))
        object.__setattr__(
            self,
            "oneway_variances",
            tuple(float(value) for value in self.oneway_variances),
        )
        object.__setattr__(
            self,
            "oneway_rho",
            tuple(float(value) for value in self.oneway_rho),
        )
        object.__setattr__(self, "rho_total", total)
        object.__setattr__(self, "rho_spent", spent)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(observation.pair for observation in self.pair_observations)

    def pair(self, pair: tuple[int, int]) -> CoarsePairObservation:
        canonical = (int(pair[0]), int(pair[1]))
        for observation in self.pair_observations:
            if observation.pair == canonical:
                return observation
        raise KeyError(canonical)

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "method": "released_support_direct_coarse_causal_diagnostic_v1",
            "mode": self.mode,
            "adjacency": "add_remove",
            "accounting": "zcdp_actual_spend_v1",
            "accounting_theorem": (
                "gaussian_zcdp_rho_equals_delta2_over_2_noise_variance"
            ),
            "public_total": int(self.public_total),
            "rho_total": self.rho_total,
            "rho_spent": self.rho_spent,
            "support": self.support.to_dict(),
            "oneway_blocks": [
                {
                    "attribute": attribute,
                    "mechanism": "reused_released_oneway_gaussian_vector",
                    "rho": self.oneway_rho[attribute],
                    "coefficient_variance": self.oneway_variances[attribute],
                    "noisy_coefficients": self.oneway_components[attribute].tolist(),
                }
                for attribute in range(len(self.oneway_components))
            ],
            "pair_blocks": [
                observation.to_public_dict()
                for observation in self.pair_observations
            ],
        }
        hash_payload = dict(payload)
        payload["sha256"] = hashlib.sha256(
            json.dumps(
                hash_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_public_dict(cls, data: dict[str, Any]) -> CoarseInteractionTranscript:
        if not isinstance(data, dict):
            raise ValueError("C2 transcript must be a mapping")
        if data.get("method") != "released_support_direct_coarse_causal_diagnostic_v1":
            raise ValueError("C2 transcript has the wrong method")
        if data.get("adjacency") != "add_remove":
            raise ValueError("C2 transcript requires add_remove adjacency")
        if data.get("accounting") != "zcdp_actual_spend_v1":
            raise ValueError("C2 transcript has the wrong accounting mode")
        support = SupportCoarsening.from_dict(data["support"])
        raw_oneway = data.get("oneway_blocks")
        if not isinstance(raw_oneway, list) or len(raw_oneway) != len(
            support.attributes
        ):
            raise ValueError("C2 public transcript has incomplete one-way blocks")
        ordered_oneway = sorted(raw_oneway, key=lambda block: int(block["attribute"]))
        if [int(block["attribute"]) for block in ordered_oneway] != list(
            range(len(support.attributes))
        ):
            raise ValueError("C2 public one-way blocks have invalid attributes")
        raw_pairs = data.get("pair_blocks")
        if not isinstance(raw_pairs, list):
            raise ValueError("C2 public transcript pair_blocks must be a list")
        result = cls(
            support=support,
            public_total=int(data["public_total"]),
            oneway_components=tuple(
                np.asarray(block["noisy_coefficients"], dtype=np.float64)
                for block in ordered_oneway
            ),
            oneway_variances=tuple(
                float(block["coefficient_variance"])
                for block in ordered_oneway
            ),
            oneway_rho=tuple(float(block["rho"]) for block in ordered_oneway),
            pair_observations=tuple(
                CoarsePairObservation.from_public_dict(block)
                for block in raw_pairs
            ),
            rho_total=float(data["rho_total"]),
            rho_spent=float(data["rho_spent"]),
            mode=str(data["mode"]),
        )
        expected_hash = data.get("sha256")
        if expected_hash is not None and str(expected_hash) != result.to_public_dict()[
            "sha256"
        ]:
            raise ValueError("C2 public transcript hash mismatch")
        return result


def coarse_pair_operators(
    left_map: AttributeSupportMap,
    right_map: AttributeSupportMap,
) -> CoarsePairOperators:
    if int(left_map.attribute) >= int(right_map.attribute):
        raise ValueError("C2 pair maps must use canonical attribute order")
    aggregation = np.kron(
        left_map.aggregation_matrix(),
        right_map.aggregation_matrix(),
    )
    left_contrast = helmert_contrast(left_map.coarse_cardinality)
    right_contrast = helmert_contrast(right_map.coarse_cardinality)
    interaction = np.kron(left_contrast.T, right_contrast.T)
    return CoarsePairOperators(
        pair=(int(left_map.attribute), int(right_map.attribute)),
        full_shape=(
            int(left_map.original_cardinality),
            int(right_map.original_cardinality),
        ),
        coarse_shape=(
            int(left_map.coarse_cardinality),
            int(right_map.coarse_cardinality),
        ),
        aggregation=aggregation,
        interaction=interaction,
    )


def coupled_direct_and_aggregate_observations(
    full_counts: np.ndarray,
    left_map: AttributeSupportMap,
    right_map: AttributeSupportMap,
    *,
    rho: float,
    full_standard_draw: np.ndarray,
) -> tuple[CoarsePairObservation, CoarsePairObservation]:
    """Return exact-marginal direct and aggregate C2 Gaussian observations.

    Both standalone mechanisms spend ``rho`` on one sensitivity-one complete
    partition vector. Disjoint coarse buckets make the normalized aggregate of
    the full-domain standard-normal draw an iid coarse standard-normal draw.
    This supplies strong paired coupling without changing either arm's law.
    """

    operators = coarse_pair_operators(left_map, right_map)
    counts = np.asarray(full_counts, dtype=np.float64)
    draw = np.asarray(full_standard_draw, dtype=np.float64)
    if counts.shape != operators.full_shape:
        raise ValueError("C2 full count table does not match the support maps")
    if draw.shape != operators.full_shape:
        raise ValueError("C2 full standard draw does not match the support maps")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("C2 full count table must be finite and nonnegative")
    if not np.all(np.isfinite(draw)):
        raise ValueError("C2 full standard draw must be finite")
    rho_value = float(rho)
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("C2 rho must be finite and positive")

    aggregation = operators.aggregation
    exact_coarse = aggregation @ counts.reshape(-1)
    raw_coarse_draw = aggregation @ draw.reshape(-1)
    bucket_sizes = np.diag(aggregation @ aggregation.T)
    if np.any(bucket_sizes <= 0.0):
        raise RuntimeError("C2 support map produced an empty coarse cell")
    coupled_standard = raw_coarse_draw / np.sqrt(bucket_sizes)
    variance = C2_COMPLETE_PARTITION_SENSITIVITY_L2**2 / (2.0 * rho_value)

    direct_covariance = variance * np.eye(operators.coarse_dimension)
    aggregate_covariance = variance * (aggregation @ aggregation.T)
    direct_table = exact_coarse + np.sqrt(variance) * coupled_standard
    aggregate_table = exact_coarse + np.sqrt(variance) * raw_coarse_draw
    interaction = operators.interaction

    def observation(
        mode: CoarseMeasurementMode,
        table: np.ndarray,
        covariance: np.ndarray,
    ) -> CoarsePairObservation:
        symmetric_covariance = 0.5 * (covariance + covariance.T)
        center = interaction @ table
        interaction_covariance = interaction @ symmetric_covariance @ interaction.T
        interaction_covariance = 0.5 * (
            interaction_covariance + interaction_covariance.T
        )
        return CoarsePairObservation(
            pair=operators.pair,
            mode=mode,
            rho=rho_value,
            noisy_coarse_table=table.reshape(operators.coarse_shape),
            coarse_covariance=symmetric_covariance,
            interaction_center=center.reshape(
                operators.coarse_shape[0] - 1,
                operators.coarse_shape[1] - 1,
            ),
            interaction_covariance=interaction_covariance,
            coupled_standard_draw=coupled_standard,
        )

    return (
        observation(C2_DIRECT_MODE, direct_table, direct_covariance),
        observation(C2_AGGREGATE_MODE, aggregate_table, aggregate_covariance),
    )


def measure_paired_coarse_interaction_transcripts(
    rows: np.ndarray,
    released_transcript: HierarchicalInteractionTranscript,
    support: SupportCoarsening,
    *,
    rng: np.random.Generator,
) -> tuple[CoarseInteractionTranscript, CoarseInteractionTranscript]:
    """Measure paired standalone C2 transcripts with an unchanged rho ledger."""

    if support.method != RELEASED_SUPPORT_METHOD or support.private_diagnostic:
        raise ValueError("C2 measurement requires nonprivate released support")
    cards = tuple(int(value) for value in released_transcript.strategy.cardinalities)
    if support.original_cardinalities != cards:
        raise ValueError("C2 support cardinalities do not match the source transcript")
    X = np.asarray(rows, dtype=np.int64)
    if X.ndim != 2 or X.shape != (released_transcript.public_total, len(cards)):
        raise ValueError("C2 private rows do not match the public transcript dimensions")
    for attribute, cardinality in enumerate(cards):
        if np.any(X[:, attribute] < 0) or np.any(X[:, attribute] >= cardinality):
            raise ValueError("C2 private rows contain out-of-domain values")

    direct_pairs: list[CoarsePairObservation] = []
    aggregate_pairs: list[CoarsePairObservation] = []
    for left, right in released_transcript.strategy.pairs:
        full_counts = np.zeros((cards[left], cards[right]), dtype=np.float64)
        np.add.at(full_counts, (X[:, left], X[:, right]), 1.0)
        full_draw = rng.standard_normal(full_counts.shape)
        name = f"pair_interaction:{left}:{right}"
        direct, aggregate = coupled_direct_and_aggregate_observations(
            full_counts,
            support.attributes[left],
            support.attributes[right],
            rho=float(released_transcript.rho_by_block[name]),
            full_standard_draw=full_draw,
        )
        direct_pairs.append(direct)
        aggregate_pairs.append(aggregate)

    oneway_components = tuple(
        released_transcript.noisy_components[f"oneway_contrast:{attribute}"]
        for attribute in range(len(cards))
    )
    oneway_variances = tuple(
        float(released_transcript.component_variances[f"oneway_contrast:{attribute}"])
        for attribute in range(len(cards))
    )
    oneway_rho = tuple(
        float(released_transcript.rho_by_block[f"oneway_contrast:{attribute}"])
        for attribute in range(len(cards))
    )

    def transcript(
        mode: CoarseMeasurementMode,
        observations: list[CoarsePairObservation],
    ) -> CoarseInteractionTranscript:
        spent = float(sum(oneway_rho)) + float(
            sum(observation.rho for observation in observations)
        )
        return CoarseInteractionTranscript(
            support=support,
            public_total=int(released_transcript.public_total),
            oneway_components=oneway_components,
            oneway_variances=oneway_variances,
            oneway_rho=oneway_rho,
            pair_observations=tuple(observations),
            rho_total=float(released_transcript.rho_total),
            rho_spent=spent,
            mode=mode,
        )

    direct_transcript = transcript(C2_DIRECT_MODE, direct_pairs)
    aggregate_transcript = transcript(C2_AGGREGATE_MODE, aggregate_pairs)
    if not np.isclose(
        direct_transcript.rho_spent,
        released_transcript.rho_spent,
        rtol=1.0e-11,
        atol=1.0e-12,
    ) or not np.isclose(
        aggregate_transcript.rho_spent,
        released_transcript.rho_spent,
        rtol=1.0e-11,
        atol=1.0e-12,
    ):
        raise RuntimeError("C2 paired transcript changed the frozen rho spend")
    return direct_transcript, aggregate_transcript


def target_from_coarse_interaction_transcript(
    qcat: QueryCatalogue,
    workload_groups: list[WorkloadGroup],
    base_released_target: np.ndarray,
    transcript: CoarseInteractionTranscript,
) -> np.ndarray:
    """Represent a coarse transcript in the original complete-cell catalogue."""

    target = np.asarray(base_released_target, dtype=np.float64).copy()
    if target.shape != (qcat.m,) or not np.all(np.isfinite(target)):
        raise ValueError("C2 base released target must match the query catalogue")
    cards = transcript.support.original_cardinalities
    qcat.validate(np.asarray(cards, dtype=np.int64))
    oneway_counts: dict[int, np.ndarray] = {}
    pair_groups: dict[tuple[int, int], WorkloadGroup] = {}
    replacement_coverage = np.zeros(qcat.m, dtype=np.int32)
    for group in workload_groups:
        parts = str(group.name).split(":")
        indices = np.asarray(group.query_indices, dtype=np.int64)
        if group.family == "oneway" and len(parts) == 2:
            attribute = int(parts[1])
            values = reconstruct_oneway(
                float(transcript.public_total),
                transcript.oneway_components[attribute],
                cards[attribute],
            )
            if values.shape != (cards[attribute],):
                raise ValueError("C2 one-way target group has the wrong shape")
            target[indices] = values
            oneway_counts[attribute] = values.copy()
            replacement_coverage[indices] += 1
        elif group.family == "twoway" and len(parts) == 3:
            pair_groups[(int(parts[1]), int(parts[2]))] = group
            replacement_coverage[indices] += 1
        else:
            raise ValueError(
                "C2 target supports only complete one-way and two-way partitions"
            )
    if not np.all(replacement_coverage == 1):
        raise ValueError("C2 target groups must replace every query exactly once")
    if set(oneway_counts) != set(range(len(cards))):
        raise ValueError("C2 target requires every one-way partition")
    if set(pair_groups) != set(transcript.pairs):
        raise ValueError("C2 target pair groups do not match the transcript")

    for pair in transcript.pairs:
        left, right = pair
        left_map = transcript.support.attributes[left]
        right_map = transcript.support.attributes[right]
        left_counts = oneway_counts[left]
        right_counts = oneway_counts[right]
        coarse_left = left_map.aggregation_matrix() @ left_counts
        coarse_right = right_map.aggregation_matrix() @ right_counts
        coarse_table = reconstruct_pair(
            float(transcript.public_total),
            oneway_contrast_from_counts(coarse_left),
            oneway_contrast_from_counts(coarse_right),
            transcript.pair(pair).interaction_center,
            left_map.coarse_cardinality,
            right_map.coarse_cardinality,
        )
        q_left = _released_lift_probabilities(
            left_counts,
            transcript.public_total,
        )
        q_right = _released_lift_probabilities(
            right_counts,
            transcript.public_total,
        )
        lifted = lift_coarse_pair_distribution(
            coarse_table,
            q_left,
            q_right,
            left_map,
            right_map,
        )
        indices = np.asarray(pair_groups[pair].query_indices, dtype=np.int64)
        if indices.size != lifted.size:
            raise ValueError("C2 pair target group has the wrong shape")
        target[indices] = lifted.reshape(-1)
    return target


def _released_lift_probabilities(
    counts: np.ndarray,
    public_total: int,
) -> np.ndarray:
    projected = project_simplex(counts, float(public_total)).astype(np.float64)
    smoothed = projected + 1.0
    return smoothed / float(np.sum(smoothed))


def _coarse_interaction_lift_map(
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray,
    left_map: AttributeSupportMap,
    right_map: AttributeSupportMap,
) -> np.ndarray:
    q_left = np.asarray(left_probabilities, dtype=np.float64)
    q_right = np.asarray(right_probabilities, dtype=np.float64)
    coarse_left = left_map.aggregate_probabilities(q_left)
    coarse_right = right_map.aggregate_probabilities(q_right)
    conditional_left = q_left / coarse_left[left_map.original_to_coarse]
    conditional_right = q_right / coarse_right[right_map.original_to_coarse]
    full_dimension = left_map.original_cardinality * right_map.original_cardinality
    coarse_dimension = left_map.coarse_cardinality * right_map.coarse_cardinality
    lift = np.zeros((full_dimension, coarse_dimension), dtype=np.float64)
    row = 0
    for left in range(left_map.original_cardinality):
        for right in range(right_map.original_cardinality):
            coarse_index = (
                int(left_map.original_to_coarse[left])
                * right_map.coarse_cardinality
                + int(right_map.original_to_coarse[right])
            )
            lift[row, coarse_index] = (
                conditional_left[left] * conditional_right[right]
            )
            row += 1
    _, _, interaction_map = pair_reconstruction_maps(
        left_map.coarse_cardinality,
        right_map.coarse_cardinality,
    )
    return lift @ interaction_map


def proposal_variances_from_coarse_interaction_transcript(
    qcat: QueryCatalogue,
    source_measurements: Measurements,
    transcript: CoarseInteractionTranscript,
) -> np.ndarray:
    """Return the declared diagonal surrogate used only by proposal logic.

    The common one-way contribution is retained from the frozen source
    reconstruction. The old full-domain interaction contribution is removed
    and replaced by the exact diagonal induced by the C2 coarse interaction
    covariance and the frozen released one-way lifting rule. The RCE objective
    itself never diagonalizes this covariance.
    """

    if source_measurements.strategy_transcript is None:
        raise ValueError("C2 proposal variances require the source strategy transcript")
    source_transcript = HierarchicalInteractionTranscript.from_public_dict(
        source_measurements.strategy_transcript
    )
    if source_transcript.strategy.cardinalities != transcript.support.original_cardinalities:
        raise ValueError("C2 source strategy cardinalities do not match")
    variances = np.asarray(source_measurements.variances, dtype=np.float64).copy()
    cards = transcript.support.original_cardinalities
    oneway_counts: dict[int, np.ndarray] = {}
    pair_groups: dict[tuple[int, int], WorkloadGroup | MeasurementGroup] = {}
    for group in source_measurements.groups:
        parts = str(group.name).split(":")
        indices = np.asarray(group.query_indices, dtype=np.int64)
        if group.family == "oneway" and len(parts) == 2:
            attribute = int(parts[1])
            oneway_counts[attribute] = np.asarray(
                source_measurements.target_projected[indices],
                dtype=np.float64,
            )
        elif group.family == "twoway" and len(parts) == 3:
            pair_groups[(int(parts[1]), int(parts[2]))] = group
    if set(oneway_counts) != set(range(len(cards))) or set(pair_groups) != set(
        transcript.pairs
    ):
        raise ValueError("C2 proposal variance groups do not match the transcript")

    for pair in transcript.pairs:
        left, right = pair
        group = pair_groups[pair]
        indices = np.asarray(group.query_indices, dtype=np.int64)
        old_variance = float(
            source_transcript.component_variances[
                f"pair_interaction:{left}:{right}"
            ]
        )
        old_interaction_diagonal = old_variance * (
            1.0 - 1.0 / float(cards[left])
        ) * (1.0 - 1.0 / float(cards[right]))
        common = np.maximum(
            variances[indices] - old_interaction_diagonal,
            0.0,
        )
        q_left = _released_lift_probabilities(
            oneway_counts[left],
            transcript.public_total,
        )
        q_right = _released_lift_probabilities(
            oneway_counts[right],
            transcript.public_total,
        )
        interaction_lift = _coarse_interaction_lift_map(
            q_left,
            q_right,
            transcript.support.attributes[left],
            transcript.support.attributes[right],
        )
        interaction_covariance = transcript.pair(pair).interaction_covariance
        new_interaction_diagonal = np.einsum(
            "ij,jk,ik->i",
            interaction_lift,
            interaction_covariance,
            interaction_lift,
            optimize=True,
        )
        variances[indices] = common + np.maximum(new_interaction_diagonal, 0.0)
    if variances.shape != (qcat.m,) or not np.all(np.isfinite(variances)):
        raise RuntimeError("C2 proposal variance surrogate is invalid")
    return np.maximum(variances, 1.0e-12)


def measurements_from_coarse_interaction_transcript(
    qcat: QueryCatalogue,
    source_measurements: Measurements,
    transcript: CoarseInteractionTranscript,
) -> Measurements:
    """Build one standalone public C2 measurement artifact.

    The source artifact contributes only the already released one-way target,
    query catalogue grouping, public row count, and privacy metadata. Its pair
    observations are replaced by the direct or aggregate C2 mechanism. The
    serialized diagonal variances are retained only for compatibility; C2
    generation must use ``CoarsenedInteractionPrecision`` from the embedded
    full-covariance transcript.
    """

    if source_measurements.mode != "dp":
        raise ValueError("C2 public measurements require a DP source artifact")
    if source_measurements.num_rows != transcript.public_total:
        raise ValueError("C2 source and transcript public row counts differ")
    if float(
        np.max(
            np.abs(
                np.asarray(source_measurements.target_projected, dtype=np.float64)
                - np.asarray(source_measurements.target_noisy, dtype=np.float64)
            )
        )
    ) > 1.0e-7:
        raise ValueError("C2 requires the frozen raw hierarchical source target")
    if not np.isclose(
        source_measurements.rho_total,
        transcript.rho_total,
        rtol=1.0e-11,
        atol=1.0e-12,
    ) or not np.isclose(
        source_measurements.rho_spent,
        transcript.rho_spent,
        rtol=1.0e-11,
        atol=1.0e-12,
    ):
        raise ValueError("C2 source and transcript rho ledgers differ")

    groups: list[MeasurementGroup] = []
    observed_pairs = set(transcript.pairs)
    for group in source_measurements.groups:
        parts = str(group.name).split(":")
        is_replaced_pair = (
            group.family == "twoway"
            and len(parts) == 3
            and (int(parts[1]), int(parts[2])) in observed_pairs
        )
        if is_replaced_pair:
            rho = float(transcript.pair((int(parts[1]), int(parts[2]))).rho)
            sigma = C2_COMPLETE_PARTITION_SENSITIVITY_L2 / np.sqrt(2.0 * rho)
            sensitivity_l2 = C2_COMPLETE_PARTITION_SENSITIVITY_L2
            noise_std = sigma
        else:
            rho = float(group.rho)
            sigma = float(group.sigma)
            sensitivity_l2 = float(group.sensitivity_l2)
            noise_std = float(group.noise_std)
        groups.append(
            MeasurementGroup(
                query_indices=np.asarray(group.query_indices, dtype=np.int32).copy(),
                sensitivity_l2=sensitivity_l2,
                rho=rho,
                sigma=sigma,
                noise_std=noise_std,
                name=str(group.name),
                family=str(group.family),
                is_partition=bool(group.is_partition),
            )
        )
    target = target_from_coarse_interaction_transcript(
        qcat,
        groups,
        source_measurements.target_projected,
        transcript,
    )

    entries: list[dict[str, Any]] = []
    for attribute, rho in enumerate(transcript.oneway_rho):
        block = transcript.support.attributes[attribute]
        source_block_name = f"oneway_contrast:{attribute}"
        entries.append(
            {
                "label": source_block_name,
                "mechanism": "gaussian_vector",
                "rho": float(rho),
                "public_metadata": {
                    "adjacency": "add_remove",
                    "scope": [attribute],
                    "measurement": "reused_released_oneway_contrast",
                    "original_cardinality": int(block.original_cardinality),
                    "sensitivity_l2": float(
                        np.sqrt(1.0 - 1.0 / block.original_cardinality)
                    ),
                    "coefficient_variance": float(
                        transcript.oneway_variances[attribute]
                    ),
                },
            }
        )
    for observation in transcript.pair_observations:
        entries.append(
            {
                "label": (
                    f"c2_{observation.mode}:"
                    f"{observation.pair[0]}:{observation.pair[1]}"
                ),
                "mechanism": "gaussian_vector",
                "rho": float(observation.rho),
                "public_metadata": {
                    "adjacency": "add_remove",
                    "scope": list(observation.pair),
                    "measurement": observation.mode,
                    "sensitivity_l2": C2_COMPLETE_PARTITION_SENSITIVITY_L2,
                    "coarse_shape": list(observation.noisy_coarse_table.shape),
                    "cell_noise_variance": float(1.0 / (2.0 * observation.rho)),
                },
            }
        )
    entry_sum = float(sum(float(entry["rho"]) for entry in entries))
    if not np.isclose(
        entry_sum,
        transcript.rho_spent,
        rtol=1.0e-11,
        atol=1.0e-12,
    ):
        raise RuntimeError("C2 public ledger entries do not sum to rho_spent")
    epsilon = zcdp_epsilon(transcript.rho_spent, source_measurements.delta)
    ledger = {
        "accounting": "zcdp_actual_spend_v1",
        "accounting_theorem": (
            "adaptive_zcdp_composition_then_epsilon_delta_conversion"
        ),
        "adjacency": "add_remove",
        "rho_limit": float(transcript.rho_total),
        "rho_spent": float(transcript.rho_spent),
        "delta": float(source_measurements.delta),
        "epsilon_from_actual_spend": float(epsilon),
        "entries": entries,
    }
    compatibility_variances = proposal_variances_from_coarse_interaction_transcript(
        qcat,
        source_measurements,
        transcript,
    )
    if compatibility_variances.shape != target.shape or np.any(
        compatibility_variances <= 0.0
    ):
        raise ValueError("C2 compatibility variances are invalid")
    return Measurements(
        target_noisy=target.copy(),
        target_projected=target.copy(),
        variances=compatibility_variances,
        inv_variances=1.0 / compatibility_variances,
        groups=groups,
        mode="dp",
        rho_total=float(transcript.rho_total),
        rho_spent=float(transcript.rho_spent),
        epsilon_delta=float(epsilon),
        delta=float(source_measurements.delta),
        projection_diagnostics={
            "method": "c2_released_support_coarse_interaction_adapter_v1",
            "measurement_mode": transcript.mode,
            "source_pair_observations_excluded": True,
            "serialized_diagonal_variances_active": "proposal_only_marginal_surrogate",
            "lift_conditionals": "released_oneway_simplex_dirichlet1",
            "required_precision_operator": (
                "released_support_coarsened_interaction_full_covariance"
            ),
        },
        num_rows=int(transcript.public_total),
        strategy_transcript=transcript.to_public_dict(),
        privacy_ledger=ledger,
    )


__all__ = [
    "C2_AGGREGATE_MODE",
    "C2_COMPLETE_PARTITION_SENSITIVITY_L2",
    "C2_DIRECT_MODE",
    "CoarseInteractionTranscript",
    "CoarsePairObservation",
    "CoarsePairOperators",
    "coarse_pair_operators",
    "coupled_direct_and_aggregate_observations",
    "measure_paired_coarse_interaction_transcripts",
    "measurements_from_coarse_interaction_transcript",
    "proposal_variances_from_coarse_interaction_transcript",
    "target_from_coarse_interaction_transcript",
]
