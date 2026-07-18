from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from qdte.queries.orthogonal import (
    helmert_contrast,
    interaction_coefficients,
    interaction_sensitivity,
    oneway_contrast_from_counts,
    pair_reconstruction_maps,
    reconstruct_oneway,
    reconstruct_pair,
)


AllocationMode = Literal[
    "equal",
    "public_optimal",
    "adaptive_distinct",
    "coverage_refined",
    "workload_optimal",
    "oracle_precision_homotopy",
]


ORACLE_PRECISION_HOMOTOPY_METHOD = "offline_oracle_interaction_precision_homotopy_v1"


@dataclass(frozen=True)
class StrategyBlock:
    name: str
    kind: str
    scope: tuple[int, ...]
    coefficient_shape: tuple[int, ...]
    sensitivity_l2: float
    public_importance: float

    @property
    def dimension(self) -> int:
        return int(np.prod(self.coefficient_shape, dtype=np.int64))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "scope": list(self.scope),
            "coefficient_shape": list(self.coefficient_shape),
            "dimension": self.dimension,
            "sensitivity_l2": self.sensitivity_l2,
            "public_importance": self.public_importance,
        }


@dataclass(frozen=True)
class HierarchicalPairStrategy:
    cardinalities: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    blocks: tuple[StrategyBlock, ...]

    def block(self, name: str) -> StrategyBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)


@dataclass
class ReconstructedMarginals:
    oneway: dict[int, np.ndarray]
    pairs: dict[tuple[int, int], np.ndarray]
    oneway_variances: dict[int, np.ndarray]
    pair_variances: dict[tuple[int, int], np.ndarray]

    def max_consistency_violation(self) -> float:
        violation = 0.0
        for (left, right), table in self.pairs.items():
            violation = max(
                violation,
                float(np.max(np.abs(np.sum(table, axis=1) - self.oneway[left]))),
                float(np.max(np.abs(np.sum(table, axis=0) - self.oneway[right]))),
            )
        return violation


@dataclass
class HierarchicalInteractionTranscript:
    strategy: HierarchicalPairStrategy
    public_total: int
    noisy_components: dict[str, np.ndarray]
    component_variances: dict[str, float]
    rho_by_block: dict[str, float]
    rho_total: float
    rho_spent: float
    allocation_mode: AllocationMode
    adjacency: str = "add_remove"
    refinement_diagnostics: dict[str, Any] | None = None

    def reconstruct(self) -> ReconstructedMarginals:
        cards = self.strategy.cardinalities
        oneway: dict[int, np.ndarray] = {}
        oneway_variances: dict[int, np.ndarray] = {}
        for attr, cardinality in enumerate(cards):
            name = _oneway_name(attr)
            theta = self.noisy_components[name]
            oneway[attr] = reconstruct_oneway(self.public_total, theta, cardinality)
            projector_diagonal = 1.0 - 1.0 / float(cardinality)
            oneway_variances[attr] = np.full(
                cardinality,
                self.component_variances[name] * projector_diagonal,
                dtype=np.float64,
            )

        pairs: dict[tuple[int, int], np.ndarray] = {}
        pair_variances: dict[tuple[int, int], np.ndarray] = {}
        for left, right in self.strategy.pairs:
            interaction_name = _pair_name(left, right)
            pairs[(left, right)] = reconstruct_pair(
                self.public_total,
                self.noisy_components[_oneway_name(left)],
                self.noisy_components[_oneway_name(right)],
                self.noisy_components[interaction_name],
                cards[left],
                cards[right],
            )
            left_map, right_map, interaction_map = pair_reconstruction_maps(
                cards[left], cards[right]
            )
            diagonal = (
                self.component_variances[_oneway_name(left)] * np.sum(left_map * left_map, axis=1)
                + self.component_variances[_oneway_name(right)] * np.sum(right_map * right_map, axis=1)
                + self.component_variances[interaction_name]
                * np.sum(interaction_map * interaction_map, axis=1)
            )
            pair_variances[(left, right)] = diagonal.reshape(cards[left], cards[right])
        return ReconstructedMarginals(
            oneway=oneway,
            pairs=pairs,
            oneway_variances=oneway_variances,
            pair_variances=pair_variances,
        )

    def pair_covariance(self, pair: tuple[int, int]) -> np.ndarray:
        left, right = _canonical_pair(pair, len(self.strategy.cardinalities))
        if (left, right) not in self.strategy.pairs:
            raise KeyError((left, right))
        cards = self.strategy.cardinalities
        left_map, right_map, interaction_map = pair_reconstruction_maps(cards[left], cards[right])
        return (
            self.component_variances[_oneway_name(left)] * (left_map @ left_map.T)
            + self.component_variances[_oneway_name(right)] * (right_map @ right_map.T)
            + self.component_variances[_pair_name(left, right)]
            * (interaction_map @ interaction_map.T)
        )

    def to_public_dict(self) -> dict[str, Any]:
        oracle_homotopy = self.allocation_mode == "oracle_precision_homotopy"
        payload = {
            "adjacency": self.adjacency,
            "accounting": (
                "offline_oracle_diagnostic_no_privacy_claim"
                if oracle_homotopy
                else "zcdp_actual_spend_v1"
            ),
            "accounting_theorem": (
                None
                if oracle_homotopy
                else "gaussian_zcdp_rho_equals_delta2_over_2_noise_variance"
            ),
            "public_total": self.public_total,
            "cardinalities": list(self.strategy.cardinalities),
            "pairs": [list(pair) for pair in self.strategy.pairs],
            "allocation_mode": self.allocation_mode,
            "rho_total": self.rho_total,
            "rho_spent": self.rho_spent,
            "blocks": [
                {
                    **block.to_dict(),
                    "mechanism": (
                        "offline_oracle_precision_homotopy"
                        if oracle_homotopy
                        else "gaussian_vector"
                    ),
                    "accounting": (
                        "offline_oracle_diagnostic_no_privacy_claim"
                        if oracle_homotopy
                        else "gaussian_zcdp_exact_v1"
                    ),
                    "rho": self.rho_by_block[block.name],
                    "coefficient_variance": self.component_variances[block.name],
                    "noisy_coefficients": self.noisy_components[block.name].tolist(),
                }
                for block in self.strategy.blocks
            ],
        }
        if self.refinement_diagnostics is not None:
            payload["refinement_diagnostics"] = self.refinement_diagnostics
        return payload

    @classmethod
    def from_public_dict(cls, data: dict[str, Any]) -> HierarchicalInteractionTranscript:
        if not isinstance(data, dict):
            raise ValueError("Interaction transcript must be a mapping")
        adjacency = str(data.get("adjacency", ""))
        if adjacency != "add_remove":
            raise ValueError("Interaction transcript requires add_remove adjacency")
        public_total = int(data.get("public_total", 0))
        if public_total <= 0:
            raise ValueError("Interaction transcript public_total must be positive")
        cardinalities = tuple(int(value) for value in data.get("cardinalities", []))
        raw_pairs = data.get("pairs", [])
        if not isinstance(raw_pairs, list):
            raise ValueError("Interaction transcript pairs must be a list")
        pairs = tuple(tuple(int(value) for value in pair) for pair in raw_pairs)
        strategy = compile_hierarchical_pair_strategy(cardinalities, pairs)
        allocation_mode = str(data.get("allocation_mode", ""))
        if allocation_mode not in {
            "equal",
            "public_optimal",
            "adaptive_distinct",
            "coverage_refined",
            "workload_optimal",
            "oracle_precision_homotopy",
        }:
            raise ValueError("Interaction transcript has an invalid allocation_mode")
        oracle_homotopy = allocation_mode == "oracle_precision_homotopy"
        rho_total = float(data.get("rho_total", 0.0))
        rho_spent = float(data.get("rho_spent", 0.0))
        if not np.isfinite(rho_total) or rho_total <= 0.0:
            raise ValueError("Interaction transcript rho_total must be positive and finite")
        if not np.isfinite(rho_spent) or rho_spent <= 0.0:
            raise ValueError("Interaction transcript rho_spent must be positive and finite")
        tolerance = 1.0e-10 * max(1.0, rho_total)
        if rho_spent > rho_total + tolerance:
            raise ValueError("Interaction transcript overspends rho_total")

        raw_blocks = data.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise ValueError("Interaction transcript blocks must be a list")
        by_name: dict[str, dict[str, Any]] = {}
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                raise ValueError("Each interaction transcript block must be a mapping")
            name = str(raw.get("name", ""))
            if not name or name in by_name:
                raise ValueError(f"Duplicate or empty interaction transcript block name {name!r}")
            by_name[name] = raw
        expected_names = {block.name for block in strategy.blocks}
        if set(by_name) != expected_names:
            raise ValueError(
                "Interaction transcript block set mismatch: "
                f"missing={sorted(expected_names - set(by_name))}, "
                f"extra={sorted(set(by_name) - expected_names)}"
            )

        noisy_components: dict[str, np.ndarray] = {}
        component_variances: dict[str, float] = {}
        rho_by_block: dict[str, float] = {}
        for block in strategy.blocks:
            raw = by_name[block.name]
            if str(raw.get("kind", "")) != block.kind:
                raise ValueError(f"Interaction transcript kind mismatch for {block.name!r}")
            if tuple(int(value) for value in raw.get("scope", [])) != block.scope:
                raise ValueError(f"Interaction transcript scope mismatch for {block.name!r}")
            if tuple(int(value) for value in raw.get("coefficient_shape", [])) != block.coefficient_shape:
                raise ValueError(f"Interaction transcript shape metadata mismatch for {block.name!r}")
            if int(raw.get("dimension", -1)) != block.dimension:
                raise ValueError(f"Interaction transcript dimension mismatch for {block.name!r}")
            sensitivity = float(raw.get("sensitivity_l2", float("nan")))
            if not np.isclose(sensitivity, block.sensitivity_l2, rtol=1.0e-12, atol=1.0e-12):
                raise ValueError(f"Interaction transcript sensitivity mismatch for {block.name!r}")
            rho = float(raw.get("rho", 0.0))
            variance = float(raw.get("coefficient_variance", 0.0))
            if not np.isfinite(rho) or rho <= 0.0:
                raise ValueError(f"Interaction transcript rho must be positive for {block.name!r}")
            if not np.isfinite(variance) or variance < 0.0:
                raise ValueError(
                    f"Interaction transcript coefficient variance must be nonnegative for {block.name!r}"
                )
            expected_variance = block.sensitivity_l2**2 / (2.0 * rho)
            if oracle_homotopy:
                if block.kind == "oneway_contrast" and not np.isclose(
                    variance,
                    expected_variance,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "Oracle precision homotopy must preserve one-way covariance"
                    )
            elif variance <= 0.0 or not np.isclose(
                variance,
                expected_variance,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"Interaction transcript variance/rho mismatch for {block.name!r}"
                )
            coefficients = np.asarray(raw.get("noisy_coefficients", []), dtype=np.float64)
            if coefficients.shape != block.coefficient_shape or not np.all(np.isfinite(coefficients)):
                raise ValueError(
                    f"Interaction transcript coefficients for {block.name!r} must have shape "
                    f"{block.coefficient_shape} and be finite"
                )
            noisy_components[block.name] = coefficients
            component_variances[block.name] = variance
            rho_by_block[block.name] = rho

        summed_rho = float(sum(rho_by_block.values()))
        if not np.isclose(summed_rho, rho_spent, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("Interaction transcript rho_spent does not match its block ledger")
        refinement_diagnostics = data.get("refinement_diagnostics")
        if refinement_diagnostics is not None and not isinstance(
            refinement_diagnostics, dict
        ):
            raise ValueError("refinement_diagnostics must be a mapping when present")
        if allocation_mode == "coverage_refined":
            if not isinstance(refinement_diagnostics, dict):
                raise ValueError(
                    "coverage_refined transcript requires refinement_diagnostics"
                )
            if refinement_diagnostics.get("method") != (
                "independent_precision_weighted_combination"
            ):
                raise ValueError("coverage_refined transcript has an invalid method")
            base_spent = float(refinement_diagnostics.get("base_rho_spent", -1.0))
            refinement_spent = float(
                refinement_diagnostics.get("refinement_rho_spent", -1.0)
            )
            if min(base_spent, refinement_spent) < 0.0 or not np.isclose(
                base_spent + refinement_spent,
                rho_spent,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("coverage_refined spend diagnostics are inconsistent")
            observation_counts = refinement_diagnostics.get("observation_counts")
            refinement_rho_by_block = refinement_diagnostics.get(
                "refinement_rho_by_block"
            )
            if not isinstance(observation_counts, dict) or set(
                observation_counts
            ) != expected_names:
                raise ValueError("coverage_refined observation counts are incomplete")
            if not isinstance(refinement_rho_by_block, dict) or set(
                refinement_rho_by_block
            ) != expected_names:
                raise ValueError("coverage_refined block spend diagnostics are incomplete")
            if any(int(value) < 1 for value in observation_counts.values()):
                raise ValueError("coverage_refined observation counts must be positive")
        if oracle_homotopy:
            if not isinstance(refinement_diagnostics, dict):
                raise ValueError(
                    "oracle_precision_homotopy requires diagnostic metadata"
                )
            if refinement_diagnostics.get("method") != ORACLE_PRECISION_HOMOTOPY_METHOD:
                raise ValueError("Oracle precision homotopy method metadata is invalid")
            if refinement_diagnostics.get("promotion_eligible") is not False:
                raise ValueError("Oracle precision homotopy must be promotion-ineligible")
            if refinement_diagnostics.get("private_truth_used") is not True:
                raise ValueError("Oracle precision homotopy must declare private truth use")
            gamma = refinement_diagnostics.get("gamma")
            exact = bool(refinement_diagnostics.get("exact_interaction_equality", False))
            if exact:
                if gamma != "infinity":
                    raise ValueError("Exact homotopy metadata requires gamma='infinity'")
                pair_variances = [
                    component_variances[block.name]
                    for block in strategy.blocks
                    if block.kind == "pair_interaction"
                ]
                if not pair_variances or any(value != 0.0 for value in pair_variances):
                    raise ValueError("Exact homotopy requires zero pair variances")
            else:
                numeric_gamma = float(gamma)
                if not np.isfinite(numeric_gamma) or numeric_gamma < 1.0:
                    raise ValueError("Finite homotopy gamma must be finite and at least one")
                if any(
                    component_variances[block.name] <= 0.0
                    for block in strategy.blocks
                    if block.kind == "pair_interaction"
                ):
                    raise ValueError("Finite homotopy requires positive pair variances")
        return cls(
            strategy=strategy,
            public_total=public_total,
            noisy_components=noisy_components,
            component_variances=component_variances,
            rho_by_block=rho_by_block,
            rho_total=rho_total,
            rho_spent=rho_spent,
            allocation_mode=allocation_mode,
            adjacency=adjacency,
            refinement_diagnostics=refinement_diagnostics,
        )


def _oneway_name(attr: int) -> str:
    return f"oneway_contrast:{int(attr)}"


def _pair_name(left: int, right: int) -> str:
    return f"pair_interaction:{int(left)}:{int(right)}"


def _canonical_pair(pair: tuple[int, int], width: int) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError("Each pair scope must contain exactly two attributes")
    left, right = (int(pair[0]), int(pair[1]))
    if left == right or min(left, right) < 0 or max(left, right) >= int(width):
        raise ValueError(f"Invalid pair scope {pair!r}")
    return (left, right) if left < right else (right, left)


def compile_hierarchical_pair_strategy(
    cardinalities: np.ndarray | tuple[int, ...] | list[int],
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> HierarchicalPairStrategy:
    cards = tuple(int(value) for value in np.asarray(cardinalities, dtype=np.int64).tolist())
    if not cards or any(cardinality < 2 for cardinality in cards):
        raise ValueError("Hierarchical interaction strategy requires all cardinalities >= 2")
    canonical_pairs = tuple(sorted({_canonical_pair(pair, len(cards)) for pair in pairs}))
    blocks: list[StrategyBlock] = []
    for attr, cardinality in enumerate(cards):
        importance = float(cardinality - 1)
        for left, right in canonical_pairs:
            if attr == left:
                importance += float(cardinality - 1) / float(cards[right])
            elif attr == right:
                importance += float(cardinality - 1) / float(cards[left])
        blocks.append(
            StrategyBlock(
                name=_oneway_name(attr),
                kind="oneway_contrast",
                scope=(attr,),
                coefficient_shape=(cardinality - 1,),
                sensitivity_l2=interaction_sensitivity((cardinality,)),
                public_importance=importance,
            )
        )
    for left, right in canonical_pairs:
        shape = (cards[left] - 1, cards[right] - 1)
        blocks.append(
            StrategyBlock(
                name=_pair_name(left, right),
                kind="pair_interaction",
                scope=(left, right),
                coefficient_shape=shape,
                sensitivity_l2=interaction_sensitivity((cards[left], cards[right])),
                public_importance=float(np.prod(shape, dtype=np.int64)),
            )
        )
    return HierarchicalPairStrategy(cards, canonical_pairs, tuple(blocks))


def allocate_public_precision(
    sensitivities: np.ndarray | list[float],
    public_importance: np.ndarray | list[float],
    rho_total: float,
) -> np.ndarray:
    sensitivity = np.asarray(sensitivities, dtype=np.float64)
    importance = np.asarray(public_importance, dtype=np.float64)
    if sensitivity.ndim != 1 or importance.shape != sensitivity.shape or sensitivity.size == 0:
        raise ValueError("sensitivities and public_importance must be matching non-empty vectors")
    if not np.all(np.isfinite(sensitivity)) or np.any(sensitivity <= 0.0):
        raise ValueError("sensitivities must be finite and positive")
    if not np.all(np.isfinite(importance)) or np.any(importance <= 0.0):
        raise ValueError("public_importance must be finite and positive")
    total = float(rho_total)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("rho_total must be finite and positive")
    weights = sensitivity * np.sqrt(importance)
    allocation = total * weights / float(np.sum(weights))
    allocation[-1] += total - float(np.sum(allocation))
    return allocation


def allocate_strategy_rho(
    strategy: HierarchicalPairStrategy,
    rho_total: float,
    mode: AllocationMode,
) -> dict[str, float]:
    if mode == "equal":
        allocation = np.full(len(strategy.blocks), float(rho_total) / len(strategy.blocks))
        allocation[-1] += float(rho_total) - float(np.sum(allocation))
    elif mode == "public_optimal":
        allocation = allocate_public_precision(
            [block.sensitivity_l2 for block in strategy.blocks],
            [block.public_importance for block in strategy.blocks],
            rho_total,
        )
    else:
        raise ValueError("allocation mode must be 'equal' or 'public_optimal'")
    return {block.name: float(rho) for block, rho in zip(strategy.blocks, allocation, strict=True)}


def validate_strategy_rho_override(
    strategy: HierarchicalPairStrategy,
    rho_total: float,
    rho_by_block: Mapping[str, float],
) -> dict[str, float]:
    expected = {block.name for block in strategy.blocks}
    if set(rho_by_block) != expected:
        raise ValueError(
            "rho override must cover every strategy block exactly: "
            f"missing={sorted(expected - set(rho_by_block))}, "
            f"extra={sorted(set(rho_by_block) - expected)}"
        )
    allocation = {name: float(rho_by_block[name]) for name in expected}
    if any(not np.isfinite(value) or value <= 0.0 for value in allocation.values()):
        raise ValueError("rho override values must be finite and positive")
    total = float(rho_total)
    spent = float(sum(allocation.values()))
    tolerance = 1.0e-15 * max(1.0, abs(total))
    if not np.isclose(spent, total, rtol=1.0e-11, atol=tolerance):
        raise ValueError(
            f"rho override spends {spent:.17g}, expected rho_total={total:.17g}"
        )
    return {
        block.name: allocation[block.name]
        for block in strategy.blocks
    }


def measure_hierarchical_pair_interactions(
    rows: np.ndarray,
    strategy: HierarchicalPairStrategy,
    *,
    public_total: int,
    rho_total: float,
    rng: np.random.Generator,
    allocation_mode: AllocationMode = "public_optimal",
    rho_by_block_override: Mapping[str, float] | None = None,
) -> HierarchicalInteractionTranscript:
    """Release one-way contrasts and pure pair interactions under add/remove adjacency."""
    X = np.asarray(rows, dtype=np.int64)
    cards = np.asarray(strategy.cardinalities, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != len(cards):
        raise ValueError("rows must match the strategy width")
    if int(public_total) <= 0 or X.shape[0] != int(public_total):
        raise ValueError("rows must match the explicitly supplied positive public_total")
    for attr, cardinality in enumerate(cards.tolist()):
        if np.any(X[:, attr] < 0) or np.any(X[:, attr] >= cardinality):
            raise ValueError(f"rows contain values outside public cardinality for attribute {attr}")

    if rho_by_block_override is None:
        rho_by_block = allocate_strategy_rho(strategy, rho_total, allocation_mode)
    else:
        if allocation_mode != "workload_optimal":
            raise ValueError(
                "rho_by_block_override requires allocation_mode='workload_optimal'"
            )
        rho_by_block = validate_strategy_rho_override(
            strategy,
            rho_total,
            rho_by_block_override,
        )
    noisy_components: dict[str, np.ndarray] = {}
    component_variances: dict[str, float] = {}
    for block in strategy.blocks:
        if block.kind == "oneway_contrast":
            attr = block.scope[0]
            counts = np.bincount(X[:, attr], minlength=cards[attr]).astype(np.float64)
            exact = oneway_contrast_from_counts(counts)
        elif block.kind == "pair_interaction":
            exact = interaction_coefficients(X, block.scope, cards)
        else:
            raise RuntimeError(f"Unsupported strategy block kind {block.kind!r}")
        rho = rho_by_block[block.name]
        variance = block.sensitivity_l2**2 / (2.0 * rho)
        noisy_components[block.name] = exact + rng.normal(
            loc=0.0,
            scale=np.sqrt(variance),
            size=block.coefficient_shape,
        )
        component_variances[block.name] = float(variance)

    rho_spent = float(sum(rho_by_block.values()))
    tolerance = 1.0e-12 * max(1.0, abs(float(rho_total)))
    if rho_spent > float(rho_total) + tolerance:
        raise RuntimeError("Hierarchical interaction measurement exceeded rho_total")
    return HierarchicalInteractionTranscript(
        strategy=strategy,
        public_total=int(public_total),
        noisy_components=noisy_components,
        component_variances=component_variances,
        rho_by_block=rho_by_block,
        rho_total=float(rho_total),
        rho_spent=rho_spent,
        allocation_mode=allocation_mode,
    )
