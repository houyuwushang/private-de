from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    compile_hierarchical_pair_strategy,
)
from qdte.queries.orthogonal import interaction_coefficients, interaction_sensitivity


@dataclass(frozen=True)
class InteractionActionRelease:
    pair: tuple[int, int]
    noisy_coefficients: np.ndarray
    coefficient_variance: float
    rho: float
    sensitivity_l2: float

    @property
    def name(self) -> str:
        return f"pair_interaction:{self.pair[0]}:{self.pair[1]}"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pair": list(self.pair),
            "coefficient_shape": list(self.noisy_coefficients.shape),
            "coefficient_variance": self.coefficient_variance,
            "rho": self.rho,
            "sensitivity_l2": self.sensitivity_l2,
            "noisy_coefficients": self.noisy_coefficients.tolist(),
        }


def _validate_rows(
    rows: np.ndarray,
    cardinalities: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    table = np.asarray(rows, dtype=np.int64)
    cards = np.asarray(cardinalities, dtype=np.int64)
    if table.ndim != 2 or cards.shape != (table.shape[1],) or np.any(cards < 2):
        raise ValueError("rows and cardinalities have incompatible shapes")
    for attr, cardinality in enumerate(cards.tolist()):
        if np.any(table[:, attr] < 0) or np.any(table[:, attr] >= cardinality):
            raise ValueError(f"rows contain out-of-domain values for attribute {attr}")
    return table, cards


def _canonical_pair(pair: Sequence[int], width: int) -> tuple[int, int]:
    values = tuple(int(value) for value in pair)
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError("pair must contain two distinct attributes")
    left, right = sorted(values)
    if left < 0 or right >= int(width):
        raise ValueError("pair contains an out-of-range attribute")
    return left, right


def measure_interaction_action(
    rows: np.ndarray,
    pair: Sequence[int],
    cardinalities: Sequence[int],
    *,
    rho: float,
    rng: np.random.Generator,
) -> InteractionActionRelease:
    """Release one pure pair-interaction vector under add/remove adjacency."""
    table, cards = _validate_rows(rows, cardinalities)
    scope = _canonical_pair(pair, table.shape[1])
    amount = float(rho)
    if not math.isfinite(amount) or amount <= 0.0:
        raise ValueError("rho must be finite and positive")
    sensitivity = interaction_sensitivity(
        (int(cards[scope[0]]), int(cards[scope[1]]))
    )
    variance = sensitivity * sensitivity / (2.0 * amount)
    exact = interaction_coefficients(table, scope, cards)
    noisy = exact + rng.normal(0.0, math.sqrt(variance), size=exact.shape)
    noisy = np.asarray(noisy, dtype=np.float64)
    noisy.setflags(write=False)
    return InteractionActionRelease(
        pair=scope,
        noisy_coefficients=noisy,
        coefficient_variance=float(variance),
        rho=amount,
        sensitivity_l2=float(sensitivity),
    )


def _validate_action_release(
    action: InteractionActionRelease,
    cardinalities: Sequence[int],
) -> tuple[int, int]:
    cards = tuple(int(value) for value in cardinalities)
    pair = _canonical_pair(action.pair, len(cards))
    expected_shape = (cards[pair[0]] - 1, cards[pair[1]] - 1)
    coefficients = np.asarray(action.noisy_coefficients, dtype=np.float64)
    if coefficients.shape != expected_shape or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"interaction action {pair} has invalid coefficients")
    expected_sensitivity = interaction_sensitivity(
        (cards[pair[0]], cards[pair[1]])
    )
    amount = float(action.rho)
    if not math.isfinite(amount) or amount <= 0.0:
        raise ValueError(f"interaction action {pair} has invalid rho")
    expected_variance = expected_sensitivity**2 / (2.0 * amount)
    if not np.isclose(
        action.sensitivity_l2,
        expected_sensitivity,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(f"interaction action {pair} has invalid sensitivity")
    if not np.isclose(
        action.coefficient_variance,
        expected_variance,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(f"interaction action {pair} has invalid variance/rho")
    return pair


def assemble_adaptive_interaction_transcript(
    anchors: HierarchicalInteractionTranscript,
    actions: Sequence[InteractionActionRelease],
    *,
    rho_total: float,
) -> HierarchicalInteractionTranscript:
    """Combine one-way anchors and distinct released pair actions."""
    if anchors.strategy.pairs:
        raise ValueError("anchors must contain one-way blocks only")
    if anchors.allocation_mode not in {"equal", "public_optimal"}:
        raise ValueError("anchors must use a static public allocation")
    total = float(rho_total)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("rho_total must be finite and positive")
    cards = anchors.strategy.cardinalities
    by_pair: dict[tuple[int, int], InteractionActionRelease] = {}
    for action in actions:
        pair = _validate_action_release(action, cards)
        if pair in by_pair:
            raise ValueError(f"duplicate adaptive interaction action {pair}")
        by_pair[pair] = action

    strategy = compile_hierarchical_pair_strategy(cards, tuple(sorted(by_pair)))
    noisy_components: dict[str, np.ndarray] = {}
    component_variances: dict[str, float] = {}
    rho_by_block: dict[str, float] = {}
    for block in strategy.blocks:
        if block.kind == "oneway_contrast":
            noisy_components[block.name] = np.asarray(
                anchors.noisy_components[block.name], dtype=np.float64
            ).copy()
            component_variances[block.name] = float(
                anchors.component_variances[block.name]
            )
            rho_by_block[block.name] = float(anchors.rho_by_block[block.name])
        elif block.kind == "pair_interaction":
            action = by_pair[(int(block.scope[0]), int(block.scope[1]))]
            noisy_components[block.name] = np.asarray(
                action.noisy_coefficients, dtype=np.float64
            ).copy()
            component_variances[block.name] = float(action.coefficient_variance)
            rho_by_block[block.name] = float(action.rho)
        else:
            raise RuntimeError(f"unsupported strategy block kind {block.kind!r}")

    spent = float(math.fsum(rho_by_block.values()))
    tolerance = 1.0e-12 * max(1.0, total)
    if spent > total + tolerance:
        raise RuntimeError(
            f"adaptive interaction transcript overspends rho: spent={spent}, total={total}"
        )
    return HierarchicalInteractionTranscript(
        strategy=strategy,
        public_total=int(anchors.public_total),
        noisy_components=noisy_components,
        component_variances=component_variances,
        rho_by_block=rho_by_block,
        rho_total=total,
        rho_spent=spent,
        allocation_mode="adaptive_distinct",
        adjacency="add_remove",
    )


def combine_coverage_refinements(
    base: HierarchicalInteractionTranscript,
    refinements: Sequence[InteractionActionRelease],
    *,
    rho_total: float,
) -> HierarchicalInteractionTranscript:
    """Precision-combine independent refinements without dropping base scopes."""
    if base.adjacency != "add_remove":
        raise ValueError("coverage refinement requires add_remove adjacency")
    if base.allocation_mode not in {"equal", "public_optimal"}:
        raise ValueError("base must be a static hierarchical interaction release")
    if not base.strategy.pairs:
        raise ValueError("coverage refinement requires a broad pair strategy")
    total = float(rho_total)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("rho_total must be finite and positive")

    cards = base.strategy.cardinalities
    supported_pairs = set(base.strategy.pairs)
    by_pair: dict[tuple[int, int], list[InteractionActionRelease]] = {}
    for action in refinements:
        pair = _validate_action_release(action, cards)
        if pair not in supported_pairs:
            raise ValueError(f"refinement pair {pair} is outside the broad base")
        by_pair.setdefault(pair, []).append(action)

    noisy_components: dict[str, np.ndarray] = {}
    component_variances: dict[str, float] = {}
    rho_by_block: dict[str, float] = {}
    observation_counts: dict[str, int] = {}
    refinement_rho_by_block: dict[str, float] = {}
    for block in base.strategy.blocks:
        base_coefficients = np.asarray(
            base.noisy_components[block.name], dtype=np.float64
        )
        base_variance = float(base.component_variances[block.name])
        base_rho = float(base.rho_by_block[block.name])
        if block.kind == "oneway_contrast":
            noisy_components[block.name] = base_coefficients.copy()
            component_variances[block.name] = base_variance
            rho_by_block[block.name] = base_rho
            observation_counts[block.name] = 1
            refinement_rho_by_block[block.name] = 0.0
            continue
        if block.kind != "pair_interaction":
            raise RuntimeError(f"unsupported strategy block kind {block.kind!r}")

        pair = (int(block.scope[0]), int(block.scope[1]))
        pair_refinements = by_pair.get(pair, [])
        precision_sum = 1.0 / base_variance
        weighted_sum = base_coefficients / base_variance
        refinement_rho = 0.0
        for action in pair_refinements:
            variance = float(action.coefficient_variance)
            precision_sum += 1.0 / variance
            weighted_sum = weighted_sum + np.asarray(
                action.noisy_coefficients, dtype=np.float64
            ) / variance
            refinement_rho += float(action.rho)
        combined_variance = 1.0 / precision_sum
        effective_rho = base_rho + refinement_rho
        expected_variance = block.sensitivity_l2**2 / (2.0 * effective_rho)
        if not np.isclose(
            combined_variance,
            expected_variance,
            rtol=1.0e-11,
            atol=1.0e-14,
        ):
            raise RuntimeError(
                f"precision/rho combination mismatch for {block.name}"
            )
        noisy_components[block.name] = weighted_sum / precision_sum
        component_variances[block.name] = float(combined_variance)
        rho_by_block[block.name] = float(effective_rho)
        observation_counts[block.name] = 1 + len(pair_refinements)
        refinement_rho_by_block[block.name] = float(refinement_rho)

    refinement_spent = float(math.fsum(float(action.rho) for action in refinements))
    spent = float(base.rho_spent + refinement_spent)
    tolerance = 1.0e-12 * max(1.0, total)
    if spent > total + tolerance:
        raise RuntimeError(
            f"coverage refinement overspends rho: spent={spent}, total={total}"
        )
    summed_rho = float(math.fsum(rho_by_block.values()))
    if not np.isclose(summed_rho, spent, rtol=1.0e-11, atol=1.0e-14):
        raise RuntimeError("combined block rho does not match actual Gaussian spend")

    return HierarchicalInteractionTranscript(
        strategy=base.strategy,
        public_total=int(base.public_total),
        noisy_components=noisy_components,
        component_variances=component_variances,
        rho_by_block=rho_by_block,
        rho_total=total,
        rho_spent=spent,
        allocation_mode="coverage_refined",
        adjacency="add_remove",
        refinement_diagnostics={
            "method": "independent_precision_weighted_combination",
            "base_rho_spent": float(base.rho_spent),
            "refinement_rho_spent": refinement_spent,
            "num_refinement_observations": len(refinements),
            "observation_counts": observation_counts,
            "refinement_rho_by_block": refinement_rho_by_block,
        },
    )
