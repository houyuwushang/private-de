import math

import numpy as np

from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    compile_hierarchical_pair_strategy,
)
from scripts.audit_static_ice_wp9_public_routes import (
    maximum_spanning_tree,
    released_tree_scores,
)


def test_maximum_spanning_tree_is_deterministic_and_reports_margin() -> None:
    scores = {
        (0, 1): 10.0,
        (0, 2): 8.0,
        (0, 3): 1.0,
        (1, 2): 7.0,
        (1, 3): 6.0,
        (2, 3): 5.0,
    }
    tree, margin = maximum_spanning_tree(4, scores)
    assert tree == ((0, 1), (0, 2), (1, 3))
    assert margin == 1.0


def test_zero_score_tree_uses_lexicographic_ties_without_false_margin() -> None:
    scores = {
        (0, 1): 0.0,
        (0, 2): 0.0,
        (1, 2): 0.0,
    }
    tree, margin = maximum_spanning_tree(3, scores)
    assert tree == ((0, 1), (0, 2))
    assert margin == 0.0


def test_released_tree_scores_use_debiased_chi_square_excess() -> None:
    strategy = compile_hierarchical_pair_strategy((2, 3, 2), [(0, 1), (0, 2), (1, 2)])
    components = {
        block.name: np.zeros(block.coefficient_shape, dtype=np.float64)
        for block in strategy.blocks
    }
    variances = {block.name: 2.0 for block in strategy.blocks}
    rho = {
        block.name: block.sensitivity_l2**2 / (2.0 * variances[block.name])
        for block in strategy.blocks
    }
    components["pair_interaction:0:1"] = np.asarray([[3.0, 1.0]])
    transcript = HierarchicalInteractionTranscript(
        strategy=strategy,
        public_total=100,
        noisy_components=components,
        component_variances=variances,
        rho_by_block=rho,
        rho_total=sum(rho.values()),
        rho_spent=sum(rho.values()),
        allocation_mode="public_optimal",
    )
    total = released_tree_scores(transcript, "total_excess")
    standardized = released_tree_scores(transcript, "null_standardized")
    expected_excess = (3.0**2 + 1.0**2) / 2.0 - 2.0
    assert total[(0, 1)] == expected_excess
    assert standardized[(0, 1)] == expected_excess / math.sqrt(4.0)
    assert total[(0, 2)] == 0.0
    assert total[(1, 2)] == 0.0
