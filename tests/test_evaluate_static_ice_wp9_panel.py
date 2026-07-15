from __future__ import annotations

import numpy as np

from scripts import evaluate_static_ice_wp9_panel as evaluator


def test_hierarchical_bootstrap_is_deterministic() -> None:
    values = np.zeros(
        (
            len(evaluator.DATASETS),
            len(evaluator.EPSILONS),
            len(evaluator.SEEDS),
        ),
        dtype=np.float64,
    )
    values[0, 0, 0] = np.log(0.9)
    first = evaluator.hierarchical_bootstrap(values, replicates=200, seed=7)
    second = evaluator.hierarchical_bootstrap(values, replicates=200, seed=7)
    assert first == second
    assert first["lower"] > 0.0
    assert first["upper"] > 0.0


def test_wp9_gate_requires_every_frozen_condition() -> None:
    cells = {f"cell-{index}": 0.9 for index in range(8)}
    datasets = {dataset: 0.95 for dataset in evaluator.DATASETS}
    tails = {
        dataset: {metric: 1.0 for metric in evaluator.TAIL_METRICS}
        for dataset in evaluator.DATASETS
    }
    passing = evaluator.apply_gate(
        cell_ratios=cells,
        overall_primary=0.95,
        bootstrap_upper=0.99,
        dataset_primary=datasets,
        dataset_tail=tails,
    )
    assert passing["passed"]

    cells["cell-0"] = 1.1
    cells["cell-1"] = 1.1
    cells["cell-2"] = 1.1
    rejected = evaluator.apply_gate(
        cell_ratios=cells,
        overall_primary=0.95,
        bootstrap_upper=0.99,
        dataset_primary=datasets,
        dataset_tail=tails,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["at_least_6_of_8_cell_wins"]


def test_wp9_gate_rejects_dataset_and_tail_regression() -> None:
    cells = {f"cell-{index}": 0.9 for index in range(8)}
    datasets = {dataset: 0.95 for dataset in evaluator.DATASETS}
    tails = {
        dataset: {metric: 1.0 for metric in evaluator.TAIL_METRICS}
        for dataset in evaluator.DATASETS
    }
    datasets["adult"] = 1.051
    tails["acs"][evaluator.TAIL_METRICS[0]] = 1.151
    gate = evaluator.apply_gate(
        cell_ratios=cells,
        overall_primary=0.95,
        bootstrap_upper=0.99,
        dataset_primary=datasets,
        dataset_tail=tails,
    )
    assert not gate["passed"]
    assert not gate["checks"]["every_dataset_primary_at_most_1p05"]
    assert not gate["checks"]["every_dataset_tail_at_most_1p15"]
