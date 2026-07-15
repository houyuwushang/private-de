from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.queries.orthogonal import interaction_coefficients, oneway_contrast_from_counts
from scripts import run_static_ice_wp10a_workload_allocation as runner


def _args(**changes):
    values = {
        "dataset": "adult",
        "epsilon": 0.1,
        "delta": 1.0e-9,
        "seed": 0,
        "stage_iters": 5_000,
        "max_pair_cells": 20_000,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_wp10a_request_is_fail_closed() -> None:
    runner.validate_request(_args())
    with pytest.raises(ValueError, match="dataset"):
        runner.validate_request(_args(dataset="acs"))
    with pytest.raises(ValueError, match="epsilon"):
        runner.validate_request(_args(epsilon=0.3))
    with pytest.raises(ValueError, match="stage_iters"):
        runner.validate_request(_args(stage_iters=50))


def test_wp10a_control_reproduction_and_noise_coupling_helpers() -> None:
    cards = (2, 2)
    rows = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]] * 3, dtype=np.int32)
    strategy = compile_hierarchical_pair_strategy(cards, [(0, 1)])
    control = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=0.3,
        rng=np.random.default_rng(7),
        allocation_mode="public_optimal",
    )
    restored = HierarchicalInteractionTranscript.from_public_dict(control.to_public_dict())
    reproduction = runner._transcript_reproduction(control, restored)
    assert reproduction["passed"]

    candidate = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=0.3,
        rng=np.random.default_rng(7),
        allocation_mode="workload_optimal",
        rho_by_block_override={
            "oneway_contrast:0": 0.12,
            "oneway_contrast:1": 0.08,
            "pair_interaction:0:1": 0.10,
        },
    )
    exact = {
        "oneway_contrast:0": oneway_contrast_from_counts(
            np.bincount(rows[:, 0], minlength=2)
        ),
        "oneway_contrast:1": oneway_contrast_from_counts(
            np.bincount(rows[:, 1], minlength=2)
        ),
        "pair_interaction:0:1": interaction_coefficients(rows, (0, 1), cards),
    }
    coupling = runner._noise_coupling(control, candidate, exact)
    assert coupling["passed"]
    assert coupling["standard_normal_max_abs"] < 1.0e-12


def test_wp10a_generation_source_cannot_evaluate_truth() -> None:
    source = inspect.getsource(runner)
    assert "evaluate_external_synthetic" not in source
    assert "answer_queries" not in source
    assert "true_answers_cache" not in source
    assert runner.EXPECTED_CANDIDATES == 20_480_000
    assert runner.EXPECTED_RISK_RATIO == pytest.approx(0.8455718531684706)
