from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qdte.privacy.accountant import ZCDPPrivacyFilter
from scripts import run_static_ice_wp9_cell as runner


def test_wp9_request_is_fail_closed() -> None:
    runner.validate_request(
        dataset="adult",
        epsilon=0.1,
        delta=1.0e-9,
        seed=2,
        max_pair_cells=20_000,
        stage_iters=5_000,
    )
    with pytest.raises(ValueError, match="epsilon"):
        runner.validate_request(
            dataset="adult",
            epsilon=1.0,
            delta=1.0e-9,
            seed=2,
            max_pair_cells=20_000,
            stage_iters=5_000,
        )
    with pytest.raises(ValueError, match="stage_iters"):
        runner.validate_request(
            dataset="adult",
            epsilon=0.1,
            delta=1.0e-9,
            seed=2,
            max_pair_cells=20_000,
            stage_iters=50,
        )


def test_wp9_mechanism_gate_checks_exact_objective_and_full_compute(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "measurement"
    generation = tmp_path / "generate"
    artifact.mkdir()
    generation.mkdir()
    payload = b"released transcript"
    (artifact / "measurements.json").write_bytes(payload)
    (generation / "measurements.json").write_bytes(payload)

    transcript = SimpleNamespace(
        rho_spent=0.2,
        component_variances={"oneway:0": 2.0, "pair:0:1": 3.0},
        strategy=SimpleNamespace(blocks=(object(), object())),
    )
    ledger = ZCDPPrivacyFilter(0.2)
    ledger.spend(label="measurement", mechanism="gaussian", rho=0.2)
    metrics = {
        "precision_operator": "orthogonal_interaction",
        "precision_operator_active": True,
        "precision_operator_diagnostics": {"effective_rank": 2},
        "num_candidates_requested": runner.EXPECTED_CANDIDATES,
        "num_candidates_scored": runner.EXPECTED_CANDIDATES,
        "final_incremental_answer_drift": 0.0,
    }
    run_status = {"num_iterations": runner.STAGE_ITERS}
    runtime = {
        "precision_operator": "orthogonal_interaction",
        "score_backend": "precision_operator",
        "num_candidates_requested": runner.EXPECTED_CANDIDATES,
        "num_candidates_scored": runner.EXPECTED_CANDIDATES,
    }
    gate = runner._mechanism_gate(
        transcript=transcript,
        ledger=ledger,
        rho_total=0.2,
        pairs=((0, 1),),
        cards=np.asarray([2, 2], dtype=np.int32),
        metrics=metrics,
        run_status=run_status,
        runtime=runtime,
        artifact_dir=artifact,
        generation_dir=generation,
    )
    assert gate["passed"]

    runtime["num_candidates_scored"] -= 1
    rejected = runner._mechanism_gate(
        transcript=transcript,
        ledger=ledger,
        rho_total=0.2,
        pairs=((0, 1),),
        cards=np.asarray([2, 2], dtype=np.int32),
        metrics=metrics,
        run_status=run_status,
        runtime=runtime,
        artifact_dir=artifact,
        generation_dir=generation,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["full_candidate_count"]


def test_wp9_generation_source_cannot_evaluate_truth() -> None:
    source = inspect.getsource(runner)
    assert "evaluate_external_synthetic" not in source
    assert "answer_queries" not in source
    assert "exact_low_order_marginals" not in source
    assert '"true_utility_evaluated": False' in source
    assert runner.EXPECTED_CANDIDATES == 20_480_000
