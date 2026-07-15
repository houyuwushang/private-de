from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from qdte.measurement.factorization import compile_hierarchical_pair_strategy
from qdte.measurement.gcea import GCEAProfile, optimize_gcea_allocation
from scripts import run_static_ice_wp10b_gcea_public_gate as runner


def _passing_fixture() -> tuple[GCEAProfile, object]:
    strategy = compile_hierarchical_pair_strategy((2, 2), [])
    profile = GCEAProfile(
        strategy=strategy,
        public_total=1_000,
        beta_tail=0.05,
        query_ids=np.asarray([0, 1], dtype=np.int64),
        unit_variance=csr_matrix(np.asarray([[9.0, 0.0], [0.0, 1.0]])),
        fixed_variance=np.zeros(2, dtype=np.float64),
        partition_names=("pair",),
        partition_rows=(np.asarray([0, 1], dtype=np.int64),),
        movable_block_indices=np.asarray([0, 1], dtype=np.int64),
        fixed_block_indices=np.asarray([], dtype=np.int64),
        control_rho=np.asarray([0.5, 0.5], dtype=np.float64),
        reconstruction_max_abs=0.0,
        unsupported_query_ids=(),
    )
    return profile, optimize_gcea_allocation(profile)


def test_wp10b_protocol_and_gate_constants_are_frozen() -> None:
    assert runner.PROTOCOL_ID == "SAGE-QDTE-ICE-WP10B-GCEA-20260715-v1"
    assert runner.CANDIDATE_ID == "SAGE-QDTE-Static-ICE-GCEA-v1"
    assert runner.DATASETS == ("adult", "br2000")
    assert runner.EPSILONS == (0.1, 0.3)
    assert runner.DELTA_DP == 1.0e-9
    assert runner.BETA_TAIL == 0.05
    assert runner.T_STAR_MAX == 0.97
    assert runner.KKT_GAP_MAX == 1.0e-8
    assert runner.RECONSTRUCTION_RESIDUAL_MAX == 1.0e-10


def test_wp10b_public_runner_has_no_private_or_transcript_loader() -> None:
    source = inspect.getsource(runner)
    assert "np.load" not in source
    assert "real_encoded" not in source
    assert "measurements.json" not in source
    assert "offline_true" not in source
    assert "evaluate_external" not in source


def test_wp10b_public_plan_hashes_only_declared_public_inputs() -> None:
    packaged_input_root = Path(__file__).resolve().parents[1] / "external_inputs"
    input_root = (
        packaged_input_root
        if (packaged_input_root / "adult_sage_strong" / "queries_full.json").is_file()
        else runner._external_input_root()
    )
    plan = runner.build_plan(input_root)
    assert plan["private_table_read"] is False
    assert plan["released_transcript_read"] is False
    assert plan["true_utility_evaluated"] is False
    for dataset in runner.DATASETS:
        assert set(plan["public_inputs"][dataset]) == {
            "schema.json",
            "metadata.json",
            "queries_full.json",
            "workload_groups.json",
        }
        assert all(
            len(record["sha256"]) == 64
            for record in plan["public_inputs"][dataset].values()
        )


def test_wp10b_public_gate_requires_every_frozen_certificate() -> None:
    profile, result = _passing_fixture()
    checks = runner.public_gate_checks(profile, result)
    assert all(checks.values())

    tail_failure = replace(
        result,
        tail_ratios={**result.tail_ratios, "max_error": 1.0001},
    )
    assert runner.public_gate_checks(profile, tail_failure)[
        "max_error_envelope_not_worse"
    ] is False

    kkt_failure = replace(result, kkt_gap=1.1e-8)
    assert runner.public_gate_checks(profile, kkt_failure)["allocation_kkt_gap"] is False
