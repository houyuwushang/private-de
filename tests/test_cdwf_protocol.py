from __future__ import annotations

import itertools

import numpy as np
import pytest

from qdte.measurement.cdwf import derive_cdwf_budget_plan
from qdte.measurement.cdwf_protocol import run_cdwf_adaptive_measurement
from qdte.measurement.factorization import compile_hierarchical_pair_strategy
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import ColumnSchema, TableSchema


def test_cdwf_uniform_and_dual_share_noise_streams_on_single_block() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2),
            ColumnSchema(name="b", kind="categorical", cardinality=2),
        ]
    )
    rows = np.asarray(list(itertools.product(range(2), repeat=2)) * 40, dtype=np.int32)
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, ((0, 1),))
    control = {block.name: 0.2 for block in strategy.blocks}
    plan = derive_cdwf_budget_plan(strategy, control, epsilon=0.1)
    qcat, groups = build_selected_pair_partition_workload(schema, strategy.pairs)
    common = {
        "rows": rows,
        "schema": schema,
        "qcat": qcat,
        "workload_groups": groups,
        "strategy": strategy,
        "plan": plan,
        "public_total": len(rows),
        "base_noise_seed": 101,
        "refinement_noise_seed": 202,
        "generation_seed": 303,
        "shadow_max_iterations": 2_000,
    }

    uniform = run_cdwf_adaptive_measurement(**common, arm="split_uniform")
    dual = run_cdwf_adaptive_measurement(**common, arm="dual_water_fill")

    assert uniform.transcript.budget.frozen_blocks
    assert uniform.transcript.final_rho_by_block() == pytest.approx(control)
    assert dual.transcript.final_rho_by_block() == pytest.approx(control)
    assert uniform.dual_fallback_count == 0
    assert dual.dual_fallback_count == 0
    assert dual.promotion_eligible_dual is True
    assert dual.shadow_dictionary is not None
    assert dual.shadow_dictionary.path_diagnostics["entered_confidence_set"] is True
    assert all(record.shadow is not None for record in dual.rounds)
    assert all(
        record.shadow["dual"]["eligible"] is True for record in dual.rounds
    )
    for uniform_stream, dual_stream in zip(
        uniform.transcript.streams,
        dual.transcript.streams,
        strict=True,
    ):
        assert uniform_stream.rho_spent == pytest.approx(dual_stream.rho_spent)
        for name, uniform_observation in uniform_stream.by_name().items():
            dual_observation = dual_stream.by_name()[name]
            assert uniform_observation.rho == pytest.approx(dual_observation.rho)
            np.testing.assert_array_equal(
                uniform_observation.noisy_coefficients,
                dual_observation.noisy_coefficients,
            )
    payload = dual.to_public_dict()
    assert payload["selection_rho"] == 0.0
    assert payload["truth_accessed_by_allocation"] is False
    assert payload["per_round_full_qdte"] is False
    assert len(payload["rounds"]) == 4
