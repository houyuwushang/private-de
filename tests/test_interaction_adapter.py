from __future__ import annotations

import itertools

import numpy as np

from qdte.measurement.factorization import (
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
    interaction_transcript_to_local_polytope_bootdiag_measurements,
    interaction_transcript_to_local_polytope_measurements,
    interaction_transcript_to_shrunk_measurements,
)
from qdte.measurement.measure import measurements_from_public_dict
from qdte.schema import ColumnSchema, TableSchema
from scripts.run_orthogonal_low_budget_pilot import build_complete_low_order_workload


def test_diagonal_adapter_is_explicit_and_preserves_legacy_objective_shapes() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"]),
            ColumnSchema(name="b", kind="categorical", cardinality=3, categories=["0", "1", "2"]),
        ]
    )
    rows = np.asarray(list(itertools.product(range(2), range(3))) * 5, dtype=np.int32)
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=20)
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, [(0, 1)])
    transcript = measure_hierarchical_pair_interactions(
        rows,
        strategy,
        public_total=len(rows),
        rho_total=0.4,
        rng=np.random.default_rng(19),
    )

    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )

    assert measurements.target_noisy.shape == (qcat.m,)
    assert measurements.target_projected.shape == (qcat.m,)
    assert measurements.variances.shape == (qcat.m,)
    assert np.all(measurements.variances > 0.0)
    assert np.all(np.isfinite(measurements.inv_variances))
    assert np.isclose(measurements.rho_spent, 0.4)
    assert all(group.rho == 0.0 and group.noise_std == 0.0 for group in measurements.groups)
    strategy_diag = measurements.projection_diagnostics["measurement_strategy"]
    assert strategy_diag["covariance_adapter"] == "marginal_diagonal_only"
    assert strategy_diag["covariance_approximation"] is True
    assert measurements.projection_diagnostics["uncertainty"]["drops_cross_scope_covariance"] is True
    assert measurements.strategy_transcript == transcript.to_public_dict()
    assert measurements.privacy_ledger is not None
    assert measurements.privacy_ledger["accounting"] == "zcdp_actual_spend_v1"
    assert measurements.privacy_ledger["adjacency"] == "add_remove"
    assert np.isclose(measurements.privacy_ledger["rho_spent"], 0.4)
    assert len(measurements.privacy_ledger["entries"]) == len(transcript.strategy.blocks)

    restored = measurements_from_public_dict(measurements.to_public_dict())
    assert restored.strategy_transcript == transcript.to_public_dict()
    assert restored.privacy_ledger == measurements.privacy_ledger

    for group in measurements.groups:
        assert np.isclose(np.sum(measurements.target_projected[group.query_indices]), len(rows))


def test_raw_reconstruction_target_preserves_exact_hierarchical_consistency() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"]),
            ColumnSchema(name="b", kind="categorical", cardinality=3, categories=["0", "1", "2"]),
        ]
    )
    rows = np.asarray(list(itertools.product(range(2), range(3))) * 5, dtype=np.int32)
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=20)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, [(0, 1)]),
        public_total=len(rows),
        rho_total=0.05,
        rng=np.random.default_rng(29),
    )

    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )

    np.testing.assert_array_equal(measurements.target_projected, measurements.target_noisy)
    assert measurements.projection_diagnostics["measurement_strategy"]["target_projection"] == (
        "raw_reconstruction"
    )
    assert measurements.projection_diagnostics["consistency"]["method"] == (
        "raw_hierarchical_reconstruction"
    )


def test_local_polytope_adapter_replaces_target_but_labels_raw_uncertainty() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"]),
            ColumnSchema(name="b", kind="categorical", cardinality=3, categories=["0", "1", "2"]),
        ]
    )
    rows = np.asarray(list(itertools.product(range(2), range(3))) * 5, dtype=np.int32)
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=20)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, [(0, 1)]),
        public_total=len(rows),
        rho_total=0.05,
        rng=np.random.default_rng(29),
    )
    transcript.noisy_components["oneway_contrast:0"] += np.asarray([40.0])

    measurements, projection = interaction_transcript_to_local_polytope_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )

    assert projection.diagnostics["certificate_passed"] is True
    assert projection.diagnostics["initial_min_cell"] < 0.0
    assert projection.marginals.min_cell() >= -1.0e-7
    assert not np.array_equal(measurements.target_projected, measurements.target_noisy)
    assert measurements.projection_diagnostics["consistency"]["certificate_passed"] is True
    assert measurements.projection_diagnostics["uncertainty"] == {
        "enabled": True,
        "method": "raw_coefficient_covariance_control",
        "projection_covariance_propagated": False,
    }
    assert measurements.strategy_transcript == transcript.to_public_dict()

    boot_measurements, boot_projection, bootstrap = (
        interaction_transcript_to_local_polytope_bootdiag_measurements(
            transcript,
            qcat,
            groups,
            delta=1.0e-9,
            rng=np.random.default_rng(101),
            num_samples=16,
            min_variance=1.0e-6,
            min_raw_variance_fraction=0.02,
        )
    )
    np.testing.assert_array_equal(
        boot_measurements.target_projected,
        measurements.target_projected,
    )
    assert boot_projection.diagnostics["certificate_passed"] is True
    assert bootstrap.diagnostics["num_samples"] == 16
    assert boot_measurements.projection_diagnostics["uncertainty"][
        "projection_covariance_propagated"
    ] is True
    restored = measurements_from_public_dict(boot_measurements.to_public_dict())
    assert (
        restored.projection_diagnostics["coefficient_bootstrap_diagonal"]["diagnostics"][
            "num_samples"
        ]
        == 16
    )


def test_shrinkage_adapter_preserves_raw_transcript_and_changes_only_pair_target() -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=4),
            ColumnSchema(name="b", kind="categorical", cardinality=3),
        ]
    )
    rows = np.asarray(list(itertools.product(range(4), range(3))) * 5, dtype=np.int32)
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=20)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, [(0, 1)]),
        public_total=len(rows),
        rho_total=0.005,
        rng=np.random.default_rng(20260714),
    )
    raw = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    measurements, shrinkage = interaction_transcript_to_shrunk_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )

    assert measurements.strategy_transcript == transcript.to_public_dict()
    np.testing.assert_array_equal(measurements.target_noisy, raw.target_noisy)
    oneway_indices = np.concatenate(
        [group.query_indices for group in groups if group.family == "oneway"]
    )
    pair_indices = np.concatenate(
        [group.query_indices for group in groups if group.family == "twoway"]
    )
    np.testing.assert_array_equal(
        measurements.target_projected[oneway_indices],
        raw.target_projected[oneway_indices],
    )
    assert not np.array_equal(
        measurements.target_projected[pair_indices],
        raw.target_projected[pair_indices],
    )
    assert shrinkage.diagnostics()["num_affected_pair_blocks"] == 1
    assert measurements.projection_diagnostics["interaction_shrinkage"]["diagnostics"][
        "method"
    ] == "positive_part_block_james_stein_v1"
    assert measurements.projection_diagnostics["uncertainty"][
        "shrinkage_covariance_propagated"
    ] is False
    restored = measurements_from_public_dict(measurements.to_public_dict())
    assert restored.strategy_transcript == transcript.to_public_dict()
