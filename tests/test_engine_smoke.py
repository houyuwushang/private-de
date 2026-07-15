from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import orjson
import pytest

from qdte.evolution.engine import (
    _run_rng_streams,
    _scheduled_accept_limit,
    _search_adjusted_noise_guard_kappa,
    run_qdte,
)
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
from qdte.queries.types import QueryCatalogue, query_key
from qdte.schema import ColumnSchema, TableSchema
from scripts.run_orthogonal_low_budget_pilot import build_complete_low_order_workload
from scripts.run_static_ice_qdte_pilot import _generation_config, _write_measurement_artifact


def test_scheduled_accept_limit_cosine_anneals_to_one() -> None:
    cfg = {
        "accepted_per_iter_schedule": "cosine",
        "accepted_per_iter_start": 8,
        "accepted_per_iter_end": 1,
        "accepted_per_iter_anneal_iters": 5,
    }

    limits = [
        _scheduled_accept_limit(iteration=i, max_iters=5, base_accept=8, qdte_cfg=cfg)
        for i in range(1, 6)
    ]

    assert limits[0] == 8
    assert limits[-1] == 1
    assert limits == sorted(limits, reverse=True)


def test_search_adjusted_noise_guard_uses_candidate_family_size() -> None:
    fixed = _search_adjusted_noise_guard_kappa(
        mode="fixed",
        fixed_kappa=2.0,
        family_size=2048,
        alpha=0.05,
    )
    corrected = _search_adjusted_noise_guard_kappa(
        mode="bonferroni",
        fixed_kappa=2.0,
        family_size=2048,
        alpha=0.05,
    )

    assert fixed == 2.0
    assert 4.0 < corrected < 4.2


def test_measurement_random_draws_do_not_perturb_generation_rng_stream() -> None:
    measurement_rng, generation_rng = _run_rng_streams(17)
    measurement_rng.normal(size=10_000)
    generation_draws_after_measurement = generation_rng.integers(0, 1_000_000, size=32)

    _, fresh_generation_rng = _run_rng_streams(17)
    fresh_generation_draws = fresh_generation_rng.integers(0, 1_000_000, size=32)

    assert np.array_equal(generation_draws_after_measurement, fresh_generation_draws)


def test_split_rng_preserves_historical_generation_seed_mapping() -> None:
    _, generation_rng = _run_rng_streams(17)
    expected_rng = np.random.default_rng(17)

    assert np.array_equal(
        generation_rng.integers(0, 1_000_000, size=32),
        expected_rng.integers(0, 1_000_000, size=32),
    )


def test_disabled_structured_swap_preserves_base_output(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2],
            "b": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )
    data_path = tmp_path / "base_compat.csv"
    df.to_csv(data_path, index=False)
    base_out = tmp_path / "base"
    disabled_out = tmp_path / "disabled"
    config = {
        "run": {
            "dataset_name": "base_compat",
            "input_csv": str(data_path),
            "output_dir": str(base_out),
            "seed": 23,
        },
        "preprocess": {
            "categorical_columns": ["a", "b"],
            "numerical_columns": [],
        },
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 32,
            "max_terms": 2,
            "max_2way_cells": 32,
        },
        "privacy": {
            "mode": "dp",
            "rho_total": 1.0,
            "delta": 1.0e-9,
            "measurement_allocation": {"oneway": 0.5, "twoway": 0.5},
        },
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 4,
            "num_active_targets": 4,
            "total_candidates_per_iter": 32,
            "accepted_per_iter": 4,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "full_recompute_every": 2,
            "stop_patience": 4,
            "min_advantage": 1.0e-6,
            "log_every": 4,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 32,
            "answer_batch_size": 32,
            "xla_preallocate": False,
        },
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }

    base_metrics = run_qdte(config)
    disabled_config = dict(config)
    disabled_config["run"] = dict(config["run"], output_dir=str(disabled_out))
    disabled_config["qdte"] = dict(config["qdte"], structured_swap_enabled=False)
    disabled_metrics = run_qdte(disabled_config)

    assert np.array_equal(
        np.load(base_out / "synthetic_initial_encoded.npy"),
        np.load(disabled_out / "synthetic_initial_encoded.npy"),
    )
    assert np.array_equal(
        np.load(base_out / "synthetic_encoded.npy"),
        np.load(disabled_out / "synthetic_encoded.npy"),
    )
    assert base_metrics["final_measured_loss"] == disabled_metrics["final_measured_loss"]
    assert base_metrics["num_candidates_scored"] == disabled_metrics["num_candidates_scored"]
    assert disabled_metrics["structured_swap_total_candidates"] == 0


def test_dp_release_engine_uses_declared_public_row_count_and_writes_ledger(
    tmp_path: Path,
) -> None:
    rows = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]],
        dtype=np.int32,
    )
    data_path = tmp_path / "release.csv"
    pd.DataFrame(rows, columns=["a", "b"]).to_csv(data_path, index=False)
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"]),
            ColumnSchema(name="b", kind="categorical", cardinality=2, categories=["0", "1"]),
        ]
    )
    schema_path = tmp_path / "schema.json"
    schema.save_json(schema_path)
    output_dir = tmp_path / "release_run"
    config = {
        "run": {
            "dataset_name": "release_contract",
            "input_csv": str(data_path),
            "output_dir": str(output_dir),
            "seed": 7,
        },
        "preprocess": {"public_schema_json": str(schema_path)},
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 32,
            "max_terms": 2,
            "max_2way_cells": 16,
        },
        "privacy": {
            "mode": "dp",
            "dp_release_mode": True,
            "public_row_count": True,
            "public_n_rows": len(rows),
            "adjacency": "add_remove",
            "rho_total": 0.1,
            "delta": 1.0e-9,
            "measurement_allocation": {"oneway": 0.5, "twoway": 0.5},
        },
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 1,
            "num_active_targets": 4,
            "total_candidates_per_iter": 16,
            "accepted_per_iter": 2,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "full_recompute_every": 1,
            "stop_patience": 1,
            "min_advantage": 0.0,
            "log_every": 1,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 16,
            "answer_batch_size": 32,
            "xla_preallocate": False,
        },
        "evaluation": {
            "compute_true_query_error": False,
            "compute_heldout_query_error": False,
            "downstream_ml": False,
            "save_synthetic_csv": False,
        },
    }

    metrics = run_qdte(config)
    measurement = orjson.loads((output_dir / "measurements.json").read_bytes())

    assert metrics["dp_release_mode"] is True
    assert metrics["public_n_rows_declared"] == len(rows)
    assert metrics["num_rows_synthetic"] == len(rows)
    assert measurement["num_rows"] == len(rows)
    assert measurement["privacy_ledger"]["accounting"] == "zcdp_actual_spend_v1"
    assert measurement["privacy_ledger"]["adjacency"] == "add_remove"
    assert measurement["privacy_ledger"]["rho_spent"] == pytest.approx(0.1)

    mismatch = copy.deepcopy(config)
    mismatch["run"]["output_dir"] = str(tmp_path / "release_mismatch")
    mismatch["privacy"]["public_n_rows"] = len(rows) + 1
    with pytest.raises(ValueError, match="does not match declared"):
        run_qdte(mismatch)


def test_exact_orthogonal_precision_engine_smoke(tmp_path: Path) -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name=name,
                kind="categorical",
                cardinality=2,
                categories=["0", "1"],
            )
            for name in ("a", "b", "c")
        ]
    )
    rng = np.random.default_rng(71)
    rows = rng.integers(0, 2, size=(96, 3), dtype=np.int32)
    data_path = tmp_path / "raw.csv"
    pd.DataFrame(rows, columns=["a", "b", "c"]).to_csv(data_path, index=False)
    schema_path = tmp_path / "schema.json"
    schema.save_json(schema_path)
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=16)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, [(0, 1), (0, 2), (1, 2)]),
        public_total=len(rows),
        rho_total=0.2,
        rng=np.random.default_rng(72),
        allocation_mode="public_optimal",
    )
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    artifact = tmp_path / "measurement"
    _write_measurement_artifact(
        artifact,
        qcat=qcat,
        schema=schema,
        measurements=measurements,
    )
    output_dir = tmp_path / "run"
    config = {
        "run": {
            "dataset_name": "exact_precision_smoke",
            "input_csv": str(data_path),
            "output_dir": str(output_dir),
            "seed": 73,
        },
        "preprocess": {
            "public_schema_json": str(schema_path),
            "force_all_categorical": True,
        },
        "workload": {"reuse_from_measurement": True},
        "privacy": {
            "mode": "dp",
            "adjacency": "add_remove",
            "rho_total": 0.2,
            "delta": 1.0e-9,
            "measurement_mode": "static_all",
        },
        "measurement": {"reuse_from": str(artifact), "fission": {"enabled": False}},
        "projection": {"uncertainty": {"enabled": False}},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "precision_operator": "orthogonal_interaction",
            "objective_loss": "quadratic",
            "objective_weighting": "variance",
            "objective_weight_profile": "none",
            "candidate_backend": "cpu_repair",
            "score_backend": "precision_operator",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "atom_flow_pool_multiplier": 8,
            "transport_prefix_strategy": "best_advantage",
            "max_iters": 3,
            "num_active_targets": 12,
            "total_candidates_per_iter": 96,
            "accepted_per_iter": 8,
            "kappa_noise": 0.0,
            "allow_below_noise_fallback": True,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "source_over_sample_factor": 8,
            "full_recompute_every": 1,
            "stop_patience": 3,
            "min_advantage": 0.0,
            "log_every": 1,
            "structured_swap_enabled": False,
            "candidate_diagnostics": False,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 32,
            "answer_batch_size": 128,
            "xla_preallocate": False,
        },
        "debug": {
            "recompute_after_batch": True,
            "assert_batch_loss_decrease": True,
        },
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }

    metrics = run_qdte(config)

    assert metrics["precision_operator"] == "orthogonal_interaction"
    assert metrics["precision_operator_active"] is True
    assert metrics["precision_operator_diagnostics"]["effective_rank"] == 6
    assert metrics["num_accepted_edits"] > 0
    assert metrics["final_optimization_objective"] < metrics["initial_optimization_objective"]
    assert (output_dir / "measurements.json").exists()
    output_measurement = orjson.loads((output_dir / "measurements.json").read_bytes())
    assert output_measurement["strategy_transcript"]["adjacency"] == "add_remove"

    stop_output_dir = tmp_path / "run_confidence_stop"
    stop_config = copy.deepcopy(config)
    stop_config["run"]["output_dir"] = str(stop_output_dir)
    stop_config["qdte"]["max_iters"] = 100
    stop_config["qdte"]["confidence_stop"] = {
        "enabled": True,
        "method": "chi_square",
        "alpha": 0.05,
    }
    stop_metrics = run_qdte(stop_config)

    assert stop_metrics["confidence_stop_enabled"] is True
    assert stop_metrics["confidence_stop_triggered"] is True
    assert stop_metrics["confidence_stop_iteration"] is not None
    assert stop_metrics["confidence_stop_iteration"] < 100
    assert stop_metrics["final_optimization_objective"] <= (
        stop_metrics["confidence_stop_diagnostics"]["objective_threshold"] + 1.0e-9
    )
    stop_runtime = orjson.loads((stop_output_dir / "runtime.json").read_bytes())
    assert stop_runtime["confidence_stop_triggered"] is True
    assert stop_runtime["confidence_stop_iteration"] == stop_metrics["confidence_stop_iteration"]

    shrink_measurements, shrinkage = interaction_transcript_to_shrunk_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )
    assert shrinkage.diagnostics()["no_op"] is True
    shrink_artifact = tmp_path / "measurement_shrink"
    _write_measurement_artifact(
        shrink_artifact,
        qcat=qcat,
        schema=schema,
        measurements=shrink_measurements,
    )
    for precision_name in (
        "orthogonal_interaction_shrink_raw",
        "orthogonal_interaction_shrink_analytic",
    ):
        shrink_output = tmp_path / precision_name
        shrink_config = copy.deepcopy(config)
        shrink_config["run"]["output_dir"] = str(shrink_output)
        shrink_config["measurement"]["reuse_from"] = str(shrink_artifact)
        shrink_config["qdte"]["precision_operator"] = precision_name
        shrink_metrics = run_qdte(shrink_config)
        assert shrink_metrics["precision_operator"] == precision_name
        assert shrink_metrics["final_optimization_objective"] <= (
            shrink_metrics["initial_optimization_objective"] + 1.0e-9
        )
        assert shrink_metrics["precision_operator_diagnostics"]["interaction_shrinkage"][
            "method"
        ] == "positive_part_block_james_stein_v1"

    p3_transcript = copy.deepcopy(transcript)
    p3_transcript.noisy_components["oneway_contrast:0"] += np.asarray([100.0])
    p3_measurements, p3_projection = interaction_transcript_to_local_polytope_measurements(
        p3_transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )
    assert p3_projection.diagnostics["active_cell_count"] > 0
    p3_artifact = tmp_path / "measurement_p3"
    _write_measurement_artifact(
        p3_artifact,
        qcat=qcat,
        schema=schema,
        measurements=p3_measurements,
    )
    p3_boot_measurements, _, _ = (
        interaction_transcript_to_local_polytope_bootdiag_measurements(
            p3_transcript,
            qcat,
            groups,
            delta=1.0e-9,
            rng=np.random.default_rng(741),
            num_samples=16,
            min_variance=1.0e-6,
            min_raw_variance_fraction=0.02,
        )
    )
    p3_boot_artifact = tmp_path / "measurement_p3_bootdiag"
    _write_measurement_artifact(
        p3_boot_artifact,
        qcat=qcat,
        schema=schema,
        measurements=p3_boot_measurements,
    )
    for precision_name, precision_artifact in (
        ("orthogonal_interaction_p3_raw", p3_artifact),
        ("orthogonal_interaction_p3_bootdiag", p3_boot_artifact),
        ("orthogonal_interaction_p3_active_set", p3_artifact),
    ):
        p3_output = tmp_path / precision_name
        p3_config = copy.deepcopy(config)
        p3_config["run"]["output_dir"] = str(p3_output)
        p3_config["measurement"]["reuse_from"] = str(precision_artifact)
        p3_config["qdte"]["precision_operator"] = precision_name
        p3_metrics = run_qdte(p3_config)
        assert p3_metrics["precision_operator"] == precision_name
        assert p3_metrics["final_optimization_objective"] <= (
            p3_metrics["initial_optimization_objective"] + 1.0e-9
        )
        assert p3_metrics["precision_operator_diagnostics"]["local_polytope_p3"][
            "certificate_passed"
        ] is True
        if precision_name.endswith("active_set"):
            assert p3_metrics["precision_operator_diagnostics"][
                "active_set_covariance_certificate"
            ]["certificate_passed"] is True
        if precision_name.endswith("bootdiag"):
            assert p3_metrics["precision_operator_diagnostics"][
                "coefficient_bootstrap_diagonal"
            ]["num_samples"] == 16


def test_active_interaction_shrinkage_engine_smoke(tmp_path: Path) -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=4),
            ColumnSchema(name="b", kind="categorical", cardinality=3),
        ]
    )
    rows = np.asarray(
        [[left, left % 3] for left in range(4) for _ in range(32)],
        dtype=np.int32,
    )
    data_path = tmp_path / "raw.csv"
    pd.DataFrame(rows, columns=["a", "b"]).to_csv(data_path, index=False)
    schema_path = tmp_path / "schema.json"
    schema.save_json(schema_path)
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=16)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, [(0, 1)]),
        public_total=len(rows),
        rho_total=50.0,
        rng=np.random.default_rng(1702),
        allocation_mode="public_optimal",
    )
    measurements, shrinkage = interaction_transcript_to_shrunk_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )
    diagnostics = shrinkage.diagnostics()
    assert diagnostics["num_affected_pair_blocks"] == 1
    assert diagnostics["no_op"] is False
    assert diagnostics["analytic_covariance_stable"] is True
    assert 0.0 < diagnostics["factor_min"] < 1.0
    artifact = tmp_path / "shrink_measurement"
    _write_measurement_artifact(
        artifact,
        qcat=qcat,
        schema=schema,
        measurements=measurements,
    )

    base = {
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 3,
            "num_active_targets": 12,
            "total_candidates_per_iter": 96,
            "accepted_per_iter": 8,
            "kappa_noise": 0.0,
            "allow_below_noise_fallback": True,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "source_over_sample_factor": 8,
            "full_recompute_every": 1,
            "stop_patience": 3,
            "min_advantage": 0.0,
            "log_every": 1,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 32,
            "answer_batch_size": 128,
        },
        "debug": {
            "recompute_after_batch": True,
            "assert_batch_loss_decrease": True,
        },
        "evaluation": {"compute_true_query_error": False},
    }
    for precision_name in (
        "orthogonal_interaction_shrink_raw",
        "orthogonal_interaction_shrink_analytic",
    ):
        output = tmp_path / precision_name
        config = _generation_config(
            base,
            input_csv=data_path,
            public_schema=schema_path,
            artifact_dir=artifact,
            output_dir=output,
            dataset_name="active_shrinkage_smoke",
            epsilon=10.0,
            rho_total=50.0,
            delta=1.0e-9,
            seed=1703,
            max_iters=3,
            precision_operator=precision_name,
            exact_scoring_chunk_size=32,
        )
        metrics = run_qdte(config)
        assert metrics["precision_operator"] == precision_name
        assert metrics["precision_operator_diagnostics"]["interaction_shrinkage"][
            "no_op"
        ] is False
        assert metrics["final_optimization_objective"] <= (
            metrics["initial_optimization_objective"] + 1.0e-9
        )


def test_confidence_constrained_entropy_nonbinary_engine_smoke(tmp_path: Path) -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=4),
            ColumnSchema(name="b", kind="categorical", cardinality=3),
            ColumnSchema(name="c", kind="categorical", cardinality=2),
        ]
    )
    rows = np.asarray(
        [
            [left, (left + repeat) % 3, (left + repeat) % 2]
            for left in range(4)
            for repeat in range(24)
        ],
        dtype=np.int32,
    )
    data_path = tmp_path / "entropy_raw.csv"
    pd.DataFrame(rows, columns=["a", "b", "c"]).to_csv(data_path, index=False)
    schema_path = tmp_path / "entropy_schema.json"
    schema.save_json(schema_path)
    pairs = [(0, 1), (0, 2), (1, 2)]
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=16)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, pairs),
        public_total=len(rows),
        rho_total=0.2,
        rng=np.random.default_rng(1801),
        allocation_mode="public_optimal",
    )
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    artifact = tmp_path / "entropy_measurement"
    _write_measurement_artifact(
        artifact,
        qcat=qcat,
        schema=schema,
        measurements=measurements,
    )
    output = tmp_path / "entropy_run"
    base = {
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 4,
            "num_active_targets": 16,
            "total_candidates_per_iter": 128,
            "accepted_per_iter": 8,
            "kappa_noise": 0.0,
            "allow_below_noise_fallback": True,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "source_over_sample_factor": 8,
            "atom_flow_pool_multiplier": 8,
            "full_recompute_every": 1,
            "stop_patience": 4,
            "min_advantage": 0.0,
            "log_every": 1,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 64,
            "answer_batch_size": 128,
        },
        "debug": {
            "recompute_after_batch": True,
            "assert_batch_loss_decrease": True,
        },
        "evaluation": {"compute_true_query_error": False},
    }
    config = _generation_config(
        base,
        input_csv=data_path,
        public_schema=schema_path,
        artifact_dir=artifact,
        output_dir=output,
        dataset_name="entropy_nonbinary_smoke",
        epsilon=0.1,
        rho_total=0.2,
        delta=1.0e-9,
        seed=1802,
        max_iters=4,
        precision_operator="orthogonal_interaction",
        exact_scoring_chunk_size=64,
        entropy=True,
    )

    metrics = run_qdte(config)

    assert metrics["entropy_enabled"] is True
    assert metrics["precision_operator"] == "orthogonal_interaction"
    assert np.isfinite(metrics["initial_entropy_regularizer"])
    assert np.isfinite(metrics["final_entropy_regularizer"])
    assert metrics["entropy_diagnostics"]["prior"]["domain_size"] == 24
    assert metrics["entropy_diagnostics"]["state"]["num_rows"] == len(rows)
    assert metrics["entropy_diagnostics"]["controller"]["dual_updates"] == 4
    timeseries = pd.read_csv(output / "metrics_timeseries.csv")
    assert "entropy_regularizer" in timeseries.columns
    assert "entropy_dual_weight" in timeseries.columns
    assert timeseries["atom_flow_custom_prefix"].max() == 1
    runtime = orjson.loads((output / "runtime.json").read_bytes())
    assert runtime["entropy_enabled"] is True
    assert runtime["entropy_diagnostics"]["state"]["num_rows"] == len(rows)

    rce_output = tmp_path / "rce_run"
    rce_config = copy.deepcopy(config)
    rce_config["run"]["output_dir"] = str(rce_output)
    rce_config["qdte"]["entropy"]["enabled"] = False
    rce_config["qdte"]["rce"] = {
        "enabled": True,
        "method": "row_realizable_confidence_set_entropic_primal_dual_v1",
        "alpha_l2": 0.025,
        "alpha_linf": 0.025,
        "product_prior_smoothing": 1.0,
    }

    rce_metrics = run_qdte(rce_config)

    assert rce_metrics["rce_enabled"] is True
    assert rce_metrics["entropy_enabled"] is False
    assert rce_metrics["rce_selected_iteration"] <= 4
    assert np.isfinite(rce_metrics["initial_rce_regularizer"])
    assert np.isfinite(rce_metrics["final_rce_regularizer"])
    assert rce_metrics["rce_regularizer_definition"] == "D_KL(p_empirical||p0)"
    assert rce_metrics["initial_rce_regularizer"] == pytest.approx(
        rce_metrics["initial_rce_kl_per_row"]
    )
    assert rce_metrics["rce_effective_lambda_cost"] == pytest.approx(
        rce_config["qdte"]["lambda_cost"] / len(rows)
    )
    assert rce_metrics["rce_diagnostics"]["prior"]["domain_size"] == 24
    assert rce_metrics["rce_diagnostics"]["dual"]["dual_updates"] == 4
    assert rce_metrics["rce_diagnostics"]["integer_incumbent"][
        "globally_certified"
    ] is False
    rce_timeseries = pd.read_csv(rce_output / "metrics_timeseries.csv")
    assert "rce_confidence_slack" in rce_timeseries.columns
    assert "rce_ellipsoid_dual" in rce_timeseries.columns
    assert rce_timeseries["atom_flow_custom_prefix"].max() == 1
    rce_runtime = orjson.loads((rce_output / "runtime.json").read_bytes())
    assert rce_runtime["rce_enabled"] is True
    assert rce_runtime["rce_diagnostics"]["state"]["num_rows"] == len(rows)


def test_exact_interaction_cycle_nonbinary_engine_smoke(tmp_path: Path) -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=4),
            ColumnSchema(name="b", kind="categorical", cardinality=3),
            ColumnSchema(name="c", kind="categorical", cardinality=2),
        ]
    )
    rows = np.asarray(
        [
            [left, (left + repeat) % 3, (left + 2 * repeat) % 2]
            for left in range(4)
            for repeat in range(32)
        ],
        dtype=np.int32,
    )
    data_path = tmp_path / "cycle_raw.csv"
    pd.DataFrame(rows, columns=["a", "b", "c"]).to_csv(data_path, index=False)
    schema_path = tmp_path / "cycle_schema.json"
    schema.save_json(schema_path)
    pairs = [(0, 1), (0, 2), (1, 2)]
    qcat, groups = build_complete_low_order_workload(schema, max_pair_cells=16)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, pairs),
        public_total=len(rows),
        rho_total=0.2,
        rng=np.random.default_rng(1901),
        allocation_mode="public_optimal",
    )
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    artifact = tmp_path / "cycle_measurement"
    _write_measurement_artifact(
        artifact,
        qcat=qcat,
        schema=schema,
        measurements=measurements,
    )
    output = tmp_path / "cycle_run"
    base = {
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 4,
            "num_active_targets": 16,
            "total_candidates_per_iter": 128,
            "accepted_per_iter": 8,
            "kappa_noise": 0.0,
            "allow_below_noise_fallback": True,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "source_over_sample_factor": 8,
            "atom_flow_pool_multiplier": 8,
            "full_recompute_every": 1,
            "stop_patience": 4,
            "min_advantage": 0.0,
            "log_every": 1,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 64,
            "answer_batch_size": 128,
        },
        "debug": {
            "recompute_after_batch": True,
            "assert_batch_loss_decrease": True,
        },
        "evaluation": {"compute_true_query_error": False},
    }
    config = _generation_config(
        base,
        input_csv=data_path,
        public_schema=schema_path,
        artifact_dir=artifact,
        output_dir=output,
        dataset_name="cycle_nonbinary_smoke",
        epsilon=0.1,
        rho_total=0.2,
        delta=1.0e-9,
        seed=1902,
        max_iters=4,
        precision_operator="orthogonal_interaction",
        exact_scoring_chunk_size=64,
        interaction_cycles=True,
    )
    config["qdte"].update(
        {
            "structured_swap_interval": 1,
            "structured_swap_candidate_units": 64,
            "structured_swap_transport_pool": 64,
            "structured_swap_accept_start": 8,
            "structured_swap_accept_end": 1,
            "structured_swap_noise_guard_kappa": 0.0,
            "structured_swap_trigger_rms": 0.0,
        }
    )

    metrics = run_qdte(config)

    assert metrics["structured_swap_compiler"] == "interaction_rectangle_v1"
    assert metrics["structured_swap_compiler_calls"] > 0
    assert metrics["structured_swap_total_candidates"] > 0
    assert metrics["structured_swap_total_positive_rectangles"] > 0
    assert metrics["final_optimization_objective"] <= (
        metrics["initial_optimization_objective"] + 1.0e-9
    )
    assert metrics["final_incremental_answer_drift"] == 0.0
    timeseries = pd.read_csv(output / "metrics_timeseries.csv")
    assert "structured_swap_positive_rectangles" in timeseries.columns
    assert timeseries["structured_swap_candidates"].max() > 0
    runtime = orjson.loads((output / "runtime.json").read_bytes())
    assert runtime["structured_swap_compiler"] == "interaction_rectangle_v1"


def test_measurement_fission_selects_reproducible_checkpoint(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2],
            "b": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )
    data_path = tmp_path / "fission.csv"
    df.to_csv(data_path, index=False)

    def config_for(output_dir: Path) -> dict:
        return {
            "run": {
                "dataset_name": "fission",
                "input_csv": str(data_path),
                "output_dir": str(output_dir),
                "seed": 29,
            },
            "preprocess": {
                "categorical_columns": ["a", "b"],
                "numerical_columns": [],
            },
            "workload": {
                "include_oneway": True,
                "include_2way_cat": True,
                "include_prefix": False,
                "include_range": False,
                "include_mixed": False,
                "max_queries": 32,
                "max_terms": 2,
                "max_2way_cells": 32,
            },
            "privacy": {
                "mode": "dp",
                "rho_total": 1.0,
                "delta": 1.0e-9,
                "measurement_allocation": {"oneway": 0.5, "twoway": 0.5},
            },
            "measurement": {
                "fission": {
                    "enabled": True,
                    "train_fraction": 0.8,
                    "checkpoint_interval": 2,
                    "selection_rule": "earliest_within_one_se",
                    "one_se_multiplier": 1.0,
                    "seed_offset": 51_771,
                }
            },
            "projection": {"project_partitions": True, "clip_nonpartition": True},
            "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
            "qdte": {
                "max_iters": 4,
                "num_active_targets": 4,
                "total_candidates_per_iter": 32,
                "accepted_per_iter": 4,
                "kappa_noise": 0.0,
                "lambda_cost": 0.0,
                "random_candidate_fraction": 0.1,
                "full_recompute_every": 2,
                "stop_patience": 4,
                "min_advantage": 1.0e-6,
                "log_every": 4,
                "objective_weighting": "variance",
                "objective_loss": "quadratic",
            },
            "runtime": {
                "use_pmap": False,
                "scoring_chunk_size": 32,
                "answer_batch_size": 32,
                "xla_preallocate": False,
            },
            "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
        }

    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    reverse_out = tmp_path / "reverse"
    first_metrics = run_qdte(config_for(first_out))
    second_metrics = run_qdte(config_for(second_out))
    reverse_config = config_for(reverse_out)
    reverse_config["measurement"]["fission"]["optimization_branch"] = "validation"
    reverse_config["qdte"]["objective_weight_profile"] = "joint_utility_envelope"
    reverse_metrics = run_qdte(reverse_config)
    fission_artifact = orjson.loads((first_out / "measurements_fission.json").read_bytes())
    reverse_artifact = orjson.loads((reverse_out / "measurements_fission.json").read_bytes())
    checkpoints = orjson.loads((first_out / "fission_checkpoints.json").read_bytes())

    assert first_metrics["measurement_fission_enabled"] is True
    assert reverse_metrics["measurement_fission_optimization_branch"] == "validation"
    assert reverse_metrics["measurement_fission_holdout_branch"] == "train"
    assert reverse_metrics["objective_weight_profile"] == "joint_utility_envelope"
    assert reverse_metrics["objective_query_weight_multiplier_max"] > 1.0
    assert reverse_artifact["optimization_branch"] == "validation"
    assert reverse_artifact["holdout_branch"] == "train"
    assert np.allclose(
        np.asarray(reverse_artifact["optimization_noisy"]),
        np.asarray(reverse_artifact["validation_noisy"]),
    )
    assert first_metrics["measurement_fission_reconstruction_max_abs"] < 1.0e-4
    assert checkpoints["selected_iteration"] in {0, 2, 4}
    assert sum(int(row["is_selected"]) for row in checkpoints["checkpoints"]) == 1
    assert (first_out / "synthetic_train_terminal_encoded.npy").exists()
    assert np.array_equal(
        np.load(first_out / "synthetic_encoded.npy"),
        np.load(second_out / "synthetic_encoded.npy"),
    )
    assert first_metrics["measurement_fission_selected_iteration"] == second_metrics[
        "measurement_fission_selected_iteration"
    ]
    reconstructed = (
        0.8 * np.asarray(fission_artifact["train_noisy"])
        + 0.2 * np.asarray(fission_artifact["validation_noisy"])
    )
    original = orjson.loads((first_out / "measurements.json").read_bytes())
    assert np.allclose(reconstructed, np.asarray(original["target_noisy"]), atol=1.0e-4)


def test_engine_smoke_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "a": rng.integers(0, 3, size=n),
            "b": rng.integers(0, 4, size=n),
            "c": rng.integers(0, 6, size=n),
            "label": rng.integers(0, 2, size=n),
        }
    )
    data_path = tmp_path / "smoke.csv"
    out_dir = tmp_path / "out"
    df.to_csv(data_path, index=False)
    config = {
        "run": {"dataset_name": "smoke", "input_csv": str(data_path), "output_dir": str(out_dir), "seed": 0},
        "preprocess": {
            "numerical_bins": 6,
            "label_column": "label",
            "numerical_columns": ["c"],
            "categorical_columns": ["a", "b", "label"],
        },
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": True,
            "include_range": True,
            "include_mixed": True,
            "max_queries": 120,
            "max_terms": 4,
            "range_intervals_per_num_attr": 4,
            "mixed_queries_per_pair": 4,
            "max_2way_cells": 50,
            "random_seed": 0,
        },
        "privacy": {
            "mode": "dp",
            "rho_total": 1.0,
            "delta": 1e-9,
            "measurement_allocation": {"oneway": 0.3, "twoway": 0.3, "prefix": 0.15, "range": 0.1, "mixed": 0.15},
        },
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real"},
        "qdte": {
            "max_iters": 3,
            "num_active_targets": 4,
            "candidates_per_target": 4,
            "total_candidates_per_iter": 32,
            "accepted_per_iter": 2,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "candidate_diagnostics": True,
            "full_recompute_every": 2,
            "stop_patience": 3,
            "min_advantage": 1e-6,
            "log_every": 1,
            "structured_swap_enabled": True,
            "structured_swap_start_iter": 1,
            "structured_swap_interval": 1,
            "structured_swap_candidate_units": 16,
            "structured_swap_transport_pool": 8,
            "structured_swap_accept_start": 2,
            "structured_swap_accept_end": 1,
            "structured_swap_accept_schedule": "cosine",
            "structured_swap_noise_guard_kappa": 0.0,
            "structured_swap_trigger_rms": 0.0,
            "structured_swap_exploration_floor": 0.2,
            "structured_swap_policy_decay": 0.95,
            "structured_swap_policy_prior_strength": 4.0,
        },
        "debug": {"recompute_after_batch": True, "assert_batch_loss_decrease": True},
        "runtime": {"use_pmap": False, "scoring_chunk_size": 32, "answer_batch_size": 64, "xla_preallocate": False},
        "evaluation": {
            "compute_true_query_error": True,
            "compute_heldout_query_error": True,
            "heldout_exclude_measured_queries": True,
            "save_synthetic_csv": True,
            "oracle_projection_bias": {"enabled": True, "num_samples": 4, "seed": 123},
            "heldout_workload": {
                "include_oneway": True,
                "include_2way_cat": True,
                "include_prefix": True,
                "include_range": True,
                "include_mixed": True,
                "include_halfspace": False,
                "max_queries": 180,
                "max_terms": 4,
                "max_2way_cells": 80,
                "range_intervals_per_num_attr": 8,
                "mixed_queries_per_pair": 8,
                "random_seed": 10000,
            },
        },
    }
    metrics = run_qdte(config)
    assert metrics["num_queries"] > 0
    assert (out_dir / "synthetic_decoded.csv").exists()
    assert (out_dir / "metrics_final.json").exists()
    assert (out_dir / "synthetic_initial_encoded.npy").exists()
    assert orjson.loads((out_dir / "run_status.json").read_bytes())["status"] == "completed"
    assert (out_dir / "metrics_by_family.json").exists()
    assert (out_dir / "oracle_projection_bias.json").exists()
    assert (out_dir / "workload_summary.json").exists()
    assert (out_dir / "queries_holdout.json").exists()
    assert (out_dir / "workload_summary_holdout.json").exists()
    assert (out_dir / "metrics_holdout.json").exists()
    assert (out_dir / "metrics_by_family_holdout.json").exists()
    assert (out_dir / "metrics_timeseries.csv").exists()
    assert (out_dir / "candidate_diagnostics_timeseries.csv").exists()
    assert (out_dir / "runtime.json").exists()
    metrics_json = orjson.loads((out_dir / "metrics_final.json").read_bytes())
    assert "final_rms_standardized_residual" in metrics_json
    assert "final_unweighted_measured_loss" in metrics_json
    assert "final_rms_unweighted_residual" in metrics_json
    assert "initial_true_query_mae" in metrics_json
    assert "final_true_query_mae" in metrics_json
    assert "positive_returned_rate_is_topk_biased" in metrics_json
    assert metrics_json["candidate_diagnostics_enabled"] is True
    assert "heldout_final_true_query_mae" in metrics_json
    assert "oracle_projection_bias_oracle_bias_l2" in metrics_json
    assert "oracle_projection_bias_observed_projected_error_l2" in metrics_json
    assert metrics_json["true_query_mae"] == metrics_json["final_true_query_mae"]
    assert metrics["final_incremental_answer_drift"] == 0.0
    assert metrics_json["structured_swap_enabled"] is True
    assert metrics_json["structured_swap_total_candidates"] > 0
    assert metrics_json["structured_swap_total_accepted"] >= 0
    by_family_json = orjson.loads((out_dir / "metrics_by_family.json").read_bytes())
    assert "oneway" in by_family_json
    assert by_family_json["oneway"]["num_queries"] > 0
    assert "true_query_mae_reduction" in by_family_json["oneway"]
    assert "true_query_rmse_reduction" in by_family_json["oneway"]
    workload_json = orjson.loads((out_dir / "workload_summary.json").read_bytes())
    assert workload_json["total_num_queries"] == metrics["num_queries"]
    assert "num_queries_by_family" in workload_json
    assert workload_json["total_queries"] == workload_json["total_num_queries"]
    assert workload_json["queries_by_family"] == workload_json["num_queries_by_family"]
    holdout_workload_json = orjson.loads((out_dir / "workload_summary_holdout.json").read_bytes())
    assert holdout_workload_json["total_queries"] > 0
    assert holdout_workload_json["num_removed_as_measured_duplicates"] > 0
    assert holdout_workload_json["heldout_exclude_measured_queries"] is True
    holdout_metrics_json = orjson.loads((out_dir / "metrics_holdout.json").read_bytes())
    assert holdout_metrics_json["num_queries"] == holdout_workload_json["total_queries"]
    assert holdout_metrics_json["num_queries"] > 0
    assert "final_true_query_mae" in holdout_metrics_json
    by_family_holdout_json = orjson.loads((out_dir / "metrics_by_family_holdout.json").read_bytes())
    assert by_family_holdout_json
    measured_qcat = QueryCatalogue.from_dict(orjson.loads((out_dir / "queries.json").read_bytes()))
    heldout_qcat = QueryCatalogue.from_dict(orjson.loads((out_dir / "queries_holdout.json").read_bytes()))
    measured_keys = {query_key(measured_qcat, qid) for qid in range(measured_qcat.m)}
    heldout_keys = {query_key(heldout_qcat, qid) for qid in range(heldout_qcat.m)}
    assert measured_keys.isdisjoint(heldout_keys)
    measurement_json = orjson.loads((out_dir / "measurements.json").read_bytes())
    assert "true_answers_debug" not in measurement_json
    assert "true_answers" not in measurement_json
    assert "oracle_projection_bias" not in measurement_json
    assert measurement_json["rho_spent"] <= measurement_json["rho_total"] + 1.0e-12
    runtime_json = orjson.loads((out_dir / "runtime.json").read_bytes())
    assert runtime_json["num_candidates_requested"] >= runtime_json["num_candidates_scored"]
    assert runtime_json["candidate_diagnostics_enabled"] is True
    assert "accepted_per_scored_candidate" in runtime_json
    assert "candidate_funnel" in runtime_json
    assert runtime_json["candidate_funnel"]["requested"] == runtime_json["num_candidates_requested"]
    assert "mean_debt" in runtime_json
    candidate_diagnostics = pd.read_csv(out_dir / "candidate_diagnostics_timeseries.csv")
    assert "diag_mean_target_component" in candidate_diagnostics.columns
    assert "diag_mean_collateral_component" in candidate_diagnostics.columns
    assert "diag_target_positive_full_negative_rate" in candidate_diagnostics.columns
    metrics_timeseries = pd.read_csv(out_dir / "metrics_timeseries.csv")
    assert "unweighted_measured_loss" in metrics_timeseries.columns
    assert "rms_unweighted_residual" in metrics_timeseries.columns


def test_engine_no_active_queries_skip_candidate_generation(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [0, 1, 0, 1], "label": [0, 1, 0, 1]})
    data_path = tmp_path / "tiny.csv"
    out_dir = tmp_path / "out"
    df.to_csv(data_path, index=False)
    config = {
        "run": {"dataset_name": "tiny", "input_csv": str(data_path), "output_dir": str(out_dir), "seed": 0},
        "preprocess": {"label_column": "label", "categorical_columns": ["a", "label"], "numerical_columns": []},
        "workload": {
            "include_oneway": True,
            "include_2way_cat": False,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 10,
            "max_terms": 2,
        },
        "privacy": {"mode": "dp", "rho_total": 1.0, "delta": 1e-9, "measurement_allocation": {"oneway": 1.0}},
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 2,
            "num_active_targets": 2,
            "total_candidates_per_iter": 8,
            "accepted_per_iter": 1,
            "kappa_noise": 1.0e9,
            "allow_below_noise_fallback": False,
            "stop_patience": 1,
            "log_every": 1,
            "objective_weighting": "unweighted",
        },
        "runtime": {"use_pmap": False, "scoring_chunk_size": 8, "answer_batch_size": 8, "xla_preallocate": False},
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }

    metrics = run_qdte(config)

    runtime_json = orjson.loads((out_dir / "runtime.json").read_bytes())
    metrics_json = orjson.loads((out_dir / "metrics_final.json").read_bytes())
    logs = (out_dir / "logs.txt").read_text(encoding="utf-8")
    assert metrics["objective_weighting"] == "unweighted"
    assert metrics_json["objective_inv_variance_mean"] == 1.0
    assert runtime_json["objective_weighting"] == "unweighted"
    assert np.isclose(metrics["final_measured_loss"], metrics["final_unweighted_measured_loss"])
    assert runtime_json["num_candidates_requested"] == 0
    assert runtime_json["num_candidates_scored"] == 0
    assert "No active queries above noise threshold" in logs


def test_engine_can_reuse_measurement_artifact(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [0, 1, 0, 1, 1, 0], "label": [0, 1, 0, 1, 0, 1]})
    data_path = tmp_path / "reuse.csv"
    measurement_dir = tmp_path / "measurement"
    reuse_dir = tmp_path / "reuse_out"
    df.to_csv(data_path, index=False)
    base_config = {
        "run": {"dataset_name": "reuse", "input_csv": str(data_path), "output_dir": str(measurement_dir), "seed": 7},
        "preprocess": {"label_column": "label", "categorical_columns": ["a", "label"], "numerical_columns": []},
        "workload": {
            "include_oneway": True,
            "include_2way_cat": False,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 10,
            "max_terms": 2,
        },
        "privacy": {"mode": "dp", "rho_total": 1.0, "delta": 1e-9, "measurement_allocation": {"oneway": 1.0}},
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 0,
            "num_active_targets": 2,
            "total_candidates_per_iter": 8,
            "accepted_per_iter": 1,
            "objective_weighting": "unweighted",
        },
        "runtime": {"use_pmap": False, "scoring_chunk_size": 8, "answer_batch_size": 8, "xla_preallocate": False},
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }

    first_metrics = run_qdte(base_config)
    reuse_config = dict(base_config)
    reuse_config["run"] = dict(base_config["run"], output_dir=str(reuse_dir))
    reuse_config["measurement"] = {"reuse_from": str(measurement_dir)}
    reuse_metrics = run_qdte(reuse_config)

    assert first_metrics["measurement_reused"] is False
    assert reuse_metrics["measurement_reused"] is True
    assert reuse_metrics["measurement_reuse_from"] == str(measurement_dir)
    assert np.isclose(first_metrics["final_measured_loss"], reuse_metrics["final_measured_loss"])
    first_measurements = orjson.loads((measurement_dir / "measurements.json").read_bytes())
    reused_measurements = orjson.loads((reuse_dir / "measurements.json").read_bytes())
    assert first_measurements["target_projected"] == reused_measurements["target_projected"]
    assert first_measurements["num_rows"] == len(df)
    assert np.array_equal(
        np.load(measurement_dir / "synthetic_encoded.npy"),
        np.load(reuse_dir / "synthetic_encoded.npy"),
    )
    logs = (reuse_dir / "logs.txt").read_text(encoding="utf-8")
    assert "Reused measurement artifact" in logs


def test_dp_mode_rejects_reused_oracle_measurement_artifact(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [0, 1, 0, 1]})
    data_path = tmp_path / "oracle_reuse.csv"
    oracle_dir = tmp_path / "oracle_measurement"
    rejected_dir = tmp_path / "rejected_dp_reuse"
    df.to_csv(data_path, index=False)
    base_config = {
        "run": {"dataset_name": "oracle_reuse", "input_csv": str(data_path), "output_dir": str(oracle_dir), "seed": 3},
        "preprocess": {"categorical_columns": ["a"], "numerical_columns": []},
        "workload": {
            "include_oneway": True,
            "include_2way_cat": False,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 10,
            "max_terms": 1,
        },
        "privacy": {"mode": "oracle", "oracle_variance": 1.0},
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "max_iters": 0,
            "num_active_targets": 2,
            "total_candidates_per_iter": 4,
            "accepted_per_iter": 1,
        },
        "runtime": {"use_pmap": False, "scoring_chunk_size": 4, "answer_batch_size": 4},
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }
    run_qdte(base_config)

    rejected_config = dict(base_config)
    rejected_config["run"] = dict(base_config["run"], output_dir=str(rejected_dir))
    rejected_config["privacy"] = {"mode": "dp", "rho_total": 1.0, "delta": 1.0e-9}
    rejected_config["measurement"] = {"reuse_from": str(oracle_dir)}

    with pytest.raises(ValueError, match="privacy.mode=dp cannot reuse"):
        run_qdte(rejected_config)


def test_engine_atom_flow_transport_smoke(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2, 0, 1],
            "b": [0, 1, 0, 1, 0, 1, 1, 0],
            "label": [0, 1, 0, 1, 0, 1, 1, 0],
        }
    )
    data_path = tmp_path / "atom.csv"
    out_dir = tmp_path / "atom_out"
    df.to_csv(data_path, index=False)
    config = {
        "run": {"dataset_name": "atom", "input_csv": str(data_path), "output_dir": str(out_dir), "seed": 1},
        "preprocess": {"label_column": "label", "categorical_columns": ["a", "b", "label"], "numerical_columns": []},
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 40,
            "max_terms": 2,
            "max_2way_cells": 40,
        },
        "privacy": {
            "mode": "dp",
            "rho_total": 1.0,
            "delta": 1e-9,
            "measurement_allocation": {"oneway": 0.5, "twoway": 0.5},
        },
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "atom_flow_pool_multiplier": 0,
            "atom_flow_max_pool": 0,
            "transport_prefix_strategy": "best_advantage",
            "max_iters": 2,
            "num_active_targets": 4,
            "total_candidates_per_iter": 24,
            "accepted_per_iter": 4,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.0,
            "full_recompute_every": 1,
            "stop_patience": 2,
            "min_advantage": 1e-6,
            "log_every": 1,
        },
        "debug": {"recompute_after_batch": True, "assert_batch_loss_decrease": True},
        "runtime": {"use_pmap": False, "scoring_chunk_size": 24, "answer_batch_size": 16, "xla_preallocate": False},
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }

    metrics = run_qdte(config)

    runtime_json = orjson.loads((out_dir / "runtime.json").read_bytes())
    timeseries = pd.read_csv(out_dir / "metrics_timeseries.csv")
    assert metrics["transport_mode"] == "atom_flow"
    assert metrics["atom_flow_update_mode"] == "batch"
    assert runtime_json["transport_mode"] == "atom_flow"
    assert runtime_json["atom_flow_update_mode"] == "batch"
    assert "atom_flow_edges" in timeseries.columns
    assert "atom_flow_batch_mode" in timeseries.columns
    assert timeseries["atom_flow_pool_candidates"].max() > 0
    assert timeseries["atom_flow_batch_mode"].max() == 1
    assert runtime_json["last_atom_flow_edges"] > 0
    assert runtime_json["last_atom_flow_batch_mode"] == 1


def test_engine_tvd_l1_objective_reports_and_decreases_its_actual_objective(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "a": [0, 0, 0, 1, 1, 2, 2, 2],
            "b": [0, 0, 1, 0, 1, 0, 1, 1],
        }
    )
    data_path = tmp_path / "tvd.csv"
    out_dir = tmp_path / "tvd_out"
    df.to_csv(data_path, index=False)
    config = {
        "run": {"dataset_name": "tvd", "input_csv": str(data_path), "output_dir": str(out_dir), "seed": 4},
        "preprocess": {"categorical_columns": ["a", "b"], "numerical_columns": []},
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 40,
            "max_terms": 2,
            "max_2way_cells": 40,
        },
        "privacy": {
            "mode": "dp",
            "rho_total": 1.0,
            "delta": 1.0e-9,
            "measurement_allocation": {"oneway": 0.5, "twoway": 0.5},
        },
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "objective_loss": "tvd_l1",
            "structured_swap_enabled": True,
            "structured_swap_start_iter": 1,
            "structured_swap_interval": 1,
            "structured_swap_candidate_units": 32,
            "structured_swap_transport_pool": 32,
            "structured_swap_accept_start": 4,
            "structured_swap_accept_end": 1,
            "structured_swap_accept_schedule": "cosine",
            "structured_swap_noise_guard_kappa": 0.0,
            "structured_swap_trigger_rms": 0.0,
            "structured_swap_delta_backend": "dense_gpu",
            "candidate_backend": "cpu_repair",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "max_iters": 3,
            "num_active_targets": 4,
            "candidates_per_target": 4,
            "total_candidates_per_iter": 24,
            "accepted_per_iter": 4,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.0,
            "full_recompute_every": 1,
            "stop_patience": 3,
            "min_advantage": 1.0e-6,
            "log_every": 1,
        },
        "debug": {"recompute_after_batch": True, "assert_batch_loss_decrease": True},
        "runtime": {"use_pmap": False, "scoring_chunk_size": 24, "answer_batch_size": 16},
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }

    metrics = run_qdte(config)

    assert metrics["objective_loss"] == "tvd_l1"
    assert metrics["final_optimization_objective"] <= metrics["initial_optimization_objective"] + 1.0e-6
    assert np.isclose(
        metrics["optimization_objective_reduction"],
        metrics["initial_optimization_objective"] - metrics["final_optimization_objective"],
    )
    timeseries = pd.read_csv(out_dir / "metrics_timeseries.csv")
    assert "optimization_objective" in timeseries.columns
    assert timeseries["structured_swap_candidates"].max() > 0


def test_engine_halfspace_cpu_smoke_with_consistency_projection(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5, 0, 2, 4, 5],
            "y": [5, 4, 3, 2, 1, 0, 3, 3, 1, 2],
            "label": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    data_path = tmp_path / "halfspace.csv"
    out_dir = tmp_path / "halfspace_out"
    df.to_csv(data_path, index=False)
    config = {
        "run": {"dataset_name": "halfspace", "input_csv": str(data_path), "output_dir": str(out_dir), "seed": 2},
        "preprocess": {
            "numerical_bins": 6,
            "label_column": "label",
            "categorical_columns": ["label"],
            "numerical_columns": ["x", "y"],
        },
        "workload": {
            "include_oneway": True,
            "include_2way_cat": False,
            "include_prefix": True,
            "include_range": True,
            "include_mixed": False,
            "include_halfspace": True,
            "halfspace_queries": 5,
            "max_queries": 50,
            "max_terms": 2,
            "range_intervals_per_num_attr": 3,
            "random_seed": 2,
        },
        "privacy": {
            "mode": "dp",
            "rho_total": 1.0,
            "delta": 1e-9,
            "measurement_allocation": {"oneway": 0.25, "prefix": 0.25, "range": 0.25, "halfspace": 0.25},
        },
        "projection": {
            "project_partitions": True,
            "clip_nonpartition": True,
            "prefix_monotonicity": True,
            "consistency": {"enabled": True, "max_iterations": 30, "tolerance": 1.0e-2},
        },
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "candidate_backend": "cpu_repair",
            "transport_mode": "microbatch_greedy",
            "max_iters": 2,
            "num_active_targets": 4,
            "total_candidates_per_iter": 32,
            "accepted_per_iter": 2,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "full_recompute_every": 1,
            "stop_patience": 2,
            "min_advantage": 1e-6,
            "log_every": 1,
        },
        "debug": {"recompute_after_batch": True, "assert_batch_loss_decrease": True},
        "runtime": {"use_pmap": False, "scoring_chunk_size": 32, "answer_batch_size": 16, "xla_preallocate": False},
        "evaluation": {"compute_true_query_error": False, "save_synthetic_csv": False},
    }

    metrics = run_qdte(config)

    measurements = orjson.loads((out_dir / "measurements.json").read_bytes())
    qcat = QueryCatalogue.from_dict(orjson.loads((out_dir / "queries.json").read_bytes()))
    assert "halfspace" in qcat.families
    assert metrics["final_incremental_answer_drift"] == 0.0
    consistency = measurements["projection_diagnostics"]["consistency"]
    assert consistency["enabled"] is True
    assert consistency["known_total_count"] == len(df)
