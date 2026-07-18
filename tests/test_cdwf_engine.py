from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

import qdte.evolution.engine as engine_module
from qdte.dataio import write_json
from qdte.evolution.engine import run_qdte
from qdte.measurement.cdwf import derive_cdwf_budget_plan
from qdte.measurement.cdwf_protocol import run_cdwf_adaptive_measurement
from qdte.measurement.cdwf_transcript import cdwf_transcript_to_measurements
from qdte.measurement.factorization import compile_hierarchical_pair_strategy
from qdte.measurement.public_artifact import (
    verify_public_transcript,
    write_public_transcript_manifest,
)
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.rce.streamwise_controller import STREAMWISE_RCE_CONTROLLER_METHOD
from qdte.schema import ColumnSchema, TableSchema
from scripts.generate_qdte_from_transcript import _prepare_generation_config


def test_cdwf_sealed_transcript_runs_streamwise_rce_without_private_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name=name,
                kind="categorical",
                cardinality=2,
            )
            for name in ("a", "b")
        ]
    )
    rows = np.asarray(list(itertools.product(range(2), repeat=2)) * 40, dtype=np.int32)
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, ((0, 1),))
    control_rho = {block.name: 0.2 for block in strategy.blocks}
    plan = derive_cdwf_budget_plan(strategy, control_rho, epsilon=0.1)
    qcat, groups = build_selected_pair_partition_workload(schema, strategy.pairs)
    common_measurement = dict(
        rows=rows,
        schema=schema,
        qcat=qcat,
        workload_groups=groups,
        strategy=strategy,
        plan=plan,
        public_total=len(rows),
        base_noise_seed=101,
        refinement_noise_seed=202,
        generation_seed=303,
        shadow_max_iterations=2_000,
    )
    measurement_run = run_cdwf_adaptive_measurement(
        **common_measurement,
        arm="dual_water_fill",
    )
    assert measurement_run.dual_fallback_count == 0, [
        record.fallback_reasons for record in measurement_run.rounds
    ]
    measurements = cdwf_transcript_to_measurements(
        measurement_run.transcript,
        qcat,
        groups,
        delta=1.0e-9,
    )

    artifact = tmp_path / "sealed_cdwf"
    artifact.mkdir()
    schema.save_json(artifact / "schema.json")
    qcat.save_json(artifact / "queries.json")
    write_json(measurements.to_public_dict(), artifact / "measurements.json")
    write_public_transcript_manifest(artifact)
    verified = verify_public_transcript(artifact)
    assert verified.measurements.sequential_transcript is not None

    base_config = {
        "run": {
            "dataset_name": "cdwf_engine_smoke",
            "input_csv": str(tmp_path / "must_not_be_loaded.csv"),
            "output_dir": str(tmp_path / "unused"),
            "seed": 303,
        },
        "preprocess": {"public_schema_json": str(artifact / "schema.json")},
        "workload": {"reuse_from_measurement": True},
        "privacy": {
            "mode": "dp",
            "dp_release_mode": True,
            "public_row_count": True,
            "public_n_rows": len(rows),
            "adjacency": "add_remove",
            "rho_total": plan.rho_total,
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
            "max_iters": 2,
            "num_active_targets": 4,
            "total_candidates_per_iter": 48,
            "accepted_per_iter": 4,
            "kappa_noise": 0.0,
            "allow_below_noise_fallback": True,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "source_over_sample_factor": 8,
            "full_recompute_every": 1,
            "stop_patience": 2,
            "min_advantage": 0.0,
            "log_every": 1,
            "structured_swap_enabled": False,
            "candidate_diagnostics": False,
            "rce": {
                "enabled": True,
                "method": "row_realizable_confidence_set_entropic_primal_dual_v1",
                "reference_prior": "released_confidence_forest_v1",
                "prefix_backend": "feature_cpu_exact",
                "alpha_l2": 0.025,
                "alpha_linf": 0.025,
                "product_prior_smoothing": 1.0,
                "streamwise_cdwf": {"enabled": True},
            },
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 32,
            "answer_batch_size": 128,
            "xla_preallocate": False,
            "log_measurement_groups": False,
        },
        "debug": {
            "recompute_after_batch": True,
            "assert_batch_loss_decrease": True,
        },
        "evaluation": {
            "compute_true_query_error": False,
            "compute_heldout_query_error": False,
            "downstream_ml": False,
            "save_synthetic_csv": False,
        },
    }
    output_dir = tmp_path / "generation"
    config = _prepare_generation_config(base_config, artifact, output_dir)

    def _private_loader_is_forbidden(_config):
        raise AssertionError("C3 transcript-only generation loaded private rows")

    monkeypatch.setattr(engine_module, "load_and_preprocess_csv", _private_loader_is_forbidden)
    metrics = run_qdte(config)

    assert metrics["private_input_loaded_by_process"] is False
    assert metrics["precision_operator"] == "orthogonal_interaction"
    assert metrics["rce_diagnostics"]["method"] == STREAMWISE_RCE_CONTROLLER_METHOD
    assert metrics["rce_diagnostics"]["confidence_set"]["method"] == (
        "adaptive_safe_streamwise_rce_confidence_v1"
    )
    assert metrics["rce_diagnostics"]["prior"]["method"] == (
        "released_sequential_confidence_calibrated_forest_v1"
    )
    assert metrics["final_incremental_answer_drift"] <= 1.0e-6
    assert (output_dir / "synthetic_encoded.npy").is_file()
