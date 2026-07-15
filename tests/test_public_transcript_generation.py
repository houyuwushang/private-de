from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pytest

import qdte.evolution.engine as engine_module
from qdte.evolution.engine import run_qdte
from qdte.dataio import write_json
from qdte.measurement.factorization import (
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
)
from qdte.measurement.public_artifact import (
    materialize_public_transcript,
    verify_public_transcript,
    write_public_transcript_manifest,
)
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import ColumnSchema, TableSchema
from scripts.generate_qdte_from_transcript import _prepare_generation_config


def _release_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    rows = np.asarray(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    private_csv = tmp_path / "private.csv"
    pd.DataFrame(rows, columns=["a", "b", "c"]).to_csv(private_csv, index=False)
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
    schema_path = tmp_path / "public_schema.json"
    schema.save_json(schema_path)
    legacy_output = tmp_path / "legacy"
    config = {
        "run": {
            "dataset_name": "public_transcript_test",
            "input_csv": str(private_csv),
            "output_dir": str(legacy_output),
            "seed": 37,
        },
        "preprocess": {"public_schema_json": str(schema_path)},
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "max_queries": 64,
            "max_terms": 2,
            "max_2way_cells": 32,
            "random_seed": 5,
        },
        "privacy": {
            "mode": "dp",
            "dp_release_mode": True,
            "public_row_count": True,
            "public_n_rows": len(rows),
            "adjacency": "add_remove",
            "rho_total": 0.2,
            "delta": 1.0e-9,
            "measurement_mode": "static_all",
            "measurement_allocation": {"oneway": 0.5, "twoway": 0.5},
        },
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "candidate_backend": "cpu_repair",
            "score_backend": "dense_gpu",
            "transport_mode": "microbatch_greedy",
            "max_iters": 4,
            "num_active_targets": 6,
            "total_candidates_per_iter": 48,
            "accepted_per_iter": 4,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "full_recompute_every": 2,
            "stop_patience": 4,
            "min_advantage": 0.0,
            "log_every": 2,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 48,
            "answer_batch_size": 64,
            "xla_preallocate": False,
            "log_measurement_groups": False,
        },
        "evaluation": {
            "compute_true_query_error": False,
            "compute_heldout_query_error": False,
            "downstream_ml": False,
            "save_synthetic_csv": False,
        },
    }
    return config, private_csv, legacy_output


def test_split_process_generation_is_byte_identical_and_loads_no_private_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, private_csv, legacy_output = _release_fixture(tmp_path)
    legacy_metrics = run_qdte(config)
    transcript = tmp_path / "transcript"
    materialize_public_transcript(config, transcript)
    assert {path.name for path in transcript.iterdir()} == {
        "schema.json",
        "queries.json",
        "measurements.json",
        "transcript_manifest.json",
    }

    for name in ("schema.json", "queries.json", "measurements.json"):
        assert (transcript / name).read_bytes() == (legacy_output / name).read_bytes()

    private_csv.unlink()

    def _private_loader_is_forbidden(_config):
        raise AssertionError("Transcript-only generation attempted to load private rows")

    monkeypatch.setattr(engine_module, "load_and_preprocess_csv", _private_loader_is_forbidden)
    split_output = tmp_path / "split_generation"
    generation_config = _prepare_generation_config(
        config,
        transcript,
        split_output,
    )
    split_metrics = run_qdte(generation_config)

    assert np.array_equal(
        np.load(legacy_output / "synthetic_initial_encoded.npy"),
        np.load(split_output / "synthetic_initial_encoded.npy"),
    )
    assert np.array_equal(
        np.load(legacy_output / "synthetic_encoded.npy"),
        np.load(split_output / "synthetic_encoded.npy"),
    )
    assert split_metrics["final_measured_loss"] == legacy_metrics["final_measured_loss"]
    assert split_metrics["num_candidates_scored"] == legacy_metrics["num_candidates_scored"]
    assert split_metrics["transcript_only_generation"] is True
    assert split_metrics["private_input_loaded_by_process"] is False


def test_public_transcript_rejects_tampered_artifact(tmp_path: Path) -> None:
    config, _, _ = _release_fixture(tmp_path)
    transcript = tmp_path / "transcript"
    materialize_public_transcript(config, transcript)
    query_path = transcript / "queries.json"
    query_path.write_bytes(query_path.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="changed after sealing"):
        verify_public_transcript(transcript)


def test_public_transcript_requires_sealed_manifest(tmp_path: Path) -> None:
    config, _, _ = _release_fixture(tmp_path)
    transcript = tmp_path / "transcript"
    materialize_public_transcript(config, transcript)
    (transcript / "transcript_manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="transcript_manifest.json"):
        verify_public_transcript(transcript)


def test_public_transcript_accepts_sealed_legacy_ledger_override(
    tmp_path: Path,
) -> None:
    config, _, _ = _release_fixture(tmp_path)
    transcript = tmp_path / "transcript"
    materialize_public_transcript(config, transcript)
    measurement_path = transcript / "measurements.json"
    payload = orjson.loads(measurement_path.read_bytes())
    ledger = payload.pop("privacy_ledger")
    measurement_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    manifest = write_public_transcript_manifest(
        transcript,
        privacy_ledger_override=ledger,
    )
    verified = verify_public_transcript(transcript)

    assert manifest["privacy_ledger_override"] == ledger
    assert verified.measurements.privacy_ledger is None
    assert verified.measurements.rho_spent == ledger["rho_spent"]


def test_public_transcript_rejects_inconsistent_legacy_ledger_override(
    tmp_path: Path,
) -> None:
    config, _, _ = _release_fixture(tmp_path)
    transcript = tmp_path / "transcript"
    materialize_public_transcript(config, transcript)
    measurement_path = transcript / "measurements.json"
    payload = orjson.loads(measurement_path.read_bytes())
    ledger = payload.pop("privacy_ledger")
    measurement_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    ledger["rho_spent"] *= 0.5

    with pytest.raises(ValueError, match="ledger spend"):
        write_public_transcript_manifest(
            transcript,
            privacy_ledger_override=ledger,
        )


def test_public_transcript_rejects_invalid_privacy_ledger(tmp_path: Path) -> None:
    config, _, _ = _release_fixture(tmp_path)
    transcript = tmp_path / "transcript"
    materialize_public_transcript(config, transcript)
    measurement_path = transcript / "measurements.json"
    payload = orjson.loads(measurement_path.read_bytes())
    payload["privacy_ledger"]["adjacency"] = "replace_one"
    measurement_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    manifest_path = transcript / "transcript_manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    measurement_record = manifest["artifacts"]["measurements"]
    measurement_record["bytes"] = measurement_path.stat().st_size
    measurement_record["sha256"] = hashlib.sha256(measurement_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    with pytest.raises(ValueError, match="add_remove adjacency"):
        verify_public_transcript(transcript)


def test_transcript_generation_rejects_configured_public_count_mismatch(
    tmp_path: Path,
) -> None:
    config, _, _ = _release_fixture(tmp_path)
    transcript = tmp_path / "transcript"
    materialize_public_transcript(config, transcript)
    generation_config = _prepare_generation_config(
        config,
        transcript,
        tmp_path / "split_generation",
    )
    generation_config["privacy"]["public_n_rows"] += 1

    with pytest.raises(ValueError, match="does not match the sealed transcript"):
        run_qdte(generation_config)


def test_split_process_supports_static_ice_exact_precision_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    rows = np.asarray(
        [[a, b, a ^ b] for a in range(2) for b in range(2) for _ in range(8)],
        dtype=np.int32,
    )
    private_csv = tmp_path / "private_static_ice.csv"
    pd.DataFrame(rows, columns=["a", "b", "c"]).to_csv(private_csv, index=False)
    public_schema = tmp_path / "public_schema.json"
    schema.save_json(public_schema)
    pairs = ((0, 1), (0, 2), (1, 2))
    qcat, groups = build_selected_pair_partition_workload(schema, pairs)
    transcript = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, pairs),
        public_total=len(rows),
        rho_total=0.2,
        rng=np.random.default_rng(411),
        allocation_mode="public_optimal",
    )
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=1.0e-9,
        target_projection="raw_reconstruction",
    )
    artifact = tmp_path / "static_ice_transcript"
    artifact.mkdir()
    schema.save_json(artifact / "schema.json")
    qcat.save_json(artifact / "queries.json")
    write_json(measurements.to_public_dict(), artifact / "measurements.json")
    write_public_transcript_manifest(artifact)

    legacy_output = tmp_path / "static_ice_legacy"
    base_config = {
        "run": {
            "dataset_name": "static_ice_split_test",
            "input_csv": str(private_csv),
            "output_dir": str(legacy_output),
            "seed": 412,
        },
        "preprocess": {"public_schema_json": str(public_schema)},
        "workload": {"reuse_from_measurement": True},
        "privacy": {
            "mode": "dp",
            "dp_release_mode": True,
            "public_row_count": True,
            "public_n_rows": len(rows),
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
            "log_measurement_groups": False,
        },
        "debug": {
            "recompute_after_batch": True,
            "assert_batch_loss_decrease": True,
        },
        "evaluation": {
            "compute_true_query_error": False,
            "compute_heldout_query_error": False,
            "save_synthetic_csv": False,
        },
    }
    legacy_metrics = run_qdte(base_config)
    private_csv.unlink()

    def _private_loader_is_forbidden(_config):
        raise AssertionError("Static-ICE transcript generation loaded private rows")

    monkeypatch.setattr(engine_module, "load_and_preprocess_csv", _private_loader_is_forbidden)
    split_output = tmp_path / "static_ice_split"
    split_config = _prepare_generation_config(
        base_config,
        artifact,
        split_output,
    )
    split_metrics = run_qdte(split_config)

    assert split_metrics["precision_operator"] == "orthogonal_interaction"
    assert split_metrics["precision_operator_active"] is True
    assert split_metrics["private_input_loaded_by_process"] is False
    assert split_metrics["final_optimization_objective"] == legacy_metrics[
        "final_optimization_objective"
    ]
    assert np.array_equal(
        np.load(legacy_output / "synthetic_initial_encoded.npy"),
        np.load(split_output / "synthetic_initial_encoded.npy"),
    )
    assert np.array_equal(
        np.load(legacy_output / "synthetic_encoded.npy"),
        np.load(split_output / "synthetic_encoded.npy"),
    )
