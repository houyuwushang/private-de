#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, set_nested
from qdte.dataio import write_json
from qdte.evolution.engine import run_qdte
from qdte.measurement.public_artifact import (
    PUBLIC_TRANSCRIPT_FILES,
    verify_public_transcript,
    write_public_transcript_manifest,
)
from scripts.generate_qdte_from_transcript import _prepare_generation_config
from scripts.run_coverage_refinement_wp8a import sha256_file


PROTOCOL_ID = "SAGE-QDTE-RCE-C1-20260715-v3"
METHOD_ID = "SAGE-QDTE-RCE-v1"
SOURCE_PROTOCOL_ID = "SAGE-QDTE-ICE-WP9-STATIC-CONFIRMATION-20260715-v1"
PROTOCOL_PATH = ROOT / "docs" / "SAGE_QDTE_RCE_C0_C1_PROTOCOL_20260715.md"
DEFAULT_SOURCE_ROOT = ROOT / "outputs" / "static_ice_wp9_formal_20260715"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_20260715"
DATASETS = ("adult", "br2000")
EPSILONS = (0.1, 0.3)
SEEDS = (0, 1, 2)
FORMAL_ITERATIONS = 5_000
FORMAL_CANDIDATES_PER_ITERATION = 4_096


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _verify_record(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(record["bytes"]) != path.stat().st_size:
        raise RuntimeError(f"Sealed source artifact size changed: {path}")
    if str(record["sha256"]) != sha256_file(path):
        raise RuntimeError(f"Sealed source artifact hash changed: {path}")
    return path


def _epsilon_dir(epsilon: float) -> str:
    return f"epsilon_{float(epsilon):g}"


def _forbidden_metric_keys(metrics: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in metrics
        if key.startswith("final_true_")
        or key.startswith("true_query_")
        or key.startswith("offline_")
    )


def _source_cell(
    source_root: Path,
    dataset: str,
    epsilon: float,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    cell = source_root.resolve() / dataset / _epsilon_dir(epsilon) / f"seed_{seed}"
    manifest = _read_json(cell / "mechanism_manifest.json")
    expected = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "dataset": dataset,
        "epsilon": float(epsilon),
        "seed": int(seed),
        "true_utility_evaluated": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"WP9 source cell violates {key}: {manifest.get(key)!r} != {value!r}"
            )
    if manifest.get("mechanism_gate", {}).get("passed") is not True:
        raise RuntimeError("WP9 source cell did not pass its mechanism gate")
    for key in ("measurement", "queries", "runtime", "metrics", "resolved_config"):
        _verify_record(manifest["artifacts"][key])
    _verify_record(manifest["artifacts"]["synthetic"])
    return cell, manifest


def _rce_config(
    source_cell: Path,
    public_transcript: Path,
    output_dir: Path,
    *,
    iterations: int,
) -> dict[str, Any]:
    source_config = load_yaml(source_cell / "generate" / "config_resolved.yaml")
    config = _prepare_generation_config(
        copy.deepcopy(source_config),
        public_transcript,
        output_dir.resolve(),
    )
    set_nested(config, "evaluation.compute_true_query_error", False)
    set_nested(config, "evaluation.compute_heldout_query_error", False)
    set_nested(config, "evaluation.save_synthetic_csv", False)
    set_nested(config, "measurement.fission.enabled", False)
    set_nested(config, "projection.uncertainty.enabled", False)
    set_nested(config, "qdte.max_iters", int(iterations))
    set_nested(config, "qdte.stop_patience", max(int(iterations), 1))
    set_nested(config, "qdte.entropy.enabled", False)
    set_nested(config, "qdte.confidence_stop.enabled", False)
    set_nested(config, "qdte.structured_swap_enabled", False)
    set_nested(config, "qdte.rce.enabled", True)
    set_nested(
        config,
        "qdte.rce.method",
        "row_realizable_confidence_set_entropic_primal_dual_v1",
    )
    set_nested(config, "qdte.rce.alpha_l2", 0.025)
    set_nested(config, "qdte.rce.alpha_linf", 0.025)
    set_nested(config, "qdte.rce.product_prior_smoothing", 1.0)
    set_nested(config, "method.protocol_id", PROTOCOL_ID)
    set_nested(config, "method.method_id", METHOD_ID)
    set_nested(
        config,
        "method.artifact_role",
        (
            "sealed_same_transcript_rce_c1_cell"
            if int(iterations) == FORMAL_ITERATIONS
            else "development_rce_c1_smoke"
        ),
    )
    return config


def _stage_public_transcript(source_cell: Path, destination: Path) -> dict[str, Any]:
    source_root = source_cell / "measurement"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"RCE C1 public transcript directory must be new or empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for relative_path in PUBLIC_TRANSCRIPT_FILES.values():
        source_path = source_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, destination / relative_path)

    source_manifest = _read_json(source_cell / "mechanism_manifest.json")
    raw_source_ledger = source_manifest.get("privacy")
    if not isinstance(raw_source_ledger, dict):
        raise ValueError("WP9 source manifest is missing its sealed privacy ledger")
    source_ledger = copy.deepcopy(raw_source_ledger)
    if source_ledger.get("adjacency") is None:
        entries = source_ledger.get("entries")
        entry_adjacencies = {
            entry.get("public_metadata", {}).get("adjacency")
            for entry in entries or []
            if isinstance(entry, dict)
            and isinstance(entry.get("public_metadata"), dict)
        }
        source_config = load_yaml(source_cell / "generate" / "config_resolved.yaml")
        configured_adjacency = source_config.get("privacy", {}).get("adjacency")
        if entry_adjacencies != {"add_remove"} or configured_adjacency != "add_remove":
            raise ValueError(
                "Legacy WP9 ledger adjacency is not uniquely certified as add_remove"
            )
        source_ledger["adjacency"] = "add_remove"
    manifest = write_public_transcript_manifest(
        destination,
        privacy_ledger_override=source_ledger,
    )
    verified = verify_public_transcript(destination)
    checks = {
        relative_path: (
            sha256_file(source_root / relative_path)
            == sha256_file(verified.root / relative_path)
        )
        for relative_path in PUBLIC_TRANSCRIPT_FILES.values()
    }
    if not all(checks.values()):
        raise RuntimeError(f"Staged public transcript differs from WP9 source: {checks}")
    return manifest


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    dataset = str(args.dataset).lower()
    epsilon = float(args.epsilon)
    seed = int(args.seed)
    iterations = int(args.iterations)
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    if epsilon not in EPSILONS:
        raise ValueError(f"epsilon must be one of {EPSILONS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if iterations <= 0 or iterations > FORMAL_ITERATIONS:
        raise ValueError(f"iterations must be in [1, {FORMAL_ITERATIONS}]")

    source_cell, source_manifest = _source_cell(
        args.source_root,
        dataset,
        epsilon,
        seed,
    )
    output_cell = (
        args.output_root.resolve()
        / dataset
        / _epsilon_dir(epsilon)
        / f"seed_{seed}"
    )
    if output_cell.exists() and any(output_cell.iterdir()):
        raise FileExistsError(f"RCE C1 output cell must be new or empty: {output_cell}")
    output_cell.mkdir(parents=True, exist_ok=True)
    public_transcript = output_cell / "public_transcript"
    transcript_manifest = _stage_public_transcript(source_cell, public_transcript)
    generation_dir = output_cell / "generate_rce"
    config = _rce_config(
        source_cell,
        public_transcript,
        generation_dir,
        iterations=iterations,
    )
    metrics = run_qdte(config)
    forbidden = _forbidden_metric_keys(metrics)
    if forbidden:
        raise RuntimeError(f"RCE C1 generator emitted forbidden true metrics: {forbidden}")

    runtime = _read_json(generation_dir / "runtime.json")
    run_status = _read_json(generation_dir / "run_status.json")
    persisted_metrics = _read_json(generation_dir / "metrics_final.json")
    source_initial = source_cell / "generate" / "synthetic_initial_encoded.npy"
    rce_initial = generation_dir / "synthetic_initial_encoded.npy"
    source_measurement = source_cell / "measurement" / "measurements.json"
    generated_measurement = generation_dir / "measurements.json"
    expected_candidates = (
        int(iterations)
        * int(config["qdte"].get("total_candidates_per_iter", FORMAL_CANDIDATES_PER_ITERATION))
    )
    checks = {
        "source_wp9_gate_passed": source_manifest["mechanism_gate"]["passed"] is True,
        "public_transcript_sealed": (
            transcript_manifest.get("generation_authorized") is True
            and transcript_manifest.get("private_input_hashes_excluded") is True
            and transcript_manifest.get("true_utility_evaluated") is False
        ),
        "public_transcript_matches_source_bytewise": all(
            sha256_file(source_cell / "measurement" / relative_path)
            == sha256_file(public_transcript / relative_path)
            for relative_path in PUBLIC_TRANSCRIPT_FILES.values()
        ),
        "measurement_reused_bytewise": (
            sha256_file(source_measurement) == sha256_file(generated_measurement)
        ),
        "initial_table_matches_control_bytewise": (
            sha256_file(source_initial) == sha256_file(rce_initial)
        ),
        "exact_raw_orthogonal_precision": (
            persisted_metrics.get("precision_operator") == "orthogonal_interaction"
            and persisted_metrics.get("precision_operator_active") is True
        ),
        "rce_v1_active": (
            persisted_metrics.get("rce_enabled") is True
            and persisted_metrics.get("rce_diagnostics", {}).get("method")
            == "row_realizable_confidence_set_entropic_primal_dual_v1"
        ),
        "rate_space_kl_scaling": (
            persisted_metrics.get("rce_regularizer_definition")
            == "D_KL(p_empirical||p0)"
            and math.isclose(
                float(persisted_metrics.get("initial_rce_regularizer", math.nan)),
                float(persisted_metrics.get("initial_rce_kl_per_row", math.nan)),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                float(persisted_metrics.get("rce_effective_lambda_cost", math.nan)),
                float(config["qdte"].get("lambda_cost", 0.0))
                / float(persisted_metrics["num_rows_synthetic"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ),
        "integer_certificate_kl_scaling": (
            persisted_metrics.get("rce_diagnostics", {})
            .get("integer_incumbent", {})
            .get("regularizer_definition")
            == "D_KL(p_empirical||p0)"
            and math.isclose(
                float(
                    persisted_metrics["rce_diagnostics"]["integer_incumbent"][
                        "best_kl_per_row"
                    ]
                ),
                float(
                    persisted_metrics["rce_diagnostics"]["integer_incumbent"][
                        "best_regularizer"
                    ]
                ),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ),
        "legacy_extensions_disabled": (
            persisted_metrics.get("entropy_enabled") is False
            and persisted_metrics.get("confidence_stop_enabled") is False
            and runtime.get("measurement_fission_enabled") is False
        ),
        "iteration_count": int(run_status.get("num_iterations", -1)) == iterations,
        "candidate_count": (
            int(persisted_metrics.get("num_candidates_requested", -1))
            == expected_candidates
            and int(persisted_metrics.get("num_candidates_scored", -1))
            == expected_candidates
        ),
        "no_true_utility_emitted": not _forbidden_metric_keys(persisted_metrics),
        "transcript_only_process_isolation": (
            persisted_metrics.get("transcript_only_generation") is True
            and persisted_metrics.get("private_input_loaded_by_process") is False
            and runtime.get("transcript_only_generation") is True
            and runtime.get("private_input_loaded_by_process") is False
            and config.get("run", {}).get("input_csv") in {None, ""}
        ),
        "finite_rce_certificate": all(
            math.isfinite(float(value))
            for value in (
                persisted_metrics["final_rce_regularizer"],
                persisted_metrics["final_rce_lagrangian"],
                persisted_metrics["rce_diagnostics"]["final_confidence"]["slack"],
            )
        ),
    }
    gate = {
        "passed": all(checks.values()),
        "checks": checks,
        "iterations": iterations,
        "candidates_per_iteration": int(
            config["qdte"].get("total_candidates_per_iter", FORMAL_CANDIDATES_PER_ITERATION)
        ),
        "expected_candidates": expected_candidates,
        "observed_candidates": int(persisted_metrics.get("num_candidates_scored", -1)),
    }
    if not gate["passed"]:
        raise RuntimeError(f"RCE C1 mechanism gate failed: {checks}")

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": (
            "sealed_same_transcript_rce_c1_cell"
            if iterations == FORMAL_ITERATIONS
            else "development_rce_c1_smoke"
        ),
        "dataset": dataset,
        "declared_dataset": source_manifest["declared_dataset"],
        "epsilon": epsilon,
        "seed": seed,
        "iterations": iterations,
        "measurement_seed": source_manifest["measurement_seed"],
        "generation_seed": source_manifest["generation_seed"],
        "mechanism_gate": gate,
        "source_control": {
            "manifest": _file_record(source_cell / "mechanism_manifest.json"),
            "measurement": _file_record(source_measurement),
            "initial_table": _file_record(source_initial),
            "synthetic": _file_record(source_cell / "generate" / "synthetic_encoded.npy"),
        },
        "artifacts": {
            "protocol": _file_record(PROTOCOL_PATH),
            "measurement": _file_record(generated_measurement),
            "public_transcript_manifest": _file_record(
                public_transcript / "transcript_manifest.json"
            ),
            "public_transcript_schema": _file_record(public_transcript / "schema.json"),
            "public_transcript_queries": _file_record(public_transcript / "queries.json"),
            "public_transcript_measurements": _file_record(
                public_transcript / "measurements.json"
            ),
            "initial_table": _file_record(rce_initial),
            "synthetic": _file_record(generation_dir / "synthetic_encoded.npy"),
            "metrics": _file_record(generation_dir / "metrics_final.json"),
            "runtime": _file_record(generation_dir / "runtime.json"),
            "run_status": _file_record(generation_dir / "run_status.json"),
            "resolved_config": _file_record(generation_dir / "config_resolved.yaml"),
        },
        "true_utility_evaluated": False,
        "private_input_hashes_excluded": True,
    }
    write_json(manifest, output_cell / "mechanism_manifest.json")
    write_json(
        {
            "status": "completed",
            "protocol_id": PROTOCOL_ID,
            "dataset": dataset,
            "epsilon": epsilon,
            "seed": seed,
            "iterations": iterations,
            "formal": iterations == FORMAL_ITERATIONS,
            "true_utility_evaluated": False,
        },
        output_cell / "cell_status.json",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one sealed same-transcript RCE C1 cell")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--epsilon", required=True, type=float, choices=EPSILONS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--iterations", type=int, default=FORMAL_ITERATIONS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    manifest = run_cell(parse_args())
    print(json.dumps(manifest["mechanism_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
