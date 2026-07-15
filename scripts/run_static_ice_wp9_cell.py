#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, set_nested
from qdte.dataio import ensure_dir, write_json
from qdte.evolution.engine import run_qdte
from qdte.measurement.factorization import (
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
)
from qdte.privacy.accountant import ZCDPPrivacyFilter
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import TableSchema
from scripts.run_coverage_refinement_wp8a import _seed, sha256_file
from scripts.run_static_ice_measurement_pilot import public_pair_scopes, rho_for_epsilon
from scripts.run_static_ice_qdte_pilot import _generation_config, _write_measurement_artifact


PROTOCOL_ID = "SAGE-QDTE-ICE-WP9-STATIC-CONFIRMATION-20260715-v1"
METHOD_ID = "SAGE-QDTE-Static-ICE-Exact-v1"
PROTOCOL_PATH = (
    ROOT / "docs" / "SAGE_QDTE_ICE_WP9_STATIC_CONFIRMATION_PROTOCOL_20260715.md"
)
DATASETS = ("nltcs", "acs", "br2000", "adult")
EPSILONS = (0.1, 0.3)
SEEDS = (0, 1, 2)
DELTA_DP = 1.0e-9
MAX_PAIR_CELLS = 20_000
STAGE_ITERS = 5_000
CANDIDATES_PER_ITER = 4_096
EXPECTED_CANDIDATES = STAGE_ITERS * CANDIDATES_PER_ITER


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
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


def validate_request(
    *,
    dataset: str,
    epsilon: float,
    delta: float,
    seed: int,
    max_pair_cells: int,
    stage_iters: int,
) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"WP9 dataset must be one of {DATASETS}")
    if not any(
        math.isclose(float(epsilon), value, rel_tol=0.0, abs_tol=1.0e-12)
        for value in EPSILONS
    ):
        raise ValueError(f"WP9 epsilon must be one of {EPSILONS}")
    if not math.isclose(float(delta), DELTA_DP, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"WP9 fixes delta={DELTA_DP}")
    if int(seed) not in SEEDS:
        raise ValueError(f"WP9 seed must be one of {SEEDS}")
    if int(max_pair_cells) != MAX_PAIR_CELLS:
        raise ValueError(f"WP9 fixes max_pair_cells={MAX_PAIR_CELLS}")
    if int(stage_iters) != STAGE_ITERS:
        raise ValueError(f"WP9 fixes stage_iters={STAGE_ITERS}")


def _write_transcript(
    artifact_dir: Path,
    *,
    transcript,
    schema: TableSchema,
    delta: float,
    privacy_ledger: dict[str, Any],
) -> None:
    qcat, groups = build_selected_pair_partition_workload(
        schema,
        transcript.strategy.pairs,
    )
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=float(delta),
        target_projection="raw_reconstruction",
    )
    diagnostics = dict(measurements.projection_diagnostics or {})
    diagnostics["wp9_static_confirmation"] = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "exact_orthogonal_precision_required": True,
        "privacy_ledger": privacy_ledger,
        "true_utility_evaluated": False,
    }
    measurements.projection_diagnostics = diagnostics
    _write_measurement_artifact(
        artifact_dir,
        qcat=qcat,
        schema=schema,
        measurements=measurements,
        extra_public=transcript.to_public_dict(),
    )


def _generation_config_wp9(
    *,
    base_config: dict[str, Any],
    input_dir: Path,
    artifact_dir: Path,
    output_dir: Path,
    dataset_name: str,
    epsilon: float,
    rho_total: float,
    seed: int,
) -> dict[str, Any]:
    config = _generation_config(
        base_config,
        input_csv=input_dir / "raw.csv",
        public_schema=input_dir / "schema.json",
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        dataset_name=dataset_name,
        epsilon=float(epsilon),
        rho_total=float(rho_total),
        delta=DELTA_DP,
        seed=int(seed),
        max_iters=STAGE_ITERS,
        precision_operator="orthogonal_interaction",
        exact_scoring_chunk_size=512,
    )
    set_nested(config, "qdte.stop_patience", STAGE_ITERS)
    set_nested(config, "evaluation.compute_true_query_error", False)
    set_nested(config, "evaluation.compute_heldout_query_error", False)
    set_nested(config, "evaluation.save_synthetic_csv", False)
    set_nested(config, "method.protocol_id", PROTOCOL_ID)
    set_nested(config, "method.method_id", METHOD_ID)
    set_nested(config, "method.artifact_role", "blind_dp_postprocessing_confirmation")
    return config


def _forbidden_metric_keys(metrics: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in metrics
        if key.startswith("final_true_")
        or key.startswith("true_query_")
        or key.startswith("offline_")
    )


def _mechanism_gate(
    *,
    transcript,
    ledger: ZCDPPrivacyFilter,
    rho_total: float,
    pairs: tuple[tuple[int, int], ...],
    cards: np.ndarray,
    metrics: dict[str, Any],
    run_status: dict[str, Any],
    runtime: dict[str, Any],
    artifact_dir: Path,
    generation_dir: Path,
) -> dict[str, Any]:
    pair_cap_ok = all(
        int(cards[left]) * int(cards[right]) <= MAX_PAIR_CELLS
        for left, right in pairs
    )
    generated_measurement = generation_dir / "measurements.json"
    source_measurement = artifact_dir / "measurements.json"
    checks = {
        "rho_spent_matches_declared": bool(
            np.isclose(transcript.rho_spent, rho_total, rtol=1.0e-11, atol=1.0e-15)
            and np.isclose(ledger.rho_spent, rho_total, rtol=1.0e-11, atol=1.0e-15)
        ),
        "all_component_variances_positive": bool(
            all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in transcript.component_variances.values()
            )
        ),
        "all_pairs_within_public_cap": bool(pair_cap_ok),
        "generation_reused_transcript_bytewise": bool(
            generated_measurement.is_file()
            and sha256_file(generated_measurement) == sha256_file(source_measurement)
        ),
        "exact_precision_operator_active": bool(
            metrics.get("precision_operator") == "orthogonal_interaction"
            and metrics.get("precision_operator_active") is True
            and runtime.get("precision_operator") == "orthogonal_interaction"
            and runtime.get("score_backend") == "precision_operator"
        ),
        "full_iteration_count": int(run_status.get("num_iterations", -1)) == STAGE_ITERS,
        "full_candidate_count": bool(
            int(metrics.get("num_candidates_requested", -1)) == EXPECTED_CANDIDATES
            and int(metrics.get("num_candidates_scored", -1)) == EXPECTED_CANDIDATES
            and int(runtime.get("num_candidates_requested", -1)) == EXPECTED_CANDIDATES
            and int(runtime.get("num_candidates_scored", -1)) == EXPECTED_CANDIDATES
        ),
        "incremental_answer_drift_zero": bool(
            abs(float(metrics.get("final_incremental_answer_drift", math.inf)))
            <= 1.0e-8
        ),
        "no_true_utility_emitted": not _forbidden_metric_keys(metrics),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "num_strategy_blocks": len(transcript.strategy.blocks),
        "num_pair_scopes": len(pairs),
        "effective_rank": int(
            metrics.get("precision_operator_diagnostics", {}).get("effective_rank", -1)
        ),
        "rho_declared": float(rho_total),
        "rho_spent": float(ledger.rho_spent),
        "num_iterations": int(run_status.get("num_iterations", -1)),
        "num_candidates_scored": int(metrics.get("num_candidates_scored", -1)),
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    validate_request(
        dataset=str(args.dataset),
        epsilon=float(args.epsilon),
        delta=float(args.delta),
        seed=int(args.seed),
        max_pair_cells=int(args.max_pair_cells),
        stage_iters=int(args.stage_iters),
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"WP9 cell output must be new or empty: {output_dir}")
    ensure_dir(output_dir)

    input_dir = args.input_dir.resolve()
    config_path = args.config.resolve()
    schema = TableSchema.load_json(input_dir / "schema.json")
    private_rows = np.load(input_dir / "real_encoded.npy", allow_pickle=False).astype(
        np.int32
    )
    metadata = _read_json(input_dir / "metadata.json")
    public_total = int(metadata["n_rows"])
    declared_dataset = f"{args.dataset}_sage_strong"
    if str(metadata.get("dataset")) != declared_dataset:
        raise ValueError(
            f"input dataset {metadata.get('dataset')!r} != {declared_dataset!r}"
        )
    if private_rows.shape != (public_total, schema.d):
        raise ValueError("encoded private table does not match public schema/n")

    pairs = public_pair_scopes(schema.cardinalities, MAX_PAIR_CELLS)
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, pairs)
    rho_total = rho_for_epsilon(float(args.epsilon), DELTA_DP)
    measurement_seed = _seed(int(args.seed), 0xB453)
    generation_seed = _seed(int(args.seed), 0x57A63, 0)
    transcript = measure_hierarchical_pair_interactions(
        private_rows,
        strategy,
        public_total=public_total,
        rho_total=rho_total,
        rng=np.random.default_rng(measurement_seed),
        allocation_mode="public_optimal",
    )
    ledger = ZCDPPrivacyFilter(rho_total)
    ledger.spend(
        label="full_static_ice_exact",
        mechanism="gaussian_hierarchical_orthogonal_vectors",
        rho=rho_total,
        public_metadata={
            "num_blocks": len(strategy.blocks),
            "num_pairs": len(pairs),
            "adjacency": "add_remove",
            "allocation": "public_optimal",
        },
    )

    artifact_dir = output_dir / "measurement"
    _write_transcript(
        artifact_dir,
        transcript=transcript,
        schema=schema,
        delta=DELTA_DP,
        privacy_ledger=ledger.to_public_dict(delta=DELTA_DP),
    )
    generation_dir = output_dir / "generate"
    config = _generation_config_wp9(
        base_config=load_yaml(config_path),
        input_dir=input_dir,
        artifact_dir=artifact_dir,
        output_dir=generation_dir,
        dataset_name=declared_dataset,
        epsilon=float(args.epsilon),
        rho_total=rho_total,
        seed=generation_seed,
    )
    metrics = run_qdte(config)
    forbidden = _forbidden_metric_keys(metrics)
    if forbidden:
        raise RuntimeError(f"WP9 generator emitted forbidden true metrics: {forbidden}")

    synthetic_path = generation_dir / "synthetic_encoded.npy"
    run_status_path = generation_dir / "run_status.json"
    runtime_path = generation_dir / "runtime.json"
    metrics_path = generation_dir / "metrics_final.json"
    for path in (synthetic_path, run_status_path, runtime_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_status = _read_json(run_status_path)
    runtime = _read_json(runtime_path)
    persisted_metrics = _read_json(metrics_path)
    gate = _mechanism_gate(
        transcript=transcript,
        ledger=ledger,
        rho_total=rho_total,
        pairs=pairs,
        cards=schema.cardinalities,
        metrics=persisted_metrics,
        run_status=run_status,
        runtime=runtime,
        artifact_dir=artifact_dir,
        generation_dir=generation_dir,
    )
    if not gate["passed"]:
        raise RuntimeError(f"WP9 mechanism gate failed: {gate['checks']}")

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "sealed_blind_static_ice_cell",
        "dataset": str(args.dataset),
        "declared_dataset": declared_dataset,
        "epsilon": float(args.epsilon),
        "delta": DELTA_DP,
        "seed": int(args.seed),
        "measurement_seed": measurement_seed,
        "generation_seed": generation_seed,
        "public_total": public_total,
        "privacy": ledger.to_public_dict(delta=DELTA_DP),
        "strategy": {
            "allocation": "public_optimal",
            "max_pair_cells": MAX_PAIR_CELLS,
            "num_blocks": len(strategy.blocks),
            "num_pairs": len(pairs),
        },
        "generator": {
            "precision_operator": "orthogonal_interaction",
            "iterations": STAGE_ITERS,
            "candidates_per_iteration": CANDIDATES_PER_ITER,
            "expected_candidates": EXPECTED_CANDIDATES,
        },
        "mechanism_gate": gate,
        "artifacts": {
            "protocol": _file_record(PROTOCOL_PATH),
            "config_source": _file_record(config_path),
            "public_schema": _file_record(input_dir / "schema.json"),
            "public_metadata": _file_record(input_dir / "metadata.json"),
            "measurement": _file_record(artifact_dir / "measurements.json"),
            "queries": _file_record(artifact_dir / "queries.json"),
            "synthetic": _file_record(synthetic_path),
            "run_status": _file_record(run_status_path),
            "runtime": _file_record(runtime_path),
            "metrics": _file_record(metrics_path),
            "resolved_config": _file_record(generation_dir / "config_resolved.yaml"),
        },
        "true_utility_evaluated": False,
        "private_input_hashes_excluded": True,
    }
    write_json(manifest, output_dir / "mechanism_manifest.json")
    write_json(
        {
            "status": "completed",
            "protocol_id": PROTOCOL_ID,
            "dataset": str(args.dataset),
            "epsilon": float(args.epsilon),
            "seed": int(args.seed),
            "true_utility_evaluated": False,
        },
        output_dir / "cell_status.json",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one blind Static-ICE-Exact WP9 confirmation cell."
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--delta", type=float, default=DELTA_DP)
    parser.add_argument("--max-pair-cells", type=int, default=MAX_PAIR_CELLS)
    parser.add_argument("--stage-iters", type=int, default=STAGE_ITERS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_cell(args)
    except Exception as error:
        output_dir = args.output_dir.resolve()
        ensure_dir(output_dir)
        write_json(
            {
                "status": "failed",
                "protocol_id": PROTOCOL_ID,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "true_utility_evaluated": False,
            },
            output_dir / "cell_status.json",
        )
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "dataset": result["dataset"],
                "epsilon": result["epsilon"],
                "seed": result["seed"],
                "true_utility_evaluated": result["true_utility_evaluated"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
