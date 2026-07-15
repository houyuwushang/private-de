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
    HierarchicalInteractionTranscript,
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import interaction_transcript_to_diagonal_measurements
from qdte.measurement.workload_factorization import (
    allocation_risk,
    factorize_public_workload,
    optimal_rho_for_importance,
)
from qdte.privacy.accountant import ZCDPPrivacyFilter
from qdte.queries.orthogonal import interaction_coefficients, oneway_contrast_from_counts
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema
from scripts.run_coverage_refinement_wp8a import _seed, sha256_file
from scripts.run_static_ice_measurement_pilot import public_pair_scopes, rho_for_epsilon
from scripts.run_static_ice_qdte_pilot import _write_measurement_artifact
from scripts.run_static_ice_wp9_cell import (
    CANDIDATES_PER_ITER,
    EXPECTED_CANDIDATES,
    MAX_PAIR_CELLS,
    METHOD_ID as CONTROL_METHOD_ID,
    PROTOCOL_ID as CONTROL_PROTOCOL_ID,
    STAGE_ITERS,
    _forbidden_metric_keys,
    _generation_config_wp9,
    _mechanism_gate,
)


PROTOCOL_ID = "SAGE-QDTE-ICE-WP10A-WORKLOAD-ALLOCATION-20260715-v1"
CANDIDATE_METHOD_ID = "SAGE-QDTE-Static-ICE-WorkloadL2-v1"
PROTOCOL_PATH = ROOT / "docs" / "SAGE_QDTE_ICE_WP10A_WORKLOAD_ALLOCATION_PROTOCOL_20260715.md"
EVALUATOR_PATH = ROOT / "scripts" / "evaluate_static_ice_wp10a_workload_allocation.py"
DATASET = "adult"
DECLARED_DATASET = "adult_sage_strong"
EPSILON = 0.1
DELTA_DP = 1.0e-9
SEED = 0
EXPECTED_FULL_QUERIES = 28_654
EXPECTED_SUPPORTED_QUERIES = 22_534
EXPECTED_UNSUPPORTED_QUERIES = 6_120
EXPECTED_RISK_RATIO = 0.8455718531684706
EXPECTED_CURRENT_PAIR_SHARE = 0.9197821438450282
EXPECTED_WORKLOAD_PAIR_SHARE = 0.8465777443293414


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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
    path = Path(str(record["path"])).resolve()
    if _file_record(path) != record:
        raise RuntimeError(f"frozen artifact changed: {path}")
    return path


def validate_request(args: argparse.Namespace) -> None:
    if str(args.dataset) != DATASET:
        raise ValueError(f"WP10a fixes dataset={DATASET}")
    if not math.isclose(float(args.epsilon), EPSILON, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"WP10a fixes epsilon={EPSILON}")
    if not math.isclose(float(args.delta), DELTA_DP, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"WP10a fixes delta={DELTA_DP}")
    if int(args.seed) != SEED:
        raise ValueError(f"WP10a fixes seed={SEED}")
    if int(args.stage_iters) != STAGE_ITERS:
        raise ValueError(f"WP10a fixes stage_iters={STAGE_ITERS}")
    if int(args.max_pair_cells) != MAX_PAIR_CELLS:
        raise ValueError(f"WP10a fixes max_pair_cells={MAX_PAIR_CELLS}")


def _control_records(control_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = control_dir / "mechanism_manifest.json"
    manifest = _read_json(manifest_path)
    expected = {
        "protocol_id": CONTROL_PROTOCOL_ID,
        "method_id": CONTROL_METHOD_ID,
        "dataset": DATASET,
        "epsilon": EPSILON,
        "seed": SEED,
        "true_utility_evaluated": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"WP10a control violates {key}: {manifest.get(key)!r}")
    if manifest.get("mechanism_gate", {}).get("passed") is not True:
        raise RuntimeError("WP10a control mechanism gate did not pass")
    artifacts = manifest.get("artifacts", {})
    records = {
        "manifest": _file_record(manifest_path),
        "measurement": dict(artifacts["measurement"]),
        "synthetic": dict(artifacts["synthetic"]),
        "metrics": dict(artifacts["metrics"]),
        "runtime": dict(artifacts["runtime"]),
        "initial": _file_record(control_dir / "generate" / "synthetic_initial_encoded.npy"),
    }
    for record in records.values():
        _verify_record(record)
    return records


def _build_plan(
    *,
    input_dir: Path,
    config_path: Path,
    control_dir: Path,
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "candidate_method_id": CANDIDATE_METHOD_ID,
        "artifact_role": "frozen_pre_generation_workload_allocation_plan",
        "dataset": DATASET,
        "declared_dataset": DECLARED_DATASET,
        "epsilon": EPSILON,
        "delta": DELTA_DP,
        "seed": SEED,
        "public_input_dir": str(input_dir.resolve()),
        "offline_evaluator_batch_size": 8192,
        "sources": {
            "protocol": _file_record(PROTOCOL_PATH),
            "runner": _file_record(Path(__file__)),
            "evaluator": _file_record(EVALUATOR_PATH),
            "factorization": _file_record(ROOT / "qdte" / "measurement" / "factorization.py"),
            "workload_factorization": _file_record(
                ROOT / "qdte" / "measurement" / "workload_factorization.py"
            ),
            "config": _file_record(config_path),
            "public_schema": _file_record(input_dir / "schema.json"),
            "public_metadata": _file_record(input_dir / "metadata.json"),
            "public_queries": _file_record(input_dir / "queries_full.json"),
            "public_groups": _file_record(input_dir / "workload_groups.json"),
        },
        "control": _control_records(control_dir),
        "true_utility_evaluated": False,
    }


def _exact_components(
    rows: np.ndarray,
    strategy,
) -> dict[str, np.ndarray]:
    cards = np.asarray(strategy.cardinalities, dtype=np.int64)
    exact: dict[str, np.ndarray] = {}
    for block in strategy.blocks:
        if block.kind == "oneway_contrast":
            attr = int(block.scope[0])
            counts = np.bincount(rows[:, attr], minlength=int(cards[attr])).astype(np.float64)
            exact[block.name] = oneway_contrast_from_counts(counts)
        elif block.kind == "pair_interaction":
            exact[block.name] = interaction_coefficients(rows, block.scope, cards)
        else:
            raise RuntimeError(f"unsupported strategy block {block.kind!r}")
    return exact


def _transcript_reproduction(
    regenerated: HierarchicalInteractionTranscript,
    sealed: HierarchicalInteractionTranscript,
) -> dict[str, Any]:
    structure_equal = bool(
        regenerated.strategy.cardinalities == sealed.strategy.cardinalities
        and regenerated.strategy.pairs == sealed.strategy.pairs
        and regenerated.allocation_mode == sealed.allocation_mode == "public_optimal"
    )
    coefficient_max_abs = 0.0
    variance_max_abs = 0.0
    rho_max_abs = 0.0
    for block in regenerated.strategy.blocks:
        name = block.name
        coefficient_max_abs = max(
            coefficient_max_abs,
            float(np.max(np.abs(regenerated.noisy_components[name] - sealed.noisy_components[name]))),
        )
        variance_max_abs = max(
            variance_max_abs,
            abs(regenerated.component_variances[name] - sealed.component_variances[name]),
        )
        rho_max_abs = max(
            rho_max_abs,
            abs(regenerated.rho_by_block[name] - sealed.rho_by_block[name]),
        )
    return {
        "passed": bool(
            structure_equal
            and coefficient_max_abs <= 1.0e-12
            and variance_max_abs <= 1.0e-12
            and rho_max_abs <= 1.0e-15
        ),
        "structure_equal": structure_equal,
        "coefficient_max_abs": coefficient_max_abs,
        "variance_max_abs": variance_max_abs,
        "rho_max_abs": rho_max_abs,
    }


def _noise_coupling(
    control: HierarchicalInteractionTranscript,
    candidate: HierarchicalInteractionTranscript,
    exact: dict[str, np.ndarray],
) -> dict[str, Any]:
    max_abs = 0.0
    for block in control.strategy.blocks:
        name = block.name
        control_noise = (control.noisy_components[name] - exact[name]) / math.sqrt(
            control.component_variances[name]
        )
        candidate_noise = (candidate.noisy_components[name] - exact[name]) / math.sqrt(
            candidate.component_variances[name]
        )
        max_abs = max(max_abs, float(np.max(np.abs(control_noise - candidate_noise))))
    return {"passed": max_abs <= 1.0e-12, "standard_normal_max_abs": max_abs}


def _write_candidate_transcript(
    artifact_dir: Path,
    *,
    transcript: HierarchicalInteractionTranscript,
    schema: TableSchema,
    ledger: ZCDPPrivacyFilter,
    allocation_certificate: dict[str, Any],
) -> None:
    qcat, groups = build_selected_pair_partition_workload(schema, transcript.strategy.pairs)
    measurements = interaction_transcript_to_diagonal_measurements(
        transcript,
        qcat,
        groups,
        delta=DELTA_DP,
        target_projection="raw_reconstruction",
    )
    diagnostics = dict(measurements.projection_diagnostics or {})
    diagnostics["wp10a_workload_allocation"] = {
        "protocol_id": PROTOCOL_ID,
        "candidate_method_id": CANDIDATE_METHOD_ID,
        "allocation_certificate": allocation_certificate,
        "privacy_ledger": ledger.to_public_dict(delta=DELTA_DP),
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_request(args)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"WP10a output must be new or empty: {output}")
    ensure_dir(output)
    input_dir = args.input_dir.resolve()
    config_path = args.config.resolve()
    control_dir = args.control_dir.resolve()
    plan = _build_plan(input_dir=input_dir, config_path=config_path, control_dir=control_dir)
    plan_path = output / "plan.json"
    write_json(plan, plan_path)

    schema = TableSchema.load_json(input_dir / "schema.json")
    metadata = _read_json(input_dir / "metadata.json")
    if str(metadata.get("dataset")) != DECLARED_DATASET:
        raise ValueError("WP10a public metadata is not Adult strong")
    public_total = int(metadata["n_rows"])
    private_rows = np.load(input_dir / "real_encoded.npy", allow_pickle=False).astype(np.int32)
    if private_rows.shape != (public_total, schema.d):
        raise ValueError("private measurement rows do not match public schema/n")

    pairs = public_pair_scopes(schema.cardinalities, MAX_PAIR_CELLS)
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, pairs)
    full_qcat = QueryCatalogue.from_dict(_read_json(input_dir / "queries_full.json"))
    cell_mask = np.asarray(
        [family in {"oneway", "twoway"} for family in full_qcat.families],
        dtype=bool,
    )
    cell_reference = factorize_public_workload(strategy, full_qcat, include=cell_mask)
    cell_reference_error = max(
        abs(cell_reference.importance_by_block[block.name] - block.public_importance)
        / max(block.public_importance, 1.0e-300)
        for block in strategy.blocks
    )
    workload = factorize_public_workload(strategy, full_qcat)
    if not (
        full_qcat.m == EXPECTED_FULL_QUERIES
        and len(workload.supported_query_ids) == EXPECTED_SUPPORTED_QUERIES
        and len(workload.unsupported_query_ids) == EXPECTED_UNSUPPORTED_QUERIES
        and cell_reference_error <= 1.0e-10
    ):
        raise RuntimeError("WP10a public workload factorization changed from the protocol")

    rho_total = rho_for_epsilon(EPSILON, DELTA_DP)
    measurement_seed = _seed(SEED, 0xB453)
    generation_seed = _seed(SEED, 0x57A63, 0)
    regenerated_control = measure_hierarchical_pair_interactions(
        private_rows,
        strategy,
        public_total=public_total,
        rho_total=rho_total,
        rng=np.random.default_rng(measurement_seed),
        allocation_mode="public_optimal",
    )
    control_measurement = _read_json(_verify_record(plan["control"]["measurement"]))
    sealed_control = HierarchicalInteractionTranscript.from_public_dict(
        control_measurement["strategy_transcript"]
    )
    reproduction = _transcript_reproduction(regenerated_control, sealed_control)
    if not reproduction["passed"]:
        raise RuntimeError(f"WP10a failed to reproduce the frozen control: {reproduction}")

    workload_rho = optimal_rho_for_importance(
        strategy,
        workload.importance_by_block,
        rho_total,
    )
    current_risk = allocation_risk(
        strategy,
        workload.importance_by_block,
        regenerated_control.rho_by_block,
    )
    optimal_risk = allocation_risk(
        strategy,
        workload.importance_by_block,
        workload_rho,
    )
    risk_ratio = optimal_risk / current_risk
    current_pair_share = sum(
        regenerated_control.rho_by_block[block.name]
        for block in strategy.blocks
        if block.kind == "pair_interaction"
    ) / rho_total
    workload_pair_share = sum(
        workload_rho[block.name]
        for block in strategy.blocks
        if block.kind == "pair_interaction"
    ) / rho_total
    if not (
        math.isclose(risk_ratio, EXPECTED_RISK_RATIO, rel_tol=1.0e-12, abs_tol=1.0e-12)
        and math.isclose(
            current_pair_share,
            EXPECTED_CURRENT_PAIR_SHARE,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            workload_pair_share,
            EXPECTED_WORKLOAD_PAIR_SHARE,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise RuntimeError("WP10a public allocation certificate changed from the protocol")

    candidate = measure_hierarchical_pair_interactions(
        private_rows,
        strategy,
        public_total=public_total,
        rho_total=rho_total,
        rng=np.random.default_rng(measurement_seed),
        allocation_mode="workload_optimal",
        rho_by_block_override=workload_rho,
    )
    coupling = _noise_coupling(
        regenerated_control,
        candidate,
        _exact_components(private_rows, strategy),
    )
    if not coupling["passed"]:
        raise RuntimeError(f"WP10a standard-normal coupling failed: {coupling}")

    ledger = ZCDPPrivacyFilter(rho_total)
    ledger.spend(
        label="full_static_ice_workload_l2",
        mechanism="gaussian_hierarchical_orthogonal_vectors",
        rho=rho_total,
        public_metadata={
            "num_blocks": len(strategy.blocks),
            "num_pairs": len(pairs),
            "adjacency": "add_remove",
            "allocation": "workload_optimal",
            "public_workload_queries": full_qcat.m,
            "supported_queries": len(workload.supported_query_ids),
        },
    )
    allocation_certificate = {
        "profile": "full_reconstructable_query_l2",
        "full_queries": full_qcat.m,
        "supported_queries": len(workload.supported_query_ids),
        "unsupported_queries": len(workload.unsupported_query_ids),
        "cell_reference_max_relative_error": cell_reference_error,
        "current_risk": current_risk,
        "optimal_risk": optimal_risk,
        "optimal_over_current": risk_ratio,
        "current_pair_rho_share": current_pair_share,
        "workload_pair_rho_share": workload_pair_share,
    }
    artifact_dir = output / "measurement"
    _write_candidate_transcript(
        artifact_dir,
        transcript=candidate,
        schema=schema,
        ledger=ledger,
        allocation_certificate=allocation_certificate,
    )

    generation_dir = output / "generate"
    config = _generation_config_wp9(
        base_config=load_yaml(config_path),
        input_dir=input_dir,
        artifact_dir=artifact_dir,
        output_dir=generation_dir,
        dataset_name=DECLARED_DATASET,
        epsilon=EPSILON,
        rho_total=rho_total,
        seed=generation_seed,
    )
    set_nested(config, "init.encoded_npy", str(_verify_record(plan["control"]["initial"])))
    set_nested(config, "method.protocol_id", PROTOCOL_ID)
    set_nested(config, "method.method_id", CANDIDATE_METHOD_ID)
    set_nested(config, "method.artifact_role", "blind_workload_allocation_candidate")
    metrics = run_qdte(config)
    forbidden = _forbidden_metric_keys(metrics)
    if forbidden:
        raise RuntimeError(f"WP10a generator emitted forbidden true metrics: {forbidden}")

    metrics_path = generation_dir / "metrics_final.json"
    runtime_path = generation_dir / "runtime.json"
    run_status_path = generation_dir / "run_status.json"
    synthetic_path = generation_dir / "synthetic_encoded.npy"
    initial_path = generation_dir / "synthetic_initial_encoded.npy"
    persisted_metrics = _read_json(metrics_path)
    runtime = _read_json(runtime_path)
    run_status = _read_json(run_status_path)
    base_gate = _mechanism_gate(
        transcript=candidate,
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
    extra_checks = {
        "control_transcript_reproduced": reproduction["passed"],
        "standard_normal_noise_coupled": coupling["passed"],
        "cell_reference_identity": cell_reference_error <= 1.0e-10,
        "workload_factorization_frozen": bool(
            full_qcat.m == EXPECTED_FULL_QUERIES
            and len(workload.supported_query_ids) == EXPECTED_SUPPORTED_QUERIES
            and len(workload.unsupported_query_ids) == EXPECTED_UNSUPPORTED_QUERIES
        ),
        "closed_form_risk_certificate": math.isclose(
            risk_ratio,
            EXPECTED_RISK_RATIO,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        "allocation_mode_is_workload_optimal": candidate.allocation_mode == "workload_optimal",
        "initial_table_hash_matches_control": bool(
            sha256_file(initial_path) == plan["control"]["initial"]["sha256"]
        ),
    }
    mechanism_gate = {
        "passed": bool(base_gate["passed"] and all(extra_checks.values())),
        "base_gate": base_gate,
        "extra_checks": extra_checks,
        "allocation_certificate": allocation_certificate,
        "control_reproduction": reproduction,
        "noise_coupling": coupling,
    }
    if not mechanism_gate["passed"]:
        raise RuntimeError(f"WP10a mechanism gate failed: {mechanism_gate}")

    seal = {
        "protocol_id": PROTOCOL_ID,
        "candidate_method_id": CANDIDATE_METHOD_ID,
        "artifact_role": "sealed_blind_workload_allocation_candidate",
        "dataset": DATASET,
        "epsilon": EPSILON,
        "delta": DELTA_DP,
        "seed": SEED,
        "measurement_seed": measurement_seed,
        "generation_seed": generation_seed,
        "plan": _file_record(plan_path),
        "control": plan["control"],
        "candidate": {
            "measurement": _file_record(artifact_dir / "measurements.json"),
            "queries": _file_record(artifact_dir / "queries.json"),
            "synthetic": _file_record(synthetic_path),
            "initial": _file_record(initial_path),
            "metrics": _file_record(metrics_path),
            "runtime": _file_record(runtime_path),
            "run_status": _file_record(run_status_path),
            "resolved_config": _file_record(generation_dir / "config_resolved.yaml"),
        },
        "mechanism_gate": mechanism_gate,
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": True,
        "private_input_hashes_excluded": True,
    }
    seal_path = output / "sealed_manifest.json"
    write_json(seal, seal_path)
    write_json(
        {
            "status": "completed",
            "protocol_id": PROTOCOL_ID,
            "seal": _file_record(seal_path),
            "true_utility_evaluated": False,
            "offline_evaluation_authorized": True,
        },
        output / "wp10a_status.json",
    )
    return seal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the blind WP10a workload-L2 allocation candidate.")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--delta", type=float, default=DELTA_DP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stage-iters", type=int, default=STAGE_ITERS)
    parser.add_argument("--max-pair-cells", type=int, default=MAX_PAIR_CELLS)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        seal = run(args)
    except Exception as error:
        output = args.output_dir.resolve()
        ensure_dir(output)
        write_json(
            {
                "status": "failed",
                "protocol_id": PROTOCOL_ID,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "true_utility_evaluated": False,
            },
            output / "wp10a_status.json",
        )
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "protocol_id": seal["protocol_id"],
                "mechanism_gate": seal["mechanism_gate"]["passed"],
                "true_utility_evaluated": seal["true_utility_evaluated"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
