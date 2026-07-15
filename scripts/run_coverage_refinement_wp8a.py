#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from qdte.measurement.adaptive_interactions import (
    InteractionActionRelease,
    combine_coverage_refinements,
    measure_interaction_action,
)
from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    allocate_strategy_rho,
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import (
    interaction_transcript_to_diagonal_measurements,
)
from qdte.measurement.precision_refinement import (
    CoverageRefinementBudget,
    coverage_rounds_for_epsilon,
    derive_coverage_refinement_budget,
)
from qdte.privacy.accountant import (
    ZCDPPrivacyFilter,
    bounded_range_epsilon,
)
from qdte.privacy.exponential import exponential_mechanism_probabilities
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import TableSchema
from qdte.selection.voi import partition_l1_score
from scripts.run_static_ice_measurement_pilot import public_pair_scopes, rho_for_epsilon
from scripts.run_static_ice_qdte_pilot import (
    _generation_config,
    _write_measurement_artifact,
)


PROTOCOL_ID = "SAGE-QDTE-ICE-WP8A-COVERAGE-REFINEMENT-20260714-v1"
PROTOCOL_PATH = (
    ROOT / "docs" / "SAGE_QDTE_ICE_WP8A_COVERAGE_REFINEMENT_PROTOCOL_20260714.md"
)
SMOKE_STAGE_ITERS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int, domain: int, *values: int) -> int:
    return int(
        np.random.SeedSequence(
            [int(seed) & 0xFFFFFFFF, int(domain), *(int(value) for value in values)]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def validate_request(
    *,
    protocol_mode: str,
    dataset: str,
    epsilon: float,
    delta: float,
    seed: int,
    stage_iters: int,
) -> None:
    if protocol_mode != "smoke":
        raise ValueError("WP8a v1 currently authorizes mechanism smoke only")
    if dataset != "acs_sage_strong":
        raise ValueError("the frozen WP8a smoke requires ACS")
    if not math.isclose(float(epsilon), 0.1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("the frozen WP8a smoke requires epsilon=0.1")
    if not math.isclose(float(delta), 1.0e-9, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the frozen WP8a smoke requires delta=1e-9")
    if int(seed) != 0 or int(stage_iters) != SMOKE_STAGE_ITERS:
        raise ValueError("the frozen WP8a smoke requires seed0 and 20 stage iterations")


def _write_transcript_artifact(
    artifact_dir: Path,
    *,
    transcript: HierarchicalInteractionTranscript,
    schema: TableSchema,
    delta: float,
    round_index: int,
    selected_actions: list[tuple[int, int]],
    full_privacy_ledger: dict[str, Any],
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
    diagnostics["coverage_precision_refinement"] = {
        "protocol_id": PROTOCOL_ID,
        "round": int(round_index),
        "selected_actions": [list(pair) for pair in selected_actions],
        "broad_strategy_preserved": True,
        "full_privacy_ledger": full_privacy_ledger,
        "true_utility_evaluated": False,
    }
    measurements.projection_diagnostics = diagnostics
    _write_measurement_artifact(
        artifact_dir,
        qcat=qcat,
        schema=schema,
        measurements=measurements,
    )


def _run_generation_stage(
    *,
    base_config: dict[str, Any],
    input_dir: Path,
    artifact_dir: Path,
    output_dir: Path,
    init_path: Path | None,
    dataset: str,
    epsilon: float,
    rho_total: float,
    delta: float,
    seed: int,
    stage_iters: int,
) -> tuple[Path, dict[str, Any]]:
    config = _generation_config(
        base_config,
        input_csv=input_dir / "raw.csv",
        public_schema=input_dir / "schema.json",
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        dataset_name=dataset,
        epsilon=float(epsilon),
        rho_total=float(rho_total),
        delta=float(delta),
        seed=int(seed),
        max_iters=int(stage_iters),
        precision_operator="orthogonal_interaction",
        exact_scoring_chunk_size=512,
    )
    set_nested(config, "qdte.stop_patience", int(stage_iters))
    set_nested(config, "evaluation.compute_true_query_error", False)
    set_nested(config, "evaluation.compute_heldout_query_error", False)
    set_nested(config, "evaluation.save_synthetic_csv", False)
    set_nested(config, "method.protocol_id", PROTOCOL_ID)
    set_nested(config, "method.artifact_role", "dp_transcript_postprocessing_smoke")
    if init_path is not None:
        set_nested(config, "init.encoded_npy", str(init_path))
    metrics = run_qdte(config)
    forbidden = [
        key
        for key in metrics
        if key.startswith("final_true_")
        or key.startswith("true_query_")
        or key.startswith("offline_")
    ]
    if forbidden:
        raise RuntimeError(f"generator emitted forbidden true metrics: {forbidden}")
    synthetic_path = output_dir / "synthetic_encoded.npy"
    if not synthetic_path.is_file():
        raise RuntimeError(f"generator did not write {synthetic_path}")
    return synthetic_path, metrics


def _mechanism_gate(
    *,
    base: HierarchicalInteractionTranscript,
    final: HierarchicalInteractionTranscript,
    full_static_rho: dict[str, float],
    budget: CoverageRefinementBudget,
    ledger: ZCDPPrivacyFilter,
    refinements: list[InteractionActionRelease],
) -> dict[str, Any]:
    expected_names = {block.name for block in base.strategy.blocks}
    full_names = set(full_static_rho)
    final_names = set(final.rho_by_block)
    all_blocks_preserved = expected_names == full_names == final_names
    all_positive = all(
        final.rho_by_block[name] > 0.0
        and final.component_variances[name] > 0.0
        for name in expected_names
    )
    std_inflation = {
        name: math.sqrt(float(full_static_rho[name]) / float(base.rho_by_block[name]))
        for name in expected_names
    }
    max_std_inflation = max(std_inflation.values())
    effective_rho_matches = all(
        np.isclose(
            final.rho_by_block[name],
            base.rho_by_block[name]
            + math.fsum(
                action.rho
                for action in refinements
                if action.name == name
            ),
            rtol=1.0e-11,
            atol=1.0e-15,
        )
        for name in expected_names
    )
    gaussian_spend_matches = np.isclose(
        final.rho_spent,
        budget.rho_base + budget.rho_refinement,
        rtol=1.0e-11,
        atol=1.0e-15,
    )
    total_spend_matches = np.isclose(
        ledger.rho_spent,
        budget.rho_total,
        rtol=1.0e-11,
        atol=1.0e-15,
    )
    checks = {
        "all_strategy_blocks_preserved": bool(all_blocks_preserved),
        "all_final_precisions_positive": bool(all_positive),
        "coverage_std_inflation_within_cap": bool(
            max_std_inflation
            <= budget.coverage_std_inflation_cap + 1.0e-11
        ),
        "effective_block_rho_matches_observations": bool(effective_rho_matches),
        "gaussian_spend_matches_budget": bool(gaussian_spend_matches),
        "total_spend_matches_budget": bool(total_spend_matches),
        "privacy_filter_within_limit": bool(
            ledger.rho_spent <= ledger.rho_limit + 1.0e-15
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_coverage_std_inflation": max_std_inflation,
        "coverage_std_inflation_cap": budget.coverage_std_inflation_cap,
        "num_strategy_blocks": len(expected_names),
        "num_pair_scopes": len(base.strategy.pairs),
        "num_refinement_observations": len(refinements),
        "num_distinct_refined_pairs": len({action.pair for action in refinements}),
        "gaussian_rho_spent": final.rho_spent,
        "total_rho_spent": ledger.rho_spent,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    ensure_dir(output_dir)

    schema = TableSchema.load_json(input_dir / "schema.json")
    private_rows = np.load(input_dir / "real_encoded.npy", allow_pickle=False).astype(
        np.int32
    )
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    public_total = int(metadata["n_rows"])
    dataset = str(metadata.get("dataset", input_dir.name))
    if private_rows.shape != (public_total, schema.d):
        raise ValueError("encoded rows do not match public schema/n")
    validate_request(
        protocol_mode=args.protocol_mode,
        dataset=dataset,
        epsilon=float(args.epsilon),
        delta=float(args.delta),
        seed=int(args.seed),
        stage_iters=int(args.stage_iters),
    )

    pairs = public_pair_scopes(schema.cardinalities, int(args.max_pair_cells))
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, pairs)
    rounds = coverage_rounds_for_epsilon(float(args.epsilon))
    rho_total = rho_for_epsilon(float(args.epsilon), float(args.delta))
    budget = derive_coverage_refinement_budget(
        strategy,
        rho_total=rho_total,
        rounds=rounds,
    )
    full_static_rho = allocate_strategy_rho(strategy, rho_total, "public_optimal")
    base = measure_hierarchical_pair_interactions(
        private_rows,
        strategy,
        public_total=public_total,
        rho_total=budget.rho_base,
        rng=np.random.default_rng(_seed(int(args.seed), 0xB453)),
        allocation_mode="public_optimal",
    )
    ledger = ZCDPPrivacyFilter(rho_total)
    ledger.spend(
        label="broad_static_ice_base",
        mechanism="gaussian_hierarchical_orthogonal_vectors",
        rho=budget.rho_base,
        public_metadata={
            "num_blocks": len(strategy.blocks),
            "num_pairs": len(strategy.pairs),
            "adjacency": "add_remove",
        },
    )

    base_artifact = output_dir / "shared" / "base_measurement"
    _write_transcript_artifact(
        base_artifact,
        transcript=base,
        schema=schema,
        delta=float(args.delta),
        round_index=-1,
        selected_actions=[],
        full_privacy_ledger=ledger.to_public_dict(delta=float(args.delta)),
    )
    base_config = load_yaml(args.config)
    current_synthetic, base_metrics = _run_generation_stage(
        base_config=base_config,
        input_dir=input_dir,
        artifact_dir=base_artifact,
        output_dir=output_dir / "shared" / "base_generate",
        init_path=None,
        dataset=dataset,
        epsilon=float(args.epsilon),
        rho_total=rho_total,
        delta=float(args.delta),
        seed=_seed(int(args.seed), 0x57A63, 0),
        stage_iters=int(args.stage_iters),
    )

    selection_epsilon = bounded_range_epsilon(
        budget.rho_selection_per_round
    )
    refinements: list[InteractionActionRelease] = []
    selected_actions: list[tuple[int, int]] = []
    round_records: list[dict[str, Any]] = []
    final_transcript = combine_coverage_refinements(
        base,
        refinements,
        rho_total=rho_total,
    )
    for round_index in range(rounds):
        synthetic_rows = np.load(current_synthetic, allow_pickle=False).astype(
            np.int32
        )
        scores = np.asarray(
            [
                partition_l1_score(
                    private_rows,
                    synthetic_rows,
                    pair,
                    schema.cardinalities,
                )
                for pair in pairs
            ],
            dtype=np.float64,
        )
        probabilities = exponential_mechanism_probabilities(
            scores,
            selection_epsilon,
            1.0,
        )
        selection_rng = np.random.default_rng(
            _seed(int(args.seed), 0x5E1EC7, round_index)
        )
        pair = pairs[int(selection_rng.choice(len(pairs), p=probabilities))]
        ledger.spend(
            label=f"partition_l1_selection:{round_index}",
            mechanism="bounded_range_exponential_mechanism",
            rho=budget.rho_selection_per_round,
            public_metadata={
                "round": round_index,
                "support_size": len(pairs),
                "sensitivity": 1.0,
                "epsilon": selection_epsilon,
            },
        )
        action = measure_interaction_action(
            private_rows,
            pair,
            schema.cardinalities,
            rho=budget.rho_refinement_per_round,
            rng=np.random.default_rng(
                _seed(int(args.seed), 0xAC710, round_index, *pair)
            ),
        )
        ledger.spend(
            label=f"interaction_refinement:{round_index}:{pair[0]}:{pair[1]}",
            mechanism="gaussian_pure_interaction_vector",
            rho=budget.rho_refinement_per_round,
            public_metadata={
                "round": round_index,
                "pair": list(pair),
                "sensitivity_l2": action.sensitivity_l2,
            },
        )
        refinements.append(action)
        selected_actions.append(pair)
        final_transcript = combine_coverage_refinements(
            base,
            refinements,
            rho_total=rho_total,
        )
        artifact_dir = output_dir / "private_l1_refine" / f"round_{round_index:02d}" / "measurement"
        _write_transcript_artifact(
            artifact_dir,
            transcript=final_transcript,
            schema=schema,
            delta=float(args.delta),
            round_index=round_index,
            selected_actions=selected_actions,
            full_privacy_ledger=ledger.to_public_dict(delta=float(args.delta)),
        )
        generation_dir = output_dir / "private_l1_refine" / f"round_{round_index:02d}" / "generate"
        current_synthetic, metrics = _run_generation_stage(
            base_config=base_config,
            input_dir=input_dir,
            artifact_dir=artifact_dir,
            output_dir=generation_dir,
            init_path=current_synthetic,
            dataset=dataset,
            epsilon=float(args.epsilon),
            rho_total=rho_total,
            delta=float(args.delta),
            seed=_seed(int(args.seed), 0x57A63, round_index + 1),
            stage_iters=int(args.stage_iters),
        )
        round_records.append(
            {
                "round": round_index,
                "selected_action": list(pair),
                "support_size": len(pairs),
                "selection_epsilon": selection_epsilon,
                "selection_rho": budget.rho_selection_per_round,
                "refinement_rho": budget.rho_refinement_per_round,
                "gaussian_rho_spent": final_transcript.rho_spent,
                "total_rho_spent": ledger.rho_spent,
                "num_times_selected": selected_actions.count(pair),
                "final_optimization_objective": metrics.get(
                    "final_optimization_objective"
                ),
            }
        )

    gate = _mechanism_gate(
        base=base,
        final=final_transcript,
        full_static_rho=full_static_rho,
        budget=budget,
        ledger=ledger,
        refinements=refinements,
    )
    if not gate["passed"]:
        raise RuntimeError(f"WP8a mechanism gate failed: {gate['checks']}")

    final_manifest = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "deployable_dp_transcript_and_postprocessing_smoke",
        "selected_actions": [list(pair) for pair in selected_actions],
        "final_synthetic_path": str(current_synthetic),
        "final_synthetic_sha256": sha256_file(current_synthetic),
        "privacy": ledger.to_public_dict(delta=float(args.delta)),
        "gaussian_transcript_rho_spent": final_transcript.rho_spent,
        "true_utility_evaluated": False,
        "private_input_hashes_excluded": True,
    }
    write_json(final_manifest, output_dir / "arm_manifest.json")
    write_json(gate, output_dir / "mechanism_gate.json")
    summary = {
        "protocol_id": PROTOCOL_ID,
        "protocol_mode": args.protocol_mode,
        "artifact_role": "mechanism_smoke_not_utility_evidence",
        "dataset": dataset,
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "seed": int(args.seed),
        "adjacency": "add_remove",
        "public_total": public_total,
        "stage_iters": int(args.stage_iters),
        "budget": budget.to_public_dict(),
        "num_strategy_blocks": len(strategy.blocks),
        "num_pair_scopes": len(strategy.pairs),
        "selected_actions": [list(pair) for pair in selected_actions],
        "rounds": round_records,
        "base_final_optimization_objective": base_metrics.get(
            "final_optimization_objective"
        ),
        "mechanism_gate": gate,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "public_input_hashes": {
            "schema": sha256_file(input_dir / "schema.json"),
        },
        "private_input_hashes_excluded": True,
        "true_utility_evaluated": False,
    }
    write_json(summary, output_dir / "summary.json")
    write_json(
        {"status": "completed", "protocol_id": PROTOCOL_ID},
        output_dir / "run_status.json",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen WP8a coverage-refinement mechanism smoke."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol-mode", choices=["smoke"], default="smoke")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage-iters", type=int, default=SMOKE_STAGE_ITERS)
    parser.add_argument("--max-pair-cells", type=int, default=20_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_smoke(args)
    except Exception as error:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            {
                "status": "failed",
                "protocol_id": PROTOCOL_ID,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
            output / "run_status.json",
        )
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output_dir),
                "mechanism_gate": summary["mechanism_gate"]["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
