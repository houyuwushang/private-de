#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from dataclasses import dataclass
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
    assemble_adaptive_interaction_transcript,
    measure_interaction_action,
)
from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.interaction_adapter import interaction_transcript_to_diagonal_measurements
from qdte.privacy.accountant import (
    ZCDPPrivacyFilter,
    bounded_range_epsilon,
    bounded_range_rho,
)
from qdte.privacy.exponential import exponential_mechanism_probabilities
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import TableSchema
from qdte.selection.voi import (
    orthogonal_interaction_score,
    partition_l1_score,
    public_pair_action_order,
)
from scripts.run_static_ice_measurement_pilot import public_pair_scopes, rho_for_epsilon
from scripts.run_static_ice_qdte_pilot import _generation_config, _write_measurement_artifact


PROTOCOL_ID = "SAGE-QDTE-ICE-WP7-OI-SELECTOR-20260714-v1"
PROTOCOL_PATH = ROOT / "docs" / "SAGE_QDTE_ICE_WP7_SELECTOR_PROTOCOL_20260714.md"
ARMS = (
    "public_static_R",
    "private_partition_l1_R",
    "private_orthogonal_interaction_R",
)
ANCHOR_RHO_FRACTION = 0.10
SELECTION_RHO_FRACTION = 0.27
ACTION_MEASUREMENT_RHO_FRACTION = 0.63
SMOKE_STAGE_ITERS = 20


@dataclass(frozen=True)
class SelectorBudget:
    rho_total: float
    rounds: int
    anchor_rho: float
    selection_rho_total: float
    selection_rho_per_round: float
    selection_epsilon_per_round: float
    action_rho_total: float
    action_rho_per_round: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho_total": self.rho_total,
            "rounds": self.rounds,
            "anchor_rho": self.anchor_rho,
            "selection_rho_total": self.selection_rho_total,
            "selection_rho_per_round": self.selection_rho_per_round,
            "selection_epsilon_per_round": self.selection_epsilon_per_round,
            "selection_rho_roundtrip": bounded_range_rho(
                self.selection_epsilon_per_round
            ),
            "action_rho_total": self.action_rho_total,
            "action_rho_per_round": self.action_rho_per_round,
        }


def rounds_for_epsilon(epsilon: float) -> int:
    value = float(epsilon)
    for declared, rounds in ((0.1, 4), (0.3, 6), (1.0, 8), (3.0, 12), (10.0, 12)):
        if math.isclose(value, declared, rel_tol=0.0, abs_tol=1.0e-12):
            return rounds
    raise ValueError("epsilon must be one of {0.1, 0.3, 1, 3, 10}")


def selector_budget(epsilon: float, delta: float) -> SelectorBudget:
    rho_total = rho_for_epsilon(float(epsilon), float(delta))
    rounds = rounds_for_epsilon(float(epsilon))
    selection_total = SELECTION_RHO_FRACTION * rho_total
    selection_round = selection_total / float(rounds)
    action_total = ACTION_MEASUREMENT_RHO_FRACTION * rho_total
    return SelectorBudget(
        rho_total=rho_total,
        rounds=rounds,
        anchor_rho=ANCHOR_RHO_FRACTION * rho_total,
        selection_rho_total=selection_total,
        selection_rho_per_round=selection_round,
        selection_epsilon_per_round=bounded_range_epsilon(selection_round),
        action_rho_total=action_total,
        action_rho_per_round=action_total / float(rounds),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_seed(seed: int, stage: int) -> int:
    if int(stage) < 0:
        raise ValueError("stage must be non-negative")
    return int(
        np.random.SeedSequence(
            [int(seed) & 0xFFFFFFFF, 0x575037, int(stage) & 0xFFFFFFFF]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _action_rng(seed: int, pair: tuple[int, int]) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [int(seed) & 0xFFFFFFFF, 0xAC710, int(pair[0]), int(pair[1])]
        )
    )


def _selection_rng(seed: int, arm: str, round_index: int) -> np.random.Generator:
    arm_id = ARMS.index(arm)
    return np.random.default_rng(
        np.random.SeedSequence(
            [int(seed) & 0xFFFFFFFF, 0x5E1EC7, arm_id, int(round_index)]
        )
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
    if protocol_mode not in {"smoke", "formal"}:
        raise ValueError("protocol_mode must be smoke or formal")
    rounds_for_epsilon(float(epsilon))
    if not math.isclose(float(delta), 1.0e-9, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("WP7 fixes delta to 1e-9")
    if int(stage_iters) <= 0:
        raise ValueError("stage_iters must be positive")
    if protocol_mode == "smoke":
        if dataset != "acs_sage_strong":
            raise ValueError("the frozen WP7 smoke requires ACS")
        if not math.isclose(float(epsilon), 0.1, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("the frozen WP7 smoke requires epsilon=0.1")
        if int(seed) != 0 or int(stage_iters) != SMOKE_STAGE_ITERS:
            raise ValueError("the frozen WP7 smoke requires seed0 and 20 stage iterations")
    elif int(stage_iters) != 5000:
        raise ValueError("formal WP7 runs require 5000 QDTE iterations per stage")


def _write_transcript_artifact(
    artifact_dir: Path,
    *,
    transcript: HierarchicalInteractionTranscript,
    schema: TableSchema,
    delta: float,
    selected_actions: list[tuple[int, int]],
    arm: str,
    round_index: int,
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
    diagnostics["adaptive_selector"] = {
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "round": int(round_index),
        "selected_actions": [list(pair) for pair in selected_actions],
        "partial_support_formulation": True,
        "fake_unmeasured_variance": False,
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
    budget: SelectorBudget,
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
        rho_total=float(budget.rho_total),
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
    set_nested(config, "method.artifact_role", "released_transcript_postprocessing")
    if init_path is not None:
        set_nested(config, "init.encoded_npy", str(init_path))
    metrics = run_qdte(config)
    synthetic_path = output_dir / "synthetic_encoded.npy"
    if not synthetic_path.is_file():
        raise RuntimeError(f"generator did not write {synthetic_path}")
    forbidden = [key for key in metrics if key.startswith("final_true_") or key.startswith("true_query_")]
    if forbidden:
        raise RuntimeError(f"generator emitted forbidden true metrics: {forbidden}")
    return synthetic_path, metrics


def _score_actions(
    *,
    arm: str,
    private_rows: np.ndarray,
    synthetic_rows: np.ndarray,
    available: list[tuple[int, int]],
    cardinalities: np.ndarray,
    action_rho: float,
) -> np.ndarray:
    if arm == "private_partition_l1_R":
        return np.asarray(
            [
                partition_l1_score(private_rows, synthetic_rows, pair, cardinalities)
                for pair in available
            ],
            dtype=np.float64,
        )
    if arm == "private_orthogonal_interaction_R":
        return np.asarray(
            [
                orthogonal_interaction_score(
                    private_rows,
                    synthetic_rows,
                    pair,
                    cardinalities,
                    measurement_rho=float(action_rho),
                    noise_floor_multiplier=1.0,
                )
                for pair in available
            ],
            dtype=np.float64,
        )
    raise ValueError(f"arm {arm!r} has no private score")


def _private_selection_diagnostic(
    *,
    arm: str,
    round_index: int,
    available: list[tuple[int, int]],
    scores: np.ndarray,
    probabilities: np.ndarray,
    chosen_index: int,
    epsilon: float,
) -> dict[str, Any]:
    order = np.argsort(-scores, kind="stable")
    rank = int(np.flatnonzero(order == int(chosen_index))[0]) + 1
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive), dtype=np.float64))
    return {
        "artifact_role": "non_releasable_private_selector_diagnostic",
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "round": int(round_index),
        "available_actions": [list(pair) for pair in available],
        "scores": scores.tolist(),
        "probabilities": probabilities.tolist(),
        "selected_index": int(chosen_index),
        "selected_action": list(available[int(chosen_index)]),
        "selected_private_rank": rank,
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "em_exponent_gap": float(epsilon * (np.max(scores) - np.min(scores)) / 2.0),
        "selection_entropy": entropy,
        "selection_entropy_fraction": float(entropy / math.log(len(scores)))
        if len(scores) > 1
        else 0.0,
    }


def _run_arm(
    *,
    arm: str,
    base_config: dict[str, Any],
    input_dir: Path,
    output_root: Path,
    schema: TableSchema,
    private_rows: np.ndarray,
    anchors: HierarchicalInteractionTranscript,
    initial_synthetic_path: Path,
    candidate_pairs: tuple[tuple[int, int], ...],
    public_order: tuple[tuple[int, int], ...],
    dataset: str,
    epsilon: float,
    delta: float,
    seed: int,
    stage_iters: int,
    budget: SelectorBudget,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown WP7 arm {arm!r}")
    arm_root = ensure_dir(output_root / "arms" / arm)
    offline_root = ensure_dir(output_root / "offline_private_diagnostics" / arm)
    ledger = ZCDPPrivacyFilter(budget.rho_total)
    ledger.spend(
        label="oneway_anchors",
        mechanism="gaussian_orthogonal_vector",
        rho=budget.anchor_rho,
        public_metadata={"num_blocks": schema.d, "adjacency": "add_remove"},
    )
    selected: list[tuple[int, int]] = []
    releases: list[InteractionActionRelease] = []
    current_synthetic_path = initial_synthetic_path
    round_rows: list[dict[str, Any]] = []
    for round_index in range(budget.rounds):
        available = [pair for pair in candidate_pairs if pair not in set(selected)]
        if not available:
            raise RuntimeError("adaptive selector exhausted its distinct action support")
        if arm == "public_static_R":
            pair = next(pair for pair in public_order if pair in available)
            selection_mechanism = "public_deterministic"
        else:
            synthetic_rows = np.load(
                current_synthetic_path,
                allow_pickle=False,
            ).astype(np.int32)
            scores = _score_actions(
                arm=arm,
                private_rows=private_rows,
                synthetic_rows=synthetic_rows,
                available=available,
                cardinalities=schema.cardinalities,
                action_rho=budget.action_rho_per_round,
            )
            probabilities = exponential_mechanism_probabilities(
                scores,
                budget.selection_epsilon_per_round,
                1.0,
            )
            selection_rng = _selection_rng(seed, arm, round_index)
            chosen_index = int(selection_rng.choice(len(available), p=probabilities))
            pair = available[chosen_index]
            selection_mechanism = "bounded_range_exponential"
            diagnostic = _private_selection_diagnostic(
                arm=arm,
                round_index=round_index,
                available=available,
                scores=scores,
                probabilities=probabilities,
                chosen_index=chosen_index,
                epsilon=budget.selection_epsilon_per_round,
            )
            write_json(diagnostic, offline_root / f"round_{round_index:02d}.json")
            ledger.spend(
                label=f"selection:{round_index}",
                mechanism="bounded_range_exponential",
                rho=budget.selection_rho_per_round,
                public_metadata={
                    "epsilon": budget.selection_epsilon_per_round,
                    "sensitivity": 1.0,
                    "support_size": len(available),
                    "score": "partition_l1" if arm == "private_partition_l1_R" else "orthogonal_interaction",
                },
            )

        release = measure_interaction_action(
            private_rows,
            pair,
            schema.cardinalities,
            rho=budget.action_rho_per_round,
            rng=_action_rng(seed, pair),
        )
        selected.append(pair)
        releases.append(release)
        ledger.spend(
            label=f"measurement:{round_index}",
            mechanism="gaussian_orthogonal_interaction",
            rho=budget.action_rho_per_round,
            public_metadata={
                "pair": list(pair),
                "sensitivity_l2": release.sensitivity_l2,
                "coefficient_dimension": int(release.noisy_coefficients.size),
            },
        )
        transcript = assemble_adaptive_interaction_transcript(
            anchors,
            releases,
            rho_total=budget.rho_total,
        )
        artifact_dir = arm_root / f"round_{round_index:02d}_measurement"
        _write_transcript_artifact(
            artifact_dir,
            transcript=transcript,
            schema=schema,
            delta=delta,
            selected_actions=selected,
            arm=arm,
            round_index=round_index,
        )
        generation_dir = arm_root / f"round_{round_index:02d}_generate"
        current_synthetic_path, metrics = _run_generation_stage(
            base_config=base_config,
            input_dir=input_dir,
            artifact_dir=artifact_dir,
            output_dir=generation_dir,
            init_path=current_synthetic_path,
            dataset=dataset,
            epsilon=epsilon,
            budget=budget,
            delta=delta,
            seed=_stage_seed(seed, round_index + 1),
            stage_iters=stage_iters,
        )
        round_rows.append(
            {
                "round": round_index,
                "selected_action": list(pair),
                "selection_mechanism": selection_mechanism,
                "action_rho": budget.action_rho_per_round,
                "gaussian_rho_spent": transcript.rho_spent,
                "total_rho_spent": ledger.rho_spent,
                "num_measured_pairs": len(selected),
                "coefficient_dimension": sum(
                    int(block.dimension) for block in transcript.strategy.blocks
                ),
                "final_optimization_objective": metrics.get("final_optimization_objective"),
                "num_accepted_edits": metrics.get("num_accepted_edits"),
                "generator_dir": str(generation_dir),
            }
        )

    expected_spent = budget.anchor_rho + budget.action_rho_total
    if arm != "public_static_R":
        expected_spent += budget.selection_rho_total
    if not np.isclose(ledger.rho_spent, expected_spent, rtol=1.0e-11, atol=1.0e-15):
        raise RuntimeError("arm ledger does not match its frozen budget")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "deployable_dp_transcript_and_postprocessing",
        "arm": arm,
        "selection_is_private": arm != "public_static_R",
        "selected_actions": [list(pair) for pair in selected],
        "rounds": round_rows,
        "privacy": ledger.to_public_dict(delta=delta),
        "final_synthetic_path": str(current_synthetic_path),
        "offline_private_diagnostics_excluded": True,
    }
    write_json(manifest, arm_root / "arm_manifest.json")
    return manifest


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_root}")
    ensure_dir(output_root)
    schema = TableSchema.load_json(input_dir / "schema.json")
    rows = np.load(input_dir / "real_encoded.npy", allow_pickle=False).astype(np.int32)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset = str(metadata.get("dataset", input_dir.name))
    public_total = int(metadata["n_rows"])
    if rows.shape != (public_total, schema.d):
        raise ValueError("encoded rows do not match public schema/n")
    validate_request(
        protocol_mode=args.protocol_mode,
        dataset=dataset,
        epsilon=float(args.epsilon),
        delta=float(args.delta),
        seed=int(args.seed),
        stage_iters=int(args.stage_iters),
    )
    budget = selector_budget(float(args.epsilon), float(args.delta))
    candidate_pairs = public_pair_scopes(
        schema.cardinalities,
        int(args.max_pair_cells),
    )
    if len(candidate_pairs) < budget.rounds:
        raise ValueError("public pair support is smaller than the frozen round count")
    public_order = public_pair_action_order(candidate_pairs, schema.cardinalities)
    base_config = load_yaml(args.config)

    anchors = measure_hierarchical_pair_interactions(
        rows,
        compile_hierarchical_pair_strategy(schema.cardinalities, ()),
        public_total=public_total,
        rho_total=budget.anchor_rho,
        rng=np.random.default_rng(
            np.random.SeedSequence([int(args.seed), 0xA11C0])
        ),
        allocation_mode="public_optimal",
    )
    anchor_transcript = assemble_adaptive_interaction_transcript(
        anchors,
        (),
        rho_total=budget.rho_total,
    )
    anchor_artifact = output_root / "shared" / "oneway_anchor_measurement"
    _write_transcript_artifact(
        anchor_artifact,
        transcript=anchor_transcript,
        schema=schema,
        delta=float(args.delta),
        selected_actions=[],
        arm="shared_oneway_anchor",
        round_index=-1,
    )
    initial_dir = output_root / "shared" / "oneway_anchor_generate"
    initial_synthetic_path, initial_metrics = _run_generation_stage(
        base_config=base_config,
        input_dir=input_dir,
        artifact_dir=anchor_artifact,
        output_dir=initial_dir,
        init_path=None,
        dataset=dataset,
        epsilon=float(args.epsilon),
        budget=budget,
        delta=float(args.delta),
        seed=_stage_seed(int(args.seed), 0),
        stage_iters=int(args.stage_iters),
    )

    manifests: dict[str, Any] = {}
    for arm in ARMS:
        manifests[arm] = _run_arm(
            arm=arm,
            base_config=base_config,
            input_dir=input_dir,
            output_root=output_root,
            schema=schema,
            private_rows=rows,
            anchors=anchors,
            initial_synthetic_path=initial_synthetic_path,
            candidate_pairs=candidate_pairs,
            public_order=public_order,
            dataset=dataset,
            epsilon=float(args.epsilon),
            delta=float(args.delta),
            seed=int(args.seed),
            stage_iters=int(args.stage_iters),
            budget=budget,
        )

    offline_root = ensure_dir(output_root / "offline_private_diagnostics")
    write_json(
        {
            "artifact_role": "non_releasable_private_input_provenance",
            "protocol_id": PROTOCOL_ID,
            "input_hashes": {
                "metadata": sha256_file(input_dir / "metadata.json"),
                "real_encoded": sha256_file(input_dir / "real_encoded.npy"),
                "raw_csv": sha256_file(input_dir / "raw.csv"),
            },
        },
        offline_root / "private_input_provenance.json",
    )

    summary = {
        "protocol_id": PROTOCOL_ID,
        "protocol_mode": args.protocol_mode,
        "artifact_role": "mechanism_smoke_not_utility_evidence"
        if args.protocol_mode == "smoke"
        else "blind_formal_generation_before_offline_evaluation",
        "dataset": dataset,
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "seed": int(args.seed),
        "adjacency": "add_remove",
        "public_total": public_total,
        "stage_iters": int(args.stage_iters),
        "budget": budget.to_dict(),
        "num_candidate_pairs": len(candidate_pairs),
        "public_action_order": [list(pair) for pair in public_order],
        "shared_anchor": {
            "artifact": str(anchor_artifact),
            "synthetic": str(initial_synthetic_path),
            "final_optimization_objective": initial_metrics.get(
                "final_optimization_objective"
            ),
        },
        "arms": manifests,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "public_input_hashes": {
            "schema": sha256_file(input_dir / "schema.json"),
        },
        "private_input_hashes_excluded": True,
        "true_utility_evaluated": False,
    }
    write_json(summary, output_root / "summary.json")
    write_json({"status": "completed", "protocol_id": PROTOCOL_ID}, output_root / "run_status.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen WP7 SAGE-VOI selector pilot.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol-mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage-iters", type=int, default=SMOKE_STAGE_ITERS)
    parser.add_argument("--max-pair-cells", type=int, default=20_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_pilot(args)
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
    print(json.dumps({"status": "completed", "output": str(args.output_dir), "arms": list(summary["arms"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
