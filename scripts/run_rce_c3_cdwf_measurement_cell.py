#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

# SciPy/HiGHS and JAX otherwise compete for every host core during the CCF
# solves. Candidate scoring still uses the visible JAX GPU.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from qdte.measurement.cdwf import derive_cdwf_budget_plan
from qdte.measurement.cdwf_protocol import (
    CDWF_ARMS,
    run_cdwf_adaptive_measurement,
)
from qdte.measurement.cdwf_transcript import cdwf_transcript_to_measurements
from qdte.measurement.factorization import (
    allocate_strategy_rho,
    compile_hierarchical_pair_strategy,
)
from qdte.measurement.public_artifact import (
    verify_public_transcript,
    write_public_transcript_manifest,
)
from qdte.queries.partitions import build_selected_pair_partition_workload
from qdte.schema import TableSchema
from scripts.path_defaults import external_inputs
from scripts.run_static_ice_measurement_pilot import (
    public_pair_scopes,
    rho_for_epsilon,
)
from scripts.run_static_ice_qdte_pilot import _write_measurement_artifact


PROTOCOL_ID = "SAGE-QDTE-RCE-C3-CDWF-20260718-v1"
METHOD_ID = "SAGE-QDTE-RCE-C3-CDWF-v1"
DATASETS = ("adult", "br2000")
EPSILONS = (0.1, 0.3)
FORMAL_SEEDS = (100, 101, 102, 103, 104)
DELTA_DP = 1.0e-9
MAX_PAIR_CELLS = 20_000


def _epsilon_is_supported(value: float) -> bool:
    return any(
        np.isclose(float(value), declared, rtol=0.0, atol=1.0e-12)
        for declared in EPSILONS
    )


def _read_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _log(stage: str, started: float, **values: Any) -> None:
    payload = {
        "stage": stage,
        "elapsed_seconds": time.perf_counter() - started,
        **values,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def run_measurement_cell(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = str(args.dataset).lower()
    epsilon = float(args.epsilon)
    formal = bool(args.formal)
    if dataset not in DATASETS or not _epsilon_is_supported(epsilon):
        raise ValueError("C3-CDWF supports Adult/BR2000 at epsilon 0.1/0.3")
    if formal and int(args.seed) not in FORMAL_SEEDS:
        raise ValueError(f"Formal C3-CDWF seed must be one of {FORMAL_SEEDS}")
    if formal and args.sealed_plan is None:
        raise ValueError("Formal C3-CDWF measurement requires --sealed-plan")

    input_dir = (
        args.input_dir.resolve()
        if args.input_dir is not None
        else (external_inputs() / f"{dataset}_sage_strong").resolve()
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"C3-CDWF output must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = TableSchema.load_json(input_dir / "schema.json")
    rows = np.load(input_dir / "real_encoded.npy", allow_pickle=False).astype(
        np.int32
    )
    metadata = _read_metadata(input_dir / "metadata.json")
    public_total = int(metadata["n_rows"])
    if rows.shape != (public_total, schema.d):
        raise ValueError("Private C3 input does not match the public schema/row count")
    pairs = public_pair_scopes(schema.cardinalities, MAX_PAIR_CELLS)
    strategy = compile_hierarchical_pair_strategy(schema.cardinalities, pairs)
    qcat, groups = build_selected_pair_partition_workload(schema, pairs)
    rho_total = rho_for_epsilon(epsilon, DELTA_DP)
    control = allocate_strategy_rho(strategy, rho_total, "public_optimal")
    plan = derive_cdwf_budget_plan(strategy, control, epsilon=epsilon)
    _log(
        "loaded_private_measurement_input",
        started,
        dataset=dataset,
        rows=public_total,
        blocks=len(strategy.blocks),
        pairs=len(pairs),
        queries=qcat.m,
        coefficient_dimension=sum(block.dimension for block in strategy.blocks),
    )

    run = run_cdwf_adaptive_measurement(
        rows,
        schema,
        qcat,
        groups,
        strategy,
        plan,
        public_total=public_total,
        arm=str(args.arm),
        base_noise_seed=int(args.base_noise_seed),
        refinement_noise_seed=int(args.refinement_noise_seed),
        generation_seed=int(args.generation_seed),
        progress_callback=lambda stage, values: _log(
            stage,
            started,
            **values,
        ),
    )
    _log(
        "adaptive_measurement_complete",
        started,
        dual_fallback_count=run.dual_fallback_count,
        promotion_eligible_dual=run.promotion_eligible_dual,
    )

    transcript_dir = output_dir / "public_transcript"
    measurements = cdwf_transcript_to_measurements(
        run.transcript,
        qcat,
        list(groups),
        delta=DELTA_DP,
    )
    _write_measurement_artifact(
        transcript_dir,
        qcat=qcat,
        schema=schema,
        measurements=measurements,
    )
    write_public_transcript_manifest(transcript_dir)
    verified = verify_public_transcript(transcript_dir)
    runtime = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": (
            "formal_private_measurement_cell"
            if formal
            else "development_private_measurement_smoke"
        ),
        "dataset": dataset,
        "epsilon": epsilon,
        "delta": DELTA_DP,
        "seed": int(args.seed),
        "arm": str(args.arm),
        "base_noise_seed": int(args.base_noise_seed),
        "refinement_noise_seed": int(args.refinement_noise_seed),
        "generation_seed": int(args.generation_seed),
        "formal": formal,
        "sealed_plan": (
            str(args.sealed_plan.resolve()) if args.sealed_plan is not None else None
        ),
        "truth_utility_evaluated": False,
        "allocation_truth_access": False,
        "selection_rho": 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024.0 * 1024.0),
        "public_transcript_manifest": verified.manifest,
        "measurement_run": run.to_public_dict(),
    }
    write_json(runtime, output_dir / "measurement_run.json")
    _log(
        "public_transcript_sealed",
        started,
        max_rss_gib=runtime["max_rss_gib"],
        transcript=str(transcript_dir),
    )
    return runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Release and seal one C3-CDWF public measurement transcript."
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--epsilon", type=float, choices=EPSILONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=CDWF_ARMS, required=True)
    parser.add_argument("--base-noise-seed", type=int, required=True)
    parser.add_argument("--refinement-noise-seed", type=int, required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--sealed-plan", type=Path)
    return parser.parse_args()


def main() -> None:
    run_measurement_cell(parse_args())


if __name__ == "__main__":
    main()
