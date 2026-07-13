#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, save_yaml, set_nested
from qdte.config_validation import validate_config
from qdte.dataio import ensure_dir, read_json, write_json
from qdte.evolution.initialization import initialize_independent_oneway
from qdte.preprocess import load_and_preprocess_csv
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.eval_jax import answer_queries
from scripts.run_adaptive_selection_ablation import (
    BudgetConfig,
    _build_blocks,
    _run_scheme,
)
from scripts.run_orthogonal_low_budget_pilot import build_complete_low_order_workload
from scripts.run_orthogonal_low_budget_pilot import _measure_oneway_warmup


PROTOCOL_ID = "SAGE-QDTE-FULL-20260713-v4-development"
PROTOCOL_PATH = ROOT / "docs" / "SAGE_QDTE_FULL_PROTOCOL_V4_20260713.json"
SCHEME = "voi_sageordergain_harmonic_qproject_repeat"
FINAL_EPSILON_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
FINAL_DELTA = 1.0e-9
FINAL_ROUNDS = 50
FINAL_INITIAL_FIT_ITERS = 5000
FINAL_INNER_ITERS = 5000
FINAL_REFIT_ITERS = 5000
COVERAGE_RHO_FRACTION = 0.1
ADAPTIVE_MEASUREMENT_FRACTION = 0.9


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    log_term = math.log(1.0 / delta)
    return (math.sqrt(log_term + epsilon) - math.sqrt(log_term)) ** 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: Path) -> str:
    paths = sorted((root / "qdte").rglob("*.py"))
    paths.extend(
        [
            root / "scripts" / "run_adaptive_selection_ablation.py",
            root / "scripts" / "run_integrated_sage_qdte.py",
            root / "scripts" / "run_orthogonal_low_budget_pilot.py",
        ]
    )
    digest = hashlib.sha256(b"sage-qdte-source-tree-v1\0")
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    try:
        commit = run("rev-parse", "HEAD").strip()
        status = run("status", "--porcelain=v1")
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": commit,
            "worktree_dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "commit": None,
            "worktree_dirty": None,
            "provenance_error": str(error),
        }


def apply_projection_profile(config: dict[str, Any], profile: str) -> None:
    if profile == "P0_raw":
        set_nested(config, "projection.project_partitions", False)
        set_nested(config, "projection.clip_nonpartition", False)
        set_nested(config, "projection.prefix_monotonicity", False)
        set_nested(config, "projection.consistency.enabled", False)
    elif profile == "P1_lightweight":
        set_nested(config, "projection.project_partitions", True)
        set_nested(config, "projection.clip_nonpartition", True)
        set_nested(config, "projection.consistency.enabled", False)
    elif profile == "P2_equality":
        set_nested(config, "projection.consistency.enabled", True)
        set_nested(config, "projection.consistency.method", "query_space_lsq")
    elif profile == "P3_nonnegative":
        set_nested(config, "projection.consistency.enabled", True)
        set_nested(config, "projection.consistency.method", "query_space_feasible_lsq")
        set_nested(config, "projection.consistency.certificate_feasibility_tolerance", 1.0e-6)
        set_nested(config, "projection.consistency.certificate_gap_absolute_tolerance", 1.0e-7)
        set_nested(config, "projection.consistency.certificate_gap_relative_tolerance", 1.0e-8)
        set_nested(config, "projection.consistency.certificate_max_iterations", 1_000)
    else:
        raise ValueError(f"Unsupported projection profile: {profile!r}")


def build_integrated_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
    rho_total: float,
    delta: float,
    projection_profile: str,
    input_csv_override: Path | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    set_nested(config, "run.seed", int(seed))
    set_nested(config, "run.output_dir", str(output_dir))
    set_nested(config, "privacy.mode", "dp")
    set_nested(config, "privacy.rho_total", float(rho_total))
    set_nested(config, "privacy.delta", float(delta))
    set_nested(config, "privacy.measurement_mode", "static_all")
    if input_csv_override is not None:
        set_nested(config, "run.input_csv", str(input_csv_override))
    apply_projection_profile(config, projection_profile)
    if str(config.get("qdte", {}).get("objective_weighting", "variance")) != "variance":
        raise ValueError("Integrated SAGE-QDTE requires variance-weighted QDTE")
    if str(config.get("qdte", {}).get("transport_mode", "")) != "atom_flow":
        raise ValueError("Integrated SAGE-QDTE requires qdte.transport_mode='atom_flow'")
    validate_config(config)
    return config


def validate_protocol_request(
    *,
    protocol_mode: str,
    epsilon: float,
    delta: float,
    rounds: int,
    inner_iters: int,
    final_refit_iters: int,
    projection_profile: str,
) -> None:
    if protocol_mode not in {"final", "smoke"}:
        raise ValueError("protocol_mode must be final or smoke")
    if rounds <= 0 or inner_iters < 0 or final_refit_iters < 0:
        raise ValueError("rounds must be positive and QDTE iteration counts non-negative")
    if protocol_mode == "final":
        if not any(math.isclose(epsilon, value, rel_tol=0.0, abs_tol=1.0e-12) for value in FINAL_EPSILON_GRID):
            raise ValueError(f"final epsilon must be one of {FINAL_EPSILON_GRID}")
        if not math.isclose(delta, FINAL_DELTA, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"final delta must be {FINAL_DELTA}")
        if (rounds, inner_iters, final_refit_iters) != (
            FINAL_ROUNDS,
            FINAL_INNER_ITERS,
            FINAL_REFIT_ITERS,
        ):
            raise ValueError("final mode fixes rounds/inner/final-refit to 50/5000/5000")
        if projection_profile != "P1_lightweight":
            raise ValueError(
                "final v4 development protocol uses P1_lightweight until the P3+BootDiag gate passes"
            )


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required integrated artifact {name}: {path}")
        hashes[name] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return hashes


def _run_checks(
    *,
    summary: dict[str, Any],
    measurement: dict[str, Any],
    rho_total: float,
    rounds: int,
    final_refit_iters: int,
    projection_profile: str,
    coverage_rho_fraction: float,
    adaptive_measurement_fraction: float,
) -> dict[str, bool]:
    selected = summary.get("selected_blocks", [])
    selected_ids = [int(item["block_id"]) for item in selected]
    adaptive_projection = measurement.get("projection_diagnostics", {}).get(
        "adaptive_selection", {}
    )
    consistency = measurement.get("projection_diagnostics", {}).get("consistency", {})
    adaptive_ledger = adaptive_projection.get("adaptive_measurement_ledger", [])
    coverage_rho = float(adaptive_projection.get("coverage_rho", float("nan")))
    adaptive_rho = float(adaptive_projection.get("adaptive_rho_spent", float("nan")))
    adaptive_measurement_rho = float(
        adaptive_projection.get("adaptive_measurement_rho", float("nan"))
    )
    selection_rho_schedule = [
        float(value) for value in adaptive_projection.get("selection_rho_schedule", [])
    ]
    adaptive_selection_rho = float(
        adaptive_projection.get("adaptive_selection_rho", float("nan"))
    )
    ledger_rho = float(sum(float(item.get("rho", 0.0)) for item in adaptive_ledger))
    variances = np.asarray(measurement.get("variances", []), dtype=np.float64)
    checks = {
        "mode_is_dp": measurement.get("mode") == "dp",
        "rounds_complete": int(summary.get("rounds_completed", -1)) == int(rounds),
        "selected_block_count_matches_rounds": len(selected_ids) == int(rounds),
        "selection_rho_schedule_matches": math.isclose(
            sum(selection_rho_schedule),
            adaptive_selection_rho,
            rel_tol=1.0e-10,
            abs_tol=1.0e-15,
        ),
        "selection_rho_matches": math.isclose(
            adaptive_selection_rho,
            float(rho_total)
            * (1.0 - float(coverage_rho_fraction))
            * (1.0 - float(adaptive_measurement_fraction)),
            rel_tol=1.0e-10,
            abs_tol=1.0e-15,
        ),
        "rho_matches": math.isclose(
            float(measurement.get("rho_spent", float("nan"))),
            float(rho_total),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ),
        "coverage_rho_matches": math.isclose(
            coverage_rho,
            float(rho_total) * float(coverage_rho_fraction),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ),
        "adaptive_rho_matches": math.isclose(
            adaptive_rho,
            float(rho_total) * (1.0 - float(coverage_rho_fraction)),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ),
        "adaptive_measurement_rho_matches": math.isclose(
            adaptive_measurement_rho,
            float(rho_total)
            * (1.0 - float(coverage_rho_fraction))
            * float(adaptive_measurement_fraction),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ),
        "adaptive_ledger_reconstructs_rho": math.isclose(
            ledger_rho,
            adaptive_measurement_rho,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ),
        "partial_oneway_warmup_recorded": (
            adaptive_projection.get("base_measurements_present") is True
            and adaptive_projection.get("coverage_complete") is False
        ),
        "unmeasured_queries_remain_low_precision": bool(
            variances.size > 0 and np.any(variances >= 1.0e11)
        ),
        "all_query_variances_are_valid": bool(
            variances.size > 0
            and np.all(np.isfinite(variances))
            and np.all(variances > 0.0)
            and np.all(variances <= 1.000001e12)
        ),
        "selector_is_private_data": summary.get("selection_input") == "oracle",
        "selection_mechanism_is_exponential": summary.get("selection_mechanism")
        == "exponential",
        "selection_score_unit_sensitivity": math.isclose(
            float(summary.get("selection_score_sensitivity", float("nan"))),
            1.0,
            abs_tol=1.0e-15,
        ),
        "selection_rule_is_sample": summary.get("selection_rule") == "sample",
        "selection_ledger_charges_em": summary.get("selection_ledger") == "conservative",
        "generator_is_qdte_standard": summary.get("generator_profile") == "qdte_standard",
        "initial_fit_requested": int(summary.get("initial_fit", {}).get("requested_iters", -1))
        == FINAL_INITIAL_FIT_ITERS,
        "final_refit_requested": int(summary.get("final_refit", {}).get("requested_iters", -1))
        == int(final_refit_iters),
        "projection_certificate": (
            bool(consistency.get("feasible_projection_certificate_passed", False))
            if projection_profile == "P3_nonnegative"
            else True
        ),
    }
    return checks


def run_integrated(
    *,
    config_path: Path,
    output_dir: Path,
    seed: int,
    epsilon: float,
    delta: float,
    protocol_mode: str,
    rounds: int,
    inner_iters: int,
    final_refit_iters: int,
    projection_profile: str,
    input_csv_override: Path | None = None,
) -> dict[str, Any]:
    wall_start = time.time()
    validate_protocol_request(
        protocol_mode=protocol_mode,
        epsilon=epsilon,
        delta=delta,
        rounds=rounds,
        inner_iters=inner_iters,
        final_refit_iters=final_refit_iters,
        projection_profile=projection_profile,
    )
    rho_total = rho_for_epsilon(float(epsilon), float(delta))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Integrated output directory must be new or empty: {output_dir}"
        )
    output_dir = ensure_dir(output_dir)
    config = build_integrated_config(
        load_yaml(config_path),
        seed=int(seed),
        output_dir=output_dir,
        rho_total=rho_total,
        delta=float(delta),
        projection_profile=projection_profile,
        input_csv_override=input_csv_override,
    )
    resolved_config_path = output_dir / "config_integrated_resolved.yaml"
    save_yaml(config, resolved_config_path)

    preprocess_result = load_and_preprocess_csv(config)
    qcat, groups = build_complete_low_order_workload(
        preprocess_result.schema,
        max_pair_cells=int(config.get("workload", {}).get("max_pair_cells", 10_000)),
    )
    qcat_path = output_dir / "queries_full.json"
    schema_path = output_dir / "schema.json"
    qcat.save_json(qcat_path)
    preprocess_result.schema.save_json(schema_path)
    blocks = _build_blocks(groups, qcat)
    if any(not np.isfinite(block.delta_l2) or block.delta_l2 <= 0.0 for block in blocks):
        raise ValueError("Every integrated measurement block must have positive finite L2 sensitivity")

    true_answers = answer_queries(
        preprocess_result.X,
        qcat,
        batch_size=int(config.get("runtime", {}).get("answer_batch_size", 8192)),
    )
    coverage_rho = COVERAGE_RHO_FRACTION * rho_total
    adaptive_rho = (1.0 - COVERAGE_RHO_FRACTION) * rho_total
    base_measurements = _measure_oneway_warmup(
        true_answers=true_answers,
        qcat=qcat,
        groups=groups,
        schema=preprocess_result.schema,
        total_rows=int(preprocess_result.X.shape[0]),
        rho_total=coverage_rho,
        delta=float(delta),
        projection_cfg=dict(config.get("projection", {})),
        rng=np.random.default_rng(int(seed) + 5_000),
    )
    coverage_dir = ensure_dir(output_dir / "coverage_measurement")
    coverage_measurement_path = coverage_dir / "measurements.json"
    write_json(base_measurements.to_public_dict(), coverage_measurement_path)
    qcat.save_json(coverage_dir / "queries.json")
    preprocess_result.schema.save_json(coverage_dir / "schema.json")

    initial_syn = initialize_independent_oneway(
        qcat,
        base_measurements.target_projected,
        preprocess_result.schema,
        int(preprocess_result.X.shape[0]),
        np.random.default_rng(int(seed)),
    )
    initial_path = output_dir / "initial_synthetic_encoded.npy"
    np.save(initial_path, initial_syn)

    adaptive_measurement_rho = ADAPTIVE_MEASUREMENT_FRACTION * adaptive_rho
    adaptive_selection_rho = adaptive_rho - adaptive_measurement_rho
    measurement_sigma = math.sqrt(float(rounds) / (2.0 * adaptive_measurement_rho))
    selection_epsilon = math.sqrt(8.0 * adaptive_selection_rho / float(rounds))
    budget = BudgetConfig(
        mode="total_zcdp_private_em_selection",
        epsilon=selection_epsilon,
        measurement_sigma=measurement_sigma,
        rho_total=rho_total,
        budget_split_mu=None,
    )
    setup_path = output_dir / "integrated_setup.json"
    write_json(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_mode": protocol_mode,
            "paper_evidence_candidate": False,
            "scheme": SCHEME,
            "selection_input": "private_true_answers",
            "selection_mechanism": "exponential",
            "selection_score_sensitivity": 1.0,
            "selection_ledger": "conservative_bounded_range_em",
            "selection_rule": "sample",
            "public_bootstrap_rounds": 0,
            "nonpositive_score_fallback": "none",
            "generator_profile": "qdte_standard",
            "initialization": "dp_oneway_independent",
            "coverage_mode": "oneway_only",
            "initial_fit_iters": FINAL_INITIAL_FIT_ITERS,
            "generator_seed_mode": "per_stage",
            "coverage_rho_fraction": COVERAGE_RHO_FRACTION,
            "adaptive_rho_fraction": 1.0 - COVERAGE_RHO_FRACTION,
            "adaptive_measurement_fraction": ADAPTIVE_MEASUREMENT_FRACTION,
            "coverage_rho": coverage_rho,
            "adaptive_rho": adaptive_rho,
            "adaptive_measurement_rho": adaptive_measurement_rho,
            "adaptive_selection_rho": adaptive_selection_rho,
            "selection_epsilon_per_round": selection_epsilon,
            "projection_profile": projection_profile,
            "epsilon": float(epsilon),
            "delta": float(delta),
            "rho_total": rho_total,
            "rounds": int(rounds),
            "inner_iters": int(inner_iters),
            "final_refit_iters": int(final_refit_iters),
            "num_queries": int(qcat.m),
            "num_blocks": int(len(blocks)),
            "workload": "complete_orthogonal_oneway_twoway",
            "measurement_sigma_multiplier": measurement_sigma,
        },
        setup_path,
    )

    summary = _run_scheme(
        base_config=config,
        qcat=qcat,
        schema=preprocess_result.schema,
        blocks=blocks,
        true_answers=true_answers,
        initial_syn=initial_syn,
        cardinalities=preprocess_result.schema.cardinalities,
        output_dir=output_dir,
        scheme=SCHEME,
        rounds=int(rounds),
        inner_iters=int(inner_iters),
        budget=budget,
        rng=np.random.default_rng(int(seed) + 10_000),
        generator_profile="qdte_standard",
        initial_fit_iters=FINAL_INITIAL_FIT_ITERS,
        final_refit_iters=int(final_refit_iters),
        base_measurements=base_measurements,
        generator_seed_mode="per_stage",
        selection_input="oracle",
        selection_ledger="conservative",
        selection_rule="sample",
        public_bootstrap_rounds=0,
        nonpositive_score_fallback="none",
    )

    scheme_dir = output_dir / SCHEME
    final_measurement_dir = scheme_dir / f"round_{int(rounds):03d}_measurement"
    measurement_path = final_measurement_dir / "measurements.json"
    summary_path = scheme_dir / "adaptive_summary.json"
    timeseries_path = scheme_dir / "adaptive_timeseries.csv"
    if int(final_refit_iters) > 0:
        final_generation_dir = scheme_dir / "final_refit"
    else:
        final_generation_dir = scheme_dir / f"round_{int(rounds):03d}_generate"
    final_synthetic_path = final_generation_dir / "synthetic_encoded.npy"
    measurement = read_json(measurement_path)
    checks = _run_checks(
        summary=summary,
        measurement=measurement,
        rho_total=rho_total,
        rounds=rounds,
        final_refit_iters=final_refit_iters,
        projection_profile=projection_profile,
        coverage_rho_fraction=COVERAGE_RHO_FRACTION,
        adaptive_measurement_fraction=ADAPTIVE_MEASUREMENT_FRACTION,
    )
    artifact_hashes = _artifact_hashes(
        {
            "protocol": PROTOCOL_PATH,
            "resolved_config": resolved_config_path,
            "setup": setup_path,
            "schema": schema_path,
            "query_catalogue": qcat_path,
            "initial_synthetic": initial_path,
            "coverage_measurement": coverage_measurement_path,
            "final_measurement": measurement_path,
            "adaptive_summary": summary_path,
            "adaptive_timeseries": timeseries_path,
            "final_synthetic": final_synthetic_path,
            "final_generator_config": final_generation_dir / "config_resolved.yaml",
            "final_runtime": final_generation_dir / "runtime.json",
            "final_metrics": final_generation_dir / "metrics_final.json",
            "final_logs": final_generation_dir / "logs.txt",
        }
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_mode": protocol_mode,
        "paper_evidence_candidate": False,
        "paper_evidence_qualified": False,
        "config_path": str(config_path),
        "input_csv_override": str(input_csv_override) if input_csv_override is not None else None,
        "output_dir": str(output_dir),
        "seed": int(seed),
        "privacy": {
            "mode": "dp",
            "adjacency": "add_remove_one",
            "epsilon": float(epsilon),
            "delta": float(delta),
            "rho_total": rho_total,
            "epsilon_recomputed": zcdp_epsilon(rho_total, float(delta)),
            "selection_epsilon_per_round": selection_epsilon,
            "selection_rho_per_round": adaptive_selection_rho / float(rounds),
            "measurement_rho_per_round": adaptive_measurement_rho / float(rounds),
            "coverage_rho": coverage_rho,
        },
        "method": {
            "label": "SAGE-QDTE",
            "scheme": SCHEME,
            "selection_input": "private_true_answers",
            "selection_mechanism": "exponential",
            "selection_score_sensitivity": 1.0,
            "selection_ledger": "bounded_range_em_zcdp",
            "selection_rule": "sample",
            "public_bootstrap_rounds": 0,
            "rounds": int(rounds),
            "initial_fit_iters": FINAL_INITIAL_FIT_ITERS,
            "inner_iters": int(inner_iters),
            "final_refit_iters": int(final_refit_iters),
            "generator_profile": "qdte_standard",
            "initialization": "dp_oneway_independent",
            "coverage_mode": "oneway_only",
            "generator_seed_mode": "per_stage",
            "workload": "complete_orthogonal_oneway_twoway",
            "projection_profile": projection_profile,
            "coverage_rho_fraction": COVERAGE_RHO_FRACTION,
            "adaptive_rho_fraction": 1.0 - COVERAGE_RHO_FRACTION,
            "adaptive_measurement_fraction": ADAPTIVE_MEASUREMENT_FRACTION,
        },
        "checks": checks,
        "artifacts": artifact_hashes,
        "source_tree_sha256": source_tree_hash(ROOT),
        "runtime_seconds": float(time.time() - wall_start),
        "git": git_revision(ROOT),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "configured_device": config.get("run", {}).get("device"),
        },
    }
    manifest_path = output_dir / "integrated_manifest.json"
    write_json(manifest, manifest_path)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Integrated run checks failed: {failed}; see {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen integrated SAGE-QDTE pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=FINAL_DELTA)
    parser.add_argument("--protocol-mode", choices=["final", "smoke"], default="final")
    parser.add_argument("--rounds", type=int, default=FINAL_ROUNDS)
    parser.add_argument("--inner-iters", type=int, default=FINAL_INNER_ITERS)
    parser.add_argument("--final-refit-iters", type=int, default=FINAL_REFIT_ITERS)
    parser.add_argument(
        "--projection-profile",
        choices=["P0_raw", "P1_lightweight", "P2_equality", "P3_nonnegative"],
        default="P1_lightweight",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_integrated(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=int(args.seed),
        epsilon=float(args.epsilon),
        delta=float(args.delta),
        protocol_mode=args.protocol_mode,
        rounds=int(args.rounds),
        inner_iters=int(args.inner_iters),
        final_refit_iters=int(args.final_refit_iters),
        projection_profile=args.projection_profile,
        input_csv_override=args.input_csv,
    )
    print(f"integrated manifest: {manifest['output_dir']}/integrated_manifest.json")
    print(f"paper evidence qualified: {manifest['paper_evidence_qualified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
