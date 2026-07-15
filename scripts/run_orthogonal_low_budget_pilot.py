#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, set_nested
from qdte.config_validation import validate_config
from qdte.dataio import ensure_dir, write_json
from qdte.eval.external import write_external_evaluation
from qdte.evolution.initialization import initialize_independent_oneway
from qdte.measurement.measure import (
    MeasurementGroup,
    Measurements,
    _apply_configured_projection,
    measure_real_dataset,
)
from qdte.preprocess import load_and_preprocess_csv
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import OP_EQ, QueryBuilder, QueryCatalogue
from qdte.queries.workload import WorkloadGroup
from qdte.schema import TableSchema
from scripts.run_adaptive_selection_ablation import BudgetConfig, _build_blocks, _run_scheme


PROTOCOL_ID = "SAGE-QDTE-ORTHOGONAL-LOW-BUDGET-20260713-v4"
DEFAULT_SCHEME = "voi_sageordergain_harmonic_qproject"


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


def build_complete_low_order_workload(
    schema: TableSchema,
    *,
    max_pair_cells: int,
) -> tuple[QueryCatalogue, list[WorkloadGroup]]:
    """Build complete equality partitions; every returned group has L2 sensitivity one."""
    if int(max_pair_cells) <= 0:
        raise ValueError("max_pair_cells must be positive")
    builder = QueryBuilder(max_terms=2)
    groups: list[WorkloadGroup] = []

    def add_scope(scope: tuple[int, ...], family: str) -> None:
        start = len(builder.names)
        axes = [range(int(schema.columns[attr].cardinality)) for attr in scope]
        group_name = f"{family}:" + ":".join(str(attr) for attr in scope)
        for values in itertools.product(*axes):
            terms = [
                (int(attr), OP_EQ, int(value), int(value), int(value))
                for attr, value in zip(scope, values, strict=True)
            ]
            name = "&".join(
                f"{schema.columns[attr].name}={value}"
                for attr, value in zip(scope, values, strict=True)
            )
            if not builder.add(terms, name=name, group=group_name, family=family):
                raise RuntimeError(f"Duplicate query while constructing partition {group_name}")
        stop = len(builder.names)
        groups.append(
            WorkloadGroup(
                name=group_name,
                family=family,
                query_indices=np.arange(start, stop, dtype=np.int32),
                sensitivity_l2=1.0,
                is_partition=True,
            )
        )

    for attr in range(schema.d):
        add_scope((int(attr),), "oneway")
    for left, right in itertools.combinations(range(schema.d), 2):
        cells = int(schema.columns[left].cardinality) * int(schema.columns[right].cardinality)
        if cells <= int(max_pair_cells):
            add_scope((int(left), int(right)), "twoway")

    qcat = builder.build()
    qcat.validate(schema.cardinalities)
    return qcat, groups


def _prepare_config(
    base: dict[str, Any],
    *,
    input_csv: Path,
    output_dir: Path,
    seed: int,
    rho_total: float,
    delta: float,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    set_nested(config, "run.input_csv", str(input_csv))
    set_nested(config, "run.output_dir", str(output_dir))
    set_nested(config, "run.seed", int(seed))
    set_nested(config, "privacy.mode", "dp")
    set_nested(config, "privacy.rho_total", float(rho_total))
    set_nested(config, "privacy.delta", float(delta))
    set_nested(config, "privacy.measurement_mode", "static_all")
    set_nested(config, "workload.reuse_from_measurement", True)
    set_nested(config, "projection.project_partitions", True)
    set_nested(config, "projection.clip_nonpartition", True)
    set_nested(config, "projection.prefix_monotonicity", False)
    set_nested(config, "projection.consistency.enabled", False)
    set_nested(config, "init.method", "independent_oneway")
    set_nested(config, "evaluation.compute_true_query_error", False)
    set_nested(config, "evaluation.compute_heldout_query_error", False)
    set_nested(config, "evaluation.save_synthetic_csv", False)
    set_nested(config, "runtime.log_measurement_groups", False)
    set_nested(config, "runtime.xla_preallocate", False)
    validate_config(config)
    return config


def _measurement_config(
    config: dict[str, Any],
    *,
    coverage_rho: float,
    oneway_weight: float,
    twoway_weight: float,
) -> dict[str, Any]:
    measured = copy.deepcopy(config)
    set_nested(measured, "privacy.rho_total", float(coverage_rho))
    set_nested(measured, "privacy.measurement_allocation.oneway", float(oneway_weight))
    set_nested(measured, "privacy.measurement_allocation.twoway", float(twoway_weight))
    allocation = measured["privacy"]["measurement_allocation"]
    for family in list(allocation):
        if family not in {"oneway", "twoway"}:
            del allocation[family]
    return measured


def _write_base_artifact(
    path: Path,
    *,
    qcat: QueryCatalogue,
    schema: TableSchema,
    measurements: Any,
) -> None:
    ensure_dir(path)
    qcat.save_json(path / "queries.json")
    schema.save_json(path / "schema.json")
    write_json(measurements.to_public_dict(), path / "measurements.json")


def _measure_oneway_warmup(
    *,
    true_answers: np.ndarray,
    qcat: QueryCatalogue,
    groups: list[WorkloadGroup],
    schema: TableSchema,
    total_rows: int,
    rho_total: float,
    delta: float,
    projection_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> Measurements:
    """Measure every one-way partition once and leave other queries untrusted."""
    oneway = [group for group in groups if group.family == "oneway"]
    if not oneway:
        raise ValueError("one-way warm-up requires at least one one-way group")
    if not np.isfinite(rho_total) or float(rho_total) <= 0.0:
        raise ValueError("one-way warm-up rho must be positive and finite")
    target = np.zeros(int(qcat.m), dtype=np.float32)
    variances = np.full(int(qcat.m), 1.0e12, dtype=np.float32)
    measured_groups: list[MeasurementGroup] = []
    covered = np.zeros(int(qcat.m), dtype=bool)
    rho_per_group = float(rho_total) / float(len(oneway))
    sigma = math.sqrt(1.0 / (2.0 * rho_per_group))
    for group in oneway:
        idx = group.query_indices.astype(np.int32, copy=False)
        if not group.is_partition or not math.isclose(
            float(group.sensitivity_l2), 1.0, abs_tol=1.0e-12
        ):
            raise ValueError("one-way warm-up groups must be unit-sensitivity partitions")
        target[idx] = (
            true_answers[idx].astype(np.float64, copy=False)
            + rng.normal(0.0, sigma, size=int(idx.size))
        ).astype(np.float32)
        variances[idx] = np.float32(sigma * sigma)
        covered[idx] = True
        measured_groups.append(
            MeasurementGroup(
                query_indices=idx.astype(np.int32, copy=True),
                sensitivity_l2=1.0,
                rho=rho_per_group,
                sigma=sigma,
                noise_std=sigma,
                name=f"warmup:{group.name}",
                family=group.family,
                is_partition=True,
            )
        )
    unmeasured = np.flatnonzero(~covered).astype(np.int32)
    if unmeasured.size:
        unmeasured_sigma = float(math.sqrt(1.0e12))
        measured_groups.append(
            MeasurementGroup(
                query_indices=unmeasured,
                sensitivity_l2=1.0,
                rho=0.0,
                sigma=unmeasured_sigma,
                noise_std=unmeasured_sigma,
                name="warmup:unmeasured",
                family="unmeasured",
                is_partition=False,
            )
        )
    projected, diagnostics = _apply_configured_projection(
        target,
        qcat,
        measured_groups,
        int(total_rows),
        projection_cfg,
        variances,
        schema.cardinalities,
    )
    diagnostics["initialization"] = {
        "mode": "oneway_only",
        "num_oneway_groups": len(oneway),
        "rho_per_group": rho_per_group,
        "sigma": sigma,
        "unmeasured_queries": int(unmeasured.size),
        "unmeasured_variance": 1.0e12,
    }
    return Measurements(
        target_noisy=target,
        target_projected=projected.astype(np.float32),
        variances=variances,
        inv_variances=(1.0 / variances.astype(np.float64)).astype(np.float32),
        groups=measured_groups,
        mode="dp",
        rho_total=float(rho_total),
        rho_spent=float(rho_total),
        epsilon_delta=zcdp_epsilon(float(rho_total), float(delta)),
        delta=float(delta),
        projection_diagnostics=diagnostics,
        num_rows=int(total_rows),
    )


def _all_groups_are_complete_partitions(groups: list[WorkloadGroup], qcat: QueryCatalogue) -> bool:
    coverage = np.zeros(qcat.m, dtype=np.int32)
    for group in groups:
        if not group.is_partition or not math.isclose(float(group.sensitivity_l2), 1.0, abs_tol=1.0e-12):
            return False
        coverage[group.query_indices] += 1
    return bool(np.all(coverage == 1))


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    if not 0.0 < float(args.coverage_fraction) < 1.0:
        raise ValueError("coverage_fraction must be in (0, 1)")
    if float(args.oneway_weight) <= 0.0 or float(args.twoway_weight) <= 0.0:
        raise ValueError("coverage family weights must be positive")
    if (
        int(args.rounds) <= 0
        or int(args.initial_fit_iters) < 0
        or int(args.inner_iters) <= 0
        or int(args.final_refit_iters) <= 0
    ):
        raise ValueError("rounds/inner/final must be positive and initial fit must be non-negative")
    if (
        int(args.initial_fit_stop_patience) < 0
        or int(args.inner_stop_patience) < 0
        or int(args.final_stop_patience) < 0
    ):
        raise ValueError("generator stop patience values must be non-negative")

    scheme = str(args.scheme)
    if bool(args.allow_repeats) and not scheme.endswith("_repeat"):
        scheme = f"{scheme}_repeat"
    allow_repeats = scheme.endswith("_repeat")

    output_dir = ensure_dir(args.output_dir)
    rho_total = rho_for_epsilon(float(args.epsilon), float(args.delta))
    config = _prepare_config(
        load_yaml(args.config),
        input_csv=args.input_dir / "raw.csv",
        output_dir=output_dir,
        seed=int(args.seed),
        rho_total=rho_total,
        delta=float(args.delta),
    )
    preprocessed = load_and_preprocess_csv(config)
    qcat, groups = build_complete_low_order_workload(
        preprocessed.schema,
        max_pair_cells=int(args.max_pair_cells),
    )
    if not _all_groups_are_complete_partitions(groups, qcat):
        raise RuntimeError("O1 workload failed the complete partition invariant")
    candidate_groups = (
        groups
        if args.coverage_mode == "oneway"
        else [group for group in groups if group.family == "twoway"]
    )
    blocks = _build_blocks(candidate_groups, qcat)
    if not allow_repeats and len(blocks) < int(args.rounds):
        raise ValueError(
            f"O1 no-repeat selection needs at least {args.rounds} two-way blocks, found {len(blocks)}"
        )

    coverage_rho = float(args.coverage_fraction) * rho_total
    adaptive_rho = rho_total - coverage_rho
    if args.selection_mode == "private_em":
        measurement_fraction = float(args.adaptive_measurement_fraction)
        if not 0.0 < measurement_fraction < 1.0:
            raise ValueError("adaptive_measurement_fraction must be in (0, 1)")
        adaptive_measurement_rho = measurement_fraction * adaptive_rho
        adaptive_selection_rho = adaptive_rho - adaptive_measurement_rho
        selection_epsilon = math.sqrt(
            8.0 * adaptive_selection_rho / float(args.rounds)
        )
        selection_input = "oracle"
        selection_ledger = "conservative"
        selection_rule = "sample"
    else:
        adaptive_measurement_rho = adaptive_rho
        adaptive_selection_rho = 0.0
        selection_epsilon = 1.0
        selection_input = "transcript"
        selection_ledger = "measurement_only"
        selection_rule = "argmax"
    exact_measurement_answers = answer_queries(
        preprocessed.X,
        qcat,
        batch_size=int(config.get("runtime", {}).get("answer_batch_size", 8192)),
    )
    if args.coverage_mode == "oneway":
        base_measurements = _measure_oneway_warmup(
            true_answers=exact_measurement_answers,
            qcat=qcat,
            groups=groups,
            schema=preprocessed.schema,
            total_rows=int(preprocessed.X.shape[0]),
            rho_total=coverage_rho,
            delta=float(args.delta),
            projection_cfg=dict(config.get("projection", {})),
            rng=np.random.default_rng([int(args.seed), 0x4F315F434F564552]),
        )
    else:
        measurement_config = _measurement_config(
            config,
            coverage_rho=coverage_rho,
            oneway_weight=float(args.oneway_weight),
            twoway_weight=float(args.twoway_weight),
        )
        base_measurements = measure_real_dataset(
            preprocessed.X,
            qcat,
            groups,
            measurement_config,
            np.random.default_rng([int(args.seed), 0x4F315F434F564552]),
            batch_size=int(config.get("runtime", {}).get("answer_batch_size", 8192)),
            cardinalities=preprocessed.schema.cardinalities,
        )
    base_dir = output_dir / "orthogonal_coverage"
    _write_base_artifact(
        base_dir,
        qcat=qcat,
        schema=preprocessed.schema,
        measurements=base_measurements,
    )

    initial = initialize_independent_oneway(
        qcat,
        base_measurements.target_projected,
        preprocessed.schema,
        int(preprocessed.X.shape[0]),
        np.random.default_rng(int(args.seed)),
    )
    initial_path = output_dir / "initial_synthetic_encoded.npy"
    np.save(initial_path, initial)

    # The exact answers remain inside the trusted DP mechanism: EM consumes them
    # for selection and Gaussian measurements release only noisy query blocks.
    measurement_sigma = math.sqrt(
        float(args.rounds) / (2.0 * adaptive_measurement_rho)
    )
    budget = BudgetConfig(
        mode=(
            "total_zcdp_private_em_selection"
            if args.selection_mode == "private_em"
            else "total_zcdp_measurement_only_selection"
        ),
        epsilon=selection_epsilon,
        measurement_sigma=measurement_sigma,
        rho_total=rho_total,
        budget_split_mu=None,
    )
    summary = _run_scheme(
        base_config=config,
        qcat=qcat,
        schema=preprocessed.schema,
        blocks=blocks,
        true_answers=exact_measurement_answers,
        initial_syn=initial,
        cardinalities=preprocessed.schema.cardinalities,
        output_dir=output_dir,
        scheme=scheme,
        rounds=int(args.rounds),
        inner_iters=int(args.inner_iters),
        budget=budget,
        rng=np.random.default_rng([int(args.seed), 0x4F315F524546494E45]),
        generator_profile="qdte_standard",
        initial_fit_iters=int(args.initial_fit_iters),
        final_refit_iters=int(args.final_refit_iters),
        base_measurements=base_measurements,
        generator_seed_mode=str(args.generator_seed_mode),
        initial_fit_stop_patience=(
            None
            if int(args.initial_fit_stop_patience) == 0
            else int(args.initial_fit_stop_patience)
        ),
        inner_stop_patience=(
            None if int(args.inner_stop_patience) == 0 else int(args.inner_stop_patience)
        ),
        final_stop_patience=(
            None if int(args.final_stop_patience) == 0 else int(args.final_stop_patience)
        ),
        measurement_schedule=str(args.measurement_schedule),
        selection_input=selection_input,
        bootstrap_strategy="low_order_coverage",
        transcript_reliability="all",
        selection_ledger=selection_ledger,
        selection_rule=selection_rule,
        public_bootstrap_rounds=0,
        nonpositive_score_fallback="none",
        transcript_sage_prior_odds=float(args.transcript_sage_prior_odds),
    )

    synthetic_path = output_dir / scheme / "final_refit" / "synthetic_encoded.npy"
    evaluation_path = output_dir / "evaluation.json"
    write_external_evaluation(
        args.input_dir,
        synthetic_path,
        evaluation_path,
        true_answers_cache_path=args.input_dir / "true_answers_cache.npz",
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    final_measurement_dir = summary.get("final_measurement_dir")
    if final_measurement_dir is None:
        raise RuntimeError("O1 completed without a final measurement artifact")
    final_measurement_path = Path(str(final_measurement_dir)) / "measurements.json"
    final_measurement = json.loads(final_measurement_path.read_text(encoding="utf-8"))
    rho_spent = float(final_measurement["rho_spent"])
    adaptive_ledger = final_measurement["projection_diagnostics"]["adaptive_selection"]
    selection_rho_schedule = [float(value) for value in adaptive_ledger["selection_rho_schedule"]]
    measurement_rho_spent = float(adaptive_ledger["adaptive_measurement_rho"])
    selection_rho_spent = float(adaptive_ledger["adaptive_selection_rho"])
    checks = {
        "all_direct_groups_are_complete_partitions": _all_groups_are_complete_partitions(groups, qcat),
        "all_group_sensitivities_are_one": all(
            math.isclose(float(group.sensitivity_l2), 1.0, abs_tol=1.0e-12) for group in groups
        ),
        "only_oneway_twoway_families": set(qcat.families) == {"oneway", "twoway"},
        "selection_mechanism_matches_protocol": (
            summary.get("selection_mechanism")
            == ("exponential" if args.selection_mode == "private_em" else "postprocessing")
        ),
        "selection_score_sensitivity_certified": (
            args.selection_mode != "private_em"
            or math.isclose(
                float(summary.get("selection_score_sensitivity", float("nan"))),
                1.0,
                abs_tol=1.0e-15,
            )
        ),
        "selection_rho_schedule_matches": math.isclose(
            sum(selection_rho_schedule),
            selection_rho_spent,
            rel_tol=1.0e-10,
            abs_tol=1.0e-15,
        ),
        "adaptive_rho_matches": math.isclose(
            selection_rho_spent + measurement_rho_spent,
            adaptive_rho,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ),
        "rho_spent_matches_total": math.isclose(rho_spent, rho_total, rel_tol=1.0e-10, abs_tol=1.0e-12),
        "synthetic_exists": synthetic_path.is_file(),
        "evaluation_exists": evaluation_path.is_file(),
    }
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "completed" if all(checks.values()) else "failed_checks",
        "paper_evidence_qualified": False,
        "development_only": True,
        "dataset": str(args.dataset),
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "rho_total": rho_total,
        "rho_spent": rho_spent,
        "seed": int(args.seed),
        "scheme": scheme,
        "selection_mode": str(args.selection_mode),
        "coverage_mode": str(args.coverage_mode),
        "selection_input": selection_input,
        "selection_rule": selection_rule,
        "selection_score_sensitivity": summary.get("selection_score_sensitivity"),
        "transcript_sage_prior": summary.get("transcript_sage_prior"),
        "selection_epsilon_per_round": selection_epsilon,
        "selection_rho_planned": adaptive_selection_rho,
        "selection_rho_spent": selection_rho_spent,
        "adaptive_measurement_rho_planned": adaptive_measurement_rho,
        "adaptive_measurement_rho_spent": measurement_rho_spent,
        "adaptive_measurement_fraction": measurement_rho_spent / adaptive_rho,
        "coverage_fraction": float(args.coverage_fraction),
        "adaptive_fraction": 1.0 - float(args.coverage_fraction),
        "coverage_family_weights": (
            {"oneway": 1.0}
            if args.coverage_mode == "oneway"
            else {
                "oneway": float(args.oneway_weight),
                "twoway": float(args.twoway_weight),
            }
        ),
        "rounds": int(args.rounds),
        "rounds_completed": int(summary["rounds_completed"]),
        "initial_fit_iters": int(args.initial_fit_iters),
        "initial_fit_stop_patience": int(args.initial_fit_stop_patience),
        "inner_iters": int(args.inner_iters),
        "inner_stop_patience": int(args.inner_stop_patience),
        "final_refit_iters": int(args.final_refit_iters),
        "final_stop_patience": int(args.final_stop_patience),
        "generator_seed_mode": str(args.generator_seed_mode),
        "measurement_schedule": str(args.measurement_schedule),
        "allow_repeats": allow_repeats,
        "num_queries": int(qcat.m),
        "num_oneway_blocks": int(sum(group.family == "oneway" for group in groups)),
        "num_twoway_blocks": int(sum(group.family == "twoway" for group in groups)),
        "measurement_sigma_refinement": measurement_sigma,
        "checks": checks,
        "summary": summary,
        "evaluation": evaluation,
        "artifacts": {
            "coverage_measurements": {
                "path": str(base_dir / "measurements.json"),
                "sha256": sha256_file(base_dir / "measurements.json"),
            },
            "final_measurements": {
                "path": str(final_measurement_path),
                "sha256": sha256_file(final_measurement_path),
            },
            "initial_synthetic": {
                "path": str(initial_path),
                "sha256": sha256_file(initial_path),
            },
            "final_synthetic": {
                "path": str(synthetic_path),
                "sha256": sha256_file(synthetic_path),
            },
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": sha256_file(evaluation_path),
            },
        },
    }
    write_json(manifest, output_dir / "orthogonal_low_budget_manifest.json")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"O1 manifest checks failed: {failed}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen O1 orthogonal low-budget SAGE-QDTE pilot.")
    parser.add_argument("--dataset", required=True, choices=["adult", "acs", "br2000", "nltcs"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epsilon", required=True, type=float, choices=[0.1, 0.3])
    parser.add_argument("--delta", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument(
        "--initial-fit-iters",
        type=int,
        default=5000,
        help="QDTE iterations used to fit the released coverage transcript before round-one selection.",
    )
    parser.add_argument(
        "--initial-fit-stop-patience",
        type=int,
        default=0,
        help="No-progress patience for the preselection fit; 0 uses initial-fit-iters.",
    )
    parser.add_argument("--inner-iters", type=int, default=5000)
    parser.add_argument(
        "--inner-stop-patience",
        type=int,
        default=0,
        help="No-progress patience per adaptive QDTE stage; 0 uses inner-iters.",
    )
    parser.add_argument("--final-refit-iters", type=int, default=5000)
    parser.add_argument(
        "--final-stop-patience",
        type=int,
        default=0,
        help="No-progress patience for the final refit; 0 uses final-refit-iters.",
    )
    parser.add_argument(
        "--generator-seed-mode",
        choices=["restart", "per_stage"],
        default="per_stage",
        help="Use the legacy restarted RNG or a reproducible fresh stream per QDTE stage.",
    )
    parser.add_argument(
        "--measurement-schedule",
        choices=["fixed", "aim_released_change"],
        default="fixed",
        help="Use fixed per-round noise or DP-safe AIM-style annealing from released model change.",
    )
    parser.add_argument(
        "--allow-repeats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow a previously selected complete query block to be remeasured and precision-combined.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["private_em", "released_transcript"],
        default="private_em",
        help="Use certified private SAGE selection or the superseded transcript-only diagnostic.",
    )
    parser.add_argument(
        "--adaptive-measurement-fraction",
        type=float,
        default=0.9,
        help="Fraction of adaptive rho assigned to Gaussian measurement; the rest pays for EM selection.",
    )
    parser.add_argument(
        "--coverage-mode",
        choices=["oneway", "all_low_order"],
        default="oneway",
        help="Use AIM-style one-way warm-up or the superseded all-low-order coverage diagnostic.",
    )
    parser.add_argument("--coverage-fraction", type=float, default=0.1)
    parser.add_argument("--oneway-weight", type=float, default=0.4)
    parser.add_argument("--twoway-weight", type=float, default=0.6)
    parser.add_argument("--max-pair-cells", type=int, default=10000)
    parser.add_argument("--scheme", default=DEFAULT_SCHEME)
    parser.add_argument(
        "--transcript-sage-prior-odds",
        type=float,
        default=1.0,
        help=(
            "Use released-transcript SAGE ranks as a bounded conditional base measure; "
            "the rescue pilot predeclares 4.0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pilot(args)
    print(json.dumps({"manifest": str(args.output_dir / "orthogonal_low_budget_manifest.json"), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
