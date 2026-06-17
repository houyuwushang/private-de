#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/qianqiu/.anaconda3/bin/conda"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "robust_ab_20260612"
PRIVATE_GSD_REF = Path("/home/qianqiu/my_life/baseline/private-de/private-gsd")


@dataclass(frozen=True)
class Scenario:
    name: str
    config: str
    seeds: tuple[int, ...]
    variants: tuple[str, ...]
    overrides: tuple[str, ...]
    milestones: tuple[int, ...]
    requires_input: str | None = None


BASE_VARIANTS = ("constructive_pair", "constructive_partner_b2", "random_best_edit", "pgsd_style_mutate50")


def _kv(key: str, value: object) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return f"--{key}={text}"


def _common_smoke_overrides(max_iters: int, log_every: int = 100) -> tuple[str, ...]:
    return (
        _kv("projection.consistency.enabled", True),
        _kv("projection.consistency.method", "local_table_feasible_jax"),
        _kv("projection.consistency.max_scope_cells", 200000),
        _kv("projection.consistency.max_dense_constraint_cells", 20000000),
        _kv("qdte.max_iters", max_iters),
        _kv("qdte.stop_patience", max_iters),
        _kv("qdte.log_every", log_every),
        _kv("qdte.candidate_diagnostics", False),
        _kv("evaluation.save_synthetic_csv", False),
    )


def default_scenarios(include_adult: bool) -> list[Scenario]:
    scenarios = [
        Scenario(
            name="smoke_default_2000",
            config="configs/smoke.yaml",
            seeds=(0, 1, 2),
            variants=BASE_VARIANTS,
            overrides=_common_smoke_overrides(2000, log_every=100),
            milestones=(100, 500, 1000, 2000),
        ),
        Scenario(
            name="smoke_low_candidates_1000",
            config="configs/smoke.yaml",
            seeds=(0, 1, 2),
            variants=BASE_VARIANTS,
            overrides=(
                *_common_smoke_overrides(1000, log_every=100),
                _kv("qdte.total_candidates_per_iter", 64),
                _kv("qdte.candidates_per_target", 4),
                _kv("qdte.accepted_per_iter", 4),
                _kv("qdte.constructive_partner_side_budget", 32),
            ),
            milestones=(100, 500, 1000),
        ),
        Scenario(
            name="smoke_strict_privacy_1000",
            config="configs/smoke.yaml",
            seeds=(0, 1, 2),
            variants=BASE_VARIANTS,
            overrides=(
                *_common_smoke_overrides(1000, log_every=100),
                _kv("privacy.rho_total", 0.25),
            ),
            milestones=(100, 500, 1000),
        ),
        Scenario(
            name="smoke_larger_workload_1000",
            config="configs/smoke.yaml",
            seeds=(0, 1),
            variants=BASE_VARIANTS,
            overrides=(
                *_common_smoke_overrides(1000, log_every=100),
                _kv("workload.max_queries", 800),
                _kv("workload.range_intervals_per_num_attr", 12),
                _kv("workload.mixed_queries_per_pair", 16),
                _kv("workload.max_2way_cells", 300),
            ),
            milestones=(100, 500, 1000),
        ),
    ]
    if include_adult:
        scenarios.append(
            Scenario(
                name="adult_small_200",
                config="configs/adult_qdte.yaml",
                seeds=(0,),
                variants=BASE_VARIANTS,
                overrides=(
                    _kv("qdte.max_iters", 200),
                    _kv("qdte.stop_patience", 200),
                    _kv("qdte.log_every", 50),
                    _kv("qdte.eval_every", 50),
                    _kv("qdte.num_active_targets", 32),
                    _kv("qdte.candidates_per_target", 16),
                    _kv("qdte.total_candidates_per_iter", 512),
                    _kv("qdte.accepted_per_iter", 16),
                    _kv("qdte.constructive_partner_side_budget", 128),
                    _kv("runtime.use_pmap", False),
                    _kv("runtime.xla_preallocate", False),
                    _kv("runtime.scoring_chunk_size", 2048),
                    _kv("runtime.answer_batch_size", 4096),
                    _kv("evaluation.save_synthetic_csv", False),
                    _kv("qdte.candidate_diagnostics", False),
                ),
                milestones=(50, 100, 200),
                requires_input="/home/qianqiu/rerun-experiment/dataset/adult.csv",
            )
        )
    return scenarios


def _run_dir(output_root: Path, scenario: str, seed: int, variant: str) -> Path:
    return output_root / scenario / f"seed{seed}_{variant}"


def _base_output(output_root: Path, scenario: str, seed: int) -> Path:
    return output_root / scenario / f"seed{seed}"


def _command(scenario: Scenario, seed: int, variant: str, output_root: Path) -> list[str]:
    variant_overrides: tuple[str, ...] = ()
    if variant in {"random_best_edit", "random_best_mutation", "private_gsd_mutate", "pgsd_mutate"}:
        # This is a QDTE-internal random-best-edit ablation: random one-row
        # mutations are scored by the QDTE edit-advantage / global loss drop,
        # then the best improving single edit is accepted.
        variant_overrides = (_kv("qdte.accepted_per_iter", 1),)
    elif variant in {"pgsd_style_mutate50", "private_gsd_mutate50"}:
        # Closer mutate-only Private-GSD-style setting: a 50-member random
        # mutation population and one best improving dataset update per
        # generation. It still uses the local QDTE workload/measurement stack,
        # so it is not a drop-in run of the upstream Private-GSD package.
        variant_overrides = (
            _kv("qdte.total_candidates_per_iter", 50),
            _kv("qdte.candidates_per_target", 50),
            _kv("qdte.num_active_targets", 1),
            _kv("qdte.accepted_per_iter", 1),
            _kv("qdte.kappa_noise", 0.0),
        )
    return [
        PYTHON,
        "run",
        "-n",
        "qdte",
        "python",
        "scripts/run_ablation.py",
        "--config",
        scenario.config,
        "--variant",
        variant,
        _kv("run.output_dir", _base_output(output_root, scenario.name, seed)),
        _kv("run.seed", seed),
        _kv("workload.random_seed", seed),
        *scenario.overrides,
        *variant_overrides,
    ]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_timeseries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _milestone_row(rows: list[dict[str, str]], milestone: int) -> dict[str, str] | None:
    best: dict[str, str] | None = None
    best_iter = -1
    for row in rows:
        try:
            iteration = int(float(row.get("iteration", "")))
        except ValueError:
            continue
        if iteration <= milestone and iteration > best_iter:
            best = row
            best_iter = iteration
    return best


def summarize(output_root: Path, scenarios: list[Scenario]) -> None:
    final_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in scenario.seeds:
            for variant in scenario.variants:
                run_dir = _run_dir(output_root, scenario.name, seed, variant)
                metrics_path = run_dir / "metrics_final.json"
                runtime_path = run_dir / "runtime.json"
                timeseries_path = run_dir / "metrics_timeseries.csv"
                if not metrics_path.exists():
                    final_rows.append(
                        {
                            "scenario": scenario.name,
                            "seed": seed,
                            "variant": variant,
                            "status": "missing",
                            "run_dir": str(run_dir),
                        }
                    )
                    continue
                metrics = _load_json(metrics_path)
                runtime = _load_json(runtime_path) if runtime_path.exists() else {}
                final_rows.append(
                    {
                        "scenario": scenario.name,
                        "seed": seed,
                        "variant": variant,
                        "status": "ok",
                        "final_measured_loss": metrics.get("final_measured_loss"),
                        "final_rms_standardized_residual": metrics.get("final_rms_standardized_residual"),
                        "true_query_rmse": metrics.get("true_query_rmse"),
                        "true_query_mae": metrics.get("true_query_mae"),
                        "num_queries": metrics.get("num_queries"),
                        "num_candidates_scored": metrics.get("num_candidates_scored"),
                        "num_candidates_requested": metrics.get("num_candidates_requested"),
                        "num_accepted_edits": metrics.get("num_accepted_edits"),
                        "wall_clock_seconds": runtime.get("wall_clock_seconds"),
                        "positive_returned_rate": runtime.get("positive_returned_rate"),
                        "selected_nonconflicting": runtime.get("num_selected_nonconflicting_candidates"),
                        "source_filter_failure_rate": runtime.get("source_filter_failure_rate"),
                        "last_explicit_pairs_evaluated": runtime.get("last_constructive_pair_explicit_pairs_evaluated"),
                        "last_explicit_positive_pairs": runtime.get("last_constructive_pair_explicit_positive_pairs"),
                        "gpu_devices": ",".join(str(x) for x in runtime.get("gpu_devices", [])),
                        "run_dir": str(run_dir),
                    }
                )
                rows = _read_timeseries(timeseries_path)
                for milestone in scenario.milestones:
                    row = _milestone_row(rows, milestone)
                    if row is None:
                        continue
                    convergence_rows.append(
                        {
                            "scenario": scenario.name,
                            "seed": seed,
                            "variant": variant,
                            "milestone": milestone,
                            "iteration": row.get("iteration"),
                            "measured_loss": row.get("measured_loss"),
                            "rms_standardized_residual": row.get("rms_standardized_residual"),
                            "accepted_edits_this_iter": row.get("accepted_edits"),
                            "positive_returned_rate": row.get("positive_returned_rate"),
                            "num_candidates": row.get("num_candidates"),
                            "run_dir": str(run_dir),
                        }
                    )
    _write_csv(output_root / "robustness_summary.csv", final_rows)
    _write_csv(output_root / "convergence_summary.csv", convergence_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--include-adult", action="store_true")
    parser.add_argument("--only-scenario", action="append", default=None)
    parser.add_argument("--only-variant", action="append", default=None)
    parser.add_argument("--only-seed", action="append", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scenarios = default_scenarios(include_adult=bool(args.include_adult))
    if args.only_scenario:
        wanted = set(args.only_scenario)
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]
    if args.only_variant:
        wanted_variants = tuple(args.only_variant)
        scenarios = [
            Scenario(
                name=scenario.name,
                config=scenario.config,
                seeds=scenario.seeds,
                variants=tuple(v for v in scenario.variants if v in wanted_variants),
                overrides=scenario.overrides,
                milestones=scenario.milestones,
                requires_input=scenario.requires_input,
            )
            for scenario in scenarios
        ]
    if args.only_seed:
        wanted_seeds = set(args.only_seed)
        scenarios = [
            Scenario(
                name=scenario.name,
                config=scenario.config,
                seeds=tuple(seed for seed in scenario.seeds if seed in wanted_seeds),
                variants=scenario.variants,
                overrides=scenario.overrides,
                milestones=scenario.milestones,
                requires_input=scenario.requires_input,
            )
            for scenario in scenarios
        ]
    scenarios = [scenario for scenario in scenarios if scenario.seeds and scenario.variants]

    manifest = {
        "output_root": str(output_root),
        "private_gsd_reference": str(PRIVATE_GSD_REF),
        "private_gsd_reference_exists": PRIVATE_GSD_REF.exists(),
        "scenarios": [
            {
                "name": scenario.name,
                "config": scenario.config,
                "seeds": list(scenario.seeds),
                "variants": list(scenario.variants),
                "overrides": list(scenario.overrides),
                "milestones": list(scenario.milestones),
                "requires_input": scenario.requires_input,
            }
            for scenario in scenarios
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    runner_log = output_root / "runner.log"
    with runner_log.open("a", encoding="utf-8") as log:
        for scenario in scenarios:
            if scenario.requires_input and not Path(scenario.requires_input).exists():
                msg = f"SKIP scenario={scenario.name}: missing {scenario.requires_input}"
                print(msg, flush=True)
                log.write(msg + "\n")
                continue
            for seed in scenario.seeds:
                for variant in scenario.variants:
                    run_dir = _run_dir(output_root, scenario.name, seed, variant)
                    if args.skip_existing and (run_dir / "metrics_final.json").exists():
                        msg = f"SKIP existing scenario={scenario.name} seed={seed} variant={variant}"
                        print(msg, flush=True)
                        log.write(msg + "\n")
                        continue
                    cmd = _command(scenario, seed, variant, output_root)
                    msg = "RUN " + " ".join(str(part) for part in cmd)
                    print(msg, flush=True)
                    log.write(msg + "\n")
                    log.flush()
                    if args.dry_run:
                        continue
                    proc = subprocess.run(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    log.write(proc.stdout)
                    log.write(f"\nEXIT code={proc.returncode} scenario={scenario.name} seed={seed} variant={variant}\n")
                    log.flush()
                    if proc.returncode != 0:
                        print(proc.stdout[-4000:], flush=True)
                        raise SystemExit(proc.returncode)
    if not args.dry_run:
        summarize(output_root, scenarios)
        print(f"Wrote {output_root / 'robustness_summary.csv'}", flush=True)
        print(f"Wrote {output_root / 'convergence_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
