#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from path_defaults import external_inputs, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_inputs, external_runs

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = external_inputs()
RUNS_ROOT = external_runs() / "sage_ablation"
SAGE_WRAPPER = REPO_ROOT / "scripts" / "run_sage_external.py"
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_external_synthetic.py"


@dataclass(frozen=True)
class SageAblation:
    slug: str
    label: str
    overrides: tuple[str, ...]


ABLATIONS: dict[str, SageAblation] = {
    "no_projection": SageAblation(
        slug="no_projection",
        label="sage_no_projection",
        overrides=(
            "projection.project_partitions=false",
            "projection.clip_nonpartition=false",
            "projection.prefix_monotonicity=false",
            "projection.consistency.enabled=false",
        ),
    ),
    "unweighted_objective": SageAblation(
        slug="unweighted_objective",
        label="sage_unweighted_objective",
        overrides=("qdte.objective_weighting=unweighted",),
    ),
    "low_order_workload": SageAblation(
        slug="low_order_workload",
        label="sage_low_order_workload",
        overrides=(
            "workload.include_prefix=false",
            "workload.include_range=false",
            "workload.include_mixed=false",
            "workload.include_kway=false",
            "workload.include_kway_prefix=false",
            "workload.include_kway_range=false",
            "workload.include_kway_mixed=false",
            "workload.include_orthogonal_kway_mixed=false",
            "workload.include_halfspace=false",
        ),
    ),
}


def _parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _rho_label(rho: float) -> str:
    return str(float(rho)).replace(".", "p")


def _run_dir(ablation: SageAblation, dataset: str, rho: float, seed: int) -> Path:
    return RUNS_ROOT / ablation.slug / dataset / f"rho{_rho_label(rho)}" / f"seed{seed}"


def _sage_command(
    ablation: SageAblation,
    dataset: str,
    rho: float,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    input_dir = INPUT_ROOT / dataset
    cmd = [
        "conda",
        "run",
        "-n",
        "qdte",
        "python",
        str(SAGE_WRAPPER),
        "--method",
        "sage",
        "--method-label",
        ablation.label,
        "--dataset",
        dataset,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--rho-total",
        str(float(rho)),
        "--delta",
        str(float(args.delta)),
        "--seed",
        str(int(seed)),
        "--n-syn",
        str(args.n_syn),
        "--max-iters",
        str(int(args.max_iters)),
    ]
    for override in ablation.overrides:
        cmd.extend(["--override", override])
    return cmd


def _eval_command(dataset: str, run_dir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        "conda",
        "run",
        "-n",
        "qdte",
        "python",
        str(EVALUATOR),
        "--input-dir",
        str(INPUT_ROOT / dataset),
        "--synthetic",
        str(run_dir / "synthetic_encoded.npy"),
        "--output",
        str(run_dir / "evaluation.json"),
        "--no-block-details",
    ]
    if args.cache_true_answers:
        cmd.extend(["--true-answers-cache", str(INPUT_ROOT / dataset / "true_answers_cache.npz")])
    return cmd


def _print_or_execute(cmd: list[str], execute: bool) -> None:
    print(" ".join(cmd), flush=True)
    if execute:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run SAGE component ablations on canonical inputs.")
    parser.add_argument("--ablations", default="no_projection,unweighted_objective,low_order_workload")
    parser.add_argument("--datasets", default="adult_sage_strong,acs_sage_strong,br2000_sage_strong,nltcs_sage_strong")
    parser.add_argument("--rhos", default="1.0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--delta", type=float, default=1.0e-9)
    parser.add_argument("--n-syn", default="same_as_real")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--cache-true-answers", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_names = _parse_csv(args.ablations)
    datasets = _parse_csv(args.datasets)
    rhos = _parse_csv(args.rhos, float)
    seeds = _parse_csv(args.seeds, int)
    for name in ablation_names:
        if name not in ABLATIONS:
            raise ValueError(f"Unknown ablation {name!r}. Available: {sorted(ABLATIONS)}")
        ablation = ABLATIONS[name]
        for dataset in datasets:
            if not (INPUT_ROOT / dataset).exists():
                raise FileNotFoundError(f"Missing canonical input directory: {INPUT_ROOT / dataset}")
            for rho in rhos:
                for seed in seeds:
                    run_dir = _run_dir(ablation, dataset, rho, seed)
                    _print_or_execute(_sage_command(ablation, dataset, rho, seed, run_dir, args), args.execute)
                    if args.evaluate:
                        _print_or_execute(_eval_command(dataset, run_dir, args), args.execute)


if __name__ == "__main__":
    main()
