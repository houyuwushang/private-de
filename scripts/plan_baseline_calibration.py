#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from path_defaults import baseline_root, external_calibration_runs, external_inputs
except ModuleNotFoundError:
    from scripts.path_defaults import baseline_root, external_calibration_runs, external_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = baseline_root()
INPUT_ROOT = external_inputs()
CALIBRATION_ROOT = external_calibration_runs()
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_external_synthetic.py"


def _parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _rho_label(rho: float) -> str:
    return str(float(rho)).replace(".", "p")


def _privmrf_data_name(dataset: str, configured: str) -> str:
    if configured != "auto":
        return configured
    for prefix in ("adult", "acs", "br2000", "nltcs"):
        if dataset == prefix or dataset.startswith(f"{prefix}_"):
            return prefix
    return dataset


def _print_or_execute(cmd: list[str], execute: bool) -> None:
    print(" ".join(cmd))
    if execute:
        subprocess.run(cmd, check=True)


def _eval_cmd(dataset: str, run_dir: Path) -> list[str]:
    return [
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


def _gsd_cmd(args: argparse.Namespace, dataset: str, rho: float, seed: int, n_prime: int) -> tuple[str, Path, list[str]]:
    setting = (
        f"nprime{n_prime}_depth{args.gsd_tree_query_depth}_"
        f"ops{str(args.gsd_genetic_operators).replace(',', '-')}_stop{str(args.gsd_early_stop_threshold).replace('.', 'p')}"
    )
    run_dir = CALIBRATION_ROOT / "private_gsd" / dataset / f"rho{_rho_label(rho)}" / f"seed{seed}" / setting
    cmd = [
        "conda",
        "run",
        "-n",
        "baseline_gsd",
        "python",
        str(BASELINE_ROOT / "external_wrappers" / "run_private_gsd.py"),
        "--method",
        "gsd",
        "--dataset",
        dataset,
        "--input-dir",
        str(INPUT_ROOT / dataset),
        "--output-dir",
        str(run_dir),
        "--rho-total",
        str(float(rho)),
        "--delta",
        str(float(args.delta)),
        "--seed",
        str(int(seed)),
        "--n-syn",
        args.n_syn,
        "--n-prime",
        str(int(n_prime)),
        "--early-stop-threshold",
        str(float(args.gsd_early_stop_threshold)),
        "--tree-query-depth",
        str(int(args.gsd_tree_query_depth)),
        "--genetic-operators",
        str(args.gsd_genetic_operators),
    ]
    return setting, run_dir, cmd


def _parse_privmrf_setting(value: str) -> tuple[int, int]:
    if "x" not in value:
        raise ValueError(f"PrivMRF setting must be ROWSxITERS, got {value!r}")
    rows, iters = value.lower().split("x", 1)
    return int(rows), int(iters)


def _privmrf_cmd(
    args: argparse.Namespace,
    dataset: str,
    rho: float,
    seed: int,
    max_rows: int,
    estimation_iters: int,
) -> tuple[str, Path, list[str]]:
    setting = (
        f"rows{max_rows}_iters{estimation_iters}_"
        f"t{str(args.privmrf_entropy_descent_t).replace('.', 'p')}_init{args.privmrf_init_measure}"
    )
    run_dir = CALIBRATION_ROOT / "privmrf" / dataset / f"rho{_rho_label(rho)}" / f"seed{seed}" / setting
    cmd = [
        "conda",
        "run",
        "-n",
        "baseline_privmrf",
        "python",
        str(BASELINE_ROOT / "external_wrappers" / "run_privmrf.py"),
        "--method",
        "privmrf",
        "--dataset",
        dataset,
        "--input-dir",
        str(INPUT_ROOT / dataset),
        "--output-dir",
        str(run_dir),
        "--rho-total",
        str(float(rho)),
        "--delta",
        str(float(args.delta)),
        "--seed",
        str(int(seed)),
        "--n-syn",
        args.n_syn,
        "--max-train-rows",
        str(int(max_rows)),
        "--estimation-iters",
        str(int(estimation_iters)),
        "--print-interval",
        str(int(args.privmrf_print_interval)),
        "--entropy-descent-t",
        str(float(args.privmrf_entropy_descent_t)),
        "--privmrf-data-name",
        _privmrf_data_name(dataset, str(args.privmrf_data_name)),
        "--init-measure",
        str(int(args.privmrf_init_measure)),
        "--theta",
        str(float(args.privmrf_theta)),
        "--max-measure-attr-num",
        str(int(args.privmrf_max_measure_attr_num)),
        "--max-measure-attr-num-privbayes",
        str(int(args.privmrf_max_measure_attr_num_privbayes)),
        "--convergence-ratio",
        str(float(args.privmrf_convergence_ratio)),
        "--final-convergence-ratio",
        str(float(args.privmrf_final_convergence_ratio)),
    ]
    return setting, run_dir, cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute diagnostic baseline runtime/quality calibrations.")
    parser.add_argument("--methods", default="private_gsd,privmrf")
    parser.add_argument("--datasets", default="adult")
    parser.add_argument("--rhos", default="1.0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--delta", type=float, default=1e-9)
    parser.add_argument("--n-syn", default="same_as_real")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    parser.add_argument("--gsd-n-primes", default="64,256,1024")
    parser.add_argument("--gsd-tree-query-depth", type=int, default=1)
    parser.add_argument("--gsd-genetic-operators", default="mutate")
    parser.add_argument("--gsd-early-stop-threshold", type=float, default=0.2)

    parser.add_argument("--privmrf-settings", default="512x20,2048x20,4096x20")
    parser.add_argument("--privmrf-print-interval", type=int, default=5)
    parser.add_argument("--privmrf-entropy-descent-t", type=float, default=0.0)
    parser.add_argument(
        "--privmrf-data-name",
        default="auto",
        help="PrivMRF internal dataset name. Use 'auto' to map adult_sage_strong -> adult, etc.",
    )
    parser.add_argument("--privmrf-init-measure", type=int, default=0)
    parser.add_argument("--privmrf-theta", type=float, default=6.0)
    parser.add_argument("--privmrf-max-measure-attr-num", type=int, default=3)
    parser.add_argument("--privmrf-max-measure-attr-num-privbayes", type=int, default=3)
    parser.add_argument("--privmrf-convergence-ratio", type=float, default=5.0)
    parser.add_argument("--privmrf-final-convergence-ratio", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = _parse_csv(args.methods)
    datasets = _parse_csv(args.datasets)
    rhos = _parse_csv(args.rhos, float)
    seeds = _parse_csv(args.seeds, int)
    gsd_n_primes = _parse_csv(args.gsd_n_primes, int)
    privmrf_settings = [_parse_privmrf_setting(item) for item in _parse_csv(args.privmrf_settings)]

    for dataset in datasets:
        if not (INPUT_ROOT / dataset).exists():
            raise FileNotFoundError(f"Missing canonical input directory: {INPUT_ROOT / dataset}")
        for rho in rhos:
            for seed in seeds:
                if "private_gsd" in methods:
                    for n_prime in gsd_n_primes:
                        _, run_dir, cmd = _gsd_cmd(args, dataset, rho, seed, n_prime)
                        if args.skip_existing and (run_dir / "evaluation.json").exists():
                            continue
                        _print_or_execute(cmd, args.execute)
                        if args.evaluate:
                            _print_or_execute(_eval_cmd(dataset, run_dir), args.execute)
                if "privmrf" in methods:
                    for max_rows, estimation_iters in privmrf_settings:
                        _, run_dir, cmd = _privmrf_cmd(args, dataset, rho, seed, max_rows, estimation_iters)
                        if args.skip_existing and (run_dir / "evaluation.json").exists():
                            continue
                        _print_or_execute(cmd, args.execute)
                        if args.evaluate:
                            _print_or_execute(_eval_cmd(dataset, run_dir), args.execute)


if __name__ == "__main__":
    main()
