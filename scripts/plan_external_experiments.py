#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from path_defaults import baseline_root, external_inputs, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import baseline_root, external_inputs, external_runs

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = baseline_root()
INPUT_ROOT = external_inputs()
RUNS_ROOT = external_runs()
QdteEvaluator = REPO_ROOT / "scripts" / "evaluate_external_synthetic.py"


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    env: str
    wrapper: Path
    wrapper_method: str
    status: str = "ready"
    gpu_probe: str | None = None


METHODS: dict[str, MethodSpec] = {
    "sage": MethodSpec(
        slug="sage",
        env="qdte",
        wrapper=REPO_ROOT / "scripts" / "run_sage_external.py",
        wrapper_method="sage",
        gpu_probe="jax",
    ),
    "dpmm_mst": MethodSpec(
        slug="dpmm_mst",
        env="baseline_dpmm",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_dpmm.py",
        wrapper_method="mst",
    ),
    "dpmm_privbayes": MethodSpec(
        slug="dpmm_privbayes",
        env="baseline_dpmm",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_dpmm.py",
        wrapper_method="privbayes",
    ),
    "dpmm_aim": MethodSpec(
        slug="dpmm_aim",
        env="baseline_dpmm",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_dpmm.py",
        wrapper_method="aim",
    ),
    "datasynth_privbayes": MethodSpec(
        slug="datasynth_privbayes",
        env="baseline_datasynth",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_datasynthesizer.py",
        wrapper_method="privbayes",
    ),
    "private_pgm_mst": MethodSpec(
        slug="private_pgm_mst",
        env="baseline_mbi",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_private_pgm.py",
        wrapper_method="mst",
    ),
    "private_pgm_aim": MethodSpec(
        slug="private_pgm_aim",
        env="baseline_mbi",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_private_pgm.py",
        wrapper_method="aim",
    ),
    "private_gsd": MethodSpec(
        slug="private_gsd",
        env="baseline_gsd",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_private_gsd.py",
        wrapper_method="gsd",
    ),
    "private_gsd_stronger": MethodSpec(
        slug="private_gsd_stronger",
        env="baseline_gsd",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_private_gsd.py",
        wrapper_method="gsd",
    ),
    "private_gsd_gpu": MethodSpec(
        slug="private_gsd_gpu",
        env="gsd",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_private_gsd.py",
        wrapper_method="gsd",
        gpu_probe="jax",
    ),
    "private_gsd_gpu_1m_fulln_audit": MethodSpec(
        slug="private_gsd_gpu_1m_fulln_audit",
        env="gsd",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_private_gsd.py",
        wrapper_method="gsd",
        gpu_probe="jax",
    ),
    "rap": MethodSpec(
        slug="rap",
        env="tddpm",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_rap.py",
        wrapper_method="rap",
        gpu_probe="torch",
    ),
    "gem": MethodSpec(
        slug="gem",
        env="tddpm",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_gem.py",
        wrapper_method="gem",
        gpu_probe="torch",
    ),
    "privmrf": MethodSpec(
        slug="privmrf",
        env="baseline_privmrf",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_privmrf.py",
        wrapper_method="privmrf",
    ),
    "privmrf_gpu": MethodSpec(
        slug="privmrf_gpu",
        env="baseline_privmrf",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_privmrf.py",
        wrapper_method="privmrf",
        gpu_probe="cupy",
    ),
    "privsyn_unofficial": MethodSpec(
        slug="privsyn_unofficial",
        env="tddpm",
        wrapper=BASELINE_ROOT / "external_wrappers" / "run_privsyn.py",
        wrapper_method="privsyn",
    ),
}


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


def _paper_rap_num_marginals(dataset: str, configured: int) -> int:
    if int(configured) >= 0:
        return int(configured)
    return 364 if dataset.startswith("br2000") else 445


def _run_dir(method: str, dataset: str, rho: float, seed: int, phase: str, args: argparse.Namespace) -> Path:
    suffix = f"seed{seed}" if phase == "full" else f"seed{seed}_{phase}"
    if phase == "full" and method == "rap":
        num_marginals = _paper_rap_num_marginals(dataset, int(args.rap_num_marginals))
        suffix = f"seed{seed}_T{int(args.rap_T)}K{int(args.rap_K)}M{num_marginals}I{int(args.rap_max_iters)}"
    return RUNS_ROOT / method / dataset / f"rho{_rho_label(rho)}" / suffix


def _build_run_command(
    spec: MethodSpec,
    dataset: str,
    rho: float,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    input_dir = INPUT_ROOT / dataset
    n_iters = args.full_n_iters if args.phase == "full" else args.smoke_n_iters
    cmd = [
        "conda",
        "run",
        "-n",
        spec.env,
        "python",
        str(spec.wrapper),
        "--method",
        spec.wrapper_method,
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
        args.n_syn,
    ]
    if spec.wrapper.name == "run_dpmm.py":
        cmd.extend(["--n-iters", str(int(n_iters)), "--n-jobs", str(int(args.n_jobs))])
    if spec.wrapper.name == "run_datasynthesizer.py":
        cmd.extend(["--degree", str(int(args.datasynth_degree))])
    if spec.wrapper.name == "run_privsyn.py":
        consistency_iters = (
            args.privsyn_consistency_iterations
            if args.phase == "full"
            else args.smoke_privsyn_consistency_iterations
        )
        view_iters = args.privsyn_view_iterations if args.phase == "full" else args.smoke_privsyn_view_iterations
        gum_iters = args.privsyn_gum_iterations if args.phase == "full" else args.smoke_privsyn_gum_iterations
        cmd.extend(["--consistency-iterations", str(int(consistency_iters))])
        cmd.extend(["--view-iterations", str(int(view_iters))])
        cmd.extend(["--gum-iterations", str(int(gum_iters))])
    if spec.slug == "sage":
        max_iters = args.sage_max_iters if args.phase == "full" else args.smoke_sage_max_iters
        if max_iters is not None:
            cmd.extend(["--max-iters", str(int(max_iters))])
    if spec.wrapper_method == "aim" and args.phase != "full" and spec.slug != "private_pgm_aim":
        cmd.extend(["--rounds", str(int(args.smoke_aim_rounds))])
    if spec.slug == "private_pgm_aim":
        rounds = args.private_pgm_aim_rounds if args.phase == "full" else args.smoke_private_pgm_aim_rounds
        if rounds is not None:
            cmd.extend(["--rounds", str(int(rounds))])
        cmd.extend(["--max-iters", str(int(args.smoke_private_pgm_aim_max_iters if args.phase != "full" else args.private_pgm_aim_max_iters))])
        cmd.extend(["--max-model-size", str(float(args.private_pgm_aim_max_model_size))])
    if spec.wrapper.name == "run_private_gsd.py":
        n_prime = args.private_gsd_n_prime if args.phase == "full" else args.smoke_private_gsd_n_prime
        early_stop = args.private_gsd_early_stop_threshold if args.phase == "full" else args.smoke_private_gsd_early_stop_threshold
        tree_query_depth = args.private_gsd_tree_query_depth if args.phase == "full" else args.smoke_private_gsd_tree_query_depth
        genetic_operators = args.private_gsd_genetic_operators if args.phase == "full" else args.smoke_private_gsd_genetic_operators
        num_generations = args.private_gsd_num_generations if args.phase == "full" else args.smoke_private_gsd_num_generations
        stop_early_min_generation = (
            args.private_gsd_stop_early_min_generation
            if args.phase == "full"
            else args.smoke_private_gsd_stop_early_min_generation
        )
        if args.phase == "full" and spec.slug == "private_gsd_gpu_1m_fulln_audit":
            n_prime = "same_as_real"
            early_stop = 0.01
            tree_query_depth = 2
            genetic_operators = "mutate,swap,cross"
            num_generations = 1000000
            stop_early_min_generation = 1000000
        cmd.extend(["--n-prime", str(n_prime)])
        cmd.extend(["--early-stop-threshold", str(float(early_stop))])
        cmd.extend(["--tree-query-depth", str(int(tree_query_depth))])
        cmd.extend(["--genetic-operators", str(genetic_operators)])
        cmd.extend(["--method-label", str(spec.slug)])
        cmd.extend(["--conda-env-label", str(spec.env)])
        if num_generations is not None:
            cmd.extend(["--num-generations", str(int(num_generations))])
        if stop_early_min_generation is not None:
            cmd.extend(["--stop-early-min-generation", str(int(stop_early_min_generation))])
    if spec.slug == "rap":
        model_rows = args.rap_model_rows if args.phase == "full" else args.smoke_rap_model_rows
        T = args.rap_T if args.phase == "full" else args.smoke_rap_T
        K = args.rap_K if args.phase == "full" else args.smoke_rap_K
        max_iters = args.rap_max_iters if args.phase == "full" else args.smoke_rap_max_iters
        num_marginals = args.rap_num_marginals if args.phase == "full" else args.smoke_rap_num_marginals
        if args.phase == "full":
            num_marginals = _paper_rap_num_marginals(dataset, int(num_marginals))
        cmd.extend(["--model-rows", str(model_rows)])
        cmd.extend(["--degree", str(int(args.rap_degree))])
        cmd.extend(["--max-cells", str(int(args.rap_max_cells))])
        cmd.extend(["--num-marginals", str(int(num_marginals))])
        cmd.extend(["--T", str(int(T))])
        cmd.extend(["--K", str(int(K))])
        cmd.extend(["--lr", str(float(args.rap_lr))])
        cmd.extend(["--max-iters", str(int(max_iters))])
        cmd.extend(["--decode-mode", str(args.rap_decode_mode)])
    if spec.slug == "gem":
        T = args.gem_T if args.phase == "full" else args.smoke_gem_T
        num_marginals = args.gem_num_marginals if args.phase == "full" else args.smoke_gem_num_marginals
        dim = args.gem_dim if args.phase == "full" else args.smoke_gem_dim
        syndata_size = args.gem_syndata_size if args.phase == "full" else args.smoke_gem_syndata_size
        max_iters = args.gem_max_iters if args.phase == "full" else args.smoke_gem_max_iters
        max_idxs = args.gem_max_idxs if args.phase == "full" else args.smoke_gem_max_idxs
        cmd.extend(["--degree", str(int(args.gem_degree))])
        cmd.extend(["--max-cells", str(int(args.gem_max_cells))])
        cmd.extend(["--num-marginals", str(int(num_marginals))])
        cmd.extend(["--T", str(int(T))])
        cmd.extend(["--alpha", str(float(args.gem_alpha))])
        cmd.extend(["--dim", str(int(dim))])
        cmd.extend(["--syndata-size", str(int(syndata_size))])
        cmd.extend(["--lr", str(float(args.gem_lr))])
        cmd.extend(["--max-iters", str(int(max_iters))])
        cmd.extend(["--max-idxs", str(int(max_idxs))])
        cmd.extend(["--decode-mode", str(args.gem_decode_mode)])
        cmd.extend(["--latent-mode", str(args.gem_latent_mode)])
    if spec.wrapper.name == "run_privmrf.py":
        max_train_rows = args.privmrf_max_train_rows if args.phase == "full" else args.smoke_privmrf_max_train_rows
        estimation_iters = args.privmrf_estimation_iters if args.phase == "full" else args.smoke_privmrf_estimation_iters
        print_interval = args.privmrf_print_interval if args.phase == "full" else args.smoke_privmrf_print_interval
        entropy_t = args.privmrf_entropy_descent_t if args.phase == "full" else args.smoke_privmrf_entropy_descent_t
        data_name = _privmrf_data_name(dataset, str(args.privmrf_data_name))
        cmd.extend(["--max-train-rows", str(max_train_rows)])
        cmd.extend(["--estimation-iters", str(int(estimation_iters))])
        cmd.extend(["--print-interval", str(int(print_interval))])
        cmd.extend(["--entropy-descent-t", str(float(entropy_t))])
        cmd.extend(["--privmrf-data-name", str(data_name)])
        cmd.extend(["--method-label", str(spec.slug)])
        cmd.extend(["--conda-env-label", str(spec.env)])
        cmd.extend(["--init-measure", str(int(args.privmrf_init_measure))])
        cmd.extend(["--theta", str(float(args.privmrf_theta))])
        cmd.extend(["--max-measure-attr-num", str(int(args.privmrf_max_measure_attr_num))])
        cmd.extend(["--max-measure-attr-num-privbayes", str(int(args.privmrf_max_measure_attr_num_privbayes))])
        cmd.extend(["--convergence-ratio", str(float(args.privmrf_convergence_ratio))])
        cmd.extend(["--final-convergence-ratio", str(float(args.privmrf_final_convergence_ratio))])
    return cmd


def _build_eval_command(dataset: str, run_dir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        "conda",
        "run",
        "-n",
        "qdte",
        "python",
        str(QdteEvaluator),
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


def _build_gpu_preflight_command(spec: MethodSpec) -> list[str] | None:
    if spec.gpu_probe is None:
        return None
    if spec.gpu_probe == "jax":
        probe = (
            "import jax; "
            "devices=jax.devices(); "
            "print('jax_backend', jax.default_backend()); "
            "print('jax_devices', ','.join(str(d) for d in devices)); "
            "assert any(getattr(d, 'platform', '') == 'gpu' for d in devices), devices"
        )
    elif spec.gpu_probe == "torch":
        probe = (
            "import torch; "
            "print('torch_cuda_available', torch.cuda.is_available()); "
            "print('torch_device_count', torch.cuda.device_count()); "
            "assert torch.cuda.is_available() and torch.cuda.device_count() > 0"
        )
    elif spec.gpu_probe == "cupy":
        probe = (
            "import cupy; "
            "count=cupy.cuda.runtime.getDeviceCount(); "
            "print('cupy_device_count', count); "
            "assert count > 0"
        )
    else:
        raise ValueError(f"Unknown gpu_probe policy {spec.gpu_probe!r} for {spec.slug}")
    return ["conda", "run", "-n", spec.env, "python", "-c", probe]


def _gpu_preflight_commands(methods: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for method in methods:
        spec = METHODS[method]
        if spec.gpu_probe is None:
            continue
        key = (spec.env, spec.gpu_probe)
        if key in seen:
            continue
        seen.add(key)
        command = _build_gpu_preflight_command(spec)
        if command is not None:
            commands.append(command)
    return commands


def _print_or_execute(cmd: list[str], execute: bool) -> None:
    printable = shlex.join(cmd)
    print(printable)
    if execute:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute canonical external baseline experiments.")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--methods", default="dpmm_mst,dpmm_privbayes,dpmm_aim")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--rhos", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--delta", type=float, default=1e-9)
    parser.add_argument("--n-syn", default="same_as_real")
    parser.add_argument("--smoke-n-iters", type=int, default=10)
    parser.add_argument("--full-n-iters", type=int, default=1000)
    parser.add_argument("--datasynth-degree", type=int, default=2)
    parser.add_argument("--smoke-privsyn-consistency-iterations", type=int, default=2)
    parser.add_argument("--privsyn-consistency-iterations", type=int, default=5)
    parser.add_argument("--smoke-privsyn-view-iterations", type=int, default=5)
    parser.add_argument("--privsyn-view-iterations", type=int, default=100)
    parser.add_argument("--smoke-privsyn-gum-iterations", type=int, default=5)
    parser.add_argument("--privsyn-gum-iterations", type=int, default=50)
    parser.add_argument("--smoke-sage-max-iters", type=int, default=50)
    parser.add_argument("--sage-max-iters", type=int, default=5000)
    parser.add_argument("--smoke-aim-rounds", type=int, default=1)
    parser.add_argument("--smoke-private-pgm-aim-rounds", type=int, default=32)
    parser.add_argument("--smoke-private-pgm-aim-max-iters", type=int, default=10)
    parser.add_argument("--private-pgm-aim-rounds", type=int, default=220)
    parser.add_argument("--private-pgm-aim-max-iters", type=int, default=100)
    parser.add_argument("--private-pgm-aim-max-model-size", type=float, default=80.0)
    parser.add_argument("--smoke-private-gsd-n-prime", default="64")
    parser.add_argument("--private-gsd-n-prime", default="512")
    parser.add_argument("--smoke-private-gsd-early-stop-threshold", type=float, default=0.2)
    parser.add_argument("--private-gsd-early-stop-threshold", type=float, default=0.05)
    parser.add_argument("--smoke-private-gsd-tree-query-depth", type=int, default=1)
    parser.add_argument("--private-gsd-tree-query-depth", type=int, default=1)
    parser.add_argument("--smoke-private-gsd-genetic-operators", default="mutate")
    parser.add_argument("--private-gsd-genetic-operators", default="mutate")
    parser.add_argument("--smoke-private-gsd-num-generations", type=int, default=None)
    parser.add_argument("--private-gsd-num-generations", type=int, default=None)
    parser.add_argument("--smoke-private-gsd-stop-early-min-generation", type=int, default=None)
    parser.add_argument("--private-gsd-stop-early-min-generation", type=int, default=None)
    parser.add_argument("--smoke-rap-model-rows", default="128")
    parser.add_argument("--rap-model-rows", default="1000")
    parser.add_argument("--rap-degree", type=int, default=3)
    parser.add_argument("--rap-max-cells", type=int, default=10000)
    parser.add_argument("--smoke-rap-num-marginals", type=int, default=8)
    parser.add_argument(
        "--rap-num-marginals",
        type=int,
        default=-1,
        help="Full-run RAP marginals. -1 uses paper defaults: 445 except 364 for BR2000.",
    )
    parser.add_argument("--smoke-rap-T", type=int, default=1)
    parser.add_argument("--rap-T", type=int, default=30)
    parser.add_argument("--smoke-rap-K", type=int, default=1)
    parser.add_argument("--rap-K", type=int, default=30)
    parser.add_argument("--rap-lr", type=float, default=0.1)
    parser.add_argument("--smoke-rap-max-iters", type=int, default=50)
    parser.add_argument("--rap-max-iters", type=int, default=1000)
    parser.add_argument("--rap-decode-mode", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--gem-degree", type=int, default=3)
    parser.add_argument("--gem-max-cells", type=int, default=10000)
    parser.add_argument("--smoke-gem-num-marginals", type=int, default=8)
    parser.add_argument("--gem-num-marginals", type=int, default=286)
    parser.add_argument("--smoke-gem-T", type=int, default=2)
    parser.add_argument("--gem-T", type=int, default=30)
    parser.add_argument("--gem-alpha", type=float, default=0.67)
    parser.add_argument("--smoke-gem-dim", type=int, default=64)
    parser.add_argument("--gem-dim", type=int, default=512)
    parser.add_argument("--smoke-gem-syndata-size", type=int, default=128)
    parser.add_argument("--gem-syndata-size", type=int, default=1000)
    parser.add_argument("--gem-lr", type=float, default=1e-4)
    parser.add_argument("--smoke-gem-max-iters", type=int, default=10)
    parser.add_argument("--gem-max-iters", type=int, default=100)
    parser.add_argument("--smoke-gem-max-idxs", type=int, default=20)
    parser.add_argument("--gem-max-idxs", type=int, default=100)
    parser.add_argument("--gem-decode-mode", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--gem-latent-mode", choices=["cached", "fresh"], default="cached")
    parser.add_argument("--smoke-privmrf-max-train-rows", default="2048")
    parser.add_argument("--privmrf-max-train-rows", default="512")
    parser.add_argument("--smoke-privmrf-estimation-iters", type=int, default=20)
    parser.add_argument("--privmrf-estimation-iters", type=int, default=20)
    parser.add_argument("--smoke-privmrf-print-interval", type=int, default=5)
    parser.add_argument("--privmrf-print-interval", type=int, default=5)
    parser.add_argument("--smoke-privmrf-entropy-descent-t", type=float, default=0.0)
    parser.add_argument("--privmrf-entropy-descent-t", type=float, default=0.8)
    parser.add_argument("--privmrf-init-measure", type=int, default=0)
    parser.add_argument("--privmrf-theta", type=float, default=6.0)
    parser.add_argument("--privmrf-max-measure-attr-num", type=int, default=3)
    parser.add_argument("--privmrf-max-measure-attr-num-privbayes", type=int, default=3)
    parser.add_argument("--privmrf-convergence-ratio", type=float, default=5.0)
    parser.add_argument("--privmrf-final-convergence-ratio", type=float, default=5.0)
    parser.add_argument(
        "--privmrf-data-name",
        default="auto",
        help="PrivMRF internal dataset name. Use 'auto' to map adult_sage_strong -> adult, etc.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--gpu-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print and, under --execute, run one GPU visibility probe per selected GPU-required environment.",
    )
    parser.add_argument("--cache-true-answers", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_default = "adult" if args.phase == "smoke" else "adult,acs,br2000,nltcs"
    rhos_default = "1.0" if args.phase == "smoke" else "0.25,0.5,1.0,2.0"
    seeds_default = "0" if args.phase == "smoke" else "0,1,2"
    methods = _parse_csv(args.methods)
    datasets = _parse_csv(args.datasets or datasets_default)
    rhos = _parse_csv(args.rhos or rhos_default, float)
    seeds = _parse_csv(args.seeds or seeds_default, int)

    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown or not-yet-wrapped method {method!r}. Available: {sorted(METHODS)}")

    if args.gpu_preflight:
        for preflight_cmd in _gpu_preflight_commands(methods):
            _print_or_execute(preflight_cmd, args.execute)

    for method in methods:
        spec = METHODS[method]
        for dataset in datasets:
            if not (INPUT_ROOT / dataset).exists():
                raise FileNotFoundError(f"Missing canonical input directory: {INPUT_ROOT / dataset}")
            for rho in rhos:
                for seed in seeds:
                    run_dir = _run_dir(method, dataset, rho, seed, args.phase, args)
                    run_cmd = _build_run_command(spec, dataset, rho, seed, run_dir, args)
                    _print_or_execute(run_cmd, args.execute)
                    if args.evaluate:
                        eval_cmd = _build_eval_command(dataset, run_dir, args)
                        _print_or_execute(eval_cmd, args.execute)


if __name__ == "__main__":
    main()
