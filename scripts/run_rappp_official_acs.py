#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import read_json
from qdte.eval.external import write_external_evaluation

try:
    from path_defaults import rappp_root
except ModuleNotFoundError:
    from scripts.path_defaults import rappp_root

RAPPP_REPO = rappp_root()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _epsilon_from_rho_zcdp(rho: float, delta: float) -> float:
    if rho <= 0.0:
        raise ValueError("--rho-total must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))


def _rho_from_epsilon_delta(epsilon: float, delta: float) -> float:
    log_inv_delta = math.log(1.0 / delta)
    rho0 = -2.0 * math.sqrt(epsilon * log_inv_delta + log_inv_delta**2) + epsilon + 2.0 * log_inv_delta
    rho1 = 2.0 * math.sqrt(epsilon * log_inv_delta + log_inv_delta**2) + epsilon + 2.0 * log_inv_delta
    return float(rho0 if rho0 > 0.0 else rho1)


def _jax_metadata() -> dict[str, Any]:
    import jax

    return {
        "jax_version": getattr(jax, "__version__", "unknown"),
        "jax_default_backend": jax.default_backend() if hasattr(jax, "default_backend") else "unknown",
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _upstream_output_dir(
    rappp_root: Path,
    *,
    state: str,
    target: str,
    epsilon: float,
    top_q: int,
    dp_select_epochs: int,
    num_random_projections: int,
    seed: int,
) -> Path:
    return (
        rappp_root
        / "results"
        / "sync_data"
        / "RAP(Marginal&Halfspace)"
        / f"acs_{state}_{target}"
        / f"{epsilon:.2f}"
        / f"topq{top_q}_epochs{dp_select_epochs}_rp{num_random_projections}"
        / str(seed)
    )


def _run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official RAP++ ACS/Folktables and evaluate on a SAGE package.")
    parser.add_argument("--method", default="rappp_official_acs", choices=["rappp_official_acs"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rappp-root", type=Path, default=RAPPP_REPO)
    parser.add_argument("--state", default="CA")
    parser.add_argument("--target", default="income")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--rho-total", type=float)
    budget.add_argument("--upstream-epsilon", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--num-random-projections", type=int, default=200_000)
    parser.add_argument("--top-q", type=int, default=5)
    parser.add_argument("--dp-select-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--true-answers-cache", type=Path)
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--xla-preallocate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rappp_root = args.rappp_root.resolve()
    input_dir = args.input_dir.resolve()
    start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    real = np.load(input_dir / "real_encoded.npy")
    upstream_delta = 1.0 / float(real.shape[0]) ** 2
    if args.upstream_epsilon is not None:
        epsilon = float(args.upstream_epsilon)
        rho_total = _rho_from_epsilon_delta(epsilon, upstream_delta)
        budget_conversion = "upstream_epsilon_direct; rho computed from upstream RAP++ zCDP conversion"
    else:
        rho_total = float(args.rho_total)
        epsilon = _epsilon_from_rho_zcdp(rho_total, upstream_delta)
        budget_conversion = "epsilon=rho+2*sqrt(rho*log(1/delta)); upstream RAP++ converts back to zCDP"

    upstream_cmd = [
        sys.executable,
        "main.py",
        "--states",
        str(args.state),
        "--targets",
        str(args.target),
        "--algorithm",
        "RAP++",
        "--seed",
        str(int(args.seed)),
        "--epsilon",
        repr(float(epsilon)),
        "--k",
        str(int(args.k)),
        "--num_random_projections",
        str(int(args.num_random_projections)),
        "--top_q",
        str(int(args.top_q)),
        "--dp_select_epochs",
        str(int(args.dp_select_epochs)),
    ]
    env = os.environ.copy()
    if not bool(args.xla_preallocate):
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    metadata: dict[str, Any] = {
        "method": "rappp_official_acs",
        "source_repository": str(rappp_root),
        "source_commit": _git_commit(rappp_root),
        "wrapper_repository": str(ROOT),
        "wrapper_commit": _git_commit(ROOT),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "dataset": str(args.dataset),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "state": str(args.state),
        "target": str(args.target),
        "seed": int(args.seed),
        "rho_total": float(rho_total),
        "upstream_delta": float(upstream_delta),
        "upstream_epsilon": float(epsilon),
        "budget_conversion": budget_conversion,
        "k": int(args.k),
        "num_random_projections": int(args.num_random_projections),
        "top_q": int(args.top_q),
        "dp_select_epochs": int(args.dp_select_epochs),
        "n_real": int(real.shape[0]),
        "n_synthetic": None,
        "command": " ".join(sys.argv),
        "upstream_command": " ".join(upstream_cmd),
        "start_time": start_iso,
        "end_time": None,
        "runtime_seconds": None,
        "status": "running",
        "failure_reason": None,
        "notes": {
            "algorithm_scope": "official upstream RAP++ ACS/Folktables path with Marginal&Halfspace",
            "xla_preallocate": bool(args.xla_preallocate),
            **_jax_metadata(),
        },
    }
    _write_json(output_dir / "run_metadata.json", metadata)

    try:
        result = _run_command(upstream_cmd, cwd=rappp_root, env=env)
        (output_dir / "upstream_stdout.log").write_text(result.stdout)
        (output_dir / "upstream_stderr.log").write_text(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"upstream RAP++ failed with return code {result.returncode}")

        upstream_dir = _upstream_output_dir(
            rappp_root,
            state=str(args.state),
            target=str(args.target),
            epsilon=float(epsilon),
            top_q=int(args.top_q),
            dp_select_epochs=int(args.dp_select_epochs),
            num_random_projections=int(args.num_random_projections),
            seed=int(args.seed),
        )
        upstream_csv = upstream_dir / "synthetic.csv"
        upstream_runtime = upstream_dir / "runtime.txt"
        if not upstream_csv.exists():
            raise FileNotFoundError(f"upstream synthetic.csv not found: {upstream_csv}")

        shutil.copy2(upstream_csv, output_dir / "synthetic_raw.csv")
        if upstream_runtime.exists():
            shutil.copy2(upstream_runtime, output_dir / "upstream_runtime.txt")

        encode_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "encode_external_synthetic_csv.py"),
            "--input-dir",
            str(input_dir),
            "--csv",
            str(upstream_csv),
            "--output",
            str(output_dir / "synthetic_encoded.npy"),
            "--decoded-output",
            str(output_dir / "synthetic_decoded.csv"),
        ]
        encode_result = _run_command(encode_cmd, cwd=ROOT, env=env)
        (output_dir / "encode_stdout.log").write_text(encode_result.stdout)
        (output_dir / "encode_stderr.log").write_text(encode_result.stderr)
        if encode_result.returncode != 0:
            raise RuntimeError(f"encoding RAP++ synthetic CSV failed with return code {encode_result.returncode}")

        synthetic = np.load(output_dir / "synthetic_encoded.npy")
        metadata["n_synthetic"] = int(synthetic.shape[0])
        metadata["upstream_output_dir"] = str(upstream_dir)
        metadata["upstream_runtime_seconds"] = (
            float(upstream_runtime.read_text().strip()) if upstream_runtime.exists() else None
        )

        if not bool(args.no_evaluate):
            cache = args.true_answers_cache
            if cache is None:
                cache = input_dir / "true_answers_cache.npz"
            evaluation = write_external_evaluation(
                input_dir,
                output_dir / "synthetic_encoded.npy",
                output_dir / "evaluation.json",
                run_metadata_path=output_dir / "run_metadata.json",
                batch_size=int(args.batch_size),
                include_block_details=False,
                true_answers_cache_path=cache,
            )
            metadata["evaluation_summary"] = {
                key: float(evaluation[key])
                for key in [
                    "full_true_mae",
                    "full_true_rmse",
                    "full_true_avg_tvd",
                    "full_true_max_error",
                    "full_true_max_tvd",
                ]
                if key in evaluation
            }

        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        metadata["runtime_seconds"] = float(time.time() - start)
        _write_json(output_dir / "run_metadata.json", metadata)

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
