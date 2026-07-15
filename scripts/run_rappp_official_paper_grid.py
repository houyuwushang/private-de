#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from path_defaults import external_runs, rappp_root
except ModuleNotFoundError:
    from scripts.path_defaults import external_runs, rappp_root

ROOT = Path(__file__).resolve().parents[1]
RAPPP_ROOT = rappp_root()
DEFAULT_OUTPUT_ROOT = external_runs() / "rappp_official_paper_grid"


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _probe_jax_devices(*, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, jax; "
            "devices = jax.devices(); "
            "print(json.dumps({"
            "'jax_version': jax.__version__, "
            "'jax_local_device_count': jax.local_device_count(), "
            "'jax_devices': [str(device) for device in devices], "
            "'jax_device_platforms': sorted({device.platform for device in devices})"
            "}, sort_keys=True))"
        ),
    ]
    try:
        result = _run_command(cmd, cwd=cwd, env=env)
        if result.returncode != 0:
            return {
                "probe_status": "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        data = json.loads(result.stdout)
        data["probe_status"] = "ok"
        return data
    except Exception as exc:
        return {"probe_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}


def _probe_nvidia_smi(*, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = _run_command(cmd, cwd=cwd, env=env)
        if result.returncode != 0:
            return {
                "probe_status": "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        rows = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index, name, memory_used, memory_total, utilization_gpu = [item.strip() for item in line.split(",", 4)]
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_used_mib": int(memory_used),
                    "memory_total_mib": int(memory_total),
                    "utilization_gpu_pct": int(utilization_gpu),
                }
            )
        return {"probe_status": "ok", "gpus": rows}
    except Exception as exc:
        return {"probe_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}


def _run_one(args: argparse.Namespace, state: str, target: str) -> dict[str, Any]:
    label = f"acs_{state}_{target}"
    run_dir = args.output_root / label / f"eps{str(float(args.upstream_epsilon)).replace('.', 'p')}" / f"seed{args.seed}"
    metadata_path = run_dir / "run_metadata.json"
    metrics_path = run_dir / "paper_metrics" / "rappp_paper_metrics.json"
    if bool(args.resume) and metrics_path.exists() and metadata_path.exists():
        metadata = _read_json(metadata_path)
        if metadata.get("status") == "completed":
            return metadata

    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("RAPPP_FULL_STATS_CHUNK_SIZE", str(int(args.chunk_size)))
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    jax_device_probe = _probe_jax_devices(cwd=args.rappp_root, env=env)
    nvidia_smi_probe = _probe_nvidia_smi(cwd=args.rappp_root, env=env)

    upstream_cmd = [
        sys.executable,
        "main.py",
        "--states",
        state,
        "--targets",
        target,
        "--algorithm",
        "RAP++",
        "--seed",
        str(int(args.seed)),
        "--epsilon",
        repr(float(args.upstream_epsilon)),
        "--k",
        str(int(args.k)),
        "--num_random_projections",
        str(int(args.num_random_projections)),
        "--top_q",
        str(int(args.top_q)),
        "--dp_select_epochs",
        str(int(args.dp_select_epochs)),
    ]
    upstream_dir = _upstream_output_dir(
        args.rappp_root,
        state=state,
        target=target,
        epsilon=float(args.upstream_epsilon),
        top_q=int(args.top_q),
        dp_select_epochs=int(args.dp_select_epochs),
        num_random_projections=int(args.num_random_projections),
        seed=int(args.seed),
    )
    synthetic_csv = upstream_dir / "synthetic.csv"
    start = time.time()
    metadata: dict[str, Any] = {
        "method": "rappp_official_paper_grid",
        "state": state,
        "target": target,
        "dataset_name": label,
        "seed": int(args.seed),
        "upstream_epsilon": float(args.upstream_epsilon),
        "k": int(args.k),
        "num_random_projections": int(args.num_random_projections),
        "top_q": int(args.top_q),
        "dp_select_epochs": int(args.dp_select_epochs),
        "chunk_size": int(args.chunk_size),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "xla_python_client_preallocate": env.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "rappp_full_stats_chunk_size": env.get("RAPPP_FULL_STATS_CHUNK_SIZE"),
        "jax_device_probe": jax_device_probe,
        "nvidia_smi_probe": nvidia_smi_probe,
        "rappp_root": str(args.rappp_root),
        "rappp_commit": _git_commit(args.rappp_root),
        "wrapper_repository": str(ROOT),
        "wrapper_commit": _git_commit(ROOT),
        "run_dir": str(run_dir),
        "upstream_output_dir": str(upstream_dir),
        "synthetic_csv": str(synthetic_csv),
        "upstream_command": " ".join(upstream_cmd),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "end_time": None,
        "runtime_seconds": None,
        "status": "running",
        "failure_reason": None,
    }
    _write_json(metadata_path, metadata)

    try:
        if not synthetic_csv.exists() or bool(args.force):
            result = _run_command(upstream_cmd, cwd=args.rappp_root, env=env)
            (run_dir / "upstream_stdout.log").write_text(result.stdout)
            (run_dir / "upstream_stderr.log").write_text(result.stderr)
            if result.returncode != 0:
                raise RuntimeError(f"upstream RAP++ failed with return code {result.returncode}")
        else:
            (run_dir / "upstream_stdout.log").write_text("skipped: synthetic.csv already exists\n")
            (run_dir / "upstream_stderr.log").write_text("")

        if not synthetic_csv.exists():
            raise FileNotFoundError(f"missing upstream synthetic.csv: {synthetic_csv}")

        metrics_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_rappp_paper_metrics.py"),
            "--rappp-root",
            str(args.rappp_root),
            "--state",
            state,
            "--target",
            target,
            "--seed",
            str(int(args.seed)),
            "--synthetic-csv",
            str(synthetic_csv),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir / "paper_metrics"),
            "--prefix-queries",
            str(int(args.prefix_queries)),
        ]
        if bool(args.skip_ml):
            metrics_cmd.append("--skip-ml")
        metrics_result = _run_command(metrics_cmd, cwd=ROOT, env=env)
        (run_dir / "paper_metrics_stdout.log").write_text(metrics_result.stdout)
        (run_dir / "paper_metrics_stderr.log").write_text(metrics_result.stderr)
        if metrics_result.returncode != 0:
            raise RuntimeError(f"paper metric evaluation failed with return code {metrics_result.returncode}")

        metadata["paper_metrics_path"] = str(metrics_path)
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        metadata["runtime_seconds"] = float(time.time() - start)
        _write_json(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAP++ upstream-default reproduction over the ACS paper grid.")
    parser.add_argument("--rappp-root", type=Path, default=RAPPP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--states", default="NY,CA,TX,FL,PA")
    parser.add_argument("--targets", default="income,travel,coverage,employment,mobility")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upstream-epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--num-random-projections", type=int, default=200_000)
    parser.add_argument("--top-q", type=int, default=5)
    parser.add_argument("--dp-select-epochs", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--prefix-queries", type=int, default=20_000)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.rappp_root = args.rappp_root.resolve()
    args.output_root = args.output_root.resolve()
    states = _parse_csv(args.states)
    targets = _parse_csv(args.targets)
    tasks = [(state, target) for state in states for target in targets]
    if args.limit is not None:
        tasks = tasks[: int(args.limit)]

    summaries = []
    for index, (state, target) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] RAP++ official grid: {state} {target}", flush=True)
        metadata = _run_one(args, state, target)
        summaries.append(metadata)
        print(json.dumps({"state": state, "target": target, "status": metadata["status"], "runtime_seconds": metadata["runtime_seconds"]}, sort_keys=True), flush=True)

    summary_path = args.output_root / f"grid_seed{args.seed}_summary.json"
    _write_json(summary_path, {"tasks": summaries})
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
