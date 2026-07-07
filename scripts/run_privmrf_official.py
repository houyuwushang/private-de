#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from path_defaults import external_runs, privmrf_root
except ModuleNotFoundError:
    from scripts.path_defaults import external_runs, privmrf_root

ROOT = Path(__file__).resolve().parents[1]
PRIVMRF_ROOT = privmrf_root()
DEFAULT_OUTPUT_ROOT = external_runs() / "privmrf_official"


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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


def _label(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PrivMRF through its upstream public experiment API.")
    parser.add_argument("--privmrf-root", type=Path, default=PRIVMRF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conda-env", default="baseline_privmrf")
    parser.add_argument("--datasets", default="nltcs,acs,adult,br2000")
    parser.add_argument("--epsilons", default="0.1,0.2,0.4,0.8,1.6,3.2")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--marginal-num", type=int, default=300)
    parser.add_argument("--classifier-num", type=int, default=25)
    parser.add_argument("--task", choices=("TVD", "SVM"), default="TVD")
    parser.add_argument("--exp-name")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.privmrf_root = args.privmrf_root.resolve()
    args.output_root = args.output_root.resolve()
    datasets = _parse_csv(args.datasets)
    epsilons = _parse_float_csv(args.epsilons)
    if not datasets:
        raise ValueError("at least one dataset is required")
    if not epsilons:
        raise ValueError("at least one epsilon is required")

    exp_name = args.exp_name
    if not exp_name:
        dataset_label = "_".join(datasets)
        eps_label = "_".join(_label(eps) for eps in epsilons)
        exp_name = f"official_{args.task.lower()}_{dataset_label}_eps{eps_label}_r{args.repeat}_m{args.marginal_num}"

    run_dir = args.output_root / exp_name
    metadata_path = run_dir / "run_metadata.json"
    result_name = f"{exp_name}_{args.task}.json"
    upstream_result_path = args.privmrf_root / "result" / result_name
    copied_result_path = run_dir / result_name
    if copied_result_path.exists() and metadata_path.exists() and not bool(args.force):
        metadata = _read_json(metadata_path)
        if metadata.get("status") == "completed":
            print(f"already completed: {run_dir}")
            return

    run_dir.mkdir(parents=True, exist_ok=True)
    for relative in ("temp", "result", "out", "exp_data"):
        (args.privmrf_root / relative).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    code = "\n".join(
        [
            "import json",
            "from exp.evaluate import run_experiment",
            f"datasets = json.loads({json.dumps(json.dumps(datasets))})",
            f"epsilons = json.loads({json.dumps(json.dumps(epsilons))})",
            "run_experiment(",
            "    datasets,",
            "    ['PrivMRF'],",
            f"    {exp_name!r},",
            "    epsilon_list=epsilons,",
            f"    task={args.task!r},",
            f"    repeat={int(args.repeat)},",
            f"    marginal_num={int(args.marginal_num)},",
            f"    classifier_num={int(args.classifier_num)},",
            ")",
        ]
    )
    cmd = ["conda", "run", "-n", args.conda_env, "python", "-c", code]
    metadata: dict[str, Any] = {
        "method": "privmrf_official",
        "task": args.task,
        "datasets": datasets,
        "epsilons": epsilons,
        "repeat": int(args.repeat),
        "marginal_num": int(args.marginal_num),
        "classifier_num": int(args.classifier_num),
        "conda_env": args.conda_env,
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "privmrf_root": str(args.privmrf_root),
        "privmrf_commit": _git_commit(args.privmrf_root),
        "wrapper_repository": str(ROOT),
        "wrapper_commit": _git_commit(ROOT),
        "run_dir": str(run_dir),
        "upstream_result_path": str(upstream_result_path),
        "copied_result_path": str(copied_result_path),
        "command": " ".join(cmd),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "end_time": None,
        "runtime_seconds": None,
        "status": "running",
        "failure_reason": None,
    }
    start = time.time()
    _write_json(metadata_path, metadata)
    try:
        result = _run_command(cmd, cwd=args.privmrf_root, env=env)
        (run_dir / "stdout.log").write_text(result.stdout)
        (run_dir / "stderr.log").write_text(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"PrivMRF official command failed with return code {result.returncode}")
        if upstream_result_path.exists():
            shutil.copy2(upstream_result_path, copied_result_path)
            metadata["status"] = "completed"
        else:
            raise FileNotFoundError(f"missing expected result: {upstream_result_path}")
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        metadata["runtime_seconds"] = float(time.time() - start)
        _write_json(metadata_path, metadata)
    print(json.dumps({"status": metadata["status"], "run_dir": str(run_dir), "runtime_seconds": metadata["runtime_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
