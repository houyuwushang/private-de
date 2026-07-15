#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAGE_RUNNER = ROOT / "scripts" / "run_sage_external.py"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _stage_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    method_label: str,
    overlays: list[Path],
    max_iters: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SAGE_RUNNER),
        "--method",
        "sage",
        "--method-label",
        method_label,
        "--dataset",
        str(args.dataset),
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(output_dir),
        "--rho-total",
        str(args.rho_total),
        "--delta",
        str(args.delta),
        "--seed",
        str(args.seed),
        "--n-syn",
        str(args.n_syn),
        "--config",
        str(args.config),
    ]
    for overlay in overlays:
        command.extend(["--overlay", str(overlay)])
    for override in list(args.override or []):
        command.extend(["--override", str(override)])
    if max_iters is not None:
        command.extend(["--max-iters", str(max_iters)])
    if args.xla_preallocate:
        command.append("--xla-preallocate")
    return command


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _require_completed_stage(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"stage did not write {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError(f"stage is not completed: {metadata_path}")
    return metadata


def _read_selected_iteration(selection_dir: Path) -> tuple[int, dict[str, Any]]:
    checkpoint_path = selection_dir / "fission_checkpoints.json"
    if not checkpoint_path.exists():
        raise RuntimeError(f"selection stage did not write {checkpoint_path}")
    checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    selected_iteration = int(checkpoints["selected_iteration"])
    terminal_iteration = int(checkpoints["terminal_iteration"])
    if selected_iteration < 0 or selected_iteration > terminal_iteration:
        raise RuntimeError(
            "invalid fission selection: "
            f"selected={selected_iteration}, terminal={terminal_iteration}"
        )
    return selected_iteration, checkpoints


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    selection_dir = output_dir / "selection"
    refit_dir = output_dir / "refit"
    status_path = output_dir / "pipeline_metadata.json"
    started_at = time.time()

    selection_overlays = list(args.base_overlay or []) + [Path(args.fission_overlay)]
    refit_overlays = list(args.base_overlay or [])
    selection_command = _stage_command(
        args,
        output_dir=selection_dir,
        method_label=f"{args.method_label}-selection",
        overlays=selection_overlays,
    )

    running = {
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "status": "running",
        "selection_dir": str(selection_dir),
        "refit_dir": str(refit_dir),
        "selection_command": selection_command,
    }
    _write_json(status_path, running)

    _run_command(selection_command)
    selection_metadata = _require_completed_stage(selection_dir)
    selected_iteration, checkpoints = _read_selected_iteration(selection_dir)

    refit_command = _stage_command(
        args,
        output_dir=refit_dir,
        method_label=str(args.method_label),
        overlays=refit_overlays,
        max_iters=selected_iteration,
    )
    running.update(
        {
            "selected_iteration": selected_iteration,
            "terminal_iteration": int(checkpoints["terminal_iteration"]),
            "selection_rule": str(checkpoints["selection_rule"]),
            "refit_command": refit_command,
        }
    )
    _write_json(status_path, running)

    _run_command(refit_command)
    refit_metadata = _require_completed_stage(refit_dir)

    completed = {
        **running,
        "status": "completed",
        "runtime_seconds": time.time() - started_at,
        "selection_runtime_seconds": selection_metadata.get("runtime_seconds"),
        "refit_runtime_seconds": refit_metadata.get("runtime_seconds"),
    }
    _write_json(status_path, completed)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select QDTE search complexity by Gaussian fission, then refit full released measurements."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rho-total", required=True, type=float)
    parser.add_argument("--delta", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-syn", default="same_as_real")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method-label", default="QDTE-Structured-FissionRefit-v1")
    parser.add_argument("--base-overlay", action="append", type=Path, default=[])
    parser.add_argument("--fission-overlay", required=True, type=Path)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--xla-preallocate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metadata = run_pipeline(args)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except Exception as exc:
        status_path = Path(args.output_dir) / "pipeline_metadata.json"
        existing: dict[str, Any] = {}
        if status_path.exists():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        existing.update(
            {
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(status_path, existing)
        raise


if __name__ == "__main__":
    main()
