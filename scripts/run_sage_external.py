#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, parse_scalar, set_nested
from qdte.dataio import read_json
from qdte.evolution.engine import run_qdte


DEFAULT_CONFIGS = {
    "adult": ROOT / "configs" / "adult_qdte.yaml",
    "acs": ROOT / "configs" / "acs_qdte.yaml",
    "br2000": ROOT / "configs" / "br2000_qdte.yaml",
    "nltcs": ROOT / "configs" / "nltcs_qdte.yaml",
}


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


def _schema_columns(schema: dict[str, Any]) -> list[dict[str, Any]]:
    columns = list(schema["columns"])
    names = [str(col["name"]) for col in columns]
    if len(names) != len(set(names)):
        raise ValueError("schema column names must be unique for SAGE external runs")
    return columns


def _parse_n_syn(value: str, n_real: int) -> str | int:
    if value == "same_as_real":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--n-syn must be positive or same_as_real")
    return parsed


def _default_config_path(dataset: str, input_dir: Path, explicit_config: Path | None) -> Path:
    if explicit_config is not None:
        return explicit_config
    metadata_path = input_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        candidate = Path(str(metadata.get("config_path", "")))
        if candidate.exists():
            return candidate
    if dataset not in DEFAULT_CONFIGS:
        raise ValueError(f"No default SAGE config is registered for dataset {dataset!r}")
    return DEFAULT_CONFIGS[dataset]


def _apply_override_pairs(config: dict[str, Any], override_pairs: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for item in override_pairs:
        if "=" not in item:
            raise ValueError(f"--override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        set_nested(resolved, key, parse_scalar(value))
    return resolved


def _prepare_config(args: argparse.Namespace, n_real: int) -> tuple[dict[str, Any], Path]:
    config_path = _default_config_path(str(args.dataset), args.input_dir, args.config)
    config = load_yaml(config_path)
    config = _apply_override_pairs(config, list(args.override or []))
    config = copy.deepcopy(config)

    n_syn = _parse_n_syn(str(args.n_syn), n_real)
    set_nested(config, "run.dataset_name", str(args.dataset))
    set_nested(config, "run.input_csv", str(args.input_dir / "raw.csv"))
    set_nested(config, "run.output_dir", str(args.output_dir))
    set_nested(config, "run.seed", int(args.seed))
    set_nested(config, "privacy.mode", "dp")
    set_nested(config, "privacy.rho_total", float(args.rho_total))
    set_nested(config, "privacy.delta", float(args.delta))
    set_nested(config, "init.N_syn", n_syn)
    if args.max_iters is not None:
        set_nested(config, "qdte.max_iters", int(args.max_iters))

    # Keep exact true answers out of the active SAGE run. The shared external
    # evaluator computes true metrics after generation as an offline step.
    set_nested(config, "evaluation.compute_true_query_error", False)
    set_nested(config, "evaluation.compute_heldout_query_error", False)
    set_nested(config, "evaluation.downstream_ml", False)
    set_nested(config, "evaluation.save_synthetic_csv", bool(args.save_synthetic_csv))
    set_nested(config, "runtime.xla_preallocate", bool(args.xla_preallocate))
    set_nested(config, "runtime.log_measurement_groups", False)
    return config, config_path


def _validate_synthetic(output_dir: Path, input_dir: Path) -> np.ndarray:
    synthetic_path = output_dir / "synthetic_encoded.npy"
    if not synthetic_path.exists():
        raise FileNotFoundError(f"SAGE did not write {synthetic_path}")
    schema = read_json(input_dir / "schema.json")
    columns = _schema_columns(schema)
    cardinalities = [int(col["cardinality"]) for col in columns]
    synthetic = np.load(synthetic_path)
    if synthetic.ndim != 2 or synthetic.shape[1] != len(columns):
        raise ValueError(
            f"synthetic_encoded.npy shape {synthetic.shape} is incompatible with schema width {len(columns)}"
        )
    if not np.issubdtype(synthetic.dtype, np.integer):
        raise ValueError(f"synthetic_encoded.npy must be integer encoded, got {synthetic.dtype}")
    for idx, cardinality in enumerate(cardinalities):
        values = synthetic[:, idx]
        if values.size and (int(values.min()) < 0 or int(values.max()) >= cardinality):
            raise ValueError(
                f"synthetic column {idx} violates [0,{cardinality}): "
                f"min={int(values.min())}, max={int(values.max())}"
            )
    return synthetic


def _metadata_base(args: argparse.Namespace) -> dict[str, Any]:
    input_metadata = read_json(args.input_dir / "metadata.json")
    return {
        "method": str(args.method_label),
        "wrapper_method": "sage",
        "source_repository": str(ROOT),
        "commit": _git_commit(ROOT),
        "conda_env": "qdte",
        "dataset": str(args.dataset),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "rho_total": float(args.rho_total),
        "delta": float(args.delta),
        "epsilon_delta": None,
        "budget_conversion": "native_zcdp_rho_total",
        "schema_hash": input_metadata.get("schema_sha256"),
        "queries_hash": input_metadata.get("queries_sha256"),
        "command": " ".join(sys.argv),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (args.input_dir / "raw.csv").exists():
        raise FileNotFoundError(f"Missing canonical raw.csv: {args.input_dir / 'raw.csv'}")
    real = np.load(args.input_dir / "real_encoded.npy")
    start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    config, config_path = _prepare_config(args, int(real.shape[0]))

    if not bool(config.get("runtime", {}).get("xla_preallocate", True)):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    final_metrics = run_qdte(config)
    synthetic = _validate_synthetic(args.output_dir, args.input_dir)
    end = time.time()
    runtime_path = args.output_dir / "runtime.json"
    runtime_metadata: dict[str, Any] = {}
    if runtime_path.exists():
        try:
            runtime_metadata = json.loads(runtime_path.read_text())
        except Exception:
            runtime_metadata = {}
    gpu_devices = runtime_metadata.get("gpu_devices") or []
    gpu_device_count = final_metrics.get("gpu_device_count", len(gpu_devices))
    score_backend = final_metrics.get("score_backend") or config.get("qdte", {}).get("score_backend")

    metadata = _metadata_base(args)
    metadata.update(
        {
            "config_path": str(config_path),
            "start_time": start_iso,
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "runtime_seconds": float(end - start),
            "status": "completed",
            "failure_reason": None,
            "n_real": int(real.shape[0]),
            "n_synthetic": int(synthetic.shape[0]),
            "num_columns": int(real.shape[1]),
            "epsilon_delta": final_metrics.get("epsilon_delta"),
            "rho_spent": final_metrics.get("rho_spent"),
            "device": config.get("run", {}).get("device"),
            "score_backend": score_backend,
            "gpu_device_count": gpu_device_count,
            "gpu_devices": gpu_devices,
            "notes": {
                "config_path": str(config_path),
                "max_iters": config.get("qdte", {}).get("max_iters"),
                "n_syn": config.get("init", {}).get("N_syn"),
                "true_metrics_disabled_during_run": True,
                "offline_external_evaluator_required": True,
                "xla_preallocate": bool(config.get("runtime", {}).get("xla_preallocate", True)),
                "log_measurement_groups": bool(config.get("runtime", {}).get("log_measurement_groups", True)),
                "device": config.get("run", {}).get("device"),
                "score_backend": score_backend,
                "gpu_device_count": gpu_device_count,
                "gpu_devices": gpu_devices,
            },
        }
    )
    _write_json(args.output_dir / "run_metadata.json", metadata)
    (args.output_dir / "stdout.log").write_text(
        f"completed sage in {metadata['runtime_seconds']:.3f}s\n",
        encoding="utf-8",
    )
    (args.output_dir / "stderr.log").write_text("", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAGE/QDTE on canonical external experiment inputs.")
    parser.add_argument("--method", required=True, choices=["sage"])
    parser.add_argument(
        "--method-label",
        default="sage",
        help="Method name written to run metadata; useful for SAGE ablations that reuse the same wrapper.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rho-total", required=True, type=float)
    parser.add_argument("--delta", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-syn", default="same_as_real")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--save-synthetic-csv", action="store_true")
    parser.add_argument("--xla-preallocate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metadata = run(args)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (args.output_dir / "stderr.log").write_text(tb, encoding="utf-8")
        metadata = _metadata_base(args)
        metadata.update(
            {
                "runtime_seconds": None,
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "notes": {},
            }
        )
        _write_json(args.output_dir / "run_metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
