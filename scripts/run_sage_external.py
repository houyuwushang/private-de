#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
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
from qdte.privacy.accountant import zcdp_epsilon


DEFAULT_CONFIGS = {
    "adult": ROOT / "configs" / "adult_qdte.yaml",
    "acs": ROOT / "configs" / "acs_qdte.yaml",
    "br2000": ROOT / "configs" / "br2000_qdte.yaml",
    "nltcs": ROOT / "configs" / "nltcs_qdte.yaml",
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_hash(root: Path) -> str:
    paths = sorted((root / "qdte").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    digest = hashlib.sha256(b"sage-qdte-static-source-tree-v1\0")
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": commit,
            "worktree_dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "worktree_dirty": None, "provenance_error": str(error)}


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required static SAGE artifact {name}: {path}")
        artifacts[name] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    return artifacts


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


def _parse_n_syn(value: str, n_real: int) -> int:
    if value == "same_as_real":
        return int(n_real)
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


def _merge_config_overlay(config: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(resolved.get(key), dict):
            resolved[key] = _merge_config_overlay(resolved[key], value)
        else:
            resolved[key] = copy.deepcopy(value)
    return resolved


def _prepare_config(args: argparse.Namespace, n_real: int) -> tuple[dict[str, Any], Path]:
    config_path = _default_config_path(str(args.dataset), args.input_dir, args.config)
    config = load_yaml(config_path)
    for overlay_path in list(getattr(args, "overlay", []) or []):
        config = _merge_config_overlay(config, load_yaml(Path(overlay_path)))
    config = _apply_override_pairs(config, list(args.override or []))
    config = copy.deepcopy(config)

    n_syn = _parse_n_syn(str(args.n_syn), n_real)
    set_nested(config, "run.dataset_name", str(args.dataset))
    set_nested(config, "run.input_csv", str(args.input_dir / "raw.csv"))
    set_nested(config, "run.output_dir", str(args.output_dir))
    set_nested(config, "run.seed", int(args.seed))
    set_nested(config, "privacy.mode", "dp")
    set_nested(config, "privacy.dp_release_mode", True)
    set_nested(config, "privacy.public_row_count", True)
    set_nested(config, "privacy.public_n_rows", int(n_real))
    set_nested(config, "privacy.adjacency", "add_remove")
    set_nested(config, "privacy.rho_total", float(args.rho_total))
    set_nested(config, "privacy.delta", float(args.delta))
    set_nested(config, "preprocess.public_schema_json", str(args.input_dir / "schema.json"))
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
    if not (args.input_dir / "schema.json").exists():
        raise FileNotFoundError(f"Missing public schema.json: {args.input_dir / 'schema.json'}")
    input_metadata = read_json(args.input_dir / "metadata.json")
    n_real = int(input_metadata.get("n_rows", 0))
    if n_real <= 0:
        raise ValueError("External input metadata must declare a positive public n_rows")
    start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    config, config_path = _prepare_config(args, n_real)

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
            "config_overlays": [str(path) for path in list(getattr(args, "overlay", []) or [])],
            "start_time": start_iso,
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "runtime_seconds": float(end - start),
            "status": "completed",
            "failure_reason": None,
            "n_real": int(n_real),
            "n_synthetic": int(synthetic.shape[0]),
            "num_columns": int(synthetic.shape[1]),
            "epsilon_delta": final_metrics.get("epsilon_delta"),
            "rho_spent": final_metrics.get("rho_spent"),
            "device": config.get("run", {}).get("device"),
            "score_backend": score_backend,
            "gpu_device_count": gpu_device_count,
            "gpu_devices": gpu_devices,
            "notes": {
                "config_path": str(config_path),
                "config_overlays": [str(path) for path in list(getattr(args, "overlay", []) or [])],
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
    protocol_id = getattr(args, "protocol_id", None)
    if protocol_id is not None:
        input_metadata = read_json(args.input_dir / "metadata.json")
        artifacts = _artifact_hashes(
            {
                "resolved_config": args.output_dir / "config_resolved.yaml",
                "schema": args.output_dir / "schema.json",
                "query_catalogue": args.output_dir / "queries.json",
                "measurement": args.output_dir / "measurements.json",
                "synthetic": args.output_dir / "synthetic_encoded.npy",
                "generator_metrics": args.output_dir / "metrics_final.json",
                "generator_timeseries": args.output_dir / "metrics_timeseries.csv",
                "runtime": args.output_dir / "runtime.json",
                "logs": args.output_dir / "logs.txt",
                "run_metadata": args.output_dir / "run_metadata.json",
                "stdout": args.output_dir / "stdout.log",
                "stderr": args.output_dir / "stderr.log",
            }
        )
        expected_schema_hash = input_metadata.get("schema_sha256")
        expected_queries_hash = input_metadata.get("queries_sha256")
        measurement_payload = read_json(args.output_dir / "measurements.json")
        privacy_ledger = measurement_payload.get("privacy_ledger", {})
        rho_spent = float(final_metrics.get("rho_spent", float("nan")))
        epsilon_reported = float(final_metrics.get("epsilon_delta", float("nan")))
        rho_tolerance = 1.0e-12 * max(1.0, abs(float(args.rho_total)))
        checks = {
            "mode_is_dp": str(config.get("privacy", {}).get("mode")) == "dp",
            "dp_release_mode_enabled": config.get("privacy", {}).get("dp_release_mode") is True,
            "public_row_count_declared": config.get("privacy", {}).get("public_row_count") is True,
            "public_n_rows_matches_metadata": int(
                config.get("privacy", {}).get("public_n_rows", 0)
            )
            == int(n_real),
            "adjacency_is_add_remove": str(config.get("privacy", {}).get("adjacency"))
            == "add_remove",
            "rho_matches": float(config.get("privacy", {}).get("rho_total"))
            == float(args.rho_total),
            "delta_matches": float(config.get("privacy", {}).get("delta")) == float(args.delta),
            "actual_spend_within_declared_rho": bool(
                np.isfinite(rho_spent)
                and rho_spent <= float(args.rho_total) + rho_tolerance
            ),
            "epsilon_uses_actual_spend": bool(
                np.isfinite(epsilon_reported)
                and np.isclose(
                    epsilon_reported,
                    zcdp_epsilon(rho_spent, float(args.delta)),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            ),
            "measurement_ledger_matches_actual_spend": bool(
                isinstance(privacy_ledger, dict)
                and privacy_ledger.get("accounting") == "zcdp_actual_spend_v1"
                and privacy_ledger.get("adjacency") == "add_remove"
                and np.isclose(
                    float(privacy_ledger.get("rho_spent", float("nan"))),
                    rho_spent,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            ),
            "true_metrics_disabled_during_generation": not any(
                bool(config.get("evaluation", {}).get(key, False))
                for key in ("compute_true_query_error", "compute_heldout_query_error", "downstream_ml")
            ),
            "objective_is_variance": str(config.get("qdte", {}).get("objective_weighting"))
            == "variance",
            "transport_is_atom_flow": str(config.get("qdte", {}).get("transport_mode"))
            == "atom_flow",
            "schema_matches_canonical_input": expected_schema_hash is None
            or artifacts["schema"]["sha256"] == expected_schema_hash,
            "queries_match_canonical_input": expected_queries_hash is None
            or artifacts["query_catalogue"]["sha256"] == expected_queries_hash,
            "synthetic_row_count_matches_public_request": int(synthetic.shape[0])
            == int(config.get("init", {}).get("N_syn", 0)),
        }
        manifest = {
            "protocol_id": str(protocol_id),
            "paper_evidence_candidate": True,
            "paper_evidence_qualified": bool(all(checks.values())),
            "method": {
                "label": str(args.method_label),
                "wrapper": "sage",
                "measurement_schedule": "public_static_all_workload_groups",
                "projection_profile": "P1_lightweight",
                "generator_profile": "qdte_standard",
                "max_iters": int(config.get("qdte", {}).get("max_iters", 0)),
            },
            "dataset": str(args.dataset),
            "seed": int(args.seed),
            "privacy": {
                "mode": "dp",
                "adjacency": "add_remove_one",
                "public_n_rows": int(n_real),
                "rho_total": float(args.rho_total),
                "rho_spent": rho_spent,
                "delta": float(args.delta),
                "epsilon_recomputed": final_metrics.get("epsilon_delta"),
            },
            "canonical_input": {
                "path": str(args.input_dir),
                "raw_sha256": input_metadata.get("raw_sha256"),
                "schema_sha256": expected_schema_hash,
                "queries_sha256": expected_queries_hash,
            },
            "checks": checks,
            "artifacts": artifacts,
            "source_tree_sha256": _source_tree_hash(ROOT),
            "git": _git_revision(ROOT),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "configured_device": config.get("run", {}).get("device"),
            },
        }
        _write_json(args.output_dir / "sage_static_manifest.json", manifest)
        if not manifest["paper_evidence_qualified"]:
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise RuntimeError(f"Static SAGE evidence checks failed: {failed}")
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
    parser.add_argument("--overlay", action="append", type=Path, default=[])
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--save-synthetic-csv", action="store_true")
    parser.add_argument("--xla-preallocate", action="store_true")
    parser.add_argument(
        "--protocol-id",
        help="Write and enforce a paper-evidence manifest under this frozen protocol ID.",
    )
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
