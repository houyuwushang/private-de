#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, parse_scalar, set_nested
from qdte.evolution.engine import run_qdte
from qdte.measurement.public_artifact import verify_public_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic rows from a sealed public DP transcript only."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply a dotted generation config override; may be repeated.",
    )
    return parser.parse_args()


def _prepare_generation_config(
    config: dict[str, Any],
    transcript_root: Path,
    output_dir: Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    verified = verify_public_transcript(transcript_root)
    resolved = copy.deepcopy(config)
    for raw in overrides or []:
        if "=" not in raw:
            raise ValueError(f"Override must use KEY=VALUE syntax: {raw!r}")
        key, value = raw.split("=", 1)
        set_nested(resolved, key, parse_scalar(value))

    run_cfg = resolved.setdefault("run", {})
    run_cfg.pop("input_csv", None)
    run_cfg["output_dir"] = str(output_dir)
    run_cfg["transcript_only_generation"] = True

    preprocess_cfg = resolved.setdefault("preprocess", {})
    preprocess_cfg["public_schema_json"] = str(verified.root / "schema.json")

    privacy_cfg = resolved.setdefault("privacy", {})
    privacy_cfg.update(
        {
            "mode": "dp",
            "dp_release_mode": True,
            "public_row_count": True,
            "public_n_rows": int(verified.measurements.num_rows),
            "adjacency": "add_remove",
            "rho_total": float(verified.measurements.rho_total),
            "delta": float(verified.measurements.delta),
        }
    )

    measurement_cfg = resolved.setdefault("measurement", {})
    measurement_cfg.pop("artifact_dir", None)
    measurement_cfg["reuse_from"] = str(verified.root)
    resolved.setdefault("workload", {})["reuse_from_measurement"] = True

    init_cfg = resolved.setdefault("init", {})
    init_cfg.pop("encoded_npy", None)
    evaluation_cfg = resolved.setdefault("evaluation", {})
    evaluation_cfg.update(
        {
            "compute_true_query_error": False,
            "compute_heldout_query_error": False,
            "downstream_ml": False,
        }
    )
    oracle_bias = evaluation_cfg.get("oracle_projection_bias")
    if isinstance(oracle_bias, dict):
        oracle_bias["enabled"] = False
    return resolved


def main() -> None:
    args = parse_args()
    config = _prepare_generation_config(
        load_yaml(args.config),
        args.transcript,
        args.output_dir,
        args.override,
    )
    metrics = run_qdte(config)
    print(
        "Generated from sealed public transcript: "
        f"{args.output_dir.resolve()} | rows={metrics['num_rows_synthetic']} | "
        f"final_measured_loss={metrics['final_measured_loss']:.12g}"
    )


if __name__ == "__main__":
    main()
