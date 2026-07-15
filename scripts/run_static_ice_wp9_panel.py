#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from scripts.run_coverage_refinement_wp8a import sha256_file
from scripts.run_static_ice_wp9_cell import (
    DATASETS,
    DELTA_DP,
    EPSILONS,
    METHOD_ID,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    SEEDS,
)


CELL_RUNNER = ROOT / "scripts" / "run_static_ice_wp9_cell.py"
EVALUATOR = ROOT / "scripts" / "evaluate_static_ice_wp9_panel.py"


@dataclass(frozen=True)
class ConfirmationCell:
    dataset: str
    epsilon: float
    seed: int

    @property
    def declared_dataset(self) -> str:
        return f"{self.dataset}_sage_strong"

    @property
    def relative_path(self) -> Path:
        return Path(self.dataset) / f"epsilon_{self.epsilon:g}" / f"seed_{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "declared_dataset": self.declared_dataset,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "relative_path": self.relative_path.as_posix(),
        }


def expected_cells() -> tuple[ConfirmationCell, ...]:
    return tuple(
        ConfirmationCell(dataset, epsilon, seed)
        for dataset in DATASETS
        for epsilon in EPSILONS
        for seed in SEEDS
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _verify_record(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"artifact size changed: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"artifact hash changed: {path}")
    return path


def _plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_panel_plan(
    *,
    input_root: Path,
    config_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    cells = []
    for cell in expected_cells():
        input_dir = input_root.resolve() / cell.declared_dataset
        config = config_root.resolve() / f"{cell.dataset}_sage_strong.yaml"
        cells.append(
            {
                **cell.to_dict(),
                "input_dir": str(input_dir),
                "config": _file_record(config),
                "public_schema": _file_record(input_dir / "schema.json"),
                "public_metadata": _file_record(input_dir / "metadata.json"),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "blind_static_ice_confirmation_plan",
        "protocol": _file_record(PROTOCOL_PATH),
        "cell_runner": _file_record(CELL_RUNNER),
        "evaluator": _file_record(EVALUATOR),
        "matrix": {
            "datasets": list(DATASETS),
            "epsilons": list(EPSILONS),
            "seeds": list(SEEDS),
            "delta": DELTA_DP,
            "num_cells": len(expected_cells()),
        },
        "offline_evaluator_batch_size": int(batch_size),
        "cells": cells,
        "true_utility_evaluated": False,
    }


def _same_plan(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    if existing != expected:
        raise RuntimeError("WP9 panel plan changed; use a new output root")


def validate_cell_output(
    output_dir: Path,
    cell: ConfirmationCell,
) -> dict[str, Any]:
    manifest_path = output_dir / "mechanism_manifest.json"
    status_path = output_dir / "cell_status.json"
    manifest = _read_json(manifest_path)
    status = _read_json(status_path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "sealed_blind_static_ice_cell",
        "dataset": cell.dataset,
        "declared_dataset": cell.declared_dataset,
        "epsilon": cell.epsilon,
        "seed": cell.seed,
        "true_utility_evaluated": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"WP9 cell {cell} violates {key}: {manifest.get(key)!r}")
    if status.get("status") != "completed" or status.get("true_utility_evaluated") is not False:
        raise RuntimeError(f"WP9 cell {cell} status is not a blind completion")
    if manifest.get("mechanism_gate", {}).get("passed") is not True:
        raise RuntimeError(f"WP9 cell {cell} failed its mechanism gate")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"WP9 cell {cell} has no artifact records")
    for record in artifacts.values():
        _verify_record(record)
    return {
        **cell.to_dict(),
        "mechanism_manifest": _file_record(manifest_path),
        "cell_status": _file_record(status_path),
        "synthetic": artifacts["synthetic"],
        "measurement": artifacts["measurement"],
        "runtime": artifacts["runtime"],
        "metrics": artifacts["metrics"],
        "privacy": manifest["privacy"],
        "mechanism_gate": manifest["mechanism_gate"],
    }


def _status(
    *,
    plan: dict[str, Any],
    completed: list[dict[str, Any]],
    active: ConfirmationCell | None,
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": _plan_hash(plan),
        "completed_cells": len(completed),
        "total_cells": len(expected_cells()),
        "active_cell": active.to_dict() if active is not None else None,
        "true_utility_evaluated": False,
    }


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = build_panel_plan(
        input_root=args.input_root,
        config_root=args.config_root,
        batch_size=int(args.batch_size),
    )
    plan_path = output_root / "panel_plan.json"
    if plan_path.exists():
        _same_plan(_read_json(plan_path), plan)
    else:
        write_json(plan, plan_path)

    plan_cells = {
        (str(value["dataset"]), float(value["epsilon"]), int(value["seed"])): value
        for value in plan["cells"]
    }
    completed: list[dict[str, Any]] = []
    for cell in expected_cells():
        cell_dir = output_root / cell.relative_path
        if (cell_dir / "mechanism_manifest.json").is_file():
            completed.append(validate_cell_output(cell_dir, cell))
            continue
        if cell_dir.exists() and any(cell_dir.iterdir()):
            raise RuntimeError(f"partial WP9 cell requires archival: {cell_dir}")
        write_json(
            _status(plan=plan, completed=completed, active=cell),
            output_root / "panel_status.json",
        )
        record = plan_cells[(cell.dataset, cell.epsilon, cell.seed)]
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "qdte",
            "python",
            str(CELL_RUNNER),
            "--dataset",
            cell.dataset,
            "--epsilon",
            str(cell.epsilon),
            "--seed",
            str(cell.seed),
            "--input-dir",
            str(record["input_dir"]),
            "--config",
            str(record["config"]["path"]),
            "--output-dir",
            str(cell_dir),
        ]
        environment = dict(os.environ)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        completed.append(validate_cell_output(cell_dir, cell))
        write_json(
            _status(plan=plan, completed=completed, active=None),
            output_root / "panel_status.json",
        )

    sealed = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "sealed_blind_static_ice_confirmation_panel",
        "panel_plan": _file_record(plan_path),
        "num_cells": len(completed),
        "cells": completed,
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": len(completed) == len(expected_cells()),
    }
    write_json(sealed, output_root / "sealed_panel_manifest.json")
    write_json(
        _status(plan=plan, completed=completed, active=None),
        output_root / "panel_status.json",
    )
    return sealed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and seal the blind Static-ICE-Exact WP9 confirmation panel."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("external_inputs"),
    )
    parser.add_argument("--config-root", type=Path, default=ROOT / "configs")
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    result = run_panel(parse_args())
    print(
        json.dumps(
            {
                "status": "sealed",
                "num_cells": result["num_cells"],
                "offline_evaluation_authorized": result[
                    "offline_evaluation_authorized"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
