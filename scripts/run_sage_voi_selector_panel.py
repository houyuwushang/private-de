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
from scripts.run_sage_voi_selector_pilot import ARMS, PROTOCOL_ID


EVALUATION_PROTOCOL = (
    ROOT / "docs" / "SAGE_QDTE_ICE_WP7_FORMAL_EVALUATION_PROTOCOL_20260714.md"
)
RUNNER = ROOT / "scripts" / "run_sage_voi_selector_pilot.py"
STAGE_ITERS = 5000
EPSILONS = (0.1, 0.3)
SEEDS = (0, 1, 2)
MECHANISM_SOURCES = (
    RUNNER,
    ROOT / "scripts" / "run_static_ice_qdte_pilot.py",
    ROOT / "qdte" / "privacy" / "accountant.py",
    ROOT / "qdte" / "privacy" / "exponential.py",
    ROOT / "qdte" / "selection" / "voi.py",
    ROOT / "qdte" / "measurement" / "adaptive_interactions.py",
    ROOT / "qdte" / "measurement" / "factorization.py",
    ROOT / "qdte" / "measurement" / "interaction_adapter.py",
    ROOT / "qdte" / "queries" / "orthogonal.py",
    ROOT / "qdte" / "queries" / "partitions.py",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: Path
    input_dir: Path


@dataclass(frozen=True)
class PanelCell:
    dataset: str
    epsilon: float
    seed: int

    @property
    def relative_path(self) -> Path:
        return Path(self.dataset) / f"epsilon_{self.epsilon:g}" / f"seed_{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "relative_path": self.relative_path.as_posix(),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_cells() -> tuple[PanelCell, ...]:
    return tuple(
        PanelCell(dataset, epsilon, seed)
        for dataset in ("acs", "adult")
        for epsilon in EPSILONS
        for seed in SEEDS
    )


def dataset_specs(input_root: Path) -> dict[str, DatasetSpec]:
    base = input_root.resolve()
    return {
        "acs": DatasetSpec(
            name="acs_sage_strong",
            config=ROOT / "configs" / "acs_sage_strong.yaml",
            input_dir=base / "acs_sage_strong",
        ),
        "adult": DatasetSpec(
            name="adult_sage_strong",
            config=ROOT / "configs" / "adult_sage_strong.yaml",
            input_dir=base / "adult_sage_strong",
        ),
    }


def _public_file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def build_panel_plan(input_root: Path) -> dict[str, Any]:
    specs = dataset_specs(input_root)
    return {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "blind_formal_generation_plan",
        "stage_iters": STAGE_ITERS,
        "epsilons": list(EPSILONS),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "cells": [cell.to_dict() for cell in expected_cells()],
        "runner": _public_file_record(RUNNER),
        "mechanism_sources": {
            path.relative_to(ROOT).as_posix(): _public_file_record(path)
            for path in MECHANISM_SOURCES
        },
        "evaluation_protocol": _public_file_record(EVALUATION_PROTOCOL),
        "datasets": {
            key: {
                "declared_name": spec.name,
                "config": _public_file_record(spec.config),
                "public_schema": _public_file_record(spec.input_dir / "schema.json"),
            }
            for key, spec in specs.items()
        },
        "private_input_hashes_excluded": True,
        "true_utility_evaluated": False,
    }


def _assert_same_plan(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    if existing != expected:
        raise RuntimeError(
            "panel_plan.json does not match the frozen runner/config/protocol; "
            "use a new output root"
        )


def validate_completed_cell(
    cell_dir: Path,
    cell: PanelCell,
    spec: DatasetSpec,
) -> dict[str, Any]:
    status_path = cell_dir / "run_status.json"
    summary_path = cell_dir / "summary.json"
    if not status_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"incomplete formal cell: {cell_dir}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise RuntimeError(f"formal cell is not completed: {cell_dir}")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "protocol_mode": "formal",
        "dataset": spec.name,
        "epsilon": cell.epsilon,
        "delta": 1.0e-9,
        "seed": cell.seed,
        "stage_iters": STAGE_ITERS,
        "true_utility_evaluated": False,
        "private_input_hashes_excluded": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(
                f"formal cell {cell_dir} violates frozen field {key}: "
                f"{summary.get(key)!r} != {value!r}"
            )
    if set(summary.get("arms", {})) != set(ARMS):
        raise RuntimeError(f"formal cell has the wrong arm set: {cell_dir}")

    arm_records: dict[str, Any] = {}
    for arm in ARMS:
        manifest_path = cell_dir / "arms" / arm / "arm_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_role") != "deployable_dp_transcript_and_postprocessing":
            raise RuntimeError(f"arm manifest has an invalid role: {manifest_path}")
        synthetic = Path(str(manifest["final_synthetic_path"]))
        if not synthetic.is_file():
            raise FileNotFoundError(synthetic)
        arm_records[arm] = {
            "arm_manifest": _public_file_record(manifest_path),
            "final_synthetic": _public_file_record(synthetic),
            "selected_actions": manifest["selected_actions"],
            "rho_spent": manifest["privacy"]["rho_spent"],
            "epsilon_from_actual_spend": manifest["privacy"][
                "epsilon_from_actual_spend"
            ],
        }
    return {
        **cell.to_dict(),
        "summary": _public_file_record(summary_path),
        "arms": arm_records,
    }


def _status_payload(
    plan: dict[str, Any],
    completed: list[dict[str, Any]],
    active: PanelCell | None,
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "completed_cells": len(completed),
        "total_cells": len(expected_cells()),
        "active_cell": active.to_dict() if active is not None else None,
        "true_utility_evaluated": False,
    }


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = build_panel_plan(args.input_root)
    plan_path = output_root / "panel_plan.json"
    if plan_path.exists():
        _assert_same_plan(json.loads(plan_path.read_text(encoding="utf-8")), plan)
    else:
        write_json(plan, plan_path)

    specs = dataset_specs(args.input_root)
    completed: list[dict[str, Any]] = []
    for cell in expected_cells():
        spec = specs[cell.dataset]
        cell_dir = output_root / cell.relative_path
        if (cell_dir / "run_status.json").is_file():
            completed.append(validate_completed_cell(cell_dir, cell, spec))
            continue
        if cell_dir.exists() and any(cell_dir.iterdir()):
            raise RuntimeError(
                f"partial/failed cell requires manual archival before retry: {cell_dir}"
            )
        write_json(
            _status_payload(plan, completed, cell),
            output_root / "panel_status.json",
        )
        command = [
            sys.executable,
            str(RUNNER),
            "--config",
            str(spec.config),
            "--input-dir",
            str(spec.input_dir),
            "--output-dir",
            str(cell_dir),
            "--protocol-mode",
            "formal",
            "--epsilon",
            str(cell.epsilon),
            "--delta",
            "1e-9",
            "--seed",
            str(cell.seed),
            "--stage-iters",
            str(STAGE_ITERS),
            "--max-pair-cells",
            "20000",
        ]
        environment = dict(os.environ)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        completed.append(validate_completed_cell(cell_dir, cell, spec))
        write_json(
            _status_payload(plan, completed, None),
            output_root / "panel_status.json",
        )

    sealed = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "sealed_blind_formal_panel",
        "panel_plan": _public_file_record(plan_path),
        "num_cells": len(completed),
        "cells": completed,
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": len(completed) == len(expected_cells()),
    }
    write_json(sealed, output_root / "sealed_panel_manifest.json")
    write_json(
        _status_payload(plan, completed, None),
        output_root / "panel_status.json",
    )
    return sealed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/resume the frozen blind WP7 selector panel."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("external_inputs"),
    )
    return parser.parse_args()


def main() -> int:
    sealed = run_panel(parse_args())
    print(
        json.dumps(
            {
                "status": "sealed",
                "num_cells": sealed["num_cells"],
                "offline_evaluation_authorized": sealed[
                    "offline_evaluation_authorized"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
