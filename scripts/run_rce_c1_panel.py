#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from scripts.run_coverage_refinement_wp8a import sha256_file
from scripts.run_rce_c1_cell import (
    DATASETS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    EPSILONS,
    FORMAL_CANDIDATES_PER_ITERATION,
    FORMAL_ITERATIONS,
    METHOD_ID,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    SEEDS,
)


CELL_RUNNER = ROOT / "scripts" / "run_rce_c1_cell.py"
EVALUATOR = ROOT / "scripts" / "evaluate_rce_c1_panel.py"
IMPLEMENTATION_FILES = (
    ROOT / "qdte" / "evolution" / "engine.py",
    ROOT / "qdte" / "evolution" / "rce_scoring.py",
    ROOT / "qdte" / "evolution" / "transport.py",
    ROOT / "qdte" / "rce" / "confidence_set.py",
    ROOT / "qdte" / "rce" / "controller.py",
    ROOT / "qdte" / "rce" / "dual.py",
    ROOT / "qdte" / "rce" / "integer.py",
    ROOT / "qdte" / "rce" / "prior.py",
    ROOT / "qdte" / "rce" / "relaxed.py",
)


@dataclass(frozen=True)
class RCECell:
    dataset: str
    epsilon: float
    seed: int

    @property
    def relative_path(self) -> Path:
        return Path(self.dataset) / f"epsilon_{self.epsilon:g}" / f"seed_{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "declared_dataset": f"{self.dataset}_sage_strong",
            "epsilon": self.epsilon,
            "seed": self.seed,
            "relative_path": self.relative_path.as_posix(),
        }


def expected_cells() -> tuple[RCECell, ...]:
    return tuple(
        RCECell(dataset, epsilon, seed)
        for dataset in DATASETS
        for epsilon in EPSILONS
        for seed in SEEDS
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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
        raise RuntimeError(f"Sealed artifact size changed: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"Sealed artifact hash changed: {path}")
    return path


def _plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_panel_plan(source_root: Path, batch_size: int) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    cells = []
    for cell in expected_cells():
        source_manifest = source_root.resolve() / cell.relative_path / "mechanism_manifest.json"
        cells.append({**cell.to_dict(), "source_manifest": _file_record(source_manifest)})
    return {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "blind_same_transcript_rce_c1_plan",
        "protocol": _file_record(PROTOCOL_PATH),
        "cell_runner": _file_record(CELL_RUNNER),
        "evaluator": _file_record(EVALUATOR),
        "implementation": [_file_record(path) for path in IMPLEMENTATION_FILES],
        "matrix": {
            "datasets": list(DATASETS),
            "epsilons": list(EPSILONS),
            "seeds": list(SEEDS),
            "iterations": FORMAL_ITERATIONS,
            "candidates_per_iteration": FORMAL_CANDIDATES_PER_ITERATION,
            "num_cells": len(expected_cells()),
        },
        "offline_evaluator_batch_size": int(batch_size),
        "cells": cells,
        "true_utility_evaluated": False,
    }


def prepare_panel(output_root: Path, source_root: Path, batch_size: int) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    expected = build_panel_plan(source_root, batch_size)
    plan_path = output_root / "panel_plan.json"
    if plan_path.is_file():
        if _read_json(plan_path) != expected:
            raise RuntimeError("RCE C1 panel plan changed; use a new output root")
    else:
        write_json(expected, plan_path)
    write_json(
        {
            "protocol_id": PROTOCOL_ID,
            "plan_sha256": _plan_hash(expected),
            "completed_cells": 0,
            "total_cells": len(expected_cells()),
            "true_utility_evaluated": False,
        },
        output_root / "panel_status.json",
    )
    return expected


def validate_cell(output_root: Path, cell: RCECell) -> dict[str, Any]:
    cell_root = output_root / cell.relative_path
    manifest_path = cell_root / "mechanism_manifest.json"
    status_path = cell_root / "cell_status.json"
    manifest = _read_json(manifest_path)
    status = _read_json(status_path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "sealed_same_transcript_rce_c1_cell",
        "dataset": cell.dataset,
        "epsilon": cell.epsilon,
        "seed": cell.seed,
        "iterations": FORMAL_ITERATIONS,
        "true_utility_evaluated": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"RCE C1 cell {cell} violates {key}: {manifest.get(key)!r}")
    if status.get("status") != "completed" or status.get("formal") is not True:
        raise RuntimeError(f"RCE C1 cell {cell} is not a formal completion")
    if manifest.get("mechanism_gate", {}).get("passed") is not True:
        raise RuntimeError(f"RCE C1 cell {cell} failed its mechanism gate")
    for record in manifest.get("artifacts", {}).values():
        _verify_record(record)
    return {
        **cell.to_dict(),
        "mechanism_manifest": _file_record(manifest_path),
        "cell_status": _file_record(status_path),
        "synthetic": manifest["artifacts"]["synthetic"],
        "metrics": manifest["artifacts"]["metrics"],
        "runtime": manifest["artifacts"]["runtime"],
        "source_control": manifest["source_control"],
        "mechanism_gate": manifest["mechanism_gate"],
    }


def seal_panel(output_root: Path, source_root: Path, batch_size: int) -> dict[str, Any]:
    output_root = output_root.resolve()
    plan_path = output_root / "panel_plan.json"
    expected_plan = build_panel_plan(source_root, batch_size)
    if _read_json(plan_path) != expected_plan:
        raise RuntimeError("RCE C1 panel plan or frozen code changed")
    completed = [validate_cell(output_root, cell) for cell in expected_cells()]
    sealed = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "sealed_blind_same_transcript_rce_c1_panel",
        "panel_plan": _file_record(plan_path),
        "num_cells": len(completed),
        "cells": completed,
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": len(completed) == len(expected_cells()),
    }
    write_json(sealed, output_root / "sealed_panel_manifest.json")
    write_json(
        {
            "protocol_id": PROTOCOL_ID,
            "plan_sha256": _plan_hash(expected_plan),
            "completed_cells": len(completed),
            "total_cells": len(expected_cells()),
            "true_utility_evaluated": False,
        },
        output_root / "panel_status.json",
    )
    return sealed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or seal the blind RCE C1 panel")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = (
        seal_panel(args.output_root, args.source_root, args.batch_size)
        if args.seal
        else prepare_panel(args.output_root, args.source_root, args.batch_size)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
