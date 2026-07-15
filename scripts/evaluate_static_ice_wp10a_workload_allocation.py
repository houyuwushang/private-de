#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdte.dataio import write_json
from qdte.eval.external import evaluate_external_synthetic
from scripts.evaluate_sage_voi_selector_panel import PRIMARY_METRICS, TAIL_METRICS
from scripts.run_coverage_refinement_wp8a import sha256_file


PROTOCOL_ID = "SAGE-QDTE-ICE-WP10A-WORKLOAD-ALLOCATION-20260715-v1"
CANDIDATE_METHOD_ID = "SAGE-QDTE-Static-ICE-WorkloadL2-v1"
EVALUATOR_ID = "SAGE-QDTE-ICE-WP10A-EVALUATOR-20260715-v1"
PRIMARY_COMPOSITE_MAX = 0.97
PRIMARY_METRIC_MAX = 1.05
TAIL_METRIC_MAX = 1.15


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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
    path = Path(str(record["path"])).resolve()
    if _file_record(path) != record:
        raise RuntimeError(f"sealed artifact changed: {path}")
    return path


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(payload[name])
        for name in PRIMARY_METRICS + TAIL_METRICS
    }


def _ratio(candidate: dict[str, float], control: dict[str, float], names: tuple[str, ...]) -> float:
    values = []
    for name in names:
        left = float(candidate[name])
        right = float(control[name])
        if not all(math.isfinite(value) and value > 0.0 for value in (left, right)):
            raise ValueError(f"metric {name} must be finite and positive")
        values.append(math.log(left / right))
    return float(math.exp(sum(values) / len(values)))


def apply_gate(candidate: dict[str, float], control: dict[str, float]) -> dict[str, Any]:
    ratios = {
        name: float(candidate[name] / control[name])
        for name in PRIMARY_METRICS + TAIL_METRICS
    }
    primary = _ratio(candidate, control, PRIMARY_METRICS)
    checks = {
        "primary_composite_at_most_0p97": primary <= PRIMARY_COMPOSITE_MAX,
        "each_primary_at_most_1p05": all(
            ratios[name] <= PRIMARY_METRIC_MAX for name in PRIMARY_METRICS
        ),
        "each_tail_at_most_1p15": all(
            ratios[name] <= TAIL_METRIC_MAX for name in TAIL_METRICS
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "primary_composite_ratio": primary,
        "metric_ratios": ratios,
        "thresholds": {
            "primary_composite_max": PRIMARY_COMPOSITE_MAX,
            "primary_metric_max": PRIMARY_METRIC_MAX,
            "tail_metric_max": TAIL_METRIC_MAX,
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    seal_path = args.seal.resolve()
    seal = _read_json(seal_path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "candidate_method_id": CANDIDATE_METHOD_ID,
        "artifact_role": "sealed_blind_workload_allocation_candidate",
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": True,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise RuntimeError(f"WP10a seal violates {key}: {seal.get(key)!r}")
    if seal.get("mechanism_gate", {}).get("passed") is not True:
        raise RuntimeError("WP10a mechanism gate did not pass")

    plan_path = _verify_record(seal["plan"])
    plan = _read_json(plan_path)
    evaluator_record = plan.get("sources", {}).get("evaluator")
    if not isinstance(evaluator_record, dict):
        raise RuntimeError("WP10a plan did not freeze its evaluator")
    _verify_record(evaluator_record)
    if evaluator_record != _file_record(Path(__file__)):
        raise RuntimeError("WP10a evaluator changed after generation was planned")
    for section in ("control", "candidate"):
        records = seal.get(section)
        if not isinstance(records, dict):
            raise RuntimeError(f"WP10a seal is missing {section} records")
        for record in records.values():
            if isinstance(record, dict) and {"path", "bytes", "sha256"} <= set(record):
                _verify_record(record)

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"WP10a evaluation output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    input_dir = Path(str(plan["public_input_dir"])).resolve()
    cache = output / "offline_true_answer_cache.npz"
    control_raw = evaluate_external_synthetic(
        input_dir,
        _verify_record(seal["control"]["synthetic"]),
        batch_size=int(plan["offline_evaluator_batch_size"]),
        include_block_details=False,
        true_answers_cache_path=cache,
    )
    candidate_raw = evaluate_external_synthetic(
        input_dir,
        _verify_record(seal["candidate"]["synthetic"]),
        batch_size=int(plan["offline_evaluator_batch_size"]),
        include_block_details=False,
        true_answers_cache_path=cache,
    )
    control = _metrics(control_raw)
    candidate = _metrics(candidate_raw)
    gate = apply_gate(candidate, control)
    decision = (
        "authorize_adult_eps0p3_seed0"
        if gate["passed"]
        else "freeze_workload_l2_allocation"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "candidate_method_id": CANDIDATE_METHOD_ID,
        "evaluator_id": EVALUATOR_ID,
        "artifact_role": "offline_true_utility_development_gate",
        "dataset": "adult",
        "epsilon": 0.1,
        "seed": 0,
        "control_metrics": control,
        "candidate_metrics": candidate,
        "gate": gate,
        "decision": decision,
        "source_seal": _file_record(seal_path),
        "true_utility_evaluated": True,
        "utility_used_during_generation": False,
    }
    write_json(result, output / "gate_summary.json")
    write_json(
        {
            "protocol_id": PROTOCOL_ID,
            "artifact_role": "wp10a_evaluation_manifest",
            "decision": decision,
            "gate_summary": _file_record(output / "gate_summary.json"),
        },
        output / "manifest.json",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the sealed WP10a workload allocation pilot.")
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps({"decision": result["decision"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
