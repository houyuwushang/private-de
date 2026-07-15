#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from qdte.eval.external import evaluate_external_synthetic
from scripts.evaluate_sage_voi_selector_panel import PRIMARY_METRICS, TAIL_METRICS
from scripts.run_coverage_refinement_wp8a import sha256_file
from scripts.run_rce_c1_cell import DATASETS, EPSILONS, METHOD_ID, PROTOCOL_ID, SEEDS
from scripts.run_rce_c1_panel import EVALUATOR, expected_cells


EVALUATOR_ID = "SAGE-QDTE-RCE-C1-EVALUATOR-20260715-v3"
DEFAULT_PANEL_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_20260715"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_eval_20260715"
DEFAULT_INPUT_ROOT = Path("external_inputs")
DEFAULT_AIM_ROOT = ROOT / "outputs" / "sage_qdte_wp4" / "E1" / "aim"


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


def verify_sealed_panel(panel_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = _read_json(panel_root.resolve() / "sealed_panel_manifest.json")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "sealed_blind_same_transcript_rce_c1_panel",
        "num_cells": len(expected_cells()),
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": True,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise RuntimeError(f"RCE C1 seal violates {key}: {seal.get(key)!r}")
    plan = _read_json(_verify_record(seal["panel_plan"]))
    if plan.get("evaluator") != _file_record(EVALUATOR):
        raise RuntimeError("RCE C1 evaluator changed after the blind panel was frozen")
    cells = seal.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected_cells()):
        raise RuntimeError("RCE C1 sealed panel is incomplete")
    for cell in cells:
        for key in ("mechanism_manifest", "cell_status", "synthetic", "metrics", "runtime"):
            _verify_record(cell[key])
        for record in cell["source_control"].values():
            _verify_record(record)
        if cell.get("mechanism_gate", {}).get("passed") is not True:
            raise RuntimeError("RCE C1 sealed panel contains a failed cell")
    return seal, plan


def _epsilon_token(epsilon: float) -> str:
    return f"{float(epsilon):g}".replace(".", "p")


def _metric_payload(raw: dict[str, Any]) -> dict[str, float]:
    return {metric: float(raw[metric]) for metric in PRIMARY_METRICS + TAIL_METRICS}


def _primary_log_ratio(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        np.mean(
            [math.log(float(left[metric]) / float(right[metric])) for metric in PRIMARY_METRICS]
        )
    )


def _geometric_ratio(
    records: dict[tuple[str, float, int], dict[str, dict[str, float]]],
    left_arm: str,
    right_arm: str,
    *,
    dataset: str | None = None,
    epsilon: float | None = None,
    metrics: tuple[str, ...] = PRIMARY_METRICS,
) -> float:
    logs: list[float] = []
    for (cell_dataset, cell_epsilon, _), arms in records.items():
        if dataset is not None and cell_dataset != dataset:
            continue
        if epsilon is not None and cell_epsilon != epsilon:
            continue
        for metric in metrics:
            logs.append(math.log(arms[left_arm][metric] / arms[right_arm][metric]))
    if not logs:
        raise ValueError("No metrics matched the requested ratio slice")
    return float(math.exp(float(np.mean(logs))))


def evaluate_panel(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = args.panel_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"RCE C1 evaluation output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    seal, plan = verify_sealed_panel(panel_root)
    if int(args.batch_size) != int(plan["offline_evaluator_batch_size"]):
        raise RuntimeError("RCE C1 evaluator batch size differs from the frozen plan")
    cell_index = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"])): cell
        for cell in seal["cells"]
    }

    records: dict[tuple[str, float, int], dict[str, dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for expected in expected_cells():
        key = (expected.dataset, expected.epsilon, expected.seed)
        cell = cell_index[key]
        input_dir = args.input_root.resolve() / f"{expected.dataset}_sage_strong"
        cache = output_root / "offline_true_answer_cache" / f"{expected.dataset}.npz"
        rce_path = _verify_record(cell["synthetic"])
        control_path = _verify_record(cell["source_control"]["synthetic"])
        aim_path = (
            args.aim_root.resolve()
            / expected.dataset
            / f"epsilon_{_epsilon_token(expected.epsilon)}"
            / f"seed_{expected.seed}"
            / "evaluation.json"
        )
        raw_rce = evaluate_external_synthetic(
            input_dir,
            rce_path,
            batch_size=int(args.batch_size),
            include_block_details=False,
            true_answers_cache_path=cache,
        )
        raw_control = evaluate_external_synthetic(
            input_dir,
            control_path,
            batch_size=int(args.batch_size),
            include_block_details=False,
            true_answers_cache_path=cache,
        )
        arms = {
            "rce_v1": _metric_payload(raw_rce),
            "wp9_control": _metric_payload(raw_control),
            "official_aim": _metric_payload(_read_json(aim_path)),
        }
        records[key] = arms
        for arm, metrics in arms.items():
            rows.append(
                {
                    "dataset": expected.dataset,
                    "epsilon": expected.epsilon,
                    "seed": expected.seed,
                    "arm": arm,
                    **metrics,
                }
            )
        raw_root = (
            output_root
            / "raw"
            / expected.dataset
            / f"epsilon_{expected.epsilon:g}"
            / f"seed_{expected.seed}"
        )
        write_json(raw_rce, raw_root / "rce_v1.json")
        write_json(raw_control, raw_root / "wp9_control.json")
        provenance[f"{expected.dataset}|{expected.epsilon:g}|{expected.seed}"] = {
            "rce": _file_record(rce_path),
            "control": _file_record(control_path),
            "official_aim": _file_record(aim_path),
        }

    metrics_path = output_root / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "epsilon", "seed", "arm", *PRIMARY_METRICS, *TAIL_METRICS],
        )
        writer.writeheader()
        writer.writerows(rows)

    comparisons: dict[str, Any] = {}
    for denominator in ("wp9_control", "official_aim"):
        label = f"rce_v1_vs_{denominator}"
        comparisons[label] = {
            "overall_primary_ratio": _geometric_ratio(records, "rce_v1", denominator),
            "dataset_primary_ratios": {
                dataset: _geometric_ratio(records, "rce_v1", denominator, dataset=dataset)
                for dataset in DATASETS
            },
            "cell_primary_ratios": {
                f"{dataset}|{epsilon:g}": _geometric_ratio(
                    records,
                    "rce_v1",
                    denominator,
                    dataset=dataset,
                    epsilon=epsilon,
                )
                for dataset in DATASETS
                for epsilon in EPSILONS
            },
            "seed_primary_ratios": {
                f"{dataset}|{epsilon:g}|{seed}": math.exp(
                    _primary_log_ratio(
                        records[(dataset, epsilon, seed)]["rce_v1"],
                        records[(dataset, epsilon, seed)][denominator],
                    )
                )
                for dataset in DATASETS
                for epsilon in EPSILONS
                for seed in SEEDS
            },
            "dataset_tail_ratios": {
                dataset: {
                    metric: _geometric_ratio(
                        records,
                        "rce_v1",
                        denominator,
                        dataset=dataset,
                        metrics=(metric,),
                    )
                    for metric in TAIL_METRICS
                }
                for dataset in DATASETS
            },
        }

    adult_aim = comparisons["rce_v1_vs_official_aim"]
    adult_cells = {
        key: value
        for key, value in adult_aim["cell_primary_ratios"].items()
        if key.startswith("adult|")
    }
    summary = {
        "evaluator_id": EVALUATOR_ID,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "offline_rce_c1_causal_evaluation",
        "classification": {
            "rce_improves_point_target_control": (
                comparisons["rce_v1_vs_wp9_control"]["overall_primary_ratio"] < 1.0
            ),
            "adult_beats_aim_aggregate": (
                adult_aim["dataset_primary_ratios"]["adult"] < 1.0
            ),
            "adult_beats_aim_at_both_epsilons": all(
                value < 1.0 for value in adult_cells.values()
            ),
        },
        "comparisons": comparisons,
        "provenance": provenance,
        "metrics": _file_record(metrics_path),
    }
    write_json(summary, output_root / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fully sealed RCE C1 panel")
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--aim-root", type=Path, default=DEFAULT_AIM_ROOT)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    summary = evaluate_panel(parse_args())
    print(json.dumps(summary["classification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
