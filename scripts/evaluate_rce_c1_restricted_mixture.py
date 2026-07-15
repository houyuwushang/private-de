#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from qdte.eval.metrics import query_error_metrics
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import QueryCatalogue
from scripts.evaluate_rce_c1_panel import PRIMARY_METRICS, TAIL_METRICS
from scripts.run_coverage_refinement_wp8a import sha256_file
from scripts.run_rce_c1_panel import expected_cells
from scripts.run_rce_c1_restricted_mixture import PROTOCOL_ID


EVALUATOR_ID = "SAGE-QDTE-RCE-C1-RESTRICTED-MIXTURE-EVAL-20260715-v2"
EVALUATION_PLAN_ID = "SAGE-QDTE-RCE-C1-RESTRICTED-MIXTURE-EVAL-PLAN-20260715-v2"
DEFAULT_RESTRICTED_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_restricted_mixture_v2_cpu_20260715"
DEFAULT_C1_EVAL_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_eval_20260715"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715"
DEFAULT_INPUT_ROOT = Path("external_inputs")


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
    path = Path(str(record["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(record["bytes"]) != path.stat().st_size:
        raise RuntimeError(f"Sealed artifact size changed: {path}")
    if str(record["sha256"]) != sha256_file(path):
        raise RuntimeError(f"Sealed artifact hash changed: {path}")
    return path


def verify_restricted_seal(root: Path) -> dict[str, Any]:
    seal = _read_json(root.resolve() / "sealed_manifest.json")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "sealed_released_only_restricted_mixture_rce_panel",
        "num_cells": len(expected_cells()),
        "global_relaxed_certificate": False,
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": True,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise RuntimeError(f"Restricted-mixture seal violates {key}: {seal.get(key)!r}")
    _verify_record(seal["summary"])
    cells = seal.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected_cells()):
        raise RuntimeError("Restricted-mixture seal has an incomplete cell set")
    for cell in cells:
        _verify_record(cell["restricted_result"])
        _verify_record(cell["fractional_query_answers"])
    return seal


def prepare_evaluation_plan(args: argparse.Namespace) -> dict[str, Any]:
    restricted_root = args.restricted_root.resolve()
    c1_eval_root = args.c1_eval_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Evaluation-plan output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    verify_restricted_seal(restricted_root)
    plan = {
        "plan_id": EVALUATION_PLAN_ID,
        "evaluator_id": EVALUATOR_ID,
        "artifact_role": "frozen_offline_restricted_mixture_evaluation_plan",
        "true_utility_evaluated": False,
        "batch_size": int(args.batch_size),
        "input_root": str(args.input_root.resolve()),
        "evaluator": _file_record(Path(__file__)),
        "restricted_seal": _file_record(restricted_root / "sealed_manifest.json"),
        "source_c1_evaluation": _file_record(c1_eval_root / "summary.json"),
    }
    write_json(plan, output_root / "evaluation_plan.json")
    return plan


def _verify_evaluation_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = _read_json(args.output_root.resolve() / "evaluation_plan.json")
    expected = {
        "plan_id": EVALUATION_PLAN_ID,
        "evaluator_id": EVALUATOR_ID,
        "artifact_role": "frozen_offline_restricted_mixture_evaluation_plan",
        "true_utility_evaluated": False,
        "batch_size": int(args.batch_size),
        "input_root": str(args.input_root.resolve()),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"Evaluation plan violates {key}: {plan.get(key)!r}")
    if _verify_record(plan["evaluator"]) != Path(__file__).resolve():
        raise RuntimeError("Restricted-mixture evaluator path changed after plan freeze")
    if _verify_record(plan["restricted_seal"]) != (
        args.restricted_root.resolve() / "sealed_manifest.json"
    ):
        raise RuntimeError("Restricted-mixture seal differs from the frozen plan")
    if _verify_record(plan["source_c1_evaluation"]) != (
        args.c1_eval_root.resolve() / "summary.json"
    ):
        raise RuntimeError("Source C1 evaluation differs from the frozen plan")
    return plan


def _partition_blocks(input_dir: Path, num_queries: int) -> list[np.ndarray]:
    payload = json.loads((input_dir / "workload_groups.json").read_text(encoding="utf-8"))
    raw_groups = payload["groups"] if isinstance(payload, dict) else payload
    blocks: list[np.ndarray] = []
    coverage = np.zeros(num_queries, dtype=np.int32)
    for group in raw_groups:
        indices = np.asarray(group["query_indices"], dtype=np.int32)
        if np.any(indices < 0) or np.any(indices >= num_queries):
            raise ValueError("workload_groups.json contains out-of-range query indices")
        coverage[indices] += 1
        if len(indices) > 1 and bool(group.get("is_partition", False)):
            blocks.append(indices)
    if np.any(coverage != 1) or not blocks:
        raise ValueError("Evaluator workload groups must partition all queries exactly once")
    return blocks


def _metrics_from_answers(
    true_answers: np.ndarray,
    synthetic_answers: np.ndarray,
    n_rows: int,
    blocks: list[np.ndarray],
) -> dict[str, float]:
    metrics = query_error_metrics(
        true_answers,
        synthetic_answers,
        n_rows,
        n_rows,
        prefix="full_true",
    )
    true_rate = np.asarray(true_answers, dtype=np.float64) / float(n_rows)
    synthetic_rate = np.asarray(synthetic_answers, dtype=np.float64) / float(n_rows)
    tvd = np.asarray(
        [0.5 * np.sum(np.abs(synthetic_rate[idx] - true_rate[idx])) for idx in blocks],
        dtype=np.float64,
    )
    return {
        "full_true_mae": float(metrics["full_true_mae"]),
        "full_true_rmse": float(metrics["full_true_rmse"]),
        "full_true_avg_tvd": float(np.mean(tvd)),
        "full_true_max_tvd": float(np.max(tvd)),
        "full_true_max_error": float(metrics["full_true_max_error"]),
    }


def _primary_ratio(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    dataset: str | None = None,
    epsilon: float | None = None,
) -> float:
    indexed = {
        (str(row["dataset"]), float(row["epsilon"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    logs: list[float] = []
    cells = sorted({key[:3] for key in indexed})
    for cell_dataset, cell_epsilon, seed in cells:
        if dataset is not None and cell_dataset != dataset:
            continue
        if epsilon is not None and cell_epsilon != float(epsilon):
            continue
        if (cell_dataset, cell_epsilon, seed, left) not in indexed:
            continue
        for metric in PRIMARY_METRICS:
            logs.append(
                math.log(
                    float(indexed[(cell_dataset, cell_epsilon, seed, left)][metric])
                    / float(indexed[(cell_dataset, cell_epsilon, seed, right)][metric])
                )
            )
    if not logs:
        raise ValueError("No rows matched the requested ratio")
    return float(math.exp(float(np.mean(logs))))


def _ratio_summary(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    epsilons = sorted({float(row["epsilon"]) for row in rows})
    return {
        "overall": _primary_ratio(rows, left, right),
        "dataset": {
            dataset: _primary_ratio(rows, left, right, dataset=dataset)
            for dataset in datasets
        },
        "cell": {
            f"{dataset}|{epsilon:g}": _primary_ratio(
                rows,
                left,
                right,
                dataset=dataset,
                epsilon=epsilon,
            )
            for dataset in datasets
            for epsilon in epsilons
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    restricted_root = args.restricted_root.resolve()
    c1_eval_root = args.c1_eval_root.resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError("Prepare and freeze the evaluation plan before evaluation")
    unexpected = {
        path.name for path in output_root.iterdir() if path.name != "evaluation_plan.json"
    }
    if unexpected:
        raise FileExistsError(
            f"Restricted-mixture evaluation output contains unexpected files: {unexpected}"
        )

    evaluation_plan = _verify_evaluation_plan(args)
    seal = verify_restricted_seal(restricted_root)
    source_panel_root = Path(str(seal["source_sealed_panel"]["path"])).resolve().parent
    cell_index = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"])): cell
        for cell in seal["cells"]
    }
    rows: list[dict[str, Any]] = []
    with (c1_eval_root / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "epsilon": float(row["epsilon"]),
                    "seed": int(row["seed"]),
                    "arm": str(row["arm"]),
                    **{
                        metric: float(row[metric])
                        for metric in PRIMARY_METRICS + TAIL_METRICS
                    },
                }
            )

    true_answers_by_dataset: dict[str, np.ndarray] = {}
    weights: list[dict[str, Any]] = []
    for expected in expected_cells():
        key = (expected.dataset, expected.epsilon, expected.seed)
        cell = cell_index[key]
        result = _read_json(_verify_record(cell["restricted_result"]))
        if result.get("private_truth_used") is not False:
            raise RuntimeError("Restricted solve was not truth-isolated")
        answers = np.load(_verify_record(cell["fractional_query_answers"])).astype(np.float64)
        input_dir = args.input_root.resolve() / f"{expected.dataset}_sage_strong"
        full_qcat = QueryCatalogue.from_dict(_read_json(input_dir / "queries_full.json"))
        released_query_payload = _read_json(
            source_panel_root
            / expected.dataset
            / f"epsilon_{expected.epsilon:g}"
            / f"seed_{expected.seed}"
            / "public_transcript"
            / "queries.json"
        )
        measured_qcat = QueryCatalogue.from_dict(released_query_payload)
        if answers.shape != (measured_qcat.m,):
            raise RuntimeError("Stored fractional answers do not match the measured catalogue")
        component_names = tuple(str(name) for name in result["component_order"])
        component_weights = np.asarray(
            [float(cell["component_weights"][name]) for name in component_names],
            dtype=np.float64,
        )
        component_tables = tuple(
            np.load(_verify_record(result["component_tables"][name])).astype(np.int32)
            for name in component_names
        )
        measured_component_answers = np.stack(
            [
                answer_queries(
                    table,
                    measured_qcat,
                    batch_size=int(args.batch_size),
                ).astype(np.float64)
                for table in component_tables
            ],
            axis=0,
        )
        if not np.allclose(
            component_weights @ measured_component_answers,
            answers,
            rtol=1.0e-10,
            atol=1.0e-8,
        ):
            raise RuntimeError("Frozen weights do not reproduce measured fractional answers")
        full_component_answers = np.stack(
            [
                answer_queries(
                    table,
                    full_qcat,
                    batch_size=int(args.batch_size),
                ).astype(np.float64)
                for table in component_tables
            ],
            axis=0,
        )
        full_fractional_answers = component_weights @ full_component_answers
        real = np.load(input_dir / "real_encoded.npy").astype(np.int32)
        if len(real) <= 0:
            raise RuntimeError("Offline evaluator received an empty real table")
        if expected.dataset not in true_answers_by_dataset:
            true_answers_by_dataset[expected.dataset] = answer_queries(
                real,
                full_qcat,
                batch_size=int(args.batch_size),
            ).astype(np.float64)
        metrics = _metrics_from_answers(
            true_answers_by_dataset[expected.dataset],
            full_fractional_answers,
            len(real),
            _partition_blocks(input_dir, full_qcat.m),
        )
        rows.append(
            {
                "dataset": expected.dataset,
                "epsilon": expected.epsilon,
                "seed": expected.seed,
                "arm": "restricted_mixture_rce",
                **metrics,
            }
        )
        weights.append(
            {
                "dataset": expected.dataset,
                "epsilon": expected.epsilon,
                "seed": expected.seed,
                **{
                    f"weight_{name}": float(value)
                    for name, value in cell["component_weights"].items()
                },
                "slack_star": float(cell["slack_star"]),
                "kl_objective": float(cell["kl_objective"]),
            }
        )

    metric_path = output_root / "metrics.csv"
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    weight_path = output_root / "weights.csv"
    with weight_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weights[0]))
        writer.writeheader()
        writer.writerows(weights)

    ratios = {
        denominator: _ratio_summary(rows, "restricted_mixture_rce", denominator)
        for denominator in ("initial", "wp9_control", "rce_v1", "official_aim")
        if any(str(row["arm"]) == denominator for row in rows)
    }
    mean_weights = {
        name: float(np.mean([row[f"weight_{name}"] for row in weights]))
        for name in ("initial", "wp9_control", "rce_v1")
    }
    summary = {
        "evaluator_id": EVALUATOR_ID,
        "artifact_role": "offline_nonreleasable_restricted_mixture_rce_evaluation",
        "private_truth_used": True,
        "allowed_use": "post-seal diagnosis only; no method promotion or tuning",
        "global_relaxed_certificate": False,
        "mean_component_weights": mean_weights,
        "primary_ratios": ratios,
        "classification": {
            "beats_wp9_overall": bool(ratios["wp9_control"]["overall"] < 1.0),
            "beats_rce_v1_overall": bool(ratios["rce_v1"]["overall"] < 1.0),
            "product_kl_prefers_wp9_on_average": bool(
                mean_weights["wp9_control"]
                > max(mean_weights["initial"], mean_weights["rce_v1"])
            ),
        },
        "artifacts": {
            "evaluation_plan": _file_record(output_root / "evaluation_plan.json"),
            "restricted_seal": _file_record(restricted_root / "sealed_manifest.json"),
            "source_c1_evaluation": _file_record(c1_eval_root / "summary.json"),
            "metrics": _file_record(metric_path),
            "weights": _file_record(weight_path),
        },
    }
    write_json(summary, output_root / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for the sealed restricted-mixture RCE diagnostic"
    )
    parser.add_argument("--restricted-root", type=Path, default=DEFAULT_RESTRICTED_ROOT)
    parser.add_argument("--c1-eval-root", type=Path, default=DEFAULT_C1_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--prepare-plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_evaluation_plan(args) if args.prepare_plan else evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
