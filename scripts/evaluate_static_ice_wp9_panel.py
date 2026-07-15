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
from scripts.run_static_ice_wp9_cell import METHOD_ID, PROTOCOL_ID
from scripts.run_static_ice_wp9_panel import DATASETS, EPSILONS, SEEDS, expected_cells


EVALUATOR_ID = "SAGE-QDTE-ICE-WP9-EVALUATOR-20260715-v1"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 0x1CE20260715
CELL_WIN_REQUIRED = 6
DATASET_PRIMARY_MAX = 1.05
DATASET_TAIL_MAX = 1.15


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
        raise RuntimeError(f"sealed artifact size changed: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"sealed artifact hash changed: {path}")
    return path


def verify_sealed_panel(panel_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seal_path = panel_root.resolve() / "sealed_panel_manifest.json"
    seal = _read_json(seal_path)
    expected_header = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "sealed_blind_static_ice_confirmation_panel",
        "num_cells": len(expected_cells()),
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": True,
    }
    for key, value in expected_header.items():
        if seal.get(key) != value:
            raise RuntimeError(f"WP9 seal violates {key}: {seal.get(key)!r}")
    plan_path = _verify_record(seal["panel_plan"])
    plan = _read_json(plan_path)
    evaluator_record = plan.get("evaluator")
    if not isinstance(evaluator_record, dict):
        raise RuntimeError("WP9 plan did not freeze the evaluator")
    _verify_record(evaluator_record)
    current = _file_record(Path(__file__))
    if evaluator_record != current:
        raise RuntimeError("WP9 evaluator changed after the panel plan was frozen")

    cells = seal.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected_cells()):
        raise RuntimeError("WP9 sealed matrix is incomplete")
    expected = {
        (cell.dataset, cell.epsilon, cell.seed) for cell in expected_cells()
    }
    observed = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"]))
        for cell in cells
    }
    if observed != expected:
        raise RuntimeError("WP9 sealed matrix differs from the protocol")
    for cell in cells:
        for key in ("mechanism_manifest", "cell_status", "synthetic", "measurement", "runtime", "metrics"):
            _verify_record(cell[key])
        if cell.get("mechanism_gate", {}).get("passed") is not True:
            raise RuntimeError("WP9 seal contains a failed mechanism cell")
    return seal, plan


def _metric_payload(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(metrics[metric]) for metric in PRIMARY_METRICS + TAIL_METRICS
    }


def _ratio(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
    metrics: tuple[str, ...] = PRIMARY_METRICS,
) -> float:
    logs = []
    for metric in metrics:
        left = float(numerator[metric])
        right = float(denominator[metric])
        if not all(math.isfinite(value) and value > 0.0 for value in (left, right)):
            raise ValueError(f"metric {metric} must be finite and positive")
        logs.append(math.log(left / right))
    return float(math.exp(sum(logs) / len(logs)))


def _epsilon_token(epsilon: float) -> str:
    return f"{float(epsilon):g}".replace(".", "p")


def paired_primary_log_ratio(
    static_metrics: dict[str, Any],
    aim_metrics: dict[str, Any],
) -> float:
    return float(
        sum(
            math.log(float(static_metrics[metric]) / float(aim_metrics[metric]))
            for metric in PRIMARY_METRICS
        )
        / len(PRIMARY_METRICS)
    )


def hierarchical_bootstrap(
    log_ratios: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    values = np.asarray(log_ratios, dtype=np.float64)
    expected_shape = (len(DATASETS), len(EPSILONS), len(SEEDS))
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(f"log_ratios must have shape {expected_shape}")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        dataset_indices = rng.integers(0, len(DATASETS), size=len(DATASETS))
        selected: list[float] = []
        for dataset_index in dataset_indices:
            epsilon_indices = rng.integers(0, len(EPSILONS), size=len(EPSILONS))
            for epsilon_index in epsilon_indices:
                seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
                selected.extend(values[dataset_index, epsilon_index, seed_indices])
        samples[index] = float(np.mean(selected))
    quantiles = np.quantile(samples, [0.025, 0.975])
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "lower": float(math.exp(float(quantiles[0]))),
        "upper": float(math.exp(float(quantiles[1]))),
    }


def apply_gate(
    *,
    cell_ratios: dict[str, float],
    overall_primary: float,
    bootstrap_upper: float,
    dataset_primary: dict[str, float],
    dataset_tail: dict[str, dict[str, float]],
) -> dict[str, Any]:
    wins = sum(value < 1.0 for value in cell_ratios.values())
    checks = {
        "at_least_6_of_8_cell_wins": wins >= CELL_WIN_REQUIRED,
        "overall_primary_below_1": overall_primary < 1.0,
        "bootstrap_upper_below_1": bootstrap_upper < 1.0,
        "every_dataset_primary_at_most_1p05": all(
            value <= DATASET_PRIMARY_MAX for value in dataset_primary.values()
        ),
        "every_dataset_tail_at_most_1p15": all(
            ratio <= DATASET_TAIL_MAX
            for values in dataset_tail.values()
            for ratio in values.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cell_wins": wins,
        "cell_total": len(cell_ratios),
    }


def evaluate_panel(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = args.panel_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"WP9 evaluation output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    seal, plan = verify_sealed_panel(panel_root)
    if int(args.batch_size) != int(plan["offline_evaluator_batch_size"]):
        raise RuntimeError("WP9 evaluator batch size differs from the frozen plan")
    cell_index = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"])): cell
        for cell in seal["cells"]
    }

    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, float, int], dict[str, dict[str, float]]] = {}
    reference_records: dict[str, Any] = {}
    for cell in expected_cells():
        key = (cell.dataset, cell.epsilon, cell.seed)
        sealed = cell_index[key]
        synthetic = _verify_record(sealed["synthetic"])
        input_dir = args.input_root.resolve() / cell.declared_dataset
        cache = output_root / "offline_true_answer_cache" / f"{cell.dataset}.npz"
        raw = evaluate_external_synthetic(
            input_dir,
            synthetic,
            batch_size=int(args.batch_size),
            include_block_details=False,
            true_answers_cache_path=cache,
        )
        static_metrics = _metric_payload(raw)
        aim_path = (
            args.aim_root.resolve()
            / cell.dataset
            / f"epsilon_{_epsilon_token(cell.epsilon)}"
            / f"seed_{cell.seed}"
            / "evaluation.json"
        )
        aim_metrics = _metric_payload(_read_json(aim_path))
        by_key[key] = {
            "static_ice_exact": static_metrics,
            "official_aim": aim_metrics,
        }
        reference_records[f"{cell.dataset}|{cell.epsilon:g}|{cell.seed}"] = {
            "official_aim": _file_record(aim_path)
        }
        for arm, metrics in (
            ("static_ice_exact", static_metrics),
            ("official_aim", aim_metrics),
        ):
            rows.append(
                {
                    "dataset": cell.dataset,
                    "epsilon": cell.epsilon,
                    "seed": cell.seed,
                    "arm": arm,
                    **metrics,
                }
            )
        raw_path = (
            output_root
            / "raw"
            / cell.dataset
            / f"epsilon_{cell.epsilon:g}"
            / f"seed_{cell.seed}"
            / "static_ice_exact.json"
        )
        write_json(raw, raw_path)

    csv_path = output_root / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "epsilon", "seed", "arm", *PRIMARY_METRICS, *TAIL_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    log_ratios = np.empty(
        (len(DATASETS), len(EPSILONS), len(SEEDS)), dtype=np.float64
    )
    per_seed: dict[str, float] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        for epsilon_index, epsilon in enumerate(EPSILONS):
            for seed_index, seed in enumerate(SEEDS):
                values = by_key[(dataset, epsilon, seed)]
                log_ratio = paired_primary_log_ratio(
                    values["static_ice_exact"], values["official_aim"]
                )
                log_ratios[dataset_index, epsilon_index, seed_index] = log_ratio
                per_seed[f"{dataset}|{epsilon:g}|{seed}"] = float(math.exp(log_ratio))

    cell_ratios = {
        f"{dataset}|{epsilon:g}": float(
            math.exp(
                np.mean(
                    log_ratios[dataset_index, epsilon_index, :]
                )
            )
        )
        for dataset_index, dataset in enumerate(DATASETS)
        for epsilon_index, epsilon in enumerate(EPSILONS)
    }
    dataset_primary = {
        dataset: float(math.exp(np.mean(log_ratios[dataset_index, :, :])))
        for dataset_index, dataset in enumerate(DATASETS)
    }
    overall_primary = float(math.exp(np.mean(log_ratios)))
    bootstrap = hierarchical_bootstrap(log_ratios)
    dataset_tail: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        dataset_tail[dataset] = {}
        for metric in TAIL_METRICS:
            logs = []
            for epsilon in EPSILONS:
                for seed in SEEDS:
                    values = by_key[(dataset, epsilon, seed)]
                    logs.append(
                        math.log(
                            values["static_ice_exact"][metric]
                            / values["official_aim"][metric]
                        )
                    )
            dataset_tail[dataset][metric] = float(math.exp(sum(logs) / len(logs)))

    gate = apply_gate(
        cell_ratios=cell_ratios,
        overall_primary=overall_primary,
        bootstrap_upper=bootstrap["upper"],
        dataset_primary=dataset_primary,
        dataset_tail=dataset_tail,
    )
    summary = {
        "evaluator_id": EVALUATOR_ID,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "offline_true_utility_confirmation_gate",
        "decision": (
            "confirm_low_budget_static_ice"
            if gate["passed"]
            else "retain_static_ice_as_development_candidate"
        ),
        "promotion_gate": gate,
        "thresholds": {
            "cell_wins_required": CELL_WIN_REQUIRED,
            "cell_total": len(DATASETS) * len(EPSILONS),
            "overall_primary_max_exclusive": 1.0,
            "bootstrap_upper_max_exclusive": 1.0,
            "dataset_primary_max": DATASET_PRIMARY_MAX,
            "dataset_tail_metric_max": DATASET_TAIL_MAX,
        },
        "overall_primary_ratio": overall_primary,
        "hierarchical_bootstrap_95": bootstrap,
        "dataset_primary_ratios": dataset_primary,
        "dataset_tail_ratios": dataset_tail,
        "cell_primary_ratios": cell_ratios,
        "seed_primary_ratios": per_seed,
        "by_cell": {
            f"{dataset}|{epsilon:g}|{seed}": {
                "primary_ratio": per_seed[f"{dataset}|{epsilon:g}|{seed}"],
                "tail_ratio": _ratio(
                    values["static_ice_exact"], values["official_aim"], TAIL_METRICS
                ),
                "metrics": values,
            }
            for (dataset, epsilon, seed), values in sorted(by_key.items())
        },
        "provenance": {
            "sealed_panel": _file_record(panel_root / "sealed_panel_manifest.json"),
            "panel_plan": seal["panel_plan"],
            "evaluator": _file_record(Path(__file__)),
            "metrics_csv": _file_record(csv_path),
            "official_aim": reference_records,
        },
    }
    write_json(summary, output_root / "gate_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the sealed Static-ICE-Exact WP9 confirmation panel."
    )
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("external_inputs"),
    )
    parser.add_argument(
        "--aim-root",
        type=Path,
        default=ROOT / "outputs" / "sage_qdte_wp4" / "E1" / "aim",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    result = evaluate_panel(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
