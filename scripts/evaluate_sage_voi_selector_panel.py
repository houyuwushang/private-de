#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from qdte.eval.external import evaluate_external_synthetic
from scripts.run_sage_voi_selector_panel import (
    EVALUATION_PROTOCOL,
    PROTOCOL_ID,
    expected_cells,
    sha256_file,
)
from scripts.run_sage_voi_selector_pilot import ARMS


PRIMARY_METRICS = (
    "full_true_mae",
    "full_true_rmse",
    "full_true_avg_tvd",
)
TAIL_METRICS = ("full_true_max_tvd", "full_true_max_error")
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20_260_714


def _verify_record(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(record["bytes"]) != path.stat().st_size:
        raise RuntimeError(f"sealed artifact size changed: {path}")
    if str(record["sha256"]) != sha256_file(path):
        raise RuntimeError(f"sealed artifact hash changed: {path}")
    return path


def load_and_verify_sealed_panel(panel_root: Path) -> dict[str, Any]:
    seal_path = panel_root / "sealed_panel_manifest.json"
    if not seal_path.is_file():
        raise FileNotFoundError("offline evaluation requires sealed_panel_manifest.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("sealed panel uses the wrong protocol")
    if seal.get("artifact_role") != "sealed_blind_formal_panel":
        raise RuntimeError("sealed panel has the wrong artifact role")
    if seal.get("true_utility_evaluated") is not False:
        raise RuntimeError("sealed panel was not blind")
    if seal.get("offline_evaluation_authorized") is not True:
        raise RuntimeError("sealed panel does not authorize offline evaluation")
    cells = list(seal.get("cells", []))
    if len(cells) != len(expected_cells()):
        raise RuntimeError("sealed panel is incomplete")
    expected_keys = {
        (cell.dataset, float(cell.epsilon), int(cell.seed))
        for cell in expected_cells()
    }
    actual_keys = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"]))
        for cell in cells
    }
    if actual_keys != expected_keys:
        raise RuntimeError("sealed panel matrix differs from the frozen matrix")
    _verify_record(seal["panel_plan"])
    for cell in cells:
        _verify_record(cell["summary"])
        if set(cell["arms"]) != set(ARMS):
            raise RuntimeError("sealed cell has the wrong arms")
        for arm in ARMS:
            _verify_record(cell["arms"][arm]["arm_manifest"])
            _verify_record(cell["arms"][arm]["final_synthetic"])
    return seal


def _cell_key(row: dict[str, Any]) -> tuple[str, float, int]:
    return str(row["dataset"]), float(row["epsilon"]), int(row["seed"])


def _positive_metric(row: dict[str, Any], metric: str) -> float:
    value = float(row[metric])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"metric {metric} must be finite and positive for log ratios")
    return value


def paired_log_ratios(
    rows: Iterable[dict[str, Any]],
    *,
    numerator: str,
    denominator: str,
    metrics: tuple[str, ...] = PRIMARY_METRICS,
) -> dict[tuple[str, float, int], np.ndarray]:
    by_key: dict[tuple[str, float, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[_cell_key(row)][str(row["arm"])] = row
    expected = {
        (cell.dataset, float(cell.epsilon), int(cell.seed))
        for cell in expected_cells()
    }
    if set(by_key) != expected:
        raise ValueError("evaluation rows do not cover the frozen matrix")
    output: dict[tuple[str, float, int], np.ndarray] = {}
    for key, arms in by_key.items():
        if numerator not in arms or denominator not in arms:
            raise ValueError(f"missing paired arms for {key}")
        output[key] = np.asarray(
            [
                math.log(
                    _positive_metric(arms[numerator], metric)
                    / _positive_metric(arms[denominator], metric)
                )
                for metric in metrics
            ],
            dtype=np.float64,
        )
    return output


def bootstrap_upper_ratio(
    cell_log_ratios: dict[tuple[str, float, int], np.ndarray],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    if int(samples) <= 0:
        raise ValueError("bootstrap samples must be positive")
    matrix = np.stack([cell_log_ratios[key] for key in sorted(cell_log_ratios)])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, matrix.shape[0], size=(int(samples), matrix.shape[0]))
    means = np.mean(matrix[indices], axis=(1, 2), dtype=np.float64)
    return float(math.exp(float(np.quantile(means, 0.95))))


def geometric_ratio(cell_log_ratios: dict[tuple[str, float, int], np.ndarray]) -> float:
    values = np.concatenate(list(cell_log_ratios.values()))
    return float(math.exp(float(np.mean(values, dtype=np.float64))))


def _grouped_ratios(
    cell_log_ratios: dict[tuple[str, float, int], np.ndarray],
) -> dict[str, dict[str, float]]:
    by_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    by_epsilon: dict[str, list[np.ndarray]] = defaultdict(list)
    for (dataset, epsilon, _seed), values in cell_log_ratios.items():
        by_dataset[dataset].append(values)
        by_epsilon[f"{epsilon:g}"].append(values)
    return {
        "by_dataset": {
            key: float(math.exp(float(np.mean(np.concatenate(values)))))
            for key, values in sorted(by_dataset.items())
        },
        "by_epsilon": {
            key: float(math.exp(float(np.mean(np.concatenate(values)))))
            for key, values in sorted(by_epsilon.items())
        },
    }


def summarize_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oi_vs_l1 = paired_log_ratios(
        rows,
        numerator="private_orthogonal_interaction_R",
        denominator="private_partition_l1_R",
    )
    primary_ratio = geometric_ratio(oi_vs_l1)
    upper = bootstrap_upper_ratio(oi_vs_l1)
    public_vs_l1 = paired_log_ratios(
        rows,
        numerator="private_partition_l1_R",
        denominator="public_static_R",
    )
    tail = paired_log_ratios(
        rows,
        numerator="private_orthogonal_interaction_R",
        denominator="private_partition_l1_R",
        metrics=TAIL_METRICS,
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "comparison": "private_orthogonal_interaction_R/private_partition_l1_R",
        "primary_metrics": list(PRIMARY_METRICS),
        "primary_geometric_ratio": primary_ratio,
        "bootstrap": {
            "unit": "dataset_epsilon_seed_cell",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "one_sided_upper_quantile": 0.95,
            "upper_ratio": upper,
        },
        "promotion_threshold": 0.97,
        "promotion_passed": bool(primary_ratio <= 0.97 and upper < 1.0),
        "oi_vs_l1": {
            **_grouped_ratios(oi_vs_l1),
            "by_cell": {
                f"{dataset}|{epsilon:g}|{seed}": float(math.exp(float(np.mean(values))))
                for (dataset, epsilon, seed), values in sorted(oi_vs_l1.items())
            },
        },
        "private_l1_vs_public_primary_ratio": geometric_ratio(public_vs_l1),
        "oi_vs_l1_tail_geometric_ratio": geometric_ratio(tail),
    }


def evaluate_panel(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = args.panel_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"offline evaluation output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    seal = load_and_verify_sealed_panel(panel_root)
    plan = json.loads(_verify_record(seal["panel_plan"]).read_text(encoding="utf-8"))
    input_dirs = {
        dataset: Path(str(record["public_schema"]["path"])).parent
        for dataset, record in plan["datasets"].items()
    }
    rows: list[dict[str, Any]] = []
    for cell in sorted(
        seal["cells"],
        key=lambda item: (str(item["dataset"]), float(item["epsilon"]), int(item["seed"])),
    ):
        dataset = str(cell["dataset"])
        cache_path = output_root / "offline_true_answer_cache" / f"{dataset}.npz"
        for arm in ARMS:
            synthetic = _verify_record(cell["arms"][arm]["final_synthetic"])
            metrics = evaluate_external_synthetic(
                input_dirs[dataset],
                synthetic,
                batch_size=int(args.batch_size),
                include_block_details=False,
                true_answers_cache_path=cache_path,
            )
            row = {
                "dataset": dataset,
                "epsilon": float(cell["epsilon"]),
                "seed": int(cell["seed"]),
                "arm": arm,
                **{metric: float(metrics[metric]) for metric in PRIMARY_METRICS + TAIL_METRICS},
            }
            rows.append(row)
            destination = (
                output_root
                / "raw"
                / dataset
                / f"epsilon_{float(cell['epsilon']):g}"
                / f"seed_{int(cell['seed'])}"
                / f"{arm}.json"
            )
            write_json(metrics, destination)

    csv_path = output_root / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["dataset", "epsilon", "seed", "arm", *PRIMARY_METRICS, *TAIL_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    gate = summarize_gate(rows)
    gate.update(
        {
            "artifact_role": "offline_true_utility_gate",
            "sealed_panel_sha256": sha256_file(panel_root / "sealed_panel_manifest.json"),
            "evaluation_protocol_sha256": sha256_file(EVALUATION_PROTOCOL),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "num_rows": len(rows),
        }
    )
    write_json(gate, output_root / "gate_summary.json")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fully sealed WP7 selector panel exactly once."
    )
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    result = evaluate_panel(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
