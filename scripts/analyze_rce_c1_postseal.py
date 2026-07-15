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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from qdte.eval.external import evaluate_external_synthetic
from qdte.evolution.entropy import AtomEntropyState, ReleasedProductPrior
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.measure import Measurements
from qdte.measurement.public_artifact import verify_public_transcript
from qdte.queries.eval_jax import answer_queries
from qdte.queries.workload import WorkloadGroup
from qdte.rce.confidence_set import RCEConfidenceSet
from scripts.evaluate_rce_c1_panel import PRIMARY_METRICS, TAIL_METRICS, verify_sealed_panel
from scripts.run_coverage_refinement_wp8a import sha256_file
from scripts.run_rce_c1_panel import expected_cells


ANALYZER_ID = "SAGE-QDTE-RCE-C1-POSTSEAL-DIAGNOSTIC-20260715-v1"
DEFAULT_PANEL_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_20260715"
DEFAULT_EVAL_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_eval_20260715"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_postseal_20260715"
DEFAULT_INPUT_ROOT = Path("external_inputs")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _groups_from_measurements(measurements: Measurements) -> list[WorkloadGroup]:
    return [
        WorkloadGroup(
            name=group.name,
            family=group.family,
            query_indices=np.asarray(group.query_indices, dtype=np.int32).copy(),
            sensitivity_l2=float(group.sensitivity_l2),
            is_partition=bool(group.is_partition),
        )
        for group in measurements.groups
    ]


def _primary_ratio_rows(
    rows: list[dict[str, Any]],
    left_arm: str,
    right_arm: str,
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
        left = indexed[(cell_dataset, cell_epsilon, seed, left_arm)]
        right = indexed[(cell_dataset, cell_epsilon, seed, right_arm)]
        logs.extend(
            math.log(float(left[metric]) / float(right[metric]))
            for metric in PRIMARY_METRICS
        )
    if not logs:
        raise ValueError("No rows matched the requested primary-ratio slice")
    return float(math.exp(float(np.mean(logs))))


def _ratio_summary(
    rows: list[dict[str, Any]],
    left_arm: str,
    right_arm: str,
) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    epsilons = sorted({float(row["epsilon"]) for row in rows})
    return {
        "overall": _primary_ratio_rows(rows, left_arm, right_arm),
        "dataset": {
            dataset: _primary_ratio_rows(rows, left_arm, right_arm, dataset=dataset)
            for dataset in datasets
        },
        "cell": {
            f"{dataset}|{epsilon:g}": _primary_ratio_rows(
                rows,
                left_arm,
                right_arm,
                dataset=dataset,
                epsilon=epsilon,
            )
            for dataset in datasets
            for epsilon in epsilons
        },
    }


def _confidence_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("truth", "initial", "wp9_control", "rce_v1"):
        selected = [row for row in rows if row["arm"] == arm]
        result[arm] = {
            "inside": int(sum(bool(row["inside"]) for row in selected)),
            "total": len(selected),
            "mean_slack": float(np.mean([float(row["slack"]) for row in selected])),
            "max_slack": float(np.max([float(row["slack"]) for row in selected])),
            "mean_kl_to_product_prior": float(
                np.mean([float(row["kl_to_product_prior"]) for row in selected])
            ),
        }
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = args.panel_root.resolve()
    eval_root = args.eval_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Post-seal diagnostic output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    seal, _ = verify_sealed_panel(panel_root)
    eval_summary = _read_json(eval_root / "summary.json")
    if eval_summary.get("artifact_role") != "offline_rce_c1_causal_evaluation":
        raise RuntimeError("RCE C1 frozen offline evaluation is missing")
    cell_index = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"])): cell
        for cell in seal["cells"]
    }

    metric_rows: list[dict[str, Any]] = []
    with (eval_root / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric_rows.append(
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

    confidence_rows: list[dict[str, Any]] = []
    true_answers_by_dataset: dict[str, np.ndarray] = {}
    for expected in expected_cells():
        key = (expected.dataset, expected.epsilon, expected.seed)
        cell = cell_index[key]
        transcript = verify_public_transcript(
            panel_root
            / expected.dataset
            / f"epsilon_{expected.epsilon:g}"
            / f"seed_{expected.seed}"
            / "public_transcript"
        )
        measurements = transcript.measurements
        if measurements.strategy_transcript is None:
            raise RuntimeError("RCE C1 transcript is missing its orthogonal strategy")
        strategy = HierarchicalInteractionTranscript.from_public_dict(
            measurements.strategy_transcript
        )
        precision = OrthogonalInteractionPrecision(
            transcript.queries,
            _groups_from_measurements(measurements),
            strategy,
        )
        confidence = RCEConfidenceSet.from_diagonal_variances(
            precision.coefficient_variances,
            alpha_l2=0.025,
            alpha_linf=0.025,
        )
        prior = ReleasedProductPrior.from_released_oneway(
            transcript.queries,
            measurements.target_projected,
            transcript.schema.cardinalities,
            public_total=int(measurements.num_rows),
            smoothing=1.0,
        )

        input_dir = args.input_root.resolve() / f"{expected.dataset}_sage_strong"
        real = np.load(input_dir / "real_encoded.npy").astype(np.int32)
        if len(real) != int(measurements.num_rows):
            raise RuntimeError("Offline true table row count differs from the declared public n")
        if expected.dataset not in true_answers_by_dataset:
            true_answers_by_dataset[expected.dataset] = answer_queries(
                real,
                transcript.queries,
                batch_size=int(args.batch_size),
            ).astype(np.float64)

        initial_path = Path(cell["source_control"]["initial_table"]["path"])
        table_paths = {
            "truth": input_dir / "real_encoded.npy",
            "initial": initial_path,
            "wp9_control": Path(cell["source_control"]["synthetic"]["path"]),
            "rce_v1": Path(cell["synthetic"]["path"]),
        }
        for arm, path in table_paths.items():
            rows = real if arm == "truth" else np.load(path).astype(np.int32)
            answers = (
                true_answers_by_dataset[expected.dataset]
                if arm == "truth"
                else answer_queries(
                    rows,
                    transcript.queries,
                    batch_size=int(args.batch_size),
                ).astype(np.float64)
            )
            residual = measurements.target_projected.astype(np.float64) - answers
            evaluation = confidence.evaluate(
                precision.coefficient_coordinates(residual)
            ).to_dict()
            entropy_state = AtomEntropyState.from_rows(rows, prior)
            confidence_rows.append(
                {
                    "dataset": expected.dataset,
                    "epsilon": expected.epsilon,
                    "seed": expected.seed,
                    "arm": arm,
                    **evaluation,
                    "kl_to_product_prior": float(
                        entropy_state.regularizer / entropy_state.n_rows
                    ),
                }
            )

        initial_metrics = evaluate_external_synthetic(
            input_dir,
            initial_path,
            batch_size=int(args.batch_size),
            include_block_details=False,
            true_answers_cache_path=(eval_root / "offline_true_answer_cache" / f"{expected.dataset}.npz"),
        )
        metric_rows.append(
            {
                "dataset": expected.dataset,
                "epsilon": expected.epsilon,
                "seed": expected.seed,
                "arm": "initial",
                **{
                    metric: float(initial_metrics[metric])
                    for metric in PRIMARY_METRICS + TAIL_METRICS
                },
            }
        )

    confidence_path = output_root / "confidence_and_kl.csv"
    with confidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confidence_rows[0]))
        writer.writeheader()
        writer.writerows(confidence_rows)

    metrics_path = output_root / "metrics_with_initial.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "analyzer_id": ANALYZER_ID,
        "artifact_role": "offline_nonreleasable_rce_c1_postseal_diagnostic",
        "private_truth_used": True,
        "allowed_use": "offline diagnosis only; not generation, tuning, or promotion",
        "confidence": _confidence_aggregate(confidence_rows),
        "primary_ratios": {
            "rce_v1_vs_initial": _ratio_summary(metric_rows, "rce_v1", "initial"),
            "wp9_control_vs_initial": _ratio_summary(metric_rows, "wp9_control", "initial"),
            "rce_v1_vs_wp9_control": _ratio_summary(metric_rows, "rce_v1", "wp9_control"),
        },
        "artifacts": {
            "sealed_panel": _file_record(panel_root / "sealed_panel_manifest.json"),
            "frozen_evaluation": _file_record(eval_root / "summary.json"),
            "confidence_and_kl": _file_record(confidence_path),
            "metrics_with_initial": _file_record(metrics_path),
        },
    }
    write_json(summary, output_root / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-seal RCE C1 confidence/KL diagnostics")
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    summary = analyze(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
