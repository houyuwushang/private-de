#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from qdte.evolution.entropy import ReleasedProductPrior
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.measure import Measurements
from qdte.measurement.public_artifact import verify_public_transcript
from qdte.queries.eval_jax import answer_queries
from qdte.queries.workload import WorkloadGroup
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.relaxed import solve_restricted_mixture_rce
from scripts.evaluate_rce_c1_panel import verify_sealed_panel
from scripts.run_coverage_refinement_wp8a import sha256_file
from scripts.run_rce_c1_panel import expected_cells


PROTOCOL_ID = "SAGE-QDTE-RCE-C1-RESTRICTED-MIXTURE-20260715-v2"
PROTOCOL_PATH = ROOT / "docs" / "SAGE_QDTE_RCE_C1_RESTRICTED_MIXTURE_PROTOCOL_20260715.md"
DEFAULT_PANEL_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_v3_20260715"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sage_qdte_rce_c1_restricted_mixture_v2_cpu_20260715"
COMPONENT_NAMES = ("initial", "wp9_control", "rce_v1")


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


def _component_distributions(
    tables: tuple[np.ndarray, ...],
    prior: ReleasedProductPrior,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if len(tables) != len(COMPONENT_NAMES):
        raise ValueError("The restricted hull requires exactly three frozen tables")
    first_shape = tables[0].shape
    if len(first_shape) != 2 or first_shape[0] <= 0:
        raise ValueError("Component tables must be non-empty matrices")
    for table in tables:
        if table.shape != first_shape:
            raise ValueError("All component tables must have identical shape")

    codes_by_component = [prior.encode_rows(table) for table in tables]
    concatenated_codes = np.concatenate(codes_by_component)
    concatenated_rows = np.concatenate(tables, axis=0)
    support_codes, first_indices, inverse = np.unique(
        concatenated_codes,
        return_index=True,
        return_inverse=True,
    )
    support_rows = concatenated_rows[first_indices]
    component_probabilities = np.zeros(
        (len(tables), len(support_codes)),
        dtype=np.float64,
    )
    offset = 0
    for index, table in enumerate(tables):
        local_inverse = inverse[offset : offset + len(table)]
        component_probabilities[index] = np.bincount(
            local_inverse,
            minlength=len(support_codes),
        ).astype(np.float64) / float(len(table))
        offset += len(table)
    log_prior = prior.log_probability_rows(support_rows)

    digest = hashlib.sha256()
    digest.update(np.asarray(support_codes, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(component_probabilities, dtype="<f8").tobytes(order="C"))
    return component_probabilities, support_codes, log_prior, digest.hexdigest()


def _cell_directory(root: Path, dataset: str, epsilon: float, seed: int) -> Path:
    return root / dataset / f"epsilon_{epsilon:g}" / f"seed_{seed}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = args.panel_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Restricted-mixture output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    seal, _ = verify_sealed_panel(panel_root)
    cell_index = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"])): cell
        for cell in seal["cells"]
    }
    manifests: list[dict[str, Any]] = []
    for expected in expected_cells():
        key = (expected.dataset, expected.epsilon, expected.seed)
        cell = cell_index[key]
        transcript_dir = (
            panel_root
            / expected.dataset
            / f"epsilon_{expected.epsilon:g}"
            / f"seed_{expected.seed}"
            / "public_transcript"
        )
        transcript = verify_public_transcript(transcript_dir)
        measurements = transcript.measurements
        if measurements.strategy_transcript is None:
            raise RuntimeError("RCE transcript is missing its interaction strategy")
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

        component_records = {
            "initial": cell["source_control"]["initial_table"],
            "wp9_control": cell["source_control"]["synthetic"],
            "rce_v1": cell["synthetic"],
        }
        component_paths = tuple(
            _verify_record(component_records[name]) for name in COMPONENT_NAMES
        )
        tables = tuple(np.load(path).astype(np.int32) for path in component_paths)
        if any(len(table) != int(measurements.num_rows) for table in tables):
            raise RuntimeError("A component table differs from the public row count")

        query_answers = np.stack(
            [
                answer_queries(
                    table,
                    transcript.queries,
                    batch_size=int(args.batch_size),
                ).astype(np.float64)
                for table in tables
            ],
            axis=0,
        )
        query_residuals = (
            measurements.target_projected.astype(np.float64).reshape(1, -1)
            - query_answers
        )
        coefficient_residuals = precision.coefficient_coordinates_many(query_residuals)
        component_probabilities, support_codes, log_prior, support_hash = (
            _component_distributions(tables, prior)
        )
        result = solve_restricted_mixture_rce(
            component_probabilities,
            coefficient_residuals,
            log_prior,
            confidence,
            component_names=COMPONENT_NAMES,
            max_iterations=int(args.max_iterations),
        )
        if not result.stage_one["success"] or not result.stage_two["success"]:
            raise RuntimeError(f"Restricted-mixture solver failed for {key}: {result.to_dict()}")
        fractional_answers = result.component_weights @ query_answers
        check_residual = precision.coefficient_coordinates(
            measurements.target_projected.astype(np.float64) - fractional_answers
        )
        if not np.allclose(check_residual, result.residual, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError("Fractional query answers do not reproduce solver residual")

        cell_dir = _cell_directory(
            output_root,
            expected.dataset,
            expected.epsilon,
            expected.seed,
        )
        cell_dir.mkdir(parents=True, exist_ok=False)
        answer_path = cell_dir / "fractional_query_answers.npy"
        np.save(answer_path, fractional_answers.astype(np.float64))
        result_path = cell_dir / "restricted_result.json"
        payload = {
            "protocol_id": PROTOCOL_ID,
            "artifact_role": "released_only_restricted_mixture_rce",
            "dataset": expected.dataset,
            "epsilon": expected.epsilon,
            "seed": expected.seed,
            "private_truth_used": False,
            "allowed_use": "post-seal estimator-vs-integer diagnostic only",
            "component_order": list(COMPONENT_NAMES),
            "component_tables": {
                name: dict(component_records[name]) for name in COMPONENT_NAMES
            },
            "component_query_answer_sha256": hashlib.sha256(
                np.asarray(query_answers, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
            "union_support_size": int(len(support_codes)),
            "union_support_sha256": support_hash,
            "global_relaxed_certificate": False,
            "restricted_result": result.to_dict(),
        }
        write_json(payload, result_path)
        manifest = {
            "dataset": expected.dataset,
            "epsilon": expected.epsilon,
            "seed": expected.seed,
            "restricted_result": _file_record(result_path),
            "fractional_query_answers": _file_record(answer_path),
            "component_weights": result.to_dict()["component_weights"],
            "slack_star": float(result.slack_star),
            "kl_objective": float(result.kl_objective),
            "confidence": dict(result.confidence),
            "union_support_size": int(len(support_codes)),
            "union_support_sha256": support_hash,
        }
        manifests.append(manifest)

    summary_path = output_root / "summary.json"
    summary = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "sealed_released_only_restricted_mixture_rce_panel",
        "num_cells": len(manifests),
        "component_order": list(COMPONENT_NAMES),
        "global_relaxed_certificate": False,
        "true_utility_evaluated": False,
        "offline_evaluation_authorized": True,
        "source_sealed_panel": _file_record(panel_root / "sealed_panel_manifest.json"),
        "protocol": _file_record(PROTOCOL_PATH),
        "solver": _file_record(ROOT / "qdte" / "rce" / "relaxed.py"),
        "runner": _file_record(Path(__file__)),
        "cells": manifests,
    }
    write_json(summary, summary_path)
    seal_payload = {
        **summary,
        "summary": _file_record(summary_path),
    }
    seal_path = output_root / "sealed_manifest.json"
    write_json(seal_payload, seal_path)
    return seal_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve released-only RCE on the fixed initial/WP9/RCE convex hull"
    )
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
