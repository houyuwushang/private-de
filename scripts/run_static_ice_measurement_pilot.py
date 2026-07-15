#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import ensure_dir, write_json
from qdte.measurement.factorization import (
    HierarchicalInteractionTranscript,
    allocate_public_precision,
    compile_hierarchical_pair_strategy,
    measure_hierarchical_pair_interactions,
)
from qdte.measurement.projection import project_simplex
from qdte.schema import TableSchema


PROTOCOL_ID = "SAGE-QDTE-STATIC-ICE-MEASUREMENT-20260714-v1"


@dataclass(frozen=True)
class FullPartitionBlock:
    name: str
    scope: tuple[int, ...]
    shape: tuple[int, ...]
    public_importance: float


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    log_term = math.log(1.0 / float(delta))
    return (math.sqrt(log_term + float(epsilon)) - math.sqrt(log_term)) ** 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def public_pair_scopes(cardinalities: np.ndarray, max_pair_cells: int) -> tuple[tuple[int, int], ...]:
    cards = np.asarray(cardinalities, dtype=np.int64)
    if cards.ndim != 1 or cards.size < 2 or np.any(cards < 2):
        raise ValueError("cardinalities must be a vector of values >= 2")
    if int(max_pair_cells) <= 0:
        raise ValueError("max_pair_cells must be positive")
    return tuple(
        (left, right)
        for left, right in itertools.combinations(range(len(cards)), 2)
        if int(cards[left] * cards[right]) <= int(max_pair_cells)
    )


def exact_low_order_marginals(
    rows: np.ndarray,
    cardinalities: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    X = np.asarray(rows, dtype=np.int64)
    cards = np.asarray(cardinalities, dtype=np.int64)
    oneway = {
        attr: np.bincount(X[:, attr], minlength=int(cards[attr])).astype(np.float64)
        for attr in range(X.shape[1])
    }
    pair_tables: dict[tuple[int, int], np.ndarray] = {}
    for left, right in pairs:
        flat = X[:, left] * int(cards[right]) + X[:, right]
        pair_tables[(left, right)] = np.bincount(
            flat,
            minlength=int(cards[left] * cards[right]),
        ).astype(np.float64).reshape(int(cards[left]), int(cards[right]))
    return oneway, pair_tables


def _full_partition_blocks(
    cardinalities: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[FullPartitionBlock, ...]:
    cards = np.asarray(cardinalities, dtype=np.int64)
    blocks = [
        FullPartitionBlock(
            name=f"oneway:{attr}",
            scope=(attr,),
            shape=(int(cards[attr]),),
            public_importance=float(cards[attr] - 1),
        )
        for attr in range(len(cards))
    ]
    blocks.extend(
        FullPartitionBlock(
            name=f"pair:{left}:{right}",
            scope=(left, right),
            shape=(int(cards[left]), int(cards[right])),
            public_importance=float(cards[left] * cards[right] - 1),
        )
        for left, right in pairs
    )
    return tuple(blocks)


def _full_partition_rho(
    blocks: tuple[FullPartitionBlock, ...],
    rho_total: float,
    mode: str,
) -> dict[str, float]:
    if mode == "legacy_family":
        oneway = [block for block in blocks if len(block.scope) == 1]
        pairs = [block for block in blocks if len(block.scope) == 2]
        oneway_fraction = 0.20 / (0.20 + 0.35)
        allocation = {
            block.name: float(rho_total) * oneway_fraction / len(oneway)
            for block in oneway
        }
        allocation.update(
            {
                block.name: float(rho_total) * (1.0 - oneway_fraction) / len(pairs)
                for block in pairs
            }
        )
        return allocation
    if mode == "public_optimal":
        values = allocate_public_precision(
            np.ones(len(blocks), dtype=np.float64),
            [block.public_importance for block in blocks],
            rho_total,
        )
        return {block.name: float(rho) for block, rho in zip(blocks, values, strict=True)}
    raise ValueError("Full partition allocation must be 'legacy_family' or 'public_optimal'")


def measure_full_partitions(
    rows: np.ndarray,
    cardinalities: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    *,
    public_total: int,
    rho_total: float,
    rng: np.random.Generator,
    allocation_mode: str,
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    exact_oneway, exact_pairs = exact_low_order_marginals(rows, cardinalities, pairs)
    blocks = _full_partition_blocks(cardinalities, pairs)
    rho_by_block = _full_partition_rho(blocks, rho_total, allocation_mode)
    oneway: dict[int, np.ndarray] = {}
    pair_tables: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        rho = rho_by_block[block.name]
        noise_std = math.sqrt(1.0 / (2.0 * rho))
        if len(block.scope) == 1:
            attr = block.scope[0]
            noisy = exact_oneway[attr] + rng.normal(0.0, noise_std, size=block.shape)
            oneway[attr] = project_simplex(noisy, float(public_total)).astype(np.float64)
        else:
            pair = (block.scope[0], block.scope[1])
            noisy = exact_pairs[pair] + rng.normal(0.0, noise_std, size=block.shape)
            pair_tables[pair] = project_simplex(noisy.ravel(), float(public_total)).reshape(block.shape)
    diagnostics = {
        "allocation_mode": allocation_mode,
        "rho_spent": float(sum(rho_by_block.values())),
        "rho_oneway": float(
            sum(rho for name, rho in rho_by_block.items() if name.startswith("oneway:"))
        ),
        "rho_pair": float(sum(rho for name, rho in rho_by_block.items() if name.startswith("pair:"))),
        "num_blocks": len(blocks),
    }
    return oneway, pair_tables, diagnostics


def project_reconstructed_marginals(
    oneway: dict[int, np.ndarray],
    pairs: dict[tuple[int, int], np.ndarray],
    public_total: int,
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    projected_oneway = {
        attr: project_simplex(table, float(public_total)).astype(np.float64)
        for attr, table in oneway.items()
    }
    projected_pairs = {
        pair: project_simplex(table.ravel(), float(public_total)).reshape(table.shape).astype(np.float64)
        for pair, table in pairs.items()
    }
    return projected_oneway, projected_pairs


def target_metrics(
    estimated_oneway: dict[int, np.ndarray],
    estimated_pairs: dict[tuple[int, int], np.ndarray],
    true_oneway: dict[int, np.ndarray],
    true_pairs: dict[tuple[int, int], np.ndarray],
    public_total: int,
) -> dict[str, float]:
    if estimated_oneway.keys() != true_oneway.keys() or estimated_pairs.keys() != true_pairs.keys():
        raise ValueError("Estimated and true marginal collections must have matching scopes")
    normalized_errors: list[np.ndarray] = []
    group_tvds: list[float] = []
    pair_tvds: list[float] = []
    for attr in sorted(true_oneway):
        error = (estimated_oneway[attr] - true_oneway[attr]) / float(public_total)
        normalized_errors.append(error.ravel())
        group_tvds.append(0.5 * float(np.sum(np.abs(error))))
    for pair in sorted(true_pairs):
        error = (estimated_pairs[pair] - true_pairs[pair]) / float(public_total)
        normalized_errors.append(error.ravel())
        tvd = 0.5 * float(np.sum(np.abs(error)))
        group_tvds.append(tvd)
        pair_tvds.append(tvd)
    errors = np.concatenate(normalized_errors)
    return {
        "cell_mae": float(np.mean(np.abs(errors))),
        "cell_rmse": float(np.sqrt(np.mean(errors * errors))),
        "max_cell_error": float(np.max(np.abs(errors))),
        "avg_group_tvd": float(np.mean(group_tvds)),
        "avg_pair_tvd": float(np.mean(pair_tvds)),
        "max_tvd": float(np.max(group_tvds)),
    }


def marginal_consistency_violation(
    oneway: dict[int, np.ndarray],
    pairs: dict[tuple[int, int], np.ndarray],
) -> float:
    violation = 0.0
    for (left, right), table in pairs.items():
        violation = max(
            violation,
            float(np.max(np.abs(np.sum(table, axis=1) - oneway[left]))),
            float(np.max(np.abs(np.sum(table, axis=0) - oneway[right]))),
        )
    return violation


def _allocation_summary(transcript: HierarchicalInteractionTranscript) -> dict[str, Any]:
    return {
        "allocation_mode": transcript.allocation_mode,
        "rho_spent": transcript.rho_spent,
        "rho_oneway": float(
            sum(
                transcript.rho_by_block[block.name]
                for block in transcript.strategy.blocks
                if block.kind == "oneway_contrast"
            )
        ),
        "rho_pair": float(
            sum(
                transcript.rho_by_block[block.name]
                for block in transcript.strategy.blocks
                if block.kind == "pair_interaction"
            )
        ),
        "num_blocks": len(transcript.strategy.blocks),
        "mean_component_variance": float(np.mean(list(transcript.component_variances.values()))),
    }


def _arm_summary(
    *,
    raw_oneway: dict[int, np.ndarray],
    raw_pairs: dict[tuple[int, int], np.ndarray],
    projected_oneway: dict[int, np.ndarray],
    projected_pairs: dict[tuple[int, int], np.ndarray],
    true_oneway: dict[int, np.ndarray],
    true_pairs: dict[tuple[int, int], np.ndarray],
    public_total: int,
    allocation: dict[str, Any],
) -> dict[str, Any]:
    raw_values = np.concatenate(
        [table.ravel() for table in raw_oneway.values()]
        + [table.ravel() for table in raw_pairs.values()]
    )
    return {
        "metrics": target_metrics(
            projected_oneway,
            projected_pairs,
            true_oneway,
            true_pairs,
            public_total,
        ),
        "raw_negative_cells": int(np.sum(raw_values < 0.0)),
        "raw_min_cell": float(np.min(raw_values)),
        "raw_consistency_linf": marginal_consistency_violation(raw_oneway, raw_pairs),
        "p1_consistency_linf": marginal_consistency_violation(projected_oneway, projected_pairs),
        "allocation": allocation,
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    ensure_dir(output_dir)

    schema_path = input_dir / "schema.json"
    encoded_path = input_dir / "real_encoded.npy"
    metadata_path = input_dir / "metadata.json"
    schema = TableSchema.load_json(schema_path)
    rows = np.load(encoded_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    public_total = int(metadata["n_rows"])
    if rows.shape != (public_total, schema.d):
        raise ValueError("Encoded rows do not match public metadata/schema dimensions")
    cards = schema.cardinalities.astype(np.int64)
    pairs = public_pair_scopes(cards, int(args.max_pair_cells))
    if not pairs:
        raise ValueError("Public pair-cell cap selected no pair scopes")
    rho_total = rho_for_epsilon(float(args.epsilon), float(args.delta))
    true_oneway, true_pairs = exact_low_order_marginals(rows, cards, pairs)

    summary: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "offline_target_diagnostic_not_for_release",
        "dataset": str(metadata.get("dataset", input_dir.name)),
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "rho_total": rho_total,
        "seed": int(args.seed),
        "adjacency": "add_remove",
        "public_total": public_total,
        "cardinalities": cards.tolist(),
        "num_pairs": len(pairs),
        "max_pair_cells": int(args.max_pair_cells),
        "input_hashes": {
            "schema": sha256_file(schema_path),
            "real_encoded": sha256_file(encoded_path),
            "metadata": sha256_file(metadata_path),
        },
        "arms": {},
    }

    for arm_index, mode in enumerate(("legacy_family", "public_optimal")):
        oneway, pair_tables, allocation = measure_full_partitions(
            rows,
            cards,
            pairs,
            public_total=public_total,
            rho_total=rho_total,
            rng=np.random.default_rng(int(args.seed) * 10_000 + 100 + arm_index),
            allocation_mode=mode,
        )
        summary["arms"][f"o1_{mode}"] = _arm_summary(
            raw_oneway=oneway,
            raw_pairs=pair_tables,
            projected_oneway=oneway,
            projected_pairs=pair_tables,
            true_oneway=true_oneway,
            true_pairs=true_pairs,
            public_total=public_total,
            allocation=allocation,
        )

    strategy = compile_hierarchical_pair_strategy(cards, pairs)
    for arm_index, mode in enumerate(("equal", "public_optimal")):
        transcript = measure_hierarchical_pair_interactions(
            rows,
            strategy,
            public_total=public_total,
            rho_total=rho_total,
            rng=np.random.default_rng(int(args.seed) * 10_000 + 200 + arm_index),
            allocation_mode=mode,
        )
        reconstructed = transcript.reconstruct()
        projected_oneway, projected_pairs = project_reconstructed_marginals(
            reconstructed.oneway,
            reconstructed.pairs,
            public_total,
        )
        arm_name = f"ice_{mode}"
        summary["arms"][arm_name] = _arm_summary(
            raw_oneway=reconstructed.oneway,
            raw_pairs=reconstructed.pairs,
            projected_oneway=projected_oneway,
            projected_pairs=projected_pairs,
            true_oneway=true_oneway,
            true_pairs=true_pairs,
            public_total=public_total,
            allocation=_allocation_summary(transcript),
        )
        write_json(transcript.to_public_dict(), output_dir / f"{arm_name}_transcript.json")

    write_json(summary, output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare complete partitions with hierarchical pure-interaction measurements."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pair-cells", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    summary = run_pilot(parse_args())
    compact = {
        name: arm["metrics"]
        for name, arm in summary["arms"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
