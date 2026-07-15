#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import write_json
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.measurement.workload_factorization import (
    allocation_risk,
    factorize_public_workload,
    optimal_rho_for_importance,
    query_public_scope,
)
from qdte.queries.types import QueryCatalogue
from scripts.run_coverage_refinement_wp8a import sha256_file
from scripts.run_static_ice_wp9_cell import METHOD_ID, PROTOCOL_ID as WP9_PROTOCOL_ID


PROTOCOL_ID = "SAGE-QDTE-ICE-WP9A-PUBLIC-ROUTE-DIAGNOSTIC-20260715-v1"
SCORE_PROFILES = ("total_excess", "null_standardized")
PROTOCOL_PATH = ROOT / "docs" / "SAGE_QDTE_ICE_WP9A_PUBLIC_ROUTE_DIAGNOSTIC_PROTOCOL_20260715.md"


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
    current = _file_record(path)
    if current != record:
        raise RuntimeError(f"sealed artifact changed: {path}")
    return path


def _jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def maximum_spanning_tree(
    width: int,
    scores: dict[tuple[int, int], float],
) -> tuple[tuple[tuple[int, int], ...], float]:
    nodes = int(width)
    if nodes < 2:
        raise ValueError("tree width must be at least two")
    expected = {(left, right) for left in range(nodes) for right in range(left + 1, nodes)}
    if set(scores) != expected:
        raise ValueError("scores must define the complete public pair graph")
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in scores.values()):
        raise ValueError("tree scores must be finite and nonnegative")

    parent = list(range(nodes))
    rank = [0] * nodes

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    selected: list[tuple[int, int]] = []
    for edge, _ in sorted(scores.items(), key=lambda item: (-float(item[1]), item[0])):
        if union(*edge):
            selected.append(edge)
            if len(selected) == nodes - 1:
                break
    if len(selected) != nodes - 1:
        raise RuntimeError("complete graph did not yield a spanning tree")
    tree = tuple(sorted(selected))
    return tree, minimum_tree_replacement_margin(nodes, scores, tree)


def minimum_tree_replacement_margin(
    width: int,
    scores: dict[tuple[int, int], float],
    tree: tuple[tuple[int, int], ...],
) -> float:
    tree_set = set(tree)
    adjacency: dict[int, set[int]] = {node: set() for node in range(int(width))}
    for left, right in tree:
        adjacency[left].add(right)
        adjacency[right].add(left)
    margins: list[float] = []
    for removed in tree:
        seen = {removed[0]}
        stack = [removed[0]]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                edge = (min(node, neighbor), max(node, neighbor))
                if edge == removed or neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        alternatives = [
            float(score)
            for edge, score in scores.items()
            if edge not in tree_set and ((edge[0] in seen) != (edge[1] in seen))
        ]
        best_alternative = max(alternatives) if alternatives else 0.0
        margins.append(float(scores[removed]) - best_alternative)
    return float(min(margins))


def released_tree_scores(
    transcript: HierarchicalInteractionTranscript,
    profile: str,
) -> dict[tuple[int, int], float]:
    if profile not in SCORE_PROFILES:
        raise ValueError(f"unknown tree score profile {profile!r}")
    scores: dict[tuple[int, int], float] = {}
    for block in transcript.strategy.blocks:
        if block.kind != "pair_interaction":
            continue
        values = np.asarray(transcript.noisy_components[block.name], dtype=np.float64)
        variance = float(transcript.component_variances[block.name])
        dimension = int(block.dimension)
        excess = max(0.0, float(np.dot(values.reshape(-1), values.reshape(-1))) / variance - dimension)
        score = excess if profile == "total_excess" else excess / math.sqrt(2.0 * dimension)
        scores[(int(block.scope[0]), int(block.scope[1]))] = float(score)
    return scores


def _family_counts(qcat: QueryCatalogue, qids: tuple[int, ...]) -> dict[str, int]:
    return dict(sorted(Counter(qcat.families[qid] for qid in qids).items()))


def _partition_coverage(
    raw_groups: list[dict[str, Any]],
    supported: set[int],
) -> dict[str, int]:
    partition = [group for group in raw_groups if bool(group.get("is_partition", False))]
    fully_supported = 0
    partially_supported = 0
    unsupported = 0
    for group in partition:
        qids = {int(value) for value in group.get("query_indices", [])}
        overlap = len(qids & supported)
        if overlap == len(qids):
            fully_supported += 1
        elif overlap:
            partially_supported += 1
        else:
            unsupported += 1
    return {
        "total_partition_blocks": len(partition),
        "fully_supported_partition_blocks": fully_supported,
        "partially_supported_partition_blocks": partially_supported,
        "unsupported_partition_blocks": unsupported,
    }


def _movement_summary(
    transcript: HierarchicalInteractionTranscript,
    optimal: dict[str, float],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for block in transcript.strategy.blocks:
        current = float(transcript.rho_by_block[block.name])
        proposed = float(optimal[block.name])
        ratio = proposed / current if current > 0.0 else math.inf
        records.append(
            {
                "block": block.name,
                "kind": block.kind,
                "scope": list(block.scope),
                "dimension": block.dimension,
                "sensitivity_l2": block.sensitivity_l2,
                "current_rho": current,
                "optimal_rho": proposed,
                "optimal_over_current": ratio,
            }
        )
    ordered = sorted(records, key=lambda item: (float(item["optimal_over_current"]), item["block"]))
    current_pair = sum(
        float(transcript.rho_by_block[block.name])
        for block in transcript.strategy.blocks
        if block.kind == "pair_interaction"
    )
    optimal_pair = sum(
        float(optimal[block.name])
        for block in transcript.strategy.blocks
        if block.kind == "pair_interaction"
    )
    return {
        "current_pair_rho_share": current_pair / transcript.rho_total,
        "optimal_pair_rho_share": optimal_pair / transcript.rho_total,
        "largest_decreases": ordered[:10],
        "largest_increases": list(reversed(ordered[-10:])),
    }


def _risk_profile(
    transcript: HierarchicalInteractionTranscript,
    importance: dict[str, float],
) -> dict[str, Any]:
    optimal = optimal_rho_for_importance(
        transcript.strategy,
        importance,
        transcript.rho_total,
    )
    current_risk = allocation_risk(
        transcript.strategy,
        importance,
        transcript.rho_by_block,
    )
    optimal_risk = allocation_risk(transcript.strategy, importance, optimal)
    return {
        "current_risk": current_risk,
        "optimal_risk": optimal_risk,
        "optimal_over_current": optimal_risk / current_risk,
        "movement": _movement_summary(transcript, optimal),
    }


def _tree_stability(
    trees: dict[int, tuple[tuple[int, int], ...]],
    margins: dict[int, float],
) -> dict[str, Any]:
    seeds = sorted(trees)
    sets = {seed: set(trees[seed]) for seed in seeds}
    pairwise = {
        f"{left}|{right}": _jaccard(sets[left], sets[right])
        for position, left in enumerate(seeds)
        for right in seeds[position + 1 :]
    }
    frequencies: Counter[tuple[int, int]] = Counter(
        edge for seed in seeds for edge in sets[seed]
    )
    intersection = set.intersection(*(sets[seed] for seed in seeds))
    union = set.union(*(sets[seed] for seed in seeds))
    return {
        "pairwise_jaccard": pairwise,
        "mean_pairwise_jaccard": float(np.mean(list(pairwise.values()))),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "selected_in_all_seeds": [list(edge) for edge in sorted(intersection)],
        "selected_in_at_least_two_seeds": [
            list(edge) for edge, count in sorted(frequencies.items()) if count >= 2
        ],
        "minimum_replacement_margin_by_seed": {
            str(seed): float(margins[seed]) for seed in seeds
        },
        "trees": {
            str(seed): [list(edge) for edge in trees[seed]] for seed in seeds
        },
    }


def run_audit(panel_root: Path, output_root: Path) -> dict[str, Any]:
    panel = panel_root.resolve()
    output = output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diagnostic output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seal_path = panel / "sealed_panel_manifest.json"
    seal = _read_json(seal_path)
    if seal.get("protocol_id") != WP9_PROTOCOL_ID or seal.get("method_id") != METHOD_ID:
        raise RuntimeError("input is not the frozen WP9 panel")
    if seal.get("true_utility_evaluated") is not False:
        raise RuntimeError("WP9 panel was not sealed before utility evaluation")
    plan_path = _verify_record(seal["panel_plan"])
    plan = _read_json(plan_path)
    plan_cells = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"])): cell
        for cell in plan.get("cells", [])
    }
    sealed_cells = {
        (str(cell["dataset"]), float(cell["epsilon"]), int(cell["seed"])): cell
        for cell in seal.get("cells", [])
    }
    if set(plan_cells) != set(sealed_cells) or len(sealed_cells) != 24:
        raise RuntimeError("WP9 public-route audit requires the complete 24-cell seal")

    transcript_by_cell: dict[tuple[str, float, int], HierarchicalInteractionTranscript] = {}
    measurement_records: dict[str, Any] = {}
    for key, cell in sorted(sealed_cells.items()):
        measurement_path = _verify_record(cell["measurement"])
        measurement_records[f"{key[0]}|{key[1]:g}|{key[2]}"] = cell["measurement"]
        measurement = _read_json(measurement_path)
        transcript_by_cell[key] = HierarchicalInteractionTranscript.from_public_dict(
            measurement["strategy_transcript"]
        )

    datasets = sorted({key[0] for key in transcript_by_cell})
    epsilons = sorted({key[1] for key in transcript_by_cell})
    seeds = sorted({key[2] for key in transcript_by_cell})
    risk_results: dict[str, Any] = {}
    public_inputs: dict[str, Any] = {}
    risk_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        representative_key = (dataset, epsilons[0], seeds[0])
        representative = transcript_by_cell[representative_key]
        input_dir = Path(str(plan_cells[representative_key]["input_dir"])).resolve()
        query_path = input_dir / "queries_full.json"
        group_path = input_dir / "workload_groups.json"
        schema_path = input_dir / "schema.json"
        public_inputs[dataset] = {
            "queries_full": _file_record(query_path),
            "workload_groups": _file_record(group_path),
            "schema": _file_record(schema_path),
        }
        qcat = QueryCatalogue.from_dict(_read_json(query_path))
        raw_groups = json.loads(group_path.read_text(encoding="utf-8"))
        if not isinstance(raw_groups, list) or not all(isinstance(item, dict) for item in raw_groups):
            raise ValueError(f"invalid public workload groups: {group_path}")

        cell_mask = np.asarray(
            [family in {"oneway", "twoway"} for family in qcat.families],
            dtype=bool,
        )
        cell_reference = factorize_public_workload(
            representative.strategy,
            qcat,
            include=cell_mask,
        )
        reference_differences = [
            abs(
                float(cell_reference.importance_by_block[block.name])
                - float(block.public_importance)
            )
            for block in representative.strategy.blocks
        ]
        reference_relative = [
            difference / max(float(block.public_importance), 1.0e-300)
            for difference, block in zip(
                reference_differences,
                representative.strategy.blocks,
                strict=True,
            )
        ]
        if max(reference_relative, default=0.0) > 1.0e-10:
            raise RuntimeError(f"cell-reference importance mismatch for {dataset}")

        full = factorize_public_workload(representative.strategy, qcat)
        group_counts = Counter(qcat.groups)
        group_weights = np.asarray(
            [1.0 / float(group_counts[group]) for group in qcat.groups],
            dtype=np.float64,
        )
        equal_group = factorize_public_workload(
            representative.strategy,
            qcat,
            weights=group_weights,
        )
        scope_counts = Counter(len(query_public_scope(qcat, qid)) for qid in range(qcat.m))
        factorization_summary = {
            "num_queries": int(qcat.m),
            "scope_order_counts": {
                str(order): int(count) for order, count in sorted(scope_counts.items())
            },
            "cell_reference_max_relative_error": max(reference_relative, default=0.0),
            "full_reconstructable_query_l2": {
                "supported_queries": len(full.supported_query_ids),
                "unsupported_queries": len(full.unsupported_query_ids),
                "supported_weight": full.supported_weight,
                "unsupported_weight": full.unsupported_weight,
                "supported_by_family": _family_counts(qcat, full.supported_query_ids),
                "unsupported_by_family": _family_counts(qcat, full.unsupported_query_ids),
                "unsupported_scope_orders": {
                    str(order): int(count)
                    for order, count in full.unsupported_scope_orders.items()
                },
                "partition_coverage": _partition_coverage(
                    raw_groups,
                    set(full.supported_query_ids),
                ),
            },
            "equal_public_group_l2": {
                "supported_queries": len(equal_group.supported_query_ids),
                "unsupported_queries": len(equal_group.unsupported_query_ids),
                "supported_weight": equal_group.supported_weight,
                "unsupported_weight": equal_group.unsupported_weight,
            },
        }

        profile_results: dict[str, Any] = {}
        for name, factorization in (
            ("full_reconstructable_query_l2", full),
            ("equal_public_group_l2", equal_group),
        ):
            per_cell = []
            for epsilon in epsilons:
                for seed in seeds:
                    transcript = transcript_by_cell[(dataset, epsilon, seed)]
                    result = _risk_profile(
                        transcript,
                        factorization.importance_by_block,
                    )
                    per_cell.append(
                        {
                            "epsilon": epsilon,
                            "seed": seed,
                            "optimal_over_current": result["optimal_over_current"],
                        }
                    )
            representative_profile = _risk_profile(
                representative,
                factorization.importance_by_block,
            )
            ratios = [float(item["optimal_over_current"]) for item in per_cell]
            representative_profile["all_cell_ratios"] = per_cell
            representative_profile["ratio_min"] = min(ratios)
            representative_profile["ratio_max"] = max(ratios)
            profile_results[name] = representative_profile
            risk_rows.append(
                {
                    "dataset": dataset,
                    "profile": name,
                    "supported_queries": len(factorization.supported_query_ids),
                    "unsupported_queries": len(factorization.unsupported_query_ids),
                    "optimal_over_current": representative_profile["optimal_over_current"],
                    "current_pair_rho_share": representative_profile["movement"][
                        "current_pair_rho_share"
                    ],
                    "optimal_pair_rho_share": representative_profile["movement"][
                        "optimal_pair_rho_share"
                    ],
                }
            )
        risk_results[dataset] = {
            "factorization": factorization_summary,
            "profiles": profile_results,
        }

    tree_results: dict[str, Any] = {}
    tree_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        by_epsilon: dict[str, Any] = {}
        width = len(transcript_by_cell[(dataset, epsilons[0], seeds[0])].strategy.cardinalities)
        for epsilon in epsilons:
            by_profile: dict[str, Any] = {}
            profile_trees: dict[str, dict[int, tuple[tuple[int, int], ...]]] = {}
            for profile in SCORE_PROFILES:
                trees: dict[int, tuple[tuple[int, int], ...]] = {}
                margins: dict[int, float] = {}
                positive_edges: dict[str, int] = {}
                positive_tree_edges: dict[str, int] = {}
                tree_score_totals: dict[str, float] = {}
                tree_dimension_means: dict[str, float] = {}
                for seed in seeds:
                    transcript = transcript_by_cell[(dataset, epsilon, seed)]
                    scores = released_tree_scores(transcript, profile)
                    tree, margin = maximum_spanning_tree(width, scores)
                    dimensions = {
                        (int(block.scope[0]), int(block.scope[1])): int(block.dimension)
                        for block in transcript.strategy.blocks
                        if block.kind == "pair_interaction"
                    }
                    trees[seed] = tree
                    margins[seed] = margin
                    positive_edges[str(seed)] = sum(value > 0.0 for value in scores.values())
                    positive_tree_edges[str(seed)] = sum(scores[edge] > 0.0 for edge in tree)
                    tree_score_totals[str(seed)] = float(sum(scores[edge] for edge in tree))
                    tree_dimension_means[str(seed)] = float(
                        np.mean([dimensions[edge] for edge in tree])
                    )
                stability = _tree_stability(trees, margins)
                stability["positive_score_edges_by_seed"] = positive_edges
                stability["positive_tree_edges_by_seed"] = positive_tree_edges
                stability["tree_score_total_by_seed"] = tree_score_totals
                stability["mean_selected_edge_dimension_by_seed"] = tree_dimension_means
                by_profile[profile] = stability
                profile_trees[profile] = trees
                tree_rows.append(
                    {
                        "dataset": dataset,
                        "epsilon": epsilon,
                        "profile": profile,
                        "mean_pairwise_jaccard": stability["mean_pairwise_jaccard"],
                        "intersection_count": stability["intersection_count"],
                        "union_count": stability["union_count"],
                        "selected_in_at_least_two_count": len(
                            stability["selected_in_at_least_two_seeds"]
                        ),
                        "minimum_replacement_margin": min(margins.values()),
                        "mean_selected_edge_dimension": float(
                            np.mean(list(tree_dimension_means.values()))
                        ),
                    }
                )
            by_profile["score_profile_overlap_by_seed"] = {
                str(seed): _jaccard(
                    set(profile_trees[SCORE_PROFILES[0]][seed]),
                    set(profile_trees[SCORE_PROFILES[1]][seed]),
                )
                for seed in seeds
            }
            by_epsilon[f"{epsilon:g}"] = by_profile
        tree_results[dataset] = by_epsilon

    summary = {
        "protocol_id": PROTOCOL_ID,
        "source_wp9_protocol_id": WP9_PROTOCOL_ID,
        "source_method_id": METHOD_ID,
        "artifact_role": "public_and_released_only_route_diagnostic",
        "true_data_loaded": False,
        "true_answers_loaded": False,
        "true_utility_evaluated": False,
        "method_promoted": False,
        "diagnostic_source": _file_record(Path(__file__)),
        "protocol_document": _file_record(PROTOCOL_PATH),
        "source_panel": _file_record(seal_path),
        "source_plan": _file_record(plan_path),
        "measurement_records": measurement_records,
        "public_inputs": public_inputs,
        "risk_results": risk_results,
        "released_tree_results": tree_results,
    }
    write_json(summary, output / "public_route_diagnostics.json")
    with (output / "allocation_risk_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risk_rows[0]))
        writer.writeheader()
        writer.writerows(risk_rows)
    with (output / "released_tree_stability.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tree_rows[0]))
        writer.writeheader()
        writer.writerows(tree_rows)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "public_route_diagnostic_manifest",
        "true_data_loaded": False,
        "true_utility_evaluated": False,
        "artifacts": {
            name: _file_record(output / name)
            for name in (
                "public_route_diagnostics.json",
                "allocation_risk_summary.csv",
                "released_tree_stability.csv",
            )
        },
    }
    write_json(manifest, output / "manifest.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit WP9 public allocation risk and released sparse-tree stability."
    )
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=Path("outputs/static_ice_wp9_formal_20260715"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args.panel_root, args.output_root)
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "artifact_role": result["artifact_role"],
                "datasets": sorted(result["risk_results"]),
                "true_utility_evaluated": result["true_utility_evaluated"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
