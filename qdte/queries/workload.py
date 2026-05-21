from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.queries.types import OP_EQ, OP_LE, OP_RANGE, QueryBuilder, QueryCatalogue
from qdte.schema import TableSchema


@dataclass
class WorkloadGroup:
    name: str
    family: str
    query_indices: np.ndarray
    sensitivity_l2: float
    is_partition: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "query_indices": self.query_indices.tolist(),
            "sensitivity_l2": self.sensitivity_l2,
            "is_partition": self.is_partition,
        }


def filter_workload_groups(groups: list[WorkloadGroup], keep_indices: np.ndarray) -> list[WorkloadGroup]:
    keep = np.asarray(keep_indices, dtype=np.int32)
    index_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(keep.tolist())}
    filtered: list[WorkloadGroup] = []
    for group in groups:
        remapped = [index_map[int(idx)] for idx in group.query_indices if int(idx) in index_map]
        if remapped:
            filtered.append(
                WorkloadGroup(
                    name=group.name,
                    family=group.family,
                    query_indices=np.asarray(remapped, dtype=np.int32),
                    sensitivity_l2=group.sensitivity_l2,
                    is_partition=group.is_partition,
                )
            )
    return filtered


def _range_intervals(cardinality: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if cardinality <= 1:
        return [(0, 0)]
    all_ranges: list[tuple[int, int]] = []
    for lo in range(cardinality):
        for hi in range(lo, cardinality):
            all_ranges.append((lo, hi))
    if len(all_ranges) <= count:
        return all_ranges
    idx = rng.choice(len(all_ranges), size=count, replace=False)
    return [all_ranges[int(i)] for i in idx]


def _cap_queries(builder: QueryBuilder, max_queries: int) -> bool:
    return len(builder.names) >= max_queries


def build_workload(schema: TableSchema, config: dict[str, Any]) -> tuple[QueryCatalogue, list[WorkloadGroup]]:
    cfg = config.get("workload", {})
    max_terms = int(cfg.get("max_terms", 4))
    max_queries = int(cfg.get("max_queries", 10000))
    max_2way_cells = int(cfg.get("max_2way_cells", 5000))
    range_per_attr = int(cfg.get("range_intervals_per_num_attr", 64))
    mixed_per_pair = int(cfg.get("mixed_queries_per_pair", 64))
    random_seed = int(cfg.get("random_seed", 0))
    rng = np.random.default_rng(random_seed)
    builder = QueryBuilder(max_terms=max_terms)

    groups: list[WorkloadGroup] = []

    def mark_group(name: str, family: str, start: int, is_partition: bool, sensitivity: float) -> None:
        end = len(builder.names)
        if end > start:
            groups.append(
                WorkloadGroup(
                    name=name,
                    family=family,
                    query_indices=np.arange(start, end, dtype=np.int32),
                    sensitivity_l2=float(sensitivity),
                    is_partition=is_partition,
                )
            )

    if bool(cfg.get("include_oneway", True)):
        for attr, col in enumerate(schema.columns):
            start = len(builder.names)
            for val in range(col.cardinality):
                builder.add(
                    [(attr, OP_EQ, val, val, val)],
                    name=f"{col.name}={val}",
                    group=f"oneway:{attr}",
                    family="oneway",
                )
            mark_group(f"oneway:{attr}", "oneway", start, True, 1.0)
            if _cap_queries(builder, max_queries):
                break

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_2way_cat", True)):
        pairs = list(itertools.combinations(range(schema.d), 2))
        rng.shuffle(pairs)
        used_cells = 0
        for a, b in pairs:
            ka = schema.columns[a].cardinality
            kb = schema.columns[b].cardinality
            cells = ka * kb
            if used_cells + cells > max_2way_cells:
                continue
            start = len(builder.names)
            for va in range(ka):
                for vb in range(kb):
                    builder.add(
                        [(a, OP_EQ, va, va, va), (b, OP_EQ, vb, vb, vb)],
                        name=f"{schema.columns[a].name}={va}&{schema.columns[b].name}={vb}",
                        group=f"twoway:{a}:{b}",
                        family="twoway",
                    )
                    if _cap_queries(builder, max_queries):
                        break
                if _cap_queries(builder, max_queries):
                    break
            mark_group(f"twoway:{a}:{b}", "twoway", start, True, 1.0)
            used_cells += cells
            if _cap_queries(builder, max_queries):
                break

    numerical = schema.numerical_indices
    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_prefix", True)):
        for attr in numerical:
            col = schema.columns[attr]
            start = len(builder.names)
            for threshold in range(col.cardinality):
                builder.add(
                    [(attr, OP_LE, threshold, 0, threshold)],
                    name=f"{col.name}<={threshold}",
                    group=f"prefix:{attr}",
                    family="prefix",
                )
                if _cap_queries(builder, max_queries):
                    break
            size = max(1, len(builder.names) - start)
            mark_group(f"prefix:{attr}", "prefix", start, False, math.sqrt(size))
            if _cap_queries(builder, max_queries):
                break

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_range", True)):
        for attr in numerical:
            col = schema.columns[attr]
            start = len(builder.names)
            for lo, hi in _range_intervals(col.cardinality, range_per_attr, rng):
                builder.add(
                    [(attr, OP_RANGE, lo, lo, hi)],
                    name=f"{col.name}[{lo},{hi}]",
                    group=f"range:{attr}",
                    family="range",
                )
                if _cap_queries(builder, max_queries):
                    break
            size = max(1, len(builder.names) - start)
            mark_group(f"range:{attr}", "range", start, False, math.sqrt(size))
            if _cap_queries(builder, max_queries):
                break

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_mixed", True)) and numerical:
        categorical = [idx for idx in range(schema.d) if idx not in numerical]
        pairs = [(c, n) for c in categorical for n in numerical]
        rng.shuffle(pairs)
        for c_attr, n_attr in pairs:
            start = len(builder.names)
            c_col = schema.columns[c_attr]
            n_col = schema.columns[n_attr]
            for _ in range(mixed_per_pair):
                c_val = int(rng.integers(0, c_col.cardinality))
                if rng.random() < 0.5:
                    threshold = int(rng.integers(0, n_col.cardinality))
                    term = (n_attr, OP_LE, threshold, 0, threshold)
                    suffix = f"<={threshold}"
                else:
                    lo = int(rng.integers(0, n_col.cardinality))
                    hi = int(rng.integers(lo, n_col.cardinality))
                    term = (n_attr, OP_RANGE, lo, lo, hi)
                    suffix = f"[{lo},{hi}]"
                builder.add(
                    [(c_attr, OP_EQ, c_val, c_val, c_val), term],
                    name=f"{c_col.name}={c_val}&{n_col.name}{suffix}",
                    group=f"mixed:{c_attr}:{n_attr}",
                    family="mixed",
                )
                if _cap_queries(builder, max_queries):
                    break
            size = max(1, len(builder.names) - start)
            mark_group(f"mixed:{c_attr}:{n_attr}", "mixed", start, False, math.sqrt(size))
            if _cap_queries(builder, max_queries):
                break

    qcat = builder.build()
    groups = [g for g in groups if len(g.query_indices) > 0 and int(g.query_indices.max()) < qcat.m]
    return qcat, groups
