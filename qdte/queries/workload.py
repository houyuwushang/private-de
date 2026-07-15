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


def _partition_intervals(cardinality: int, bins: int) -> list[tuple[int, int]]:
    if cardinality <= 0:
        return []
    num_bins = max(1, min(int(bins), int(cardinality)))
    intervals: list[tuple[int, int]] = []
    for bin_idx in range(num_bins):
        lo = int(bin_idx * cardinality // num_bins)
        hi = int((bin_idx + 1) * cardinality // num_bins) - 1
        if hi >= lo:
            intervals.append((lo, hi))
    return intervals


def _cap_queries(builder: QueryBuilder, max_queries: int) -> bool:
    return len(builder.names) >= max_queries


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(default)
        return [int(part.strip()) for part in stripped.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(f"Expected integer/list/comma-separated string, got {type(value).__name__}")


def _scope_label(schema: TableSchema, attrs: tuple[int, ...]) -> str:
    return ":".join(str(int(attr)) for attr in attrs)


def _term_label(schema: TableSchema, term: tuple[int, int, int, int, int]) -> str:
    attr, op, value, lo, hi = term
    name = schema.columns[int(attr)].name
    if op == OP_EQ:
        return f"{name}={int(value)}"
    if op == OP_LE:
        return f"{name}<={int(value)}"
    if op == OP_RANGE:
        return f"{name}[{int(lo)},{int(hi)}]"
    return f"{name}?{int(op)}"


def _sample_eq_terms(
    schema: TableSchema,
    attrs: tuple[int, ...],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int, int]]:
    terms: list[tuple[int, int, int, int, int]] = []
    for attr in attrs:
        card = int(schema.columns[int(attr)].cardinality)
        val = int(rng.integers(0, card))
        terms.append((int(attr), OP_EQ, val, val, val))
    return terms


def _sample_kway_eq_queries(
    *,
    builder: QueryBuilder,
    groups: list[WorkloadGroup],
    schema: TableSchema,
    rng: np.random.Generator,
    orders: list[int],
    queries_per_order: int,
    max_queries: int,
) -> None:
    attrs_all = list(range(schema.d))
    for order in sorted(set(int(order) for order in orders)):
        if order <= 0 or order > len(attrs_all) or order > builder.max_terms:
            continue
        start = len(builder.names)
        attempts = 0
        max_attempts = max(queries_per_order * 20, 100)
        while len(builder.names) - start < queries_per_order and attempts < max_attempts:
            attempts += 1
            attrs = tuple(sorted(int(x) for x in rng.choice(attrs_all, size=order, replace=False).tolist()))
            terms = _sample_eq_terms(schema, attrs, rng)
            builder.add(
                terms,
                name="&".join(_term_label(schema, term) for term in terms),
                group=f"kway:{order}:{_scope_label(schema, attrs)}",
                family="kway",
            )
            if _cap_queries(builder, max_queries):
                break
        size = len(builder.names) - start
        if size > 0:
            groups.append(
                WorkloadGroup(
                    name=f"kway:{order}",
                    family="kway",
                    query_indices=np.arange(start, len(builder.names), dtype=np.int32),
                    sensitivity_l2=math.sqrt(size),
                    is_partition=False,
                )
            )
        if _cap_queries(builder, max_queries):
            break


def _sample_kway_numeric_queries(
    *,
    builder: QueryBuilder,
    groups: list[WorkloadGroup],
    schema: TableSchema,
    rng: np.random.Generator,
    family: str,
    orders: list[int],
    queries_per_order: int,
    max_queries: int,
) -> None:
    numerical = list(int(attr) for attr in schema.numerical_indices)
    if not numerical:
        return
    attrs_all = list(range(schema.d))
    for order in sorted(set(int(order) for order in orders)):
        if order <= 0 or order > len(attrs_all) or order > builder.max_terms:
            continue
        start = len(builder.names)
        attempts = 0
        max_attempts = max(queries_per_order * 30, 100)
        while len(builder.names) - start < queries_per_order and attempts < max_attempts:
            attempts += 1
            n_attr = int(rng.choice(numerical))
            remaining = [attr for attr in attrs_all if attr != n_attr]
            if order - 1 > len(remaining):
                continue
            cond_attrs = tuple(
                sorted(int(x) for x in rng.choice(remaining, size=order - 1, replace=False).tolist())
            )
            terms = _sample_eq_terms(schema, cond_attrs, rng)
            n_col = schema.columns[n_attr]
            if family == "kway_prefix":
                threshold = int(rng.integers(0, n_col.cardinality))
                numeric_term = (n_attr, OP_LE, threshold, 0, threshold)
            elif family == "kway_range":
                lo = int(rng.integers(0, n_col.cardinality))
                hi = int(rng.integers(lo, n_col.cardinality))
                numeric_term = (n_attr, OP_RANGE, lo, lo, hi)
            else:
                raise ValueError(f"Unknown k-way numeric family {family!r}")
            full_terms = sorted([*terms, numeric_term], key=lambda term: int(term[0]))
            attrs = tuple(int(term[0]) for term in full_terms)
            builder.add(
                full_terms,
                name="&".join(_term_label(schema, term) for term in full_terms),
                group=f"{family}:{order}:{_scope_label(schema, attrs)}",
                family=family,
            )
            if _cap_queries(builder, max_queries):
                break
        size = len(builder.names) - start
        if size > 0:
            groups.append(
                WorkloadGroup(
                    name=f"{family}:{order}",
                    family=family,
                    query_indices=np.arange(start, len(builder.names), dtype=np.int32),
                    sensitivity_l2=math.sqrt(size),
                    is_partition=False,
                )
            )
        if _cap_queries(builder, max_queries):
            break


def _sample_kway_mixed_queries(
    *,
    builder: QueryBuilder,
    groups: list[WorkloadGroup],
    schema: TableSchema,
    rng: np.random.Generator,
    orders: list[int],
    queries_per_order: int,
    max_queries: int,
) -> None:
    numerical = set(int(attr) for attr in schema.numerical_indices)
    if not numerical:
        return
    attrs_all = list(range(schema.d))
    for order in sorted(set(int(order) for order in orders)):
        if order <= 0 or order > len(attrs_all) or order > builder.max_terms:
            continue
        start = len(builder.names)
        attempts = 0
        max_attempts = max(queries_per_order * 40, 200)
        while len(builder.names) - start < queries_per_order and attempts < max_attempts:
            attempts += 1
            attrs = tuple(sorted(int(x) for x in rng.choice(attrs_all, size=order, replace=False).tolist()))
            if not any(attr in numerical for attr in attrs):
                continue
            terms: list[tuple[int, int, int, int, int]] = []
            for attr in attrs:
                col = schema.columns[int(attr)]
                if attr not in numerical:
                    val = int(rng.integers(0, col.cardinality))
                    terms.append((int(attr), OP_EQ, val, val, val))
                elif rng.random() < 0.5:
                    threshold = int(rng.integers(0, col.cardinality))
                    terms.append((int(attr), OP_LE, threshold, 0, threshold))
                else:
                    lo = int(rng.integers(0, col.cardinality))
                    hi = int(rng.integers(lo, col.cardinality))
                    terms.append((int(attr), OP_RANGE, lo, lo, hi))
            builder.add(
                terms,
                name="&".join(_term_label(schema, term) for term in terms),
                group=f"kway_mixed:{order}:{_scope_label(schema, attrs)}",
                family="kway_mixed",
            )
            if _cap_queries(builder, max_queries):
                break
        size = len(builder.names) - start
        if size > 0:
            groups.append(
                WorkloadGroup(
                    name=f"kway_mixed:{order}",
                    family="kway_mixed",
                    query_indices=np.arange(start, len(builder.names), dtype=np.int32),
                    sensitivity_l2=math.sqrt(size),
                    is_partition=False,
                )
            )
        if _cap_queries(builder, max_queries):
            break


def _sample_orthogonal_kway_mixed_queries(
    *,
    builder: QueryBuilder,
    groups: list[WorkloadGroup],
    schema: TableSchema,
    rng: np.random.Generator,
    orders: list[int],
    scopes_per_order: int,
    range_bins: int,
    max_cells_per_group: int,
    max_queries: int,
) -> None:
    if scopes_per_order <= 0:
        return
    numerical = set(int(attr) for attr in schema.numerical_indices)
    categorical = [attr for attr in range(schema.d) if attr not in numerical]
    if not categorical or not numerical:
        return
    attrs_all = list(range(schema.d))
    seen = getattr(builder, "_seen")
    for order in sorted(set(int(order) for order in orders)):
        if order <= 1 or order > len(attrs_all) or order > builder.max_terms:
            continue
        scopes = [
            tuple(int(attr) for attr in attrs)
            for attrs in itertools.combinations(attrs_all, order)
            if any(attr in numerical for attr in attrs) and any(attr not in numerical for attr in attrs)
        ]
        rng.shuffle(scopes)
        used_scopes = 0
        for attrs in scopes:
            axes: list[list[tuple[int, int, int, int, int]]] = []
            expected_cells = 1
            for attr in attrs:
                col = schema.columns[int(attr)]
                if attr in numerical:
                    intervals = _partition_intervals(int(col.cardinality), range_bins)
                    if not intervals:
                        expected_cells = 0
                        break
                    axes.append([(int(attr), OP_RANGE, lo, lo, hi) for lo, hi in intervals])
                    expected_cells *= len(intervals)
                else:
                    values = list(range(int(col.cardinality)))
                    axes.append([(int(attr), OP_EQ, val, val, val) for val in values])
                    expected_cells *= len(values)
                if expected_cells > max_cells_per_group:
                    break
            if expected_cells <= 0 or expected_cells > max_cells_per_group:
                continue
            if len(builder.names) + expected_cells > max_queries:
                continue

            all_terms: list[list[tuple[int, int, int, int, int]]] = []
            has_duplicate = False
            for terms_tuple in itertools.product(*axes):
                terms = sorted([tuple(int(part) for part in term) for term in terms_tuple], key=lambda term: term[0])
                key = tuple(sorted(terms))
                if key in seen:
                    has_duplicate = True
                    break
                all_terms.append(terms)
            if has_duplicate or len(all_terms) != expected_cells:
                continue

            start = len(builder.names)
            group_name = f"orthogonal_kway_mixed:{order}:{_scope_label(schema, attrs)}"
            for terms in all_terms:
                added = builder.add(
                    terms,
                    name="&".join(_term_label(schema, term) for term in terms),
                    group=group_name,
                    family="orthogonal_kway_mixed",
                )
                if not added:
                    raise RuntimeError("Unexpected duplicate while adding orthogonal k-way mixed workload")
            groups.append(
                WorkloadGroup(
                    name=group_name,
                    family="orthogonal_kway_mixed",
                    query_indices=np.arange(start, len(builder.names), dtype=np.int32),
                    sensitivity_l2=1.0,
                    is_partition=True,
                )
            )
            used_scopes += 1
            if used_scopes >= scopes_per_order or _cap_queries(builder, max_queries):
                break
        if _cap_queries(builder, max_queries):
            break


def _group_query_scope(qcat: QueryCatalogue, query_indices: np.ndarray) -> tuple[int, ...]:
    attrs: set[int] = set()
    for qid_raw in query_indices.tolist():
        qid = int(qid_raw)
        for term_idx in range(int(qcat.num_terms[qid])):
            attr = int(qcat.attrs[qid, term_idx])
            if attr >= 0:
                attrs.add(attr)
        for term_idx in range(int(qcat.linear_num_terms[qid])):
            attr = int(qcat.linear_attrs[qid, term_idx])
            if attr >= 0:
                attrs.add(attr)
    return tuple(sorted(attrs))


def _exact_group_l2_sensitivity(
    qcat: QueryCatalogue,
    schema: TableSchema,
    query_indices: np.ndarray,
    max_cells: int,
) -> float | None:
    idx = np.asarray(query_indices, dtype=np.int32)
    if idx.size == 0:
        return 0.0
    scope = _group_query_scope(qcat, idx)
    if not scope:
        return 0.0
    num_cells = 1
    for attr in scope:
        num_cells *= int(schema.columns[attr].cardinality)
        if num_cells > max_cells:
            return None

    axes = [np.arange(int(schema.columns[attr].cardinality), dtype=np.int32) for attr in scope]
    mesh = np.meshgrid(*axes, indexing="ij")
    X_cells = np.zeros((num_cells, schema.d), dtype=np.int32)
    for attr, values in zip(scope, mesh, strict=True):
        X_cells[:, attr] = values.reshape(-1)

    hits = np.zeros(num_cells, dtype=np.int32)
    for qid in idx.tolist():
        hits += qcat.eval_query_np(X_cells, int(qid)).astype(np.int32)
    return math.sqrt(float(int(hits.max()) if hits.size else 0))


def _refine_group_sensitivities(
    qcat: QueryCatalogue,
    schema: TableSchema,
    groups: list[WorkloadGroup],
    max_cells: int,
) -> list[WorkloadGroup]:
    refined: list[WorkloadGroup] = []
    for group in groups:
        sensitivity = _exact_group_l2_sensitivity(qcat, schema, group.query_indices, max_cells)
        refined.append(
            WorkloadGroup(
                name=group.name,
                family=group.family,
                query_indices=group.query_indices,
                sensitivity_l2=float(group.sensitivity_l2 if sensitivity is None else sensitivity),
                is_partition=group.is_partition,
            )
        )
    return refined


def build_workload(schema: TableSchema, config: dict[str, Any]) -> tuple[QueryCatalogue, list[WorkloadGroup]]:
    cfg = config.get("workload", {})
    max_terms = int(cfg.get("max_terms", 4))
    max_queries = int(cfg.get("max_queries", 10000))
    max_2way_cells = int(cfg.get("max_2way_cells", 5000))
    range_per_attr = int(cfg.get("range_intervals_per_num_attr", 64))
    mixed_per_pair = int(cfg.get("mixed_queries_per_pair", 64))
    kway_orders = _as_int_list(cfg.get("kway_orders"), [3])
    kway_prefix_orders = _as_int_list(cfg.get("kway_prefix_orders"), kway_orders)
    kway_range_orders = _as_int_list(cfg.get("kway_range_orders"), kway_orders)
    kway_mixed_orders = _as_int_list(cfg.get("kway_mixed_orders"), kway_orders)
    orthogonal_kway_mixed_orders = _as_int_list(cfg.get("orthogonal_kway_mixed_orders"), [2])
    kway_queries_per_order = int(cfg.get("kway_queries_per_order", mixed_per_pair))
    kway_prefix_queries_per_order = int(cfg.get("kway_prefix_queries_per_order", mixed_per_pair))
    kway_range_queries_per_order = int(cfg.get("kway_range_queries_per_order", mixed_per_pair))
    kway_mixed_queries_per_order = int(cfg.get("kway_mixed_queries_per_order", mixed_per_pair))
    orthogonal_kway_mixed_scopes_per_order = int(cfg.get("orthogonal_kway_mixed_scopes_per_order", 4))
    orthogonal_kway_mixed_range_bins = int(cfg.get("orthogonal_kway_mixed_range_bins", 4))
    orthogonal_kway_mixed_max_cells_per_group = int(cfg.get("orthogonal_kway_mixed_max_cells_per_group", 4096))
    halfspace_queries = int(cfg.get("halfspace_queries", cfg.get("mixed_queries_per_pair", 64)))
    exact_group_sensitivity_max_cells = int(cfg.get("exact_group_sensitivity_max_cells", 200_000))
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
            if len(builder.names) + int(col.cardinality) > max_queries:
                continue
            start = len(builder.names)
            for val in range(col.cardinality):
                builder.add(
                    [(attr, OP_EQ, val, val, val)],
                    name=f"{col.name}={val}",
                    group=f"oneway:{attr}",
                    family="oneway",
                )
            mark_group(f"oneway:{attr}", "oneway", start, True, 1.0)

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
            if len(builder.names) + cells > max_queries:
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
            mark_group(f"twoway:{a}:{b}", "twoway", start, True, 1.0)
            used_cells += cells

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

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_kway", False)):
        _sample_kway_eq_queries(
            builder=builder,
            groups=groups,
            schema=schema,
            rng=rng,
            orders=kway_orders,
            queries_per_order=max(0, kway_queries_per_order),
            max_queries=max_queries,
        )

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_kway_prefix", False)):
        _sample_kway_numeric_queries(
            builder=builder,
            groups=groups,
            schema=schema,
            rng=rng,
            family="kway_prefix",
            orders=kway_prefix_orders,
            queries_per_order=max(0, kway_prefix_queries_per_order),
            max_queries=max_queries,
        )

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_kway_range", False)):
        _sample_kway_numeric_queries(
            builder=builder,
            groups=groups,
            schema=schema,
            rng=rng,
            family="kway_range",
            orders=kway_range_orders,
            queries_per_order=max(0, kway_range_queries_per_order),
            max_queries=max_queries,
        )

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_kway_mixed", False)):
        _sample_kway_mixed_queries(
            builder=builder,
            groups=groups,
            schema=schema,
            rng=rng,
            orders=kway_mixed_orders,
            queries_per_order=max(0, kway_mixed_queries_per_order),
            max_queries=max_queries,
        )

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_orthogonal_kway_mixed", False)):
        _sample_orthogonal_kway_mixed_queries(
            builder=builder,
            groups=groups,
            schema=schema,
            rng=rng,
            orders=orthogonal_kway_mixed_orders,
            scopes_per_order=max(0, orthogonal_kway_mixed_scopes_per_order),
            range_bins=max(1, orthogonal_kway_mixed_range_bins),
            max_cells_per_group=max(1, orthogonal_kway_mixed_max_cells_per_group),
            max_queries=max_queries,
        )

    if not _cap_queries(builder, max_queries) and bool(cfg.get("include_halfspace", False)) and numerical:
        start = len(builder.names)
        max_halfspace_terms = max(1, min(max_terms, len(numerical)))
        for query_id in range(halfspace_queries):
            num_terms = int(rng.integers(1, max_halfspace_terms + 1))
            attrs = [int(x) for x in rng.choice(numerical, size=num_terms, replace=False).tolist()]
            weights = rng.normal(0.0, 1.0, size=num_terms)
            weights = np.where(np.abs(weights) < 0.1, np.sign(weights + 1.0e-12) * 0.1, weights)
            lows = np.asarray([0 for _ in attrs], dtype=np.float64)
            highs = np.asarray([schema.columns[attr].cardinality - 1 for attr in attrs], dtype=np.float64)
            min_score = float(np.sum(np.where(weights >= 0.0, weights * lows, weights * highs)))
            max_score = float(np.sum(np.where(weights >= 0.0, weights * highs, weights * lows)))
            if max_score <= min_score:
                threshold = min_score
            else:
                threshold = float(rng.uniform(min_score, max_score))
            terms = [(attr, float(weight)) for attr, weight in zip(attrs, weights.tolist(), strict=True)]
            builder.add_halfspace(
                terms,
                threshold=threshold,
                name=" + ".join(f"{weight:.3g}*{schema.columns[attr].name}" for attr, weight in terms)
                + f" <= {threshold:.3g}",
                group="halfspace",
                family="halfspace",
            )
            if _cap_queries(builder, max_queries):
                break
        size = max(1, len(builder.names) - start)
        mark_group("halfspace", "halfspace", start, False, math.sqrt(size))

    qcat = builder.build()
    groups = [g for g in groups if len(g.query_indices) > 0 and int(g.query_indices.max()) < qcat.m]
    groups = _refine_group_sensitivities(qcat, schema, groups, max(1, exact_group_sensitivity_max_cells))
    return qcat, groups
