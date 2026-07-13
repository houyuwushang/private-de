#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import apply_overrides, load_yaml, set_nested
from qdte.dataio import ensure_dir, write_json
from qdte.measurement.measure import MeasurementGroup, Measurements
from qdte.preprocess import load_and_preprocess_csv
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import filter_query_catalogue
from qdte.queries.workload import WorkloadGroup, build_workload


def _prepare_gsd_workload_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(config)
    workload = dict(prepared.get("workload", {}) or {})
    workload.update(
        {
            "include_oneway": False,
            "include_2way_cat": True,
            "include_prefix": False,
            "include_range": False,
            "include_mixed": False,
            "include_kway": False,
            "include_kway_prefix": False,
            "include_kway_range": False,
            "include_kway_mixed": False,
            "include_orthogonal_kway_mixed": False,
            "include_halfspace": False,
            "max_queries": int(workload.get("gsd_max_queries", 10_000_000)),
            "max_2way_cells": int(workload.get("gsd_max_2way_cells", 10_000_000)),
            "max_terms": 2,
        }
    )
    prepared["workload"] = workload
    return prepared


def _resolve_n_prime(value: str, n_real: int) -> int:
    if value in {"default", "same_as_real", ""}:
        return int(n_real)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--n-prime must be positive, same_as_real, or default")
    return parsed


def _selected_indices_by_noisy_topk(
    groups: list[WorkloadGroup],
    noisy_rates: np.ndarray,
    n_prime: int,
) -> np.ndarray:
    selected: list[int] = []
    for group in groups:
        idx = np.asarray(group.query_indices, dtype=np.int32)
        if idx.size == 0:
            continue
        top_k = min(int(idx.size), int(n_prime))
        order = np.argsort(-noisy_rates[idx], kind="mergesort")[:top_k]
        selected.extend(idx[order].astype(int).tolist())
    return np.asarray(selected, dtype=np.int32)


def _remap_groups(groups: list[WorkloadGroup], keep: np.ndarray) -> list[WorkloadGroup]:
    index_map = {int(old): new for new, old in enumerate(np.asarray(keep, dtype=np.int32).tolist())}
    remapped: list[WorkloadGroup] = []
    for group in groups:
        indices = [index_map[int(idx)] for idx in group.query_indices if int(idx) in index_map]
        if not indices:
            continue
        remapped.append(
            WorkloadGroup(
                name=group.name,
                family="gsd_twoway",
                query_indices=np.asarray(indices, dtype=np.int32),
                sensitivity_l2=math.sqrt(2.0),
                is_partition=bool(group.is_partition and len(indices) == len(group.query_indices)),
            )
        )
    return remapped


def _measurement_groups(groups: list[WorkloadGroup], rho_per_group: float, sigma_count: float) -> list[MeasurementGroup]:
    return [
        MeasurementGroup(
            query_indices=np.asarray(group.query_indices, dtype=np.int32),
            sensitivity_l2=math.sqrt(2.0),
            rho=float(rho_per_group),
            sigma=float(sigma_count),
            noise_std=float(sigma_count),
            name=group.name,
            family="gsd_twoway",
            is_partition=group.is_partition,
        )
        for group in groups
    ]


def run(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    prepared = _prepare_gsd_workload_config(config)
    preprocess = load_and_preprocess_csv(prepared)
    X_real = preprocess.X
    schema = preprocess.schema
    n_real = int(X_real.shape[0])
    qcat, groups = build_workload(schema, prepared)
    if not groups:
        raise ValueError("GSD measurement materialization produced no 2-way groups")

    noise_mode = str(args.noise_mode)
    rho_total = float(args.rho_total)
    delta = float(args.delta)
    if noise_mode == "gsd" and (not np.isfinite(rho_total) or rho_total <= 0.0):
        raise ValueError("--rho-total must be positive and finite in gsd noise mode")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("--delta must be in (0, 1)")
    n_prime = _resolve_n_prime(str(args.n_prime), n_real)
    rho_per_group = rho_total / float(len(groups)) if noise_mode == "gsd" else 0.0
    sigma_count = 1.0 / math.sqrt(rho_per_group) if noise_mode == "gsd" else 0.0
    rng = np.random.default_rng(int(args.seed))

    true_counts = answer_queries(X_real, qcat, batch_size=int(args.answer_batch_size)).astype(np.float64)
    true_rates = true_counts / float(n_real)
    if noise_mode == "gsd":
        noise_rates = rng.normal(0.0, sigma_count / float(n_real), size=qcat.m)
        noisy_rates = np.clip(true_rates + noise_rates, 0.0, 1.0)
        mode = "dp"
        rho_spent = rho_total
        epsilon_delta = zcdp_epsilon(rho_total, delta)
        variance_value = sigma_count * sigma_count
        projection_method = "private_gsd_2way_rate_clip"
        source_stat_scale = "GSD rates clipped to [0,1], multiplied by n_real"
    elif noise_mode == "no_noise":
        noisy_rates = true_rates.copy()
        mode = "non_dp_diagnostic"
        rho_spent = 0.0
        epsilon_delta = 0.0
        variance_value = 1.0
        projection_method = "private_gsd_2way_exact_truth_no_noise"
        source_stat_scale = "Exact true rates with no noise, multiplied by n_real"
    else:
        raise ValueError("--noise-mode must be one of: gsd, no_noise")
    noisy_counts = noisy_rates * float(n_real)
    keep = _selected_indices_by_noisy_topk(groups, noisy_rates, n_prime)
    qcat_selected = filter_query_catalogue(qcat, keep)
    qcat_selected.validate(schema.cardinalities)
    groups_selected = _remap_groups(groups, keep)
    target_noisy = noisy_counts[keep].astype(np.float32)
    variances = np.full(qcat_selected.m, variance_value, dtype=np.float32)
    measurements = Measurements(
        target_noisy=target_noisy,
        target_projected=target_noisy.copy(),
        variances=variances,
        inv_variances=(1.0 / np.maximum(variances, 1.0e-12)).astype(np.float32),
        groups=_measurement_groups(groups_selected, rho_per_group, sigma_count),
        mode=mode,
        rho_total=rho_total,
        rho_spent=rho_spent,
        epsilon_delta=epsilon_delta,
        delta=delta,
        num_rows=int(n_real),
        projection_diagnostics={
            "method": projection_method,
            "noise_mode": noise_mode,
            "rate_clipped_to_0_1": noise_mode == "gsd",
            "sparse_statistics_top_n_per_workload": int(n_prime),
            "num_original_queries": int(qcat.m),
            "num_selected_queries": int(qcat_selected.m),
            "num_workload_groups": int(len(groups_selected)),
            "sigma_count": float(sigma_count),
            "rho_per_group": float(rho_per_group),
        },
    )

    schema.save_json(output_dir / "schema.json")
    qcat_selected.save_json(output_dir / "queries.json")
    write_json(measurements.to_public_dict(), output_dir / "measurements.json")
    write_json([group.to_dict() for group in groups_selected], output_dir / "workload_groups.json")
    summary = {
        "method": "private_gsd_2way_measurement_materialization",
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "rho_total": rho_total,
        "delta": delta,
        "n_real": n_real,
        "n_prime": int(n_prime),
        "num_original_queries": int(qcat.m),
        "num_selected_queries": int(qcat_selected.m),
        "num_workload_groups": int(len(groups_selected)),
        "rho_per_group": float(rho_per_group),
        "sigma_count": float(sigma_count),
        "noise_mode": noise_mode,
        "measurement_mode": mode,
        "variance_value": float(variance_value),
        "target_scale": "counts",
        "source_stat_scale": source_stat_scale,
    }
    write_json(summary, output_dir / "metadata.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a Private-GSD-style 2-way measurement artifact for QDTE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rho-total", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prime", default="same_as_real")
    parser.add_argument("--noise-mode", choices=["gsd", "no_noise"], default="gsd")
    parser.add_argument("--answer-batch-size", type=int, default=8192)
    args, overrides = parser.parse_known_args()
    config = apply_overrides(load_yaml(args.config), overrides)
    set_nested(config, "privacy.mode", "dp")
    set_nested(config, "privacy.rho_total", float(args.rho_total))
    set_nested(config, "privacy.delta", float(args.delta))
    summary = run(args, config)
    print(summary)


if __name__ == "__main__":
    main()
