#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.evolution.engine import run_qdte


def make_smoke_csv(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 1000
    a = rng.integers(0, 4, size=n)
    b = rng.integers(0, 5, size=n)
    c = np.clip(a + rng.integers(0, 4, size=n), 0, 7)
    d = np.clip((b * 2 + rng.integers(0, 3, size=n)), 0, 7)
    y = ((a == 2) | (d > 4)).astype(int)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "label": y})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dp", "oracle"], default="dp")
    args = parser.parse_args()
    data_path = ROOT / "outputs" / "smoke_input.csv"
    make_smoke_csv(data_path)
    config = {
        "run": {
            "dataset_name": "smoke",
            "input_csv": str(data_path),
            "output_dir": str(ROOT / "outputs" / f"smoke_qdte_{args.mode}"),
            "seed": 0,
        },
        "preprocess": {
            "numerical_bins": 8,
            "missing_token": "__MISSING__",
            "label_column": "label",
            "numerical_columns": ["c", "d"],
            "categorical_columns": ["a", "b", "label"],
        },
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": True,
            "include_range": True,
            "include_mixed": True,
            "max_queries": 400,
            "max_terms": 4,
            "range_intervals_per_num_attr": 8,
            "mixed_queries_per_pair": 8,
            "max_2way_cells": 150,
            "random_seed": 0,
        },
        "privacy": {
            "mode": args.mode,
            "rho_total": 1.0,
            "delta": 1.0e-9,
            "measurement_mode": "static_all",
            "measurement_allocation": {
                "oneway": 0.25,
                "twoway": 0.25,
                "prefix": 0.15,
                "range": 0.15,
                "mixed": 0.20,
            },
        },
        "projection": {
            "project_partitions": True,
            "clip_nonpartition": True,
            "prefix_monotonicity": False,
        },
        "init": {"N_syn": "same_as_real", "method": "independent_oneway"},
        "qdte": {
            "score_backend": "dense_gpu",
            "transport_mode": "sequential_greedy",
            "max_iters": 100,
            "num_active_targets": 16,
            "candidates_per_target": 16,
            "total_candidates_per_iter": 256,
            "accepted_per_iter": 8,
            "kappa_noise": 0.5,
            "lambda_cost": 0.0,
            "numerical_distance_gamma": 0.1,
            "random_candidate_fraction": 0.05,
            "full_recompute_every": 20,
            "stop_patience": 15,
            "min_advantage": 1.0e-6,
            "log_every": 10,
        },
        "runtime": {
            "use_pmap": False,
            "scoring_chunk_size": 256,
            "answer_batch_size": 512,
            "xla_preallocate": False,
        },
        "evaluation": {
            "compute_true_query_error": True,
            "downstream_ml": False,
            "save_synthetic_csv": True,
        },
    }
    run_qdte(config)


if __name__ == "__main__":
    main()
