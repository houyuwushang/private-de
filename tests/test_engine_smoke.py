from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import orjson

from qdte.evolution.engine import run_qdte
from qdte.queries.types import QueryCatalogue, query_key


def test_engine_smoke_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "a": rng.integers(0, 3, size=n),
            "b": rng.integers(0, 4, size=n),
            "c": rng.integers(0, 6, size=n),
            "label": rng.integers(0, 2, size=n),
        }
    )
    data_path = tmp_path / "smoke.csv"
    out_dir = tmp_path / "out"
    df.to_csv(data_path, index=False)
    config = {
        "run": {"dataset_name": "smoke", "input_csv": str(data_path), "output_dir": str(out_dir), "seed": 0},
        "preprocess": {
            "numerical_bins": 6,
            "label_column": "label",
            "numerical_columns": ["c"],
            "categorical_columns": ["a", "b", "label"],
        },
        "workload": {
            "include_oneway": True,
            "include_2way_cat": True,
            "include_prefix": True,
            "include_range": True,
            "include_mixed": True,
            "max_queries": 120,
            "max_terms": 4,
            "range_intervals_per_num_attr": 4,
            "mixed_queries_per_pair": 4,
            "max_2way_cells": 50,
            "random_seed": 0,
        },
        "privacy": {
            "mode": "dp",
            "rho_total": 1.0,
            "delta": 1e-9,
            "measurement_allocation": {"oneway": 0.3, "twoway": 0.3, "prefix": 0.15, "range": 0.1, "mixed": 0.15},
        },
        "projection": {"project_partitions": True, "clip_nonpartition": True},
        "init": {"N_syn": "same_as_real"},
        "qdte": {
            "max_iters": 3,
            "num_active_targets": 4,
            "candidates_per_target": 4,
            "total_candidates_per_iter": 32,
            "accepted_per_iter": 2,
            "kappa_noise": 0.0,
            "lambda_cost": 0.0,
            "random_candidate_fraction": 0.1,
            "full_recompute_every": 2,
            "stop_patience": 3,
            "min_advantage": 1e-6,
            "log_every": 1,
        },
        "debug": {"recompute_after_batch": True, "assert_batch_loss_decrease": True},
        "runtime": {"use_pmap": False, "scoring_chunk_size": 32, "answer_batch_size": 64, "xla_preallocate": False},
        "evaluation": {
            "compute_true_query_error": True,
            "compute_heldout_query_error": True,
            "heldout_exclude_measured_queries": True,
            "save_synthetic_csv": True,
            "heldout_workload": {
                "include_oneway": True,
                "include_2way_cat": True,
                "include_prefix": True,
                "include_range": True,
                "include_mixed": True,
                "include_halfspace": False,
                "max_queries": 180,
                "max_terms": 4,
                "max_2way_cells": 80,
                "range_intervals_per_num_attr": 8,
                "mixed_queries_per_pair": 8,
                "random_seed": 10000,
            },
        },
    }
    metrics = run_qdte(config)
    assert metrics["num_queries"] > 0
    assert (out_dir / "synthetic_decoded.csv").exists()
    assert (out_dir / "metrics_final.json").exists()
    assert (out_dir / "metrics_by_family.json").exists()
    assert (out_dir / "workload_summary.json").exists()
    assert (out_dir / "queries_holdout.json").exists()
    assert (out_dir / "workload_summary_holdout.json").exists()
    assert (out_dir / "metrics_holdout.json").exists()
    assert (out_dir / "metrics_by_family_holdout.json").exists()
    assert (out_dir / "metrics_timeseries.csv").exists()
    assert (out_dir / "runtime.json").exists()
    metrics_json = orjson.loads((out_dir / "metrics_final.json").read_bytes())
    assert "final_rms_standardized_residual" in metrics_json
    assert "initial_true_query_mae" in metrics_json
    assert "final_true_query_mae" in metrics_json
    assert "positive_returned_rate_is_topk_biased" in metrics_json
    assert "heldout_final_true_query_mae" in metrics_json
    assert metrics_json["true_query_mae"] == metrics_json["final_true_query_mae"]
    assert metrics["final_incremental_answer_drift"] == 0.0
    by_family_json = orjson.loads((out_dir / "metrics_by_family.json").read_bytes())
    assert "oneway" in by_family_json
    assert by_family_json["oneway"]["num_queries"] > 0
    assert "true_query_mae_reduction" in by_family_json["oneway"]
    assert "true_query_rmse_reduction" in by_family_json["oneway"]
    workload_json = orjson.loads((out_dir / "workload_summary.json").read_bytes())
    assert workload_json["total_num_queries"] == metrics["num_queries"]
    assert "num_queries_by_family" in workload_json
    assert workload_json["total_queries"] == workload_json["total_num_queries"]
    assert workload_json["queries_by_family"] == workload_json["num_queries_by_family"]
    holdout_workload_json = orjson.loads((out_dir / "workload_summary_holdout.json").read_bytes())
    assert holdout_workload_json["total_queries"] > 0
    assert holdout_workload_json["num_removed_as_measured_duplicates"] > 0
    assert holdout_workload_json["heldout_exclude_measured_queries"] is True
    holdout_metrics_json = orjson.loads((out_dir / "metrics_holdout.json").read_bytes())
    assert holdout_metrics_json["num_queries"] == holdout_workload_json["total_queries"]
    assert holdout_metrics_json["num_queries"] > 0
    assert "final_true_query_mae" in holdout_metrics_json
    by_family_holdout_json = orjson.loads((out_dir / "metrics_by_family_holdout.json").read_bytes())
    assert by_family_holdout_json
    measured_qcat = QueryCatalogue.from_dict(orjson.loads((out_dir / "queries.json").read_bytes()))
    heldout_qcat = QueryCatalogue.from_dict(orjson.loads((out_dir / "queries_holdout.json").read_bytes()))
    measured_keys = {query_key(measured_qcat, qid) for qid in range(measured_qcat.m)}
    heldout_keys = {query_key(heldout_qcat, qid) for qid in range(heldout_qcat.m)}
    assert measured_keys.isdisjoint(heldout_keys)
    measurement_json = orjson.loads((out_dir / "measurements.json").read_bytes())
    assert "true_answers_debug" not in measurement_json
    assert "true_answers" not in measurement_json
    runtime_json = orjson.loads((out_dir / "runtime.json").read_bytes())
    assert runtime_json["num_candidates_requested"] >= runtime_json["num_candidates_scored"]
    assert "accepted_per_scored_candidate" in runtime_json
    assert "candidate_funnel" in runtime_json
    assert runtime_json["candidate_funnel"]["requested"] == runtime_json["num_candidates_requested"]
