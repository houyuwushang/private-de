#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jax import jit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

try:
    from path_defaults import rappp_root
except ModuleNotFoundError:
    from scripts.path_defaults import rappp_root

DEFAULT_RAPPP_ROOT = rappp_root()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _as_float_dict(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            out[key] = _as_float_dict(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _compute_stat_error(dataset_container: Any, query_class: Any, synthetic_csv: Path, *, stat_seed: int) -> dict[str, Any]:
    stat = query_class.get_statistics(domain=dataset_container.train.domain, seed=stat_seed)
    true_stats_fn = jit(stat.get_exact_statistics_fn(stat.get_num_queries()))
    all_idx = np.arange(stat.get_num_queries())

    real = dataset_container.train.get_dataset()
    synthetic_df = pd.read_csv(synthetic_csv)
    synthetic = dataset_container.from_df_to_dataset(synthetic_df).get_dataset()

    real_stats = np.asarray(true_stats_fn(all_idx, real).block_until_ready())
    synthetic_stats = np.asarray(true_stats_fn(all_idx, synthetic).block_until_ready())
    errors = real_stats - synthetic_stats
    l1_total = float(np.linalg.norm(errors.reshape(-1), ord=1))
    return {
        "num_queries": int(stat.get_num_queries()),
        "stat_shape": list(real_stats.shape),
        "max": float(np.max(np.abs(errors))) if errors.size else 0.0,
        "l1_total": l1_total,
        "ave": l1_total / float(stat.get_num_queries()) if stat.get_num_queries() else 0.0,
    }


def _binarize_columns(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    cat_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    stored_binarizers = []
    for col in cat_columns:
        lb = LabelBinarizer()
        stored_binarizers.append(lb.fit(combined_data[col].astype(int)))

    def replace_with_binarized(dataframe: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
        new_df = dataframe[feature_columns].copy()
        for idx, column_name in enumerate(column_names):
            if column_name not in new_df.columns:
                continue
            lb = stored_binarizers[idx]
            lb_results = lb.transform(new_df[column_name])
            if len(lb.classes_) <= 1:
                continue
            columns = lb.classes_ if len(lb.classes_) > 2 else [f"is {lb.classes_[1]}"]
            binarized_cols = pd.DataFrame(lb_results, columns=columns, index=new_df.index)
            new_df.drop(columns=column_name, inplace=True)
            new_df = pd.concat([new_df, binarized_cols], axis=1)
        return new_df

    x_cat_cols = [cat for cat in cat_columns if cat in feature_columns]
    x_train = replace_with_binarized(train_data, x_cat_cols)
    x_test = replace_with_binarized(test_data, x_cat_cols)
    return np.asarray(x_train), np.asarray(x_test)


def _evaluate_lr(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    cat_columns: list[str],
) -> dict[str, Any]:
    train_data = train_data.copy()
    test_data = test_data.copy()
    for col in cat_columns:
        if col in train_data.columns:
            train_data[col] = train_data[col].fillna(0)
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(0)

    x_train, x_test = _binarize_columns(train_data, test_data, feature_columns, cat_columns)
    y_train = np.asarray(train_data[[label_column]]).ravel()
    y_test = np.asarray(test_data[[label_column]]).ravel()
    model = LogisticRegression(penalty="l1", solver="liblinear")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return _as_float_dict(classification_report(y_test, y_pred, output_dict=True))


def _compute_lr_metrics(dataset_container: Any, synthetic_csv: Path) -> dict[str, Any]:
    train_df = dataset_container.from_dataset_to_df_fn(dataset_container.train)
    test_df = dataset_container.from_dataset_to_df_fn(dataset_container.test)
    synthetic_df = pd.read_csv(synthetic_csv)

    cat_cols = list(dataset_container.cat_columns)
    num_cols = list(dataset_container.num_columns)
    labels = list(dataset_container.label_column)
    results: dict[str, Any] = {}
    for label in labels:
        feature_columns = list(set(cat_cols + num_cols) - {label})
        original = _evaluate_lr(
            train_df,
            test_df,
            feature_columns=feature_columns,
            label_column=label,
            cat_columns=cat_cols,
        )
        synthetic = _evaluate_lr(
            synthetic_df,
            test_df,
            feature_columns=feature_columns,
            label_column=label,
            cat_columns=cat_cols,
        )
        results[label] = {
            "original_accuracy": float(original["accuracy"]),
            "original_macro_f1": float(original["macro avg"]["f1-score"]),
            "original_weighted_f1": float(original["weighted avg"]["f1-score"]),
            "synthetic_accuracy": float(synthetic["accuracy"]),
            "synthetic_macro_f1": float(synthetic["macro avg"]["f1-score"]),
            "synthetic_weighted_f1": float(synthetic["weighted avg"]["f1-score"]),
            "original_report": original,
            "synthetic_report": synthetic,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one RAP++ output with the original paper benchmark metrics.")
    parser.add_argument("--rappp-root", type=Path, default=DEFAULT_RAPPP_ROOT)
    parser.add_argument("--state", default="CA")
    parser.add_argument("--target", default="income")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-csv", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix-queries", type=int, default=20000)
    parser.add_argument("--stat-seed", type=int, default=123)
    parser.add_argument("--skip-ml", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rappp_root = args.rappp_root.resolve()
    sys.path.insert(0, str(rappp_root))

    from dataloading.data_functions.acs import get_acs
    from modules.marginal_queries import MarginalQueryClass
    from modules.random_prefix import RandomPrefixQueryClass

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    start = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_fn = get_acs(state=args.state, target=args.target)
    dataset_container = dataset_fn(args.seed)

    metrics: dict[str, Any] = {
        "state": args.state,
        "target": args.target,
        "seed": int(args.seed),
        "synthetic_csv": str(args.synthetic_csv.resolve()),
        "rappp_root": str(rappp_root),
        "run_metadata": _load_run_metadata(args.run_dir.resolve()) if args.run_dir else {},
        "query_error": {},
        "ml": {},
        "runtime_seconds": None,
    }

    metrics["query_error"]["marginal"] = _compute_stat_error(
        dataset_container,
        MarginalQueryClass(K=2, max_number_rows=2000),
        args.synthetic_csv,
        stat_seed=args.stat_seed,
    )
    metrics["query_error"]["prefix"] = _compute_stat_error(
        dataset_container,
        RandomPrefixQueryClass(
            num_random_projections=int(args.prefix_queries),
            k=2,
            max_number_rows=2000,
            max_number_queries=int(args.prefix_queries),
        ),
        args.synthetic_csv,
        stat_seed=args.stat_seed,
    )

    if not args.skip_ml:
        metrics["ml"]["LR"] = _compute_lr_metrics(dataset_container, args.synthetic_csv)

    metrics["runtime_seconds"] = float(time.time() - start)
    _write_json(output_dir / "rappp_paper_metrics.json", metrics)

    flat_rows = []
    for name, row in metrics["query_error"].items():
        flat_rows.append({"metric_group": "query_error", "name": name, **row})
    for model_name, model_results in metrics["ml"].items():
        for label, row in model_results.items():
            flat_rows.append({"metric_group": "ml", "name": f"{model_name}:{label}", **row})
    if flat_rows:
        pd.DataFrame(flat_rows).to_csv(output_dir / "rappp_paper_metrics.csv", index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
