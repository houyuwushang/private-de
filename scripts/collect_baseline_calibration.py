#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

try:
    from path_defaults import external_calibration_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_calibration_runs


METRICS = [
    "full_true_mae",
    "full_true_rmse",
    "full_true_avg_tvd",
    "full_true_max_error",
    "full_true_max_tvd",
    "runtime_seconds",
]


def _read_json(path: Path) -> dict[str, Any]:
    return orjson.loads(path.read_bytes())


def _row_from_evaluation(path: Path) -> dict[str, Any]:
    evaluation = _read_json(path)
    metadata = evaluation.get("run_metadata") or {}
    notes = metadata.get("notes") or {}
    row = {
        "evaluation_path": str(path),
        "run_dir": str(path.parent),
        "setting": path.parent.name,
        "method": metadata.get("method"),
        "dataset": evaluation.get("dataset") or metadata.get("dataset"),
        "rho_total": metadata.get("rho_total"),
        "epsilon_delta": metadata.get("epsilon_delta"),
        "delta": metadata.get("delta"),
        "seed": metadata.get("seed"),
        "status": metadata.get("status", "unknown"),
        "failure_reason": metadata.get("failure_reason"),
        "n_real": metadata.get("n_real"),
        "n_train": metadata.get("n_train"),
        "n_synthetic_native": metadata.get("n_synthetic_native"),
        "n_synthetic": metadata.get("n_synthetic"),
        "n_prime": notes.get("n_prime"),
        "tree_query_depth": notes.get("tree_query_depth"),
        "genetic_operators": ",".join(notes.get("genetic_operators") or [])
        if isinstance(notes.get("genetic_operators"), list)
        else notes.get("genetic_operators"),
        "early_stop_threshold": notes.get("early_stop_threshold"),
        "max_train_rows": notes.get("max_train_rows"),
        "estimation_iters": notes.get("estimation_iters"),
        "entropy_descent_t": notes.get("entropy_descent_t"),
        "init_measure": notes.get("init_measure"),
        "resampled_to_n_syn": notes.get("resampled_to_n_syn"),
    }
    for metric in METRICS:
        row[metric] = evaluation.get(metric)
    return row


def collect_results(runs_root: Path) -> pd.DataFrame:
    rows = [_row_from_evaluation(path) for path in sorted(runs_root.rglob("evaluation.json"))]
    return pd.DataFrame(rows)


def _write_markdown(df: pd.DataFrame, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text("# Baseline Calibration Summary\n\nNo `evaluation.json` files found.\n")
        return
    table_cols = [
        "method",
        "dataset",
        "rho_total",
        "seed",
        "setting",
        "n_prime",
        "max_train_rows",
        "estimation_iters",
        "full_true_mae",
        "full_true_rmse",
        "full_true_avg_tvd",
        "full_true_max_error",
        "full_true_max_tvd",
        "runtime_seconds",
    ]
    table_cols = [col for col in table_cols if col in df.columns]
    lines = [
        "# Baseline Calibration Summary",
        "",
        "Diagnostic calibration only. Do not treat these rows as final paper evidence.",
        "",
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join("---" for _ in table_cols) + " |",
    ]
    view = df.sort_values(["method", "dataset", "rho_total", "seed", "setting"])
    for _, row in view.iterrows():
        values = []
        for col in table_cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect diagnostic baseline calibration evaluations.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=external_calibration_runs(),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_results(args.runs_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    if args.output_md is not None:
        _write_markdown(df, args.output_md)
    print(f"wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
