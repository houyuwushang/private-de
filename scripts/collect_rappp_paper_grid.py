#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from path_defaults import external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, external_runs

DEFAULT_RUNS_ROOT = external_runs() / "rappp_official_paper_grid"
DEFAULT_RESULTS_ROOT = external_results()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _row_from_metrics(path: Path) -> dict[str, Any]:
    metrics = _read_json(path)
    final_metadata_path = path.parents[1] / "run_metadata.json"
    run_metadata = _read_json(final_metadata_path) if final_metadata_path.exists() else (metrics.get("run_metadata") or {})
    lr = next(iter((metrics.get("ml") or {}).get("LR", {}).values()), {})
    marginal = (metrics.get("query_error") or {}).get("marginal", {})
    prefix = (metrics.get("query_error") or {}).get("prefix", {})
    jax_probe = run_metadata.get("jax_device_probe") or {}
    return {
        "state": metrics.get("state") or run_metadata.get("state"),
        "target": metrics.get("target") or run_metadata.get("target"),
        "seed": metrics.get("seed") or run_metadata.get("seed"),
        "dataset_name": run_metadata.get("dataset_name"),
        "upstream_epsilon": run_metadata.get("upstream_epsilon"),
        "num_random_projections": run_metadata.get("num_random_projections"),
        "top_q": run_metadata.get("top_q"),
        "dp_select_epochs": run_metadata.get("dp_select_epochs"),
        "run_status": run_metadata.get("status"),
        "cuda_visible_devices": run_metadata.get("cuda_visible_devices"),
        "xla_python_client_preallocate": run_metadata.get("xla_python_client_preallocate"),
        "rappp_full_stats_chunk_size": run_metadata.get("rappp_full_stats_chunk_size"),
        "jax_version": jax_probe.get("jax_version"),
        "jax_local_device_count": jax_probe.get("jax_local_device_count"),
        "jax_device_platforms": ",".join(jax_probe.get("jax_device_platforms") or []),
        "jax_devices": ",".join(jax_probe.get("jax_devices") or []),
        "task_runtime_seconds": run_metadata.get("runtime_seconds"),
        "paper_metric_runtime_seconds": metrics.get("runtime_seconds"),
        "marginal_num_queries": marginal.get("num_queries"),
        "marginal_max_error": marginal.get("max"),
        "marginal_average_error": marginal.get("ave"),
        "prefix_num_queries": prefix.get("num_queries"),
        "prefix_max_error": prefix.get("max"),
        "prefix_average_error": prefix.get("ave"),
        "original_accuracy": lr.get("original_accuracy"),
        "original_macro_f1": lr.get("original_macro_f1"),
        "original_weighted_f1": lr.get("original_weighted_f1"),
        "synthetic_accuracy": lr.get("synthetic_accuracy"),
        "synthetic_macro_f1": lr.get("synthetic_macro_f1"),
        "synthetic_weighted_f1": lr.get("synthetic_weighted_f1"),
        "macro_f1_gap": (
            lr.get("original_macro_f1") - lr.get("synthetic_macro_f1")
            if lr.get("original_macro_f1") is not None and lr.get("synthetic_macro_f1") is not None
            else None
        ),
        "metrics_path": str(path),
        "run_dir": str(path.parents[1]),
    }


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "marginal_max_error",
        "marginal_average_error",
        "prefix_max_error",
        "prefix_average_error",
        "original_macro_f1",
        "synthetic_macro_f1",
        "macro_f1_gap",
        "synthetic_accuracy",
        "synthetic_weighted_f1",
        "task_runtime_seconds",
    ]
    grouped = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "sem", "min", "max"]).reset_index()
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join(str(part) for part in col if part))
        else:
            flat_cols.append(str(col))
    grouped.columns = flat_cols
    return grouped


def _write_markdown(raw: pd.DataFrame, target_summary: pd.DataFrame, overall: pd.DataFrame, output_md: Path) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# RAP++ Official Paper Grid Summary",
        "",
        f"Rows: {len(raw)}",
        "",
        "## Overall",
        "",
    ]
    overall_cols = [
        "marginal_average_error_mean",
        "prefix_average_error_mean",
        "synthetic_macro_f1_mean",
        "macro_f1_gap_mean",
        "task_runtime_seconds_mean",
    ]
    if not overall.empty:
        row = overall.iloc[0]
        lines.extend(
            [
                "| Metric | Value |",
                "|---|---:|",
                *[f"| {col} | {fmt(row[col])} |" for col in overall_cols if col in row],
                "",
            ]
        )

    table_cols = [
        "target",
        "marginal_average_error_mean",
        "prefix_average_error_mean",
        "synthetic_macro_f1_mean",
        "macro_f1_gap_mean",
        "task_runtime_seconds_mean",
    ]
    lines.extend(["## By Target", ""])
    lines.append("| " + " | ".join(table_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in table_cols) + " |")
    for _, row in target_summary.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in table_cols) + " |")
    lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RAP++ official paper-grid metrics.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_RESULTS_ROOT / "rappp_official_paper_grid_seed0_20260706")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_paths = sorted(args.runs_root.rglob("paper_metrics/rappp_paper_metrics.json"))
    rows = [_row_from_metrics(path) for path in metric_paths]
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError(f"No RAP++ paper metric files found under {args.runs_root}")
    raw = raw.sort_values(["state", "target"]).reset_index(drop=True)
    target_summary = _summarize(raw, ["target"]).sort_values("target").reset_index(drop=True)
    state_summary = _summarize(raw, ["state"]).sort_values("state").reset_index(drop=True)
    overall = _summarize(raw.assign(all_tasks="all"), ["all_tasks"])

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_prefix.with_suffix(".csv"), index=False)
    target_summary.to_csv(output_prefix.with_name(output_prefix.name + "_by_target.csv"), index=False)
    state_summary.to_csv(output_prefix.with_name(output_prefix.name + "_by_state.csv"), index=False)
    overall.to_csv(output_prefix.with_name(output_prefix.name + "_overall.csv"), index=False)
    _write_markdown(raw, target_summary, overall, output_prefix.with_suffix(".md"))
    print(f"wrote {len(raw)} rows to {output_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
