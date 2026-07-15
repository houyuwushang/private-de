#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

try:
    from path_defaults import external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_runs


METRICS = [
    "full_true_mae",
    "full_true_rmse",
    "full_true_avg_tvd",
    "full_true_max_error",
    "full_true_max_tvd",
    "runtime_seconds",
]

SUMMARY_METRICS = [
    ("MAE", "full_true_mae"),
    ("RMSE", "full_true_rmse"),
    ("AvgTVD", "full_true_avg_tvd"),
    ("MaxErr", "full_true_max_error"),
    ("MaxTVD", "full_true_max_tvd"),
    ("runtime_s", "runtime_seconds"),
]

DISPLAY_NAMES = {
    "private_gsd_gpu_1m_fulln_audit": "Private-GSD GPU 1M/full-N",
    "private_gsd_gpu": "Private-GSD GPU",
    "private_gsd": "Private-GSD",
    "private_pgm_aim": "Private-PGM AIM",
    "private_pgm_mst": "Private-PGM MST",
    "sage": "SAGE",
}


def _read_json(path: Path) -> dict[str, Any]:
    return orjson.loads(path.read_bytes())


def _row_from_evaluation(path: Path) -> dict[str, Any]:
    evaluation = _read_json(path)
    run_metadata = evaluation.get("run_metadata") or {}
    row = {
        "evaluation_path": str(path),
        "run_dir": str(path.parent),
        "method": run_metadata.get("method") or path.parent.parents[2].name,
        "dataset": evaluation.get("dataset") or run_metadata.get("dataset"),
        "rho_total": run_metadata.get("rho_total"),
        "epsilon_delta": run_metadata.get("epsilon_delta"),
        "delta": run_metadata.get("delta"),
        "seed": run_metadata.get("seed"),
        "status": run_metadata.get("status", "unknown"),
        "failure_reason": run_metadata.get("failure_reason"),
    }
    for metric in METRICS:
        row[metric] = evaluation.get(metric)
    return row


def collect_results(runs_root: Path) -> pd.DataFrame:
    rows = [_row_from_evaluation(path) for path in sorted(runs_root.rglob("evaluation.json"))]
    return pd.DataFrame(rows)


def _parse_csv(value: str | None, cast=str) -> set | None:
    if value is None:
        return None
    parsed = {cast(item.strip()) for item in value.split(",") if item.strip()}
    return parsed or None


def _filter_results(
    df: pd.DataFrame,
    *,
    methods: set[str] | None,
    datasets: set[str] | None,
    seeds: set[int] | None,
    rhos: set[float] | None,
    strict_seed_dirs: bool,
) -> pd.DataFrame:
    out = df.copy()
    if methods is not None and "method" in out.columns:
        out = out[out["method"].astype(str).isin(methods)]
    if datasets is not None and "dataset" in out.columns:
        out = out[out["dataset"].astype(str).isin(datasets)]
    if seeds is not None and "seed" in out.columns:
        out = out[out["seed"].astype("Int64").isin(seeds)]
        if strict_seed_dirs and "run_dir" in out.columns:
            valid_names = {f"seed{seed}" for seed in seeds}
            out = out[out["run_dir"].map(lambda value: Path(str(value)).name in valid_names)]
    if rhos is not None and "rho_total" in out.columns:
        out = out[out["rho_total"].astype(float).isin(rhos)]
    return out


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        columns = ["dataset", "method", "method_slug", "n"]
        for label, _ in SUMMARY_METRICS:
            columns.extend([f"{label}_mean", f"{label}_sem"])
        return pd.DataFrame(columns=columns)

    group_cols = ["dataset", "method"]
    rows: list[dict[str, object]] = []
    for (dataset, method_slug), group in df.groupby(group_cols, dropna=False, sort=True):
        row: dict[str, object] = {
            "dataset": dataset,
            "method": DISPLAY_NAMES.get(str(method_slug), str(method_slug)),
            "method_slug": method_slug,
            "n": int(group["seed"].nunique()) if "seed" in group.columns else int(len(group)),
        }
        for label, metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{label}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{label}_sem"] = float(values.sem()) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown_summary(df: pd.DataFrame, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text("# External Results Summary\n\nNo `evaluation.json` files found.\n")
        return
    group_cols = ["dataset", "rho_total", "method"]
    metric_cols = [col for col in METRICS if col in df.columns]
    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary = summary.fillna(0.0)
    flat_cols = []
    for col in summary.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join(str(part) for part in col if part))
        else:
            flat_cols.append(str(col))
    summary.columns = flat_cols
    table_cols = list(summary.columns)
    lines = [
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join("---" for _ in table_cols) + " |",
    ]
    for _, row in summary.iterrows():
        values = []
        for col in table_cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_md.write_text(
        "# External Results Summary\n\n"
        f"Rows: {len(df)}\n\n"
        + "\n".join(lines)
        + "\n"
    )


def _write_summary_markdown(summary: pd.DataFrame, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        output_md.write_text("# External Results Aggregate Summary\n\nNo rows.\n")
        return
    columns = list(summary.columns)
    lines = [
        "# External Results Aggregate Summary",
        "",
        f"Rows: {len(summary)}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in summary.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect canonical external-baseline evaluation JSON files.")
    parser.add_argument("--runs-root", type=Path, default=external_runs())
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--methods")
    parser.add_argument("--datasets")
    parser.add_argument("--seeds")
    parser.add_argument("--rhos")
    parser.add_argument(
        "--strict-seed-dirs",
        action="store_true",
        help="When --seeds is used, keep only run directories named exactly seedN; excludes smoke/phase suffixes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_results(args.runs_root)
    df = _filter_results(
        df,
        methods=_parse_csv(args.methods, str),
        datasets=_parse_csv(args.datasets, str),
        seeds=_parse_csv(args.seeds, int),
        rhos=_parse_csv(args.rhos, float),
        strict_seed_dirs=bool(args.strict_seed_dirs),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    if args.output_md is not None:
        _write_markdown_summary(df, args.output_md)
    if args.summary_csv is not None or args.summary_md is not None:
        summary = summarize_results(df)
        if args.summary_csv is not None:
            args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(args.summary_csv, index=False)
        if args.summary_md is not None:
            _write_summary_markdown(summary, args.summary_md)
    print(f"wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
