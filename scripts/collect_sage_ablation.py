#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

try:
    from path_defaults import external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, external_runs

RUNS_ROOT = external_runs()
RESULTS_ROOT = external_results()

DATASETS = [
    "adult_sage_strong",
    "acs_sage_strong",
    "br2000_sage_strong",
    "nltcs_sage_strong",
]

DATASET_LABELS = {
    "adult_sage_strong": "Adult",
    "acs_sage_strong": "ACS",
    "br2000_sage_strong": "BR2000",
    "nltcs_sage_strong": "NLTCS",
}

VARIANTS = [
    ("full", "sage", "SAGE", RUNS_ROOT / "sage"),
    (
        "no_projection",
        "sage_no_projection",
        "No projection",
        RUNS_ROOT / "sage_ablation" / "no_projection",
    ),
    (
        "unweighted_objective",
        "sage_unweighted_objective",
        "Unweighted objective",
        RUNS_ROOT / "sage_ablation" / "unweighted_objective",
    ),
    (
        "low_order_workload",
        "sage_low_order_workload",
        "Low-order workload",
        RUNS_ROOT / "sage_ablation" / "low_order_workload",
    ),
]

METRICS = [
    "full_true_mae",
    "full_true_rmse",
    "full_true_avg_tvd",
    "full_true_max_error",
    "full_true_max_tvd",
]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return orjson.loads(path.read_bytes())


def _run_dir(root: Path, dataset: str, rho: float, seed: int) -> Path:
    rho_label = str(float(rho)).replace(".", "p")
    return root / dataset / f"rho{rho_label}" / f"seed{seed}"


def collect(rho: float, seed: int, datasets: list[str], variant_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_variants = [variant for variant in VARIANTS if variant[0] in set(variant_names)]
    if not selected_variants:
        raise ValueError(f"No variants selected from {variant_names}. Available: {[variant[0] for variant in VARIANTS]}")
    for dataset in datasets:
        for variant, expected_method, label, root in selected_variants:
            run_dir = _run_dir(root, dataset, rho, seed)
            eval_path = run_dir / "evaluation.json"
            metadata_path = run_dir / "run_metadata.json"
            workload_path = run_dir / "workload_summary.json"
            if not eval_path.exists():
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "variant": variant,
                        "variant_label": label,
                        "method": expected_method,
                        "rho_total": rho,
                        "seed": seed,
                        "status": "missing",
                        "run_dir": str(run_dir),
                    }
                )
                continue
            evaluation = _read_json(eval_path)
            metadata = _read_json(metadata_path) if metadata_path.exists() else evaluation.get("run_metadata", {})
            workload = _read_json(workload_path) if workload_path.exists() else {}
            row = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "variant": variant,
                "variant_label": label,
                "method": metadata.get("method", expected_method),
                "rho_total": metadata.get("rho_total", rho),
                "seed": metadata.get("seed", seed),
                "status": metadata.get("status", "unknown"),
                "run_dir": str(run_dir),
                "measured_queries": workload.get("total_num_queries", workload.get("total_queries")),
                "measured_groups": sum((workload.get("num_groups_by_family") or {}).values())
                if isinstance(workload.get("num_groups_by_family"), dict)
                else None,
                "runtime_seconds": metadata.get("runtime_seconds", evaluation.get("runtime_seconds")),
            }
            for metric in METRICS:
                row[metric] = evaluation.get(metric)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for metric in METRICS:
        full_by_dataset = (
            df[df["variant"] == "full"]
            .set_index(["dataset", "seed"])[metric]
            .to_dict()
        )
        df[f"{metric}_delta_vs_full"] = [
            (float(value) - float(full_by_dataset[(dataset, seed)]))
            if value is not None and (dataset, seed) in full_by_dataset and pd.notna(value)
            else None
            for dataset, seed, value in zip(df["dataset"], df["seed"], df[metric])
        ]
        df[f"{metric}_ratio_vs_full"] = [
            (float(value) / float(full_by_dataset[(dataset, seed)]))
            if value is not None
            and (dataset, seed) in full_by_dataset
            and float(full_by_dataset[(dataset, seed)]) != 0.0
            and pd.notna(value)
            else None
            for dataset, seed, value in zip(df["dataset"], df["seed"], df[metric])
        ]
    return df


def collect_many(rho: float, seeds: list[int], datasets: list[str], variant_names: list[str]) -> pd.DataFrame:
    frames = [collect(rho, seed, datasets, variant_names) for seed in seeds]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else math.nan
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    valid = df[df["status"] != "missing"].copy()
    if valid.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "dataset_label", "variant", "variant_label", "method", "rho_total"]
    rows: list[dict[str, Any]] = []
    for keys, group in valid.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["seed_count"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique()))
        for key in ["measured_queries", "measured_groups", "runtime_seconds"]:
            values = pd.to_numeric(group[key], errors="coerce")
            row[f"{key}_mean"] = float(values.mean()) if values.notna().any() else math.nan
            row[f"{key}_sem"] = _sem(values)
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else math.nan
            row[f"{metric}_sem"] = _sem(values)
            ratio_key = f"{metric}_ratio_vs_full"
            ratios = pd.to_numeric(group[ratio_key], errors="coerce")
            row[f"{ratio_key}_mean"] = float(ratios.mean()) if ratios.notna().any() else math.nan
            row[f"{ratio_key}_sem"] = _sem(ratios)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_mean_sem(row: pd.Series, mean_col: str, sem_col: str) -> str:
    mean = row.get(mean_col)
    sem = row.get(sem_col)
    if mean is None or pd.isna(mean):
        return ""
    if sem is None or pd.isna(sem) or float(sem) == 0.0:
        return _fmt(mean)
    return f"{float(mean):.6g} +/- {float(sem):.2g}"


def write_markdown(df: pd.DataFrame, output: Path, summary: pd.DataFrame | None = None) -> None:
    lines = [
        "# SAGE Ablation Summary",
        "",
        "Lower is better for all utility metrics.",
        "",
    ]
    if df.empty:
        lines.append("No rows found.")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    if summary is not None and not summary.empty:
        lines.extend(
            [
                "Values are mean +/- SEM across available seeds. Ratio columns are computed against full SAGE at the same dataset and seed before aggregation.",
                "",
            ]
        )
        columns = [
            ("variant_label", "Variant"),
            ("seed_count", "n"),
            ("measured_queries_mean", "Measured queries"),
            ("full_true_mae_mean", "MAE"),
            ("full_true_rmse_mean", "RMSE"),
            ("full_true_avg_tvd_mean", "AvgTVD"),
            ("full_true_max_error_mean", "MaxErr"),
            ("full_true_max_tvd_mean", "MaxTVD"),
            ("runtime_seconds_mean", "runtime_s"),
            ("full_true_mae_ratio_vs_full_mean", "MAE / SAGE"),
            ("full_true_rmse_ratio_vs_full_mean", "RMSE / SAGE"),
            ("full_true_avg_tvd_ratio_vs_full_mean", "AvgTVD / SAGE"),
        ]
        sem_columns = {
            "full_true_mae_mean": "full_true_mae_sem",
            "full_true_rmse_mean": "full_true_rmse_sem",
            "full_true_avg_tvd_mean": "full_true_avg_tvd_sem",
            "full_true_max_error_mean": "full_true_max_error_sem",
            "full_true_max_tvd_mean": "full_true_max_tvd_sem",
            "runtime_seconds_mean": "runtime_seconds_sem",
            "full_true_mae_ratio_vs_full_mean": "full_true_mae_ratio_vs_full_sem",
            "full_true_rmse_ratio_vs_full_mean": "full_true_rmse_ratio_vs_full_sem",
            "full_true_avg_tvd_ratio_vs_full_mean": "full_true_avg_tvd_ratio_vs_full_sem",
        }
        for dataset in DATASETS:
            sub = summary[summary["dataset"] == dataset].copy()
            if sub.empty:
                continue
            sub["variant_order"] = sub["variant"].map({name: idx for idx, (name, *_rest) in enumerate(VARIANTS)})
            sub = sub.sort_values("variant_order")
            lines.extend([f"## {DATASET_LABELS.get(dataset, dataset)}", ""])
            lines.append("| " + " | ".join(label for _, label in columns) + " |")
            lines.append("|---" + "|---:" * (len(columns) - 1) + "|")
            for _, row in sub.iterrows():
                values = []
                for column, _label in columns:
                    if column in sem_columns:
                        values.append(_fmt_mean_sem(row, column, sem_columns[column]))
                    else:
                        values.append(_fmt(row[column]))
                lines.append("| " + " | ".join(values) + " |")
            lines.append("")
        output.write_text("\n".join(lines), encoding="utf-8")
        return
    columns = [
        "variant_label",
        "measured_queries",
        "full_true_mae",
        "full_true_rmse",
        "full_true_avg_tvd",
        "full_true_max_error",
        "full_true_max_tvd",
        "runtime_seconds",
    ]
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue
        sub["variant_order"] = sub["variant"].map({name: idx for idx, (name, *_rest) in enumerate(VARIANTS)})
        sub = sub.sort_values("variant_order")
        lines.extend([f"## {DATASET_LABELS.get(dataset, dataset)}", ""])
        lines.append("| Variant | Measured queries | MAE | RMSE | AvgTVD | MaxErr | MaxTVD | runtime_s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(_fmt(row[col]) for col in columns)
                + " |"
            )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SAGE ablation runs into paper-facing tables.")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0, help="Backward-compatible single-seed selector.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Overrides --seed when provided.")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--variants", default=",".join(variant[0] for variant in VARIANTS))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_ROOT / "sage_ablation_seed0_rho1_20260706.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_ROOT / "sage_ablation_seed0_rho1_20260706.md",
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = _parse_csv(args.datasets)
    seeds = [int(seed) for seed in _parse_csv(args.seeds)] if args.seeds else [int(args.seed)]
    variants = _parse_csv(args.variants)
    df = collect_many(float(args.rho), seeds, datasets, variants)
    summary = summarize(df) if len(seeds) > 1 else None
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        (summary if summary is not None else summarize(df)).to_csv(args.summary_csv, index=False)
    write_markdown(df, args.output_md, summary)
    print(f"wrote {len(df)} rows to {args.output_csv}")
    if summary is not None:
        print(f"summarized {len(summary)} variant/dataset rows across seeds {seeds}")


if __name__ == "__main__":
    main()
