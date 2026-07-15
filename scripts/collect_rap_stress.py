#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from path_defaults import external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, external_runs

RUNS_ROOT = external_runs()
RESULTS_ROOT = external_results()

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
    return json.loads(path.read_text())


def _run_dir(root: Path, dataset: str, rho: float, seed: int, setting: str) -> Path:
    rho_label = str(float(rho)).replace(".", "p")
    return root / "rap" / dataset / f"rho{rho_label}" / f"seed{seed}_{setting}"


def collect(dataset: str, rho: float, seeds: list[int], setting: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = _run_dir(RUNS_ROOT, dataset, rho, seed, setting)
        eval_path = run_dir / "evaluation.json"
        metadata_path = run_dir / "run_metadata.json"
        if not eval_path.exists():
            rows.append(
                {
                    "dataset": dataset,
                    "rho_total": rho,
                    "seed": seed,
                    "setting": setting,
                    "status": "missing",
                    "run_dir": str(run_dir),
                }
            )
            continue
        evaluation = _read_json(eval_path)
        metadata = _read_json(metadata_path) if metadata_path.exists() else evaluation.get("run_metadata", {})
        notes = metadata.get("notes") or {}
        row = {
            "dataset": dataset,
            "rho_total": metadata.get("rho_total", rho),
            "seed": metadata.get("seed", seed),
            "setting": setting,
            "method": metadata.get("method", "rap_softmax"),
            "status": metadata.get("status", "unknown"),
            "run_dir": str(run_dir),
            "runtime_seconds": metadata.get("runtime_seconds", evaluation.get("runtime_seconds")),
            "torch_device": metadata.get("torch_device"),
            "torch_cuda_available": metadata.get("torch_cuda_available"),
            "T": notes.get("T"),
            "K": notes.get("K"),
            "num_marginals": notes.get("num_marginals"),
            "num_queries": notes.get("num_queries"),
            "model_rows": metadata.get("model_rows"),
            "max_iters": notes.get("max_iters"),
            "decode_mode": notes.get("decode_mode"),
        }
        for metric in METRICS:
            row[metric] = evaluation.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else math.nan
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["status"] != "missing"].copy()
    if valid.empty:
        return pd.DataFrame()
    group_cols = [
        "dataset",
        "rho_total",
        "setting",
        "method",
        "T",
        "K",
        "num_marginals",
        "num_queries",
        "model_rows",
        "max_iters",
        "decode_mode",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in valid.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["seed_count"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique()))
        row["torch_devices"] = ",".join(sorted(set(str(device) for device in group["torch_device"].dropna())))
        runtime = pd.to_numeric(group["runtime_seconds"], errors="coerce")
        row["runtime_seconds_mean"] = float(runtime.mean())
        row["runtime_seconds_sem"] = _sem(runtime)
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else math.nan
            row[f"{metric}_sem"] = _sem(values)
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


def write_markdown(df: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# RAP Stress Summary",
        "",
        "Diagnostic appendix/secondary-table evidence. Lower is better for all utility metrics.",
        "",
    ]
    if not summary.empty:
        lines += [
            "## Mean Across Seeds",
            "",
            "| dataset | setting | n | T | K | M | queries | rows | MAE | RMSE | AvgTVD | MaxErr | MaxTVD | runtime_s | device |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for _, row in summary.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(row["dataset"]),
                        _fmt(row["setting"]),
                        _fmt(row["seed_count"]),
                        _fmt(row["T"]),
                        _fmt(row["K"]),
                        _fmt(row["num_marginals"]),
                        _fmt(row["num_queries"]),
                        _fmt(row["model_rows"]),
                        _fmt_mean_sem(row, "full_true_mae_mean", "full_true_mae_sem"),
                        _fmt_mean_sem(row, "full_true_rmse_mean", "full_true_rmse_sem"),
                        _fmt_mean_sem(row, "full_true_avg_tvd_mean", "full_true_avg_tvd_sem"),
                        _fmt_mean_sem(row, "full_true_max_error_mean", "full_true_max_error_sem"),
                        _fmt_mean_sem(row, "full_true_max_tvd_mean", "full_true_max_tvd_sem"),
                        _fmt_mean_sem(row, "runtime_seconds_mean", "runtime_seconds_sem"),
                        _fmt(row["torch_devices"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines += [
        "## Per-seed Rows",
        "",
        "| dataset | seed | status | MAE | RMSE | AvgTVD | MaxErr | MaxTVD | runtime_s | device | run_dir |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in df.sort_values(["dataset", "rho_total", "seed"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("dataset")),
                    _fmt(row.get("seed")),
                    _fmt(row.get("status")),
                    _fmt(row.get("full_true_mae")),
                    _fmt(row.get("full_true_rmse")),
                    _fmt(row.get("full_true_avg_tvd")),
                    _fmt(row.get("full_true_max_error")),
                    _fmt(row.get("full_true_max_tvd")),
                    _fmt(row.get("runtime_seconds")),
                    _fmt(row.get("torch_device")),
                    _fmt(row.get("run_dir")),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RAP stress-audit rows.")
    parser.add_argument("--dataset", default="adult_sage_strong")
    parser.add_argument(
        "--dataset-settings",
        default="",
        help=(
            "Comma-separated dataset:setting pairs. When provided, this overrides "
            "--dataset/--setting and allows one table to include heterogeneous "
            "stress settings such as br2000_sage_strong:T30K30M364I1000."
        ),
    )
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--setting", default="T30K30M445I1000")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_ROOT / "rap_adult_sage_strong_stress_multiseed_rho1_20260706.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=RESULTS_ROOT / "rap_adult_sage_strong_stress_multiseed_summary_rho1_20260706.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_ROOT / "rap_adult_sage_strong_stress_multiseed_rho1_20260706.md",
    )
    return parser.parse_args()


def _dataset_settings(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not str(args.dataset_settings).strip():
        return [(str(args.dataset), str(args.setting))]
    pairs: list[tuple[str, str]] = []
    for item in _parse_csv(args.dataset_settings):
        if ":" not in item:
            raise ValueError(f"--dataset-settings item must be dataset:setting, got {item!r}")
        dataset, setting = item.split(":", 1)
        dataset = dataset.strip()
        setting = setting.strip()
        if not dataset or not setting:
            raise ValueError(f"--dataset-settings item must be dataset:setting, got {item!r}")
        pairs.append((dataset, setting))
    return pairs


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in _parse_csv(args.seeds)]
    frames = [collect(dataset, args.rho, seeds, setting) for dataset, setting in _dataset_settings(args)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    write_markdown(df, summary, args.output_md)
    print(f"wrote {len(df)} rows to {args.output_csv}")
    print(f"wrote {len(summary)} summary rows to {args.summary_csv}")


if __name__ == "__main__":
    main()
