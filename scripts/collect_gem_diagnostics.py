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

DEFAULT_RUNS = [
    "adult_sage_strong:seed0_T30M445D512S1000I100_cached_sample_diag",
    "adult_sage_strong:seed0_T30M445D512S1000I100_cached_sample_remap_diag",
    "adult_sage_strong:seed0_T30M445D512S1000I100_cached_balanced_remap_diag",
    "adult_sage_strong:seed0_T30M445D512S1000I100_cached_argmax_remap_diag",
    "adult_sage_strong:seed0_T30M286D512S1000I100_cached_diag",
    "adult_sage_strong:seed0_T30M445D512S1000I100_cached_balanced_diag",
    "adult_sage_strong:seed0_T30M445D512S4096I100_cached_sample_diag",
    "adult_sage_strong:seed0_T30M445D512S1000I100_cached_argmax_diag",
    "adult_sage_strong:seed0_T30M445D512S1000I100_fresh_sample_diag",
    "adult_sage_strong:seed0_T30M445D512S1000I100_fresh_argmax_diag",
]

EXTERNAL_METRICS = [
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


def _rho_label(rho: float) -> str:
    return str(float(rho)).replace(".", "p")


def _parse_run_item(item: str) -> tuple[str, str]:
    if ":" not in item:
        raise ValueError(f"run item must be dataset:run_name, got {item!r}")
    dataset, run_name = item.split(":", 1)
    dataset = dataset.strip()
    run_name = run_name.strip()
    if not dataset or not run_name:
        raise ValueError(f"run item must be dataset:run_name, got {item!r}")
    return dataset, run_name


def _metric(prefix: str, metrics: dict[str, Any], key: str) -> tuple[str, Any]:
    return f"{prefix}_{key}", metrics.get(key)


def collect(runs: list[str], rho: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in runs:
        dataset, run_name = _parse_run_item(item)
        run_dir = RUNS_ROOT / "gem" / dataset / f"rho{_rho_label(rho)}" / run_name
        metadata_path = run_dir / "run_metadata.json"
        eval_path = run_dir / "evaluation.json"
        row: dict[str, Any] = {
            "dataset": dataset,
            "run_name": run_name,
            "rho_total": rho,
            "run_dir": str(run_dir),
        }
        if not metadata_path.exists():
            row["status"] = "missing_metadata"
            rows.append(row)
            continue
        metadata = _read_json(metadata_path)
        evaluation = _read_json(eval_path) if eval_path.exists() else {}
        notes = metadata.get("notes") or {}
        diagnostics = metadata.get("internal_diagnostics") or _read_json(run_dir / "internal_diagnostics.json")
        row.update(
            {
                "status": metadata.get("status", "unknown"),
                "seed": metadata.get("seed"),
                "method": metadata.get("method", "gem"),
                "runtime_seconds": metadata.get("runtime_seconds"),
                "torch_cuda_available": metadata.get("torch_cuda_available"),
                "torch_device": metadata.get("torch_device"),
                "T": notes.get("T"),
                "alpha": notes.get("alpha"),
                "num_marginals": notes.get("num_marginals"),
                "num_queries": notes.get("num_queries"),
                "dim": notes.get("dim"),
                "syndata_size": notes.get("syndata_size"),
                "max_iters": notes.get("max_iters"),
                "max_idxs": notes.get("max_idxs"),
                "latent_mode": notes.get("latent_mode") or ("cached" if "cached" in run_name else "fresh_legacy"),
                "decode_mode": notes.get("decode_mode"),
            }
        )
        remap = notes.get("transformer_query_remap") or {}
        row["query_remap"] = bool(remap.get("enabled", False))
        row["query_remap_non_identity_columns"] = remap.get("non_identity_columns")
        row["query_remap_non_identity_positions"] = remap.get("non_identity_positions")
        for prefix, diag_key in [
            ("internal_relaxed_cached", "relaxed_cached_batch"),
            ("internal_relaxed_fresh", "relaxed_fresh_batch"),
            ("internal_row", "row_sampled_synthetic"),
        ]:
            values = diagnostics.get(diag_key) or {}
            for key in ["mae", "rmse", "max_error"]:
                col, value = _metric(prefix, values, key)
                row[col] = value
        for metric in EXTERNAL_METRICS:
            row[metric] = evaluation.get(metric)
        cached_mae = row.get("internal_relaxed_cached_mae")
        row_mae = row.get("internal_row_mae")
        external_mae = row.get("full_true_mae")
        if cached_mae not in (None, 0) and not pd.isna(cached_mae):
            row["internal_row_mae_over_relaxed_cached"] = float(row_mae) / float(cached_mae)
            row["external_mae_over_relaxed_cached"] = float(external_mae) / float(cached_mae) if external_mae is not None else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(df: pd.DataFrame, output: Path) -> None:
    lines = [
        "# GEM Row-realization Diagnostics",
        "",
        "Adult SAGE-strong diagnostic rows. Lower is better for all utility metrics.",
        "",
        "The internal relaxed columns evaluate GEM's trained soft distribution on its own exact-marginal workload.",
        "The internal row columns evaluate the discrete row table decoded from that distribution on the same workload.",
        "The external columns evaluate the same row table with the shared SAGE external evaluator.",
        "",
        "| run | M | remap | latent | decode | relaxed MAE | row MAE | row/relaxed | external MAE | external/relaxed | external AvgTVD | external MaxErr | runtime_s | device |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.sort_values(["num_marginals", "latent_mode", "decode_mode", "run_name"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("run_name")),
                    _fmt(row.get("num_marginals")),
                    "yes" if bool(row.get("query_remap")) else "no",
                    _fmt(row.get("latent_mode")),
                    _fmt(row.get("decode_mode")),
                    _fmt(row.get("internal_relaxed_cached_mae")),
                    _fmt(row.get("internal_row_mae")),
                    _fmt(row.get("internal_row_mae_over_relaxed_cached")),
                    _fmt(row.get("full_true_mae")),
                    _fmt(row.get("external_mae_over_relaxed_cached")),
                    _fmt(row.get("full_true_avg_tvd")),
                    _fmt(row.get("full_true_max_error")),
                    _fmt(row.get("runtime_seconds")),
                    _fmt(row.get("torch_device")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The pre-remap GEM wrapper used QueryManager category-id coordinates directly against RDT one-hot positions.",
        "- RDT keeps first-seen category order, so many categorical coordinates were permuted in the relaxed GEM objective.",
        "- The remapped wrapper fixes this public coordinate mismatch; on the 445-marginal Adult diagnostic, row/relaxed MAE drops from about 2.56x to about 1.00x for sample decoding.",
        "- `sample` decoding is the best remapped row realization among the tested decoders; `balanced` is close, while `argmax` remains worse.",
        "- This Adult-only table is complemented by `gem_remap_all4_seed0_20260707`; GEM still remains appendix/diagnostic evidence because that all-dataset seed0 diagnostic is weak under the shared evaluator.",
    ]
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GEM row-realization diagnostics.")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_ROOT / "gem_row_realization_diagnostics_adult_sage_strong_20260706.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_ROOT / "gem_row_realization_diagnostics_adult_sage_strong_20260706.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect(_parse_csv(args.runs), args.rho)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    write_markdown(df, args.output_md)
    print(f"wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
