#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from path_defaults import external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, external_runs


DEFAULT_RUNS = {
    "adult_sage_strong": "seed0_T30M445D512S1000I100_cached_sample_remap_diag",
    "acs_sage_strong": "seed0_T30M445D512S1000I100_cached_sample_remap_diag",
    "br2000_sage_strong": "seed0_T30M364D512S1000I100_cached_sample_remap_diag",
    "nltcs_sage_strong": "seed0_T30M445D512S1000I100_cached_sample_remap_diag",
}

METRICS = [
    "full_true_mae",
    "full_true_rmse",
    "full_true_avg_tvd",
    "full_true_max_error",
    "full_true_max_tvd",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rho_label(rho: float) -> str:
    return str(float(rho)).replace(".", "p")


def collect(runs_root: Path, rho: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, run_name in DEFAULT_RUNS.items():
        run_dir = runs_root / "gem" / dataset / f"rho{_rho_label(rho)}" / run_name
        metadata_path = run_dir / "run_metadata.json"
        evaluation_path = run_dir / "evaluation.json"
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": "GEM remap",
            "method_slug": "gem_remap",
            "rho_total": rho,
            "seed": 0,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "metadata_path": str(metadata_path),
            "evaluation_path": str(evaluation_path),
        }
        if not metadata_path.exists() or not evaluation_path.exists():
            row["status"] = "missing"
            rows.append(row)
            continue

        metadata = _read_json(metadata_path)
        evaluation = _read_json(evaluation_path)
        diagnostics = metadata.get("internal_diagnostics") or {}
        relaxed = (diagnostics.get("relaxed_cached_batch") or {}).get("mae")
        row_mae = (diagnostics.get("row_sampled_synthetic") or {}).get("mae")
        remap = (metadata.get("notes") or {}).get("transformer_query_remap") or {}
        notes = metadata.get("notes") or {}

        row.update(
            {
                "status": metadata.get("status"),
                "runtime_seconds": metadata.get("runtime_seconds"),
                "torch_cuda_available": metadata.get("torch_cuda_available"),
                "torch_device": metadata.get("torch_device"),
                "T": notes.get("T"),
                "num_marginals": notes.get("num_marginals"),
                "num_queries": notes.get("num_queries"),
                "dim": notes.get("dim"),
                "syndata_size": notes.get("syndata_size"),
                "decode_mode": notes.get("decode_mode"),
                "latent_mode": notes.get("latent_mode"),
                "query_remap": bool(remap.get("enabled", False)),
                "query_remap_non_identity_columns": remap.get("non_identity_columns"),
                "query_remap_non_identity_positions": remap.get("non_identity_positions"),
                "internal_relaxed_cached_mae": relaxed,
                "internal_row_mae": row_mae,
                "internal_row_mae_over_relaxed_cached": (
                    float(row_mae) / float(relaxed) if relaxed not in (None, 0) and row_mae is not None else ""
                ),
            }
        )
        for metric in METRICS:
            row[metric] = evaluation.get(metric)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "method",
        "method_slug",
        "rho_total",
        "seed",
        "status",
        "T",
        "num_marginals",
        "num_queries",
        "dim",
        "syndata_size",
        "decode_mode",
        "latent_mode",
        "query_remap",
        "query_remap_non_identity_columns",
        "query_remap_non_identity_positions",
        "internal_relaxed_cached_mae",
        "internal_row_mae",
        "internal_row_mae_over_relaxed_cached",
        *METRICS,
        "runtime_seconds",
        "torch_cuda_available",
        "torch_device",
        "run_name",
        "run_dir",
        "metadata_path",
        "evaluation_path",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], output_md: Path) -> None:
    lines = [
        "# GEM Remapped Seed0 All-4 Diagnostic",
        "",
        "This table evaluates the remapped GEM wrapper on the same four SAGE-strong encoded datasets.",
        "It is a seed0 diagnostic, not a main-table multi-seed baseline.",
        "",
        "| dataset | M | queries | row/relaxed | MAE | RMSE | AvgTVD | MaxErr | MaxTVD | runtime_s | device |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("dataset")),
                    _fmt(row.get("num_marginals")),
                    _fmt(row.get("num_queries")),
                    _fmt(row.get("internal_row_mae_over_relaxed_cached")),
                    _fmt(row.get("full_true_mae")),
                    _fmt(row.get("full_true_rmse")),
                    _fmt(row.get("full_true_avg_tvd")),
                    _fmt(row.get("full_true_max_error")),
                    _fmt(row.get("full_true_max_tvd")),
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
        "- The public transformer-coordinate remap keeps internal row/relaxed MAE near 1x on all four datasets.",
        "- Despite the remap, GEM remains much weaker than the current primary SAGE/GSD/AIM/MST/RAP rows under the shared strong evaluator.",
        "- The current evidence does not justify spending GPU time on a full seed0--4 remapped GEM grid unless a reviewer specifically requests it.",
    ]
    output_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect remapped GEM seed0 all-4 diagnostics.")
    parser.add_argument("--runs-root", type=Path, default=external_runs())
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=external_results() / "gem_remap_all4_seed0_20260707.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=external_results() / "gem_remap_all4_seed0_20260707.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(args.runs_root, args.rho)
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
