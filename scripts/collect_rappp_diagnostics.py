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

RUNS_ROOT = external_runs()
RESULTS_ROOT = external_results()

DEFAULT_RUNS = [
    "nltcs_sage_strong:seed0_smoke_mr32_e1_i0",
    "nltcs_sage_strong:seed0_strong_mr512_e5_i1",
    "nltcs_sage_strong:seed0_strong_mr2048_e10_i1",
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
    return json.loads(path.read_text())


def _rho_label(rho: float) -> str:
    return str(float(rho)).replace(".", "p")


def _parse_run_item(item: str) -> tuple[str, str]:
    if ":" not in item:
        raise ValueError(f"run item must be dataset:run_name, got {item!r}")
    dataset, run_name = item.split(":", 1)
    return dataset.strip(), run_name.strip()


def collect(runs: list[str], rho: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in runs:
        dataset, run_name = _parse_run_item(item)
        run_dir = RUNS_ROOT / "rappp_marginal" / dataset / f"rho{_rho_label(rho)}" / run_name
        metadata_path = run_dir / "run_metadata.json"
        eval_path = run_dir / "evaluation.json"
        row: dict[str, Any] = {
            "nominal_dataset": dataset,
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
        input_dir = Path(str(metadata.get("input_dir", ""))).name
        eval_dataset = str(evaluation.get("dataset") or "")
        row.update(
            {
                "status": metadata.get("status", "unknown"),
                "method": metadata.get("method", "rappp_marginal"),
                "seed": metadata.get("seed"),
                "input_dataset": input_dir,
                "eval_dataset": eval_dataset,
                "strong_eval": eval_dataset.endswith("_sage_strong"),
                "runtime_seconds": metadata.get("runtime_seconds"),
                "conda_env": metadata.get("conda_env"),
                "jax_backend": notes.get("jax_default_backend"),
                "jax_devices": ",".join(str(x) for x in notes.get("jax_devices") or []),
                "model_rows": metadata.get("model_rows"),
                "n_real": metadata.get("n_real"),
                "n_synthetic": metadata.get("n_synthetic"),
                "k": notes.get("k"),
                "top_q": notes.get("top_q"),
                "dp_select_epochs": notes.get("dp_select_epochs"),
                "iterations": notes.get("iterations"),
                "sigmoid_doubles": notes.get("sigmoid_doubles"),
                "decode_mode": notes.get("decode_mode"),
                "not_full_official_rappp": notes.get("not_full_official_rappp"),
                "num_eval_queries": evaluation.get("num_queries"),
                "num_eval_vector_blocks": evaluation.get("num_vector_blocks"),
            }
        )
        for metric in METRICS:
            row[metric] = evaluation.get(metric)
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
        "# RAP++ Marginal-only Path-check",
        "",
        "These rows are not official full RAP++ evidence. They use the RAP++ projection implementation with marginal statistics only. The full RAP++ marginal-plus-halfspace/task-target interface is not represented by the current categorical benchmark wrapper.",
        "",
        "| run | input | evaluator | strong eval | rows | epochs | iters | MAE | RMSE | AvgTVD | MaxErr | MaxTVD | runtime_s | backend |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("run_name")),
                    _fmt(row.get("input_dataset")),
                    _fmt(row.get("eval_dataset")),
                    _fmt(row.get("strong_eval")).lower(),
                    _fmt(row.get("model_rows")),
                    _fmt(row.get("dp_select_epochs")),
                    _fmt(row.get("iterations")),
                    _fmt(row.get("full_true_mae")),
                    _fmt(row.get("full_true_rmse")),
                    _fmt(row.get("full_true_avg_tvd")),
                    _fmt(row.get("full_true_max_error")),
                    _fmt(row.get("full_true_max_tvd")),
                    _fmt(row.get("runtime_seconds")),
                    _fmt(row.get("jax_backend")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The original smoke row used the weak `nltcs` evaluator and should not be cited as strong-workload evidence.",
        "- The stronger `nltcs_sage_strong` probes improve substantially when model rows and select epochs increase, but remain far from competitive with SAGE, Private-GSD GPU, AIM, MST, and RAP softmax under the shared row-level evaluator.",
        "- Because the wrapper is marginal-only and not the official full RAP++ setup, these rows should remain path-check evidence unless a semantically faithful RAP++ conversion is designed.",
    ]
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RAP++ marginal-only path-check diagnostics.")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_ROOT / "rappp_marginal_pathcheck_20260706.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_ROOT / "rappp_marginal_pathcheck_20260706.md",
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
