#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PRIMARY_METRICS = ["measured_loss", "mae", "rmse", "avg_tvd", "max_tvd", "max_error"]
DISPLAY_METRICS = ["measured_loss", "MAE", "RMSE", "AvgTVD", "MaxTVD", "MaxError"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _pct_change(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return (value / baseline - 1.0) * 100.0


def _normalize_metric_key(row: dict[str, Any], key: str) -> float:
    if key in row:
        return float(row[key])
    lower = key.lower()
    if lower in row:
        return float(row[lower])
    raise KeyError(key)


def _aggregate_rows(rows: list[dict[str, Any]], baseline_variant: str) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    if baseline_variant not in by_variant:
        raise ValueError(f"baseline variant {baseline_variant!r} is missing")

    aggregates: dict[str, dict[str, Any]] = {}
    for variant, variant_rows in sorted(by_variant.items()):
        metrics = {
            metric: _mean([_normalize_metric_key(row, metric) for row in variant_rows])
            for metric in PRIMARY_METRICS
        }
        aggregates[variant] = {
            "num_seeds": len({int(row["seed"]) for row in variant_rows if "seed" in row}),
            **metrics,
        }

    baseline = aggregates[baseline_variant]
    for variant, agg in aggregates.items():
        for metric in PRIMARY_METRICS:
            agg[f"{metric}_rel_vs_{baseline_variant}_pct"] = _pct_change(float(agg[metric]), float(baseline[metric]))
    return aggregates


def _summarize_attrs0_14(path: Path, baseline_variant: str, rtp_variant: str) -> dict[str, Any]:
    data = _read_json(path)
    rows = list(data["rows"])
    aggregates = _aggregate_rows(rows, baseline_variant)
    if rtp_variant not in aggregates:
        raise ValueError(f"RTP variant {rtp_variant!r} is missing")
    rtp = aggregates[rtp_variant]
    return {
        "path": str(path),
        "baseline_variant": baseline_variant,
        "rtp_variant": rtp_variant,
        "aggregates": aggregates,
        "signals": {
            "average_metrics_improve": all(float(rtp[f"{metric}_rel_vs_{baseline_variant}_pct"]) < 0.0 for metric in ["mae", "rmse", "avg_tvd"]),
            "max_tvd_improves": float(rtp[f"max_tvd_rel_vs_{baseline_variant}_pct"]) < 0.0,
            "max_error_regresses": float(rtp[f"max_error_rel_vs_{baseline_variant}_pct"]) > 0.0,
        },
    }


def _summarize_attrs3_4(path: Path, baseline_variant: str, rtp_variant: str) -> dict[str, Any]:
    data = _read_json(path)
    rows = list(data["rows"])
    by_variant = {str(row["variant"]): row for row in rows}
    if baseline_variant not in by_variant or rtp_variant not in by_variant:
        raise ValueError(f"Expected variants {baseline_variant!r} and {rtp_variant!r} in {path}")
    base = by_variant[baseline_variant]
    rtp = by_variant[rtp_variant]
    rel = {
        metric: _pct_change(_normalize_metric_key(rtp, metric), _normalize_metric_key(base, metric))
        for metric in DISPLAY_METRICS
    }
    return {
        "path": str(path),
        "baseline_variant": baseline_variant,
        "rtp_variant": rtp_variant,
        "relative_vs_baseline_pct": rel,
        "signals": {
            "avg_tvd_improves": rel["AvgTVD"] < 0.0,
            "max_tvd_regresses": rel["MaxTVD"] > 0.0,
            "max_error_improves": rel["MaxError"] < 0.0,
        },
    }


def _transfer_row(rows: list[dict[str, str]], synthetic_variant: str, target_variant: str) -> dict[str, str]:
    for row in rows:
        if row["synthetic_variant"] == synthetic_variant and row["target_variant"] == target_variant:
            return row
    raise ValueError(f"Missing transfer row synthetic={synthetic_variant!r}, target={target_variant!r}")


def _summarize_transfer(path: Path, baseline_variant: str, rtp_variant: str) -> dict[str, Any]:
    rows = _read_csv(path)
    base = _transfer_row(rows, baseline_variant, baseline_variant)
    rtp = _transfer_row(rows, rtp_variant, rtp_variant)
    metrics = {
        "T_target_to_true_norm2": _pct_change(float(rtp["T_target_to_true_norm2"]), float(base["T_target_to_true_norm2"])),
        "F_synthetic_to_target_norm2": _pct_change(float(rtp["F_synthetic_to_target_norm2"]), float(base["F_synthetic_to_target_norm2"])),
        "E_synthetic_to_true_norm2": _pct_change(float(rtp["E_synthetic_to_true_norm2"]), float(base["E_synthetic_to_true_norm2"])),
    }
    return {
        "path": str(path),
        "relative_vs_baseline_pct": metrics,
        "signals": {
            "target_improves": metrics["T_target_to_true_norm2"] < 0.0,
            "fit_improves": metrics["F_synthetic_to_target_norm2"] < 0.0,
            "final_weighted_error_regresses": metrics["E_synthetic_to_true_norm2"] > 0.0,
        },
    }


def _summarize_teacher(path: Path, baseline_loss: float) -> dict[str, Any]:
    data = _read_json(path)
    row = dict(data["row"])
    teacher_loss = float(row["recomputed_measured_loss"])
    return {
        "path": str(path),
        "teacher_target_is_row_realizable": bool(data.get("metadata", {}).get("teacher_target_is_row_realizable", False)),
        "offline_true_answers_used": bool(data.get("metadata", {}).get("offline_true_answers_used", True)),
        "recomputed_measured_loss": teacher_loss,
        "teacher_rate_mae": float(row["teacher_rate_mae"]),
        "teacher_rate_rmse": float(row["teacher_rate_rmse"]),
        "teacher_rate_max_error": float(row["teacher_rate_max_error"]),
        "loss_ratio_vs_baseline": teacher_loss / baseline_loss if baseline_loss else float("inf"),
        "signals": {
            "fits_much_better_than_dp_target": teacher_loss < 0.1 * baseline_loss,
            "uses_no_true_answers": bool(data.get("metadata", {}).get("offline_true_answers_used", True)) is False,
        },
    }


def _derive_decision(summary: dict[str, Any]) -> dict[str, Any]:
    attrs0 = summary["attrs0_14"]["signals"]
    attrs3 = summary["attrs3_4"]["signals"]
    transfer = summary["attrs3_4_transfer"]["signals"]
    teacher = summary["teacher_target"]["signals"]
    findings = [
        "attrs0_14_improves_average_metrics" if attrs0["average_metrics_improve"] else "attrs0_14_average_metrics_not_consistent",
        "attrs0_14_has_tail_regression" if attrs0["max_error_regresses"] else "attrs0_14_tail_not_regressed",
        "attrs3_4_shows_transfer_gap" if transfer["target_improves"] and transfer["fit_improves"] and transfer["final_weighted_error_regresses"] else "attrs3_4_transfer_gap_not_proven",
        "teacher_target_fit_succeeds" if teacher["fits_much_better_than_dp_target"] else "teacher_target_fit_not_strong",
    ]
    return {
        "findings": findings,
        "conclusion": (
            "Local row-realizable projection is useful diagnostically but current single-scope RTP is not a stable paper variant; "
            "teacher-target fit suggests optimizer/transport is not the first-order bottleneck."
        ),
        "recommendation": (
            "Do not add more post-hoc true-tail scopes. Ask for a decision between public multi-scope RTP, proximal C/R coupling, "
            "QDTE-as-column-generation, or freezing RTP as diagnostic/future work."
        ),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs0 = summary["attrs0_14"]["aggregates"][summary["attrs0_14"]["rtp_variant"]]
    attrs3_rel = summary["attrs3_4"]["relative_vs_baseline_pct"]
    transfer_rel = summary["attrs3_4_transfer"]["relative_vs_baseline_pct"]
    teacher = summary["teacher_target"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# RTP Transfer Evidence Summary\n\n")
        fh.write("This file summarizes completed diagnostics only; it does not run generation.\n\n")
        fh.write("## Key Findings\n\n")
        for item in summary["decision"]["findings"]:
            fh.write(f"- {item}\n")
        fh.write(f"\nConclusion: {summary['decision']['conclusion']}\n\n")
        fh.write(f"Recommendation: {summary['decision']['recommendation']}\n\n")
        fh.write("## Adult attrs0,14 Seed0-4\n\n")
        fh.write("| Metric | RTP vs Base |\n| --- | ---: |\n")
        for metric in ["measured_loss", "mae", "rmse", "avg_tvd", "max_tvd", "max_error"]:
            fh.write(f"| {metric} | {attrs0[f'{metric}_rel_vs_QDTE-Base_pct']:+.2f}% |\n")
        fh.write("\n## Adult attrs3,4 Seed0\n\n")
        fh.write("| Metric | RTP attrs3,4 vs Base |\n| --- | ---: |\n")
        for metric in DISPLAY_METRICS:
            fh.write(f"| {metric} | {attrs3_rel[metric]:+.2f}% |\n")
        fh.write("\n## attrs3,4 Transfer\n\n")
        fh.write("| Component | RTP attrs3,4 vs Base |\n| --- | ---: |\n")
        for metric, value in transfer_rel.items():
            fh.write(f"| {metric} | {value:+.2f}% |\n")
        fh.write("\n## Teacher Target\n\n")
        fh.write("| Metric | Value |\n| --- | ---: |\n")
        for key in ["recomputed_measured_loss", "teacher_rate_mae", "teacher_rate_rmse", "teacher_rate_max_error", "loss_ratio_vs_baseline"]:
            fh.write(f"| {key} | {teacher[key]} |\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    attrs0 = _summarize_attrs0_14(args.attrs0_14_json, "QDTE-Base", "QDTE-RTP-local")
    attrs3 = _summarize_attrs3_4(args.attrs3_4_json, "QDTE-Base", "QDTE-RTP-local-3-4")
    baseline_loss = float(attrs0["aggregates"]["QDTE-Base"]["measured_loss"])
    teacher = _summarize_teacher(args.teacher_json, baseline_loss=baseline_loss)
    transfer = _summarize_transfer(args.attrs3_4_transfer_csv, "baseline", "rtp_local_attrs3_4")
    summary = {
        "attrs0_14": attrs0,
        "attrs3_4": attrs3,
        "attrs3_4_transfer": transfer,
        "teacher_target": teacher,
    }
    summary["decision"] = _derive_decision(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_md is not None:
        _write_markdown(summary, args.output_md)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed RTP transfer-gap diagnostic evidence.")
    parser.add_argument("--attrs0-14-json", required=True, type=Path)
    parser.add_argument("--attrs3-4-json", required=True, type=Path)
    parser.add_argument("--attrs3-4-transfer-csv", required=True, type=Path)
    parser.add_argument("--teacher-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(summary["decision"]["conclusion"])


if __name__ == "__main__":
    main()
