#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


MAIN_PREFIXES = {
    "random_mutation": ("rtype_random",),
    "single_query": ("rtype_single_enter", "rtype_single_exit"),
    "masked_single_query": ("rtype_masked_single",),
    "relaxed_masked_single_query": ("rtype_relaxed_masked_single",),
    "residual_weighted_mutation": ("rtype_residual_weighted",),
    "enumerated_local": ("rtype_enumerated_local",),
    "qdte_mixture": (
        "rtype_single_enter",
        "rtype_single_exit",
        "rtype_masked_single",
        "rtype_enumerated_local",
        "rtype_relaxed_masked_single",
    ),
}

MEAN_SUFFIXES = (
    "mean_full_advantage",
    "mean_target_component",
    "mean_collateral_component",
    "mean_affected_queries",
    "mean_beneficial_queries",
    "mean_harmful_queries",
)


def _sum_existing(df: pd.DataFrame, columns: Iterable[str]) -> float:
    total = 0.0
    for column in columns:
        if column in df:
            total += float(df[column].fillna(0.0).sum())
    return total


def _weighted_existing(df: pd.DataFrame, prefixes: tuple[str, ...], suffix: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for prefix in prefixes:
        value_col = f"{prefix}_{suffix}"
        weight_col = f"{prefix}_generated"
        if value_col not in df or weight_col not in df:
            continue
        values = df[value_col].fillna(0.0)
        weights = df[weight_col].fillna(0.0)
        numerator += float((values * weights).sum())
        denominator += float(weights.sum())
    return numerator / denominator if denominator > 0.0 else 0.0


def _rate_from_counts(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _aggregate_prefixes(df: pd.DataFrame, prefixes: tuple[str, ...]) -> dict[str, float]:
    generated = _sum_existing(df, (f"{prefix}_generated" for prefix in prefixes))
    positive = _sum_existing(df, (f"{prefix}_positive" for prefix in prefixes))
    selected = _sum_existing(df, (f"{prefix}_selected" for prefix in prefixes))
    accepted = _sum_existing(df, (f"{prefix}_accepted" for prefix in prefixes))

    row: dict[str, float] = {
        "main_generated": generated,
        "main_positive_rate": _rate_from_counts(positive, generated),
        "main_selected_rate": _rate_from_counts(selected, generated),
        "main_accepted_rate": _rate_from_counts(accepted, generated),
    }
    for suffix in MEAN_SUFFIXES:
        row[f"main_{suffix}"] = _weighted_existing(df, prefixes, suffix)

    target_positive = 0.0
    target_positive_full_negative = 0.0
    target_positive_collateral_negative = 0.0
    for prefix in prefixes:
        generated_col = f"{prefix}_generated"
        if generated_col not in df:
            continue
        generated_values = df[generated_col].fillna(0.0)
        target_positive_rate = df.get(f"{prefix}_target_positive_rate", pd.Series(0.0, index=df.index)).fillna(0.0)
        full_negative_rate = df.get(
            f"{prefix}_target_positive_full_negative_rate",
            pd.Series(0.0, index=df.index),
        ).fillna(0.0)
        collateral_negative_rate = df.get(
            f"{prefix}_target_positive_collateral_negative_rate",
            pd.Series(0.0, index=df.index),
        ).fillna(0.0)
        prefix_target_positive = generated_values * target_positive_rate
        target_positive += float(prefix_target_positive.sum())
        target_positive_full_negative += float((generated_values * full_negative_rate).sum())
        target_positive_collateral_negative += float((prefix_target_positive * collateral_negative_rate).sum())
    row["main_target_positive_rate"] = _rate_from_counts(target_positive, generated)
    row["main_target_positive_full_negative_rate"] = _rate_from_counts(
        target_positive_full_negative,
        generated,
    )
    row["main_target_positive_collateral_negative_rate"] = _rate_from_counts(
        target_positive_collateral_negative,
        target_positive,
    )

    conflict = 0.0
    for prefix in prefixes:
        generated_col = f"{prefix}_generated"
        conflict_col = f"{prefix}_residual_conflict_rate"
        if generated_col in df and conflict_col in df:
            conflict += float((df[generated_col].fillna(0.0) * df[conflict_col].fillna(0.0)).sum())
    row["main_residual_conflict_rate"] = _rate_from_counts(conflict, generated)
    return row


def _aggregate_overall(df: pd.DataFrame) -> dict[str, float]:
    generated = float(df["candidate_diag_count"].fillna(0.0).sum())
    row = {
        "all_generated": generated,
        "all_positive_rate": float((df["candidate_diag_count"] * df["diag_positive_full_rate"]).sum() / generated),
        "all_accepted_rate": float((df["candidate_diag_count"] * df["diag_accepted_rate"]).sum() / generated),
        "all_mean_full_advantage": float((df["candidate_diag_count"] * df["diag_mean_full_advantage"]).sum() / generated),
        "all_mean_target_component": float(
            (df["candidate_diag_count"] * df["diag_mean_target_component"].fillna(0.0)).sum() / generated
        ),
        "all_mean_collateral_component": float(
            (df["candidate_diag_count"] * df["diag_mean_collateral_component"].fillna(0.0)).sum() / generated
        ),
        "all_residual_conflict_rate": float(
            (df["candidate_diag_count"] * df["diag_residual_conflict_rate"]).sum() / generated
        ),
    }
    return row


def _load_final_metrics(run_dir: Path) -> dict[str, float | int | str]:
    with (run_dir / "metrics_final.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "final_measured_loss": float(metrics.get("final_measured_loss", 0.0)),
        "final_true_query_mae": float(metrics.get("final_true_query_mae", 0.0)),
        "final_true_query_rmse": float(metrics.get("final_true_query_rmse", 0.0)),
        "num_accepted_edits": int(metrics.get("num_accepted_edits", 0)),
        "num_candidates_scored": int(metrics.get("num_candidates_scored", 0)),
    }


def _summarize_variant(run_dir: Path, variant: str, df: pd.DataFrame, phase: str) -> dict[str, float | int | str]:
    prefixes = MAIN_PREFIXES[variant]
    total_generated = float(df["candidate_diag_count"].fillna(0.0).sum())
    random_generated = _sum_existing(df, ("rtype_random_generated",))
    planned_random = float(df.get("diag_planned_random_candidates", pd.Series(0.0, index=df.index)).fillna(0.0).sum())
    fallback_random = float(df.get("diag_fallback_random_candidates", pd.Series(0.0, index=df.index)).fillna(0.0).sum())
    mixture_random = float(df.get("diag_mixture_random_candidates", pd.Series(0.0, index=df.index)).fillna(0.0).sum())
    directed_shortfall = float(
        df.get("diag_directed_candidate_shortfall", pd.Series(0.0, index=df.index)).fillna(0.0).sum()
    )
    row: dict[str, float | int | str] = {
        "variant": variant,
        "phase": phase,
        "iterations": int(len(df)),
        "fallback_random_rate": _rate_from_counts(random_generated, total_generated),
        "planned_random_candidates": planned_random,
        "fallback_random_candidates": fallback_random,
        "mixture_random_candidates": mixture_random,
        "directed_candidate_shortfall": directed_shortfall,
    }
    if phase == "all":
        row.update(_load_final_metrics(run_dir))
    row.update(_aggregate_overall(df))
    row.update(_aggregate_prefixes(df, prefixes))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output", default="outputs/exp_conflictdiag2000")
    parser.add_argument("--phase-size", type=int, default=500)
    parser.add_argument("--output", default="outputs/exp_conflictdiag2000_candidate_diag_summary.tsv")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    variants = args.variants if args.variants is not None else list(MAIN_PREFIXES)
    for variant in variants:
        if variant not in MAIN_PREFIXES:
            raise ValueError(f"Unknown variant for diagnostics summary: {variant}")
        run_dir = Path(f"{args.base_output}_{variant}")
        diagnostics_path = run_dir / "candidate_diagnostics_timeseries.csv"
        if not diagnostics_path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(diagnostics_path)
        df = pd.read_csv(diagnostics_path)
        rows.append(_summarize_variant(run_dir, variant, df, "all"))
        if args.phase_size > 0 and len(df) >= args.phase_size:
            rows.append(_summarize_variant(run_dir, variant, df.head(args.phase_size), f"first{args.phase_size}"))
            rows.append(_summarize_variant(run_dir, variant, df.tail(args.phase_size), f"last{args.phase_size}"))

    summary = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
