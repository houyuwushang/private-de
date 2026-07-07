#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "sage": "SAGE",
    "sage_ordered_gain": "SAGE",
    "sage_ablation": "SAGE ablation",
    "dpmm_aim": "DPMM AIM",
    "private_pgm_aim": "Private-PGM AIM",
    "aim_private_pgm": "Private-PGM AIM",
    "dpmm_mst": "DPMM MST",
    "private_pgm_mst": "Private-PGM MST",
    "mst_private_pgm": "Private-PGM MST",
    "dpmm_privbayes": "DPMM PrivBayes",
    "datasynth_privbayes": "PrivBayes",
    "private_gsd": "Private-GSD",
    "private_gsd_gpu": "Private-GSD GPU",
    "private_gsd_stronger": "Private-GSD stronger",
    "privmrf": "PrivMRF",
    "privmrf_gpu": "PrivMRF GPU",
    "privsyn_unofficial": "PrivSyn unofficial",
    "rap": "RAP",
    "rappp": "RAP++",
}

METHOD_COLORS = {
    "SAGE": "#0072B2",
    "SAGE ablation": "#56B4E9",
    "DPMM AIM": "#E17C05",
    "Private-PGM AIM": "#D55E00",
    "DPMM MST": "#44AA99",
    "Private-PGM MST": "#009E73",
    "DPMM PrivBayes": "#E69F00",
    "PrivBayes": "#E69F00",
    "Private-GSD": "#CC79A7",
    "Private-GSD GPU": "#AA4499",
    "Private-GSD stronger": "#882255",
    "PrivMRF": "#6A3D9A",
    "PrivMRF GPU": "#6A3D9A",
    "PrivSyn unofficial": "#999933",
    "RAP": "#7F7F7F",
    "RAP++": "#7F7F7F",
    "Other": "#4D4D4D",
}

DATASET_LABELS = {
    "adult_sage_strong": "Adult",
    "acs_sage_strong": "ACS",
    "br2000_sage_strong": "BR2000",
    "nltcs_sage_strong": "NLTCS",
}

METRIC_LABELS = {
    "full_true_mae": "MAE",
    "full_true_rmse": "RMSE",
    "full_true_avg_tvd": "AvgTVD",
    "full_true_max_error": "Max error",
    "full_true_max_tvd": "MaxTVD",
    "runtime_seconds": "Runtime (s)",
}

DEFAULT_METRICS = ["full_true_mae", "full_true_rmse", "full_true_avg_tvd"]


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def _method_color(label: str) -> str:
    return METHOD_COLORS.get(label, METHOD_COLORS["Other"])


def _dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(str(dataset), str(dataset))


def _dataset_order(values: list[str]) -> list[str]:
    preferred = ["adult_sage_strong", "acs_sage_strong", "br2000_sage_strong", "nltcs_sage_strong"]
    present = set(values)
    return [dataset for dataset in preferred if dataset in present] + sorted(present - set(preferred))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = df.copy()
    work["method_label"] = work["method"].map(_method_label)
    grouped = (
        work.groupby(["dataset", "rho_total", "method_label"], dropna=False)[metric]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["sem"].fillna(0.0)
    return grouped


def plot_metric_vs_budget(df: pd.DataFrame, metric: str, out_dir: Path, formats: list[str]) -> None:
    agg = _aggregate(df, metric)
    datasets = _dataset_order([str(item) for item in agg["dataset"].dropna().unique().tolist()])
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = agg[agg["dataset"] == dataset]
        for label in sorted(sub["method_label"].unique().tolist()):
            part = sub[sub["method_label"] == label].sort_values("rho_total")
            color = _method_color(label)
            ax.errorbar(
                part["rho_total"],
                part["mean"],
                yerr=part["sem"],
                marker="o",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=color,
                label=label,
            )
        ax.set_title(_dataset_label(dataset))
        ax.set_xlabel(r"$\rho$ total")
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (lower is better)")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, out_dir, f"{metric}_vs_budget", formats)


def plot_metric_at_rho(df: pd.DataFrame, metric: str, rho: float, out_dir: Path, formats: list[str]) -> None:
    agg = _aggregate(df, metric)
    sub = agg[agg["rho_total"].astype(float) == float(rho)].copy()
    if sub.empty:
        return
    datasets = _dataset_order([str(item) for item in sub["dataset"].dropna().unique().tolist()])
    methods = sorted(str(item) for item in sub["method_label"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        part = sub[sub["dataset"] == dataset].set_index("method_label").reindex(methods).reset_index()
        colors = [_method_color(label) for label in part["method_label"].tolist()]
        ax.bar(part["method_label"], part["mean"], yerr=part["sem"], capsize=2, color=colors, edgecolor="white")
        ax.set_title(_dataset_label(dataset))
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (lower is better)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out_dir, f"{metric}_rho{str(float(rho)).replace('.', 'p')}_bars", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw paper-style figures from canonical external result summaries.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--formats", default="pdf,png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_style()
    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise ValueError(f"No result rows in {args.summary_csv}")
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    for metric in metrics:
        if metric not in df.columns:
            raise ValueError(f"Metric {metric!r} not found in {args.summary_csv}")
        plot_metric_vs_budget(df, metric, args.out_dir, formats)
        plot_metric_at_rho(df, metric, args.rho, args.out_dir, formats)
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
