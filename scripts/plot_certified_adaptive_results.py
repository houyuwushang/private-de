#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from path_defaults import legacy_paper_package_dir, sage_paper_dir
except ModuleNotFoundError:
    from scripts.path_defaults import legacy_paper_package_dir, sage_paper_dir

SAGE_PAPER_DIR = sage_paper_dir()
DEFAULT_PACKAGE_DIR = legacy_paper_package_dir()

DATASET_LABELS = {
    "adult": "Adult",
    "acs": "ACS",
}

SCHEME_LABELS = {
    "voi_sageordergain_harmonic_qproject": "SAGE-Select",
    "aim_l1_floor": "AIM-style",
}

SCHEME_COLORS = {
    "SAGE-Select": "#0072B2",
    "AIM-style": "#D55E00",
}

METRICS = [
    ("MAE", "full_true_mae"),
    ("RMSE", "full_true_rmse"),
    ("AvgTVD", "full_true_avg_tvd"),
    ("MaxErr", "full_true_max_error"),
    ("MaxTVD", "full_true_max_tvd"),
]

PRIMARY_METRICS = METRICS[:3]

IMPROVEMENT_COLUMNS = {
    "full_true_mae": (
        "full_true_mae_mean_rel_improvement_pct",
        "full_true_mae_std_rel_improvement_pct",
    ),
    "full_true_rmse": (
        "full_true_rmse_mean_rel_improvement_pct",
        "full_true_rmse_std_rel_improvement_pct",
    ),
    "full_true_avg_tvd": (
        "full_true_avg_tvd_mean_rel_improvement_pct",
        "full_true_avg_tvd_std_rel_improvement_pct",
    ),
    "full_true_max_error": (
        "full_true_max_error_mean_rel_improvement_pct",
        "full_true_max_error_std_rel_improvement_pct",
    ),
    "full_true_max_tvd": (
        "full_true_max_tvd_mean_rel_improvement_pct",
        "full_true_max_tvd_std_rel_improvement_pct",
    ),
}


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
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _dataset_order(values: pd.Series) -> list[str]:
    preferred = ["adult", "acs"]
    present = set(str(value) for value in values.dropna().unique())
    return [item for item in preferred if item in present] + sorted(present - set(preferred))


def plot_primary_improvement(paired: pd.DataFrame, out_dir: Path, formats: list[str]) -> list[Path]:
    datasets = _dataset_order(paired["dataset"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.3), squeeze=False)
    markers = ["o", "s", "^"]
    for ax, dataset in zip(axes[0], datasets):
        sub = paired[paired["dataset"] == dataset].sort_values("rho_total")
        for marker, (label, metric) in zip(markers, PRIMARY_METRICS):
            mean_col, std_col = IMPROVEMENT_COLUMNS[metric]
            ax.errorbar(
                sub["rho_total"],
                sub[mean_col],
                yerr=sub[std_col],
                marker=marker,
                linewidth=1.8,
                markersize=4,
                capsize=2,
                label=label,
            )
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel(r"$\rho$ total")
        ax.set_ylabel("SAGE improvement over AIM (%)")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return _save(fig, out_dir, "certified_adaptive_primary_improvement", formats)


def plot_all_metric_heatmap(paired: pd.DataFrame, out_dir: Path, formats: list[str]) -> list[Path]:
    datasets = _dataset_order(paired["dataset"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8 * len(datasets), 3.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = paired[paired["dataset"] == dataset].sort_values("rho_total")
        matrix = []
        for _, metric in METRICS:
            mean_col, _ = IMPROVEMENT_COLUMNS[metric]
            matrix.append(sub[mean_col].to_numpy(dtype=float))
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-8.0, vmax=8.0)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([f"{rho:g}" for rho in sub["rho_total"]])
        ax.set_yticks(range(len(METRICS)))
        ax.set_yticklabels([label for label, _ in METRICS])
        ax.set_xlabel(r"$\rho$ total")
        for row_idx, (_, metric) in enumerate(METRICS):
            mean_col, _ = IMPROVEMENT_COLUMNS[metric]
            for col_idx, value in enumerate(sub[mean_col].to_numpy(dtype=float)):
                color = "white" if abs(value) > 4.5 else "#222222"
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86)
    cbar.set_label("SAGE improvement over AIM (%)")
    return _save(fig, out_dir, "certified_adaptive_all_metric_improvement", formats)


def plot_primary_error_curves(aggregate: pd.DataFrame, out_dir: Path, formats: list[str]) -> list[Path]:
    datasets = _dataset_order(aggregate["dataset"])
    fig, axes = plt.subplots(
        len(PRIMARY_METRICS),
        len(datasets),
        figsize=(4.2 * len(datasets), 2.65 * len(PRIMARY_METRICS)),
        squeeze=False,
        sharex="col",
    )
    for row_idx, (metric_label, metric) in enumerate(PRIMARY_METRICS):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            sub = aggregate[aggregate["dataset"] == dataset].sort_values("rho_total")
            for scheme in ["voi_sageordergain_harmonic_qproject", "aim_l1_floor"]:
                part = sub[sub["scheme"] == scheme]
                label = SCHEME_LABELS.get(scheme, scheme)
                ax.errorbar(
                    part["rho_total"],
                    part[mean_col],
                    yerr=part[std_col],
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    capsize=2,
                    color=SCHEME_COLORS.get(label, "#4D4D4D"),
                    label=label,
                )
            if row_idx == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset))
            if row_idx == len(PRIMARY_METRICS) - 1:
                ax.set_xlabel(r"$\rho$ total")
            if col_idx == 0:
                ax.set_ylabel(f"{metric_label}\n(lower is better)")
            ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_dir, "certified_adaptive_primary_error_curves", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot certified adaptive SAGE-Select results.")
    parser.add_argument("--paired-csv", type=Path, default=SAGE_PAPER_DIR / "certified_budget_curve_paired.csv")
    parser.add_argument("--aggregate-csv", type=Path, default=SAGE_PAPER_DIR / "certified_budget_curve_aggregate.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PACKAGE_DIR / "figures")
    parser.add_argument("--formats", default="pdf,png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_style()
    paired = pd.read_csv(args.paired_csv)
    aggregate = pd.read_csv(args.aggregate_csv)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    outputs = []
    outputs += plot_primary_improvement(paired, args.out_dir, formats)
    outputs += plot_all_metric_heatmap(paired, args.out_dir, formats)
    outputs += plot_primary_error_curves(aggregate, args.out_dir, formats)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
