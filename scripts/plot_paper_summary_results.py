#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from write_paper_tables import DATASET_LABELS
from write_paper_tables import DATASET_ORDER
from write_paper_tables import HIGHPOWER_GSD_METHOD_ORDER
from write_paper_tables import MAIN_METHOD_ORDER
from write_paper_tables import load_primary_rows


METRICS = [
    ("MAE", "MAE_mean", "MAE_sem"),
    ("RMSE", "RMSE_mean", "RMSE_sem"),
    ("AvgTVD", "AvgTVD_mean", "AvgTVD_sem"),
    ("MaxErr", "MaxErr_mean", "MaxErr_sem"),
    ("MaxTVD", "MaxTVD_mean", "MaxTVD_sem"),
]

METHOD_COLORS = {
    "SAGE": "#0072B2",
    "RAP softmax": "#7F7F7F",
    "Private-GSD GPU": "#AA4499",
    "Private-GSD GPU 1M/full-N": "#882255",
    "Private-PGM AIM": "#D55E00",
    "Private-PGM MST": "#009E73",
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
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_metric(rows: list[dict[str, str]], metric: tuple[str, str, str], methods: list[str], out_dir: Path, formats: list[str]) -> None:
    label, mean_key, sem_key = metric
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(3.4 * len(DATASET_ORDER), 3.0), squeeze=False)
    for ax, dataset in zip(axes[0], DATASET_ORDER):
        values = []
        errors = []
        colors = []
        labels = []
        for method in methods:
            row = lookup[(dataset, method)]
            values.append(_as_float(row, mean_key))
            errors.append(_as_float(row, sem_key))
            colors.append(METHOD_COLORS[method])
            labels.append(method)
        x = list(range(len(methods)))
        ax.bar(x, values, yerr=errors, capsize=2, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_ylabel(f"{label} (lower is better)")
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[method]) for method in methods]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, out_dir, f"{label.lower()}_main_bars", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper summary rows with SEM bars.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gsd-main", choices=["200k", "1m-fulln"], default="1m-fulln")
    parser.add_argument("--metrics", default="MAE,RMSE,AvgTVD,MaxErr,MaxTVD")
    parser.add_argument("--formats", default="pdf,png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_style()
    methods = HIGHPOWER_GSD_METHOD_ORDER if args.gsd_main == "1m-fulln" else MAIN_METHOD_ORDER
    wanted_metrics = {item.strip() for item in args.metrics.split(",") if item.strip()}
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    rows = load_primary_rows(args.results_dir, gsd_main=args.gsd_main)
    for metric in METRICS:
        if metric[0] in wanted_metrics:
            plot_metric(rows, metric, methods, args.out_dir, formats)
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
