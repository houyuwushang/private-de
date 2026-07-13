#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

try:
    from path_defaults import external_results
except ModuleNotFoundError:
    from scripts.path_defaults import external_results


DEFAULT_PACKAGE = external_results() / "qdte_paper_package_20260711"
DATASET_ORDER = ["Adult", "ACS", "BR2000"]
QDTE_COLOR = "#147D92"
GSD_COLOR = "#C44E52"
IMPROVE_COLOR = "#2A7F62"
REGRESS_COLOR = "#C44E52"
GRID_COLOR = "#D4D8DC"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _save_figure(fig: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_structured_time_to_quality(package_dir: Path) -> list[Path]:
    rows = _read_csv(package_dir / "tables" / "structured_vs_gsd_time_curve.csv")
    endpoints = {
        row["dataset_label"]: row
        for row in _read_csv(package_dir / "tables" / "structured_vs_gsd_endpoint.csv")
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharey=True)
    for ax, dataset in zip(axes, DATASET_ORDER, strict=True):
        selected = sorted(
            (row for row in rows if row["dataset_label"] == dataset),
            key=lambda row: float(row["time_seconds"]),
        )
        if not selected or dataset not in endpoints:
            raise ValueError(f"missing controlled-generator data for {dataset}")
        gsd_loss = float(selected[0]["gsd_final_loss"])
        gsd_runtime = float(selected[0]["gsd_runtime_seconds"])
        x = [float(row["time_seconds"]) / gsd_runtime for row in selected]
        y = [float(row["loss"]) / gsd_loss for row in selected]
        endpoint = endpoints[dataset]
        first_pass = float(endpoint["time_to_first_pass_over_gsd"])

        ax.plot(x, y, color=QDTE_COLOR, linewidth=1.45, label="QDTE-Structured")
        ax.axhline(1.0, color=GSD_COLOR, linewidth=1.2, linestyle="--", label="GSD endpoint")
        ax.axvline(1.0, color="#666666", linewidth=0.9, linestyle=":", label="GSD runtime")
        ax.scatter([first_pass], [1.0], color=QDTE_COLOR, s=22, zorder=4)
        ax.scatter([x[-1]], [y[-1]], color=QDTE_COLOR, s=18, marker="s", zorder=4)
        ax.annotate(
            f"pass {first_pass:.2f}x",
            (first_pass, 1.0),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=7,
        )
        ax.set_title(dataset, fontsize=9, pad=3)
        ax.set_xlabel("wall time / GSD runtime", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 100.0)
        ax.set_xlim(0.0, max(1.05, 1.04 * max(x)))
        ax.grid(axis="y", which="both", color=GRID_COLOR, linewidth=0.55)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("QDTE target loss / GSD final loss", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(top=0.77, bottom=0.23, left=0.08, right=0.99, wspace=0.16)
    return _save_figure(
        fig,
        package_dir / "figures" / "qdte_structured_time_to_quality",
    )


def plot_transfer_gap_decomposition(package_dir: Path) -> list[Path]:
    data = _read_json(package_dir / "tables" / "transfer_gap_diagnostic.json")
    changes = data["transfer_decomposition"]["relative_vs_baseline_pct"]
    labels = ["Target T", "Fit F", "Final E"]
    values = [
        float(changes["T_target_to_true_norm2"]),
        float(changes["F_synthetic_to_target_norm2"]),
        float(changes["E_synthetic_to_true_norm2"]),
    ]
    colors = [IMPROVE_COLOR if value < 0.0 else REGRESS_COLOR for value in values]

    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_ylabel("relative change vs. Standard (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    margin = max(0.12, 0.08 * (max(values) - min(values)))
    ax.set_ylim(min(values) - margin, max(values) + 3.2 * margin)
    for bar, value in zip(bars, values, strict=True):
        offset = 3 if value >= 0.0 else 4
        ax.annotate(
            f"{value:+.2f}%",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111" if value >= 0.0 else "white",
        )
    fig.tight_layout(pad=0.35)
    return _save_figure(
        fig,
        package_dir / "figures" / "qdte_transfer_gap_decomposition",
    )


def build_figures(package_dir: Path) -> list[Path]:
    return [
        *plot_structured_time_to_quality(package_dir),
        *plot_transfer_gap_decomposition(package_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot figures from the QDTE paper package.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in build_figures(args.package_dir):
        print(path)


if __name__ == "__main__":
    main()
