#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.dataio import read_json, write_json


DEFAULT_INPUT = ROOT / "outputs" / "static_ice_wp9_eval_20260715"
DEFAULT_OUTPUT = ROOT / "outputs" / "static_ice_wp9_figures_20260715"
DATASETS = ("nltcs", "acs", "br2000", "adult")
DATASET_LABELS = {
    "nltcs": "NLTCS\n(binary)",
    "acs": "ACS\n(binary)",
    "br2000": "BR2000\n(general cardinality)",
    "adult": "Adult\n(general cardinality)",
}
METRICS = (
    "full_true_mae",
    "full_true_rmse",
    "full_true_avg_tvd",
    "full_true_max_tvd",
    "full_true_max_error",
)
PRIMARY_METRICS = METRICS[:3]
PROTOCOL_ID = "SAGE-QDTE-ICE-WP9-STATIC-CONFIRMATION-20260715-v1"
METHOD_ID = "SAGE-QDTE-Static-ICE-Exact-v1"


def _geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Geometric means require positive finite values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = 4 * 2 * 3 * 2
    if len(rows) != expected:
        raise ValueError(f"WP9 metrics must contain exactly {expected} rows, got {len(rows)}")
    return rows


def build_ratio_rows(
    metrics_rows: list[dict[str, str]],
    gate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if gate_summary.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("WP9 gate summary protocol mismatch")
    if gate_summary.get("method_id") != METHOD_ID:
        raise ValueError("WP9 gate summary method mismatch")
    if gate_summary.get("decision") != "retain_static_ice_as_development_candidate":
        raise ValueError("WP9 figure requires the frozen non-promotion decision")

    by_key: dict[tuple[str, float, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in metrics_rows:
        dataset = str(row["dataset"])
        epsilon = float(row["epsilon"])
        seed = int(row["seed"])
        arm = str(row["arm"])
        key = (dataset, epsilon, seed)
        if arm in by_key[key]:
            raise ValueError(f"Duplicate WP9 arm for {key}: {arm}")
        by_key[key][arm] = row

    expected_keys = {
        (dataset, epsilon, seed)
        for dataset in DATASETS
        for epsilon in (0.1, 0.3)
        for seed in (0, 1, 2)
    }
    if set(by_key) != expected_keys:
        raise ValueError("WP9 metrics do not contain the frozen dataset/epsilon/seed matrix")

    seed_rows: list[dict[str, Any]] = []
    for dataset, epsilon, seed in sorted(by_key):
        arms = by_key[(dataset, epsilon, seed)]
        if set(arms) != {"static_ice_exact", "official_aim"}:
            raise ValueError(f"WP9 cell has unexpected arms: {(dataset, epsilon, seed)}")
        ratios: dict[str, float] = {}
        for metric in METRICS:
            numerator = float(arms["static_ice_exact"][metric])
            denominator = float(arms["official_aim"][metric])
            if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(f"Invalid WP9 metric in {(dataset, epsilon, seed, metric)}")
            ratios[f"{metric}_ratio"] = numerator / denominator
        seed_rows.append(
            {
                "dataset": dataset,
                "epsilon": epsilon,
                "seed": seed,
                "level": "seed",
                "primary_ratio": _geometric_mean(
                    [ratios[f"{metric}_ratio"] for metric in PRIMARY_METRICS]
                ),
                **ratios,
            }
        )

    output = list(seed_rows)
    frozen_cell_ratios = gate_summary.get("cell_primary_ratios", {})
    for dataset in DATASETS:
        for epsilon in (0.1, 0.3):
            selected = [
                row
                for row in seed_rows
                if row["dataset"] == dataset and row["epsilon"] == epsilon
            ]
            aggregate: dict[str, Any] = {
                "dataset": dataset,
                "epsilon": epsilon,
                "seed": "aggregate",
                "level": "aggregate",
            }
            for key in ("primary_ratio", *(f"{metric}_ratio" for metric in METRICS)):
                values = [float(row[key]) for row in selected]
                aggregate[key] = _geometric_mean(values)
                aggregate[f"{key}_min"] = min(values)
                aggregate[f"{key}_max"] = max(values)
            frozen = float(frozen_cell_ratios[f"{dataset}|{epsilon}"])
            if not math.isclose(
                float(aggregate["primary_ratio"]),
                frozen,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise RuntimeError(
                    f"Recomputed primary ratio disagrees with frozen gate for {dataset}|{epsilon}"
                )
            output.append(aggregate)
    return output


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_gate(ratio_rows: list[dict[str, Any]]) -> plt.Figure:
    _style()
    aggregate = [row for row in ratio_rows if row["level"] == "aggregate"]
    seeds = [row for row in ratio_rows if row["level"] == "seed"]
    all_values = [
        float(row[key])
        for row in seeds
        for key in (
            "primary_ratio",
            "full_true_max_tvd_ratio",
            "full_true_max_error_ratio",
        )
    ]
    lower = max(0.25, min(all_values) * 0.82)
    upper = max(1.25, max(all_values) * 1.18)
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 3.7), sharex=True, sharey=True)
    for column, dataset in enumerate(DATASETS):
        top = axes[0, column]
        bottom = axes[1, column]
        for axis in (top, bottom):
            axis.axhspan(lower, 1.0, color="#DCEFE5", alpha=0.45, linewidth=0)
            axis.axhspan(1.0, upper, color="#F5DFD8", alpha=0.38, linewidth=0)
            axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(0.085, 0.35)
            axis.set_ylim(lower, upper)
            axis.set_xticks([0.1, 0.3], labels=["0.1", "0.3"])
            axis.xaxis.set_minor_formatter(NullFormatter())

        selected_aggregate = sorted(
            [row for row in aggregate if row["dataset"] == dataset],
            key=lambda row: float(row["epsilon"]),
        )
        selected_seeds = [row for row in seeds if row["dataset"] == dataset]
        x = np.asarray([float(row["epsilon"]) for row in selected_aggregate])
        primary = np.asarray([float(row["primary_ratio"]) for row in selected_aggregate])
        primary_low = primary - np.asarray(
            [float(row["primary_ratio_min"]) for row in selected_aggregate]
        )
        primary_high = np.asarray(
            [float(row["primary_ratio_max"]) for row in selected_aggregate]
        ) - primary
        top.errorbar(
            x,
            primary,
            yerr=np.vstack([primary_low, primary_high]),
            color="#0072B2",
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=2,
            label="Primary GM",
        )
        top.scatter(
            [float(row["epsilon"]) for row in selected_seeds],
            [float(row["primary_ratio"]) for row in selected_seeds],
            color="#0072B2",
            s=8,
            alpha=0.3,
            linewidths=0,
        )

        for key, label, color, marker in (
            ("full_true_max_tvd_ratio", "MaxTVD", "#D55E00", "s"),
            ("full_true_max_error_ratio", "MaxError", "#009E73", "^"),
        ):
            values = np.asarray([float(row[key]) for row in selected_aggregate])
            low = values - np.asarray([float(row[f"{key}_min"]) for row in selected_aggregate])
            high = np.asarray([float(row[f"{key}_max"]) for row in selected_aggregate]) - values
            bottom.errorbar(
                x,
                values,
                yerr=np.vstack([low, high]),
                color=color,
                marker=marker,
                markersize=3.7,
                linewidth=1.15,
                capsize=2,
                label=label,
            )
            bottom.scatter(
                [float(row["epsilon"]) for row in selected_seeds],
                [float(row[key]) for row in selected_seeds],
                color=color,
                s=7,
                alpha=0.25,
                linewidths=0,
            )
        top.set_title(DATASET_LABELS[dataset])
        bottom.set_xlabel(r"Privacy budget $\epsilon$")

    axes[0, 0].set_ylabel("Static-ICE / AIM\nprimary error")
    axes[1, 0].set_ylabel("Static-ICE / AIM\ntail error")
    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    tail_handles, tail_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        [*top_handles, *tail_handles],
        [*top_labels, *tail_labels],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.text(0.995, 0.012, "Lower is better; shaded green is below AIM", ha="right", fontsize=6.5)
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.085, right=0.995, wspace=0.16, hspace=0.16)
    return fig


def _write_ratio_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(input_dir: Path, output_dir: Path, *, force: bool = False) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    metrics_path = input_dir / "metrics.csv"
    gate_path = input_dir / "gate_summary.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"WP9 figure output must be new or use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = read_json(gate_path)
    ratios = build_ratio_rows(_read_metrics(metrics_path), gate)
    ratio_path = output_dir / "wp9_static_ice_vs_aim_ratios.csv"
    _write_ratio_csv(ratios, ratio_path)
    figure = plot_gate(ratios)
    pdf_path = output_dir / "wp9_static_ice_vs_aim_gate.pdf"
    png_path = output_dir / "wp9_static_ice_vs_aim_gate.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(figure)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "artifact_role": "descriptive_frozen_wp9_gate_figure",
        "source_gate_decision": gate["decision"],
        "descriptive_only": True,
        "method_promotion_authorized": False,
        "inputs": {
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "gate": {"path": str(gate_path), "sha256": _sha256(gate_path)},
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (ratio_path, pdf_path, png_path)
        },
    }
    write_json(manifest, output_dir / "figure_manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the frozen Static-ICE WP9 promotion gate.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.input_dir, args.output_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
