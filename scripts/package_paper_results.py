from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

try:
    from path_defaults import external_results, sage_paper_dir
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, sage_paper_dir

EXTERNAL_RESULTS = external_results()
SAGE_PAPER_DIR = sage_paper_dir()

DATASET_LABELS = {
    "adult_sage_strong": "Adult",
    "acs_sage_strong": "ACS",
    "br2000_sage_strong": "BR2000",
    "nltcs_sage_strong": "NLTCS",
}

METHOD_ORDER = [
    "SAGE",
    "RAP softmax",
    "Private-GSD GPU 1M/full-N",
    "Private-PGM AIM",
    "Private-PGM MST",
]

SENSITIVITY_METHOD_ORDER = [
    "SAGE",
    "Private-GSD GPU 1M/full-N",
    "Private-PGM AIM",
    "Private-PGM MST",
]

METHOD_COLORS = {
    "SAGE": "#0072B2",
    "RAP softmax": "#E69F00",
    "Private-GSD": "#CC79A7",
    "Private-GSD GPU": "#AA4499",
    "Private-GSD GPU 1M/full-N": "#882255",
    "Private-PGM AIM": "#D55E00",
    "Private-PGM MST": "#009E73",
}

TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

METRICS = [
    ("MAE", "MAE_mean", "MAE_sem"),
    ("RMSE", "RMSE_mean", "RMSE_sem"),
    ("AvgTVD", "AvgTVD_mean", "AvgTVD_sem"),
    ("MaxErr", "MaxErr_mean", "MaxErr_sem"),
    ("MaxTVD", "MaxTVD_mean", "MaxTVD_sem"),
]

APPENDIX_ARTIFACTS = [
    "datasynth_privbayes_adult_strong_seed0_20260706.csv",
    "datasynth_privbayes_adult_strong_seed0_20260706.md",
    "privsyn_unofficial_adult_strong_seed0_20260706.csv",
    "privsyn_unofficial_adult_strong_seed0_20260706.md",
    "privmrf_gpu_adult_calibration_seed0_20260706.csv",
    "privmrf_gpu_adult_calibration_seed0_20260706.md",
    "privmrf_official_full_tvd_epsgrid_m300_20260706.csv",
    "privmrf_official_full_tvd_epsgrid_m300_20260706.md",
    "dpmm_privbayes_adult_calibration_seed0_20260706.csv",
    "dpmm_privbayes_adult_calibration_seed0_20260706.md",
    "rap_adult_sage_strong_calibration_seed0_20260706.csv",
    "rap_adult_sage_strong_calibration_seed0_20260706.md",
    "gem_adult_sage_strong_calibration_seed0_20260706.csv",
    "gem_adult_sage_strong_calibration_seed0_20260706.md",
    "gem_row_realization_diagnostics_adult_sage_strong_20260706.csv",
    "gem_row_realization_diagnostics_adult_sage_strong_20260706.md",
    "gem_remap_all4_seed0_20260707.csv",
    "gem_remap_all4_seed0_20260707.md",
    "baseline_gpu_provenance_20260707.csv",
    "baseline_gpu_provenance_20260707.md",
    "baseline_admission_audit_20260706.csv",
    "baseline_admission_audit_20260706.md",
    "original_protocol_baseline_audit_20260707.csv",
    "original_protocol_baseline_audit_20260707.md",
    "primary_run_evidence_audit_20260707.csv",
    "primary_run_evidence_audit_20260707.md",
    "rappp_marginal_pathcheck_20260706.csv",
    "rappp_marginal_pathcheck_20260706.md",
    "rappp_official_paper_grid_seed0to4_20260707.csv",
    "rappp_official_paper_grid_seed0to4_20260707.md",
    "rappp_official_paper_grid_seed0to4_20260707_by_state.csv",
    "rappp_official_paper_grid_seed0to4_20260707_by_target.csv",
    "rappp_official_paper_grid_seed0to4_20260707_overall.csv",
    "private_gsd_stronger_gpu_audit_seed0_20260706.csv",
    "private_gsd_stronger_gpu_audit_seed0_20260706.md",
]

RHO_SWEEP_ARTIFACTS = [
    "gpu_rho_sweep_seed0_all4_20260706.csv",
    "gpu_rho_sweep_seed0_all4_20260706.md",
]

PRIMARY_SEED0TO4_ARTIFACTS = [
    "sage_all4_seed0to4_rho1_20260706.csv",
    "sage_all4_seed0to4_rho1_20260706.md",
    "sage_all4_seed0to4_summary_rho1_20260706.csv",
    "sage_all4_seed0to4_summary_rho1_20260706.md",
    "private_gsd_gpu_1m_fulln_all4_seed0to4_20260706.csv",
    "private_gsd_gpu_1m_fulln_all4_seed0to4_20260706.md",
    "private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
    "private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.md",
    "private_pgm_aim_all4_seed0to4_rho1_20260706.csv",
    "private_pgm_aim_all4_seed0to4_rho1_20260706.md",
    "private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv",
    "private_pgm_aim_all4_seed0to4_summary_rho1_20260706.md",
    "private_pgm_mst_all4_seed0to4_rho1_20260706.csv",
    "private_pgm_mst_all4_seed0to4_rho1_20260706.md",
    "private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv",
    "private_pgm_mst_all4_seed0to4_summary_rho1_20260706.md",
    "rap_sage_strong_stress_all4_seed0to4_rho1_20260706.csv",
    "rap_sage_strong_stress_all4_seed0to4_rho1_20260706.md",
    "rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
    "sage_vs_private_gsd_1m_fulln_seed0to4_20260706.csv",
    "sage_vs_private_gsd_1m_fulln_seed0to4_20260706.md",
    "sage_vs_private_pgm_aim_seed0to4_20260706.csv",
    "sage_vs_private_pgm_aim_seed0to4_20260706.md",
    "sage_vs_private_pgm_mst_seed0to4_20260706.csv",
    "sage_vs_private_pgm_mst_seed0to4_20260706.md",
    "sage_vs_rap_softmax_seed0to4_20260706.csv",
    "sage_vs_rap_softmax_seed0to4_20260706.md",
]

RHO_SWEEP_FIGURES_DIR = EXTERNAL_RESULTS / "figures_gpu_rho_sweep_seed0_all4_20260706"

SAGE_ABLATION_ARTIFACTS = [
    "sage_ablation_seed0_rho1_20260706.csv",
    "sage_ablation_seed0_rho1_20260706.md",
    "sage_ablation_unweighted_multiseed_rho1_20260706.csv",
    "sage_ablation_unweighted_multiseed_rho1_20260706.md",
    "sage_ablation_unweighted_multiseed_summary_rho1_20260706.csv",
    "sage_ablation_low_order_multiseed_rho1_20260706.csv",
    "sage_ablation_low_order_multiseed_rho1_20260706.md",
    "sage_ablation_low_order_multiseed_summary_rho1_20260706.csv",
    "sage_ablation_no_projection_multiseed_rho1_20260706.csv",
    "sage_ablation_no_projection_multiseed_rho1_20260706.md",
    "sage_ablation_no_projection_multiseed_summary_rho1_20260706.csv",
]

CERTIFIED_ADAPTIVE_ARTIFACTS = [
    "certified_budget_curve_raw.csv",
    "certified_budget_curve_aggregate.csv",
    "certified_budget_curve_paired.csv",
    "certified_budget_curve_summary.md",
    "certified_extra_dataset_raw.csv",
    "certified_extra_dataset_aggregate.csv",
    "certified_extra_dataset_paired.csv",
    "certified_extra_dataset_summary.md",
    "certified_r50_rho1_four_dataset_raw.csv",
    "certified_r50_rho1_four_dataset_aggregate.csv",
    "certified_r50_rho1_four_dataset_paired.csv",
    "certified_r50_rho1_four_dataset_summary.md",
]

ABLATION_COLORS = {
    "No projection": "#56B4E9",
    "Unweighted objective": "#E69F00",
    "Low-order workload": "#009E73",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


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
            "grid.alpha": 0.8,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _dataset_order(rows: Iterable[dict[str, str]]) -> list[str]:
    present = {row["dataset"] for row in rows}
    preferred = ["adult_sage_strong", "acs_sage_strong", "br2000_sage_strong", "nltcs_sage_strong"]
    return [dataset for dataset in preferred if dataset in present] + sorted(present - set(preferred))


def _lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["dataset"], row["method"]): row for row in rows}


def _primary_rows(main_csv: Path) -> list[dict[str, str]]:
    rows = _read_rows(main_csv)
    rows.extend(_read_rows(EXTERNAL_RESULTS / "private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv"))
    rows.extend(_read_rows(EXTERNAL_RESULTS / "private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv"))
    rows.extend(_read_rows(EXTERNAL_RESULTS / "private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv"))
    rap_path = EXTERNAL_RESULTS / "rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv"
    if rap_path.exists():
        for rap in _read_rows(rap_path):
            rows.append(
                {
                    "dataset": rap["dataset"],
                    "method": "RAP softmax",
                    "method_slug": "rap_softmax",
                    "n": rap["seed_count"],
                    "MAE_mean": rap["full_true_mae_mean"],
                    "MAE_sem": rap["full_true_mae_sem"],
                    "RMSE_mean": rap["full_true_rmse_mean"],
                    "RMSE_sem": rap["full_true_rmse_sem"],
                    "AvgTVD_mean": rap["full_true_avg_tvd_mean"],
                    "AvgTVD_sem": rap["full_true_avg_tvd_sem"],
                    "MaxErr_mean": rap["full_true_max_error_mean"],
                    "MaxErr_sem": rap["full_true_max_error_sem"],
                    "MaxTVD_mean": rap["full_true_max_tvd_mean"],
                    "MaxTVD_sem": rap["full_true_max_tvd_sem"],
                    "runtime_s_mean": rap["runtime_seconds_mean"],
                    "runtime_s_sem": rap["runtime_seconds_sem"],
                }
            )
    return rows


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in ["pdf", "png"]:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_metric_grid(rows: list[dict[str, str]], out_dir: Path, methods: list[str], stem: str) -> list[Path]:
    lookup = _lookup(rows)
    datasets = _dataset_order(rows)
    x = list(range(len(datasets)))
    width = 0.16
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), squeeze=False)
    axes_flat = axes.flatten()
    plotted_metrics = METRICS + [("Runtime", "runtime_s_mean", "runtime_s_sem")]

    for ax, (metric_label, mean_key, sem_key) in zip(axes_flat, plotted_metrics):
        for idx, method in enumerate(methods):
            offsets = [pos + (idx - (len(methods) - 1) / 2) * width for pos in x]
            values = []
            errors = []
            for dataset in datasets:
                row = lookup.get((dataset, method))
                values.append(_float(row, mean_key) if row else float("nan"))
                errors.append(_float(row, sem_key) if row else 0.0)
            ax.bar(
                offsets,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                color=METHOD_COLORS.get(method, "#4D4D4D"),
                edgecolor="white",
                linewidth=0.5,
                label=method,
            )
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(dataset, dataset) for dataset in datasets], rotation=20)
        ax.set_yscale("log")
        ax.set_ylabel("lower is better")
        ax.grid(axis="x", visible=False)
    for ax in axes_flat[len(plotted_metrics) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_dir, stem)


def plot_relative_to_sage(rows: list[dict[str, str]], out_dir: Path, methods: list[str]) -> list[Path]:
    lookup = _lookup(rows)
    datasets = _dataset_order(rows)
    metric_subset = METRICS[:3]
    x_labels = [f"{DATASET_LABELS.get(dataset, dataset)}\n{metric}" for dataset in datasets for metric, _, _ in metric_subset]
    x = list(range(len(x_labels)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    plot_methods = [method for method in methods if method != "SAGE"]
    for idx, method in enumerate(plot_methods):
        offsets = [pos + (idx - (len(plot_methods) - 1) / 2) * width for pos in x]
        values = []
        for dataset in datasets:
            sage = lookup[(dataset, "SAGE")]
            row = lookup.get((dataset, method))
            for _, mean_key, _ in metric_subset:
                if row is None:
                    values.append(float("nan"))
                    continue
                denom = _float(sage, mean_key)
                values.append(_float(row, mean_key) / denom if denom > 0 else float("nan"))
        ax.bar(
            offsets,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#4D4D4D"),
            edgecolor="white",
            linewidth=0.5,
            label=method,
        )
    ax.axhline(1.0, color="#222222", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Error / SAGE")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=len(plot_methods), bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    return _save(fig, out_dir, "relative_to_sage_primary_metrics")


def plot_private_gsd_fulln_sensitivity(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    sensitivity_path = EXTERNAL_RESULTS / "private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv"
    if not sensitivity_path.exists():
        return []
    sensitivity_rows = _read_rows(sensitivity_path)
    sage_lookup = {
        row["dataset"]: row
        for row in rows
        if row.get("method") == "SAGE"
    }
    gsd_lookup = {row["dataset"]: row for row in sensitivity_rows}
    datasets = _dataset_order(sensitivity_rows)
    ratio_metrics = [
        ("MAE", "MAE_mean", "MAE_mean"),
        ("RMSE", "RMSE_mean", "RMSE_mean"),
        ("AvgTVD", "AvgTVD_mean", "AvgTVD_mean"),
        ("MaxErr", "MaxErr_mean", "MaxErr_mean"),
        ("MaxTVD", "MaxTVD_mean", "MaxTVD_mean"),
    ]
    x_labels = [f"{DATASET_LABELS.get(dataset, dataset)}\n{metric}" for dataset in datasets for metric, _, _ in ratio_metrics]
    values = []
    colors = []
    for dataset in datasets:
        sage = sage_lookup[dataset]
        gsd = gsd_lookup[dataset]
        for _, sage_key, gsd_key in ratio_metrics:
            ratio = _float(gsd, gsd_key) / _float(sage, sage_key)
            values.append(ratio)
            colors.append("#AA4499" if ratio >= 1.0 else "#009E73")
    fig, ax = plt.subplots(figsize=(12.5, 3.7))
    ax.bar(
        list(range(len(values))),
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(1.0, color="#222222", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Private-GSD 1M/full-N / SAGE")
    ax.set_xticks(list(range(len(values))))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.grid(axis="x", visible=False)
    ax.text(
        0.01,
        0.95,
        "Below 1: Private-GSD lower",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    return _save(fig, out_dir, "private_gsd_1m_fulln_sensitivity_ratios")


def plot_summary_metric_bars(rows: list[dict[str, str]], out_dir: Path, methods: list[str]) -> list[Path]:
    lookup = _lookup(rows)
    datasets = _dataset_order(rows)
    outputs: list[Path] = []
    for metric_label, mean_key, sem_key in METRICS:
        fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.0), squeeze=False)
        for ax, dataset in zip(axes[0], datasets):
            values = []
            errors = []
            colors = []
            for method in methods:
                row = lookup[(dataset, method)]
                values.append(_float(row, mean_key))
                errors.append(_float(row, sem_key))
                colors.append(METHOD_COLORS.get(method, "#4D4D4D"))
            x = list(range(len(methods)))
            ax.bar(x, values, yerr=errors, capsize=2, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_title(DATASET_LABELS.get(dataset, dataset))
            ax.set_ylabel(f"{metric_label} (lower is better)")
            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.grid(axis="y")
            ax.grid(axis="x", visible=False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS.get(method, "#4D4D4D")) for method in methods]
        fig.legend(handles, methods, loc="upper center", ncol=min(len(methods), 5), bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout()
        outputs += _save(fig, out_dir, f"{metric_label.lower()}_main_bars")
    return outputs


def plot_sage_ablation_ratios(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    datasets = _dataset_order(rows)
    variants = ["No projection", "Unweighted objective", "Low-order workload"]
    metrics = [
        ("MAE", "full_true_mae_ratio_vs_full"),
        ("RMSE", "full_true_rmse_ratio_vs_full"),
        ("AvgTVD", "full_true_avg_tvd_ratio_vs_full"),
        ("MaxErr", "full_true_max_error_ratio_vs_full"),
        ("MaxTVD", "full_true_max_tvd_ratio_vs_full"),
    ]
    lookup = {(row["dataset"], row["variant_label"]): row for row in rows}
    x_labels = [f"{DATASET_LABELS.get(dataset, dataset)}\n{metric}" for dataset in datasets for metric, _ in metrics]
    x = list(range(len(x_labels)))
    width = 0.22
    fig, ax = plt.subplots(figsize=(13.0, 3.8))
    for idx, variant in enumerate(variants):
        offsets = [pos + (idx - (len(variants) - 1) / 2) * width for pos in x]
        values = []
        for dataset in datasets:
            row = lookup.get((dataset, variant))
            for _, ratio_key in metrics:
                values.append(_float(row, ratio_key) if row else float("nan"))
        ax.bar(
            offsets,
            values,
            width=width,
            color=ABLATION_COLORS.get(variant, "#4D4D4D"),
            edgecolor="white",
            linewidth=0.5,
            label=variant,
        )
    ax.axhline(1.0, color="#222222", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Ablation error / SAGE")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=len(variants), bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    return _save(fig, out_dir, "sage_ablation_ratios")


def plot_sage_variant_multiseed(
    rows: list[dict[str, str]],
    out_dir: Path,
    variant: str,
    variant_label: str,
    figure_stem: str,
) -> list[Path]:
    datasets = _dataset_order(rows)
    metrics = [
        ("MAE", "full_true_mae_ratio_vs_full_mean", "full_true_mae_ratio_vs_full_sem"),
        ("RMSE", "full_true_rmse_ratio_vs_full_mean", "full_true_rmse_ratio_vs_full_sem"),
        ("AvgTVD", "full_true_avg_tvd_ratio_vs_full_mean", "full_true_avg_tvd_ratio_vs_full_sem"),
        ("MaxErr", "full_true_max_error_ratio_vs_full_mean", "full_true_max_error_ratio_vs_full_sem"),
        ("MaxTVD", "full_true_max_tvd_ratio_vs_full_mean", "full_true_max_tvd_ratio_vs_full_sem"),
    ]
    lookup = {
        row["dataset"]: row
        for row in rows
        if row.get("variant") == variant or row.get("variant_label") == variant_label
    }
    x_labels = [f"{DATASET_LABELS.get(dataset, dataset)}\n{metric}" for dataset in datasets for metric, _, _ in metrics]
    values = []
    errors = []
    for dataset in datasets:
        row = lookup.get(dataset)
        for _, mean_key, sem_key in metrics:
            values.append(_float(row, mean_key) if row else float("nan"))
            errors.append(_float(row, sem_key) if row else 0.0)
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    ax.bar(
        list(range(len(values))),
        values,
        yerr=errors,
        capsize=2,
        color=ABLATION_COLORS.get(variant_label, "#4D4D4D"),
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(1.0, color="#222222", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel(f"{variant_label} / SAGE")
    ax.set_xticks(list(range(len(values))))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, figure_stem)


def _copy_artifact(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        dst.write_text(_sanitize_text(src.read_text()), encoding="utf-8")
    else:
        shutil.copy2(src, dst)
    return dst


def _path_replacements() -> list[tuple[str, str]]:
    roots = {
        str(EXTERNAL_RESULTS.parent),
        str(SAGE_PAPER_DIR.parent),
    }
    return [(root, "$SAGE_BASELINE_ROOT") for root in sorted(roots, key=len, reverse=True)]


def _sanitize_text(text: str) -> str:
    for source, replacement in _path_replacements():
        text = text.replace(source, replacement)
    return text


def _display_path(path: Path, package_dir: Path | None = None) -> str:
    if package_dir is not None:
        try:
            return str(path.relative_to(package_dir))
        except ValueError:
            pass
    return _sanitize_text(str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_filelist_and_hashes(out_dir: Path) -> tuple[Path, Path]:
    filelist = out_dir / "ARTIFACT_FILELIST_20260706.txt"
    sha_path = out_dir / "ARTIFACT_SHA256SUMS_20260706.txt"
    files = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path.name not in {filelist.name, sha_path.name}
    )
    filelist.write_text("\n".join(str(path.relative_to(out_dir)) for path in files) + "\n")
    sha_path.write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(out_dir)}" for path in files) + "\n"
    )
    return filelist, sha_path


def _load_write_paper_tables_module():
    module_path = Path(__file__).resolve().parent / "write_paper_tables.py"
    spec = importlib.util.spec_from_file_location("write_paper_tables_for_package", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_paper_tables_draft(results_dir: Path, tables_dir: Path) -> Path:
    module = _load_write_paper_tables_module()
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "paper_tables_draft.tex"
    table_path.write_text(module.build_latex(results_dir, gsd_main="1m-fulln"), encoding="utf-8")
    return table_path


TRACEABILITY_IDS = [
    "primary-table-coverage",
    "sage-vs-aim",
    "sage-vs-mst",
    "sage-vs-rap",
    "sage-vs-gsd",
    "gpu-provenance",
    "baseline-tiering",
    "certified-selector-boundary",
    "sage-ablation-coverage",
]


def _package_csv(package_dir: Path, relpath: str) -> list[dict[str, str]]:
    return _read_rows(package_dir / relpath)


def _artifact_status(package_dir: Path, relpaths: list[str]) -> tuple[bool, str]:
    missing = [relpath for relpath in relpaths if not (package_dir / relpath).exists()]
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, "all evidence artifacts present"


def _sum_col(rows: list[dict[str, str]], col: str) -> int:
    return sum(int(row[col]) for row in rows)


def _bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _trace_row(
    rows: list[dict[str, str]],
    claim_id: str,
    claim: str,
    scope: str,
    evidence_artifacts: list[str],
    verifier: str,
    ok: bool,
    detail: str,
) -> None:
    rows.append(
        {
            "claim_id": claim_id,
            "claim": claim,
            "scope": scope,
            "evidence_artifacts": "; ".join(evidence_artifacts),
            "verifier_or_gate": verifier,
            "status": "pass" if ok else "fail",
            "machine_check": detail,
        }
    )


def _check_summary_coverage(package_dir: Path) -> tuple[bool, str]:
    specs = [
        ("tables/sage_all4_seed0to4_summary_rho1_20260706.csv", "SAGE", "n"),
        (
            "tables/private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
            "Private-GSD GPU 1M/full-N",
            "n",
        ),
        ("tables/private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv", "Private-PGM AIM", "n"),
        ("tables/private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv", "Private-PGM MST", "n"),
        (
            "tables/rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
            "rap_softmax",
            "seed_count",
        ),
    ]
    expected_datasets = set(DATASET_LABELS)
    details = []
    for relpath, method, n_col in specs:
        rows = _package_csv(package_dir, relpath)
        datasets = {row["dataset"] for row in rows}
        counts = [int(row[n_col]) for row in rows]
        if datasets != expected_datasets or counts != [5, 5, 5, 5]:
            return False, f"{relpath}: datasets={sorted(datasets)} counts={counts}"
        details.append(f"{method}:4 datasets x 5 seeds")

    table = (package_dir / "tables/paper_tables_draft.tex").read_text()
    label_pos = table.find(r"\label{tab:main-results}")
    end_pos = table.find(r"\end{table*}", label_pos)
    if label_pos < 0 or end_pos < 0:
        return False, "paper table main-results segment is missing or malformed"
    main_segment = table[label_pos:end_pos]
    for method in METHOD_ORDER:
        if main_segment.count(f"& {method} &") != len(DATASET_LABELS):
            return False, f"paper table does not contain four rows for {method}"
    return True, "; ".join(details)


def _check_primary_run_evidence(package_dir: Path) -> tuple[bool, str]:
    rows = _package_csv(package_dir, "appendix/primary_run_evidence_audit_20260707.csv")
    passed = [row for row in rows if _bool_text(row.get("evidence_ok", ""))]
    methods = sorted({row.get("method", "") for row in rows})
    datasets = sorted({row.get("dataset", "") for row in rows})
    ok = (
        len(rows) == 100
        and len(passed) == 100
        and methods
        == [
            "private_gsd_gpu_1m_fulln_audit",
            "private_pgm_aim",
            "private_pgm_mst",
            "rap",
            "sage",
        ]
        and datasets == ["acs_sage_strong", "adult_sage_strong", "br2000_sage_strong", "nltcs_sage_strong"]
    )
    return ok, f"rows={len(rows)}; passed={len(passed)}; methods={methods}; datasets={datasets}"


def _check_pairwise(
    package_dir: Path,
    relpath: str,
    sage_col: str,
    other_col: str,
    expected_sage: int,
    expected_other: int,
) -> tuple[bool, str]:
    rows = _package_csv(package_dir, relpath)
    sage_total = _sum_col(rows, sage_col)
    other_total = _sum_col(rows, other_col)
    ok = sage_total == expected_sage and other_total == expected_other
    return ok, f"{sage_col}={sage_total}; {other_col}={other_total}"


def _check_gsd_detail(package_dir: Path) -> tuple[bool, str]:
    relpath = "tables/sage_vs_private_gsd_1m_fulln_seed0to4_20260706.csv"
    rows = _package_csv(package_dir, relpath)
    sage_total = _sum_col(rows, "sage_metric_wins")
    gsd_total = _sum_col(rows, "gsd_metric_wins")
    gsd_wins = {
        (row["dataset"], metric)
        for row in rows
        for metric in ["MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"]
        if row.get(f"winner_{metric}") == "Private-GSD GPU 1M/full-N"
    }
    expected_gsd_wins = {
        ("acs_sage_strong", "AvgTVD"),
        ("acs_sage_strong", "MaxTVD"),
        ("br2000_sage_strong", "AvgTVD"),
        ("br2000_sage_strong", "MaxTVD"),
    }
    ok = sage_total == 16 and gsd_total == 4 and gsd_wins == expected_gsd_wins
    detail = (
        f"sage_metric_wins={sage_total}; gsd_metric_wins={gsd_total}; "
        f"gsd_wins={sorted(gsd_wins)}"
    )
    return ok, detail


def _check_gpu_provenance(package_dir: Path) -> tuple[bool, str]:
    rows = _package_csv(package_dir, "appendix/baseline_gpu_provenance_20260707.csv")
    required = [row for row in rows if row["gpu_policy"] == "required"]
    cpu_native = [row for row in rows if row["gpu_policy"] == "cpu_native"]
    required_ok = [
        row
        for row in required
        if row["status"] == "completed" and _bool_text(row["gpu_ok"]) and row.get("evidence")
    ]
    cpu_ok = [
        row
        for row in cpu_native
        if row["status"] == "completed" and row["gpu_ok"] == "not_required"
    ]
    ok = len(rows) == 100 and len(required_ok) == 60 and len(cpu_ok) == 40
    return ok, f"rows={len(rows)}; required_gpu_ok={len(required_ok)}; cpu_native_ok={len(cpu_ok)}"


def _check_baseline_tiering(package_dir: Path) -> tuple[bool, str]:
    rows = {
        row["candidate"]: row
        for row in _package_csv(package_dir, "appendix/baseline_admission_audit_20260706.csv")
    }
    checks = {
        "RAP softmax": ("primary", "admitted_primary", "4", "5"),
        "RAP++ official ACS grid": ("original_protocol", "admitted_original_protocol", "25", "5"),
        "PrivMRF official TVD": ("original_protocol", "admitted_original_protocol", "4", "1"),
    }
    details = []
    for candidate, (tier, status, dataset_count, seed_count) in checks.items():
        row = rows.get(candidate)
        if row is None:
            return False, f"missing {candidate}"
        observed = (
            row.get("expected_tier"),
            row.get("machine_status"),
            row.get("dataset_count"),
            row.get("seed_count"),
        )
        expected = (tier, status, dataset_count, seed_count)
        if observed != expected:
            return False, f"{candidate}: observed={observed} expected={expected}"
        details.append(f"{candidate}:{tier}/{status}")
    return True, "; ".join(details)


def _check_original_protocol_audit(package_dir: Path) -> tuple[bool, str]:
    rows = {
        row["candidate"]: row
        for row in _package_csv(package_dir, "appendix/original_protocol_baseline_audit_20260707.csv")
    }
    checks = {
        "RAP++ official ACS grid": ("passed", "125", "25", "5"),
        "PrivMRF official TVD": ("passed", "72", "4", "1"),
    }
    details = []
    for candidate, (status, row_count, dataset_count, seed_count) in checks.items():
        row = rows.get(candidate)
        if row is None:
            return False, f"missing {candidate}"
        observed = (
            row.get("audit_status"),
            row.get("row_count"),
            row.get("dataset_count"),
            row.get("seed_count"),
        )
        expected = (status, row_count, dataset_count, seed_count)
        if observed != expected:
            return False, f"{candidate}: observed={observed} expected={expected}"
        details.append(f"{candidate}:{status}/{row_count} rows")
    return True, "; ".join(details)


def _write_claim_traceability(package_dir: Path) -> list[Path]:
    rows: list[dict[str, str]] = []

    primary_artifacts = [
        "tables/paper_tables_draft.tex",
        "tables/sage_all4_seed0to4_summary_rho1_20260706.csv",
        "tables/private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
        "tables/private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv",
        "tables/private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv",
        "tables/rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
        "appendix/primary_run_evidence_audit_20260707.csv",
    ]
    ok, detail = _artifact_status(package_dir, primary_artifacts)
    if ok:
        ok, detail = _check_summary_coverage(package_dir)
    if ok:
        ok, detail = _check_primary_run_evidence(package_dir)
    _trace_row(
        rows,
        "primary-table-coverage",
        "The primary strict same-protocol table contains SAGE, RAP softmax, high-power Private-GSD, AIM, and MST over four datasets and five seeds.",
        "strict same-protocol rho=1 seed0-4",
        primary_artifacts,
        "verify_paper_claims:_check_summary_files + _check_main_table_tex + audit_primary_run_evidence.py",
        ok,
        detail,
    )

    pairwise_specs = [
        (
            "sage-vs-aim",
            "SAGE wins all 20 dataset-metric cells against Private-PGM AIM.",
            "tables/sage_vs_private_pgm_aim_seed0to4_20260706.csv",
            "sage_wins",
            "aim_wins",
            20,
            0,
            "verify_paper_claims:_check_pairwise_win_counts",
        ),
        (
            "sage-vs-mst",
            "SAGE wins all 20 dataset-metric cells against Private-PGM MST.",
            "tables/sage_vs_private_pgm_mst_seed0to4_20260706.csv",
            "sage_wins",
            "mst_wins",
            20,
            0,
            "verify_paper_claims:_check_pairwise_win_counts",
        ),
        (
            "sage-vs-rap",
            "SAGE wins all 20 dataset-metric cells against RAP softmax.",
            "tables/sage_vs_rap_softmax_seed0to4_20260706.csv",
            "sage_metric_wins",
            "rap_metric_wins",
            20,
            0,
            "verify_paper_claims:_check_pairwise_win_counts",
        ),
    ]
    for claim_id, claim, relpath, sage_col, other_col, expected_sage, expected_other, verifier in pairwise_specs:
        ok, detail = _artifact_status(package_dir, [relpath])
        if ok:
            ok, detail = _check_pairwise(package_dir, relpath, sage_col, other_col, expected_sage, expected_other)
        _trace_row(
            rows,
            claim_id,
            claim,
            "strict same-protocol rho=1 seed0-4",
            [relpath],
            verifier,
            ok,
            detail,
        )

    gsd_artifact = "tables/sage_vs_private_gsd_1m_fulln_seed0to4_20260706.csv"
    ok, detail = _artifact_status(package_dir, [gsd_artifact])
    if ok:
        ok, detail = _check_gsd_detail(package_dir)
    _trace_row(
        rows,
        "sage-vs-gsd",
        "Against high-power Private-GSD, SAGE wins MAE, RMSE, and MaxErr on all datasets and loses AvgTVD/MaxTVD on ACS and BR2000.",
        "strict same-protocol rho=1 seed0-4",
        [gsd_artifact],
        "verify_paper_claims:_check_pairwise_win_counts + traceability winner-set check",
        ok,
        detail,
    )

    gpu_artifacts = [
        "appendix/baseline_gpu_provenance_20260707.csv",
        "appendix/baseline_gpu_provenance_20260707.md",
    ]
    ok, detail = _artifact_status(package_dir, gpu_artifacts)
    if ok:
        ok, detail = _check_gpu_provenance(package_dir)
    _trace_row(
        rows,
        "gpu-provenance",
        "Paper-facing SAGE, Private-GSD, and RAP runs carry GPU evidence; AIM and MST are explicitly CPU-native Private-PGM baselines.",
        "strict same-protocol execution provenance",
        gpu_artifacts,
        "audit_gpu_provenance.py",
        ok,
        detail,
    )

    tier_artifacts = [
        "appendix/baseline_admission_audit_20260706.csv",
        "appendix/original_protocol_baseline_audit_20260707.csv",
        "appendix/rappp_official_paper_grid_seed0to4_20260707.csv",
        "appendix/privmrf_official_full_tvd_epsgrid_m300_20260706.csv",
    ]
    ok, detail = _artifact_status(package_dir, tier_artifacts)
    if ok:
        ok, detail = _check_baseline_tiering(package_dir)
    if ok:
        ok, detail = _check_original_protocol_audit(package_dir)
    _trace_row(
        rows,
        "baseline-tiering",
        "RAP softmax is primary same-protocol evidence; RAP++ official and PrivMRF official are original-protocol reproduced evidence, not direct same-protocol rows.",
        "baseline admission and reporting tier",
        tier_artifacts,
        "verify_paper_claims:_check_admission_audit + audit_baseline_admission.py + audit_original_protocol_baselines.py",
        ok,
        detail,
    )

    selector_artifacts = [
        "tables/certified_budget_curve_summary.md",
        "tables/certified_r50_rho1_four_dataset_summary.md",
        "tables/certified_r50_rho1_four_dataset_paired.csv",
    ]
    ok, detail = _artifact_status(package_dir, selector_artifacts)
    _trace_row(
        rows,
        "certified-selector-boundary",
        "SAGE-Select certified adaptive evidence is transcript-only selector evidence and remains separate from the static all-measurement external-baseline main table.",
        "certified adaptive selector appendix",
        selector_artifacts,
        "package manifest protocol note + certified selector tables",
        ok,
        detail,
    )

    ablation_artifacts = [
        "tables/sage_ablation_unweighted_multiseed_summary_rho1_20260706.csv",
        "tables/sage_ablation_low_order_multiseed_summary_rho1_20260706.csv",
        "tables/sage_ablation_no_projection_multiseed_summary_rho1_20260706.csv",
    ]
    ok, detail = _artifact_status(package_dir, ablation_artifacts)
    _trace_row(
        rows,
        "sage-ablation-coverage",
        "The current package contains multi-seed ablations for objective weighting, low-order workload, and projection.",
        "SAGE mechanism ablation",
        ablation_artifacts,
        "package filelist + table artifacts",
        ok,
        detail,
    )

    failed = [row for row in rows if row["status"] != "pass"]
    if failed:
        details = "; ".join(f"{row['claim_id']}: {row['machine_check']}" for row in failed)
        raise ValueError(f"claim traceability checks failed: {details}")

    csv_path = package_dir / "tables" / "paper_claim_traceability_20260707.csv"
    md_path = package_dir / "tables" / "paper_claim_traceability_20260707.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Paper Claim Traceability",
        "",
        "This table maps reviewer-facing experimental claims to package artifacts and machine checks.",
        "",
        "| Claim ID | Status | Scope | Evidence Artifacts | Verifier / Gate | Machine Check |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = row["evidence_artifacts"].replace("|", "/")
        check = row["machine_check"].replace("|", "/")
        lines.append(
            f"| `{row['claim_id']}` | {row['status']} | {row['scope']} | {evidence} | "
            f"{row['verifier_or_gate']} | {check} |"
        )
    lines += [
        "",
        "## Claim Text",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['claim_id']}`: {row['claim']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def _write_manifest(
    out_dir: Path,
    main_csv: Path,
    copied_main: list[Path],
    figures: list[Path],
    certified: list[Path],
    appendix: list[Path],
) -> Path:
    manifest = out_dir / "PAPER_RESULTS_PACKAGE_20260706.md"
    lines = [
        "# Paper Results Package 2026-07-06",
        "",
        "This package collects the current paper-facing experimental artifacts for the SAGE strong-workload results.",
        "",
        "## Main Table",
        "",
        "Source:",
        "",
        f"- `{_display_path(main_csv)}`",
        f"- `{_display_path(EXTERNAL_RESULTS / 'private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv')}`",
        f"- `{_display_path(EXTERNAL_RESULTS / 'private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv')}`",
        f"- `{_display_path(EXTERNAL_RESULTS / 'private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv')}`",
        f"- `{_display_path(EXTERNAL_RESULTS / 'rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv')}`",
        "",
        "The paper-ready primary table merges the RAP softmax summary with the SAGE, Private-GSD, AIM, and MST summary.",
        "",
        "Copied artifacts:",
        "",
    ]
    lines += [f"- `{_display_path(path, out_dir)}`" for path in copied_main]
    lines += [
        "",
        "Paper-ready table entry point:",
        "",
        "- `tables/paper_tables_draft.tex`",
        "",
        "Claim traceability entry points:",
        "",
        "- `tables/paper_claim_traceability_20260707.csv`",
        "- `tables/paper_claim_traceability_20260707.md`",
        "",
        "Current generated table labels:",
        "",
        "- `tab:main-results`",
        "- `tab:main-runtime`",
        "- `tab:private-gsd-1m-fulln-sensitivity`",
        "- `tab:ablation-ratios`",
        "- `tab:certified-adaptive-paired`",
        "- `tab:certified-extra-diagnostic`",
        "- `tab:rap-secondary`",
        "- `tab:gem-diagnosis`",
        "- `tab:original-protocol-baselines`",
        "- `tab:rappp-pathcheck`",
        "- `tab:appendix-baseline-audit`",
        "",
        "## Ablation Tables",
        "",
        "- `tables/sage_ablation_seed0_rho1_20260706.csv`",
        "- `tables/sage_ablation_seed0_rho1_20260706.md`",
        "- `tables/sage_ablation_unweighted_multiseed_rho1_20260706.csv`",
        "- `tables/sage_ablation_unweighted_multiseed_rho1_20260706.md`",
        "- `tables/sage_ablation_unweighted_multiseed_summary_rho1_20260706.csv`",
        "- `tables/sage_ablation_low_order_multiseed_rho1_20260706.csv`",
        "- `tables/sage_ablation_low_order_multiseed_rho1_20260706.md`",
        "- `tables/sage_ablation_low_order_multiseed_summary_rho1_20260706.csv`",
        "- `tables/sage_ablation_no_projection_multiseed_rho1_20260706.csv`",
        "- `tables/sage_ablation_no_projection_multiseed_rho1_20260706.md`",
        "- `tables/sage_ablation_no_projection_multiseed_summary_rho1_20260706.csv`",
        "",
        "## Certified Adaptive Selector Tables",
        "",
    ]
    lines += [f"- `{_display_path(path, out_dir)}`" for path in certified]
    lines += [
        "",
        "Generated table/figure entry points:",
        "",
        "- `tab:certified-adaptive-paired` in `tables/paper_tables_draft.tex`",
        "- `tab:certified-extra-diagnostic` in `tables/paper_tables_draft.tex`",
        "- `figures/certified_adaptive_primary_improvement.pdf`",
        "- `figures/certified_adaptive_all_metric_improvement.pdf`",
        "- `figures/certified_adaptive_primary_error_curves.pdf`",
        "- `figures/private_gsd_1m_fulln_sensitivity_ratios.pdf`",
        "",
        "Original-protocol reproduced baselines:",
        "",
        "- `tab:original-protocol-baselines` in `tables/paper_tables_draft.tex`",
        "- `appendix/rappp_official_paper_grid_seed0to4_20260707.csv`",
        "- `appendix/privmrf_official_full_tvd_epsgrid_m300_20260706.csv`",
        "- `appendix/original_protocol_baseline_audit_20260707.csv`",
        "",
        "These certified adaptive results compare transcript-only SAGE-Select ordered-gain selection with AIM-style L1 floor selection. They should not be merged with the static all-measurement external-baseline main table. The BR2000/NLTCS certified transcript-only extra-dataset table is a diagnostic appendix item because the three-seed evidence is mixed: BR2000 is negative on MAE/AvgTVD and slightly positive on RMSE, while NLTCS is positive on MAE/AvgTVD but negative on RMSE and tail metrics.",
        "",
        "Recommended primary-table methods:",
        "",
        "- SAGE",
        "- RAP softmax",
        "- Private-GSD GPU 1M/full-N",
        "- Private-PGM AIM",
        "- Private-PGM MST",
        "",
        "The primary Private-GSD row uses the high-power GPU 1M/full-N audit protocol. It is the most competitive baseline: SAGE wins MAE/RMSE/MaxErr on all datasets, while Private-GSD wins AvgTVD/MaxTVD on ACS and BR2000.",
        "RAP softmax is included in the primary row-level utility comparison because it has four-dataset, five-seed GPU evidence under the shared evaluator; GEM is reported as remap-diagnosis evidence; DataSynthesizer PrivBayes, DPMM PrivBayes, unofficial PrivSyn, and PrivMRF GPU are explicit appendix audit rows.",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{_display_path(path, out_dir)}`" for path in figures]
    lines += [
        "",
        "## Appendix And Audit Tables",
        "",
    ]
    lines += [f"- `{_display_path(path, out_dir)}`" for path in appendix]
    lines += [
        "",
        "## Protocol Notes",
        "",
        "- All primary-table rows use the canonical external evaluator.",
        "- True answers are used only by the offline evaluator.",
        "- Main results use `rho=1.0`, `delta=1e-9`, and seeds 0/1/2/3/4.",
        "- Private-GSD GPU uses JAX GPU backend in the `gsd` environment.",
        "- Private-GSD GPU 1M/full-N uses `N_prime=n`, `tree_query_depth=2`, one million generations, mutate/swap/cross, and no early stop before the one-million-generation budget.",
        "- Certified adaptive SAGE-Select evidence is a separate selector-comparison table, not a replacement for the static strong-workload external-baseline table.",
        "- RAP softmax has been promoted to the primary row-level utility comparison; the current rows are GPU runs (`T=30`, `K=30`, `max_iters=1000`), not RAP++.",
        "- GEM remains diagnosis/appendix evidence because the transformer-coordinate remap seed0 all-dataset diagnostic is much weaker than the primary baselines; the old relaxed-to-row gap was a wrapper coordinate artifact, not CPU fallback.",
        "- RAP++ marginal path-check rows remain appendix audit evidence, while `rappp_official_paper_grid_seed0to4_20260707` is reported separately as upstream original-protocol reproduced evidence.",
        "- DataSynthesizer PrivBayes, DPMM PrivBayes, unofficial PrivSyn, and PrivMRF GPU are represented in the appendix audit tier; `privmrf_official_full_tvd_epsgrid_m300_20260706` is reported separately as upstream original-protocol reproduced TVD evidence.",
        "- `baseline_admission_audit_20260706` records which primary, secondary, and appendix baselines are broad enough for paper-table admission.",
        "- `original_protocol_baseline_audit_20260707` records machine-checkable protocol evidence for RAP++ official and PrivMRF official reproduced rows.",
        "- `rappp_official_paper_grid_seed0to4_20260707` records upstream RAP++ ACS/Folktables original-protocol reproduction evidence over seeds 0 through 4. Keep it separate from the strict same-protocol SAGE evaluator table.",
        "",
    ]
    manifest.write_text("\n".join(lines))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package current paper-facing SAGE experiment results.")
    parser.add_argument(
        "--main-csv",
        type=Path,
        default=EXTERNAL_RESULTS / "sage_all4_seed0to4_summary_rho1_20260706.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXTERNAL_RESULTS / "paper_package_seed0to4_20260706",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_style()
    rows = _primary_rows(args.main_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.main_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    summary_figures_dir = args.out_dir / "summary_figures"
    appendix_dir = args.out_dir / "appendix"
    for generated_dir in [tables_dir, figures_dir, summary_figures_dir, appendix_dir]:
        if generated_dir.exists():
            shutil.rmtree(generated_dir)

    _write_paper_tables_draft(EXTERNAL_RESULTS, tables_dir)

    copied_main = []
    for suffix in [".csv", ".md", ".tex"]:
        copied = _copy_artifact(args.main_csv.with_suffix(suffix), tables_dir)
        if copied is not None:
            copied_main.append(copied)
    for name in RHO_SWEEP_ARTIFACTS:
        copied = _copy_artifact(EXTERNAL_RESULTS / name, tables_dir)
        if copied is not None:
            copied_main.append(copied)
    for name in PRIMARY_SEED0TO4_ARTIFACTS:
        copied = _copy_artifact(EXTERNAL_RESULTS / name, tables_dir)
        if copied is not None:
            copied_main.append(copied)
    for name in SAGE_ABLATION_ARTIFACTS:
        copied = _copy_artifact(EXTERNAL_RESULTS / name, tables_dir)
        if copied is not None:
            copied_main.append(copied)

    figures = []
    figures += plot_metric_grid(rows, figures_dir, METHOD_ORDER, "main_metrics_log_grid")
    figures += plot_relative_to_sage(rows, figures_dir, METHOD_ORDER)
    figures += plot_metric_grid(
        rows,
        figures_dir,
        SENSITIVITY_METHOD_ORDER,
        "main_metrics_log_grid_with_lightweight_gsd",
    )
    figures += plot_private_gsd_fulln_sensitivity(rows, figures_dir)
    summary_figures = plot_summary_metric_bars(rows, summary_figures_dir, METHOD_ORDER)
    if RHO_SWEEP_FIGURES_DIR.exists():
        for path in sorted(RHO_SWEEP_FIGURES_DIR.glob("*")):
            copied = _copy_artifact(path, figures_dir)
            if copied is not None:
                figures.append(copied)
    ablation_csv = EXTERNAL_RESULTS / "sage_ablation_seed0_rho1_20260706.csv"
    if ablation_csv.exists():
        figures += plot_sage_ablation_ratios(_read_rows(ablation_csv), figures_dir)
    unweighted_summary_csv = EXTERNAL_RESULTS / "sage_ablation_unweighted_multiseed_summary_rho1_20260706.csv"
    if unweighted_summary_csv.exists():
        figures += plot_sage_variant_multiseed(
            _read_rows(unweighted_summary_csv),
            figures_dir,
            "unweighted_objective",
            "Unweighted objective",
            "sage_unweighted_multiseed_ratios",
        )
    low_order_summary_csv = EXTERNAL_RESULTS / "sage_ablation_low_order_multiseed_summary_rho1_20260706.csv"
    if low_order_summary_csv.exists():
        figures += plot_sage_variant_multiseed(
            _read_rows(low_order_summary_csv),
            figures_dir,
            "low_order_workload",
            "Low-order workload",
            "sage_low_order_multiseed_ratios",
        )
    no_projection_summary_csv = EXTERNAL_RESULTS / "sage_ablation_no_projection_multiseed_summary_rho1_20260706.csv"
    if no_projection_summary_csv.exists():
        figures += plot_sage_variant_multiseed(
            _read_rows(no_projection_summary_csv),
            figures_dir,
            "no_projection",
            "No projection",
            "sage_no_projection_multiseed_ratios",
        )
    figures += sorted(figures_dir.glob("certified_adaptive_*"))
    figures = sorted(figures_dir.glob("*"))
    all_figures = sorted([*figures, *summary_figures])

    certified = []
    for name in CERTIFIED_ADAPTIVE_ARTIFACTS:
        copied = _copy_artifact(SAGE_PAPER_DIR / name, tables_dir)
        if copied is not None:
            certified.append(copied)

    appendix = []
    for name in APPENDIX_ARTIFACTS:
        copied = _copy_artifact(EXTERNAL_RESULTS / name, appendix_dir)
        if copied is not None:
            appendix.append(copied)

    copied_main += _write_claim_traceability(args.out_dir)

    manifest = _write_manifest(args.out_dir, args.main_csv, copied_main, all_figures, certified, appendix)
    filelist, sha_path = _write_filelist_and_hashes(args.out_dir)
    print(f"wrote package manifest: {manifest}")
    print(f"wrote package filelist: {filelist}")
    print(f"wrote package sha256 sums: {sha_path}")
    print(
        f"wrote {len(figures)} figure files, {len(summary_figures)} summary figure files, "
        f"and {len(appendix)} appendix artifacts"
    )


if __name__ == "__main__":
    main()
