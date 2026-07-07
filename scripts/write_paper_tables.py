from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    from path_defaults import external_results, legacy_paper_package_dir, sage_paper_dir
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, legacy_paper_package_dir, sage_paper_dir

EXTERNAL_RESULTS = external_results()
SAGE_PAPER_DIR = sage_paper_dir()
DEFAULT_PACKAGE_DIR = legacy_paper_package_dir()
DEFAULT_DOCS_DIR = Path("docs")

SAGE_SUMMARY_SEED0TO4 = "sage_all4_seed0to4_summary_rho1_20260706.csv"
PRIVATE_GSD_FULLN_SUMMARY_SEED0TO4 = "private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv"
PRIVATE_PGM_AIM_SUMMARY_SEED0TO4 = "private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv"
PRIVATE_PGM_MST_SUMMARY_SEED0TO4 = "private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv"
RAP_SUMMARY_SEED0TO4 = "rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv"

DATASET_ORDER = [
    "adult_sage_strong",
    "acs_sage_strong",
    "br2000_sage_strong",
    "nltcs_sage_strong",
]

DATASET_LABELS = {
    "adult_sage_strong": "Adult",
    "acs_sage_strong": "ACS",
    "br2000_sage_strong": "BR2000",
    "nltcs_sage_strong": "NLTCS",
}

MAIN_METHOD_ORDER = [
    "SAGE",
    "RAP softmax",
    "Private-GSD GPU",
    "Private-PGM AIM",
    "Private-PGM MST",
]

HIGHPOWER_GSD_METHOD_ORDER = [
    "SAGE",
    "RAP softmax",
    "Private-GSD GPU 1M/full-N",
    "Private-PGM AIM",
    "Private-PGM MST",
]

METRICS = [
    ("MAE", "MAE_mean", "MAE_sem"),
    ("RMSE", "RMSE_mean", "RMSE_sem"),
    ("AvgTVD", "AvgTVD_mean", "AvgTVD_sem"),
    ("MaxErr", "MaxErr_mean", "MaxErr_sem"),
    ("MaxTVD", "MaxTVD_mean", "MaxTVD_sem"),
]

ABLATION_METRICS = [
    ("MAE", "full_true_mae_ratio_vs_full_mean", "full_true_mae_ratio_vs_full_sem"),
    ("RMSE", "full_true_rmse_ratio_vs_full_mean", "full_true_rmse_ratio_vs_full_sem"),
    ("AvgTVD", "full_true_avg_tvd_ratio_vs_full_mean", "full_true_avg_tvd_ratio_vs_full_sem"),
    ("MaxErr", "full_true_max_error_ratio_vs_full_mean", "full_true_max_error_ratio_vs_full_sem"),
    ("MaxTVD", "full_true_max_tvd_ratio_vs_full_mean", "full_true_max_tvd_ratio_vs_full_sem"),
]

ABLATION_FILES = [
    "sage_ablation_unweighted_multiseed_summary_rho1_20260706.csv",
    "sage_ablation_low_order_multiseed_summary_rho1_20260706.csv",
    "sage_ablation_no_projection_multiseed_summary_rho1_20260706.csv",
]

APPENDIX_METRICS = ["MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"]

CERTIFIED_IMPROVEMENT_METRICS = [
    ("MAE", "full_true_mae_mean_rel_improvement_pct", "full_true_mae_std_rel_improvement_pct"),
    ("RMSE", "full_true_rmse_mean_rel_improvement_pct", "full_true_rmse_std_rel_improvement_pct"),
    ("AvgTVD", "full_true_avg_tvd_mean_rel_improvement_pct", "full_true_avg_tvd_std_rel_improvement_pct"),
    ("MaxErr", "full_true_max_error_mean_rel_improvement_pct", "full_true_max_error_std_rel_improvement_pct"),
    ("MaxTVD", "full_true_max_tvd_mean_rel_improvement_pct", "full_true_max_tvd_std_rel_improvement_pct"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def appendix_float(row: dict[str, str], metric: str) -> float:
    aliases = {
        "MAE": ["MAE", "mae"],
        "RMSE": ["RMSE", "rmse"],
        "AvgTVD": ["AvgTVD", "avg_tvd"],
        "MaxErr": ["MaxErr", "max_error"],
        "MaxTVD": ["MaxTVD", "max_tvd"],
        "runtime": ["runtime_sec", "runtime_seconds", "runtime_s"],
    }
    for key in aliases[metric]:
        value = row.get(key, "")
        if value != "":
            return float(value)
    return float("nan")


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_num(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "--"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 1e-3:
        text = f"{value:.2e}"
        mantissa, exponent = text.split("e")
        return rf"${mantissa}\times 10^{{{int(exponent)}}}$"
    if abs_value < 1:
        return f"{value:.4f}"
    return f"{value:.{digits}f}"


def fmt_mean_sem(row: dict[str, str], mean_key: str, sem_key: str) -> str:
    return rf"{fmt_num(as_float(row, mean_key))} $\pm$ {fmt_num(as_float(row, sem_key))}"


def fmt_ratio(row: dict[str, str], mean_key: str, sem_key: str) -> str:
    return rf"{as_float(row, mean_key):.2f} $\pm$ {as_float(row, sem_key):.2f}"


def fmt_pct_mean_std(row: dict[str, str], mean_key: str, std_key: str) -> str:
    return rf"{as_float(row, mean_key):.2f}\% $\pm$ {as_float(row, std_key):.2f}\%"


def load_primary_rows(results_dir: Path, *, gsd_main: str = "200k") -> list[dict[str, str]]:
    rows = read_rows(results_dir / SAGE_SUMMARY_SEED0TO4)
    if gsd_main == "1m-fulln":
        rows.extend(read_rows(results_dir / PRIVATE_GSD_FULLN_SUMMARY_SEED0TO4))
    else:
        rows.extend(
            row
            for row in read_rows(results_dir / "private_gsd_gpu_multiseed_all4_20260706.csv")
            if row["method"] == "Private-GSD GPU"
        )
    rows.extend(read_rows(results_dir / PRIVATE_PGM_AIM_SUMMARY_SEED0TO4))
    rows.extend(read_rows(results_dir / PRIVATE_PGM_MST_SUMMARY_SEED0TO4))
    rap_path = results_dir / RAP_SUMMARY_SEED0TO4
    if rap_path.exists():
        for rap in read_rows(rap_path):
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


def table_env(label: str, caption: str, tabular: str, placement: str = "t") -> str:
    return "\n".join(
        [
            rf"\begin{{table*}}[{placement}]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            tabular,
            r"\end{table*}",
        ]
    )


def build_tabular(columns: list[str], rows: list[list[str]], align: str | None = None) -> str:
    if align is None:
        align = "ll" + "r" * max(0, len(columns) - 2)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(columns) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main_result_table(results_dir: Path, *, gsd_main: str = "200k") -> str:
    rows = load_primary_rows(results_dir, gsd_main=gsd_main)
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    method_order = HIGHPOWER_GSD_METHOD_ORDER if gsd_main == "1m-fulln" else MAIN_METHOD_ORDER
    table_rows: list[list[str]] = []
    for dataset in DATASET_ORDER:
        for method in method_order:
            row = lookup[(dataset, method)]
            table_rows.append(
                [
                    latex_escape(DATASET_LABELS[dataset]),
                    latex_escape(method),
                    *[fmt_mean_sem(row, mean_key, sem_key) for _, mean_key, sem_key in METRICS],
                ]
            )
    tabular = build_tabular(
        ["Dataset", "Method", "MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"],
        table_rows,
        align="llccccc",
    )
    if gsd_main == "1m-fulln":
        gsd_note = (
            "The Private-GSD GPU row uses the high-power $N'=n$, one-million-generation, "
            "no-early-stop-before-budget audit configuration."
        )
    else:
        gsd_note = (
            "The Private-GSD GPU row uses the documented 200k-generation early-stop GPU configuration."
        )
    caption = (
        "Main row-level utility comparison at $\\rho=1.0$ over seeds 0 through 4. "
        "Entries are mean $\\pm$ SEM; lower is better. The primary table includes complete "
        "row-level baselines that share the same datasets, privacy budget, seeds, synthetic "
        f"row-count convention, and external evaluator. {gsd_note}"
    )
    return table_env("tab:main-results", caption, tabular)


def runtime_table(results_dir: Path, *, gsd_main: str = "200k") -> str:
    rows = load_primary_rows(results_dir, gsd_main=gsd_main)
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    method_order = HIGHPOWER_GSD_METHOD_ORDER if gsd_main == "1m-fulln" else MAIN_METHOD_ORDER
    table_rows: list[list[str]] = []
    for dataset in DATASET_ORDER:
        for method in method_order:
            row = lookup[(dataset, method)]
            table_rows.append(
                [
                    latex_escape(DATASET_LABELS[dataset]),
                    latex_escape(method),
                    fmt_mean_sem(row, "runtime_s_mean", "runtime_s_sem"),
                ]
            )
    tabular = build_tabular(["Dataset", "Method", "Runtime (s)"], table_rows, align="llc")
    caption = (
        "Runtime for the primary row-level comparison at $\\rho=1.0$. "
        "Entries are mean $\\pm$ SEM over seeds 0 through 4."
    )
    return table_env("tab:main-runtime", caption, tabular)


def private_gsd_fulln_sensitivity_table(results_dir: Path) -> str:
    main_rows = read_rows(results_dir / SAGE_SUMMARY_SEED0TO4)
    sensitivity_path = results_dir / PRIVATE_GSD_FULLN_SUMMARY_SEED0TO4
    if not sensitivity_path.exists():
        return ""
    sensitivity_rows = read_rows(sensitivity_path)
    sage_lookup = {row["dataset"]: row for row in main_rows if row["method"] == "SAGE"}
    gsd_lookup = {row["dataset"]: row for row in sensitivity_rows}
    table_rows: list[list[str]] = []
    for dataset in DATASET_ORDER:
        sage = sage_lookup[dataset]
        gsd = gsd_lookup[dataset]
        ratios = []
        for _, mean_key, _ in METRICS:
            denom = as_float(sage, mean_key)
            value = as_float(gsd, mean_key)
            ratios.append(f"{value / denom:.2f}x")
        table_rows.append(
            [
                latex_escape(DATASET_LABELS[dataset]),
                *ratios,
                fmt_mean_sem(gsd, "runtime_s_mean", "runtime_s_sem"),
            ]
        )
    tabular = build_tabular(
        ["Dataset", "MAE/SAGE", "RMSE/SAGE", "AvgTVD/SAGE", "MaxErr/SAGE", "MaxTVD/SAGE", "Runtime (s)"],
        table_rows,
        align="lcccccc",
    )
    caption = (
        "Private-GSD GPU 1M/full-N configuration sensitivity at $\\rho=1.0$ over seeds 0 through 4. "
        "Values are ratios relative to SAGE, so values below 1 mean Private-GSD is lower. "
        "This row uses $N'=n$, one million generations, tree depth 2, mutate/swap/cross, and "
        "no early stop before the one-million-generation budget."
    )
    return table_env("tab:private-gsd-1m-fulln-sensitivity", caption, tabular)


def ablation_table(results_dir: Path) -> str:
    rows: list[dict[str, str]] = []
    for filename in ABLATION_FILES:
        rows.extend(row for row in read_rows(results_dir / filename) if row["variant"] != "full")
    variant_order = {
        "unweighted_objective": 0,
        "low_order_workload": 1,
        "no_projection": 2,
    }
    rows.sort(key=lambda row: (DATASET_ORDER.index(row["dataset"]), variant_order[row["variant"]]))
    table_rows = [
        [
            latex_escape(row["dataset_label"]),
            latex_escape(row["variant_label"]),
            *[fmt_ratio(row, mean_key, sem_key) for _, mean_key, sem_key in ABLATION_METRICS],
        ]
        for row in rows
    ]
    tabular = build_tabular(
        ["Dataset", "Variant", "MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"],
        table_rows,
        align="llccccc",
    )
    caption = (
        "Ablation ratios relative to the full SAGE/QDTE configuration at $\\rho=1.0$. "
        "Values above 1 indicate the ablated variant is worse than the full method."
    )
    return table_env("tab:ablation-ratios", caption, tabular)


def certified_adaptive_paired_table() -> str:
    rows = read_rows(SAGE_PAPER_DIR / "certified_budget_curve_paired.csv")
    rows.sort(key=lambda row: (row["dataset"], float(row["rho_total"])))
    dataset_labels = {"adult": "Adult", "acs": "ACS"}
    table_rows = [
        [
            latex_escape(dataset_labels.get(row["dataset"], row["dataset"])),
            fmt_num(as_float(row, "rho_total")),
            str(int(as_float(row, "n_pairs"))),
            *[fmt_pct_mean_std(row, mean_key, std_key) for _, mean_key, std_key in CERTIFIED_IMPROVEMENT_METRICS],
        ]
        for row in rows
    ]
    tabular = build_tabular(
        ["Dataset", "$\\rho$", "n", "MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"],
        table_rows,
        align="llcccccc",
    )
    caption = (
        "Certified transcript-selection comparison of SAGE-Select ordered-gain against AIM-style $L_1$ floor. "
        "Entries are paired relative improvement percentages over seeds 0, 1, and 2; positive means SAGE-Select "
        "has lower error. Runs use 50 rounds, transcript-only scoring, measurement-only ledger, "
        "three public bootstrap rounds, and all-projected transcript reliability."
    )
    return table_env("tab:certified-adaptive-paired", caption, tabular)


def certified_extra_dataset_diagnostic_table() -> str:
    rows = read_rows(SAGE_PAPER_DIR / "certified_extra_dataset_paired.csv")
    order = {"br2000": 0, "nltcs": 1}
    rows.sort(key=lambda row: (order.get(row["dataset"], 99), float(row["rho_total"])))
    dataset_labels = {"br2000": "BR2000", "nltcs": "NLTCS"}
    table_rows = [
        [
            latex_escape(dataset_labels.get(row["dataset"], row["dataset"])),
            fmt_num(as_float(row, "rho_total")),
            str(int(as_float(row, "n_pairs"))),
            *[fmt_pct_mean_std(row, mean_key, std_key) for _, mean_key, std_key in CERTIFIED_IMPROVEMENT_METRICS],
        ]
        for row in rows
    ]
    tabular = build_tabular(
        ["Dataset", "$\\rho$", "n", "MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"],
        table_rows,
        align="llcccccc",
    )
    caption = (
        "Certified transcript-only extra-dataset diagnostic for BR2000 and NLTCS at $\\rho=1.0$ over "
        "seeds 0, 1, and 2. Entries are paired relative improvement percentages; positive means "
        "SAGE-Select ordered-gain has lower error than AIM-style $L_1$ floor. This diagnostic is mixed "
        "appendix evidence: BR2000 is negative on MAE/AvgTVD and slightly positive on RMSE, while "
        "NLTCS is positive on MAE/AvgTVD but negative on RMSE and tail metrics."
    )
    return table_env("tab:certified-extra-diagnostic", caption, tabular)


def rap_secondary_table(results_dir: Path) -> str:
    rows = read_rows(results_dir / RAP_SUMMARY_SEED0TO4)
    rows.sort(key=lambda row: DATASET_ORDER.index(row["dataset"]))
    table_rows = []
    for row in rows:
        setting = rf"$T={int(as_float(row, 'T'))}, K={int(as_float(row, 'K'))}, M={int(as_float(row, 'num_marginals'))}$"
        table_rows.append(
            [
                latex_escape(DATASET_LABELS[row["dataset"]]),
                setting,
                fmt_mean_sem(row, "full_true_mae_mean", "full_true_mae_sem"),
                fmt_mean_sem(row, "full_true_rmse_mean", "full_true_rmse_sem"),
                fmt_mean_sem(row, "full_true_avg_tvd_mean", "full_true_avg_tvd_sem"),
                fmt_mean_sem(row, "full_true_max_error_mean", "full_true_max_error_sem"),
                fmt_mean_sem(row, "full_true_max_tvd_mean", "full_true_max_tvd_sem"),
                fmt_mean_sem(row, "runtime_seconds_mean", "runtime_seconds_sem"),
            ]
        )
    tabular = build_tabular(
        ["Dataset", "RAP setting", "MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD", "Runtime (s)"],
        table_rows,
        align="llcccccc",
    )
    caption = (
        "GPU-backed RAP softmax configuration details under the same $\\rho=1.0$ evaluation protocol. "
        "The summarized RAP rows are included in the primary utility comparison; this table records "
        "the RAP-specific $T$, $K$, marginal count, runtime, and utility values. Current rows use "
        "$T=30$, $K=30$, sample decoding, and "
        "1000 optimization iterations."
    )
    return table_env("tab:rap-secondary", caption, tabular)


def gem_diagnosis_table(results_dir: Path) -> str:
    rows = [
        row
        for row in read_rows(results_dir / "gem_row_realization_diagnostics_adult_sage_strong_20260706.csv")
        if row["num_marginals"] == "445"
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("query_remap", "")).lower() not in ("true", "1", "yes"),
            row["latent_mode"],
            row["decode_mode"],
            as_float(row, "syndata_size"),
        )
    )
    table_rows = []
    for row in rows:
        variant = f"{row['latent_mode']}/{row['decode_mode']}"
        remap = "yes" if str(row.get("query_remap", "")).lower() in ("true", "1", "yes") else "no"
        table_rows.append(
            [
                latex_escape(variant),
                remap,
                str(int(as_float(row, "syndata_size"))),
                fmt_num(as_float(row, "internal_relaxed_cached_mae")),
                fmt_num(as_float(row, "internal_row_mae")),
                f"{as_float(row, 'internal_row_mae_over_relaxed_cached'):.2f}x",
                fmt_num(as_float(row, "full_true_mae")),
                f"{as_float(row, 'external_mae_over_relaxed_cached'):.1f}x",
                fmt_num(as_float(row, "full_true_avg_tvd")),
                fmt_num(as_float(row, "full_true_max_error")),
            ]
        )
    tabular = build_tabular(
        [
            "Variant",
            "Remap",
            "Rows",
            "Relaxed MAE",
            "Row MAE",
            "Row/relaxed",
            "External MAE",
            "External/relaxed",
            "AvgTVD",
            "MaxErr",
        ],
        table_rows,
        align="lllccccccc",
    )
    caption = (
        "GEM coordinate-remap diagnosis on Adult with 445 marginals. Rows is GEM's relaxed synthetic batch size. "
        "The pre-remap wrapper used QueryManager category-id coordinates against RDT first-seen one-hot "
        "positions, creating a public coordinate mismatch. The remapped wrapper removes the relaxed-to-row "
        "gap for sample decoding. The subsequent seed0 all-dataset remap diagnostic remains weak "
        "under the shared strong evaluator, so GEM stays in appendix evidence."
    )
    return table_env("tab:gem-diagnosis", caption, tabular)


def rappp_pathcheck_table(results_dir: Path) -> str:
    rows = read_rows(results_dir / "rappp_marginal_pathcheck_20260706.csv")
    table_rows = []
    for row in rows:
        eval_label = "strong" if str(row.get("strong_eval", "")).lower() == "true" else "weak"
        table_rows.append(
            [
                latex_escape(row["run_name"].replace("seed0_", "")),
                latex_escape(row["input_dataset"]),
                latex_escape(eval_label),
                str(int(as_float(row, "model_rows"))),
                str(int(as_float(row, "dp_select_epochs"))),
                str(int(as_float(row, "iterations"))),
                fmt_num(as_float(row, "full_true_mae")),
                fmt_num(as_float(row, "full_true_avg_tvd")),
                fmt_num(as_float(row, "full_true_max_error")),
            ]
        )
    tabular = build_tabular(
        ["Run", "Input", "Eval.", "Rows", "Epochs", "Iters", "MAE", "AvgTVD", "MaxErr"],
        table_rows,
        align="lllcccccc",
    )
    caption = (
        "RAP++ marginal-only path-check on NLTCS. These rows use the RAP++ projection "
        "implementation with marginal statistics only and are not official full RAP++ "
        "marginal-plus-halfspace/task-target evidence. The first row is the old weak "
        "NLTCS smoke; the strong rows use the 8704-query NLTCS strong evaluator."
    )
    return table_env("tab:rappp-pathcheck", caption, tabular)


def original_protocol_baselines_table(results_dir: Path) -> str:
    rappp_overall = read_rows(results_dir / "rappp_official_paper_grid_seed0to4_20260707_overall.csv")[0]
    privmrf_rows = read_rows(results_dir / "privmrf_official_full_tvd_epsgrid_m300_20260706.csv")
    privmrf_eps32 = [row for row in privmrf_rows if abs(as_float(row, "epsilon") - 3.2) < 1e-9]

    def privmrf_mean(way: int) -> float:
        values = [as_float(row, "tvd_mean") for row in privmrf_eps32 if int(as_float(row, "way")) == way]
        return sum(values) / len(values)

    rows = [
        [
            "RAP++ official",
            "ACS/Folktables native protocol",
            "125 tasks: 5 states, 5 targets, 5 seeds. Seeds 1--4 record JAX GPU metadata.",
            (
                rf"2-way marginal avg. {fmt_num(as_float(rappp_overall, 'marginal_average_error_mean'))}; "
                rf"20K prefix avg. {fmt_num(as_float(rappp_overall, 'prefix_average_error_mean'))}; "
                rf"LR macro F1 {fmt_num(as_float(rappp_overall, 'synthetic_macro_f1_mean'))}; "
                rf"F1 gap {fmt_num(as_float(rappp_overall, 'macro_f1_gap_mean'))}."
            ),
        ],
        [
            "PrivMRF official",
            "Upstream TVD protocol",
            "4 datasets, epsilons 0.1--3.2, repeat 1, 300 marginals.",
            (
                rf"At $\epsilon=3.2$, mean 3/4/5-way TVD across datasets is "
                rf"{fmt_num(privmrf_mean(3))}/{fmt_num(privmrf_mean(4))}/{fmt_num(privmrf_mean(5))}."
            ),
        ],
    ]

    lines = [
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.15\textwidth}p{0.20\textwidth}p{0.31\textwidth}p{0.28\textwidth}}",
        r"\toprule",
        r"Method & Native protocol & Evidence & Native metrics \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row[:3]) + " & " + row[3] + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    caption = (
        "Original-protocol reproduced baseline evidence. These rows run upstream public-code "
        "protocols and native metrics rather than the strict SAGE row-level evaluator, so they "
        "are reported separately from Table~\\ref{tab:main-results}. They support baseline "
        "coverage and reproducibility, not direct same-protocol utility ratios."
    )
    return table_env("tab:original-protocol-baselines", caption, "\n".join(lines))


def appendix_baseline_audit_table(results_dir: Path) -> str:
    datasynth_rows = read_rows(results_dir / "datasynth_privbayes_adult_strong_seed0_20260706.csv")
    dpmm_rows = read_rows(results_dir / "dpmm_privbayes_adult_calibration_seed0_20260706.csv")
    privsyn_rows = read_rows(results_dir / "privsyn_unofficial_adult_strong_seed0_20260706.csv")
    privmrf_rows = read_rows(results_dir / "privmrf_gpu_adult_calibration_seed0_20260706.csv")

    sage_ref = next(row for row in datasynth_rows if row["method"] == "SAGE")
    selected = [
        (
            "DataSynth PrivBayes",
            "degree 2",
            "CPU",
            next(row for row in datasynth_rows if row["method"] == "DataSynthesizer PrivBayes"),
            "appendix",
        ),
        (
            "DPMM PrivBayes",
            "best MAE",
            "CPU",
            min(
                (row for row in dpmm_rows if row["method"].startswith("privbayes_")),
                key=lambda row: appendix_float(row, "MAE"),
            ),
            "calibration failure",
        ),
        (
            "PrivSyn",
            "unofficial best MAE",
            "CPU/unspecified",
            min(
                (row for row in privsyn_rows if row["method"].startswith("PrivSyn")),
                key=lambda row: appendix_float(row, "MAE"),
            ),
            "unofficial appendix",
        ),
        (
            "PrivMRF",
            "GPU best MAE",
            "CuPy/GPU",
            min(
                (row for row in privmrf_rows if row["method"].startswith("privmrf_gpu_")),
                key=lambda row: appendix_float(row, "MAE"),
            ),
            "appendix",
        ),
    ]

    table_rows = []
    for method, setting, device, row, status in selected:
        mae_ratio = appendix_float(row, "MAE") / appendix_float(sage_ref, "MAE")
        table_rows.append(
            [
                latex_escape(method),
                latex_escape(setting),
                latex_escape(device),
                *[fmt_num(appendix_float(row, metric)) for metric in APPENDIX_METRICS],
                f"{mae_ratio:.1f}x",
                latex_escape(status),
            ]
        )
    tabular = build_tabular(
        ["Method", "Setting", "Device", "MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD", "MAE/SAGE", "Tier"],
        table_rows,
        align="lllccccccc",
    )
    caption = (
        "Appendix baseline audit on Adult seed 0 at $\\rho=1.0$. "
        "These rows are reported for transparency but are not primary-table baselines because "
        "they are single-seed calibrations, unofficial implementations, or weaker wrapper variants."
    )
    return table_env("tab:appendix-baseline-audit", caption, tabular)


def build_latex(results_dir: Path, *, gsd_main: str = "200k") -> str:
    parts = [
        "% Auto-generated by scripts/write_paper_tables.py.",
        "% Requires: \\usepackage{booktabs}.",
        "",
        main_result_table(results_dir, gsd_main=gsd_main),
        "",
        runtime_table(results_dir, gsd_main=gsd_main),
        "",
        private_gsd_fulln_sensitivity_table(results_dir),
        "",
        ablation_table(results_dir),
        "",
        rap_secondary_table(results_dir),
        "",
        gem_diagnosis_table(results_dir),
        "",
        original_protocol_baselines_table(results_dir),
        "",
        rappp_pathcheck_table(results_dir),
        "",
        appendix_baseline_audit_table(results_dir),
        "",
        certified_adaptive_paired_table(),
        "",
        certified_extra_dataset_diagnostic_table(),
        "",
    ]
    return "\n".join(parts)


def build_markdown(results_dir: Path, table_path: Path, *, gsd_main: str = "200k") -> str:
    if gsd_main == "1m-fulln":
        main_gsd_label = "Private-GSD GPU 1M/full-N"
        sensitivity_note = (
            "high-power Private-GSD GPU ratio table; the same protocol is promoted "
            "into the generated primary table, and this table records ratios versus SAGE."
        )
        gsd_policy = (
            "The generated primary table promotes the high-power Private-GSD GPU "
            "1M/full-N audit row into the main comparison. Claims should therefore "
            "be framed as SAGE improving MAE/RMSE/MaxErr across datasets while "
            "Private-GSD remains competitive or better on some TVD aggregate metrics."
        )
    else:
        main_gsd_label = "Private-GSD GPU"
        sensitivity_note = (
            "high-power Private-GSD GPU sensitivity table; not merged into the "
            "primary table unless the paper chooses that protocol as the main GSD comparison."
        )
        gsd_policy = (
            "The generated primary table keeps the documented 200k-generation "
            "early-stop Private-GSD GPU row in the main comparison and reports "
            "1M/full-N as a sensitivity table."
        )
    return "\n".join(
        [
            "# Paper Tables Draft",
            "",
            "This note records the current LaTeX-ready table package for the paper draft.",
            "",
            "## Output",
            "",
            f"- LaTeX tables: `{table_path}`",
            "",
            "## Table Tiers",
            "",
            f"- `tab:main-results`: primary row-level comparison. Includes SAGE/QDTE, RAP softmax, {main_gsd_label}, Private-PGM AIM, and Private-PGM MST.",
            "- `tab:main-runtime`: runtime for the same primary row-level comparison.",
            f"- `tab:private-gsd-1m-fulln-sensitivity`: {sensitivity_note}",
            "- `tab:ablation-ratios`: contribution-oriented SAGE/QDTE ablations, reported as ratios against the full method.",
            "- `tab:rap-secondary`: RAP configuration/detail table under the same rho=1 protocol; the same RAP summary is now included in the primary row-level comparison.",
            "- `tab:gem-diagnosis`: GEM coordinate-remap diagnosis; the remapped seed0 all-dataset diagnostic remains weak, so this stays appendix/diagnostic.",
            "- `tab:original-protocol-baselines`: upstream native-protocol reproduction evidence for RAP++ official and PrivMRF official; keep separate from the strict row-level evaluator table.",
            "- `tab:rappp-pathcheck`: RAP++ marginal-only path-check; not official full RAP++ evidence.",
            "- `tab:appendix-baseline-audit`: Adult seed0 audit rows for PrivBayes/DataSynth, DPMM PrivBayes, unofficial PrivSyn, and PrivMRF GPU.",
            "- `tab:certified-adaptive-paired`: theorem-aligned transcript-only SAGE-Select versus AIM-style selection on Adult/ACS budget curves.",
            "- `tab:certified-extra-diagnostic`: BR2000/NLTCS certified adaptive three-seed diagnostic; mixed evidence, appendix only.",
            "",
            "## Input Artifacts",
            "",
            f"- `{results_dir / SAGE_SUMMARY_SEED0TO4}`",
            f"- `{results_dir / PRIVATE_GSD_FULLN_SUMMARY_SEED0TO4}`",
            f"- `{results_dir / PRIVATE_PGM_AIM_SUMMARY_SEED0TO4}`",
            f"- `{results_dir / PRIVATE_PGM_MST_SUMMARY_SEED0TO4}`",
            f"- `{results_dir / 'sage_ablation_unweighted_multiseed_summary_rho1_20260706.csv'}`",
            f"- `{results_dir / 'sage_ablation_low_order_multiseed_summary_rho1_20260706.csv'}`",
            f"- `{results_dir / 'sage_ablation_no_projection_multiseed_summary_rho1_20260706.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_budget_curve_paired.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_budget_curve_aggregate.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_budget_curve_raw.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_extra_dataset_paired.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_extra_dataset_aggregate.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_extra_dataset_raw.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_extra_dataset_summary.md'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_r50_rho1_four_dataset_paired.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_r50_rho1_four_dataset_aggregate.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_r50_rho1_four_dataset_raw.csv'}`",
            f"- `{SAGE_PAPER_DIR / 'certified_r50_rho1_four_dataset_summary.md'}`",
            f"- `{results_dir / RAP_SUMMARY_SEED0TO4}`",
            f"- `{results_dir / 'gem_row_realization_diagnostics_adult_sage_strong_20260706.csv'}`",
            f"- `{results_dir / 'rappp_official_paper_grid_seed0to4_20260707_overall.csv'}`",
            f"- `{results_dir / 'privmrf_official_full_tvd_epsgrid_m300_20260706.csv'}`",
            f"- `{results_dir / 'rappp_marginal_pathcheck_20260706.csv'}`",
            f"- `{results_dir / 'datasynth_privbayes_adult_strong_seed0_20260706.csv'}`",
            f"- `{results_dir / 'dpmm_privbayes_adult_calibration_seed0_20260706.csv'}`",
            f"- `{results_dir / 'privsyn_unofficial_adult_strong_seed0_20260706.csv'}`",
            f"- `{results_dir / 'privmrf_gpu_adult_calibration_seed0_20260706.csv'}`",
            "",
            "## Writing Notes",
            "",
            f"- {gsd_policy}",
            "- The primary table intentionally excludes the lightweight `Private-GSD` row because the paper-facing comparison should use the GPU/tuned GSD run.",
            "- When using the 200k main table, the paper-facing `Private-GSD GPU` row uses `N_prime=2048`, `tree_query_depth=2`, `num_generations=200000`, `stop_early_min_generation=2048`, and `early_stop_threshold=0.01`; do not describe it as a one-million-generation no-early-stop run.",
            "- When using the high-power main table, the `Private-GSD GPU 1M/full-N` row uses `N_prime=n`, one million generations, tree depth 2, mutate/swap/cross, and no early stop before the one-million-generation budget.",
            "- The certified adaptive table is a SAGE-Select versus AIM selector comparison, not the same claim as the static all-measurement external baseline table.",
            "- The certified extra-dataset diagnostic is mixed across three seeds: BR2000 is negative on MAE/AvgTVD and slightly positive on RMSE; NLTCS is positive on MAE/AvgTVD but negative on RMSE/tail. Keep it in appendix or limitation discussion.",
            "- RAP softmax is included in the primary row-level comparison because it has complete four-dataset, five-seed GPU evidence under the shared evaluator; it remains distinct from RAP++.",
            "- GEM's remapped wrapper removes the earlier relaxed-to-row coordinate artifact; the seed0 all-dataset diagnostic remains much weaker than the primary baselines.",
            "- RAP++ official and PrivMRF official are reported as original-protocol reproduced baselines because their native public-code metrics differ from the SAGE row-level evaluator.",
            "- RAP++ marginal-only path checks are not official full RAP++ evidence; do not present them as a main same-protocol baseline without a semantic conversion to the original halfspace/task-target interface.",
            "- PrivBayes/DataSynth, DPMM PrivBayes, unofficial PrivSyn, and PrivMRF GPU are now explicit appendix audit rows rather than omitted baselines.",
            "- Current main SAGE/QDTE results use `privacy.measurement_mode=static_all`; do not describe these tables as full adaptive SAGE-Select experiments.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write LaTeX-ready paper result tables.")
    parser.add_argument("--results-dir", type=Path, default=EXTERNAL_RESULTS)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--gsd-main", choices=["200k", "1m-fulln"], default="1m-fulln")
    args = parser.parse_args()

    table_dir = args.package_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "paper_tables_draft.tex"
    table_path.write_text(build_latex(args.results_dir, gsd_main=args.gsd_main), encoding="utf-8")

    docs_name = "PAPER_TABLES_DRAFT_EN.md" if args.gsd_main == "200k" else "PAPER_TABLES_DRAFT_HIGHPOWER_GSD_EN.md"
    docs_path = args.docs_dir / docs_name
    docs_path.write_text(build_markdown(args.results_dir, table_path, gsd_main=args.gsd_main), encoding="utf-8")

    print(f"Wrote {table_path}")
    print(f"Wrote {docs_path}")


if __name__ == "__main__":
    main()
