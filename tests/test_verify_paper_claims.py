from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


DATASETS = [
    "adult_sage_strong",
    "acs_sage_strong",
    "br2000_sage_strong",
    "nltcs_sage_strong",
]
METRICS = ["MAE", "RMSE", "AvgTVD", "MaxErr", "MaxTVD"]
DATASET_LABELS = {
    "adult_sage_strong": "Adult",
    "acs_sage_strong": "ACS",
    "br2000_sage_strong": "BR2000",
    "nltcs_sage_strong": "NLTCS",
}


def _load_claim_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_paper_claims.py"
    spec = importlib.util.spec_from_file_location("verify_paper_claims", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _value(
    dataset: str,
    metric: str,
    default: float,
    overrides: dict[tuple[str, str], float] | None,
) -> float:
    if overrides is None:
        return default
    return overrides.get((dataset, metric), default)


def _summary_csv(
    method: str,
    method_slug: str,
    n_col: str = "n",
    *,
    default: float = 1.0,
    overrides: dict[tuple[str, str], float] | None = None,
) -> str:
    lines = [
        f"dataset,method,method_slug,{n_col},MAE_mean,RMSE_mean,AvgTVD_mean,MaxErr_mean,MaxTVD_mean"
    ]
    for dataset in DATASETS:
        values = [_value(dataset, metric, default, overrides) for metric in METRICS]
        lines.append(f"{dataset},{method},{method_slug},5," + ",".join(str(value) for value in values))
    return "\n".join(lines) + "\n"


def _rap_summary_csv(default: float = 2.0) -> str:
    lines = [
        "dataset,method,seed_count,full_true_mae_mean,full_true_rmse_mean,"
        "full_true_avg_tvd_mean,full_true_max_error_mean,full_true_max_tvd_mean"
    ]
    for dataset in DATASETS:
        lines.append(f"{dataset},rap_softmax,5,{default},{default},{default},{default},{default}")
    return "\n".join(lines) + "\n"


def _other_metric_values(
    default: float,
    overrides: dict[tuple[str, str], float] | None,
) -> dict[tuple[str, str], float]:
    return {
        (dataset, metric): _value(dataset, metric, default, overrides)
        for dataset in DATASETS
        for metric in METRICS
    }


def _comparison_csv(
    *,
    other_prefix: str,
    other_label: str,
    ratio_pattern: str,
    sage_wins_col: str,
    other_wins_col: str,
    include_dataset_label: bool,
    default_other: float,
    overrides: dict[tuple[str, str], float] | None = None,
    extra_rap_setting: bool = False,
) -> str:
    fields = ["dataset"]
    if include_dataset_label:
        fields.append("dataset_label")
    fields += ["sage_n", f"{other_prefix}_n"]
    if extra_rap_setting:
        fields.append("rap_setting")
    for metric in METRICS:
        fields += [f"sage_{metric}", f"{other_prefix}_{metric}", ratio_pattern.format(metric=metric)]
        if include_dataset_label:
            fields.append(f"winner_{metric}")
    fields += [sage_wins_col, other_wins_col]
    other_values = _other_metric_values(default_other, overrides)
    lines = [",".join(fields)]
    for dataset in DATASETS:
        row: dict[str, str] = {"dataset": dataset, "sage_n": "5", f"{other_prefix}_n": "5"}
        if include_dataset_label:
            row["dataset_label"] = DATASET_LABELS[dataset]
        if extra_rap_setting:
            row["rap_setting"] = "T30K30M445I1000"
        sage_wins = 0
        other_wins = 0
        for metric in METRICS:
            sage_value = 1.0
            other_value = other_values[(dataset, metric)]
            winner = "SAGE" if sage_value <= other_value else other_label
            row[f"sage_{metric}"] = str(sage_value)
            row[f"{other_prefix}_{metric}"] = str(other_value)
            row[ratio_pattern.format(metric=metric)] = str(other_value / sage_value)
            if include_dataset_label:
                row[f"winner_{metric}"] = winner
            if winner == "SAGE":
                sage_wins += 1
            else:
                other_wins += 1
        row[sage_wins_col] = str(sage_wins)
        row[other_wins_col] = str(other_wins)
        lines.append(",".join(row[field] for field in fields))
    return "\n".join(lines) + "\n"


def _traceability_csv() -> str:
    claim_ids = [
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
    lines = ["claim_id,claim,scope,evidence_artifacts,verifier_or_gate,status,machine_check"]
    for claim_id in claim_ids:
        lines.append(f"{claim_id},claim,scope,artifact,verifier,pass,checked")
    return "\n".join(lines) + "\n"


def _write_complete_package(root: Path) -> None:
    main_rows = []
    for dataset in ["Adult", "ACS", "BR2000", "NLTCS"]:
        for method in [
            "SAGE",
            "RAP softmax",
            "Private-GSD GPU 1M/full-N",
            "Private-PGM AIM",
            "Private-PGM MST",
        ]:
            main_rows.append(f"{dataset} & {method} & 1 & 1 & 1 & 1 & 1 \\\\")
    _write(
        root / "tables" / "paper_tables_draft.tex",
        "\\begin{table*}\n"
        "\\label{tab:main-results}\n"
        + "\n".join(main_rows)
        + "\n\\end{table*}\n",
    )
    _write(
        root / "tables" / "sage_all4_seed0to4_summary_rho1_20260706.csv",
        _summary_csv("SAGE", "sage"),
    )
    gsd_overrides = {
        ("acs_sage_strong", "AvgTVD"): 0.5,
        ("acs_sage_strong", "MaxTVD"): 0.5,
        ("br2000_sage_strong", "AvgTVD"): 0.5,
        ("br2000_sage_strong", "MaxTVD"): 0.5,
    }
    _write(
        root / "tables" / "private_gsd_gpu_1m_fulln_all4_seed0to4_summary_20260706.csv",
        _summary_csv(
            "Private-GSD GPU 1M/full-N",
            "private_gsd_gpu_1m_fulln_audit",
            default=2.0,
            overrides=gsd_overrides,
        ),
    )
    _write(
        root / "tables" / "private_pgm_aim_all4_seed0to4_summary_rho1_20260706.csv",
        _summary_csv("Private-PGM AIM", "private_pgm_aim", default=2.0),
    )
    _write(
        root / "tables" / "private_pgm_mst_all4_seed0to4_summary_rho1_20260706.csv",
        _summary_csv("Private-PGM MST", "private_pgm_mst", default=2.0),
    )
    _write(
        root / "tables" / "rap_sage_strong_stress_all4_seed0to4_summary_rho1_20260706.csv",
        _rap_summary_csv(),
    )
    _write(
        root / "tables" / "sage_vs_private_pgm_aim_seed0to4_20260706.csv",
        _comparison_csv(
            other_prefix="aim",
            other_label="Private-PGM AIM",
            ratio_pattern="{metric}_ratio_aim_over_sage",
            sage_wins_col="sage_wins",
            other_wins_col="aim_wins",
            include_dataset_label=False,
            default_other=2.0,
        ),
    )
    _write(
        root / "tables" / "sage_vs_private_pgm_mst_seed0to4_20260706.csv",
        _comparison_csv(
            other_prefix="mst",
            other_label="Private-PGM MST",
            ratio_pattern="{metric}_ratio_mst_over_sage",
            sage_wins_col="sage_wins",
            other_wins_col="mst_wins",
            include_dataset_label=False,
            default_other=2.0,
        ),
    )
    _write(
        root / "tables" / "sage_vs_rap_softmax_seed0to4_20260706.csv",
        _comparison_csv(
            other_prefix="rap",
            other_label="RAP softmax",
            ratio_pattern="rap_over_sage_{metric}",
            sage_wins_col="sage_metric_wins",
            other_wins_col="rap_metric_wins",
            include_dataset_label=True,
            default_other=2.0,
            extra_rap_setting=True,
        ),
    )
    _write(
        root / "tables" / "sage_vs_private_gsd_1m_fulln_seed0to4_20260706.csv",
        _comparison_csv(
            other_prefix="gsd",
            other_label="Private-GSD GPU 1M/full-N",
            ratio_pattern="gsd_over_sage_{metric}",
            sage_wins_col="sage_metric_wins",
            other_wins_col="gsd_metric_wins",
            include_dataset_label=True,
            default_other=2.0,
            overrides=gsd_overrides,
        ),
    )
    _write(
        root / "appendix" / "baseline_admission_audit_20260706.csv",
        "candidate,expected_tier,machine_status,dataset_count,seed_count,gpu_complete\n"
        "RAP softmax,primary,admitted_primary,4,5,True\n"
        "RAP++ official ACS grid,original_protocol,admitted_original_protocol,25,5,False\n"
        "PrivMRF official TVD,original_protocol,admitted_original_protocol,4,1,False\n",
    )
    _write(
        root / "appendix" / "original_protocol_baseline_audit_20260707.csv",
        "candidate,evidence_tier,audit_status,row_count,dataset_count,state_count,target_count,seed_count,gpu_evidence,protocol,source_csv,source_runs,error_count\n"
        "RAP++ official ACS grid,original_protocol,passed,125,25,5,5,5,seed_gpu_probe=1/2/3/4,epsilon=1.0;k=2;num_random_projections=200000;top_q=5;dp_select_epochs=50,rappp.csv,rappp_runs,0\n"
        "PrivMRF official TVD,original_protocol,passed,72,4,,,1,not_claimed,datasets=acs/adult/br2000/nltcs;epsilons=0.1/0.2/0.4/0.8/1.6/3.2;ways=3/4/5;repeat=1;marginal_num=300,privmrf.csv,privmrf_runs,0\n",
    )
    _write(root / "tables" / "paper_claim_traceability_20260707.csv", _traceability_csv())
    _write(root / "tables" / "paper_claim_traceability_20260707.md", "# Paper Claim Traceability\n")


def test_verify_paper_claims_accepts_current_claim_shape(tmp_path: Path) -> None:
    mod = _load_claim_verifier()
    _write_complete_package(tmp_path)

    result = mod.verify_claims(tmp_path)

    assert result.errors == []


def test_verify_paper_claims_rejects_changed_gsd_claim_count(tmp_path: Path) -> None:
    mod = _load_claim_verifier()
    _write_complete_package(tmp_path)
    _write(
        tmp_path / "tables" / "sage_vs_private_gsd_1m_fulln_seed0to4_20260706.csv",
        _comparison_csv(
            other_prefix="gsd",
            other_label="Private-GSD GPU 1M/full-N",
            ratio_pattern="gsd_over_sage_{metric}",
            sage_wins_col="sage_metric_wins",
            other_wins_col="gsd_metric_wins",
            include_dataset_label=True,
            default_other=2.0,
        ),
    )

    result = mod.verify_claims(tmp_path)

    assert any("gsd_metric_wins total is 0, expected 4" in error for error in result.errors)


def test_verify_paper_claims_rejects_stale_pairwise_values(tmp_path: Path) -> None:
    mod = _load_claim_verifier()
    _write_complete_package(tmp_path)
    _write(
        tmp_path / "tables" / "sage_vs_private_pgm_aim_seed0to4_20260706.csv",
        _comparison_csv(
            other_prefix="aim",
            other_label="Private-PGM AIM",
            ratio_pattern="{metric}_ratio_aim_over_sage",
            sage_wins_col="sage_wins",
            other_wins_col="aim_wins",
            include_dataset_label=False,
            default_other=3.0,
        ),
    )

    result = mod.verify_claims(tmp_path)

    assert any("aim_MAE is 3.0, expected summary 2.0" in error for error in result.errors)


def test_verify_paper_claims_rejects_demoted_rap_softmax(tmp_path: Path) -> None:
    mod = _load_claim_verifier()
    _write_complete_package(tmp_path)
    _write(
        tmp_path / "appendix" / "baseline_admission_audit_20260706.csv",
        "candidate,expected_tier,machine_status,dataset_count,seed_count,gpu_complete\n"
        "RAP softmax,primary,partial_evidence,4,5,True\n"
        "RAP++ official ACS grid,original_protocol,admitted_original_protocol,25,5,False\n"
        "PrivMRF official TVD,original_protocol,admitted_original_protocol,4,1,False\n",
    )

    result = mod.verify_claims(tmp_path)

    assert any("RAP softmax: machine_status is 'partial_evidence'" in error for error in result.errors)


def test_verify_paper_claims_rejects_weakened_original_protocol_audit(tmp_path: Path) -> None:
    mod = _load_claim_verifier()
    _write_complete_package(tmp_path)
    _write(
        tmp_path / "appendix" / "original_protocol_baseline_audit_20260707.csv",
        "candidate,evidence_tier,audit_status,row_count,dataset_count,state_count,target_count,seed_count,gpu_evidence,protocol,source_csv,source_runs,error_count\n"
        "RAP++ official ACS grid,original_protocol,passed,25,25,5,5,5,seed_gpu_probe=1/2/3/4,epsilon=1.0;k=2;num_random_projections=200000;top_q=5;dp_select_epochs=50,rappp.csv,rappp_runs,0\n"
        "PrivMRF official TVD,original_protocol,passed,72,4,,,1,not_claimed,datasets=acs/adult/br2000/nltcs;epsilons=0.1/0.2/0.4/0.8/1.6/3.2;ways=3/4/5;repeat=1;marginal_num=300,privmrf.csv,privmrf_runs,0\n",
    )

    result = mod.verify_claims(tmp_path)

    assert any("RAP++ official ACS grid: row_count is '25'" in error for error in result.errors)


def test_verify_paper_claims_rejects_failed_traceability_row(tmp_path: Path) -> None:
    mod = _load_claim_verifier()
    _write_complete_package(tmp_path)
    _write(
        tmp_path / "tables" / "paper_claim_traceability_20260707.csv",
        _traceability_csv().replace("sage-vs-gsd,claim,scope,artifact,verifier,pass", "sage-vs-gsd,claim,scope,artifact,verifier,fail"),
    )

    result = mod.verify_claims(tmp_path)

    assert any("paper_claim_traceability_20260707.csv:sage-vs-gsd" in error for error in result.errors)
