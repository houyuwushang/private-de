from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_paper_result_state.py"
    spec = importlib.util.spec_from_file_location("audit_paper_result_state", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_fixture(tmp_path: Path, *, gsd_acs_avg_tvd: float = 0.93) -> tuple[Path, Path]:
    mod = _load_audit_module()
    results_dir = tmp_path / "results"
    runs_root = tmp_path / "runs"
    datasets = mod.DATASETS
    regular_fields = ["dataset", "method", "method_slug", "n", "MAE_mean", "RMSE_mean", "AvgTVD_mean", "MaxErr_mean", "MaxTVD_mean"]
    rap_fields = [
        "dataset",
        "method",
        "seed_count",
        "full_true_mae_mean",
        "full_true_rmse_mean",
        "full_true_avg_tvd_mean",
        "full_true_max_error_mean",
        "full_true_max_tvd_mean",
    ]

    sage_rows = [
        {
            "dataset": dataset,
            "method": "SAGE",
            "method_slug": "sage",
            "n": 5,
            "MAE_mean": 1.0,
            "RMSE_mean": 1.0,
            "AvgTVD_mean": 1.0,
            "MaxErr_mean": 1.0,
            "MaxTVD_mean": 1.0,
        }
        for dataset in datasets
    ]
    _write_csv(results_dir / mod.SUMMARY_FILES["sage"], regular_fields, sage_rows)
    for name, method, slug, value in [
        ("aim", "Private-PGM AIM", "private_pgm_aim", 2.0),
        ("mst", "Private-PGM MST", "private_pgm_mst", 3.0),
    ]:
        _write_csv(
            results_dir / mod.SUMMARY_FILES[name],
            regular_fields,
            [
                {
                    "dataset": dataset,
                    "method": method,
                    "method_slug": slug,
                    "n": 5,
                    "MAE_mean": value,
                    "RMSE_mean": value,
                    "AvgTVD_mean": value,
                    "MaxErr_mean": value,
                    "MaxTVD_mean": value,
                }
                for dataset in datasets
            ],
        )
    _write_csv(
        results_dir / mod.SUMMARY_FILES["rap"],
        rap_fields,
        [
            {
                "dataset": dataset,
                "method": "rap_softmax",
                "seed_count": 5,
                "full_true_mae_mean": 4.0,
                "full_true_rmse_mean": 4.0,
                "full_true_avg_tvd_mean": 4.0,
                "full_true_max_error_mean": 4.0,
                "full_true_max_tvd_mean": 4.0,
            }
            for dataset in datasets
        ],
    )

    gsd_rows = []
    for dataset in datasets:
        row = {
            "dataset": dataset,
            "method": "Private-GSD GPU 1M/full-N",
            "method_slug": "private_gsd_gpu_1m_fulln_audit",
            "n": 5,
            "MAE_mean": 2.0,
            "RMSE_mean": 2.0,
            "AvgTVD_mean": 2.0,
            "MaxErr_mean": 2.0,
            "MaxTVD_mean": 2.0,
        }
        if dataset == "acs_sage_strong":
            row["AvgTVD_mean"] = gsd_acs_avg_tvd
            row["MaxTVD_mean"] = 0.9
        if dataset == "br2000_sage_strong":
            row["AvgTVD_mean"] = 0.9
            row["MaxTVD_mean"] = 0.8
        gsd_rows.append(row)
    _write_csv(results_dir / mod.SUMMARY_FILES["gsd_1m"], regular_fields, gsd_rows)

    metadata_dir = runs_root / "private_gsd_gpu_1m_fulln_audit" / "adult_sage_strong" / "rho1p0" / "seed0"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": "private_gsd_gpu_1m_fulln_audit",
                "notes": {
                    "jax_backend": "gpu",
                    "jax_devices": ["gpu:0"],
                    "num_generations": 1000000,
                    "stop_early_min_generation": 1000000,
                    "tree_query_depth": 2,
                },
            }
        )
    )
    config_dir = runs_root / "sage" / "adult_sage_strong" / "rho1p0" / "seed0"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config_resolved.yaml").write_text(
        "\n".join(
            [
                "measurement_mode: static_all",
                "objective_weighting: variance",
                "score_backend: dense_gpu",
                "transport_mode: atom_flow",
                "max_iters: 5000",
                "project_partitions: true",
                "clip_nonpartition: true",
                "prefix_monotonicity: true",
                "projection:",
                "  consistency:",
                "    enabled: false",
            ]
        )
        + "\n"
    )
    return results_dir, runs_root


def test_print_ratio_block_accepts_mixed_summary_schemas(tmp_path: Path, capsys) -> None:
    mod = _load_audit_module()
    datasets = mod.DATASETS
    regular_fields = ["dataset", "MAE_mean", "RMSE_mean", "AvgTVD_mean", "MaxErr_mean", "MaxTVD_mean"]
    rap_fields = [
        "dataset",
        "full_true_mae_mean",
        "full_true_rmse_mean",
        "full_true_avg_tvd_mean",
        "full_true_max_error_mean",
        "full_true_max_tvd_mean",
    ]

    _write_csv(
        tmp_path / mod.SUMMARY_FILES["sage"],
        regular_fields,
        [{field: (1.0 if field != "dataset" else dataset) for field in regular_fields} for dataset in datasets],
    )
    for method in ["aim", "mst", "gsd_1m"]:
        _write_csv(
            tmp_path / mod.SUMMARY_FILES[method],
            regular_fields,
            [{field: (2.0 if field != "dataset" else dataset) for field in regular_fields} for dataset in datasets],
        )
    _write_csv(
        tmp_path / mod.SUMMARY_FILES["rap"],
        rap_fields,
        [{field: (3.0 if field != "dataset" else dataset) for field in rap_fields} for dataset in datasets],
    )

    mod.print_ratio_block(tmp_path)

    out = capsys.readouterr().out
    assert "### rap" in out
    assert "MAE=3.00" in out
    assert out.count("SAGE wins: 20/20") == 4


def test_grep_consistency_enabled_reads_nested_value(tmp_path: Path) -> None:
    mod = _load_audit_module()
    config = tmp_path / "config_resolved.yaml"
    config.write_text(
        "\n".join(
            [
                "projection:",
                "  project_partitions: true",
                "  consistency:",
                "    enabled: false",
                "    method: local_marginal_ipf",
                "qdte:",
                "  enabled: true",
            ]
        )
        + "\n"
    )

    assert mod.grep_consistency_enabled(config) == "false"


def test_audit_result_state_accepts_expected_primary_pattern(tmp_path: Path) -> None:
    mod = _load_audit_module()
    results_dir, runs_root = _write_summary_fixture(tmp_path)

    assert mod.audit_result_state(results_dir, runs_root) == []


def test_audit_result_state_rejects_unexpected_gsd_win_pattern(tmp_path: Path) -> None:
    mod = _load_audit_module()
    results_dir, runs_root = _write_summary_fixture(tmp_path, gsd_acs_avg_tvd=1.1)

    errors = mod.audit_result_state(results_dir, runs_root)

    assert any("GSD-winning cells" in error for error in errors)
