from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_original_protocol_baselines.py"
    spec = importlib.util.spec_from_file_location("audit_original_protocol_baselines", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _minimal_results_and_runs(tmp_path: Path):
    mod = _load_module()
    results_root = tmp_path / "external_results"
    runs_root = tmp_path / "external_runs"

    rappp_rows: list[dict[str, object]] = []
    for state in mod.RAPPP_STATES:
        for target in mod.RAPPP_TARGETS:
            for seed in mod.SEEDS:
                run_dir = runs_root / "rappp_official_paper_grid" / f"acs_{state}_{target}" / "eps1p0" / f"seed{seed}"
                metrics = run_dir / "paper_metrics" / "rappp_paper_metrics.json"
                metrics.parent.mkdir(parents=True, exist_ok=True)
                metrics.write_text("{}")
                _write_json(
                    run_dir / "run_metadata.json",
                    {
                        "method": "rappp_official_paper_grid",
                        "state": state,
                        "target": target,
                        "seed": seed,
                        "upstream_epsilon": 1.0,
                        "k": 2,
                        "num_random_projections": 200000,
                        "top_q": 5,
                        "dp_select_epochs": 50,
                        "status": "completed",
                        "jax_device_probe": {"jax_device_platforms": ["gpu"], "jax_devices": ["cuda:0"]}
                        if seed != 0
                        else {},
                    },
                )
                rappp_rows.append(
                    {
                        "state": state,
                        "target": target,
                        "seed": seed,
                        "dataset_name": f"acs_{state}_{target}",
                        "upstream_epsilon": 1.0,
                        "num_random_projections": 200000,
                        "top_q": 5,
                        "dp_select_epochs": 50,
                        "run_status": "completed",
                        "jax_device_platforms": "gpu" if seed != 0 else "",
                        "jax_devices": "cuda:0" if seed != 0 else "",
                        "metrics_path": str(metrics),
                        "run_dir": str(run_dir),
                    }
                )
    _write_csv(results_root / mod.RAPPP_CSV, rappp_rows)

    privmrf_rows: list[dict[str, object]] = []
    for dataset in mod.PRIVMRF_DATASETS:
        run_name = (
            "official_nltcs_tvd_epsgrid_m300_20260706"
            if dataset == "nltcs"
            else "official_acs_adult_br2000_tvd_epsgrid_m300_20260706"
        )
        run_dir = runs_root / "privmrf_official" / run_name
        result_path = run_dir / f"{run_name}_TVD.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}")
        for epsilon in mod.PRIVMRF_EPSILONS:
            for way in mod.PRIVMRF_WAYS:
                privmrf_rows.append(
                    {
                        "method": "PrivMRF official",
                        "task": "TVD",
                        "dataset": dataset,
                        "epsilon": epsilon,
                        "way": way,
                        "tvd_mean": 0.01,
                        "tvd_values": 0.01,
                        "repeat": 1,
                        "marginal_num": 300,
                        "run_dir": str(run_dir),
                        "result_path": str(result_path),
                    }
                )
    _write_csv(results_root / mod.PRIVMRF_CSV, privmrf_rows)

    _write_json(
        runs_root / "privmrf_official" / "official_nltcs_tvd_epsgrid_m300_20260706" / "run_metadata.json",
        {
            "method": "privmrf_official",
            "task": "TVD",
            "status": "completed",
            "repeat": 1,
            "marginal_num": 300,
            "conda_env": "baseline_privmrf",
            "datasets": ["nltcs"],
            "epsilons": list(mod.PRIVMRF_EPSILONS),
            "copied_result_path": str(
                runs_root
                / "privmrf_official"
                / "official_nltcs_tvd_epsgrid_m300_20260706"
                / "official_nltcs_tvd_epsgrid_m300_20260706_TVD.json"
            ),
        },
    )
    _write_json(
        runs_root / "privmrf_official" / "official_acs_adult_br2000_tvd_epsgrid_m300_20260706" / "run_metadata.json",
        {
            "method": "privmrf_official",
            "task": "TVD",
            "status": "completed",
            "repeat": 1,
            "marginal_num": 300,
            "conda_env": "baseline_privmrf",
            "datasets": ["acs", "adult", "br2000"],
            "epsilons": list(mod.PRIVMRF_EPSILONS),
            "copied_result_path": str(
                runs_root
                / "privmrf_official"
                / "official_acs_adult_br2000_tvd_epsgrid_m300_20260706"
                / "official_acs_adult_br2000_tvd_epsgrid_m300_20260706_TVD.json"
            ),
        },
    )
    return mod, results_root, runs_root


def test_original_protocol_audit_accepts_complete_fixture(tmp_path: Path) -> None:
    mod, results_root, runs_root = _minimal_results_and_runs(tmp_path)

    assert mod.audit(results_root, runs_root) == []

    rows = mod.summary_rows(results_root, runs_root, [])
    output_csv = tmp_path / "audit.csv"
    output_md = tmp_path / "audit.md"
    mod.write_csv(rows, output_csv)
    mod.write_markdown(rows, output_md)

    assert "RAP++ official ACS grid" in output_csv.read_text()
    assert "PrivMRF official TVD" in output_md.read_text()
    assert "125" in output_csv.read_text()
    assert "72" in output_csv.read_text()


def test_original_protocol_audit_rejects_weakened_rappp_projection_count(tmp_path: Path) -> None:
    mod, results_root, runs_root = _minimal_results_and_runs(tmp_path)
    path = results_root / mod.RAPPP_CSV
    rows = list(csv.DictReader(path.open()))
    rows[0]["num_random_projections"] = "1000"
    _write_csv(path, rows)

    errors = mod.audit(results_root, runs_root)

    assert any("num_random_projections" in error for error in errors)


def test_original_protocol_audit_rejects_weakened_privmrf_marginal_num(tmp_path: Path) -> None:
    mod, results_root, runs_root = _minimal_results_and_runs(tmp_path)
    path = results_root / mod.PRIVMRF_CSV
    rows = list(csv.DictReader(path.open()))
    rows[-1]["marginal_num"] = "30"
    _write_csv(path, rows)

    errors = mod.audit(results_root, runs_root)

    assert any("marginal_num" in error for error in errors)
