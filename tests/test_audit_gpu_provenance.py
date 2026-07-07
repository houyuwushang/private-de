from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_gpu_provenance.py"
    spec = importlib.util.spec_from_file_location("audit_gpu_provenance", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_audit_gpu_provenance_accepts_gpu_runs_and_cpu_native_baselines(tmp_path: Path) -> None:
    mod = _load_module()
    runs_root = tmp_path / "external_runs"
    dataset = "adult_sage_strong"

    sage_dir = runs_root / "sage" / dataset / "rho1p0" / "seed0"
    _write_json(
        sage_dir / "run_metadata.json",
        {
            "status": "completed",
            "runtime_seconds": 1.0,
            "conda_env": "qdte",
            "method": "sage",
            "rho_total": 1.0,
            "n_real": 100,
            "n_synthetic": 100,
            "command": "run_sage_external.py --max-iters 5000",
            "notes": {"max_iters": 5000},
        },
    )
    _write_json(sage_dir / "runtime.json", {"gpu_devices": ["cuda:0"], "score_backend": "dense_gpu"})
    _write_json(sage_dir / "metrics_final.json", {"gpu_device_count": 1, "score_backend": "dense_gpu"})
    (sage_dir / "config_resolved.yaml").write_text("run:\n  device: gpu\nqdte:\n  score_backend: dense_gpu\n")

    gsd_dir = runs_root / "private_gsd_gpu_1m_fulln_audit" / dataset / "rho1p0" / "seed0"
    _write_json(
        gsd_dir / "run_metadata.json",
        {
            "status": "completed",
            "conda_env": "gsd",
            "method": "private_gsd_gpu_1m_fulln_audit",
            "rho_total": 1.0,
            "n_real": 100,
            "n_synthetic": 100,
            "notes": {
                "jax_backend": "gpu",
                "jax_devices": ["gpu:0"],
                "n_prime": 100,
                "num_generations": 1_000_000,
                "stop_early_min_generation": 1_000_000,
                "tree_query_depth": 2,
                "early_stop_threshold": 0.01,
                "genetic_operators": ["mutate", "swap", "cross"],
            },
        },
    )

    rap_dir = runs_root / "rap" / dataset / "rho1p0" / "seed0_T30K30M445I1000"
    _write_json(
        rap_dir / "run_metadata.json",
        {
            "status": "completed",
            "conda_env": "tddpm",
            "method": "rap_softmax",
            "rho_total": 1.0,
            "model_rows": 1000,
            "n_real": 100,
            "n_synthetic": 100,
            "torch_cuda_available": True,
            "torch_device": "cuda:0",
            "notes": {"T": 30, "K": 30, "num_marginals": 445, "max_iters": 1000, "decode_mode": "sample"},
        },
    )

    aim_dir = runs_root / "private_pgm_aim" / dataset / "rho1p0" / "seed0"
    _write_json(
        aim_dir / "run_metadata.json",
        {
            "status": "completed",
            "conda_env": "baseline_mbi",
            "method": "private_pgm_aim",
            "rho_total": 1.0,
            "n_real": 100,
            "n_synthetic": 100,
            "notes": {"rounds": 220, "max_iters": 100, "max_model_size": 80.0},
        },
    )
    mst_dir = runs_root / "private_pgm_mst" / dataset / "rho1p0" / "seed0"
    _write_json(
        mst_dir / "run_metadata.json",
        {
            "status": "completed",
            "conda_env": "baseline_mbi",
            "method": "private_pgm_mst",
            "rho_total": 1.0,
            "n_real": 100,
            "n_synthetic": 100,
            "command": "run_private_pgm.py --method mst",
            "notes": {},
        },
    )

    for root in ("private_pgm_aim", "private_pgm_mst"):
        run_dir = runs_root / root / dataset / "rho1p0" / "seed0"
        assert (run_dir / "run_metadata.json").exists()

    rows = mod.audit(runs_root, (dataset,), (0,))

    assert len(rows) == 5
    assert not [row for row in rows if mod._is_failure(row)]
    by_method = {row["method"]: row for row in rows}
    assert by_method["sage"]["gpu_ok"] is True
    assert by_method["private_gsd_gpu_1m_fulln_audit"]["gpu_ok"] is True
    assert by_method["rap_softmax"]["gpu_ok"] is True
    assert by_method["rap_softmax"]["protocol_ok"] is True
    assert by_method["private_pgm_aim"]["gpu_ok"] == "not_required"
    assert by_method["private_pgm_mst"]["gpu_ok"] == "not_required"


def test_audit_gpu_provenance_fails_required_gpu_without_evidence(tmp_path: Path) -> None:
    mod = _load_module()
    runs_root = tmp_path / "external_runs"
    dataset = "adult_sage_strong"
    run_dir = runs_root / "private_gsd_gpu_1m_fulln_audit" / dataset / "rho1p0" / "seed0"
    _write_json(
        run_dir / "run_metadata.json",
        {
            "status": "completed",
            "conda_env": "gsd",
            "method": "private_gsd_gpu_1m_fulln_audit",
            "rho_total": 1.0,
            "n_real": 100,
            "n_synthetic": 100,
            "notes": {
                "n_prime": 100,
                "num_generations": 1_000_000,
                "stop_early_min_generation": 1_000_000,
                "tree_query_depth": 2,
                "early_stop_threshold": 0.01,
                "genetic_operators": ["mutate", "swap", "cross"],
            },
        },
    )

    row = mod._row_for_run(
        runs_root,
        next(spec for spec in mod.METHODS if spec.method == "private_gsd_gpu_1m_fulln_audit"),
        dataset,
        0,
    )

    assert row["gpu_ok"] is False
    assert mod._is_failure(row)


def test_audit_gpu_provenance_fails_weakened_private_gsd_protocol(tmp_path: Path) -> None:
    mod = _load_module()
    runs_root = tmp_path / "external_runs"
    dataset = "adult_sage_strong"
    run_dir = runs_root / "private_gsd_gpu_1m_fulln_audit" / dataset / "rho1p0" / "seed0"
    _write_json(
        run_dir / "run_metadata.json",
        {
            "status": "completed",
            "conda_env": "gsd",
            "method": "private_gsd_gpu_1m_fulln_audit",
            "rho_total": 1.0,
            "n_real": 100,
            "n_synthetic": 100,
            "notes": {
                "jax_backend": "gpu",
                "jax_devices": ["gpu:0"],
                "n_prime": 100,
                "num_generations": 200_000,
                "stop_early_min_generation": 200_000,
                "tree_query_depth": 2,
                "early_stop_threshold": 0.01,
                "genetic_operators": ["mutate", "swap", "cross"],
            },
        },
    )

    row = mod._row_for_run(
        runs_root,
        next(spec for spec in mod.METHODS if spec.method == "private_gsd_gpu_1m_fulln_audit"),
        dataset,
        0,
    )

    assert row["gpu_ok"] is True
    assert row["protocol_ok"] is False
    assert "FAIL:gsd_num_generations_not_1000000" in row["protocol_evidence"]
    assert mod._is_failure(row)
