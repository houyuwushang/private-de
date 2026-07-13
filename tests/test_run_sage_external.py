from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
import json
from pathlib import Path

import numpy as np


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sage_external.py"
    spec = importlib.util.spec_from_file_location("run_sage_external", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_external_sage_config_enforces_dp_boundary(tmp_path: Path) -> None:
    runner = _load_runner()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  dataset_name: original
  input_csv: old.csv
  output_dir: old_out
  seed: 123
privacy:
  mode: oracle
  rho_total: 99.0
  delta: 0.5
init:
  N_syn: 7
qdte:
  max_iters: 77
evaluation:
  compute_true_query_error: true
  compute_heldout_query_error: true
  downstream_ml: true
  save_synthetic_csv: true
runtime:
  xla_preallocate: true
  log_measurement_groups: true
""".lstrip(),
        encoding="utf-8",
    )
    (input_dir / "metadata.json").write_text(f'{{"config_path": "{config_path}"}}', encoding="utf-8")

    args = Namespace(
        dataset="toy_external",
        input_dir=input_dir,
        output_dir=output_dir,
        config=None,
        override=[],
        n_syn="same_as_real",
        seed=5,
        rho_total=1.25,
        delta=1e-9,
        max_iters=11,
        save_synthetic_csv=False,
        xla_preallocate=False,
    )

    config, resolved_path = runner._prepare_config(args, n_real=4)

    assert resolved_path == config_path
    assert config["run"]["dataset_name"] == "toy_external"
    assert config["run"]["input_csv"] == str(input_dir / "raw.csv")
    assert config["run"]["output_dir"] == str(output_dir)
    assert config["run"]["seed"] == 5
    assert config["privacy"]["mode"] == "dp"
    assert config["privacy"]["rho_total"] == 1.25
    assert config["privacy"]["delta"] == 1e-9
    assert config["init"]["N_syn"] == "same_as_real"
    assert config["qdte"]["max_iters"] == 11

    assert config["evaluation"]["compute_true_query_error"] is False
    assert config["evaluation"]["compute_heldout_query_error"] is False
    assert config["evaluation"]["downstream_ml"] is False
    assert config["evaluation"]["save_synthetic_csv"] is False
    assert config["runtime"]["xla_preallocate"] is False
    assert config["runtime"]["log_measurement_groups"] is False


def test_config_overlay_recursively_replaces_only_declared_fields() -> None:
    runner = _load_runner()
    base = {
        "qdte": {"total_candidates_per_iter": 4096, "max_iters": 5000},
        "runtime": {"use_pmap": True},
    }
    overlay = {"qdte": {"total_candidates_per_iter": 3584, "structured_swap_enabled": True}}

    merged = runner._merge_config_overlay(base, overlay)

    assert merged == {
        "qdte": {
            "total_candidates_per_iter": 3584,
            "max_iters": 5000,
            "structured_swap_enabled": True,
        },
        "runtime": {"use_pmap": True},
    }
    assert base["qdte"]["total_candidates_per_iter"] == 4096


def test_external_sage_protocol_manifest_hashes_dp_only_generation_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    np.save(input_dir / "real_encoded.npy", np.zeros((4, 2), dtype=np.int64))
    (input_dir / "raw.csv").write_text("a,b\n0,0\n0,0\n0,0\n0,0\n", encoding="utf-8")
    (input_dir / "schema.json").write_text(
        json.dumps(
            {
                "columns": [
                    {"name": "a", "cardinality": 2},
                    {"name": "b", "cardinality": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "metadata.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  device: cpu
privacy:
  mode: dp
init:
  N_syn: same_as_real
qdte:
  objective_weighting: variance
  transport_mode: atom_flow
  max_iters: 5
evaluation: {}
runtime: {}
""".lstrip(),
        encoding="utf-8",
    )

    def fake_run(config):
        out = Path(config["run"]["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "synthetic_encoded.npy", np.zeros((4, 2), dtype=np.int64))
        for name in (
            "config_resolved.yaml",
            "schema.json",
            "queries.json",
            "measurements.json",
            "metrics_final.json",
            "runtime.json",
        ):
            (out / name).write_text("{}", encoding="utf-8")
        (out / "metrics_timeseries.csv").write_text("iteration,loss\n", encoding="utf-8")
        (out / "logs.txt").write_text("done\n", encoding="utf-8")
        return {"epsilon_delta": 1.0, "rho_spent": 0.1, "score_backend": "cpu"}

    monkeypatch.setattr(runner, "run_qdte", fake_run)
    args = Namespace(
        method="sage",
        method_label="SAGE-QDTE-Static",
        dataset="toy",
        input_dir=input_dir,
        output_dir=output_dir,
        rho_total=0.1,
        delta=1e-9,
        seed=0,
        n_syn="same_as_real",
        config=config_path,
        overlay=[],
        override=[],
        max_iters=5,
        save_synthetic_csv=False,
        xla_preallocate=False,
        protocol_id="protocol-v1",
    )

    runner.run(args)

    manifest = json.loads((output_dir / "sage_static_manifest.json").read_text())
    assert manifest["protocol_id"] == "protocol-v1"
    assert manifest["paper_evidence_qualified"] is True
    assert manifest["checks"]["true_metrics_disabled_during_generation"] is True
    assert manifest["checks"]["objective_is_variance"] is True
    assert manifest["checks"]["transport_is_atom_flow"] is True
    assert manifest["artifacts"]["synthetic"]["sha256"]
