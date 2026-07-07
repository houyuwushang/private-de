from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


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
