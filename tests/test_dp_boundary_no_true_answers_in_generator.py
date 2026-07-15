from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reprojection_cli_does_not_enable_true_answer_metrics_by_default(monkeypatch) -> None:
    module = _load_script("reproject_measurements")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reproject_measurements.py",
            "--measurement",
            "outputs/base",
            "--output-dir",
            "outputs/reprojected",
            "--total",
            "4",
        ],
    )

    args = module.parse_args()

    assert args.compute_target_metrics is False


def test_external_sage_runner_keeps_true_answers_out_of_active_run(tmp_path: Path) -> None:
    runner = _load_script("run_sage_external")
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
privacy:
  mode: oracle
evaluation:
  compute_true_query_error: true
  compute_heldout_query_error: true
  downstream_ml: true
runtime:
  log_measurement_groups: true
""".lstrip(),
        encoding="utf-8",
    )
    (input_dir / "metadata.json").write_text(f'{{"config_path": "{config_path}"}}', encoding="utf-8")

    args = type(
        "Args",
        (),
        {
            "dataset": "toy",
            "input_dir": input_dir,
            "output_dir": output_dir,
            "config": None,
            "override": [],
            "n_syn": "same_as_real",
            "seed": 0,
            "rho_total": 1.0,
            "delta": 1.0e-9,
            "max_iters": 1,
            "save_synthetic_csv": False,
            "xla_preallocate": False,
        },
    )()

    config, _ = runner._prepare_config(args, n_real=4)

    assert config["privacy"]["mode"] == "dp"
    assert config["privacy"]["public_n_rows"] == 4
    assert config["init"]["N_syn"] == 4
    assert config["evaluation"]["compute_true_query_error"] is False
    assert config["evaluation"]["compute_heldout_query_error"] is False
    assert config["evaluation"]["downstream_ml"] is False
    assert config["runtime"]["log_measurement_groups"] is False
