from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_qdte_fission_refit_external.py"
    spec = importlib.util.spec_from_file_location("run_qdte_fission_refit_external", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        dataset="toy",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        rho_total=1.0,
        delta=1e-9,
        seed=3,
        n_syn="same_as_real",
        config=tmp_path / "config.yaml",
        method_label="QDTE-Structured-FissionRefit-v2",
        base_overlay=[tmp_path / "standard.yaml", tmp_path / "guard.yaml"],
        fission_overlay=tmp_path / "fission.yaml",
        override=["workload.reuse_from_measurement=true"],
        xla_preallocate=False,
    )


def test_pipeline_uses_blind_selection_as_refit_budget(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    args = _args(tmp_path)
    observed: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        observed.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_metadata.json").write_text(
            json.dumps({"status": "completed", "runtime_seconds": 1.5}),
            encoding="utf-8",
        )
        if output_dir.name == "selection":
            (output_dir / "fission_checkpoints.json").write_text(
                json.dumps(
                    {
                        "selected_iteration": 700,
                        "terminal_iteration": 5000,
                        "selection_rule": "earliest_within_one_se",
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(runner, "_run_command", fake_run)
    metadata = runner.run_pipeline(args)

    assert len(observed) == 2
    assert observed[0].count("--overlay") == 3
    assert str(args.fission_overlay) in observed[0]
    assert observed[1].count("--overlay") == 2
    assert str(args.fission_overlay) not in observed[1]
    label_index = observed[1].index("--method-label")
    assert observed[1][label_index + 1] == "QDTE-Structured-FissionRefit-v2"
    max_index = observed[1].index("--max-iters")
    assert observed[1][max_index + 1] == "700"
    assert metadata["selected_iteration"] == 700
    assert metadata["status"] == "completed"


def test_selected_iteration_must_be_inside_trajectory(tmp_path: Path) -> None:
    runner = _load_runner()
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    (selection_dir / "fission_checkpoints.json").write_text(
        json.dumps(
            {
                "selected_iteration": 5001,
                "terminal_iteration": 5000,
                "selection_rule": "earliest_within_one_se",
            }
        ),
        encoding="utf-8",
    )

    try:
        runner._read_selected_iteration(selection_dir)
    except RuntimeError as exc:
        assert "invalid fission selection" in str(exc)
    else:
        raise AssertionError("invalid selected iteration was accepted")
