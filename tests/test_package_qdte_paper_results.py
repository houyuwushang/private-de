from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "package_qdte_paper_results.py"
    )
    spec = importlib.util.spec_from_file_location("package_qdte_paper_results", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_stage(
    path: Path,
    *,
    runtime: float,
    final_loss: float,
    points: list[tuple[float, float]],
    source: Path | None = None,
    wall_column: str = "wall_time_seconds",
) -> None:
    path.mkdir(parents=True)
    _write_json(path / "run_status.json", {"status": "completed"})
    metadata = {"source_run": str(source)} if source is not None else {}
    _write_json(path / "run_metadata.json", metadata)
    _write_json(
        path / "metrics_final.json",
        {"runtime_seconds": runtime, "final_measured_loss": final_loss},
    )
    _write_json(path / "external_evaluation.json", {})
    (path / "synthetic_encoded.npy").write_bytes(b"synthetic")
    with (path / "metrics_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[wall_column, "measured_loss"])
        writer.writeheader()
        for wall, loss in points:
            writer.writerow({wall_column: wall, "measured_loss": loss})


def test_chain_curve_supports_legacy_and_current_wall_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "BASELINE_ROOT", tmp_path)
    base = tmp_path / "base"
    phase = tmp_path / "phase"
    _write_stage(
        base,
        runtime=10.0,
        final_loss=70.0,
        points=[(2.0, 90.0), (8.0, 75.0)],
        wall_column="wall_time",
    )
    _write_stage(
        phase,
        runtime=5.0,
        final_loss=50.0,
        points=[(1.0, 65.0), (4.0, 55.0)],
        source=base,
    )
    tracker = module.SourceTracker()

    curve, total, chain = module._chain_curve(phase, tracker, "test")

    assert chain == [base.resolve(), phase.resolve()]
    assert total == pytest.approx(15.0)
    assert any(point["time_seconds"] == pytest.approx(8.0) for point in curve)
    assert any(point["time_seconds"] == pytest.approx(11.0) for point in curve)
    first_below_68 = min(
        float(point["time_seconds"]) for point in curve if float(point["loss"]) < 68.0
    )
    assert first_below_68 == pytest.approx(11.0)


def test_relative_change_uses_candidate_over_baseline() -> None:
    module = _load_module()
    assert module._relative_change_percent(9.0, 10.0) == pytest.approx(-10.0)


def test_table_renderer_emits_label_and_rows() -> None:
    module = _load_module()
    text = module._table(
        "Caption", "tab:test", "lr", ["Name", "Value"], [["QDTE", "1.0"]]
    )
    assert r"\label{tab:test}" in text
    assert "QDTE & 1.0" in text
