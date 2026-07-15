from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scripts import run_static_ice_wp9_panel as panel


INPUT_ROOT = Path("external_inputs")


def test_wp9_panel_plan_freezes_complete_matrix_and_evaluator() -> None:
    plan = panel.build_panel_plan(
        input_root=INPUT_ROOT,
        config_root=panel.ROOT / "configs",
        batch_size=8192,
    )

    assert plan["protocol_id"] == panel.PROTOCOL_ID
    assert plan["method_id"] == panel.METHOD_ID
    assert plan["matrix"]["num_cells"] == 24
    assert len(plan["cells"]) == 24
    assert plan["evaluator"]["sha256"]
    assert plan["cell_runner"]["sha256"]
    assert plan["true_utility_evaluated"] is False
    assert {
        (cell["dataset"], cell["epsilon"], cell["seed"])
        for cell in plan["cells"]
    } == {
        (dataset, epsilon, seed)
        for dataset in panel.DATASETS
        for epsilon in panel.EPSILONS
        for seed in panel.SEEDS
    }


def test_wp9_panel_refuses_plan_drift() -> None:
    plan = {"protocol": "v1"}
    panel._same_plan(dict(plan), plan)
    with pytest.raises(RuntimeError, match="changed"):
        panel._same_plan({"protocol": "v0"}, plan)


def test_wp9_panel_source_has_no_true_evaluator() -> None:
    source = inspect.getsource(panel)
    assert "evaluate_external_synthetic" not in source
    assert '"true_utility_evaluated": False' in source
