from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "verify_qdte_paper_package.py"
    )
    spec = importlib.util.spec_from_file_location("verify_qdte_paper_package", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_qdte_paper_package_passes() -> None:
    module = _load_module()
    if not module.DEFAULT_PACKAGE.exists():
        pytest.skip("local QDTE paper package is not available")

    assert module.verify_package(module.DEFAULT_PACKAGE) == []


def test_fact_audit_rejects_universal_gsd_claim() -> None:
    module = _load_module()
    metadata = {
        "status": "complete",
        "facts": {
            "QDTE-C1": {
                "RAP softmax": {"qdte_wins": 20, "cells": 20},
                "Private-PGM AIM": {"qdte_wins": 20, "cells": 20},
                "Private-PGM MST": {"qdte_wins": 20, "cells": 20},
                "Private-GSD GPU 1M/full-N": {"qdte_wins": 16, "cells": 20},
            },
            "QDTE-C4": {
                "datasets": 3,
                "target_loss_wins": 3,
                "offline_metric_wins": 15,
                "offline_metric_cells": 15,
            },
            "QDTE-C5": {
                "QDTE-Structured-v2": {
                    "measured_loss_wins": 4,
                    "replacement_gate_passed": False,
                },
                "QDTE-Structured-SA": {
                    "measured_loss_wins": 4,
                    "replacement_gate_passed": False,
                },
            },
            "QDTE-C6": {
                "datasets": 4,
                "mae_wins": 4,
                "rmse_wins": 3,
                "combined_wins": 7,
                "combined_cells": 8,
                "largest_l2_regression_percent": 0.6172295822633345,
                "gate_passed": True,
            },
            "QDTE-C7": {
                "target_change_percent": -1.2,
                "fit_change_percent": -1.5,
                "final_change_percent": 0.3,
                "teacher_loss_ratio_vs_baseline": 0.05,
            },
        },
    }
    metadata["facts"]["QDTE-C4"]["offline_metric_wins"] = 14

    errors = module.audit_facts(metadata)

    assert any("QDTE-C4 mismatch" in error for error in errors)


def test_private_path_audit_rejects_home_path(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "bad.json").write_text(
        '{"path": "/home/' + "qian" + 'qiu/private"}', encoding="utf-8"
    )

    errors = module.audit_no_private_paths(tmp_path)

    assert any("private/local path" in error for error in errors)
