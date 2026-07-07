from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_admission_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_baseline_admission.py"
    spec = importlib.util.spec_from_file_location("audit_baseline_admission", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_baseline_root_uses_path_defaults_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SAGE_BASELINE_ROOT", str(tmp_path))

    mod = _load_admission_module()

    assert mod.BASELINE_ROOT == tmp_path
    assert mod.RESULTS_ROOT == tmp_path / "external_results"


def test_required_admission_errors_fail_only_required_tiers() -> None:
    mod = _load_admission_module()

    rows = [
        {
            "candidate": "RAP softmax",
            "expected_tier": "primary",
            "machine_status": "partial_evidence",
        },
        {
            "candidate": "RAP++ official ACS grid",
            "expected_tier": "original_protocol",
            "machine_status": "missing_or_failed",
        },
        {
            "candidate": "GEM",
            "expected_tier": "appendix_diagnosis",
            "machine_status": "partial_evidence",
        },
    ]

    errors = mod.required_admission_errors(rows)

    assert len(errors) == 2
    assert "RAP softmax" in errors[0]
    assert "admitted_primary" in errors[0]
    assert "RAP++ official ACS grid" in errors[1]
    assert "admitted_original_protocol" in errors[1]


def test_required_admission_errors_accept_required_admissions() -> None:
    mod = _load_admission_module()

    rows = [
        {
            "candidate": "RAP softmax",
            "expected_tier": "primary",
            "machine_status": "admitted_primary",
        },
        {
            "candidate": "PrivMRF official TVD",
            "expected_tier": "original_protocol",
            "machine_status": "admitted_original_protocol",
        },
    ]

    assert mod.required_admission_errors(rows) == []
