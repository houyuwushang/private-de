from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_qdte_paper_claim_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_qdte_paper_claim_matrix", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_claim_matrix_passes() -> None:
    module = _load_module()
    data = module.load_matrix(module.DEFAULT_MATRIX)
    markdown = module.DEFAULT_MARKDOWN.read_text(encoding="utf-8")

    assert module.audit_matrix(data, markdown) == []


def test_missing_forbidden_boundary_fails() -> None:
    module = _load_module()
    data = module.load_matrix(module.DEFAULT_MATRIX)
    markdown = module.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    data["claims"][0]["forbidden"] = []

    errors = module.audit_matrix(data, markdown)

    assert any("QDTE-C1: forbidden" in error for error in errors)


def test_unknown_claim_reference_fails() -> None:
    module = _load_module()
    data = module.load_matrix(module.DEFAULT_MATRIX)
    markdown = module.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    data["main_contributions"].append("QDTE-C99")

    errors = module.audit_matrix(data, markdown)

    assert any("unknown claims" in error for error in errors)
