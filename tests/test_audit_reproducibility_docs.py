from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_reproducibility_docs.py"
    spec = importlib.util.spec_from_file_location("audit_reproducibility_docs", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reproducibility_docs_audit_passes_current_repo() -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]

    assert mod.audit(root) == []


def test_reproducibility_docs_audit_rejects_missing_required_snippet(tmp_path: Path) -> None:
    mod = _load_module()
    for check in mod.CHECKS:
        path = tmp_path / check.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(check.snippets) + "\n", encoding="utf-8")
    (tmp_path / "docs" / "REPRODUCIBILITY.md").write_text(
        "missing the public smoke command\n",
        encoding="utf-8",
    )

    errors = mod.audit(tmp_path)

    assert any("docs/REPRODUCIBILITY.md" in error for error in errors)
    assert any("scripts/smoke_qdte.py" in error for error in errors)
