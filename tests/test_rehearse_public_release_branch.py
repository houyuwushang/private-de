from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rehearse_public_release_branch.py"
    spec = importlib.util.spec_from_file_location("rehearse_public_release_branch", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rehearse_public_release_branch_passes_current_surface(tmp_path: Path) -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]

    result = mod.rehearse_public_release_branch(root, tmp_path / "release_branch", force=False)

    assert result.ok
    assert "public release verification passed" in result.stdout
    assert "public release branch rehearsal passed" in result.stdout
    assert "docs/HANDOFF.md" in result.copied_internal_paths
    assert "docs/HANDOFF.md" not in result.tracked_paths
    assert (result.output_dir / "docs" / "HANDOFF.md").is_file()


def test_rehearse_public_release_branch_refuses_existing_output_without_force(tmp_path: Path) -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "release_branch"
    output_dir.mkdir()

    try:
        mod.rehearse_public_release_branch(root, output_dir, force=False)
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("release branch rehearsal should reject an existing output directory")
