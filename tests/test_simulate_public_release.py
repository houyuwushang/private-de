from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_public_release.py"
    spec = importlib.util.spec_from_file_location("simulate_public_release", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_release_paths_include_release_verifiers() -> None:
    mod = _load_module()

    paths = mod.public_release_paths()

    assert ".gitignore" in paths
    assert "qdte" in paths
    assert "qdte/eval/external.py" not in paths
    assert "scripts/run_qdte.py" in paths
    assert "scripts/smoke_qdte.py" in paths
    assert "scripts/run_integrated_sage_qdte.py" in paths
    assert "scripts/run_orthogonal_low_budget_pilot.py" in paths
    assert "scripts/run_nonnegative_projection_pilot.py" in paths
    assert "scripts/audit_public_release_plan.py" in paths
    assert "scripts/verify_public_release.py" in paths
    assert "scripts/simulate_public_release.py" in paths
    assert "scripts/rehearse_public_release_branch.py" in paths
    assert "scripts/create_public_release_repo.py" in paths
    assert "scripts/verify_paper_package_tarball.py" in paths
    assert "tests/test_simulate_public_release.py" in paths
    assert "tests/conftest.py" in paths
    assert "tests/test_rehearse_public_release_branch.py" in paths
    assert "tests/test_create_public_release_repo.py" in paths
    assert "tests/test_run_integrated_sage_qdte.py" in paths
    assert "tests/test_nonnegative_projection_theorem.py" in paths
    assert "docs/CODE_REVIEW_GUIDE.md" in paths
    assert "docs/HANDOFF.md" not in paths


def test_simulate_public_release_passes_strict_verifier_on_current_public_surface(tmp_path: Path) -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]

    result = mod.simulate_public_release(root, tmp_path / "public_release", force=False)

    assert result.ok
    assert "public release verification passed" in result.stdout
    assert "public release import smoke passed" in result.stdout
    assert "public release QDTE smoke passed" in result.stdout
    assert "docs/HANDOFF.md" not in result.stdout
    assert (result.output_dir / "qdte" / "evolution" / "engine.py").is_file()
    assert (result.output_dir / "outputs" / "public_release_smoke_qdte" / "metrics_final.json").is_file()
    assert not (result.output_dir / "qdte" / "__pycache__").exists()


def test_copy_release_surface_refuses_existing_output_without_force(tmp_path: Path) -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "public_release"
    output_dir.mkdir()

    try:
        mod.copy_release_surface(root, output_dir, paths=["README.md"], force=False)
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("copy_release_surface should reject an existing output directory")
