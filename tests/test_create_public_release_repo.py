from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "create_public_release_repo.py"
    spec = importlib.util.spec_from_file_location("create_public_release_repo", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_create_public_release_repo_passes_strict_verifier(tmp_path: Path) -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]

    result = mod.create_public_release_repo(
        root,
        tmp_path / "public_release_repo",
        branch_name="public-release-test",
        force=False,
        run_smoke=False,
    )

    assert result.ok
    assert "public release verification passed" in result.stdout
    assert "public release repository creation passed" in result.stdout
    assert result.branch_name == "public-release-test"
    assert "README.md" in result.tracked_paths
    assert result.commit_sha is None
    assert "docs/HANDOFF.md" not in result.tracked_paths
    assert not (result.output_dir / "docs" / "HANDOFF.md").exists()
    assert (result.output_dir / "scripts" / "verify_public_release.py").is_file()


def test_create_public_release_repo_refuses_existing_output_without_force(tmp_path: Path) -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "public_release_repo"
    output_dir.mkdir()

    try:
        mod.create_public_release_repo(root, output_dir, force=False, run_smoke=False)
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("release repo creation should reject an existing output directory")


def test_create_public_release_repo_can_create_initial_commit(tmp_path: Path) -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]

    result = mod.create_public_release_repo(
        root,
        tmp_path / "public_release_repo",
        branch_name="public-release-test",
        force=False,
        run_smoke=False,
        commit=True,
        commit_message="Test public release",
    )

    assert result.ok
    assert result.commit_sha
    assert "public release commit created" in result.stdout
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=result.output_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    assert observed == result.commit_sha
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=result.output_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    assert status == ""
