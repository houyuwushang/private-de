from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_release_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_public_release.py"
    spec = importlib.util.spec_from_file_location("verify_public_release", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_paths(root: Path, paths: list[str]) -> None:
    for rel in paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")


def test_release_audit_allows_known_tracked_internal_as_warning(tmp_path: Path) -> None:
    mod = _load_release_module()
    _git(tmp_path, "init")
    assert "qdte" in mod.PUBLIC_VISIBLE_PATHS
    assert "qdte/eval/external.py" not in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/run_qdte.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/smoke_qdte.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/run_ablation.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/run_adaptive_selection_ablation.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/audit_public_release_plan.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/audit_original_protocol_baselines.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/create_public_release_repo.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/simulate_public_release.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/rehearse_public_release_branch.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "tests/test_simulate_public_release.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "tests/test_rehearse_public_release_branch.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "tests/test_audit_original_protocol_baselines.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "tests/test_create_public_release_repo.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "scripts/verify_paper_package_tarball.py" in mod.PUBLIC_VISIBLE_PATHS
    assert "tests/test_verify_paper_package_tarball.py" in mod.PUBLIC_VISIBLE_PATHS

    (tmp_path / ".gitignore").write_text(
        "\n".join(
            [
                "docs/HANDOFF.md",
                "docs/archive/",
                "docs/PAPER_*",
                "docs/EXTERNAL_BASELINE_ADMISSION_*",
                "docs/GPU_EXPERIMENT_QUEUE_*",
                "docs/RESULTS_PACKAGE.md",
                "docs/USENIX_*",
                "configs/*_rappp_sage.yaml",
                "data/",
                "sage_paper/",
                "resources/",
                "outputs/",
            ]
        )
        + "\n"
    )
    _write_paths(
        tmp_path,
        [
            *mod.PUBLIC_VISIBLE_PATHS,
            "docs/HANDOFF.md",
            "docs/archive/HANDOFF_20260706_1519_full_history.md",
            "data/2014/1-Year/ss14pca.csv",
        ],
    )

    _git(tmp_path, "add", ".gitignore", "README.md")
    _git(tmp_path, "add", "-f", "docs/HANDOFF.md")

    audit = mod.verify(root=tmp_path, strict=False)
    assert audit.errors == []
    assert any("docs/HANDOFF.md" in warning for warning in audit.warnings)

    strict = mod.verify(root=tmp_path, strict=True)
    assert any("docs/HANDOFF.md" in error for error in strict.errors)


def test_release_audit_rejects_visible_internal_untracked_file(tmp_path: Path) -> None:
    mod = _load_release_module()
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text(
        "\n".join(
            [
                "docs/HANDOFF.md",
                "sage_paper/",
                "resources/",
                "outputs/",
            ]
        )
        + "\n"
    )
    _write_paths(
        tmp_path,
        [
            *mod.PUBLIC_VISIBLE_PATHS,
            "docs/PAPER_VISIBLE.md",
            "docs/USENIX_VISIBLE.md",
            "data/raw.csv",
        ],
    )

    result = mod.verify(root=tmp_path, strict=False)
    assert any("docs/PAPER_VISIBLE.md" in error for error in result.errors)
    assert any("docs/USENIX_VISIBLE.md" in error for error in result.errors)
    assert any("data/raw.csv" in error for error in result.errors)


def test_release_audit_rejects_missing_public_file(tmp_path: Path) -> None:
    mod = _load_release_module()
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("docs/HANDOFF.md\n")
    missing = "docs/PUBLIC_RELEASE_MANIFEST_20260706.md"
    _write_paths(tmp_path, [path for path in mod.PUBLIC_VISIBLE_PATHS if path != missing])

    result = mod.verify(root=tmp_path, strict=False)
    assert any(missing in error for error in result.errors)


def test_release_audit_rejects_local_paths_in_public_files(tmp_path: Path) -> None:
    mod = _load_release_module()
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("docs/HANDOFF.md\n")
    _write_paths(tmp_path, mod.PUBLIC_VISIBLE_PATHS)
    (tmp_path / "README.md").write_text("/home/" + "qianqiu/private/path\n")

    result = mod.verify(root=tmp_path, strict=False)

    assert any("README.md" in error and "/home/" + "qianqiu" in error for error in result.errors)


def test_release_audit_rejects_local_paths_in_tracked_release_files(tmp_path: Path) -> None:
    mod = _load_release_module()
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("docs/HANDOFF.md\n")
    _write_paths(tmp_path, mod.PUBLIC_VISIBLE_PATHS)
    _write_paths(tmp_path, ["scripts/non_manifest_release_helper.py"])
    (tmp_path / "scripts" / "non_manifest_release_helper.py").write_text(
        "ROOT = '/home/" + "qianqiu/private/path'\n"
    )
    _git(tmp_path, "add", ".")

    result = mod.verify(root=tmp_path, strict=False)

    assert any(
        "scripts/non_manifest_release_helper.py" in error and "/home/" + "qianqiu" in error
        for error in result.errors
    )
