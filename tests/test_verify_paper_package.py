from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


def _load_package_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_paper_package.py"
    spec = importlib.util.spec_from_file_location("verify_paper_package", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(root: Path, entries: dict[str, bytes]) -> None:
    for rel, data in entries.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "ARTIFACT_FILELIST_20260706.txt").write_text("\n".join(sorted(entries)) + "\n")
    (root / "ARTIFACT_SHA256SUMS_20260706.txt").write_text(
        "".join(f"{_sha256(entries[rel])}  {rel}\n" for rel in sorted(entries))
    )


def test_verify_paper_package_accepts_consistent_package(tmp_path: Path) -> None:
    mod = _load_package_verifier()
    _write_manifest(
        tmp_path,
        {
            "PAPER_RESULTS_PACKAGE_20260706.md": b"summary\n",
            "tables/main.csv": b"a,b\n1,2\n",
            "figures/main.txt": b"figure placeholder\n",
        },
    )

    result = mod.verify_package(tmp_path)

    assert result.errors == []


def test_verify_paper_package_rejects_checksum_mismatch(tmp_path: Path) -> None:
    mod = _load_package_verifier()
    _write_manifest(tmp_path, {"tables/main.csv": b"a,b\n1,2\n"})
    (tmp_path / "tables" / "main.csv").write_bytes(b"a,b\n9,9\n")

    result = mod.verify_package(tmp_path)

    assert any("sha256 mismatch for tables/main.csv" in error for error in result.errors)


def test_verify_paper_package_rejects_unlisted_artifact(tmp_path: Path) -> None:
    mod = _load_package_verifier()
    _write_manifest(tmp_path, {"tables/main.csv": b"a,b\n1,2\n"})
    extra = tmp_path / "tables" / "extra.csv"
    extra.write_bytes(b"x\n")

    result = mod.verify_package(tmp_path)

    assert any("artifact missing from filelist: tables/extra.csv" in error for error in result.errors)


def test_verify_paper_package_rejects_stale_source_copy(tmp_path: Path) -> None:
    mod = _load_package_verifier()
    package_dir = tmp_path / "package"
    external_results = tmp_path / "external_results"
    sage_paper = tmp_path / "sage_paper"
    external_results.mkdir()
    sage_paper.mkdir()
    _write_manifest(
        package_dir,
        {
            "appendix/baseline_gpu_provenance_20260707.csv": b"old\n",
        },
    )
    (external_results / "baseline_gpu_provenance_20260707.csv").write_bytes(b"new\n")

    result = mod.verify_package(
        package_dir,
        external_results_dir=external_results,
        sage_paper_results_dir=sage_paper,
    )

    assert any(
        "packaged artifact is stale relative to source: appendix/baseline_gpu_provenance_20260707.csv"
        in error
        for error in result.errors
    )
