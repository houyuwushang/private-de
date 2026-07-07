from __future__ import annotations

import importlib.util
import hashlib
import sys
import tarfile
from pathlib import Path


def _load_tarball_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_paper_package_tarball.py"
    spec = importlib.util.spec_from_file_location("verify_paper_package_tarball", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _tar(package_dir: Path, tarball: Path) -> None:
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(package_dir, arcname=package_dir.name)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_tarball_accepts_matching_package(tmp_path: Path) -> None:
    mod = _load_tarball_verifier()
    package_dir = tmp_path / "paper_package_seed0to4_20260706"
    _write(package_dir / "tables" / "main.csv", b"a,b\n1,2\n")
    _write(package_dir / "figures" / "main.pdf", b"%PDF placeholder\n")
    tarball = tmp_path / "paper_package_seed0to4_20260706.tar.gz"
    _tar(package_dir, tarball)

    result = mod.verify_tarball(package_dir, tarball)

    assert result.errors == []


def test_verify_tarball_accepts_matching_sha_sidecar_and_docs(tmp_path: Path) -> None:
    mod = _load_tarball_verifier()
    package_dir = tmp_path / "paper_package_seed0to4_20260706"
    _write(package_dir / "tables" / "main.csv", b"a,b\n1,2\n")
    tarball = tmp_path / "paper_package_seed0to4_20260706.tar.gz"
    _tar(package_dir, tarball)
    digest = _sha256(tarball)
    sha256_file = tmp_path / "paper_package_seed0to4_20260706.tar.gz.sha256"
    sha256_file.write_text(f"{digest}  {tarball.name}\n", encoding="utf-8")
    doc = tmp_path / "manifest.md"
    doc.write_text(f"archive sha256: {digest}\n", encoding="utf-8")

    result = mod.verify_tarball(package_dir, tarball, sha256_file=sha256_file, sha_docs=[doc])

    assert result.errors == []


def test_verify_tarball_accepts_matching_doc_with_other_snapshot_shas(tmp_path: Path) -> None:
    mod = _load_tarball_verifier()
    package_dir = tmp_path / "paper_package_seed0to4_20260706"
    _write(package_dir / "tables" / "main.csv", b"a,b\n1,2\n")
    tarball = tmp_path / "paper_package_seed0to4_20260706.tar.gz"
    _tar(package_dir, tarball)
    digest = _sha256(tarball)
    sha256_file = tmp_path / "paper_package_seed0to4_20260706.tar.gz.sha256"
    sha256_file.write_text(f"{digest}  {tarball.name}\n", encoding="utf-8")
    doc = tmp_path / "manifest.md"
    doc.write_text(
        "\n".join(
            [
                f"archive sha256: {digest}",
                "pdf sha256: " + "1" * 64,
                "public release commit: " + "2" * 64,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = mod.verify_tarball(package_dir, tarball, sha256_file=sha256_file, sha_docs=[doc])

    assert result.errors == []


def test_verify_tarball_rejects_stale_sha_sidecar(tmp_path: Path) -> None:
    mod = _load_tarball_verifier()
    package_dir = tmp_path / "paper_package_seed0to4_20260706"
    _write(package_dir / "tables" / "main.csv", b"a,b\n1,2\n")
    tarball = tmp_path / "paper_package_seed0to4_20260706.tar.gz"
    _tar(package_dir, tarball)
    sha256_file = tmp_path / "paper_package_seed0to4_20260706.tar.gz.sha256"
    sha256_file.write_text(f"{'0' * 64}  {tarball.name}\n", encoding="utf-8")

    result = mod.verify_tarball(package_dir, tarball, sha256_file=sha256_file)

    assert any("package tarball SHA-256 mismatch" in error for error in result.errors)


def test_verify_tarball_rejects_stale_documented_sha(tmp_path: Path) -> None:
    mod = _load_tarball_verifier()
    package_dir = tmp_path / "paper_package_seed0to4_20260706"
    _write(package_dir / "tables" / "main.csv", b"a,b\n1,2\n")
    tarball = tmp_path / "paper_package_seed0to4_20260706.tar.gz"
    _tar(package_dir, tarball)
    digest = _sha256(tarball)
    sha256_file = tmp_path / "paper_package_seed0to4_20260706.tar.gz.sha256"
    sha256_file.write_text(f"{digest}  {tarball.name}\n", encoding="utf-8")
    doc = tmp_path / "manifest.md"
    doc.write_text(f"archive sha256: {'0' * 64}\n", encoding="utf-8")

    result = mod.verify_tarball(package_dir, tarball, sha256_file=sha256_file, sha_docs=[doc])

    assert any("documented package archive SHA is missing current value" in error for error in result.errors)


def test_verify_tarball_rejects_stale_content(tmp_path: Path) -> None:
    mod = _load_tarball_verifier()
    package_dir = tmp_path / "paper_package_seed0to4_20260706"
    _write(package_dir / "tables" / "main.csv", b"old\n")
    tarball = tmp_path / "paper_package_seed0to4_20260706.tar.gz"
    _tar(package_dir, tarball)
    _write(package_dir / "tables" / "main.csv", b"new\n")

    result = mod.verify_tarball(package_dir, tarball)

    assert any("tarball content mismatch: paper_package_seed0to4_20260706/tables/main.csv" in error for error in result.errors)


def test_verify_tarball_rejects_missing_member(tmp_path: Path) -> None:
    mod = _load_tarball_verifier()
    package_dir = tmp_path / "paper_package_seed0to4_20260706"
    _write(package_dir / "tables" / "main.csv", b"a,b\n")
    tarball = tmp_path / "paper_package_seed0to4_20260706.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        pass

    result = mod.verify_tarball(package_dir, tarball)

    assert any("tarball missing package file: paper_package_seed0to4_20260706/tables/main.csv" in error for error in result.errors)
