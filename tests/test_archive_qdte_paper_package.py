from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "archive_qdte_paper_package.py"
    spec = importlib.util.spec_from_file_location("archive_qdte_paper_package", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_is_deterministic_and_complete(tmp_path: Path) -> None:
    mod = _load_module()
    package = tmp_path / "package"
    (package / "tables").mkdir(parents=True)
    (package / "tables/main.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (package / "metadata.json").write_text("{}\n", encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_digest = mod.create_archive(package, first)
    second_digest = mod.create_archive(package, second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    assert Path(f"{first}.sha256").read_text().startswith(first_digest)
    with tarfile.open(first, "r:gz") as archive:
        names = sorted(member.name for member in archive.getmembers() if member.isfile())
    assert names == ["package/metadata.json", "package/tables/main.csv"]
