#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path

try:
    from path_defaults import qdte_paper_package_dir
except ModuleNotFoundError:
    from scripts.path_defaults import qdte_paper_package_dir


DEFAULT_PACKAGE = qdte_paper_package_dir()
DEFAULT_TARBALL = DEFAULT_PACKAGE.with_suffix(".tar.gz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_archive(package_dir: Path, tarball: Path) -> str:
    package_dir = package_dir.resolve()
    tarball = tarball.resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(package_dir)
    files = sorted(path for path in package_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"package is empty: {package_dir}")
    tarball.parent.mkdir(parents=True, exist_ok=True)
    with tarball.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in files:
                    arcname = str(Path(package_dir.name) / path.relative_to(package_dir))
                    info = archive.gettarinfo(str(path), arcname=arcname)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    digest = _sha256(tarball)
    sidecar = Path(f"{tarball}.sha256")
    sidecar.write_text(f"{digest}  {tarball.name}\n", encoding="utf-8")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic QDTE paper-package tarball and SHA-256 sidecar."
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digest = create_archive(args.package_dir, args.tarball)
    print(f"QDTE package: {args.package_dir.resolve()}")
    print(f"QDTE archive: {args.tarball.resolve()}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
