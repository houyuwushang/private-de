#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path

try:
    from path_defaults import qdte_paper_package_dir, repo_root
except ModuleNotFoundError:
    from scripts.path_defaults import qdte_paper_package_dir, repo_root


DEFAULT_PACKAGE_DIR = qdte_paper_package_dir()
DEFAULT_TARBALL = DEFAULT_PACKAGE_DIR.with_suffix(".tar.gz")
DEFAULT_SHA256_FILE = Path(f"{DEFAULT_TARBALL}.sha256")
DEFAULT_SHA_DOC_RELPATHS = (
    "docs/PUBLIC_RELEASE_MANIFEST_20260706.md",
    "docs/QDTE_SUBMISSION_READINESS_20260711.md",
)
SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")


@dataclass
class TarballCheck:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _package_files(package_dir: Path) -> dict[str, Path]:
    prefix = package_dir.name
    return {
        f"{prefix}/{path.relative_to(package_dir)}": path
        for path in package_dir.rglob("*")
        if path.is_file()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256_sidecar(tarball: Path, sha256_file: Path) -> list[str]:
    errors: list[str] = []
    if not sha256_file.exists():
        return [f"package tarball SHA-256 sidecar is missing: {sha256_file}"]
    if not sha256_file.is_file():
        return [f"package tarball SHA-256 sidecar path is not a file: {sha256_file}"]

    text = sha256_file.read_text(encoding="utf-8").strip()
    parts = text.split()
    if not parts or not SHA256_RE.fullmatch(parts[0]):
        return [f"package tarball SHA-256 sidecar has invalid format: {sha256_file}"]

    expected = parts[0].lower()
    observed = _sha256(tarball)
    if observed != expected:
        errors.append(
            f"package tarball SHA-256 mismatch: observed {observed}, sidecar has {expected}"
        )
    if len(parts) > 1 and Path(parts[1]).name != tarball.name:
        errors.append(
            f"package tarball SHA-256 sidecar names {parts[1]!r}, expected {tarball.name!r}"
        )
    return errors


def default_sha_docs(root: Path | None = None) -> list[Path]:
    base = repo_root() if root is None else root
    return [base / relpath for relpath in DEFAULT_SHA_DOC_RELPATHS]


def _verify_documented_sha(path: Path, expected_sha: str) -> list[str]:
    if not path.exists():
        return [f"documented package archive SHA file is missing: {path}"]
    text = path.read_text(encoding="utf-8")
    observed = {match.lower() for match in SHA256_RE.findall(text)}
    if expected_sha not in observed:
        return [f"documented package archive SHA is missing current value in {path}: {expected_sha}"]
    return []


def verify_tarball(
    package_dir: Path,
    tarball: Path,
    *,
    sha256_file: Path | None = None,
    sha_docs: Iterable[Path] = (),
) -> TarballCheck:
    errors: list[str] = []
    if not package_dir.exists():
        return TarballCheck(errors=[f"package directory is missing: {package_dir}"])
    if not package_dir.is_dir():
        return TarballCheck(errors=[f"package path is not a directory: {package_dir}"])
    if not tarball.exists():
        return TarballCheck(errors=[f"package tarball is missing: {tarball}"])
    if not tarball.is_file():
        return TarballCheck(errors=[f"package tarball path is not a file: {tarball}"])

    expected = _package_files(package_dir)
    try:
        with tarfile.open(tarball, "r:gz") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile()
            }
            for rel in sorted(set(expected) - set(members)):
                errors.append(f"tarball missing package file: {rel}")
            for rel in sorted(set(members) - set(expected)):
                errors.append(f"tarball contains extra file: {rel}")
            for rel in sorted(set(expected) & set(members)):
                extracted = archive.extractfile(members[rel])
                if extracted is None:
                    errors.append(f"failed to read tarball member: {rel}")
                    continue
                observed = extracted.read()
                expected_bytes = expected[rel].read_bytes()
                if observed != expected_bytes:
                    errors.append(f"tarball content mismatch: {rel}")
    except tarfile.TarError as exc:
        errors.append(f"failed to read package tarball: {exc}")

    if sha256_file is not None:
        errors.extend(_verify_sha256_sidecar(tarball, sha256_file))
        if not errors:
            expected_sha = _sha256(tarball)
            for doc in sha_docs:
                errors.extend(_verify_documented_sha(doc, expected_sha))
    return TarballCheck(errors=errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the paper package tarball matches the current package directory."
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--sha256-file", type=Path, default=DEFAULT_SHA256_FILE)
    parser.add_argument(
        "--sha-doc",
        action="append",
        type=Path,
        default=None,
        help="Documentation file that must contain only the current package tarball SHA-256.",
    )
    parser.add_argument("--skip-sha256", action="store_true", help="Skip SHA-256 sidecar and doc checks.")
    parser.add_argument("--skip-sha-docs", action="store_true", help="Skip documented SHA reference checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sha256_file = None if args.skip_sha256 else args.sha256_file
    sha_docs: list[Path] = []
    if not args.skip_sha256 and not args.skip_sha_docs:
        sha_docs = args.sha_doc if args.sha_doc is not None else default_sha_docs()
    result = verify_tarball(args.package_dir, args.tarball, sha256_file=sha256_file, sha_docs=sha_docs)
    print(f"paper package dir: {args.package_dir}")
    print(f"paper package tarball: {args.tarball}")
    if sha256_file is not None:
        print(f"paper package tarball SHA-256 sidecar: {sha256_file}")
    for doc in sha_docs:
        print(f"documented package archive SHA: {doc}")
    if result.ok:
        print("paper package tarball verification passed")
        return 0
    for error in result.errors:
        print(f"ERROR: {error}")
    print("paper package tarball verification failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
