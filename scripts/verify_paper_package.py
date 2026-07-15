#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from path_defaults import external_results, paper_package_dir, sage_paper_dir
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, paper_package_dir, sage_paper_dir

DEFAULT_PACKAGE_DIR = paper_package_dir()
DEFAULT_EXTERNAL_RESULTS = external_results()
DEFAULT_SAGE_PAPER_DIR = sage_paper_dir()

TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class PackageCheck:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_replacements(external_results_dir: Path, sage_paper_results_dir: Path) -> list[tuple[str, str]]:
    roots = {
        str(external_results_dir.parent),
        str(sage_paper_results_dir.parent),
    }
    return [(root, "$SAGE_BASELINE_ROOT") for root in sorted(roots, key=len, reverse=True)]


def _sanitize_text(text: str, external_results_dir: Path, sage_paper_results_dir: Path) -> str:
    for source, replacement in _path_replacements(external_results_dir, sage_paper_results_dir):
        text = text.replace(source, replacement)
    return text


def _source_bytes_for_package_artifact(
    rel: str,
    *,
    external_results_dir: Path,
    sage_paper_results_dir: Path,
) -> bytes | None:
    rel_path = Path(rel)
    if len(rel_path.parts) < 2:
        return None
    section = rel_path.parts[0]
    filename = rel_path.name
    source: Path | None = None
    if section in {"tables", "appendix"}:
        candidate = external_results_dir / filename
        if candidate.exists():
            source = candidate
        else:
            candidate = sage_paper_results_dir / filename
            if candidate.exists():
                source = candidate
    if source is None or not source.is_file():
        return None
    if source.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        return _sanitize_text(
            source.read_text(encoding="utf-8"),
            external_results_dir,
            sage_paper_results_dir,
        ).encode("utf-8")
    return source.read_bytes()


def _read_filelist(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        if "  " not in line:
            raise ValueError(f"invalid checksum line {lineno}: {line!r}")
        digest, rel = line.split("  ", 1)
        if len(digest) != 64:
            raise ValueError(f"invalid sha256 digest on line {lineno}: {digest!r}")
        if rel in checksums:
            raise ValueError(f"duplicate checksum entry on line {lineno}: {rel}")
        checksums[rel] = digest
    return checksums


def _actual_files(package_dir: Path, filelist_name: str, checksums_name: str) -> set[str]:
    excluded = {filelist_name, checksums_name}
    return {
        str(path.relative_to(package_dir))
        for path in package_dir.rglob("*")
        if path.is_file() and str(path.relative_to(package_dir)) not in excluded
    }


def verify_package(
    package_dir: Path,
    *,
    filelist_name: str = "ARTIFACT_FILELIST_20260706.txt",
    checksums_name: str = "ARTIFACT_SHA256SUMS_20260706.txt",
    external_results_dir: Path = DEFAULT_EXTERNAL_RESULTS,
    sage_paper_results_dir: Path = DEFAULT_SAGE_PAPER_DIR,
) -> PackageCheck:
    errors: list[str] = []
    if not package_dir.exists():
        return PackageCheck(errors=[f"package directory is missing: {package_dir}"])
    if not package_dir.is_dir():
        return PackageCheck(errors=[f"package path is not a directory: {package_dir}"])

    filelist_path = package_dir / filelist_name
    checksums_path = package_dir / checksums_name
    if not filelist_path.exists():
        errors.append(f"filelist is missing: {filelist_path}")
    if not checksums_path.exists():
        errors.append(f"checksum manifest is missing: {checksums_path}")
    if errors:
        return PackageCheck(errors=errors)

    try:
        listed = _read_filelist(filelist_path)
    except Exception as exc:
        return PackageCheck(errors=[f"failed to read filelist: {exc}"])
    try:
        checksums = _read_checksums(checksums_path)
    except Exception as exc:
        return PackageCheck(errors=[f"failed to read checksum manifest: {exc}"])

    actual = _actual_files(package_dir, filelist_name, checksums_name)
    checksum_paths = set(checksums)

    for rel in sorted(actual - listed):
        errors.append(f"artifact missing from filelist: {rel}")
    for rel in sorted(listed - actual):
        errors.append(f"filelist entry missing from package: {rel}")
    for rel in sorted(checksum_paths - listed):
        errors.append(f"checksum entry missing from filelist: {rel}")
    for rel in sorted(listed - checksum_paths):
        errors.append(f"filelist entry missing checksum: {rel}")

    for rel in sorted(actual & checksum_paths):
        expected = checksums[rel]
        observed = _sha256(package_dir / rel)
        if observed != expected:
            errors.append(f"sha256 mismatch for {rel}: expected {expected}, observed {observed}")
        source_bytes = _source_bytes_for_package_artifact(
            rel,
            external_results_dir=external_results_dir,
            sage_paper_results_dir=sage_paper_results_dir,
        )
        if source_bytes is not None and (package_dir / rel).read_bytes() != source_bytes:
            errors.append(f"packaged artifact is stale relative to source: {rel}")

    return PackageCheck(errors=errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a packaged SAGE paper result artifact directory.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--filelist-name", default="ARTIFACT_FILELIST_20260706.txt")
    parser.add_argument("--checksums-name", default="ARTIFACT_SHA256SUMS_20260706.txt")
    parser.add_argument("--external-results-dir", type=Path, default=DEFAULT_EXTERNAL_RESULTS)
    parser.add_argument("--sage-paper-results-dir", type=Path, default=DEFAULT_SAGE_PAPER_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_package(
        args.package_dir,
        filelist_name=str(args.filelist_name),
        checksums_name=str(args.checksums_name),
        external_results_dir=args.external_results_dir,
        sage_paper_results_dir=args.sage_paper_results_dir,
    )
    print(f"paper package: {args.package_dir}")
    if result.ok:
        print("paper package verification passed")
        return 0
    for error in result.errors:
        print(f"ERROR: {error}")
    print("paper package verification failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
