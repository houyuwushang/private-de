#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from path_defaults import repo_root
    from plan_public_release_branch import build_plan
    from simulate_public_release import copy_release_surface
except ModuleNotFoundError:
    from scripts.path_defaults import repo_root
    from scripts.plan_public_release_branch import build_plan
    from scripts.simulate_public_release import copy_release_surface


DEFAULT_OUTPUT_DIR = Path("/tmp/private_de_public_release_branch_rehearsal")


@dataclass(frozen=True)
class RehearsalResult:
    output_dir: Path
    copied_public_paths: list[str]
    copied_internal_paths: list[str]
    tracked_paths: list[str]
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} failed with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _git_lines(root: Path, args: list[str]) -> list[str]:
    proc = _run(["git", *args], root)
    return [line for line in proc.stdout.splitlines() if line]


def _copy_path(source_root: Path, output_dir: Path, relpath: str) -> None:
    source = source_root / relpath
    dest = output_dir / relpath
    if not source.exists():
        raise FileNotFoundError(f"release rehearsal path is missing: {relpath}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, dest)


def copy_tracked_internal_paths(source_root: Path, output_dir: Path, paths: list[str]) -> list[str]:
    copied: list[str] = []
    for relpath in paths:
        _copy_path(source_root, output_dir, relpath)
        copied.append(relpath)
    return copied


def rehearse_public_release_branch(
    source_root: Path,
    output_dir: Path,
    *,
    branch_name: str = "public-release-sage",
    force: bool = False,
) -> RehearsalResult:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    source_plan = build_plan(source_root, branch_name)
    copied_public_paths = copy_release_surface(source_root, output_dir, force=force)
    copied_internal_paths = copy_tracked_internal_paths(
        source_root,
        output_dir,
        source_plan.tracked_internal_paths,
    )

    _run(["git", "init"], output_dir)
    _run(["git", "switch", "-c", branch_name], output_dir)
    _run(["git", "add", *source_plan.public_paths], output_dir)
    if copied_internal_paths:
        _run(["git", "add", "-f", *copied_internal_paths], output_dir)

    rehearsal_plan = build_plan(output_dir, branch_name)
    stdout_parts = [
        f"tracked internal paths before cleanup: {len(rehearsal_plan.tracked_internal_paths)}\n"
    ]
    if rehearsal_plan.tracked_internal_paths:
        rm_proc = _run(["git", "rm", "--cached", *rehearsal_plan.tracked_internal_paths], output_dir)
        stdout_parts.append(rm_proc.stdout)
    _run(["git", "add", "-u"], output_dir)
    _run(["git", "add", *rehearsal_plan.public_paths], output_dir)

    verification = _run(
        ["python3", "scripts/verify_public_release.py", "--strict", "--root", str(output_dir)],
        output_dir,
        check=False,
    )
    tracked_paths = _git_lines(output_dir, ["ls-files"])
    stdout = "".join(stdout_parts) + verification.stdout
    stderr = verification.stderr
    if verification.returncode == 0:
        stdout += "public release branch rehearsal passed\n"
    return RehearsalResult(
        output_dir=output_dir,
        copied_public_paths=copied_public_paths,
        copied_internal_paths=copied_internal_paths,
        tracked_paths=tracked_paths,
        stdout=stdout,
        stderr=stderr,
        returncode=verification.returncode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-destructively rehearse the public release branch index cleanup "
            "in a temporary git repository."
        )
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--branch-name", default="public-release-sage")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = rehearse_public_release_branch(
            args.root,
            args.output_dir,
            branch_name=args.branch_name,
            force=args.force,
        )
    except Exception as exc:
        print(f"public release branch rehearsal failed before verification: {exc}", file=sys.stderr)
        return 1

    print(f"source repo: {args.root.resolve()}")
    print(f"rehearsal repo: {result.output_dir.resolve()}")
    print(f"copied public paths: {len(result.copied_public_paths)}")
    print(f"copied tracked internal paths: {len(result.copied_internal_paths)}")
    print(f"tracked paths after cleanup: {len(result.tracked_paths)}")
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.ok:
        return 0
    print("public release branch rehearsal failed")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
