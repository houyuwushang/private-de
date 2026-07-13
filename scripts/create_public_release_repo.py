#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from path_defaults import repo_root
    from simulate_public_release import copy_release_surface
except ModuleNotFoundError:
    from scripts.path_defaults import repo_root
    from scripts.simulate_public_release import copy_release_surface


DEFAULT_BRANCH_NAME = "public-release-sage"
DEFAULT_OUTPUT_DIR = repo_root().parent / "private-de-public-release"


@dataclass(frozen=True)
class ReleaseRepoResult:
    output_dir: Path
    branch_name: str
    copied_paths: list[str]
    tracked_paths: list[str]
    commit_sha: str | None
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


def _import_smoke_command() -> list[str]:
    code = (
        "modules = ['qdte', 'qdte.config', 'qdte.privacy.gaussian', "
        "'qdte.queries.workload', 'qdte.evolution.engine', 'qdte.eval.external']\n"
        "for module in modules:\n"
        "    __import__(module)\n"
        "print(f'public release import smoke passed: {len(modules)} modules')\n"
    )
    return [sys.executable, "-B", "-c", code]


def _qdte_smoke_command() -> list[str]:
    return [
        sys.executable,
        "-B",
        "scripts/smoke_qdte.py",
        "--mode",
        "dp",
        "--rows",
        "120",
        "--max-iters",
        "2",
        "--output-dir",
        "outputs/public_release_smoke_qdte",
    ]


def create_public_release_repo(
    source_root: Path,
    output_dir: Path,
    *,
    branch_name: str = DEFAULT_BRANCH_NAME,
    force: bool = False,
    run_smoke: bool = True,
    commit: bool = False,
    commit_message: str = "Initial public QDTE release",
) -> ReleaseRepoResult:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    copied_paths = copy_release_surface(source_root, output_dir, force=force)

    _run(["git", "init"], output_dir)
    _run(["git", "switch", "-c", branch_name], output_dir)
    _run(["git", "add", "."], output_dir)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    commit_sha: str | None = None
    verification = _run(
        ["python3", "scripts/verify_public_release.py", "--strict", "--root", str(output_dir)],
        output_dir,
        check=False,
    )
    stdout_parts.append(verification.stdout)
    stderr_parts.append(verification.stderr)
    returncode = verification.returncode

    if returncode == 0 and commit:
        commit_proc = _run(
            [
                "git",
                "-c",
                "user.name=QDTE Release Bot",
                "-c",
                "user.email=qdte-release@example.invalid",
                "commit",
                "-m",
                commit_message,
            ],
            output_dir,
            check=False,
        )
        stdout_parts.append(commit_proc.stdout)
        stderr_parts.append(commit_proc.stderr)
        if commit_proc.returncode == 0:
            commit_sha = _run(["git", "rev-parse", "HEAD"], output_dir).stdout.strip()
            stdout_parts.append(f"public release commit created: {commit_sha}\n")
        else:
            returncode = commit_proc.returncode

    if returncode == 0 and run_smoke:
        import_smoke = _run(_import_smoke_command(), output_dir, check=False)
        stdout_parts.append(import_smoke.stdout)
        stderr_parts.append(import_smoke.stderr)
        if import_smoke.returncode != 0:
            returncode = import_smoke.returncode

    if returncode == 0 and run_smoke:
        qdte_smoke = _run(_qdte_smoke_command(), output_dir, check=False)
        stdout_parts.append(qdte_smoke.stdout)
        stderr_parts.append(qdte_smoke.stderr)
        if qdte_smoke.returncode == 0:
            stdout_parts.append("public release QDTE smoke passed\n")
        else:
            returncode = qdte_smoke.returncode

    tracked_paths = _git_lines(output_dir, ["ls-files"])
    if returncode == 0:
        stdout_parts.append("public release repository creation passed\n")
    return ReleaseRepoResult(
        output_dir=output_dir,
        branch_name=branch_name,
        copied_paths=copied_paths,
        tracked_paths=tracked_paths,
        commit_sha=commit_sha,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        returncode=returncode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a persistent clean public release git repository from the "
            "current public QDTE source surface without modifying the "
            "active research branch."
        )
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--branch-name", default=DEFAULT_BRANCH_NAME)
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--skip-smoke", action="store_true", help="Only run strict release verification.")
    parser.add_argument("--commit", action="store_true", help="Create an initial commit in the generated release repo.")
    parser.add_argument("--commit-message", default="Initial public QDTE release")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = create_public_release_repo(
            args.root,
            args.output_dir,
            branch_name=args.branch_name,
            force=args.force,
            run_smoke=not args.skip_smoke,
            commit=args.commit,
            commit_message=args.commit_message,
        )
    except Exception as exc:
        print(f"public release repository creation failed before verification: {exc}", file=sys.stderr)
        return 1

    print(f"source repo: {args.root.resolve()}")
    print(f"public release repo: {result.output_dir.resolve()}")
    print(f"branch: {result.branch_name}")
    print(f"copied public paths: {len(result.copied_paths)}")
    print(f"tracked public paths: {len(result.tracked_paths)}")
    if result.commit_sha is not None:
        print(f"commit: {result.commit_sha}")
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.ok:
        return 0
    print("public release repository creation failed")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
