#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from verify_public_release import KNOWN_TRACKED_INTERNAL, PUBLIC_VISIBLE_PATHS
except ModuleNotFoundError:
    from scripts.verify_public_release import KNOWN_TRACKED_INTERNAL, PUBLIC_VISIBLE_PATHS


EXTRA_PUBLIC_PATHS = [".gitignore"]
PREFLIGHT_COMMANDS = [
    ["git", "status", "--short", "--untracked-files=all"],
]
VERIFY_COMMANDS = [
    ["python3", "scripts/audit_reproducibility_docs.py"],
    ["python3", "scripts/verify_public_release.py", "--strict"],
    [
        "conda",
        "run",
        "-n",
        "qdte",
        "python",
        "scripts/simulate_public_release.py",
        "--output-dir",
        "/tmp/private_de_public_release_strict_check",
        "--force",
    ],
    [
        "conda",
        "run",
        "-n",
        "qdte",
        "python",
        "scripts/rehearse_public_release_branch.py",
        "--output-dir",
        "/tmp/private_de_public_release_branch_rehearsal",
        "--force",
    ],
    ["conda", "run", "-n", "qdte", "pytest", "-q"],
    ["git", "diff", "--check"],
    ["git", "status", "--short"],
]


@dataclass(frozen=True)
class ReleasePlan:
    branch_name: str
    internal_paths: list[str]
    public_paths: list[str]
    tracked_internal_paths: list[str]
    missing_public_paths: list[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_lines(root: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def _quote(value: str) -> str:
    return shlex.quote(value)


def build_plan(root: Path, branch_name: str) -> ReleasePlan:
    tracked = set(_git_lines(root, ["ls-files"]))
    internal_paths = sorted(KNOWN_TRACKED_INTERNAL)
    public_paths = [*EXTRA_PUBLIC_PATHS, *PUBLIC_VISIBLE_PATHS]
    missing_public_paths = [path for path in public_paths if not (root / path).exists()]
    tracked_internal_paths = [path for path in internal_paths if path in tracked]
    return ReleasePlan(
        branch_name=branch_name,
        internal_paths=internal_paths,
        public_paths=public_paths,
        tracked_internal_paths=tracked_internal_paths,
        missing_public_paths=missing_public_paths,
    )


def _render_multiline(command: list[str], paths: list[str]) -> list[str]:
    if not paths:
        return [" ".join(_quote(part) for part in command)]
    lines = [" ".join(_quote(part) for part in command) + " \\"]
    for idx, path in enumerate(paths):
        suffix = " \\" if idx < len(paths) - 1 else ""
        lines.append(f"  {_quote(path)}{suffix}")
    return lines


def render_shell_plan(plan: ReleasePlan, *, include_all_known_internal: bool = False) -> str:
    internal_paths = plan.internal_paths if include_all_known_internal else plan.tracked_internal_paths
    lines = [
        "# Public release branch cleanup plan.",
        "# Review before running. Do not execute on the active research branch.",
        "# Preflight status must contain only changes intentionally going into the release branch.",
        "",
    ]
    for command in PREFLIGHT_COMMANDS:
        lines.append(" ".join(_quote(part) for part in command))
    lines += [
        "",
        f"git switch -c {_quote(plan.branch_name)}",
        "",
    ]
    if internal_paths:
        lines.extend(_render_multiline(["git", "rm", "--cached"], internal_paths))
    else:
        lines.append("# No known tracked internal files are currently tracked.")
    lines += [
        "",
        "git add -u",
        "",
        *_render_multiline(["git", "add"], plan.public_paths),
        "",
    ]
    for command in VERIFY_COMMANDS:
        lines.append(" ".join(_quote(part) for part in command))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a non-destructive public-release branch cleanup command plan."
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--branch-name", default="public-release-sage")
    parser.add_argument(
        "--include-all-known-internal",
        action="store_true",
        help="Render every known internal path instead of only those currently tracked.",
    )
    parser.add_argument(
        "--allow-missing-public",
        action="store_true",
        help="Print the plan even if public-release paths are currently missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    plan = build_plan(root, args.branch_name)

    print(f"repo: {root}")
    print(f"branch: {plan.branch_name}")
    print(f"tracked internal paths to remove: {len(plan.tracked_internal_paths)}")
    if plan.missing_public_paths:
        print("missing public-release paths:")
        for path in plan.missing_public_paths:
            print(f"- {path}")
        if not args.allow_missing_public:
            print("refusing to render release plan until missing public paths are present", file=sys.stderr)
            return 1
    print("")
    print(render_shell_plan(plan, include_all_known_internal=args.include_all_known_internal), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
