#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
    from plan_public_release_branch import (
        EXTRA_PUBLIC_PATHS,
        PREFLIGHT_COMMANDS,
        VERIFY_COMMANDS,
        build_plan,
        render_shell_plan,
    )
    from verify_public_release import PUBLIC_VISIBLE_PATHS
except ModuleNotFoundError:
    from scripts.plan_public_release_branch import (
        EXTRA_PUBLIC_PATHS,
        PREFLIGHT_COMMANDS,
        VERIFY_COMMANDS,
        build_plan,
        render_shell_plan,
    )
    from scripts.verify_public_release import PUBLIC_VISIBLE_PATHS


MANIFEST = Path("docs/PUBLIC_RELEASE_MANIFEST_20260706.md")
REQUIRED_VERIFY_COMMANDS = [
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
REQUIRED_PREFLIGHT_COMMANDS = [
    ["git", "status", "--short", "--untracked-files=all"],
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_text(root: Path) -> str:
    path = root / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"missing public release manifest: {path}")
    return path.read_text(encoding="utf-8")


def _extract_manifest_multiline_paths(text: str, command_line: str) -> list[str]:
    paths: list[str] = []
    lines = text.splitlines()
    try:
        start = lines.index(command_line) + 1
    except ValueError:
        return []
    for line in lines[start:]:
        item = line.strip()
        if not item:
            break
        if item.endswith("\\"):
            item = item[:-1].rstrip()
        if item:
            paths.append(item)
    return paths


def _extract_manifest_git_add_paths(text: str) -> list[str]:
    return _extract_manifest_multiline_paths(text, "git add \\")


def _extract_manifest_git_rm_cached_paths(text: str) -> list[str]:
    return _extract_manifest_multiline_paths(text, "git rm --cached \\")


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def audit(root: Path) -> list[str]:
    errors: list[str] = []
    expected_public_paths = [*EXTRA_PUBLIC_PATHS, *PUBLIC_VISIBLE_PATHS]
    text = _manifest_text(root)
    manifest_paths = _extract_manifest_git_add_paths(text)
    manifest_set = set(manifest_paths)
    expected_set = set(expected_public_paths)
    missing = sorted(expected_set - manifest_set)
    extra = sorted(manifest_set - expected_set)
    if missing:
        errors.append(f"public release manifest git-add block is missing paths: {missing}")
    if extra:
        errors.append(f"public release manifest git-add block has unexpected paths: {extra}")

    plan = build_plan(root, "public-release-sage")
    manifest_internal_paths = _extract_manifest_git_rm_cached_paths(text)
    manifest_internal_set = set(manifest_internal_paths)
    expected_internal_set = set(plan.tracked_internal_paths)
    missing_internal = sorted(expected_internal_set - manifest_internal_set)
    extra_internal = sorted(manifest_internal_set - expected_internal_set)
    if missing_internal:
        errors.append(f"public release manifest git-rm block is missing tracked internal paths: {missing_internal}")
    if extra_internal:
        errors.append(f"public release manifest git-rm block has unexpected tracked internal paths: {extra_internal}")

    if VERIFY_COMMANDS != REQUIRED_VERIFY_COMMANDS:
        errors.append(
            "release plan VERIFY_COMMANDS changed without updating audit expectations: "
            f"observed={VERIFY_COMMANDS} expected={REQUIRED_VERIFY_COMMANDS}"
        )
    if PREFLIGHT_COMMANDS != REQUIRED_PREFLIGHT_COMMANDS:
        errors.append(
            "release plan PREFLIGHT_COMMANDS changed without updating audit expectations: "
            f"observed={PREFLIGHT_COMMANDS} expected={REQUIRED_PREFLIGHT_COMMANDS}"
        )

    rendered = render_shell_plan(plan)
    for command in REQUIRED_PREFLIGHT_COMMANDS:
        command_text = _command_text(command)
        if command_text not in rendered:
            errors.append(f"rendered release plan is missing preflight command: {command_text}")
        if command_text not in text:
            errors.append(f"public release manifest is missing preflight command: {command_text}")
    for path in expected_public_paths:
        if path not in plan.public_paths:
            errors.append(f"generated release plan is missing public path: {path}")
        if path not in rendered:
            errors.append(f"rendered release plan is missing public path: {path}")
    for path in plan.tracked_internal_paths:
        if path not in rendered:
            errors.append(f"rendered release plan is missing tracked internal removal path: {path}")
    for command in REQUIRED_VERIFY_COMMANDS:
        command_text = _command_text(command)
        if command_text not in rendered:
            errors.append(f"rendered release plan is missing verification command: {command_text}")
        if command_text not in text:
            errors.append(f"public release manifest is missing verification command: {command_text}")
    if "docs/HANDOFF.md" in plan.public_paths:
        errors.append("generated release plan must not publish docs/HANDOFF.md")
    if "sage_paper" in plan.public_paths:
        errors.append("generated release plan must not publish sage_paper")
    return errors


def main() -> int:
    root = repo_root()
    errors = audit(root)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("public release plan audit failed")
        return 1
    print("public release plan audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
