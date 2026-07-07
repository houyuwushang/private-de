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
    from verify_public_release import PUBLIC_VISIBLE_PATHS
except ModuleNotFoundError:
    from scripts.path_defaults import repo_root
    from scripts.verify_public_release import PUBLIC_VISIBLE_PATHS


EXTRA_PUBLIC_PATHS = [".gitignore"]
PUBLIC_IMPORT_MODULES = [
    "qdte",
    "qdte.config",
    "qdte.config_validation",
    "qdte.dataio",
    "qdte.preprocess",
    "qdte.schema",
    "qdte.privacy.accountant",
    "qdte.privacy.exponential",
    "qdte.privacy.gaussian",
    "qdte.queries.types",
    "qdte.queries.workload",
    "qdte.queries.delta_index",
    "qdte.queries.eval_jax",
    "qdte.measurement.projection",
    "qdte.measurement.consistency",
    "qdte.measurement.measure",
    "qdte.evolution.state",
    "qdte.evolution.candidates",
    "qdte.evolution.initialization",
    "qdte.evolution.scheduler",
    "qdte.evolution.scoring",
    "qdte.evolution.transport",
    "qdte.evolution.gpu_candidates",
    "qdte.evolution.engine",
    "qdte.eval.metrics",
    "qdte.eval.runtime",
    "qdte.eval.external",
]


@dataclass(frozen=True)
class SimulationResult:
    output_dir: Path
    copied_paths: list[str]
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def public_release_paths() -> list[str]:
    return [*EXTRA_PUBLIC_PATHS, *PUBLIC_VISIBLE_PATHS]


def _safe_remove_output(output_dir: Path, source_root: Path) -> None:
    output = output_dir.resolve()
    source = source_root.resolve()
    if output == source:
        raise ValueError("refusing to remove the source repository")
    if output == output.anchor:
        raise ValueError(f"refusing to remove filesystem root: {output}")
    if len(output.parts) < 3:
        raise ValueError(f"refusing to remove suspiciously broad output path: {output}")
    shutil.rmtree(output)


def copy_release_surface(
    source_root: Path,
    output_dir: Path,
    *,
    paths: list[str] | None = None,
    force: bool = False,
) -> list[str]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        _safe_remove_output(output_dir, source_root)
    output_dir.mkdir(parents=True)

    copied: list[str] = []
    for relpath in paths or public_release_paths():
        source = source_root / relpath
        if not source.exists():
            raise FileNotFoundError(f"public release path is missing: {relpath}")
        dest = output_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, dest)
        copied.append(relpath)
    return copied


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


def _import_smoke_command() -> list[str]:
    module_literal = repr(PUBLIC_IMPORT_MODULES)
    code = (
        f"modules = {module_literal}\n"
        "for module in modules:\n"
        "    __import__(module)\n"
        "print(f'public release import smoke passed: {len(modules)} qdte modules')\n"
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


def simulate_public_release(source_root: Path, output_dir: Path, *, force: bool = False) -> SimulationResult:
    copied = copy_release_surface(source_root, output_dir, force=force)
    _run(["git", "init"], output_dir)
    _run(["git", "add", "."], output_dir)
    verifier = ["python3", "scripts/verify_public_release.py", "--strict", "--root", str(output_dir)]
    verification = _run(verifier, output_dir, check=False)
    import_smoke = None
    qdte_smoke = None
    if verification.returncode == 0:
        import_smoke = _run(_import_smoke_command(), output_dir, check=False)
    if import_smoke is not None and import_smoke.returncode == 0:
        qdte_smoke = _run(_qdte_smoke_command(), output_dir, check=False)
    stdout = verification.stdout
    stderr = verification.stderr
    returncode = verification.returncode
    if import_smoke is not None:
        stdout += import_smoke.stdout
        stderr += import_smoke.stderr
        if import_smoke.returncode != 0:
            returncode = import_smoke.returncode
    if qdte_smoke is not None:
        stdout += qdte_smoke.stdout
        stderr += qdte_smoke.stderr
        if qdte_smoke.returncode == 0:
            stdout += "public release QDTE smoke passed\n"
        else:
            returncode = qdte_smoke.returncode
    return SimulationResult(
        output_dir=output_dir,
        copied_paths=copied,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-destructively simulate the clean public release branch by copying "
            "only public files into a temporary git repository and running the "
            "strict public-release verifier there."
        )
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/private_de_public_release_strict_check"),
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = simulate_public_release(args.root, args.output_dir, force=args.force)
    except Exception as exc:
        print(f"public release strict simulation failed before verification: {exc}", file=sys.stderr)
        return 1

    print(f"source repo: {args.root.resolve()}")
    print(f"simulated release repo: {result.output_dir.resolve()}")
    print(f"copied public paths: {len(result.copied_paths)}")
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.ok:
        print("public release strict simulation passed")
        return 0
    print("public release strict simulation failed")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
