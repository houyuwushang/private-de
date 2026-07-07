from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_baseline_root() -> Path:
    local_sibling = repo_root().parent / "baseline" / repo_root().name
    if local_sibling.exists():
        return local_sibling
    return repo_root() / "external_workspace"


def baseline_root() -> Path:
    return _env_path("SAGE_BASELINE_ROOT", _default_baseline_root())


def external_inputs() -> Path:
    return _env_path("SAGE_EXTERNAL_INPUTS", baseline_root() / "external_inputs")


def external_runs() -> Path:
    return _env_path("SAGE_EXTERNAL_RUNS", baseline_root() / "external_runs")


def external_results() -> Path:
    return _env_path("SAGE_EXTERNAL_RESULTS", baseline_root() / "external_results")


def external_calibration_runs() -> Path:
    return _env_path("SAGE_EXTERNAL_CALIBRATION_RUNS", baseline_root() / "external_calibration_runs")


def sage_paper_dir() -> Path:
    return _env_path("SAGE_PAPER_RESULTS_DIR", baseline_root() / "sage_paper")


def paper_package_dir() -> Path:
    return _env_path(
        "SAGE_PAPER_PACKAGE_DIR",
        external_results() / "paper_package_seed0to4_20260706",
    )


def legacy_paper_package_dir() -> Path:
    return _env_path(
        "SAGE_LEGACY_PAPER_PACKAGE_DIR",
        external_results() / "paper_package_20260706",
    )


def rappp_root() -> Path:
    return _env_path("SAGE_RAPPP_ROOT", baseline_root() / "relaxed-adaptive-projection")


def privmrf_root() -> Path:
    return _env_path("SAGE_PRIVMRF_ROOT", baseline_root() / "PrivMRF")
