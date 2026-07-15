#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PUBLIC_VISIBLE_PATHS = [
    "README.md",
    "configs/smoke.yaml",
    "configs/adult_qdte.yaml",
    "configs/acs_qdte.yaml",
    "configs/br2000_qdte.yaml",
    "configs/nltcs_qdte.yaml",
    "configs/adult_sage_strong.yaml",
    "configs/acs_sage_strong.yaml",
    "configs/br2000_sage_strong.yaml",
    "configs/nltcs_sage_strong.yaml",
    "configs/integrated_sage_qdte_smoke.yaml",
    "configs/variants/dp_release_profile_overlay.yaml",
    "configs/variants/qdte_gsd_breadth_seed0_manifest.yaml",
    "configs/variants/qdte_gsd_converged_seed0_manifest.yaml",
    "configs/variants/qdte_pa_diag16_overlay.yaml",
    "configs/variants/qdte_structured_breadth_tail_v1.yaml",
    "configs/variants/qdte_structured_converged_batch_v1.yaml",
    "configs/variants/qdte_structured_converged_tail_v1.yaml",
    "configs/variants/qdte_structured_fission_refit_v2_overlay.yaml",
    "configs/variants/qdte_structured_search_aware_v1_overlay.yaml",
    "configs/variants/qdte_structured_standard_v2_overlay.yaml",
    "docs/DP_BOUNDARY.md",
    "docs/CODE_REVIEW_GUIDE.md",
    "docs/EXTERNAL_BASELINES.md",
    "docs/PUBLIC_RELEASE_MANIFEST_20260706.md",
    "docs/QDTE_PAPER_CLAIM_MATRIX_20260711.json",
    "docs/QDTE_PAPER_CLAIM_MATRIX_20260711.md",
    "docs/REPRODUCIBILITY.md",
    "qdte",
    "scripts/check_env.py",
    "scripts/path_defaults.py",
    "scripts/run_qdte.py",
    "scripts/measure_qdte_transcript.py",
    "scripts/generate_qdte_from_transcript.py",
    "scripts/smoke_qdte.py",
    "scripts/audit_baseline_admission.py",
    "scripts/audit_gpu_provenance.py",
    "scripts/audit_integrated_sage_qdte_run.py",
    "scripts/audit_original_protocol_baselines.py",
    "scripts/audit_paper_result_state.py",
    "scripts/audit_public_release_plan.py",
    "scripts/audit_reproducibility_docs.py",
    "scripts/audit_qdte_paper_claim_matrix.py",
    "scripts/archive_qdte_paper_package.py",
    "scripts/collect_baseline_calibration.py",
    "scripts/collect_external_results.py",
    "scripts/collect_gem_diagnostics.py",
    "scripts/collect_gem_remap_all4.py",
    "scripts/collect_privmrf_official.py",
    "scripts/collect_rap_stress.py",
    "scripts/collect_rappp_diagnostics.py",
    "scripts/collect_rappp_paper_grid.py",
    "scripts/collect_sage_ablation.py",
    "scripts/create_public_release_repo.py",
    "scripts/create_external_input_package.py",
    "scripts/encode_external_synthetic_csv.py",
    "scripts/evaluate_external_synthetic.py",
    "scripts/evaluate_rappp_paper_metrics.py",
    "scripts/export_rappp_acs_folktables.py",
    "scripts/package_paper_results.py",
    "scripts/package_qdte_paper_results.py",
    "scripts/plan_baseline_calibration.py",
    "scripts/plan_external_experiments.py",
    "scripts/plan_public_release_branch.py",
    "scripts/plan_sage_ablation.py",
    "scripts/plot_certified_adaptive_results.py",
    "scripts/plot_external_results.py",
    "scripts/plot_paper_summary_results.py",
    "scripts/plot_qdte_paper_results.py",
    "scripts/rehearse_public_release_branch.py",
    "scripts/run_ablation.py",
    "scripts/run_adaptive_selection_ablation.py",
    "scripts/run_integrated_sage_qdte.py",
    "scripts/run_nonnegative_projection_pilot.py",
    "scripts/run_orthogonal_low_budget_pilot.py",
    "scripts/run_privmrf_official.py",
    "scripts/run_rappp_official_acs.py",
    "scripts/run_rappp_official_paper_grid.py",
    "scripts/run_sage_external.py",
    "scripts/run_official_gsd_on_qdte_workload.py",
    "scripts/gsd_released_target_utils.py",
    "scripts/run_qdte_fission_refit_external.py",
    "scripts/run_same_target_gsd_generator.py",
    "scripts/materialize_gsd_measurement.py",
    "scripts/materialize_row_realizable_target.py",
    "scripts/materialize_teacher_target.py",
    "scripts/compute_transfer_gap_diagnostics.py",
    "scripts/reproject_measurements.py",
    "scripts/summarize_rtp_transfer_evidence.py",
    "scripts/simulate_public_release.py",
    "scripts/verify_paper_package.py",
    "scripts/verify_paper_package_tarball.py",
    "scripts/verify_paper_claims.py",
    "scripts/verify_qdte_paper_package.py",
    "scripts/verify_public_release.py",
    "scripts/write_external_input_profile.py",
    "scripts/write_external_workload_groups.py",
    "scripts/write_paper_tables.py",
    "tests/conftest.py",
    "tests/test_external_evaluator.py",
    "tests/test_audit_baseline_admission.py",
    "tests/test_audit_gpu_provenance.py",
    "tests/test_audit_original_protocol_baselines.py",
    "tests/test_audit_paper_result_state.py",
    "tests/test_create_public_release_repo.py",
    "tests/test_run_sage_external.py",
    "tests/test_simulate_public_release.py",
    "tests/test_rehearse_public_release_branch.py",
    "tests/test_verify_paper_package.py",
    "tests/test_verify_paper_package_tarball.py",
    "tests/test_verify_paper_claims.py",
    "tests/test_verify_public_release.py",
    "tests/test_archive_qdte_paper_package.py",
    "tests/test_audit_qdte_paper_claim_matrix.py",
    "tests/test_audit_reproducibility_docs.py",
    "tests/test_compute_transfer_gap_diagnostics.py",
    "tests/test_dp_boundary_no_true_answers_in_generator.py",
    "tests/test_consistency_projection.py",
    "tests/test_config_validation.py",
    "tests/test_edit_advantage.py",
    "tests/test_engine_smoke.py",
    "tests/test_gsd_diagnostic_helpers.py",
    "tests/test_hybrid_candidates.py",
    "tests/test_hybrid_transport.py",
    "tests/test_materialize_row_realizable_target.py",
    "tests/test_materialize_teacher_target.py",
    "tests/test_measurement_fission.py",
    "tests/test_measurement.py",
    "tests/test_public_transcript_generation.py",
    "tests/test_nonnegative_projection_theorem.py",
    "tests/test_package_qdte_paper_results.py",
    "tests/test_plot_qdte_paper_results.py",
    "tests/test_preprocess.py",
    "tests/test_run_official_gsd_on_qdte_workload.py",
    "tests/test_official_gsd_released_target_adapter.py",
    "tests/test_run_integrated_sage_qdte.py",
    "tests/test_run_nonnegative_projection_pilot.py",
    "tests/test_run_orthogonal_low_budget_pilot.py",
    "tests/test_run_qdte_fission_refit_external.py",
    "tests/test_reproject_measurements.py",
    "tests/test_transport.py",
    "tests/test_verify_qdte_paper_package.py",
]

PRIVATE_IGNORED_PATHS = [
    "data/2014/1-Year/ss14pca.csv",
    "docs/HANDOFF.md",
    "docs/archive/HANDOFF_20260706_1519_full_history.md",
    "docs/EXTERNAL_BASELINE_ADMISSION_MATRIX_20260706.md",
    "docs/GPU_EXPERIMENT_QUEUE_20260706.md",
    "docs/RESULTS_PACKAGE.md",
    "docs/PAPER_PAGE_BUDGET_PLACEMENT_20260706.md",
    "configs/acs_ca_income_rappp_sage.yaml",
    "sage_paper/usenix_sage_draft/main.tex",
    "resources/private_de_reject_version/main.tex",
    "outputs/adaptive_selection_fixedrho1_20260701_smoke_s0.log",
]

INTERNAL_PATTERNS = [
    "AGENTS.md",
    "QDTE_FULL_IMPLEMENTATION_PLAN_FOR_CODEX.md",
    "architecture.md",
    "architecture_zh.md",
    "docs/HANDOFF.md",
    "docs/archive/*",
    "docs/codex_tasks/*",
    "docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md",
    "docs/QDTE_ABLATION_SUMMARY.md",
    "docs/QDTE_FINAL_EXPERIMENT_PLAN.md",
    "docs/PAPER_*",
    "docs/SAGE_*",
    "docs/RRC_VOI_*",
    "docs/OLD_REJECTION_*",
    "docs/USENIX_*",
    "docs/*EXPERT*",
    "docs/*MEMO*",
    "docs/*REJECT*",
    "configs/*_rappp_sage.yaml",
    "data/*",
    "outputs/*",
    "resources/*",
    "sage_paper/*",
]

KNOWN_TRACKED_INTERNAL = {
    "AGENTS.md",
    "QDTE_FULL_IMPLEMENTATION_PLAN_FOR_CODEX.md",
    "architecture.md",
    "architecture_zh.md",
    "docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md",
    "docs/HANDOFF.md",
    "docs/QDTE_ABLATION_SUMMARY.md",
    "docs/QDTE_FINAL_EXPERIMENT_PLAN.md",
    "docs/codex_tasks/CODEX_QDTE_HANDOFF_AND_STEP1_PROMPT.md",
    "docs/codex_tasks/QDTE_STEP1_AUDIT_PATCH_FOR_CODEX.md",
}

BANNED_PUBLIC_TOKENS = [
    "/home/" + "qian" + "qiu",
    "/train" + "34",
    "/mnt" + "/",
]


@dataclass
class CheckResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _run_git(args: list[str], root: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _repo_root(start: Path) -> Path:
    proc = _run_git(["rev-parse", "--show-toplevel"], start)
    return Path(proc.stdout.strip())


def _is_ignored(path: str, root: Path) -> bool:
    proc = _run_git(["check-ignore", "--no-index", "-q", path], root, check=False)
    return proc.returncode == 0


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _is_explicit_public_path(path: str, root: Path) -> bool:
    for public_path in PUBLIC_VISIBLE_PATHS:
        if path == public_path:
            return True
        public_root = root / public_path
        if public_root.is_dir() and path.startswith(public_path.rstrip("/") + "/"):
            return True
    return False


def _git_lines(args: list[str], root: Path) -> list[str]:
    proc = _run_git(args, root)
    return [line for line in proc.stdout.splitlines() if line]


def _scan_banned_tokens(paths: list[str], root: Path, errors: list[str]) -> None:
    for path in sorted(set(paths)):
        full_path = root / path
        if not full_path.is_file():
            continue
        try:
            text = full_path.read_text(errors="ignore")
        except Exception as exc:
            errors.append(f"failed to scan release path {path}: {exc}")
            continue
        for token in BANNED_PUBLIC_TOKENS:
            if token in text:
                errors.append(f"release path contains local/private token {token!r}: {path}")


def verify(root: Path, strict: bool) -> CheckResult:
    errors: list[str] = []
    warnings: list[str] = []

    for path in PUBLIC_VISIBLE_PATHS:
        full_path = root / path
        if not full_path.exists():
            errors.append(f"public release path is missing: {path}")
            continue
        if _is_ignored(path, root):
            errors.append(f"public release path is ignored: {path}")

    for path in PRIVATE_IGNORED_PATHS:
        if not _is_ignored(path, root):
            errors.append(f"private/internal path is visible to git: {path}")

    tracked = set(_git_lines(["ls-files"], root))
    tracked_internal = sorted(path for path in tracked if _matches_any(path, INTERNAL_PATTERNS))
    if tracked_internal:
        unknown = sorted(set(tracked_internal) - KNOWN_TRACKED_INTERNAL)
        known = sorted(set(tracked_internal) & KNOWN_TRACKED_INTERNAL)
        if unknown:
            errors.extend(f"unexpected tracked internal path: {path}" for path in unknown)
        if known and strict:
            errors.extend(f"tracked internal path must be removed on release branch: {path}" for path in known)
        elif known:
            warnings.extend(f"known tracked internal path remains for release-branch cleanup: {path}" for path in known)

    visible_untracked = set(_git_lines(["ls-files", "--others", "--exclude-standard"], root))
    visible_internal = sorted(path for path in visible_untracked if _matches_any(path, INTERNAL_PATTERNS))
    if visible_internal:
        errors.extend(f"untracked internal path is not ignored: {path}" for path in visible_internal)

    explicit_public_untracked = [
        path for path in visible_untracked if _is_explicit_public_path(path, root)
    ]
    release_paths = [
        path
        for path in [*tracked, *explicit_public_untracked, *PUBLIC_VISIBLE_PATHS]
        if not _matches_any(path, INTERNAL_PATTERNS)
    ]
    _scan_banned_tokens(release_paths, root, errors)

    return CheckResult(errors=errors, warnings=warnings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the repository surface intended for the public QDTE code release."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on all tracked internal files. Use this on the final public release branch.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository path. Defaults to the current working directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root(args.root.resolve())
    result = verify(root=root, strict=args.strict)

    mode = "strict" if args.strict else "audit"
    print(f"public release verification mode: {mode}")
    print(f"repo: {root}")

    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")

    if result.ok:
        print("public release verification passed")
        return 0
    print("public release verification failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
