#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_REPRODUCIBILITY_SNIPPETS = [
    "conda run -n qdte python scripts/check_env.py",
    "conda run -n qdte python scripts/smoke_qdte.py --mode dp --rows 120 --max-iters 2",
    "privacy.mode=dp",
    "privacy.public_n_rows",
    "offline evaluation metrics",
    "conda run -n qdte python scripts/run_paper_readiness_checks.py",
    "python3 scripts/audit_gpu_provenance.py",
    "conda run -n qdte python scripts/audit_baseline_admission.py",
    "python3 scripts/audit_original_protocol_baselines.py",
    "original-protocol baseline evidence",
    "conda run -n qdte python scripts/audit_qdte_paper_claim_matrix.py",
    "conda run -n qdte python scripts/package_qdte_paper_results.py --force",
    "conda run -n qdte python scripts/verify_qdte_paper_package.py",
    "conda run -n qdte python scripts/archive_qdte_paper_package.py",
    "conda run -n qdte python scripts/verify_paper_package_tarball.py",
    "QDTE-Standard",
    "QDTE-Structured",
    "QDTE-FissionRefit",
    "Encoded column cardinalities are treated as public known schema information",
    "The evaluator answers the public workload exactly on `real_encoded.npy` and on",
    "normalizes both answer vectors by their row counts",
    "0.5 * sum(abs(q(D_syn) / |D_syn| - q(D_real) / |D_real|))",
    "`full_true_avg_tvd` is the mean over these block TVDs",
    "These exact true answers are offline evaluation artifacts only",
    "python3 scripts/audit_public_release_plan.py",
    "conda run -n qdte python scripts/simulate_public_release.py",
    "conda run -n qdte python scripts/rehearse_public_release_branch.py",
    "conda run -n qdte python scripts/create_public_release_repo.py",
    "clean `public-release-sage` Git branch",
    "qdte_paper_package_20260711.tar.gz",
    "7bcacf49270d3f478deca9c8375b3f30b51502ca9d8c54caa812f93ce2972f7b",
    "docs/PUBLIC_RELEASE_MANIFEST_20260706.md",
    "per-group measurement",
    "privacy ledger",
    "scripts/measure_qdte_transcript.py",
    "scripts/generate_qdte_from_transcript.py",
    "tests/test_public_transcript_generation.py",
]

REQUIRED_PUBLIC_MANIFEST_SNIPPETS = [
    "scripts/smoke_qdte.py",
    "scripts/measure_qdte_transcript.py",
    "scripts/generate_qdte_from_transcript.py",
    "scripts/audit_reproducibility_docs.py",
    "scripts/audit_original_protocol_baselines.py",
    "conda run -n qdte python scripts/smoke_qdte.py --mode dp --rows 120 --max-iters 2",
    "short DP-mode QDTE end-to-end smoke",
    "scripts/rehearse_public_release_branch.py",
    "scripts/create_public_release_repo.py",
    "scripts/package_qdte_paper_results.py",
    "scripts/verify_qdte_paper_package.py",
    "scripts/archive_qdte_paper_package.py",
    "configs/variants/qdte_structured_standard_v2_overlay.yaml",
    "qdte_paper_package_20260711.tar.gz",
    "7bcacf49270d3f478deca9c8375b3f30b51502ca9d8c54caa812f93ce2972f7b",
]

REQUIRED_README_SNIPPETS = [
    "conda run -n qdte python scripts/smoke_qdte.py --mode dp --rows 120 --max-iters 2",
    "privacy.mode=dp",
    "privacy.public_n_rows",
    "external evaluator 的 `full_true_*` 指标同样只属于离线评估",
    "`real_encoded.npy` 和 `synthetic_encoded.npy`",
    "0.5 * sum(abs(q(D_syn) / |D_syn| - q(D_real) / |D_real|))",
    "`full_true_avg_tvd` 和 `full_true_max_tvd`",
    "docs/PUBLIC_RELEASE_MANIFEST_20260706.md",
    "python3 scripts/audit_original_protocol_baselines.py",
    "strict same-protocol 主表分开报告",
    "QDTE-Standard",
    "QDTE-Structured",
    "QDTE-FissionRefit",
    "Encoded attribute cardinalities are treated as public and known",
    "conda run -n qdte python scripts/package_qdte_paper_results.py --force",
    "conda run -n qdte python scripts/verify_qdte_paper_package.py",
]

REQUIRED_CODE_REVIEW_SNIPPETS = [
    "residual[q] = target_projected[q] - answer_syn[q]",
    "privacy.public_n_rows",
    "per-Gaussian-vector ledger",
    "tests/test_dp_boundary_no_true_answers_in_generator.py",
    "tests/test_public_transcript_generation.py",
    "tests/test_nonnegative_projection_theorem.py",
    "python3 scripts/audit_reproducibility_docs.py",
    "python3 scripts/verify_public_release.py",
    "QDTE-Standard",
    "Frozen Research Status",
]

REQUIRED_CODE_REVIEW_PATHS = [
    "qdte/evolution/scoring.py",
    "qdte/evolution/candidates.py",
    "qdte/evolution/transport.py",
    "qdte/evolution/engine.py",
    "qdte/measurement/measure.py",
    "qdte/measurement/consistency.py",
    "qdte/measurement/public_artifact.py",
    "scripts/measure_qdte_transcript.py",
    "scripts/generate_qdte_from_transcript.py",
    "scripts/run_nonnegative_projection_pilot.py",
    "scripts/reproject_measurements.py",
    "scripts/run_adaptive_selection_ablation.py",
    "scripts/run_integrated_sage_qdte.py",
    "scripts/audit_integrated_sage_qdte_run.py",
    "scripts/run_orthogonal_low_budget_pilot.py",
    "configs/variants/dp_release_profile_overlay.yaml",
    "docs/DP_BOUNDARY.md",
    "docs/REPRODUCIBILITY.md",
    "tests/test_edit_advantage.py",
    "tests/test_transport.py",
    "tests/test_preprocess.py",
    "tests/test_config_validation.py",
    "tests/test_public_transcript_generation.py",
    "tests/test_dp_boundary_no_true_answers_in_generator.py",
    "tests/test_run_integrated_sage_qdte.py",
    "tests/test_run_orthogonal_low_budget_pilot.py",
    "tests/test_consistency_projection.py",
    "tests/test_nonnegative_projection_theorem.py",
    "tests/test_reproject_measurements.py",
]


@dataclass(frozen=True)
class DocumentCheck:
    path: str
    snippets: list[str]


CHECKS = [
    DocumentCheck("docs/REPRODUCIBILITY.md", REQUIRED_REPRODUCIBILITY_SNIPPETS),
    DocumentCheck("docs/PUBLIC_RELEASE_MANIFEST_20260706.md", REQUIRED_PUBLIC_MANIFEST_SNIPPETS),
    DocumentCheck("README.md", REQUIRED_README_SNIPPETS),
    DocumentCheck("docs/CODE_REVIEW_GUIDE.md", REQUIRED_CODE_REVIEW_SNIPPETS),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def audit(root: Path) -> list[str]:
    errors: list[str] = []
    for check in CHECKS:
        path = root / check.path
        if not path.is_file():
            errors.append(f"missing reproducibility document: {check.path}")
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in check.snippets:
            if snippet not in text:
                errors.append(f"{check.path} is missing required snippet: {snippet}")
    for relpath in REQUIRED_CODE_REVIEW_PATHS:
        if not (root / relpath).is_file():
            errors.append(f"code review guide references missing public path: {relpath}")
    return errors


def main() -> int:
    root = repo_root()
    errors = audit(root)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("reproducibility documentation audit failed")
        return 1
    print("reproducibility documentation audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
