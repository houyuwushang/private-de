# Public Release Manifest

Date: 2026-07-06

This document defines the intended public GitHub surface for the SAGE/QDTE code release. It separates public reproducibility assets from internal handoff notes, paper drafts, expert memos, and experiment scratch artifacts.

The release goal is:

```text
public repository = code + configs + reproducibility scripts + tests + concise docs
internal workspace = paper drafts + handoff + expert discussions + raw scratch outputs
```

## Public Release Principles

1. Keep the repository code-focused.
2. Include enough configs and scripts to reproduce the paper-facing runs.
3. Keep the DP boundary visible in README and tests.
4. Do not publish handoff notes, rejected-version material, expert memos, or long internal planning documents.
5. Do not commit large generated outputs, caches, conda environments, or local baseline repositories.

## Include In Public Release

### Core Package

```text
qdte/
```

Include all source files under:

```text
qdte/config.py
qdte/config_validation.py
qdte/dataio.py
qdte/preprocess.py
qdte/schema.py
qdte/privacy/
qdte/queries/
qdte/measurement/
qdte/evolution/
qdte/eval/
```

Exclude:

```text
__pycache__/
*.pyc
```

### Configs

Include:

```text
configs/smoke.yaml
configs/adult_qdte.yaml
configs/acs_qdte.yaml
configs/br2000_qdte.yaml
configs/nltcs_qdte.yaml
configs/adult_sage_strong.yaml
configs/acs_sage_strong.yaml
configs/br2000_sage_strong.yaml
configs/nltcs_sage_strong.yaml
```

Optional GPU/debug configs can remain if documented:

```text
configs/adult_qdte_gpu_highpower.yaml
configs/adult_qdte_gpu_highpower_directed_group.yaml
```

### Reproducibility Scripts

Include core scripts:

```text
scripts/check_env.py
scripts/path_defaults.py
scripts/audit_public_release_plan.py
scripts/audit_gpu_provenance.py
scripts/audit_original_protocol_baselines.py
scripts/audit_paper_result_state.py
scripts/run_qdte.py
scripts/smoke_qdte.py
scripts/run_sage_external.py
scripts/evaluate_external_synthetic.py
scripts/collect_external_results.py
scripts/plan_external_experiments.py
scripts/write_external_input_profile.py
scripts/write_external_workload_groups.py
scripts/plot_external_results.py
scripts/plot_paper_summary_results.py
scripts/simulate_public_release.py
scripts/rehearse_public_release_branch.py
scripts/create_public_release_repo.py
scripts/write_paper_tables.py
scripts/package_paper_results.py
scripts/verify_paper_package.py
scripts/verify_paper_package_tarball.py
scripts/verify_paper_claims.py
scripts/verify_public_release.py
```

Include ablation and certified-selector scripts if the final paper reports those artifacts:

```text
scripts/run_ablation.py
scripts/plan_sage_ablation.py
scripts/collect_sage_ablation.py
scripts/run_adaptive_selection_ablation.py
scripts/plot_certified_adaptive_results.py
```

Include appendix/audit baseline collection helpers only if their external dependency paths are documented:

```text
scripts/collect_baseline_calibration.py
scripts/plan_baseline_calibration.py
scripts/collect_rap_stress.py
scripts/collect_gem_diagnostics.py
scripts/collect_gem_remap_all4.py
```

Include original-protocol reproduced-baseline helpers when reporting the RAP++
official and PrivMRF official appendix table:

```text
scripts/audit_baseline_admission.py
scripts/collect_privmrf_official.py
scripts/collect_rappp_diagnostics.py
scripts/collect_rappp_paper_grid.py
scripts/create_external_input_package.py
scripts/encode_external_synthetic_csv.py
scripts/evaluate_rappp_paper_metrics.py
scripts/export_rappp_acs_folktables.py
scripts/run_privmrf_official.py
scripts/run_rappp_official_acs.py
scripts/run_rappp_official_paper_grid.py
```

### Tests

Include:

```text
tests/
```

The minimum public verification command should remain:

```text
conda run -n qdte pytest -q
```

The release should explicitly mention that GPU tests depend on JAX/CUDA availability.

### Documentation

Include only concise public documentation:

```text
README.md
docs/PUBLIC_RELEASE_MANIFEST_20260706.md
```

Public docs now available:

```text
docs/REPRODUCIBILITY.md
docs/DP_BOUNDARY.md
docs/EXTERNAL_BASELINES.md
```

Do not publish long working notes as the primary public documentation.

Do not commit raw downloaded datasets. Public releases should provide download
or preprocessing instructions instead:

```text
data/
```

Result-package descriptions that contain local artifact paths should be stored
with the released experiment artifact, not committed to the clean source
repository:

```text
docs/RESULTS_PACKAGE.md
```

## Exclude From Public Release

### Internal Workflow and Handoff

```text
AGENTS.md
docs/HANDOFF.md
docs/archive/
docs/codex_tasks/
QDTE_FULL_IMPLEMENTATION_PLAN_FOR_CODEX.md
architecture.md
architecture_zh.md
```

### Paper Drafts and Expert Memos

```text
docs/PAPER_*
docs/SAGE_*
docs/RRC_VOI_*
docs/OLD_REJECTION_*
docs/EXTERNAL_BASELINE_ADMISSION_*
docs/GPU_EXPERIMENT_QUEUE_*
docs/RESULTS_PACKAGE.md
docs/*DRAFT*
docs/*EXPERT*
docs/*MEMO*
docs/*PLAN*
docs/*PROOF*
docs/*REJECT*
```

Exception:

If a paper-facing theorem or results document is meant to be public, convert it into a short public doc under a stable name such as:

```text
docs/DP_BOUNDARY.md
docs/REPRODUCIBILITY.md
```

Do not publish raw expert-discussion files.

### Experiment Scratch and Generated Outputs

```text
outputs/
resources/
sage_runs/
sage_paper/
configs/*_rappp_sage.yaml
```

External result packages should be archived separately or released as a versioned artifact, not committed into the source repository.

### External Baseline Repositories

Do not vendor local external baseline repositories into this repository:

```text
<external-baseline-workspace>/
```

Public release should document how to fetch or configure baselines instead.

## Current Tracked Files That Should Be Removed From Public Branch

These files are currently tracked and should be removed from a clean public branch with `git rm --cached` or by creating a fresh release branch:

```text
AGENTS.md
QDTE_FULL_IMPLEMENTATION_PLAN_FOR_CODEX.md
architecture.md
architecture_zh.md
docs/HANDOFF.md
docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md
docs/QDTE_ABLATION_SUMMARY.md
docs/QDTE_FINAL_EXPERIMENT_PLAN.md
docs/codex_tasks/CODEX_QDTE_HANDOFF_AND_STEP1_PROMPT.md
docs/codex_tasks/QDTE_STEP1_AUDIT_PATCH_FOR_CODEX.md
```

The `.gitignore` has been expanded so similar new internal files will not be added accidentally, but already tracked files still require explicit removal from the public release branch.

The release verifier checks both sides of this boundary: public docs such as
`docs/REPRODUCIBILITY.md`, `docs/DP_BOUNDARY.md`, and
`docs/EXTERNAL_BASELINES.md` must exist and must not be ignored, while internal
handoff, paper-draft, expert-note, raw-data, and generated-output paths must be
ignored or removed from the public branch.

## Research-Branch Dry Run Status

The current research branch is expected to pass `verify_public_release.py` in
audit mode and fail `--strict` mode until the public release branch removes the
already-tracked internal notes from the index.

The strict-mode removal list is:

```text
AGENTS.md
QDTE_FULL_IMPLEMENTATION_PLAN_FOR_CODEX.md
architecture.md
architecture_zh.md
docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md
docs/HANDOFF.md
docs/QDTE_ABLATION_SUMMARY.md
docs/QDTE_FINAL_EXPERIMENT_PLAN.md
docs/codex_tasks/CODEX_QDTE_HANDOFF_AND_STEP1_PROMPT.md
docs/codex_tasks/QDTE_STEP1_AUDIT_PATCH_FOR_CODEX.md
```

The current paper artifact should be shipped as a separate result archive, not
committed to the source repository:

```text
paper_package_seed0to4_20260706.tar.gz
sha256: a918441491334904c21d3b9a509164437731ddf3dddab8f1e4705bfb8d185086
```

Manuscript-local helpers such as `scripts/audit_usenix_pdf_format.py` are not
part of the explicit public `git add` list below unless the release scope is
expanded to include the paper source. Diagnostic helpers are included only when
their corresponding appendix evidence is part of the artifact package.

## Release Branch Cleanup Command Plan

Do not run these commands on the active research branch unless it has already
been intentionally converted into the public release branch. The recommended
workflow is to create a separate branch, remove already-tracked internal files
from the index only, and then verify strict release cleanliness.

```bash
git status --short --untracked-files=all

git switch -c public-release-sage

git rm --cached \
  AGENTS.md \
  QDTE_FULL_IMPLEMENTATION_PLAN_FOR_CODEX.md \
  architecture.md \
  architecture_zh.md \
  docs/HANDOFF.md \
  docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md \
  docs/QDTE_ABLATION_SUMMARY.md \
  docs/QDTE_FINAL_EXPERIMENT_PLAN.md \
  docs/codex_tasks/CODEX_QDTE_HANDOFF_AND_STEP1_PROMPT.md \
  docs/codex_tasks/QDTE_STEP1_AUDIT_PATCH_FOR_CODEX.md

git add -u

git add \
  .gitignore \
  README.md \
  configs/smoke.yaml \
  configs/adult_qdte.yaml \
  configs/acs_qdte.yaml \
  configs/br2000_qdte.yaml \
  configs/nltcs_qdte.yaml \
  configs/adult_sage_strong.yaml \
  configs/acs_sage_strong.yaml \
  configs/br2000_sage_strong.yaml \
  configs/nltcs_sage_strong.yaml \
  docs/DP_BOUNDARY.md \
  docs/EXTERNAL_BASELINES.md \
  docs/PUBLIC_RELEASE_MANIFEST_20260706.md \
  docs/REPRODUCIBILITY.md \
  qdte \
  scripts/check_env.py \
  scripts/path_defaults.py \
  scripts/run_qdte.py \
  scripts/smoke_qdte.py \
  scripts/audit_baseline_admission.py \
  scripts/audit_gpu_provenance.py \
  scripts/audit_original_protocol_baselines.py \
  scripts/audit_paper_result_state.py \
  scripts/audit_public_release_plan.py \
  scripts/collect_baseline_calibration.py \
  scripts/collect_external_results.py \
  scripts/collect_gem_diagnostics.py \
  scripts/collect_gem_remap_all4.py \
  scripts/collect_privmrf_official.py \
  scripts/collect_rap_stress.py \
  scripts/collect_rappp_diagnostics.py \
  scripts/collect_rappp_paper_grid.py \
  scripts/collect_sage_ablation.py \
  scripts/create_external_input_package.py \
  scripts/encode_external_synthetic_csv.py \
  scripts/evaluate_external_synthetic.py \
  scripts/evaluate_rappp_paper_metrics.py \
  scripts/export_rappp_acs_folktables.py \
  scripts/package_paper_results.py \
  scripts/verify_paper_package.py \
  scripts/verify_paper_package_tarball.py \
  scripts/verify_paper_claims.py \
  scripts/plan_baseline_calibration.py \
  scripts/plan_external_experiments.py \
  scripts/plan_sage_ablation.py \
  scripts/plot_certified_adaptive_results.py \
  scripts/plot_external_results.py \
  scripts/plot_paper_summary_results.py \
  scripts/simulate_public_release.py \
  scripts/rehearse_public_release_branch.py \
  scripts/create_public_release_repo.py \
  scripts/run_ablation.py \
  scripts/run_adaptive_selection_ablation.py \
  scripts/run_privmrf_official.py \
  scripts/run_rappp_official_acs.py \
  scripts/run_rappp_official_paper_grid.py \
  scripts/run_sage_external.py \
  scripts/verify_public_release.py \
  scripts/write_external_input_profile.py \
  scripts/write_external_workload_groups.py \
  scripts/write_paper_tables.py \
  tests/test_external_evaluator.py \
  tests/test_audit_baseline_admission.py \
  tests/test_audit_gpu_provenance.py \
  tests/test_audit_original_protocol_baselines.py \
  tests/test_audit_paper_result_state.py \
  tests/test_run_sage_external.py \
  tests/test_simulate_public_release.py \
  tests/test_rehearse_public_release_branch.py \
  tests/test_create_public_release_repo.py \
  tests/test_verify_paper_package.py \
  tests/test_verify_paper_package_tarball.py \
  tests/test_verify_paper_claims.py \
  tests/test_verify_public_release.py

python3 scripts/verify_public_release.py --strict
conda run -n qdte python scripts/simulate_public_release.py --output-dir /tmp/private_de_public_release_strict_check --force
conda run -n qdte python scripts/rehearse_public_release_branch.py --output-dir /tmp/private_de_public_release_branch_rehearsal --force
conda run -n qdte pytest -q
git diff --check
git status --short
```

Expected result on the release branch:

```text
python3 scripts/verify_public_release.py --strict
# public release verification passed

conda run -n qdte python scripts/simulate_public_release.py --output-dir /tmp/private_de_public_release_strict_check --force
# public release strict simulation passed

conda run -n qdte python scripts/rehearse_public_release_branch.py --output-dir /tmp/private_de_public_release_branch_rehearsal --force
# public release branch rehearsal passed

conda run -n qdte pytest -q
# all tests pass
```

The initial `git status --short --untracked-files=all` is a preflight review.
Proceed only when every changed or untracked file is intentionally part of the
release branch operation or has been parked elsewhere. After the cleanup,
`verify_public_release.py --strict`, strict simulation, branch rehearsal, and
`git status --short` together provide the final index check: no internal file
may remain tracked, and no release-visible path should contain local/private
tokens.

The `git rm --cached` operation removes the files from the public branch index
but keeps local working-tree copies because the paths are covered by
`.gitignore`. If a file is still needed internally, keep it in the private
research branch or private artifact archive, not the public release branch.

## Current Visibility Audit

The following untracked public-release files are intentionally visible and can
be added after final review:

```text
configs/adult_sage_strong.yaml
configs/acs_sage_strong.yaml
configs/br2000_sage_strong.yaml
configs/nltcs_sage_strong.yaml
docs/DP_BOUNDARY.md
docs/EXTERNAL_BASELINES.md
docs/PUBLIC_RELEASE_MANIFEST_20260706.md
docs/REPRODUCIBILITY.md
qdte/
scripts/audit_baseline_admission.py
scripts/audit_gpu_provenance.py
scripts/audit_original_protocol_baselines.py
scripts/audit_paper_result_state.py
scripts/audit_public_release_plan.py
scripts/collect_baseline_calibration.py
scripts/collect_external_results.py
scripts/collect_gem_diagnostics.py
scripts/collect_gem_remap_all4.py
scripts/collect_privmrf_official.py
scripts/collect_rap_stress.py
scripts/collect_rappp_diagnostics.py
scripts/collect_rappp_paper_grid.py
scripts/collect_sage_ablation.py
scripts/create_external_input_package.py
scripts/encode_external_synthetic_csv.py
scripts/evaluate_external_synthetic.py
scripts/evaluate_rappp_paper_metrics.py
scripts/export_rappp_acs_folktables.py
scripts/package_paper_results.py
scripts/plan_baseline_calibration.py
scripts/plan_external_experiments.py
scripts/plan_sage_ablation.py
scripts/plot_certified_adaptive_results.py
scripts/plot_external_results.py
scripts/plot_paper_summary_results.py
scripts/simulate_public_release.py
scripts/rehearse_public_release_branch.py
scripts/create_public_release_repo.py
scripts/run_ablation.py
scripts/run_adaptive_selection_ablation.py
scripts/run_privmrf_official.py
scripts/run_rappp_official_acs.py
scripts/run_rappp_official_paper_grid.py
scripts/run_qdte.py
scripts/smoke_qdte.py
scripts/run_sage_external.py
scripts/write_external_input_profile.py
scripts/write_external_workload_groups.py
scripts/write_paper_tables.py
scripts/verify_paper_package.py
scripts/verify_paper_package_tarball.py
scripts/verify_paper_claims.py
scripts/verify_public_release.py
tests/test_external_evaluator.py
tests/test_audit_baseline_admission.py
tests/test_audit_gpu_provenance.py
tests/test_audit_original_protocol_baselines.py
tests/test_audit_paper_result_state.py
tests/test_run_sage_external.py
tests/test_simulate_public_release.py
tests/test_rehearse_public_release_branch.py
tests/test_create_public_release_repo.py
tests/test_verify_paper_package.py
tests/test_verify_paper_package_tarball.py
tests/test_verify_paper_claims.py
tests/test_verify_public_release.py
```

The following generated/internal files are intentionally ignored:

```text
docs/EXTERNAL_BASELINE_ADMISSION_MATRIX_20260706.md
docs/GPU_EXPERIMENT_QUEUE_20260706.md
docs/RESULTS_PACKAGE.md
docs/PAPER_*
docs/SAGE_*
docs/RRC_VOI_*
docs/HANDOFF.md
resources/
sage_paper/
outputs/
```

## README Verification Before Public Release

The README has been updated toward the paper-facing SAGE/QDTE story. Before release, verify it still states:

1. The paper-facing generator uses variance-aware QDTE over noisy/projected measurements.
2. Strong configs are:

```text
configs/adult_sage_strong.yaml
configs/acs_sage_strong.yaml
configs/br2000_sage_strong.yaml
configs/nltcs_sage_strong.yaml
```

3. Exact true answers are offline evaluation only.
4. Main experiments use a static public heterogeneous measurement schedule.
5. Certified adaptive SAGE-Select experiments use transcript-only selection.
6. External baseline wrappers require separate environments and are not vendored.

## Release Verification Checklist

Before pushing the clean public branch:

0. Run the public-release verifier. On the active research branch, audit mode
   should pass with warnings for known historical tracked internal files:

```text
python3 scripts/verify_public_release.py
```

On the final public release branch, strict mode must pass:

```text
python3 scripts/verify_public_release.py --strict
```

The active research branch can also run a non-destructive strict simulation
without modifying the branch index. Run this inside the `qdte` environment so
the simulated release can also execute the QDTE package import smoke and a
short DP-mode QDTE end-to-end smoke:

```text
conda run -n qdte python scripts/simulate_public_release.py --force
```

The active research branch can also rehearse the release-branch index cleanup
in a temporary Git repository:

```text
conda run -n qdte python scripts/rehearse_public_release_branch.py --force
```

To create a persistent clean public release repository outside the active
research branch:

```text
PUBLIC_RELEASE_DIR="${PWD}/../private-de-public-release"
conda run -n qdte python scripts/create_public_release_repo.py --output-dir "${PUBLIC_RELEASE_DIR}" --force --commit
```

1. Confirm tracked files:

```text
git ls-files | sort
```

2. Confirm no internal docs are tracked:

```text
git ls-files | rg '(^AGENTS\.md$|HANDOFF|codex_tasks|QDTE_FULL_IMPLEMENTATION_PLAN|architecture|PAPER_|SAGE_|RRC_VOI_|OLD_REJECTION|EXPERT|MEMO|REJECT)'
```

3. Confirm new internal files are ignored and public docs are visible:

```text
for p in \
  docs/DP_BOUNDARY.md \
  docs/EXTERNAL_BASELINES.md \
  docs/PUBLIC_RELEASE_MANIFEST_20260706.md \
  docs/REPRODUCIBILITY.md \
  docs/EXTERNAL_BASELINE_ADMISSION_MATRIX_20260706.md \
  docs/GPU_EXPERIMENT_QUEUE_20260706.md \
  docs/RESULTS_PACKAGE.md \
  sage_paper/usenix_sage_draft/main.tex \
  resources/private_de_reject_version/main.tex; do
  git check-ignore --no-index -q "$p" && echo "ignored $p" || echo "visible $p"
done
```

Expected output: public docs should print `visible`; generated/internal paths
should print `ignored`.

3. Run tests:

```text
conda run -n qdte pytest -q
```

4. Run a smoke command:

```text
conda run -n qdte python scripts/smoke_qdte.py --mode dp --rows 120 --max-iters 2
```

5. Check import and GPU environment:

```text
conda run -n qdte python scripts/check_env.py
```

6. Verify generated outputs remain outside git:

```text
git status --short
```

## Recommended Next Cleanup Step

Create a clean release branch and remove internal tracked files from that branch only. Do not delete them from the working research branch unless explicitly intended.
