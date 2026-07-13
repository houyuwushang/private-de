# Reproducibility Guide

This guide describes the public reproduction path for QDTE experiments. It assumes the repository contains code, configs, scripts, and tests, while large generated outputs and external baseline repositories are stored outside this source tree.

## 1. Environment

Use the project conda environment:

```bash
conda run -n qdte python scripts/check_env.py
conda run -n qdte pytest -q
```

GPU-backed QDTE runs require a JAX/CUDA environment. `scripts/check_env.py` reports visible JAX devices.

The helper scripts use portable defaults under `external_workspace/`. Set these
environment variables to reuse an existing external experiment workspace:

```bash
export SAGE_BASELINE_ROOT=/path/to/external-baseline-workspace
export SAGE_EXTERNAL_INPUTS=$SAGE_BASELINE_ROOT/external_inputs
export SAGE_EXTERNAL_RUNS=$SAGE_BASELINE_ROOT/external_runs
export SAGE_EXTERNAL_RESULTS=$SAGE_BASELINE_ROOT/external_results
export SAGE_PAPER_PACKAGE_DIR=$SAGE_EXTERNAL_RESULTS/paper_package_seed0to4_20260706
export QDTE_PAPER_PACKAGE_DIR=$SAGE_EXTERNAL_RESULTS/qdte_paper_package_20260711
```

## 2. Input Package Format

Paper-style external experiments use a canonical input directory:

```text
external_inputs/<dataset>/
  raw.csv
  schema.json
  real_encoded.npy
  queries_full.json
  workload_groups.json
  metadata.json
  true_answers_cache.npz        # optional evaluator cache
```

The true-answer cache is for offline evaluation only. It must not be used by QDTE during measurement, generation, selection, stopping, or hyperparameter selection.

## 3. Smoke Run

Run a small DP smoke test that generates its own toy CSV and writes a short
QDTE output directory:

```bash
conda run -n qdte python scripts/smoke_qdte.py --mode dp --rows 120 --max-iters 2
```

The smoke runs in `privacy.mode=dp`. Exact true answers are computed only for
offline evaluation metrics. To run from your own CSV/config instead, call:

```bash
conda run -n qdte python scripts/run_qdte.py \
  --config configs/smoke.yaml \
  --privacy.mode dp \
  --run.input_csv /path/to/input.csv \
  --run.output_dir outputs/smoke_qdte
```

## 4. Strong SAGE Configs

The paper-facing strong configs are:

```text
configs/adult_sage_strong.yaml
configs/acs_sage_strong.yaml
configs/br2000_sage_strong.yaml
configs/nltcs_sage_strong.yaml
```

They use:

```text
privacy.mode = dp
privacy.measurement_mode = static_all
qdte.objective_weighting = variance
qdte.score_backend = dense_gpu
qdte.transport_mode = atom_flow
```

The config paths may point to local canonical input directories. For a public reproduction run, override `run.input_csv` and `run.output_dir` as needed:

```bash
conda run -n qdte python scripts/run_qdte.py \
  --config configs/adult_sage_strong.yaml \
  --run.input_csv /path/to/adult_sage_strong/raw.csv \
  --run.output_dir outputs/adult_sage_strong_qdte
```

## 5. Canonical External SAGE Run

To run SAGE on a canonical external input package:

```bash
conda run -n qdte python scripts/run_sage_external.py \
  --method sage \
  --dataset adult_sage_strong \
  --input-dir /path/to/external_inputs/adult_sage_strong \
  --output-dir /path/to/external_runs/sage/adult_sage_strong/rho1p0/seed0 \
  --rho-total 1.0 \
  --delta 1e-9 \
  --seed 0 \
  --n-syn same_as_real \
  --max-iters 5000 \
  --xla-preallocate
```

This runner disables active true-query evaluation inside SAGE and leaves true answers to the shared offline evaluator.

## 6. Shared Offline Evaluation

Evaluate any row-level synthetic table with:

```bash
conda run -n qdte python scripts/evaluate_external_synthetic.py \
  --input-dir /path/to/external_inputs/adult_sage_strong \
  --synthetic /path/to/run/synthetic_encoded.npy \
  --output /path/to/run/evaluation.json \
  --run-metadata /path/to/run/run_metadata.json \
  --true-answers-cache /path/to/external_inputs/adult_sage_strong/true_answers_cache.npz \
  --no-block-details
```

All methods should be compared through this evaluator after producing `synthetic_encoded.npy`.
The evaluator answers the public workload exactly on `real_encoded.npy` and on
`synthetic_encoded.npy`, normalizes both answer vectors by their row counts,
and writes `full_true_mae`, `full_true_rmse`, and `full_true_max_error` from
their pointwise normalized differences. For public partition/vector blocks it
computes block TVD as:

```text
0.5 * sum(abs(q(D_syn) / |D_syn| - q(D_real) / |D_real|))
```

`full_true_avg_tvd` is the mean over these block TVDs and
`full_true_max_tvd` is their maximum. These exact true answers are offline evaluation artifacts only; they must not be read by DP-mode measurement,
selection, generation, scoring, transport, stopping, or hyperparameter
selection.

## 7. Planning And Collecting External Runs

Plan commands without executing:

```bash
conda run -n qdte python scripts/plan_external_experiments.py \
  --phase full \
  --methods sage,rap,private_gsd_gpu_1m_fulln_audit,private_pgm_aim,private_pgm_mst \
  --datasets adult_sage_strong,acs_sage_strong,br2000_sage_strong,nltcs_sage_strong \
  --rhos 1.0 \
  --seeds 0,1,2,3,4
```

Execute and evaluate:

```bash
conda run -n qdte python scripts/plan_external_experiments.py \
  --phase full \
  --methods sage,rap,private_gsd_gpu_1m_fulln_audit,private_pgm_aim,private_pgm_mst \
  --datasets adult_sage_strong,acs_sage_strong,br2000_sage_strong,nltcs_sage_strong \
  --rhos 1.0 \
  --seeds 0,1,2,3,4 \
  --execute \
  --evaluate
```

Collect evaluated runs:

```bash
conda run -n qdte python scripts/collect_external_results.py \
  --runs-root /path/to/external_runs \
  --output-csv /path/to/external_results/summary_all.csv \
  --output-md /path/to/external_results/summary_all.md
```

## 8. Figures And Tables

Plot canonical result CSVs:

```bash
conda run -n qdte python scripts/plot_external_results.py \
  --summary-csv /path/to/results.csv \
  --out-dir /path/to/figures \
  --metrics full_true_mae,full_true_rmse,full_true_avg_tvd,full_true_max_error,full_true_max_tvd,runtime_seconds \
  --rho 1.0 \
  --formats pdf,png
```

Generate paper tables and package results:

```bash
conda run -n qdte python scripts/package_qdte_paper_results.py --force
conda run -n qdte python scripts/verify_qdte_paper_package.py
conda run -n qdte python scripts/plot_qdte_paper_results.py
conda run -n qdte python scripts/archive_qdte_paper_package.py
conda run -n qdte python scripts/verify_paper_package_tarball.py
```

The QDTE package builder reconstructs tables and figures from declared source
artifacts, records a source hash manifest, and parameterizes external-workspace
paths. The verifier checks package hashes, source hashes, claim facts, and the
absence of machine-local paths. The deterministic archive and sidecar are then
checked byte-for-byte against the package directory.

Audit the current paper-result state without launching experiments:

```bash
python3 scripts/audit_paper_result_state.py
```

The audit prints the current baseline/SAGE ratios, the high-power Private-GSD
configuration shift, Adult SAGE seed0 stability, and key metadata hashes. Use it
before changing table claims or investigating an apparent result discrepancy.

Audit paper-facing GPU provenance without launching experiments:

```bash
python3 scripts/audit_gpu_provenance.py
```

This audit verifies the recorded GPU metadata for QDTE-Standard, Private-GSD, and RAP
rows in the strict same-protocol evidence, and records AIM/MST as CPU-native
Private-PGM baselines.

Check baseline admission tiers without launching experiments:

```bash
conda run -n qdte python scripts/audit_baseline_admission.py
```

Audit original-protocol reproduced baseline grids without launching
experiments:

```bash
python3 scripts/audit_original_protocol_baselines.py
```

Run the full paper-readiness chain from a clean project environment:

```bash
conda run -n qdte python scripts/run_paper_readiness_checks.py
```

This chain rebuilds or checks the manuscript, focused tests, package and
tarball integrity, claim traceability, original-protocol baseline evidence, GPU
provenance, public-release audit, strict public-release simulation, and
whitespace checks.

## 9. Current QDTE Paper Package

The paper-facing method names and claim boundaries are frozen in:

```text
docs/QDTE_PAPER_CLAIM_MATRIX_20260711.json
docs/QDTE_PAPER_CLAIM_MATRIX_20260711.md
```

`QDTE-Standard` is the end-to-end DP default. `QDTE-Structured` is the stronger
same-target controlled-generator profile and is not promoted to the DP default.
`QDTE-FissionRefit` is a released-only two-pass MAE/RMSE variant. RTP and
teacher-target experiments remain transfer-gap diagnostics.

Build, verify, plot, and archive the latest evidence package with:

```bash
conda run -n qdte python scripts/audit_qdte_paper_claim_matrix.py
conda run -n qdte python scripts/package_qdte_paper_results.py --force
conda run -n qdte python scripts/verify_qdte_paper_package.py
conda run -n qdte python scripts/plot_qdte_paper_results.py
conda run -n qdte python scripts/archive_qdte_paper_package.py
conda run -n qdte python scripts/verify_paper_package_tarball.py
```

The package is written to the external-results workspace as
`qdte_paper_package_20260711/`; generated runs are not committed to the code
repository. Its deterministic archive is:

```text
qdte_paper_package_20260711.tar.gz
sha256: 7bcacf49270d3f478deca9c8375b3f30b51502ca9d8c54caa812f93ce2972f7b
```

Controlled generator reproduction uses the exact public target materializer,
the upstream official GSD runner, and frozen public manifests:

```text
scripts/materialize_gsd_measurement.py
scripts/run_official_gsd_on_qdte_workload.py
configs/variants/qdte_gsd_converged_seed0_manifest.yaml
configs/variants/qdte_gsd_breadth_seed0_manifest.yaml
```

FissionRefit uses `scripts/run_qdte_fission_refit_external.py` and the
Standard-v2, search-aware, and fission-refit-v2 overlays. These runners use
released measurements for generation and selection; exact true answers are
opened only by the completed-run evaluator.

Encoded column cardinalities are treated as public known schema information.
When no separate schema file is supplied, the loader infers them from the input
CSV as an input convenience, not as a private schema-estimation contribution.

## 10. Original-Protocol Reproduced Baselines

The strict same-protocol table and the upstream original-protocol reproduced
tables are separate. RAP++ official and PrivMRF official require their upstream
repositories and native data interfaces under the external baseline workspace;
the repositories are not vendored into this source tree.

Representative helper scripts are:

```text
scripts/run_rappp_official_paper_grid.py
scripts/collect_rappp_paper_grid.py
scripts/run_privmrf_official.py
scripts/collect_privmrf_official.py
scripts/audit_baseline_admission.py
scripts/audit_original_protocol_baselines.py
```

The generated CSV/Markdown outputs should be stored under the external results
workspace and packaged as paper artifacts, not committed to the source
repository.

## 11. Expected Output Files

A QDTE run should write:

```text
config_resolved.yaml
schema.json
queries.json
measurements.json
synthetic_encoded.npy
synthetic_decoded.csv
metrics_final.json
metrics_by_family.json
metrics_timeseries.csv
runtime.json
workload_summary.json
run_metadata.json
```

An external evaluation adds:

```text
evaluation.json
```

## 12. Public Release Notes

Large generated outputs, local baseline repositories, and paper scratch notes are not part of the source release. See:

```text
docs/PUBLIC_RELEASE_MANIFEST_20260706.md
```

Before publishing a clean source branch, simulate it without modifying the
research branch:

```bash
python3 scripts/audit_public_release_plan.py
conda run -n qdte python scripts/simulate_public_release.py --output-dir /tmp/private_de_public_release_strict_check --force
conda run -n qdte python scripts/rehearse_public_release_branch.py --output-dir /tmp/private_de_public_release_branch_rehearsal --force
PUBLIC_RELEASE_DIR="${PWD}/../private-de-public-release"
conda run -n qdte python scripts/create_public_release_repo.py --output-dir "${PUBLIC_RELEASE_DIR}" --force --commit
```

The release-plan audit checks that the manifest's explicit `git add` block
matches the generated public release path list and verification command list.
The simulation copies only the public release surface, initializes a temporary
Git repository, runs the strict public-release verifier, imports the QDTE
package, and executes the DP-mode QDTE smoke above. The branch rehearsal copies
the public surface plus currently tracked internal files into a temporary Git
repository, executes the release-branch index cleanup, and runs the strict
verifier on the resulting index. The persistent release-repo command creates a
clean `public-release-sage` Git branch in a separate repository containing only
the public source surface. The current paper package archive is:

```text
qdte_paper_package_20260711.tar.gz
sha256: 7bcacf49270d3f478deca9c8375b3f30b51502ca9d8c54caa812f93ce2972f7b
```
