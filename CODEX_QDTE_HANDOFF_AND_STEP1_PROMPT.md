# Codex handoff workflow and Step 1 audit patch prompt for QDTE

This document is intended to be pasted into a fresh Codex session, or saved in the repository as a task file such as `codex_tasks/TASK_001_audit_metrics.md`.

---

## Part A. Persistent memory strategy for future Codex windows

Do not rely on a single Codex chat window to remember the project. Maintain project memory in repository files.

Recommended files:

```text
AGENTS.md
architecture_zh.md
README.md                         # optional but recommended
codex_tasks/
  TASK_001_audit_metrics.md
  TASK_002_workload_scaling.md
  TASK_003_public_schema.md
  TASK_004_heldout_workload.md
docs/
  QDTE_PROJECT_CONTEXT.md
  QDTE_IMPLEMENTATION_STATUS.md
  QDTE_FORMULAS_AND_INVARIANTS.md
  QDTE_EXPERIMENT_PROTOCOL.md
  QDTE_CLAIM_BOUNDARY.md
  QDTE_CHANGELOG.md
  QDTE_HANDOFF_LATEST.md
```

### What each file should contain

`AGENTS.md`: persistent repo rules for Codex. Keep it concise. It should say how to run tests, what files to read before work, what invariants must never be broken, and how to update handoff notes.

`docs/QDTE_PROJECT_CONTEXT.md`: stable research context. Define QDTE-Greedy-Dense, the algorithm goal, privacy boundary, and current claim boundary.

`docs/QDTE_IMPLEMENTATION_STATUS.md`: what is implemented, what is partial, what is not implemented. Update after every task.

`docs/QDTE_FORMULAS_AND_INVARIANTS.md`: formulas that tests must protect, including residual sign, measured loss, edit advantage, batch advantage, and DP post-processing boundary.

`docs/QDTE_EXPERIMENT_PROTOCOL.md`: commands, configs, datasets, epsilon/rho conversion table, output files, and audit metrics.

`docs/QDTE_CHANGELOG.md`: short append-only log of changes and tests.

`docs/QDTE_HANDOFF_LATEST.md`: one-page summary for the next fresh Codex window.

`codex_tasks/TASK_xxx.md`: one task per file. Each task should specify goal, non-goals, files likely to change, required tests, success criteria, and expected final report.

---

## Part B. Suggested root `AGENTS.md`

Copy this into the repository root as `AGENTS.md`.

```markdown
# AGENTS.md — QDTE repository instructions

## Project identity

This repository implements QDTE-Greedy-Dense: a differentially private synthetic data generator based on edit-impact directed evolution.

The core pipeline is:

1. preprocess input data using a public or explicitly configured schema/domain;
2. build a measured workload;
3. compute true query answers on the real data;
4. add zCDP Gaussian noise in `privacy.mode=dp`;
5. project/clip noisy targets;
6. initialize a synthetic table;
7. run QDTE edit loop using only noisy/projected targets;
8. evaluate true query error after the run only for offline evaluation.

## Must-preserve mathematical invariants

Residual convention:

```text
residual[q] = target_projected[q] - answer_syn[q]
```

If `residual[q] > 0`, the synthetic data has too few records satisfying query `q`, so a directed edit should try to enter the query.
If `residual[q] < 0`, the synthetic data has too many records satisfying query `q`, so a directed edit should try to exit the query.

Measured loss:

```text
L = 0.5 * sum_q residual[q]^2 * inv_variance[q]
```

Edit impact:

```text
delta[q] = phi_q(x_new) - phi_q(x_old)
```

Edit advantage:

```text
A(e) = delta @ (residual * inv_variance)
       - 0.5 * ((delta * delta) @ inv_variance)
       - lambda_cost * edit_cost
```

Without edit cost, accepting `A(e) > 0` must decrease measured loss for the corresponding single edit. For a batch, batch advantage must be computed from the cumulative `delta_sum`.

## Privacy boundary

In `privacy.mode=dp`, QDTE optimization must never use exact true answers after measurement. Exact true answers may be computed only for final or logged evaluation metrics, never for candidate generation, scoring, scheduling, transport, early stopping, or model selection.

Generation/evolution/projection/materialization are post-processing of DP measurements and should not consume additional privacy budget.

## Current implementation status

Implemented:

- static-all DP measurement with zCDP Gaussian noise;
- diagonal inverse-variance weighting;
- dense JAX full-workload edit scoring;
- query-specific repairs for EQ/LE/GE/RANGE-style predicates;
- micro-batch greedy transport with positive batch advantage;
- final full recomputation drift check.

Partial:

- consistency projection: only simplex projection for partition groups and clipping for non-partition groups;
- GPU candidate diagnostics: top-k returned candidate stats are incomplete;
- public-schema handling: assume schema/domain/discretization are public unless explicitly marked as debug/private.

Not implemented yet:

- adaptive select-measure loop;
- full covariance/block covariance;
- halfspace workload construction;
- atom histogram transport;
- debt scheduler update;
- plausibility materializer;
- formal disclosure/privacy attack audit;
- held-out workload evaluation.

## Development discipline

For every task:

1. Read this file first.
2. Read `architecture_zh.md` and the task file under `codex_tasks/` if present.
3. Start with a short plan before editing code.
4. Make the smallest safe change that satisfies the task.
5. Do not change core optimization behavior unless the task explicitly asks for it.
6. Add or update tests for new metrics or behavior.
7. Run relevant tests and report exact commands and results.
8. Update `docs/QDTE_IMPLEMENTATION_STATUS.md`, `docs/QDTE_CHANGELOG.md`, and `docs/QDTE_HANDOFF_LATEST.md` if they exist.
9. In the final response, list changed files, tests run, outputs generated, and remaining risks.
```

---

## Part C. Step 1 task file: `codex_tasks/TASK_001_audit_metrics.md`

```markdown
# TASK_001 — Add audit metrics for QDTE without changing optimization behavior

## Goal

Add audit and interpretability metrics to the current QDTE implementation so that a fresh run can answer:

1. what the measured loss scale means;
2. whether true query error improved from initialization to final output;
3. which query families improved or failed;
4. how many candidates passed each stage of the candidate funnel;
5. what the gain/sec and candidate throughput are;
6. how much of the workload was actually constructed and whether caps were hit.

This task is audit-only. Do not change the core QDTE algorithm.

## Non-goals

Do not implement:

- public schema loader;
- held-out workload;
- new projection methods;
- halfspace queries;
- debt scheduler;
- plausibility materializer;
- adaptive select-measure;
- baseline comparison.

Do not change:

- DP measurement target;
- residual definition;
- candidate repair logic;
- edit scoring formula;
- transport acceptance logic;
- default hyperparameters, unless adding optional metrics flags.

## Required formulas

Measured loss:

```text
L = 0.5 * sum_q residual[q]^2 * inv_variance[q]
```

RMS standardized residual:

```text
rms_standardized_residual = sqrt(2 * L / m)
```

Family-level measured loss:

```text
L_family = 0.5 * sum_{q in family} residual[q]^2 * inv_variance[q]
rms_standardized_residual_family = sqrt(2 * L_family / m_family)
```

True query error on rates:

```text
true_rate[q] = true_answers[q] / n_real
syn_rate[q]  = syn_answers[q] / n_syn
MAE  = mean(abs(syn_rate - true_rate))
RMSE = sqrt(mean((syn_rate - true_rate)^2))
MaxErr = max(abs(syn_rate - true_rate))
```

Throughput:

```text
loss_reduction_per_second = (initial_measured_loss - final_measured_loss) / time_generation_seconds
candidates_scored_per_second = num_candidates_scored / time_generation_seconds
accepted_edits_per_second = num_accepted_edits / time_generation_seconds
accepted_per_scored_candidate = num_accepted_edits / num_candidates_scored
```

## Required output additions

Add fields to `metrics_final.json`:

```text
initial_measured_loss
final_measured_loss
loss_reduction
initial_rms_standardized_residual
final_rms_standardized_residual
loss_reduction_per_second
candidates_scored_per_second
accepted_edits_per_second
accepted_per_scored_candidate
initial_true_query_mae
initial_true_query_rmse
initial_true_query_max_error
final_true_query_mae
final_true_query_rmse
final_true_query_max_error
true_query_mae_delta
true_query_rmse_delta
```

Keep the existing `true_query_mae`, `true_query_rmse`, and `true_query_max_error` aliases for backward compatibility, but make them equal to the final metrics.

Create `metrics_by_family.json` with, for each query family:

```text
num_queries
initial_measured_loss
final_measured_loss
initial_rms_standardized_residual
final_rms_standardized_residual
initial_true_query_mae
initial_true_query_rmse
initial_true_query_max_error
final_true_query_mae
final_true_query_rmse
final_true_query_max_error
true_query_mae_delta
true_query_rmse_delta
```

Create or extend `workload_summary.json`:

```text
num_queries_total
num_groups_total
num_queries_by_family
num_groups_by_family
num_queries_by_group
max_queries_config
max_queries_hit
max_2way_cells_config
range_intervals_per_num_attr
mixed_queries_per_pair
include_oneway
include_2way_cat
include_prefix
include_range
include_mixed
include_halfspace
```

Add candidate funnel fields to `runtime.json` or `metrics_final.json`:

```text
num_candidates_requested
num_candidates_scored
num_candidates_returned_if_topk
num_positive_advantage_returned
num_selected_nonconflicting
num_batch_accepted
accepted_per_requested_candidate
accepted_per_scored_candidate
accepted_per_returned_candidate
positive_returned_rate
```

For GPU top-k mode, if positivity over all candidates before top-k is not available yet, report:

```text
num_positive_advantage_all_candidates: null
positive_advantage_all_rate: null
positive_advantage_all_rate_available: false
positive_returned_rate_is_topk_biased: true
```

Optional but preferred: modify the GPU candidate scoring kernel to return `num_positive_all_candidates` before top-k. If this adds too much complexity, do not risk breaking the GPU path; mark it unavailable instead.

Add fields to `metrics_timeseries.csv`:

```text
rms_standardized_residual
loss_reduction_from_initial
loss_reduction_per_second_so_far
accepted_total_so_far
scored_total_so_far
accepted_per_scored_so_far
```

## Suggested files to inspect

- `qdte/evolution/engine.py`
- `qdte/eval/metrics.py`
- `qdte/eval/runtime.py`
- `qdte/queries/workload.py`
- `qdte/evolution/gpu_candidates.py`
- `tests/test_engine_smoke.py`
- `tests/test_edit_advantage.py`
- `tests/test_measurement.py`

## Implementation plan

1. Add helper functions in `qdte/eval/metrics.py`:
   - `rms_standardized_residual_from_loss(loss, num_queries)`;
   - `query_error_metrics_prefixed(prefix, true_answers, syn_answers, n_real, n_syn)`;
   - `query_error_metrics_by_family(...)`;
   - `measured_loss_by_family(...)`.

2. Add workload summary support:
   - either return diagnostics from `build_workload`, or create a helper that summarizes `qcat` and `workload_groups` after construction;
   - write `workload_summary.json` in `engine.py`.

3. In `run_qdte`, preserve initial synthetic answers before the edit loop:
   - `initial_answer_syn = answer_syn.copy()`;
   - `initial_residual = residual.copy()`;
   - do not alter optimization logic.

4. At final evaluation, compute true answers once for evaluation only, then compute both initial and final true query metrics.

5. Add family-level metrics using `qcat.families`.

6. Add throughput metrics using `RuntimeStats` and existing counters.

7. Improve candidate funnel counters:
   - accumulate returned candidate count separately from scored count;
   - accumulate selected nonconflicting count;
   - accumulate batch accepted count;
   - keep current counters for backward compatibility.

8. Update tests:
   - smoke test should assert that `metrics_by_family.json` and `workload_summary.json` exist;
   - smoke test should assert `initial_rms_standardized_residual` and `final_rms_standardized_residual` are present;
   - add a small unit test for `rms_standardized_residual_from_loss`;
   - ensure `measurements.json` still does not include true answers.

## Required checks

Run at least:

```bash
pytest -q
```

If GPU/JAX environment is available, also run a short smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py \
  --config configs/smoke.yaml \
  --privacy.mode dp \
  --qdte.max_iters 3 \
  --runtime.xla_preallocate false \
  --run.output_dir outputs/smoke_audit_check
```

Do not require a full Adult run for this patch.

## Success criteria

The patch is successful if:

1. all tests pass;
2. default QDTE runs still produce `synthetic_decoded.csv`, `metrics_final.json`, `metrics_timeseries.csv`, and `runtime.json`;
3. new outputs `metrics_by_family.json` and `workload_summary.json` are produced;
4. optimizer behavior is unchanged except for additional logging/metrics;
5. no exact true answers are written into `measurements.json`;
6. final report clearly states changed files, tests run, and any metrics not yet available in GPU top-k mode.

## Final response format

At the end, report:

```text
Changed files:
- ...

Tests run:
- ...

New outputs:
- ...

Important notes:
- Whether GPU all-candidate positive rate is implemented or intentionally marked unavailable.
- Whether any optimization behavior changed. It should be “no”.

Next recommended task:
- TASK_002_workload_scaling or TASK_003_public_schema.
```
```

---

## Part D. Prompt to paste into a fresh Codex window

```text
You are working on the GitHub repository `houyuwushang/private-de`.

This is a QDTE implementation for differentially private synthetic data generation. A previous implementation already runs and decreases measured loss. Your task is NOT to redesign the algorithm. Your task is Step 1: add audit metrics and interpretability outputs so that we can trust and analyze the existing implementation.

First, read these files if they exist:

1. `AGENTS.md`
2. `architecture_zh.md`
3. `docs/QDTE_PROJECT_CONTEXT.md`
4. `docs/QDTE_FORMULAS_AND_INVARIANTS.md`
5. `codex_tasks/TASK_001_audit_metrics.md`

If some docs do not exist yet, continue using this prompt as the task specification.

Core invariants you must preserve:

- `residual[q] = target_projected[q] - answer_syn[q]`.
- Measured loss is `0.5 * sum_q residual[q]^2 * inv_variance[q]`.
- Candidate edit impact is `delta[q] = phi_q(x_new) - phi_q(x_old)`.
- Edit advantage is `delta @ (residual * inv_variance) - 0.5 * ((delta * delta) @ inv_variance) - lambda_cost * edit_cost`.
- In `privacy.mode=dp`, exact true answers may be computed only for offline evaluation after initialization/final output. They must not be used for scheduling, candidate generation, scoring, transport, early stopping, or model selection.
- Do not change core optimization behavior in this task.

Task goal:

Add audit metrics to explain loss scale, true query improvement, family-level behavior, candidate funnel, throughput, and workload coverage.

Required metrics:

1. `rms_standardized_residual = sqrt(2 * measured_loss / num_queries)`.
2. Initial and final true query MAE/RMSE/MaxErr on rate scale.
3. Delta between initial and final true query MAE/RMSE.
4. Family-level measured loss and true query error.
5. Throughput:
   - `loss_reduction_per_second`
   - `candidates_scored_per_second`
   - `accepted_edits_per_second`
   - `accepted_per_scored_candidate`
6. Candidate funnel:
   - requested candidates
   - scored candidates
   - returned candidates if GPU top-k is used
   - positive returned candidates
   - selected non-conflicting candidates
   - batch accepted candidates
7. Workload summary:
   - total query count
   - query count by family
   - group count by family
   - max query cap and whether it was hit
   - relevant workload config values.

Required output files:

- Keep existing: `metrics_final.json`, `metrics_timeseries.csv`, `runtime.json`.
- Add: `metrics_by_family.json`.
- Add: `workload_summary.json`.

Backward compatibility:

- Keep existing final fields `true_query_mae`, `true_query_rmse`, and `true_query_max_error`; they should remain aliases for the final true query metrics.
- Do not remove existing fields.

Suggested implementation files:

- `qdte/eval/metrics.py`
- `qdte/eval/runtime.py`
- `qdte/evolution/engine.py`
- `qdte/queries/workload.py`
- `qdte/evolution/gpu_candidates.py`
- tests under `tests/`

Tests:

- Add or update smoke tests so they assert the new output files and key metric fields exist.
- Add a unit test for RMS standardized residual.
- Ensure `measurements.json` still does not contain exact true answers.
- Run `pytest -q`.

Important caution about GPU top-k mode:

If positivity over all candidates before top-k is easy to compute safely, implement it. If not, do not risk breaking the fused GPU path. Instead, explicitly report:

- `positive_advantage_all_rate_available: false`
- `positive_advantage_all_rate: null`
- `positive_returned_rate_is_topk_biased: true`

Final response must include:

1. a short implementation plan before coding;
2. changed files;
3. tests run and results;
4. new output fields/files;
5. explicit statement that optimization behavior was not changed;
6. remaining risks or metrics still unavailable.
```
