# Handoff

## README And Repository Sync

### What Changed

- Added a Chinese root `README.md` describing the current code functionality, privacy boundary, module layout, run commands, outputs, held-out evaluation path, test status, and known limitations.

### Changed Files

- `README.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `15 passed in 1.96s`

### Current Status

- README added.
- Repository ready to commit and push.

### Next Recommended Task

- Run a representative DP experiment and compare measured workload metrics with held-out workload metrics.

## TASK 002 Held-Out Workload Evaluation

### What Changed

- Added optional `evaluation.compute_heldout_query_error` support with `evaluation.heldout_workload` config defaults.
- Built a separate held-out `QueryCatalogue` from `evaluation.heldout_workload` and saved it to `queries_holdout.json`.
- Added exact duplicate filtering against measured workload queries when `evaluation.heldout_exclude_measured_queries` is true.
- Added held-out workload metadata in `workload_summary_holdout.json`.
- Added offline-only held-out true-query evaluation after initialization and after final synthetic generation.
- Added held-out aggregate metrics in `metrics_holdout.json`.
- Added per-family held-out true-query metrics in `metrics_by_family_holdout.json`.
- Added held-out summary fields to `metrics_final.json`.
- Added query catalogue helper coverage for stable query keys and duplicate filtering.
- Extended smoke coverage for held-out output files, duplicate exclusion, and the privacy boundary.

### New Output Files

- `queries_holdout.json`
- `workload_summary_holdout.json`
- `metrics_holdout.json`
- `metrics_by_family_holdout.json`

### Changed Files

- `qdte/evolution/engine.py`
- `qdte/queries/types.py`
- `qdte/queries/workload.py`
- `tests/test_engine_smoke.py`
- `tests/test_queries.py`
- `docs/HANDOFF.md`

### Tests Run

- `pytest -q`
  - Result: failed in this shell because `pytest` is not on `PATH`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `15 passed in 19.63s`

### Current Status

- Implementation complete.
- Test verification complete in the `qdte` conda environment.
- Held-out workload is evaluation-only: it is not passed to measurement, residual computation, active query selection, candidate generation, scoring, transport, stopping, or hyperparameter selection.

### Next Recommended Task

- Run a representative DP experiment with `evaluation.compute_heldout_query_error: true` and compare measured-workload vs held-out true-query error trends for DP-noise overfitting.

## TASK 001b Audit Output Cleanup

### What Changed

- Moved one-off Codex prompt documents from the repo root into `docs/codex_tasks/`.
- Added audit-friendly aliases to `workload_summary.json`: `total_queries`, `queries_by_family`, and `groups_by_family`.
- Added a nested `candidate_funnel` object to `runtime.json` while preserving existing flat runtime fields.
- Added top-k interpretation flags to `metrics_final.json`.
- Added per-family true-query MAE/RMSE reduction fields when true-query evaluation is enabled.
- Added unit coverage for `rms_standardized_residual(loss, num_queries)`.
- Extended smoke coverage for the new audit-output fields.

### Changed Files

- `docs/codex_tasks/CODEX_QDTE_HANDOFF_AND_STEP1_PROMPT.md`
- `docs/codex_tasks/QDTE_STEP1_AUDIT_PATCH_FOR_CODEX.md`
- `qdte/eval/runtime.py`
- `qdte/evolution/engine.py`
- `tests/test_engine_smoke.py`
- `tests/test_metrics.py`
- `docs/HANDOFF.md`

### Tests Run

- `pytest -q`
  - Result: `14 passed in 1.72s`

### Current Status

- Implementation complete.
- Test verification complete.

### Next Recommended Task

- Run a representative DP configuration and inspect `workload_summary.json`, `runtime.json`, `metrics_final.json`, and `metrics_by_family.json` for audit readability.

## TASK 001 Audit Metrics And Workload Summaries

### What Changed

- Added loss-scale RMS metrics to `metrics_final.json`.
- Added runtime efficiency metrics and candidate funnel aggregates to `metrics_final.json` and `runtime.json`.
- Added initial true-query evaluation metrics while keeping the existing final true-query aliases.
- Added per-family measured and true-query evaluation output in `metrics_by_family.json`.
- Added workload coverage output in `workload_summary.json`.
- Added new `metrics_timeseries.csv` columns for RMS residuals and candidate funnel counts/rates.
- Added smoke coverage for the new audit files and privacy-boundary checks.

### Changed Files

- `qdte/eval/metrics.py`
- `qdte/eval/runtime.py`
- `qdte/evolution/engine.py`
- `tests/test_engine_smoke.py`
- `AGENTS.md`
- `docs/HANDOFF.md`

### Tests Run

- `conda run -n qdte pytest -q`
  - Result: `13 passed in 1.70s`

### Current Status

- Implementation complete.
- Test verification complete.

### Next Recommended Task

- Run a representative DP configuration and inspect the new audit JSON files for workload-family imbalance and candidate funnel bottlenecks.
