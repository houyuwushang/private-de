# Handoff

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
