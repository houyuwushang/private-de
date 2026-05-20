# QDTE Step 1: Audit Metrics and Trustworthiness Patch

This patch must not change the QDTE optimization algorithm. It only adds diagnostics, consistency checks, and output files so that we can trust and interpret the current implementation.

## Why this patch comes first

The current implementation shows decreasing measured loss, but the meaning of that loss is not obvious because it is a covariance/variance-weighted count-scale loss, while `true_query_mae` and `true_query_rmse` are normalized rate errors. Before changing the algorithm, we need to make the current run fully auditable.

The current measured loss is:

```text
L = 0.5 * sum_j residual[j]^2 * inv_variance[j]
  = 0.5 * sum_j (residual[j] / sigma[j])^2
```

Therefore a loss of a few thousand can be normal when there are thousands of queries. The diagnostic quantity that should be reported is:

```text
rms_standardized_residual = sqrt(2 * L / num_queries)
```

For example, with 8000 queries and final loss 1500:

```text
sqrt(2 * 1500 / 8000) ≈ 0.612
```

This means the synthetic residual is around 0.61 noise standard deviations per query on average, which is not large.

## Files to modify

Likely files:

```text
qdte/eval/metrics.py
qdte/eval/runtime.py
qdte/evolution/engine.py
qdte/queries/workload.py
qdte/evolution/gpu_candidates.py    # optional in this step, only if easy
configs/*.yaml                     # optional audit config additions
```

## 1. Add measured-loss scale diagnostics

In `qdte/eval/metrics.py`, add:

```python
def standardized_residual_metrics(residual, sigma, inv_variance=None):
    """Return diagnostics explaining the weighted measured loss scale.

    residual is count-scale target - synthetic answer.
    sigma is count-scale DP noise std per query.
    """
```

It should report:

```text
num_queries
measured_residual_mae_count
measured_residual_rmse_count
measured_residual_max_count
measured_residual_mae_rate     # divide by N_syn or N when N_syn == N_real
measured_residual_rmse_rate
standardized_residual_mae
standardized_residual_rmse     # should equal sqrt(2 * loss / m)
standardized_residual_p50_abs
standardized_residual_p90_abs
standardized_residual_p95_abs
standardized_residual_p99_abs
standardized_residual_max_abs
fraction_abs_standardized_residual_le_1
fraction_abs_standardized_residual_le_2
fraction_abs_standardized_residual_le_3
```

Implementation details:

```python
z = residual / np.maximum(sigma, 1e-6)
loss = measured_loss(residual, inv_variance)
assert np.isclose(np.sqrt(2 * loss / m), np.sqrt(np.mean(z*z)), rtol=1e-4, atol=1e-4)
```

Do not fail the whole run if the check fails because of floating point precision; instead record a warning in `metrics_final.json`.

## 2. Add initial true query error

Currently the engine reports final true query error only. Add initial true query error before evolution starts.

In `qdte/evolution/engine.py`:

After initial synthetic answers are computed and before `X_real` is cleared, compute:

```python
true_answers_initial_eval = answer_queries(X_real, qcat, batch_size=...)
initial_true_metrics = query_error_metrics(true_answers_initial_eval, answer_syn, n_real, n_syn)
```

Then at the end, final true query metrics already exist. Add:

```text
initial_true_query_mae
initial_true_query_rmse
initial_true_query_max_error
final_true_query_mae
final_true_query_rmse
final_true_query_max_error
true_query_mae_reduction
true_query_rmse_reduction
```

Keep the old keys `true_query_mae`, `true_query_rmse`, `true_query_max_error` for backward compatibility, but also write the explicit `final_*` keys.

## 3. Add gain/sec and throughput metrics

In final metrics and runtime JSON, add:

```text
loss_reduction_per_generation_second = loss_reduction / time_generation_seconds
loss_reduction_per_total_second = loss_reduction / total_wall_time_seconds
accepted_edits_per_generation_second = num_accepted_edits / time_generation_seconds
scored_candidates_per_generation_second = num_candidates_scored / time_generation_seconds
accepted_edits_per_scored_candidate = num_accepted_edits / num_candidates_scored
accepted_edits_per_requested_candidate = num_accepted_edits / num_candidates_requested
```

Also include:

```text
initial_rms_standardized_residual
final_rms_standardized_residual
initial_measured_residual_rmse_rate
final_measured_residual_rmse_rate
```

## 4. Add workload coverage summary

Create `workload_summary.json` in the output directory.

It should contain:

```text
num_queries_total
num_queries_by_family
num_groups_by_family
max_queries_config
max_queries_reached: bool
max_2way_cells_config
range_intervals_per_num_attr
mixed_queries_per_pair
random_seed
```

Also extend `qdte/queries/workload.py` to return or save diagnostics about skipped/generated workload items. At minimum record:

```text
2way_pairs_considered
2way_pairs_generated
2way_cells_generated
2way_pairs_skipped_by_max_2way_cells
range_queries_generated
mixed_queries_generated
```

Reason: if the run has only about 8000 queries, we need to know whether this is because the workload is genuinely small or because caps such as `max_queries` and `max_2way_cells` truncated it.

## 5. Add per-family metrics

Add `metrics_by_family.json`.

For each family, report:

```text
family
num_queries
initial_measured_loss
final_measured_loss
loss_reduction
initial_rms_standardized_residual
final_rms_standardized_residual
initial_true_query_mae
final_true_query_mae
initial_true_query_rmse
final_true_query_rmse
```

Families are available through `qcat.families` or workload groups.

This is important because QDTE may work very well for one-way and prefix queries but not for mixed/range queries. A single global number hides this.

## 6. Add candidate funnel diagnostics

Add a candidate funnel section to `runtime.json` and `metrics_final.json`:

```text
candidate_funnel:
  requested_candidates
  scored_candidates
  returned_candidates              # if GPU top-k path is used
  positive_advantage_candidates_returned
  selected_nonconflicting_candidates
  accepted_edits
  accepted_per_scored
  accepted_per_returned
  positive_returned_rate
```

Important naming rule:

If GPU `gpu_return_top_k` is used, the existing `positive_advantage_rate` is computed over returned top-k candidates, not over all scored candidates. Rename or duplicate it as:

```text
positive_advantage_rate_returned
```

Do not report it as positive rate over all generated/scored candidates unless the GPU kernel actually counts positives before top-k.

Optional stronger GPU fix:

In `qdte/evolution/gpu_candidates.py`, before `jax.lax.top_k`, compute:

```python
num_positive_all = jnp.sum(scores > min_advantage)
```

Return it from the pmap kernel and aggregate across devices. Then report:

```text
positive_advantage_candidates_all
positive_advantage_rate_all
```

If this is too much for this first patch, skip it but label current positive rate correctly as returned-only.

## 7. Add target-direction diagnostics for accepted edits

For accepted edits, report whether the candidate actually improved its target query direction:

```text
r[target_q] * delta[target_q] > 0
```

Add to runtime:

```text
target_direction_success_accepted
accepted_edits_with_target_query
accepted_target_direction_success_rate
```

This requires computing deltas for accepted edits. Deltas are already computed during transport for CPU path; for GPU prefix path, either reuse selected deltas if available or recompute only for accepted edits.

## 8. Guard count-scale residual assumption

The current implementation uses count residual:

```text
residual = target_projected - answer_syn
```

This is valid only when `N_syn == N_real`, unless targets are rescaled to the synthetic dataset size.

Add either:

Option A, strict first patch:

```python
if n_syn != n_real:
    raise ValueError("Count-scale QDTE currently requires N_syn == N_real. Use same_as_real or implement target scaling.")
```

Option B, future patch:

```python
target_for_syn = target_projected * (n_syn / n_real)
```

Use Option A now unless you are ready to test Option B carefully.

## 9. Public schema assumption logging

Do not implement full public schema loading in this patch unless easy. But add explicit logging and metrics fields:

```text
schema_assumption: public_domain_required_for_formal_dp
schema_source: learned_from_input_csv | public_schema_file | unknown
formal_dp_schema_warning: true/false
```

For now, if no `preprocess.schema_path` is provided, set:

```text
schema_source = learned_from_input_csv
formal_dp_schema_warning = privacy.mode == "dp"
```

This does not change results, but prevents us from accidentally claiming formal DP when the schema/domain/bins were learned from private data.

Later patch will implement `public_schema.json`.

## 10. Tests to add or update

Add tests:

1. `test_standardized_loss_identity`:
   - create random residual and sigma;
   - verify `standardized_residual_rmse == sqrt(2 * measured_loss / m)`.

2. `test_initial_and_final_true_metrics_present`:
   - run smoke;
   - verify output metrics include initial and final true query errors.

3. `test_count_scale_requires_same_n`:
   - configure `N_syn` different from real N;
   - expect `ValueError` unless target scaling is enabled.

4. `test_workload_summary_written`:
   - run smoke;
   - verify `workload_summary.json` exists and contains family counts.

## 11. Expected output files after this patch

A normal run should write:

```text
metrics_final.json
metrics_timeseries.csv
runtime.json
workload_summary.json
metrics_by_family.json
synthetic_encoded.npy
synthetic_decoded.csv
measurements.json
schema.json
queries.json
logs.txt
```

## 12. Success criteria

After this patch, a full run must allow us to answer:

1. Why is measured loss numerically large or small?
2. What is the average residual in noise-standard-deviation units?
3. Did true query error improve from initialization to final output?
4. Which query families improved and which did not?
5. How many candidates were requested, scored, returned, selected, and accepted?
6. What is loss reduction per second?
7. Was the workload truncated by `max_queries` or `max_2way_cells`?
8. Is the current DP claim conditional on public schema/domain?

Do not change the QDTE optimization behavior in this patch.
