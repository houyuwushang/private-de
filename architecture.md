# QDTE Implementation Architecture

This document describes the code structure implemented for the QDTE synthetic data generator in this repository. It is written to make the execution path, privacy boundary, and GPU optimization code easy to audit.

## Entry Points

### `scripts/run_qdte.py`

Main command-line entry point.

Responsibilities:

- Load a YAML config with `qdte.config.load_yaml`.
- Apply CLI overrides such as `--privacy.mode dp` using dotted keys.
- Optionally disable XLA preallocation via `runtime.xla_preallocate`.
- Call `qdte.evolution.engine.run_qdte(config)`.

The primary DP command is:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte.yaml --privacy.mode dp
```

### `scripts/smoke_qdte.py`

Small local smoke runner. It creates a synthetic toy CSV under `outputs/`, builds an in-memory config, and calls `run_qdte`.

### `scripts/run_ablation.py`

Ablation runner for comparing variants:

- `random_mutation`
- `no_edit_cost`
- `sequential_greedy`
- `target_only`
- `no_threshold`

### `scripts/check_env.py`

JAX/GPU sanity check. It prints devices and runs a matrix multiplication.

## Config Files

### `configs/adult_qdte.yaml`

Default reliable Adult DP configuration.

Important settings:

- Input CSV: `/home/qianqiu/rerun-experiment/dataset/adult.csv`
- Output directory: `outputs/adult_qdte`
- `privacy.mode: dp`
- `privacy.rho_total: 1.0`
- `privacy.delta: 1.0e-9`
- `privacy.measurement_mode: static_all`
- `qdte.candidate_backend`: omitted, so the engine uses `cpu_repair`
- `qdte.objective_weighting: unweighted`
- `qdte.score_backend: dense_gpu`
- `qdte.transport_mode: atom_flow`
- `qdte.atom_flow_update_mode: batch`
- `qdte.atom_flow_pool_multiplier: 16`
- `qdte.atom_flow_max_pool: 0`
- `qdte.max_iters: 5000`
- `qdte.total_candidates_per_iter: 4096`
- `qdte.accepted_per_iter: 64`

This is the current quality-oriented baseline. `objective_weighting:
unweighted` is the current single-layer QDTE mainline: the generation stage
directly minimizes raw residuals between the projected feasible target and the
synthetic query answers. The `variance` mode remains available as an
inverse-variance ablation.

### `configs/adult_qdte_gpu_highpower.yaml`

Experimental high-throughput GPU configuration.

Important settings:

- Output directory: `outputs/adult_qdte_gpu_highpower`
- `qdte.objective_weighting: unweighted`
- `qdte.candidate_backend: jax_repair`
- `qdte.score_backend: sparse_delta_gpu`
- `qdte.transport_mode: atom_flow`
- `qdte.atom_flow_update_mode: batch`
- `qdte.transport_delta_backend: jax_prefix`
- `qdte.transport_prefix_strategy: best_advantage`
- `qdte.max_iters: 500`
- `qdte.total_candidates_per_iter: 1572864`
- `qdte.accepted_per_iter: 1024`
- `qdte.gpu_source_draws: 32`
- `qdte.gpu_batches_per_iter: 1`
- `qdte.gpu_sparse_query_block_size: 64`
- `qdte.gpu_sparse_changed_attr_capacity: 4`
- `qdte.gpu_return_top_k: 8192`

This path scores many more candidates on GPU, uses sparse-delta GPU scoring by default, and returns only a top-k subset to CPU-side atom-flow transport.

### Other configs

- `configs/smoke.yaml`: small smoke-test configuration.
- `configs/acs_qdte.yaml`: ACS dataset configuration.

## Top-Level Package Layout

```text
qdte/
  config.py                 YAML loading, dotted CLI overrides
  config_validation.py      fail-fast validation for unsupported modes/backends
  dataio.py                 output directory, JSON, NPY helpers
  preprocess.py             CSV encoding/decoding
  schema.py                 encoded table schema dataclasses
  privacy/
    accountant.py           zCDP epsilon conversion
    gaussian.py             zCDP Gaussian mechanism
    exponential.py          exponential mechanism helper, not used in default QDTE path
  queries/
    types.py                query catalogue representation
    workload.py             workload/group construction
    eval_jax.py             JAX query evaluation
    delta_index.py          sparse affected-query index for edit deltas
  measurement/
    measure.py              real-data measurement, DP noise, projection
    projection.py           simplex projection and count clipping
    consistency.py          scope-local marginal consistency projection
  evolution/
    engine.py               end-to-end QDTE orchestration
    state.py                mutable QDTE state dataclass
    initialization.py       independent one-way synthetic initialization
    scheduler.py            active query selection
    candidates.py           CPU repair candidate generation
    gpu_candidates.py       JAX/GPU fused candidate generation and scoring
    scoring.py              dense/sparse candidate scoring and delta computation
    transport.py            nonconflicting, prefix, and atom-flow transport
  eval/
    metrics.py              measured loss and true-query evaluation metrics
    runtime.py              runtime counters and throughput stats
```

## End-to-End Execution Flow

The full run is orchestrated by `qdte.evolution.engine.run_qdte`.

High-level sequence:

1. Resolve config and output directory.
2. Load and encode the real CSV.
3. Build the query workload.
4. Measure real data:
   - compute true query answers on `X_real`;
   - if `privacy.mode=dp`, add zCDP Gaussian noise;
   - project/clamp noisy counts where configured;
   - optionally apply prefix monotonicity and scope-local marginal consistency projection.
5. Initialize synthetic data from one-way noisy targets.
6. Run QDTE edit loop:
   - choose active target queries;
   - generate candidate edits;
   - score candidate edits against the measured target;
   - select transport edits with atom-flow or prefix-greedy transport;
   - apply accepted edits;
   - update synthetic query answers incrementally;
   - periodically recompute all synthetic answers to check drift.
7. Optionally compute exact true-query and held-out query metrics for offline evaluation only.
8. Save encoded and decoded synthetic data.
9. Save final metrics, per-family metrics, timeseries, runtime profile, schema, workload, measurements, logs, held-out outputs, and resolved config.

## Privacy Boundary

The privacy-critical code is in `qdte.measurement.measure.measure_real_dataset`.

### DP mode

When `privacy.mode=dp`:

1. `answer_queries(X_real, qcat, batch_size=...)` computes exact query answers on the real encoded data.
2. The workload groups are assigned zCDP budget using `_allocate_group_budgets`.
3. Each workload group calls `add_zcdp_gaussian_noise` from `qdte/privacy/gaussian.py`.
4. The noisy target counts become `Measurements.target_noisy`.
5. The projected/clipped noisy counts become `Measurements.target_projected`.
6. QDTE only optimizes against `target_projected`, with inverse variance weights derived from the Gaussian noise variance.

The Gaussian mechanism uses:

```text
sigma = 1 / sqrt(2 * rho)
noise_std = sensitivity_l2 * sigma
noisy_answers = true_answers + Normal(0, noise_std)
```

The reported epsilon is computed in `qdte/privacy/accountant.py` as:

```text
epsilon(delta) = rho + 2 * sqrt(rho * log(1 / delta))
```

### Oracle mode

When `privacy.mode=oracle`, `measure_real_dataset` uses exact query answers as targets and logs a warning. This mode is only for debugging or upper-bound experiments. It is not a DP result path.

### Evaluation after generation

`evaluation.compute_true_query_error` computes exact true query answers after the run only for reporting final MAE/RMSE. These exact answers are not used as QDTE optimization targets in DP mode.

## Data Encoding

Implemented in `qdte/preprocess.py` and `qdte/schema.py`.

### `load_and_preprocess_csv`

Reads the configured CSV and produces:

- `PreprocessResult.X`: encoded integer table, shape `(n_rows, n_columns)`.
- `PreprocessResult.schema`: `TableSchema` with column metadata.
- `PreprocessResult.raw_columns`: original CSV column names converted to strings.

Categorical columns are mapped to sorted integer category IDs. Numerical columns are discretized into quantile-like bins unless their unique value count is already small. Missing values, empty strings, and `?` are mapped to the configured missing token.

### `decode_array`

Converts encoded synthetic records back to a CSV-friendly `pandas.DataFrame` using schema representatives.

## Query Representation and Workload

Implemented in:

- `qdte/queries/types.py`
- `qdte/queries/workload.py`
- `qdte/queries/eval_jax.py`

### Query terms

Supported operations:

- `OP_EQ`
- `OP_LE`
- `OP_GE`
- `OP_RANGE`

Each query is stored in fixed-size arrays:

- `attrs`
- `ops`
- `values`
- `lows`
- `highs`
- `num_terms`
- `linear_attrs`
- `linear_weights`
- `linear_thresholds`
- `linear_num_terms`

The ordinary arrays represent `EQ/LE/GE/RANGE` conjunctions. The linear arrays represent halfspace predicates of the form `sum_i weight_i * x[attr_i] <= threshold`. This fixed array layout is important because it can be passed directly to JAX kernels.

### Workload groups

`build_workload` constructs a `QueryCatalogue` and a list of `WorkloadGroup` objects.

Implemented families:

- `oneway`
- `twoway`
- `prefix`
- `range`
- `mixed`
- `kway`
- `kway_prefix`
- `kway_range`
- `kway_mixed`
- `orthogonal_kway_mixed`
- `halfspace`

Each `WorkloadGroup` carries:

- `query_indices`
- `family`
- `sensitivity_l2`
- `is_partition`

Partition groups such as one-way and selected two-way marginals can be projected to the simplex with total count equal to the real dataset size.

The configurable high-order families are:

- `kway`: sampled k-way equality conjunctions.
- `kway_prefix`: sampled k-way conjunctions with one numeric prefix term and equality conditions on the remaining attributes.
- `kway_range`: sampled k-way conjunctions with one numeric range term and equality conditions on the remaining attributes.
- `kway_mixed`: sampled k-way conjunctions where categorical attributes use equality terms and numerical attributes use prefix/range terms, allowing multiple numerical terms in the same query.
- `orthogonal_kway_mixed`: sampled k-way mixed scopes expanded into full Cartesian partitions. Categorical dimensions use all equality cells; numerical dimensions use equal-width disjoint range intervals. Each group is mutually exclusive and has L2 sensitivity `1`.

They are controlled by `include_kway`, `include_kway_prefix`, `include_kway_range`, `include_kway_mixed`, `include_orthogonal_kway_mixed`, the corresponding `*_orders`, and `*_queries_per_order` or `*_scopes_per_order` settings. If enabled in DP mode, their family names must also appear in `privacy.measurement_allocation`.

DP measurement is applied per `WorkloadGroup`: the group answer vector is noised with a Gaussian mechanism calibrated to that group's L2 sensitivity and allocated rho. `build_workload` refines each group sensitivity from the public schema and query definitions by enumerating the finite query scope when it is below `workload.exact_group_sensitivity_max_cells`. Orthogonal equality workloads therefore get sensitivity `1`; overlapping prefix/range/mixed workloads get `sqrt(max_overlap)`. If the scope is too large, the builder falls back to the conservative configured group sensitivity. The implementation records diagonal variances; coordinates inside a group currently receive independent Gaussian noise with the same standard deviation.

Halfspace queries are implemented for measurement, evaluation, CPU repair, GPU fused single-query repair, and consistency projection.

Orthogonal halfspace groups are not implemented yet. The current catalogue can represent a single linear threshold `sum_i w_i x_i <= t`, which gives cumulative and overlapping halfspace queries. A truly orthogonal halfspace partition would require either a linear slab predicate `lo < sum_i w_i x_i <= hi` or conjunctions of halfspace tree leaves; both require extending the query representation and JAX/GPU evaluation paths.

### JAX query evaluation

`qdte/queries/eval_jax.py` provides:

- `eval_records_queries_arrays`: JIT-compiled per-record/per-query boolean matrix evaluation.
- `eval_records_queries`: wrapper using `QueryCatalogue`.
- `answer_queries`: batches over records and sums query satisfaction indicators.

The main count identity is:

```text
answer[q] = sum_i 1{record_i satisfies query_q}
```

## Measurement and Projection

Implemented in:

- `qdte/measurement/measure.py`
- `qdte/measurement/projection.py`
- `qdte/measurement/consistency.py`

`measure_real_dataset` returns a `Measurements` dataclass:

- `target_noisy`: noisy answers before projection.
- `target_projected`: projected/clipped targets used by QDTE.
- `variances`: per-query noise variances.
- `inv_variances`: inverse variances used in weighted loss.
- `groups`: per-group budget/noise metadata.
- `mode`, `rho_total`, `rho_spent`, `epsilon_delta`, `delta`.
- `projection_diagnostics`: projection metadata, including consistency diagnostics when enabled.

Projection helpers:

- `project_simplex`: projects partition marginals onto nonnegative counts summing to dataset size.
- `clip_counts`: clips non-partition counts into `[0, N]`.
- `project_non_decreasing`: weighted PAVA projection for prefix measurements.
- `project_consistent_targets`: maps supported query scopes to local marginal tables, enforces non-negativity and known row count `N`, and reconciles overlapping scopes by shared marginals.

The consistency projection supports `EQ`, `LE`, `GE`, `RANGE`, configurable k-way conjunctions, and halfspace masks represented by `QueryCatalogue`. It fails fast when a scope exceeds `projection.consistency.max_scope_cells`; it does not silently skip large scopes. The current implementation still reports diagonal variances after projection. Full post-projection covariance propagation is future work.

## QDTE State

Implemented in `qdte/evolution/state.py`.

`QDTEState` contains:

- `X_syn`: current encoded synthetic table.
- `answer_syn`: current synthetic query answers.
- `target`: DP noisy/projected target answers.
- `residual`: `target - answer_syn`.
- `variance`, `inv_variance`, `sigma`.
- `debt`: query-level scheduling signal updated from weighted loss damage/improvement after accepted edits.
- `iteration`.

The main loss is implemented in `qdte/eval/metrics.py`:

```text
measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]
```

## Initialization

Implemented in `qdte/evolution/initialization.py`.

`initialize_independent_oneway` initializes each synthetic column independently from the noisy/projected one-way marginals when available. If one-way targets are unavailable or invalid for a column, it falls back to uniform sampling over that column's cardinality.

## Active Query Scheduling

Implemented in `qdte/evolution/scheduler.py`.

`select_active_queries` ranks queries by standardized residual:

```text
priority = (abs(residual) - kappa_noise * sigma) / sigma
```

Only positive priorities are preferred. If none are positive, the default behavior is to return no active queries. When `allow_below_noise_fallback: true`, selection falls back to absolute standardized residual. The selected top `num_active_targets` queries become the targets for directed candidate repair.

By default, `allow_below_noise_fallback: false` makes this threshold strict: if no query exceeds `kappa_noise * sigma`, no candidates are generated and the patience counter advances. High-throughput GPU configs may set `allow_below_noise_fallback: true` to keep the GPU candidate path busy.

## Candidate Generation

Candidate edits are represented by `CandidateBatch` in `qdte/evolution/candidates.py`:

- `row_ids`: synthetic row IDs to edit.
- `old_rows`: rows before edit.
- `new_rows`: proposed rows after edit.
- `target_query_ids`: query that motivated each candidate, or `-1` for random candidates.
- `edit_cost`: Hamming/numerical edit penalty.
- `repair_type`: random, enter, or exit repair type.
- `diagnostics`: counts for requested/produced candidates and filtering diagnostics.

### CPU repair backend: `qdte/evolution/candidates.py`

This is the default backend.

For each active query:

1. Decide whether the residual wants more records satisfying the query (`enter`) or fewer (`exit`).
2. Sample source rows from `X_syn`.
3. Filter source rows so enter repairs start from rows not satisfying the query and exit repairs start from rows satisfying the query.
4. Apply batched repair:
   - `_repair_enter_batch`: edits query terms so the new row satisfies the target query.
   - `_repair_exit_batch`: breaks at least one query term where possible.
5. Add random mutation candidates according to `random_candidate_fraction`.
6. Compute edit cost with `compute_edit_cost`.

The optional `qdte.candidate_compiler: masked_single_query` mode is a target-preserving masked version of the default one-query repair. For positive residuals it samples near-miss rows that satisfy the unmasked terms and fail the full query, then repairs only sampled masked terms. For negative residuals it samples rows satisfying the full query and breaks sampled masked terms, guaranteeing target-query exit when the candidate is kept. The final acceptance rule still uses the full measured-objective edit advantage.

The optional `qdte.candidate_compiler: paired_query` mode adds paired CPU proposals before the normal one-query repair fallback. It chooses negative-residual source queries and positive-residual destination queries from the active set, samples rows that satisfy the source and not the destination, repairs them into the destination, and optionally tries to break the source while preserving destination membership. `qdte.candidate_compiler: masked_paired_query` uses sampled subsets of ordinary query terms for source/destination constraints, widening the proposal space while keeping the same full measured-objective scoring. `qdte.candidate_compiler: masked_exit_query` samples rows that already satisfy a positive-residual destination mask and exits only a negative-residual source mask, preserving the destination mask. These paired proposals are not accepted just because they match the pair or mask.

The experimental exit-axis compilers isolate the candidate-generation design choices from the scoring and transport objective. `directed_exit_only` uses only negative-residual queries, samples old rows satisfying the query, and compiles directed exit repairs. `masked_exit_only` is the single-query masked version: it samples a subset of the negative query terms and exits only that mask, with final scoring still using the full measured objective. `random_source_directed_exit` keeps the negative-residual query direction for repair but removes source filtering, so the old row source is random. These variants are CPU-only ablations; they exist to compare QDTE-style directed proposal construction with Private-GSD-style random mutation under the same QDTE scorer.

The experimental `qdte.transport_mode: blind_accept` bypasses positive-advantage and batch/prefix objective gates, accepting generated nonconflicting candidates in generation order. This mode is a sanity-check ablation only; it is not equivalent to Private-GSD, which still ranks whole candidate datasets by fitness.

### GPU repair backend: `qdte/evolution/gpu_candidates.py`

Enabled by:

```yaml
qdte:
  candidate_backend: jax_repair
```

This backend fuses candidate generation and scoring inside a JAX `pmap` kernel:

- Replicates `X_syn` to each local GPU with `replicate_table_to_devices`.
- Reuses a cached GPU context so query/schema constants are device-put once per run.
- Pads active query ids to fixed `num_active_targets` capacity to reduce `pmap` recompilation.
- Samples source rows on GPU.
- Performs source satisfaction filtering using `_eval_candidate_source_satisfaction`.
- Applies directed enter/exit repairs in `_repair_directed_rows`.
- Applies random mutations in `_random_mutation_rows`.
- Computes edit costs on GPU.
- Scores all generated candidates on GPU.
- With `score_backend: dense_gpu`, uses boolean enter/exit masks and optional query-block scoring via `gpu_score_query_block_size`.
- With `score_backend: sparse_delta_gpu`, uses a precomputed attribute-to-query affected index and multi-word query-scope bitsets so each edit evaluates only query blocks touched by changed attributes.
- Supports directed halfspace enter/exit repair in the fused single-query GPU path. CPU-only structured compilers still require `candidate_backend: cpu_repair`.
- Optionally returns only local top-k candidates via `gpu_return_top_k`.
- Optionally runs multiple GPU scoring batches inside one QDTE iteration via `gpu_batches_per_iter`.

The returned `CandidateBatch.diagnostics["scored_candidates"]` records the full number of candidates scored, even if only top-k candidates are transferred back for selection.

The replicated GPU table is updated after accepted edits via `apply_edits_to_replicated_table`.

The GPU repair backend implements directed halfspace repair for the default single-query compiler. Structured compilers other than `single_query` remain CPU-only.

## Candidate Scoring

Implemented in `qdte/evolution/scoring.py`.

For candidate edit `x_old -> x_new`, define:

```text
delta[q] = 1{x_new satisfies q} - 1{x_old satisfies q}
weights[q] = residual[q] * inv_variance[q]
```

The score is:

```text
advantage =
    delta dot weights
    - 0.5 * sum_q delta[q]^2 * inv_variance[q]
    - lambda_cost * edit_cost
```

This is the one-edit improvement in the weighted quadratic objective, including an edit penalty.

Available scoring paths:

- `score_candidates`: dense JAX scoring over all queries, with optional multi-GPU `pmap`.
- `score_candidates_target_only`: cheaper ablation path that only scores against each candidate's target query.
- `score_candidates_sparse`: exact CPU sparse-delta scoring via `QueryDeltaIndex`; useful for audit/debug, not the current fastest path.
- `sparse_delta_gpu`: compiled sparse-delta scoring inside the fused GPU candidate backend.
- `compute_deltas`: computes full query deltas for selected candidates.
- `compute_deltas_sparse`: computes dense delta matrices from sparse affected-query evaluation.
- `edit_advantage_from_delta`: test/helper function for validating advantage calculations.

Dense JAX scoring computes the same formula with boolean enter/exit masks:

```text
linear = enter dot weights - exit dot weights
quad = (enter or exit) dot inv_variance
```

This is equivalent to explicitly materializing `delta = phi_new - phi_old`, but reduces large temporary float32 tensors in GPU scoring.

The sparse GPU scorer keeps the same objective. It uses multi-word `uint32` query-scope bitsets for duplicate suppression, so it no longer has the previous single-word 31 encoded-attribute limit. It still assumes the configured `gpu_sparse_changed_attr_capacity` covers all attributes a candidate can change. The current Adult highpower config sets that capacity to `4`, matching `workload.max_terms: 4`.

## Selection and Transport

Implemented in `qdte/evolution/transport.py`.

### Nonconflicting selection

`select_top_nonconflicting`:

1. Keeps finite candidates with advantage above `min_advantage`.
2. Sorts a candidate pool by score.
3. Accepts at most one candidate per synthetic row.
4. Returns up to `accepted_per_iter` candidate indices.

This prevents two edits from trying to update the same synthetic row in the same microbatch.

### CPU transport prefix

`choose_transport_batch`:

1. Sorts selected candidates by individual advantage.
2. Computes cumulative query delta prefixes.
3. Computes true batch advantage for each prefix.
4. Accepts either:
   - the largest positive prefix, or
   - the best positive prefix when `transport_prefix_strategy: best_advantage`.

The batch advantage is recomputed on the full residual, not just the per-candidate score, so the quadratic interaction among multiple edits is accounted for.

### JAX transport prefix

`choose_transport_batch_jax`:

Enabled by:

```yaml
qdte:
  transport_delta_backend: jax_prefix
```

It moves the prefix delta/advantage calculation into JAX using `_choose_transport_prefix_jit`. This reduces CPU transfer and CPU-side delta computation for high-throughput configurations.

### Atom-flow transport

`transport_mode: atom_flow` supports two update modes:

- `atom_flow_update_mode: exact`: successive atom-flow over full encoded old-row/new-row atom edges. Each accepted flow unit updates the current weighted residual and affected edge marginals before the next unit is selected.
- `atom_flow_update_mode: batch`: the default current path. It groups positive returned candidates by atom edge, estimates same-edge diminishing returns, enforces row capacity once, then uses JAX prefix objective evaluation to accept a positive batch prefix.

Both modes consume only measured residuals, inverse variances, candidate deltas, and edit costs. The accepted batch is evaluated with the original measured objective:

```text
A(B) = residual @ (Delta_B * inv_variance)
       - 0.5 * (Delta_B^2 @ inv_variance)
       - lambda_cost * total_edit_cost
```

Batch atom-flow is an approximate selector compared with exact successive atom-flow, but it still lets the engine update:

```text
answer_syn = answer_syn + transport.delta_sum
residual = target - answer_syn
```

once after accepting the prefix.

### Applying edits

`apply_edits` mutates the CPU `X_syn` table:

```text
X_syn[row_ids[accepted]] = new_rows[accepted]
```

When the GPU candidate backend is active, `apply_edits_to_replicated_table` also updates the replicated GPU table.

## Main QDTE Loop

The main loop is in `qdte/evolution/engine.py`.

Per iteration:

1. Set `state.iteration`.
2. Select active queries with `select_active_queries`.
3. Generate candidates:
   - CPU path: `generate_candidates`.
   - GPU path: `generate_and_score_candidates_gpu`.
4. Score candidates:
   - CPU repair path uses `score_candidates` or `score_candidates_target_only`.
   - GPU repair path returns fused scores from `generate_and_score_candidates_gpu`.
5. Choose transport:
   - `transport_mode: atom_flow` and `atom_flow_update_mode: batch`: `choose_atom_flow_batch_transport`;
   - `transport_mode: atom_flow` and `atom_flow_update_mode: exact`: `choose_atom_flow_transport`;
   - otherwise, `select_top_nonconflicting` followed by `choose_transport_batch` or `choose_transport_batch_jax`.
6. Apply accepted edits.
7. Incrementally update:

```text
answer_syn = answer_syn + transport.delta_sum
residual = target - answer_syn
```

8. Periodically recompute `answer_queries(state.X_syn, qcat)` to remove or detect incremental drift.
9. Log timeseries metrics at `log_every` and on selected special iterations.

The final step always recomputes synthetic query answers and reports `final_incremental_answer_drift`.

## Output Files

For a normal run, `run_qdte` writes:

- `config_resolved.yaml`: config after CLI overrides.
- `schema.json`: encoded schema.
- `queries.json`: query catalogue.
- `measurements.json`: DP noisy targets, variances, budget metadata.
- `synthetic_encoded.npy`: encoded synthetic table.
- `synthetic_decoded.csv`: decoded synthetic table, if `evaluation.save_synthetic_csv` is true.
- `metrics_final.json`: final privacy, loss, query error, and run summary.
- `metrics_by_family.json`: measured and optional true-query metrics split by query family.
- `metrics_timeseries.csv`: per-log-interval optimization metrics.
- `workload_summary.json`: measured workload summary.
- `runtime.json`: wall time, phase timings, throughput counters, backend names.
- `logs.txt`: console logs from the run.

When held-out evaluation is enabled, the run also writes:

- `queries_holdout.json`
- `workload_summary_holdout.json`
- `metrics_holdout.json`
- `metrics_by_family_holdout.json`

The user-required files are:

- `synthetic_decoded.csv`
- `metrics_final.json`
- `metrics_timeseries.csv`
- `runtime.json`

## Runtime and Metrics

Implemented in:

- `qdte/eval/metrics.py`
- `qdte/eval/runtime.py`

`RuntimeStats` tracks:

- measurement time
- initialization time
- total generation time
- candidate generation time
- scoring time
- transport time
- full recompute time
- number of iterations
- candidates requested/scored
- accepted edits
- source-filter diagnostics
- accepted rate
- scoring throughput

`query_error_metrics` reports post-run normalized query error:

- `true_query_mae`
- `true_query_rmse`
- `true_query_max_error`

These metrics compare exact true query rates to synthetic query rates for evaluation only.

## Tests

The current test suite covers the critical pieces:

- `tests/test_engine_smoke.py`: end-to-end smoke execution.
- `tests/test_measurement.py`: DP measurement/projection behavior.
- `tests/test_transport.py`: batch transport and selection behavior.
- `tests/test_queries.py`: query evaluation/workload behavior.
- `tests/test_edit_advantage.py`: edit advantage consistency.
- `tests/test_repairs.py`: candidate repair behavior, including directed enter/exit contracts for k-way conjunctions, halfspace queries, generated directed candidates, and the paired-query homogeneous-table edge case.
- `tests/test_projection.py`: prefix monotonicity projection behavior.
- `tests/test_consistency_projection.py`: scope-local consistency projection.
- `tests/test_config_validation.py`: unsupported-mode and backend validation.
- `tests/test_delta_index.py`: sparse affected-query delta index and sparse scoring.
- `tests/test_gpu_candidates.py`: GPU padding, query-block scoring, and sparse GPU scoring helpers.
- `tests/test_scheduler.py`: strict noise thresholding and debt scheduling.
- `tests/test_workload.py`: workload family construction, including halfspace.

The last verified test run passed:

```text
76 passed in 6.92s
```

## Current Verified Runs

### Default Adult DP run

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte.yaml --privacy.mode dp
```

Observed output directory:

```text
outputs/adult_qdte
```

Observed summary:

- `final_measured_loss`: about `2533.62`
- `true_query_mae`: about `0.000214`
- `true_query_rmse`: about `0.000657`
- `num_candidates_scored`: `20,480,000`
- `privacy_mode`: `dp`
- `epsilon_delta`: about `10.1046`

### High-throughput GPU Adult DP run

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte_gpu_highpower.yaml --privacy.mode dp
```

Observed output directory:

```text
outputs/adult_qdte_gpu_highpower
```

Current highpower defaults:

- `score_backend: sparse_delta_gpu`
- `candidate_backend: jax_repair`
- `transport_mode: atom_flow`
- `atom_flow_update_mode: batch`
- `total_candidates_per_iter: 1572864`
- `gpu_return_top_k: 8192`
- `accepted_per_iter: 1024`

Recent 50-iteration sparse-delta GPU probe:

- `num_candidates_scored`: `78,643,200`
- `time_generation_seconds`: about `19.91`
- `time_scoring_seconds`: about `14.26`
- `candidate_scoring_throughput_per_second`: about `5.52M`
- `candidates_scored_per_second`: about `3.95M`
- `final_measured_loss`: about `127,132`

A 1000-iteration highpower held-out run completed with zero incremental answer drift and improved measured and held-out 2-way true-query metrics. That run did not prove held-out generalization for prefix/range/mixed because duplicate filtering left only 2-way held-out queries.

This configuration is faster and uses the GPU much more heavily, but final quality should still be evaluated against the quality-oriented config for each workload and privacy setting.

## Important Implementation Notes

- The DP path is not a simulation. It computes real query answers on the real input table, adds zCDP Gaussian noise, projects/clips configured targets, and then runs QDTE against those noisy targets.
- Oracle mode is explicitly separated and logs a warning.
- The default path prioritizes final quality and robustness.
- The high-throughput path prioritizes GPU occupancy and iteration throughput.
- `measurement_mode=static_all` is the only implemented privacy measurement mode. Adaptive select-measure-generate is intentionally rejected with `NotImplementedError` in this version.
- `include_halfspace` is implemented for measurement/evaluation/CPU repair/GPU fused single-query repair/consistency projection.
- Consistency projection currently keeps the existing diagonal variance representation; full post-projection covariance propagation is not implemented.
- `sparse_delta_gpu` uses multi-word `uint32` query-scope bitsets, so the previous single-word 31 encoded-attribute limit has been removed. The remaining capacity assumption is `gpu_sparse_changed_attr_capacity`: it must cover the number of attributes a candidate can change.
- There is no repository-level dependency manifest yet; current tests and runs rely on the local `qdte` conda environment.
- The dense query evaluation uses a boolean `(batch_size, num_queries)` satisfaction matrix. This is simple and GPU-friendly, but it can be memory-bandwidth-bound.
